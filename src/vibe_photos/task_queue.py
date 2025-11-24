"""Celery task wiring for pre_process, process, and post_process stages."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import cast

from celery import Celery
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.artifact_store import (
    ArtifactManager,
    ArtifactResult,
    ArtifactSpec,
    hash_file,
)
from vibe_photos.cache_helpers import resolve_cache_root
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import (
    ArtifactRecord,
    PostProcessResult,
    ProcessResult,
    open_primary_session,
)
from vibe_photos.db import Image as ImageRow
from vibe_photos.ml.siglip_blip import SiglipBlipDetector
from vibe_photos.pipeline import PreprocessingPipeline
from vibe_photos.preprocessing import ensure_preprocessing_artifacts

LOGGER = get_logger(__name__, extra={"component": "task_queue"})


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    return load_settings()


def _default_primary_db() -> str:
    settings = _load_settings()
    return settings.databases.primary_url


def _default_cache_root() -> Path:
    settings = _load_settings()
    return resolve_cache_root(settings.cache.root)


def _init_celery() -> Celery:
    settings = _load_settings()
    app = Celery("vibe_photos")
    app.conf.update(
        broker_url=settings.queues.broker_url,
        result_backend=settings.queues.result_backend,
        worker_concurrency=settings.queues.default_concurrency,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_default_queue=settings.queues.preprocess_queue,
        task_routes={
            "vibe_photos.task_queue.pre_process": {"queue": settings.queues.preprocess_queue},
            "vibe_photos.task_queue.process": {"queue": settings.queues.main_queue},
            "vibe_photos.task_queue.post_process": {"queue": settings.queues.post_process_queue},
        },
    )
    return app


celery_app = _init_celery()


def _artifact_root() -> Path:
    return _default_cache_root() / "artifacts"


def _artifact_exists(session: Session, image_id: str, spec: ArtifactSpec) -> bool:
    stmt = (
        select(ArtifactRecord.id)
        .where(
            ArtifactRecord.image_id == image_id,
            ArtifactRecord.artifact_type == spec.artifact_type,
            ArtifactRecord.version_key == spec.version_key,
            ArtifactRecord.status == "complete",
        )
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none() is not None


@celery_app.task(name="vibe_photos.task_queue.pre_process", acks_late=True)
def pre_process(image_id: str) -> str:
    """Generate preprocessing artifacts for an image if they are missing."""

    settings = _load_settings()
    primary_path = _default_primary_db()
    artifact_root = _artifact_root()

    with open_primary_session(primary_path) as primary_session:
        row = primary_session.get(ImageRow, image_id)
        if row is None:
            LOGGER.warning("task_image_missing", extra={"image_id": image_id})
            return image_id

        image_path = Path(row.primary_path)
        if not image_path.exists():
            LOGGER.error("task_image_path_missing", extra={"image_id": image_id, "path": str(image_path)})
            return image_id

        image = Image.open(image_path).convert("RGB")
        manager = ArtifactManager(session=primary_session, root=artifact_root)
        detector = SiglipBlipDetector(settings=settings)

        artifacts = ensure_preprocessing_artifacts(
            image_id=image_id,
            image=image,
            image_path=image_path,
            settings=settings,
            manager=manager,
            detector=detector,
        )

        content_artifact = cast(ArtifactRecord, artifacts["content_hash_artifact"])
        phash_artifact = cast(ArtifactRecord, artifacts["phash_artifact"])
        thumb_large = cast(ArtifactSpec, artifacts["thumb_large_spec"])
        thumb_small = cast(ArtifactSpec, artifacts["thumb_small_spec"])
        exif_spec = cast(ArtifactSpec, artifacts["exif_spec"])
        embed_spec = cast(ArtifactSpec, artifacts["embedding_spec"])
        caption_spec = cast(ArtifactSpec, artifacts["caption_spec"])

        detection_spec = ArtifactSpec(
            artifact_type="detector_regions",
            model_name=settings.models.detection.model_name,
            params={"backend": settings.models.detection.backend, "score_threshold": settings.models.detection.score_threshold},
        )

        def _write_detections(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            detections: list[dict] = []
            if settings.models.detection.enabled:
                from vibe_photos.ml.detection import (
                    Detector,
                    build_owlvit_detector,
                    detection_priority,
                    non_max_suppression,
                )

                detector_model: Detector = build_owlvit_detector(settings=settings)
                raw = detector_model.detect(image, settings.models.siglip_labels.candidate_labels)
                filtered = [det for det in raw if det.score >= settings.models.detection.score_threshold]
                filtered = non_max_suppression(filtered, settings.models.detection.nms_iou_threshold)
                filtered = sorted(
                    filtered,
                    key=lambda det: detection_priority(
                        det,
                        area_gamma=settings.models.detection.primary_area_gamma,
                        center_penalty=settings.models.detection.primary_center_penalty,
                    ),
                    reverse=True,
                )
                for det in filtered:
                    detections.append(
                        {
                            "label": det.label,
                            "score": det.score,
                            "bbox": {
                                "x_min": det.bbox.x_min,
                                "y_min": det.bbox.y_min,
                                "x_max": det.bbox.x_max,
                                "y_max": det.bbox.y_max,
                            },
                            "backend": det.backend,
                            "model_name": det.model_name,
                        }
                    )

            out_path = out_dir / "detections.json"
            out_path.write_text(json.dumps(detections), encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))

        manager.ensure_artifact(
            image_id=image_id,
            spec=detection_spec,
            builder=_write_detections,
            dependencies=[content_artifact.id, phash_artifact.id],
        )

        LOGGER.info(
            "preprocess_artifacts_ready",
            extra={
                "image_id": image_id,
                "artifacts": [
                    thumb_large.version_key,
                    thumb_small.version_key,
                    exif_spec.version_key,
                    embed_spec.version_key,
                    caption_spec.version_key,
                    detection_spec.version_key,
                ],
            },
        )
    return image_id


@celery_app.task(name="vibe_photos.task_queue.process", acks_late=True)
def process(image_id: str) -> str:
    """Consume cached artifacts to compute labels, clusters, and indices."""

    settings = _load_settings()
    cache_root = _default_cache_root()
    artifact_root = _artifact_root()
    primary_path = _default_primary_db()

    embed_spec = ArtifactSpec(
        artifact_type="embedding",
        model_name=settings.models.embedding.resolved_model_name(),
        params={"device": settings.models.embedding.device, "batch_size": settings.models.embedding.batch_size},
    )

    # Ensure cache exists; if missing, run preprocess first.
    needs_preprocess = not cache_root.exists()
    if not needs_preprocess:
        with open_primary_session(primary_path) as session:
            needs_preprocess = not _artifact_exists(session, image_id, embed_spec)

    if needs_preprocess:
        pre_process(image_id)

    with open_primary_session(primary_path) as session:
        manager = ArtifactManager(session=session, root=artifact_root)
        embedding = manager.ensure_artifact(
            image_id=image_id,
            spec=embed_spec,
            builder=lambda path: ArtifactResult(storage_path=path / "missing.json", checksum="0"),
        )

        detection_spec = ArtifactSpec(
            artifact_type="detector_regions",
            model_name=settings.models.detection.model_name,
            params={"backend": settings.models.detection.backend, "score_threshold": settings.models.detection.score_threshold},
        )
        detection = manager.ensure_artifact(
            image_id=image_id,
            spec=detection_spec,
            builder=lambda path: ArtifactResult(storage_path=path / "missing.json", checksum="0"),
        )

        now = time.time()
        payload = {
            "classification_threshold": settings.main_processing.classification_threshold,
            "caption_threshold": settings.main_processing.caption_confidence_threshold,
        }
        result = ProcessResult(
            image_id=image_id,
            result_type="labels",
            version_key=f"main:{settings.main_processing.classification_threshold}",
            payload=json.dumps(payload),
            source_artifact_ids=json.dumps([embedding.id, detection.id]),
            created_at=now,
            updated_at=now,
        )
        session.add(result)
        session.commit()

        LOGGER.info(
            "process_complete",
            extra={"image_id": image_id, "version_key": result.version_key},
        )

    # Run the label + duplicate pipeline. The underlying implementation is
    # defensive and only recomputes embeddings, captions, scenes, and
    # duplicates when they are missing or configuration has changed, so it is
    # safe (though potentially heavy) to call from each process() invocation.
    settings = _load_settings()
    cache_root = _default_cache_root()
    primary_path = _default_primary_db()

    LOGGER.info(
        "label_pipeline_start",
        extra={"primary_db": str(primary_path), "cache_root": str(cache_root)},
    )

    pipeline = PreprocessingPipeline(settings=settings)
    # Mark the current image as "dirty" for near-duplicate recomputation. The
    # underlying pHash stage will fall back to a full rebuild when no pairs
    # exist yet, and will otherwise only refresh pairs involving this image.
    with open_primary_session(primary_path) as primary_session:
        row = primary_session.get(ImageRow, image_id)
        if row is not None and row.status == "active":
            row.near_duplicate_dirty = True
            primary_session.add(row)
            primary_session.commit()

    pipeline.run(roots=[], primary_db_url=primary_path, cache_root_path=cache_root)

    LOGGER.info("label_pipeline_complete", extra={})

    return image_id


@celery_app.task(name="vibe_photos.task_queue.post_process", acks_late=True)
def post_process(image_id: str) -> str:
    """Run optional OCR or cloud models with stricter concurrency limits."""

    settings = _load_settings()
    if not settings.post_process.enable_ocr and not settings.post_process.enable_cloud_models:
        LOGGER.info("post_process_disabled", extra={"image_id": image_id})
        return image_id

    artifact_root = _artifact_root()
    primary_path = _default_primary_db()
    with open_primary_session(primary_path) as session:
        manager = ArtifactManager(session=session, root=artifact_root)
        embed_spec = ArtifactSpec(
            artifact_type="embedding",
            model_name=settings.models.embedding.resolved_model_name(),
            params={"device": settings.models.embedding.device, "batch_size": settings.models.embedding.batch_size},
        )
        embedding = manager.ensure_artifact(
            image_id=image_id,
            spec=embed_spec,
            builder=lambda path: ArtifactResult(storage_path=path / "missing.json", checksum="0"),
        )

        now = time.time()
        out_dir = artifact_root / image_id / "post_process" / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        manifest = {
            "enable_ocr": settings.post_process.enable_ocr,
            "enable_cloud_models": settings.post_process.enable_cloud_models,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        post_process_result = PostProcessResult(
            image_id=image_id,
            post_process_type="baseline",
            version_key=f"post_process:{int(settings.post_process.enable_ocr)}:{int(settings.post_process.enable_cloud_models)}",
            storage_path=str(manifest_path),
            checksum=hash_file(manifest_path),
            source_artifact_ids=json.dumps([embedding.id]),
            created_at=now,
            updated_at=now,
        )
        session.add(post_process_result)
        session.commit()

        LOGGER.info(
            "post_process_complete",
            extra={"image_id": image_id, "version_key": post_process_result.version_key},
        )
    return image_id


__all__ = ["celery_app", "pre_process", "process", "post_process"]
