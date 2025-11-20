"""Celery task wiring for preprocessing, main, and enhancement stages."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from celery import Celery
from PIL import Image
from utils.logging import get_logger

from vibe_photos.artifact_store import ArtifactManager, ArtifactResult, ArtifactSpec, hash_file
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import Image as ImageRow
from vibe_photos.db import EnhancementOutput, MainStageResult, open_primary_session, open_projection_session
from vibe_photos.hasher import compute_content_hash, compute_perceptual_hash
from vibe_photos.ml.siglip_blip import SiglipBlipDetector


LOGGER = get_logger(__name__, extra={"component": "task_queue"})


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    return load_settings()


def _default_primary_db() -> Path:
    return Path("data/index.db")


def _default_projection_db() -> Path:
    return Path("data/projection.db")


def _init_celery() -> Celery:
    settings = _load_settings()
    app = Celery("vibe_photos")
    app.conf.broker_url = settings.queues.broker_url
    app.conf.result_backend = settings.queues.result_backend
    app.conf.task_acks_late = True
    app.conf.task_default_queue = settings.queues.preprocess_queue
    app.conf.task_routes = {
        "vibe_photos.task_queue.process_image": {"queue": settings.queues.preprocess_queue},
        "vibe_photos.task_queue.run_main_stage": {"queue": settings.queues.main_queue},
        "vibe_photos.task_queue.run_enhancement": {"queue": settings.queues.enhancement_queue},
    }
    return app


celery_app = _init_celery()


def _artifact_root() -> Path:
    return Path("cache/artifacts")


def _build_thumbnail(image: Image.Image, output_dir: Path, size: int) -> ArtifactResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    resized = image.copy()
    resized.thumbnail((size, size))
    out_path = output_dir / f"thumbnail_{size}.jpg"
    resized.save(out_path, format="JPEG", quality=90)
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def _extract_exif(image: Image.Image) -> dict:
    try:
        exif = image.getexif()
    except Exception:
        return {}

    payload: dict = {}
    for tag_id, value in dict(exif).items():
        payload[str(tag_id)] = value if isinstance(value, (int, float, str)) else str(value)
    return payload


@celery_app.task(name="vibe_photos.task_queue.process_image", acks_late=True)
def process_image(image_id: str) -> str:
    """Generate preprocessing artifacts for an image if they are missing."""

    settings = _load_settings()
    primary_path = _default_primary_db()
    projection_path = _default_projection_db()
    artifact_root = _artifact_root()

    with open_primary_session(primary_path) as primary_session, open_projection_session(projection_path) as projection_session:
        row = primary_session.get(ImageRow, image_id)
        if row is None:
            LOGGER.warning("task_image_missing", extra={"image_id": image_id})
            return image_id

        image_path = Path(row.primary_path)
        if not image_path.exists():
            LOGGER.error("task_image_path_missing", extra={"image_id": image_id, "path": str(image_path)})
            return image_id

        image = Image.open(image_path).convert("RGB")
        manager = ArtifactManager(session=projection_session, root=artifact_root)

        thumb_large = ArtifactSpec(
            artifact_type="thumbnail_large",
            model_name="pil",
            params={"size": max(1024, settings.pipeline.thumbnail_size)},
        )
        manager.ensure_artifact(
            image_id=image_id,
            spec=thumb_large,
            builder=lambda path: _build_thumbnail(image, path, max(1024, settings.pipeline.thumbnail_size)),
        )

        thumb_small = ArtifactSpec(
            artifact_type="thumbnail_small",
            model_name="pil",
            params={"size": min(256, settings.pipeline.thumbnail_size)},
        )
        manager.ensure_artifact(
            image_id=image_id,
            spec=thumb_small,
            builder=lambda path: _build_thumbnail(image, path, min(256, settings.pipeline.thumbnail_size)),
        )

        exif_spec = ArtifactSpec(
            artifact_type="exif",
            model_name="pil_exif",
            params={"datetime_format": settings.pipeline.exif_datetime_format},
        )

        def _write_exif(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = _extract_exif(image)
            out_path = out_dir / "exif.json"
            out_path.write_text(json.dumps(payload), encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path), payload_path=out_path)

        manager.ensure_artifact(image_id=image_id, spec=exif_spec, builder=_write_exif)

        hash_spec = ArtifactSpec(artifact_type="content_hash", model_name="sha256", params={})

        def _write_content_hash(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            value = compute_content_hash(image_path)
            out_path = out_dir / "content_hash.txt"
            out_path.write_text(value, encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))

        content_artifact = manager.ensure_artifact(image_id=image_id, spec=hash_spec, builder=_write_content_hash)

        phash_spec = ArtifactSpec(artifact_type="perceptual_hash", model_name="phash", params={})

        def _write_phash(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            value = compute_perceptual_hash(image_path)
            out_path = out_dir / "perceptual_hash.txt"
            out_path.write_text(value, encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))

        phash_artifact = manager.ensure_artifact(image_id=image_id, spec=phash_spec, builder=_write_phash)

        detector = SiglipBlipDetector(settings=settings)
        embed_spec = ArtifactSpec(
            artifact_type="embedding",
            model_name=settings.models.embedding.resolved_model_name(),
            params={"device": settings.models.embedding.device, "batch_size": settings.models.embedding.batch_size},
        )

        def _write_embedding(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            scores = detector._classify_with_siglip(image=image, labels=settings.models.siglip_labels.candidate_labels)
            out_path = out_dir / "embedding.json"
            out_path.write_text(json.dumps(scores), encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path), payload_path=out_path)

        embedding = manager.ensure_artifact(
            image_id=image_id, spec=embed_spec, builder=_write_embedding, dependencies=[content_artifact.id]
        )

        caption_spec = ArtifactSpec(
            artifact_type="caption",
            model_name=settings.models.caption.resolved_model_name(),
            params={"device": settings.models.caption.device},
        )

        def _write_caption(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            caption = detector._generate_caption_with_blip(image=image)
            out_path = out_dir / "caption.txt"
            out_path.write_text(caption, encoding="utf-8")
            return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))

        caption_artifact = manager.ensure_artifact(
            image_id=image_id, spec=caption_spec, builder=_write_caption, dependencies=[embedding.id]
        )

        detection_spec = ArtifactSpec(
            artifact_type="detector_regions",
            model_name=settings.models.detection.model_name,
            params={"backend": settings.models.detection.backend, "score_threshold": settings.models.detection.score_threshold},
        )

        def _write_detections(out_dir: Path) -> ArtifactResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            detections: List[dict] = []
            if settings.models.detection.enabled:
                from vibe_photos.ml.detection import (  # noqa: WPS433
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

        detection_artifact = manager.ensure_artifact(
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


@celery_app.task(name="vibe_photos.task_queue.run_main_stage", acks_late=True)
def run_main_stage(image_id: str) -> str:
    """Consume cached artifacts to compute labels, clusters, and indices."""

    settings = _load_settings()
    projection_path = _default_projection_db()
    artifact_root = _artifact_root()

    with open_projection_session(projection_path) as session:
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
        result = MainStageResult(
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
            "main_stage_complete",
            extra={"image_id": image_id, "version_key": result.version_key},
        )
    return image_id


@celery_app.task(name="vibe_photos.task_queue.run_enhancement", acks_late=True)
def run_enhancement(image_id: str) -> str:
    """Run optional OCR or cloud models with stricter concurrency limits."""

    settings = _load_settings()
    if not settings.enhancement.enable_ocr and not settings.enhancement.enable_cloud_models:
        LOGGER.info("enhancement_disabled", extra={"image_id": image_id})
        return image_id

    projection_path = _default_projection_db()
    artifact_root = _artifact_root()
    with open_projection_session(projection_path) as session:
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
        out_dir = artifact_root / image_id / "enhancement" / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        manifest = {"enable_ocr": settings.enhancement.enable_ocr, "enable_cloud_models": settings.enhancement.enable_cloud_models}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        enhancement = EnhancementOutput(
            image_id=image_id,
            enhancement_type="baseline",
            version_key=f"enhancement:{int(settings.enhancement.enable_ocr)}:{int(settings.enhancement.enable_cloud_models)}",
            storage_path=str(manifest_path),
            checksum=hash_file(manifest_path),
            source_artifact_ids=json.dumps([embedding.id]),
            created_at=now,
            updated_at=now,
        )
        session.add(enhancement)
        session.commit()

        LOGGER.info(
            "enhancement_complete",
            extra={"image_id": image_id, "version_key": enhancement.version_key},
        )
    return image_id


__all__ = ["celery_app", "process_image", "run_main_stage", "run_enhancement"]

