"""High-level preprocessing + label-layer pipeline orchestration."""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image as PILImage
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.artifact_store import ArtifactManager, ArtifactSpec
from vibe_photos.cache_manifest import CACHE_FORMAT_VERSION, ensure_cache_manifest
from vibe_photos.config import Settings, _normalize_siglip_label, load_settings
from vibe_photos.db import (
    ArtifactRecord,
    Image,
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageNearDuplicateGroup,
    ImageNearDuplicateMembership,
    ImageScene,
    Region,
    RegionEmbedding,
    open_primary_session,
)
from vibe_photos.db_helpers import dialect_insert
from vibe_photos.hasher import (
    CONTENT_HASH_ALGO,
    PHASH_ALGO,
    compute_content_hash,
    compute_perceptual_hash,
)
from vibe_photos.labels.build_object_prototypes import build_object_prototypes
from vibe_photos.labels.cluster_pass import (
    run_image_cluster_pass,
    run_region_cluster_pass,
)
from vibe_photos.labels.duplicate_propagation import propagate_duplicate_labels
from vibe_photos.labels.object_label_pass import run_object_label_pass
from vibe_photos.labels.scene_label_pass import run_scene_label_pass
from vibe_photos.ml.models import get_blip_caption_model, get_siglip_embedding_model
from vibe_photos.ml.siglip_blip import SiglipBlipDetector
from vibe_photos.preprocessing import (
    ensure_preprocessing_artifacts,
    extract_exif_and_gps,
)
from vibe_photos.scanner import FileInfo, scan_roots

LOGGER = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_storage_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return _PROJECT_ROOT / path
    return path


def _relative_to_cache_root(cache_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(cache_root))
    except ValueError:
        try:
            return str(path.relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(path)


@dataclass
class RunJournalRecord:
    """Lightweight checkpoint for resumable pipeline execution."""

    stage: str
    cursor_image_id: str | None
    updated_at: float


class ImageMeta(TypedDict):
    image_id: str
    phash_int: int
    mtime: float
    width: int
    height: int
    size_bytes: int


def _extract_exif_and_gps(
    image: PILImage.Image, datetime_format: str = "raw"
) -> tuple[str | None, str | None, dict[str, object] | None]:
    return extract_exif_and_gps(image, datetime_format)


def _siglip_label_group(
    label: str,
    label_groups: dict[str, Sequence[str]] | None = None,
    canonical_map: dict[str, str] | None = None,
) -> str | None:
    """Return a coarse semantic group name for a SigLIP label."""

    if canonical_map is not None:
        normalized = _normalize_siglip_label(label)
        return canonical_map.get(normalized)

    if not label_groups:
        return None

    normalized = _normalize_siglip_label(label)
    for group_name, members in label_groups.items():
        for member in members:
            if normalized == _normalize_siglip_label(member):
                return group_name
    return None


def load_run_journal(cache_root: Path) -> RunJournalRecord | None:
    """Load the run journal from ``cache/run_journal.json`` if it exists."""

    journal_path = cache_root / "run_journal.json"
    if not journal_path.exists():
        return None

    try:
        data = json.loads(journal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("run_journal_load_error", extra={"path": str(journal_path), "error": str(exc)})
        return None

    stage = str(data.get("stage") or "")
    cursor = data.get("cursor_image_id")
    updated_raw = data.get("updated_at")

    try:
        updated_at = float(updated_raw)
    except (TypeError, ValueError):
        updated_at = time.time()

    if not stage:
        return None

    return RunJournalRecord(stage=stage, cursor_image_id=str(cursor) if cursor is not None else None, updated_at=updated_at)


def save_run_journal(cache_root: Path, record: RunJournalRecord) -> None:
    """Persist the run journal to ``cache/run_journal.json``."""

    cache_root.mkdir(parents=True, exist_ok=True)
    journal_path = cache_root / "run_journal.json"
    payload = {
        "stage": record.stage,
        "cursor_image_id": record.cursor_image_id,
        "updated_at": record.updated_at,
    }
    journal_path.write_text(json.dumps(payload), encoding="utf-8")


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return



class PreprocessingPipeline:
    """Orchestrates preprocessing + label stages for one or more album roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with application settings.

        Args:
            settings: Optional pre-loaded Settings instance. When omitted,
                configuration is loaded from ``config/settings.yaml``.
        """

        self._settings = settings or load_settings()
        self._logger = get_logger(__name__, extra={"component": "preprocess"})
        self._cache_root: Path | None = None
        self._journal: RunJournalRecord | None = None

    def _stage_start_index(self, stage_names: Sequence[str]) -> int:
        """Return the first stage index to execute based on the run journal."""

        if self._journal is None:
            return 0

        try:
            current_idx = stage_names.index(self._journal.stage)
        except ValueError:
            return 0

        if self._journal.cursor_image_id is None:
            return min(current_idx + 1, len(stage_names))

        return current_idx

    def _purge_cache_entries(self, image_ids: Sequence[str], cache_session: Session | None) -> None:
        """Remove cached artifacts/rows for the provided image IDs."""

        ids = list({str(image_id) for image_id in image_ids})
        if not ids:
            return

        if cache_session is not None:
            # Use chunked deletes to avoid exceeding per-statement parameter limits
            # when invalidating large libraries.
            chunk_size = 400
            for offset in range(0, len(ids), chunk_size):
                chunk = ids[offset : offset + chunk_size]
                cache_session.execute(delete(ImageEmbedding).where(ImageEmbedding.image_id.in_(chunk)))
                cache_session.execute(delete(ImageCaption).where(ImageCaption.image_id.in_(chunk)))
                cache_session.execute(delete(ImageScene).where(ImageScene.image_id.in_(chunk)))
                cache_session.execute(delete(Region).where(Region.image_id.in_(chunk)))
                for image_id in chunk:
                    cache_session.execute(
                        delete(RegionEmbedding).where(RegionEmbedding.region_id.like(f"{image_id}#%"))
                    )
            cache_session.commit()

        cache_root = self._cache_root
        if cache_root:
            for image_id in ids:
                _remove_path(cache_root / "embeddings" / f"{image_id}.npy")
                _remove_path(cache_root / "embeddings" / f"{image_id}.json")
                _remove_path(cache_root / "captions" / f"{image_id}.json")
                _remove_path(cache_root / "regions" / f"{image_id}.json")
                artifacts_dir = cache_root / "artifacts" / image_id
                if artifacts_dir.exists():
                    shutil.rmtree(artifacts_dir, ignore_errors=True)

    def run(self, roots: Sequence[Path], primary_db_url: str, cache_root_path: Path) -> None:
        """Run the preprocessing pipeline for the given album roots.

        The initial implementation focuses on wiring and logging. Individual
        stages are implemented as stubs that can be expanded in dedicated
        iterations.

        Args:
            roots: Album root directories to scan.
            primary_db_url: Connection target for the primary operational database.
            cache_root_path: Path to the cache root directory.
        """

        if not isinstance(primary_db_url, str):
            raise TypeError("primary_db_url must be a PostgreSQL URL string")

        cache_root = cache_root_path
        self._cache_root = cache_root
        # Ensure cache manifest exists and matches current settings; resets cache
        # artifacts when the manifest changes.
        ensure_cache_manifest(cache_root, self._settings)
        self._logger.info(
            "pipeline_start",
            extra={
                "roots": [str(root) for root in roots],
                "primary_db": primary_db_url,
                "cache_root": str(cache_root),
            },
        )

        journal = load_run_journal(cache_root)
        self._journal = journal
        if journal is not None:
            self._logger.info(
                "pipeline_resume",
                extra={"stage": journal.stage, "cursor_image_id": journal.cursor_image_id},
            )

        with open_primary_session(primary_db_url) as primary_session:
            cache_session = primary_session
            stage_plan: list[tuple[str, Callable[[], None], bool]] = [
                (
                    "scan_and_hash",
                    lambda: self._run_scan_and_hash(roots, primary_session, cache_session),
                    True,
                ),
                (
                    "phash_and_duplicates",
                    lambda: self._run_perceptual_hashing_and_duplicates(primary_session, cache_session),
                    True,
                ),
                ("preprocess_artifacts", lambda: self._run_preprocess_artifacts(primary_session), True),
                (
                    "embeddings_and_captions",
                    lambda: self._run_embeddings_and_captions(primary_session, cache_session),
                    False,
                ),
                ("scene_classification", lambda: self._run_scene_classification(primary_session, cache_session), False),
            ]

            if self._settings.pipeline.run_detection and self._settings.models.detection.enabled:
                stage_plan.append(
                    (
                        "region_detection",
                        lambda: self._run_region_detection_and_reranking(primary_session, cache_session),
                        False,
                    )
                )
                stage_plan.append(
                    (
                        "object_label_pass",
                        lambda: self._run_object_label_pass_stage(primary_session, cache_session),
                        False,
                    )
                )
            if self._settings.pipeline.run_cluster:
                stage_plan.append(
                    (
                        "cluster_pass",
                        lambda: self._run_cluster_pass(primary_session, cache_session),
                        False,
                    )
                )

            stage_names = [name for name, _, _ in stage_plan]
            start_index = self._stage_start_index(stage_names)
            if start_index > 0:
                resume_stage = stage_names[start_index] if start_index < len(stage_names) else None
                self._logger.info(
                    "pipeline_resume_plan",
                    extra={"skipped_stages": stage_names[:start_index], "resume_stage": resume_stage},
                )

            for idx, (stage_name, stage_callable, record_completion) in enumerate(stage_plan):
                if idx < start_index:
                    self._logger.info("pipeline_stage_skipped", extra={"stage": stage_name})
                    continue

                if self._journal is not None and self._journal.stage == stage_name and self._journal.cursor_image_id:
                    self._logger.info(
                        "pipeline_stage_resume_cursor",
                        extra={"stage": stage_name, "cursor_image_id": self._journal.cursor_image_id},
                    )

                stage_callable()

                if record_completion:
                    save_run_journal(
                        cache_root,
                        RunJournalRecord(stage=stage_name, cursor_image_id=None, updated_at=time.time()),
                    )

        # Best-effort: remove run journal after a successful full run to avoid
        # stale cursors influencing future runs.
        try:
            journal_path = cache_root / "run_journal.json"
            if journal_path.exists():
                journal_path.unlink()
        except OSError:
            # Non-fatal; the next run will simply see an existing journal.
            pass

        self._logger.info("pipeline_complete", extra={})

    def _run_scan_and_hash(
        self, roots: Sequence[Path], primary_session: Session, cache_session: Session
    ) -> None:
        """Scan album roots and populate the images table with content hashes."""

        self._logger.info("scan_and_hash_start", extra={})
        files: list[FileInfo] = list(scan_roots(roots))
        total_files = len(files)
        self._logger.info("scan_and_hash_files_discovered", extra={"file_count": total_files})

        progress_interval = max(1, total_files // 20) if total_files else 0

        path_to_image_id: dict[str, str] = {}
        existing_rows = primary_session.execute(select(Image.image_id, Image.all_paths)).mappings()
        for row in existing_rows:
            raw_paths = row["all_paths"]
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []
            for stored_path in paths:
                path_to_image_id[str(stored_path)] = row["image_id"]

        now = time.time()
        processed_files = 0

        invalidated_ids: list[str] = []

        for file_info in files:
            path = file_info.path.resolve()
            path_str = str(path)
            new_image_id = compute_content_hash(path)
            old_image_id = path_to_image_id.get(path_str)

            if old_image_id is not None and old_image_id != new_image_id:
                old_image = primary_session.get(Image, old_image_id)
                if old_image is not None:
                    try:
                        old_paths = json.loads(old_image.all_paths) if old_image.all_paths else []
                    except json.JSONDecodeError:
                        old_paths = []

                    new_paths = [p for p in old_paths if p != path_str]
                    status = old_image.status
                    primary_path = old_image.primary_path

                    if new_paths:
                        if primary_path not in new_paths:
                            primary_path = new_paths[0]
                        if status == "deleted":
                            status = "active"
                    else:
                        status = "deleted"

                    old_image.all_paths = json.dumps(new_paths)
                    old_image.primary_path = primary_path
                    old_image.status = status
                    old_image.updated_at = now
                    old_image.near_duplicate_dirty = True
                    primary_session.add(old_image)

                if old_image_id is not None:
                    invalidated_ids.append(old_image_id)
                path_to_image_id.pop(path_str, None)
                old_image_id = None

            existing = primary_session.get(Image, new_image_id)

            if existing is not None:
                try:
                    paths = json.loads(existing.all_paths) if existing.all_paths else []
                except json.JSONDecodeError:
                    paths = []

                if path_str not in paths:
                    paths.append(path_str)

                primary_path = existing.primary_path or path_str
                status = existing.status or "active"
                if status == "deleted":
                    status = "active"

                existing.primary_path = primary_path
                existing.all_paths = json.dumps(paths)
                existing.size_bytes = file_info.size_bytes
                existing.mtime = file_info.mtime
                existing.hash_algo = CONTENT_HASH_ALGO
                existing.status = status
                existing.updated_at = now
                existing.near_duplicate_dirty = True
                primary_session.add(existing)
            else:
                all_paths_json = json.dumps([path_str])
                primary_session.add(
                    Image(
                        image_id=new_image_id,
                        primary_path=path_str,
                        all_paths=all_paths_json,
                        size_bytes=file_info.size_bytes,
                        mtime=file_info.mtime,
                        width=None,
                        height=None,
                        exif_datetime=None,
                        camera_model=None,
                        hash_algo=CONTENT_HASH_ALGO,
                        phash=None,
                        phash_algo=None,
                        phash_updated_at=None,
                        created_at=now,
                        updated_at=now,
                        status="active",
                        error_message=None,
                        schema_version=1,
                        near_duplicate_dirty=True,
                    )
                )

            path_to_image_id[path_str] = new_image_id

            processed_files += 1
            if progress_interval and (
                processed_files % progress_interval == 0 or processed_files == total_files
            ):
                percent = round(processed_files * 100.0 / max(total_files, 1), 1)
                self._logger.info(
                    "scan_and_hash_progress %s/%s (%.1f%%)",
                    processed_files,
                    total_files,
                    percent,
                    extra={"processed": processed_files, "total": total_files, "percent": percent},
                )

        primary_session.commit()
        if cache_session and cache_session is not primary_session:
            cache_session.commit()

        if invalidated_ids:
            self._purge_cache_entries(invalidated_ids, cache_session)

        now = time.time()
        images = primary_session.execute(select(Image)).scalars()
        for image in images:
            raw_paths = image.all_paths
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []

            if not paths:
                continue

            existing_paths: list[str] = []
            for stored_path in paths:
                stored_path_str = str(stored_path)
                try:
                    if Path(stored_path_str).exists():
                        existing_paths.append(stored_path_str)
                except OSError:
                    continue

            if existing_paths == paths:
                continue

            primary_path = image.primary_path
            status = image.status

            if existing_paths:
                if primary_path not in existing_paths:
                    primary_path = existing_paths[0]
                if status == "deleted":
                    status = "active"
            else:
                status = "deleted"

            image.all_paths = json.dumps(existing_paths)
            image.primary_path = primary_path
            image.status = status
            image.updated_at = now
            primary_session.add(image)

        primary_session.commit()
        if cache_session and cache_session is not primary_session:
            cache_session.commit()
        self._logger.info("scan_and_hash_complete", extra={"file_count": total_files})

    def _run_preprocess_artifacts(self, primary_session: Session) -> None:
        """Ensure Celery-style preprocessing artifacts exist for all active images."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        artifact_root = cache_root / "artifacts"
        manager = ArtifactManager(session=primary_session, root=artifact_root)
        detector = SiglipBlipDetector(settings=self._settings)

        rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active").order_by(Image.image_id)
        )
        total_images = primary_session.execute(select(func.count()).where(Image.status == "active")).scalar_one()

        processed = 0
        errors = 0
        progress_interval = max(1, int(total_images) // 20) if total_images else 0

        self._logger.info(
            "preprocess_artifacts_start",
            extra={"artifact_root": str(artifact_root), "image_count": int(total_images)},
        )

        metadata_updates = 0

        for row in rows:
            image_id = row.image_id
            path_str = row.primary_path
            image_path = Path(path_str)

            try:
                with PILImage.open(image_path) as pil_image:
                    image_rgb = pil_image.convert("RGB")
                    artifacts = ensure_preprocessing_artifacts(
                        image_id=image_id,
                        image=image_rgb,
                        image_path=image_path,
                        settings=self._settings,
                        manager=manager,
                        detector=detector,
                    )
                metadata_payload = artifacts.get("metadata_payload")
                if isinstance(metadata_payload, dict) and self._update_image_metadata_from_payload(
                    primary_session, image_id, metadata_payload
                ):
                    metadata_updates += 1
                processed += 1
                if progress_interval and (
                    processed % progress_interval == 0 or processed == int(total_images)
                ):
                    percent = round(processed * 100.0 / max(int(total_images), 1), 1)
                    self._logger.info(
                        "preprocess_artifacts_progress %s/%s (%.1f%%)",
                        processed,
                        int(total_images),
                        percent,
                        extra={"processed": processed, "total": int(total_images), "percent": percent},
                    )
            except FileNotFoundError as exc:
                errors += 1
                self._logger.error(
                    "preprocess_artifact_path_missing",
                    extra={"image_id": image_id, "path": str(image_path), "error": str(exc)},
                )
            except Exception as exc:  # pragma: no cover - defensive
                errors += 1
                self._logger.error(
                    "preprocess_artifact_error",
                    extra={"image_id": image_id, "path": str(image_path), "error": str(exc)},
                )

        if metadata_updates:
            primary_session.commit()

        self._logger.info(
            "preprocess_artifacts_complete",
            extra={
                "processed": processed,
                "errors": errors,
                "artifact_root": str(artifact_root),
                "metadata_updated": metadata_updates,
            },
        )

    def _update_image_metadata_from_payload(
        self, session: Session, image_id: str, payload: dict[str, object]
    ) -> bool:
        """Persist EXIF/GPS metadata derived from preprocessing artifacts."""

        row = session.get(Image, image_id)
        if row is None:
            return False

        changed = False
        exif_datetime = payload.get("exif_datetime")
        camera_model = payload.get("camera_model")
        gps_payload = payload.get("gps") or {}

        if isinstance(exif_datetime, str) and exif_datetime != row.exif_datetime:
            row.exif_datetime = exif_datetime
            changed = True

        if isinstance(camera_model, str) and camera_model != row.camera_model:
            row.camera_model = camera_model
            changed = True

        lat = gps_payload.get("latitude") if isinstance(gps_payload, dict) else None
        lon = gps_payload.get("longitude") if isinstance(gps_payload, dict) else None

        if isinstance(lat, (int, float)) and lat != row.gps_latitude:
            row.gps_latitude = float(lat)
            changed = True
        if isinstance(lon, (int, float)) and lon != row.gps_longitude:
            row.gps_longitude = float(lon)
            changed = True

        if changed:
            row.updated_at = time.time()
            session.add(row)
        return changed

    def _run_region_detection_and_reranking(
        self,
        primary_session: Session,
        cache_session: Session | None = None,
        target_image_ids: Sequence[str] | None = None,
    ) -> None:
        """Detection pass (M2): emit regions + region embeddings, no semantic labels."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        detection_cfg = self._settings.models.detection

        self._logger.info(
            "region_detection_start",
            extra={
                "detection_backend": detection_cfg.backend,
                "detection_model": detection_cfg.model_name,
            },
        )

        from vibe_photos.ml.detection import (
            BoundingBox,
            OwlVitDetector,
            build_owlvit_detector,
            detection_priority,
            filter_secondary_regions_by_priority,
            non_max_suppression,
        )
        from vibe_photos.ml.detection import (
            Detection as RegionDetection,
        )
        from vibe_photos.ml.models import get_siglip_embedding_model

        cache_root = self._cache_root
        regions_dir = cache_root / "regions"
        regions_dir.mkdir(parents=True, exist_ok=True)

        embedding_model_name = self._settings.models.embedding.resolved_model_name()
        embedding_backend = self._settings.models.embedding.backend

        # Fallback: if no cache session is provided (e.g., single-image task),
        # reuse the primary session because feature tables now live in the primary DB.
        cache_session = cache_session or primary_session

        try:
            detector: OwlVitDetector = build_owlvit_detector(settings=self._settings)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error(
                "region_detector_init_error",
                extra={"backend": detection_cfg.backend, "model_name": detection_cfg.model_name, "error": str(exc)},
            )
            return

        siglip_processor, siglip_model, siglip_device = get_siglip_embedding_model(settings=self._settings)

        siglip_labels_cfg = self._settings.models.siglip_labels
        candidate_labels = list(siglip_labels_cfg.candidate_labels)

        import numpy as np
        import torch
        from PIL import Image as _Image

        active_rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active").order_by(Image.image_id)
        )
        paths_by_id: dict[str, str] = {row.image_id: row.primary_path for row in active_rows}

        if target_image_ids is not None:
            allowed_ids = {str(image_id) for image_id in target_image_ids}
            paths_by_id = {image_id: path for image_id, path in paths_by_id.items() if image_id in allowed_ids}

        caption_cfg = self._settings.models.caption
        caption_model_name = caption_cfg.resolved_model_name()
        caption_rows = cache_session.execute(
            select(ImageCaption.image_id, ImageCaption.caption).where(ImageCaption.model_name == caption_model_name)
        )
        captions_by_id: dict[str, str] = {row.image_id: row.caption for row in caption_rows}

        if not paths_by_id:
            self._logger.info("region_detection_noop", extra={})
            return

        process_ids = sorted(paths_by_id.keys())

        non_canonical_ids = self._load_non_canonical_image_ids(primary_session)
        if non_canonical_ids:
            process_ids = [image_id for image_id in process_ids if image_id not in non_canonical_ids]

        nms_iou_threshold = float(detection_cfg.nms_iou_threshold)
        area_gamma = float(detection_cfg.primary_area_gamma)
        center_penalty = float(detection_cfg.primary_center_penalty)
        caption_fallback_enabled = bool(detection_cfg.caption_primary_enabled)
        min_primary_priority = float(detection_cfg.caption_primary_min_priority)
        margin_x = float(detection_cfg.caption_primary_box_margin_x)
        margin_y = float(detection_cfg.caption_primary_box_margin_y)
        caption_keyword_groups = detection_cfg.caption_primary_keywords
        secondary_min_priority = float(detection_cfg.secondary_min_priority)
        secondary_min_relative_to_primary = float(detection_cfg.secondary_min_relative_to_primary)
        max_regions = int(detection_cfg.max_regions_per_image)

        updated_rows = 0
        now = time.time()

        total_images = len(process_ids)
        progress_interval = max(1, total_images // 20) if total_images else 0
        processed_images = 0

        for image_id in process_ids:
            path_str = paths_by_id[image_id]
            try:
                image = _Image.open(path_str).convert("RGB")
            except Exception as exc:
                self._logger.error(
                    "region_detection_image_open_error",
                    extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                )
                continue

            try:
                detections: list[RegionDetection] = detector.detect(image=image, prompts=candidate_labels)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error(
                    "region_detection_backend_error",
                    extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                )
                continue

            detections = non_max_suppression(detections, iou_threshold=nms_iou_threshold)

            # Remove any existing regions/embeddings for this image_id to keep results consistent with the current model.
            cache_session.execute(delete(Region).where(Region.image_id == image_id))
            cache_session.execute(delete(RegionEmbedding).where(RegionEmbedding.region_id.like(f"{image_id}#%")))

            width, height = image.size

            # Prepare SigLIP inputs for region crops.
            region_images: list[_Image.Image] = []
            region_indices: list[int] = []
            priorities: list[float] = []

            for detection in detections:
                priorities.append(
                    detection_priority(
                        detection,
                        area_gamma=area_gamma,
                        center_penalty=center_penalty,
                    )
                )

            caption_text = captions_by_id.get(image_id, "")
            caption_lower = caption_text.lower()

            top_priority = max(priorities) if priorities else 0.0
            need_caption_fallback = caption_fallback_enabled and (
                not detections or top_priority < min_primary_priority
            )

            if need_caption_fallback and caption_lower and caption_keyword_groups:
                fallback_label: str | None = None
                for group_name, group_cfg in caption_keyword_groups.items():
                    label_value = group_cfg.get("label")
                    label = str(label_value).strip() if isinstance(label_value, str) else ""
                    raw_keywords = group_cfg.get("keywords", [])
                    if not isinstance(raw_keywords, list):
                        continue
                    for raw_keyword in raw_keywords:
                        keyword = str(raw_keyword).strip().lower()
                        if not keyword:
                            continue
                        if keyword in caption_lower:
                            fallback_label = label or group_name
                            break
                    if fallback_label:
                        break

                if fallback_label:
                    bbox = BoundingBox(
                        x_min=margin_x,
                        y_min=margin_y,
                        x_max=max(margin_x, 1.0 - margin_x),
                        y_max=max(margin_y, 1.0 - margin_y),
                    )
                    fallback_detection = RegionDetection(
                        bbox=bbox,
                        label=fallback_label,
                        score=0.5,
                        backend="caption-fallback",
                        model_name="caption-primary",
                    )
                    detections.append(fallback_detection)
                    priorities.append(
                        detection_priority(
                            fallback_detection,
                            area_gamma=area_gamma,
                            center_penalty=center_penalty,
                        )
                    )

            if detections:
                pairs = list(zip(detections, priorities, strict=True))
                pairs.sort(key=lambda pair: pair[1], reverse=True)
                sorted_detections, sorted_priorities = zip(*pairs, strict=True)
                detections = list(sorted_detections)
                priorities = list(sorted_priorities)

                if max_regions > 0:
                    detections, priorities = filter_secondary_regions_by_priority(
                        detections,
                        priorities,
                        max_regions=max_regions,
                        secondary_min_priority=secondary_min_priority,
                        secondary_min_relative_to_primary=secondary_min_relative_to_primary,
                    )
            else:
                detections = []
                priorities = []

            for index, det in enumerate(detections):
                x_min_px = int(max(0, min(width, round(det.bbox.x_min * width))))
                y_min_px = int(max(0, min(height, round(det.bbox.y_min * height))))
                x_max_px = int(max(0, min(width, round(det.bbox.x_max * width))))
                y_max_px = int(max(0, min(height, round(det.bbox.y_max * height))))

                if x_max_px <= x_min_px or y_max_px <= y_min_px:
                    continue

                crop = image.crop((x_min_px, y_min_px, x_max_px, y_max_px))
                region_images.append(crop)
                region_indices.append(index)

            region_payloads = []

            # Compute region embeddings in one batch to avoid redundant preprocessing.
            region_embeddings: list[np.ndarray] = []
            if region_images:
                region_inputs = siglip_processor(images=region_images, return_tensors="pt")
                region_inputs = region_inputs.to(siglip_device)

                with torch.no_grad():
                    region_emb = siglip_model.get_image_features(**region_inputs)

                region_emb = region_emb / region_emb.norm(dim=-1, keepdim=True)
                region_embeddings = [emb.detach().cpu().numpy().astype(np.float32) for emb in region_emb]

            for new_index, det in enumerate(detections):
                region_id = f"{image_id}#{new_index}"
                cache_session.add(
                    Region(
                        id=region_id,
                        image_id=image_id,
                        x_min=float(det.bbox.x_min),
                        y_min=float(det.bbox.y_min),
                        x_max=float(det.bbox.x_max),
                        y_max=float(det.bbox.y_max),
                        detector=det.backend,
                        raw_label=det.label,
                        raw_score=float(det.score),
                        created_at=now,
                    )
                )

                if new_index < len(region_embeddings):
                    emb_vec = region_embeddings[new_index]
                    region_rel_path = f"regions/{embedding_model_name}/{region_id}.npy"
                    emb_path = cache_root / region_rel_path
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(emb_path, emb_vec)

                    cache_session.add(
                        RegionEmbedding(
                            region_id=region_id,
                            model_name=embedding_model_name,
                            embedding_path=region_rel_path,
                            embedding_dim=int(emb_vec.shape[-1]),
                            backend=embedding_backend,
                            updated_at=now,
                        )
                    )

                region_payloads.append(
                    {
                        "bbox": {
                            "x_min": float(det.bbox.x_min),
                            "y_min": float(det.bbox.y_min),
                            "x_max": float(det.bbox.x_max),
                            "y_max": float(det.bbox.y_max),
                        },
                        "detector_label": det.label,
                        "detector_score": float(det.score),
                    }
                )

            cache_session.commit()
            updated_rows += len(region_payloads)

            payload = {
                "image_id": image_id,
                "backend": detection_cfg.backend,
                "model_name": detection_cfg.model_name,
                "updated_at": now,
                "detections": region_payloads,
            }
            cache_path = regions_dir / f"{image_id}.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

            processed_images += 1
            if progress_interval and (
                processed_images % progress_interval == 0 or processed_images == total_images
            ):
                percent = round(processed_images * 100.0 / max(total_images, 1), 1)
                self._logger.info(
                    "region_detection_progress %s/%s (%.1f%%)",
                    processed_images,
                    total_images,
                    percent,
                    extra={
                        "processed": processed_images,
                        "total": total_images,
                        "percent": percent,
                    },
                )

        self._logger.info(
            "region_detection_complete",
            extra={"regions_created": updated_rows},
        )

    def _run_object_label_pass_stage(self, primary_session: Session, cache_session: Session) -> None:
        """Run object label pass based on cached region embeddings."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        self._logger.info("object_label_pass_start", extra={})

        proto_name = self._settings.label_spaces.object_current
        proto_path = cache_root / "label_text_prototypes" / f"{proto_name}.npz"
        if not proto_path.exists():
            self._logger.info("object_label_pass_build_prototypes", extra={"prototype": proto_name})
            build_object_prototypes(
                session=primary_session,
                settings=self._settings,
                cache_root=cache_root,
                output_name=proto_name,
            )

        run_object_label_pass(
            primary_session=primary_session,
            cache_session=cache_session,
            settings=self._settings,
            cache_root=cache_root,
            label_space_ver=self._settings.label_spaces.object_current,
            prototype_name=proto_name,
        )

        propagated = propagate_duplicate_labels(
            primary_session=primary_session,
            label_space_ver=self._settings.label_spaces.object_current,
        )
        self._logger.info(
            "object_label_pass_complete",
            extra={"duplicate_labels_propagated": propagated},
        )

    def _run_cluster_pass(self, primary_session: Session, cache_session: Session) -> None:
        """Run image + region clustering passes."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        self._logger.info("cluster_pass_start", extra={})

        image_clusters, image_members = run_image_cluster_pass(
            primary_session=primary_session,
            cache_session=cache_session,
            settings=self._settings,
            cache_root=cache_root,
        )
        region_clusters, region_members = run_region_cluster_pass(
            primary_session=primary_session,
            cache_session=cache_session,
            settings=self._settings,
            cache_root=cache_root,
        )

        propagated = propagate_duplicate_labels(
            primary_session=primary_session,
            label_space_ver=self._settings.label_spaces.cluster_current,
        )
        self._logger.info(
            "cluster_pass_complete",
            extra={
                "image_clusters": image_clusters,
                "image_members": image_members,
                "region_clusters": region_clusters,
                "region_members": region_members,
                "duplicate_labels_propagated": propagated,
            },
        )

    def _run_scene_classification(self, primary_session: Session, cache_session: Session) -> None:
        """Run scene label pass that mirrors outputs into the label layer."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        self._logger.info("scene_classification_start", extra={})

        start_after = None
        if self._journal is not None and self._journal.stage == "scene_classification":
            start_after = self._journal.cursor_image_id
            if start_after:
                self._logger.info(
                    "scene_classification_resume_from_cursor",
                    extra={"cursor_image_id": start_after},
                )

        def _update_cursor(cursor_id: str | None) -> None:
            save_run_journal(
                cache_root,
                RunJournalRecord(stage="scene_classification", cursor_image_id=cursor_id, updated_at=time.time()),
            )

        processed = run_scene_label_pass(
            primary_session=primary_session,
            cache_session=cache_session,
            settings=self._settings,
            cache_root=cache_root,
            label_space_ver=self._settings.label_spaces.scene_current,
            start_after=start_after,
            update_cursor=_update_cursor,
        )
        _update_cursor(None)

        propagated = propagate_duplicate_labels(
            primary_session=primary_session,
            label_space_ver=self._settings.label_spaces.scene_current,
        )

        if processed == 0:
            self._logger.info("scene_classification_noop", extra={})
        else:
            self._logger.info(
                "scene_classification_complete",
                extra={"processed": processed, "duplicate_labels_propagated": propagated},
            )

    def _run_embeddings_and_captions(self, primary_session: Session, cache_session: Session) -> None:
        """Mirror artifact-backed embeddings/captions into the relational tables."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        self._logger.info("embeddings_and_captions_start", extra={})

        embedding_cfg = self._settings.models.embedding
        caption_cfg = self._settings.models.caption
        embedding_spec = ArtifactSpec(
            artifact_type="embedding",
            model_name=embedding_cfg.resolved_model_name(),
            params={},
        )
        caption_spec = ArtifactSpec(
            artifact_type="caption",
            model_name=caption_cfg.resolved_model_name(),
            params={},
        )

        active_rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active").order_by(Image.image_id)
        )
        process_ids = [row.image_id for row in active_rows]

        non_canonical_ids = self._load_non_canonical_image_ids(primary_session)
        if non_canonical_ids:
            process_ids = [image_id for image_id in process_ids if image_id not in non_canonical_ids]

        if not process_ids:
            self._logger.info("embeddings_and_captions_noop", extra={})
            return

        if self._journal is not None and self._journal.stage == "embeddings_and_captions" and self._journal.cursor_image_id:
            cursor_id = self._journal.cursor_image_id
            before = len(process_ids)
            process_ids = [image_id for image_id in process_ids if image_id > cursor_id]
            self._logger.info(
                "embeddings_resume_from_cursor",
                extra={"cursor_image_id": cursor_id, "skipped": before - len(process_ids)},
            )

        embedding_map = self._load_artifact_map(primary_session, process_ids, embedding_spec)
        caption_map = self._load_artifact_map(primary_session, process_ids, caption_spec)

        total_embeddings = 0
        total_captions = 0

        for image_id in process_ids:
            artifact_record = embedding_map.get(image_id)
            if artifact_record:
                if self._sync_embedding_from_artifact(
                    primary_session,
                    cache_root,
                    image_id,
                    artifact_record,
                    embedding_cfg,
                ):
                    total_embeddings += 1
            else:
                self._logger.warning(
                    "embedding_artifact_missing",
                    extra={"image_id": image_id, "artifact_type": embedding_spec.artifact_type},
                )

            caption_record = caption_map.get(image_id)
            if caption_record:
                if self._sync_caption_from_artifact(
                    primary_session,
                    image_id,
                    caption_record,
                    caption_cfg,
                ):
                    total_captions += 1
            else:
                self._logger.debug(
                    "caption_artifact_missing",
                    extra={"image_id": image_id, "artifact_type": caption_spec.artifact_type},
                )

            save_run_journal(
                cache_root,
                RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=image_id, updated_at=time.time()),
            )

        save_run_journal(
            cache_root,
            RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=None, updated_at=time.time()),
        )
        primary_session.commit()
        self._logger.info(
            "embeddings_and_captions_complete",
            extra={"embeddings_synced": total_embeddings, "captions_synced": total_captions},
        )

    def _sync_embedding_from_artifact(
        self,
        session: Session,
        cache_root: Path,
        image_id: str,
        record: ArtifactRecord,
        embedding_cfg: Any,
    ) -> bool:
        """Upsert an ImageEmbedding row from an existing artifact."""

        resolved_path = _resolve_storage_path(record.storage_path)
        rel_path = _relative_to_cache_root(cache_root, resolved_path)

        try:
            embedding_dim = self._load_embedding_dim(resolved_path)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error(
                "embedding_artifact_load_error",
                extra={"image_id": image_id, "path": str(resolved_path), "error": str(exc)},
            )
            return False

        updated_at = float(record.updated_at or time.time())
        values = {
            "image_id": image_id,
            "model_name": embedding_cfg.resolved_model_name(),
            "embedding_path": rel_path,
            "embedding_dim": int(embedding_dim),
            "model_backend": embedding_cfg.backend,
            "updated_at": updated_at,
        }

        stmt = dialect_insert(session, ImageEmbedding).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ImageEmbedding.image_id, ImageEmbedding.model_name],
            set_={
                "embedding_path": stmt.excluded.embedding_path,
                "embedding_dim": stmt.excluded.embedding_dim,
                "model_backend": stmt.excluded.model_backend,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        session.execute(stmt)
        return True

    def _load_embedding_dim(self, embedding_path: Path) -> int:
        """Return the embedding dimensionality derived from artifact metadata or the .npy itself."""

        meta_path = embedding_path.parent / "embedding_meta.json"
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                dim_value = payload.get("embedding_dim")
                if isinstance(dim_value, int):
                    return dim_value
                if isinstance(dim_value, (float, str)):
                    return int(dim_value)
            except Exception:
                pass

        import numpy as np

        vec = np.load(embedding_path)
        return int(vec.shape[-1])

    def _sync_caption_from_artifact(
        self,
        session: Session,
        image_id: str,
        record: ArtifactRecord,
        caption_cfg: Any,
    ) -> bool:
        """Upsert an ImageCaption row from an existing artifact."""

        resolved_path = _resolve_storage_path(record.storage_path)
        try:
            caption_text = resolved_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            self._logger.error(
                "caption_artifact_load_error",
                extra={"image_id": image_id, "path": str(resolved_path), "error": str(exc)},
            )
            return False

        updated_at = float(record.updated_at or time.time())
        values = {
            "image_id": image_id,
            "model_name": caption_cfg.resolved_model_name(),
            "caption": caption_text,
            "model_backend": caption_cfg.backend,
            "updated_at": updated_at,
        }

        stmt = dialect_insert(session, ImageCaption).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ImageCaption.image_id, ImageCaption.model_name],
            set_={
                "caption": stmt.excluded.caption,
                "model_backend": stmt.excluded.model_backend,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        session.execute(stmt)
        return True

    def _load_artifact_map(
        self,
        session: Session,
        image_ids: Sequence[str],
        spec: ArtifactSpec,
    ) -> dict[str, ArtifactRecord]:
        """Return artifact rows keyed by image_id for the provided spec."""

        if not image_ids:
            return {}

        stmt = select(ArtifactRecord).where(
            ArtifactRecord.image_id.in_(image_ids),
            ArtifactRecord.artifact_type == spec.artifact_type,
            ArtifactRecord.version_key == spec.version_key,
            ArtifactRecord.status == "complete",
        )
        rows = session.execute(stmt).scalars().all()
        return {row.image_id: row for row in rows}

    def process_embedding_task(self, primary_session: Session, image_id: str) -> None:
        """Compute SigLIP embedding for a single image and persist it."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        embedding_cfg = self._settings.models.embedding
        embedding_model_name = embedding_cfg.resolved_model_name()

        row = primary_session.get(Image, image_id)
        if row is None or row.status != "active":
            return

        path_str = row.primary_path

        existing = primary_session.execute(
            select(ImageEmbedding).where(
                and_(ImageEmbedding.image_id == image_id, ImageEmbedding.model_name == embedding_model_name)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return

        import numpy as np
        import torch
        from PIL import Image as _Image

        try:
            img = _Image.open(path_str).convert("RGB")
        except Exception as exc:
            self._logger.error(
                "embedding_image_open_error",
                extra={"image_id": image_id, "path": path_str, "error": str(exc)},
            )
            return

        siglip_processor, siglip_model, siglip_device = get_siglip_embedding_model(settings=self._settings)

        inputs = siglip_processor(images=[img], return_tensors="pt")
        inputs = inputs.to(siglip_device)

        with torch.no_grad():
            features = siglip_model.get_image_features(**inputs)

        emb = features[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.detach().cpu().numpy().astype(np.float32)

        embeddings_dir = cache_root / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        rel_path = f"{image_id}.npy"
        emb_path = embeddings_dir / rel_path
        np.save(emb_path, vec)

        now = time.time()

        meta_payload = {
            "image_id": image_id,
            "model_name": embedding_model_name,
            "model_backend": embedding_cfg.backend,
            "embedding_dim": int(vec.shape[-1]),
            "updated_at": now,
            "cache_format_version": CACHE_FORMAT_VERSION,
        }
        meta_path = embeddings_dir / f"{image_id}.json"
        meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")

        stmt = dialect_insert(primary_session, ImageEmbedding).values(
            image_id=image_id,
            model_name=embedding_model_name,
            embedding_path=rel_path,
            embedding_dim=int(vec.shape[-1]),
            model_backend=embedding_cfg.backend,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ImageEmbedding.image_id, ImageEmbedding.model_name],
            set_={
                "embedding_path": stmt.excluded.embedding_path,
                "embedding_dim": stmt.excluded.embedding_dim,
                "model_backend": stmt.excluded.model_backend,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        primary_session.execute(stmt)
        primary_session.commit()

    def process_caption_task(self, primary_session: Session, image_id: str) -> None:
        """Compute a BLIP caption for a single image and persist it."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        caption_cfg = self._settings.models.caption
        caption_model_name = caption_cfg.resolved_model_name()

        row = primary_session.get(Image, image_id)
        if row is None or row.status != "active":
            return

        existing = primary_session.execute(
            select(ImageCaption).where(
                and_(ImageCaption.image_id == image_id, ImageCaption.model_name == caption_model_name)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return

        path_str = row.primary_path

        import torch
        from PIL import Image as _ImageCaption

        try:
            img = _ImageCaption.open(path_str).convert("RGB")
        except Exception as exc:
            self._logger.error(
                "caption_image_open_error",
                extra={"image_id": image_id, "path": path_str, "error": str(exc)},
            )
            return

        blip_processor, blip_model, blip_device = get_blip_caption_model(settings=self._settings)

        inputs = blip_processor(images=[img], return_tensors="pt")
        inputs = inputs.to(blip_device)

        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs)

        token_ids = generated_ids[0]
        caption_text = blip_processor.decode(token_ids, skip_special_tokens=True)

        captions_dir = cache_root / "captions"
        captions_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        rel_path = f"{image_id}.json"
        caption_path = captions_dir / rel_path
        payload = {
            "image_id": image_id,
            "caption": caption_text,
            "model_name": caption_model_name,
            "model_backend": caption_cfg.backend,
            "updated_at": now,
            "cache_format_version": CACHE_FORMAT_VERSION,
        }
        caption_path.write_text(json.dumps(payload), encoding="utf-8")

        stmt = dialect_insert(primary_session, ImageCaption).values(
            image_id=image_id,
            model_name=caption_model_name,
            caption=caption_text,
            model_backend=caption_cfg.backend,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ImageCaption.image_id, ImageCaption.model_name],
            set_={
                "caption": stmt.excluded.caption,
                "model_backend": stmt.excluded.model_backend,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        primary_session.execute(stmt)
        primary_session.commit()

    def process_region_detection_task(self, primary_session: Session, image_id: str) -> None:
        """Run region detection for a single image via the detection pipeline."""

        if not self._settings.pipeline.run_detection or not self._settings.models.detection.enabled:
            return

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._run_region_detection_and_reranking(primary_session, target_image_ids=[image_id])

    def _run_perceptual_hashing_and_duplicates(self, primary_session: Session, _cache_session: Session) -> None:
        """Compute perceptual hashes and rebuild near-duplicate groups."""

        self._logger.info("phash_and_duplicates_start", extra={})

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        now = time.time()

        phash_query = (
            select(Image.image_id, Image.primary_path)
            .where(
                and_(
                    Image.status == "active",
                    or_(
                        Image.phash.is_(None),
                        Image.phash_algo.is_(None),
                        Image.phash_algo != PHASH_ALGO,
                        Image.width.is_(None),
                        Image.height.is_(None),
                        Image.exif_datetime.is_(None),
                        Image.camera_model.is_(None),
                    ),
                )
            )
            .order_by(Image.image_id)
        )

        # Ensure concurrent Celery workers lock rows in a consistent order on Postgres to
        # avoid deadlocks when multiple pipelines recompute pHash simultaneously.
        bind = primary_session.get_bind()
        if bind is None:
            raise RuntimeError("primary_session is not bound to an engine")
        phash_query = phash_query.with_for_update(skip_locked=True)

        rows = primary_session.execute(phash_query)

        phash_updated = 0
        for row in rows:
            image_id = row.image_id
            path_str = row.primary_path
            try:
                from PIL import Image as _ImagePhash

                image = _ImagePhash.open(path_str)
                width, height = image.size
            except Exception as exc:
                self._logger.error("phash_image_open_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                primary_session.execute(
                    update(Image)
                    .where(Image.image_id == image_id)
                    .values(status="error", error_message=f"phash_image_open_error: {exc}", updated_at=now)
                )
                continue

            exif_datetime: str | None
            camera_model: str | None
            gps_payload: dict[str, object] | None

            try:
                exif_datetime, camera_model, gps_payload = _extract_exif_and_gps(
                    image, datetime_format=self._settings.pipeline.exif_datetime_format
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("exif_parse_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                exif_datetime = None
                camera_model = None
            try:
                phash_hex = compute_perceptual_hash(image)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("phash_compute_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                primary_session.execute(
                    update(Image)
                    .where(Image.image_id == image_id)
                    .values(status="error", error_message=f"phash_compute_error: {exc}", updated_at=now)
                )
                continue

            primary_session.execute(
                update(Image)
                .where(Image.image_id == image_id)
                .values(
                    width=int(width),
                    height=int(height),
                    exif_datetime=exif_datetime,
                    camera_model=camera_model,
                    phash=phash_hex,
                    phash_algo=PHASH_ALGO,
                    phash_updated_at=now,
                    status="active",
                    error_message=None,
                    updated_at=now,
                )
            )
            phash_updated += 1

        primary_session.commit()
        self._logger.info("phash_update_complete", extra={"updated_count": phash_updated})

        rows = primary_session.execute(
            select(Image.image_id, Image.phash, Image.mtime, Image.width, Image.height, Image.size_bytes).where(
                and_(Image.status == "active", Image.phash.is_not(None), Image.phash_algo == PHASH_ALGO)
            )
        )

        threshold = self._settings.pipeline.phash_hamming_threshold
        if threshold <= 0:
            threshold = 12

        items: list[ImageMeta] = []
        image_meta: dict[str, ImageMeta] = {}
        for row in rows:
            if not row.phash:
                continue
            try:
                ph_int = int(row.phash, 16)
            except (TypeError, ValueError):
                continue
            record: ImageMeta = {
                "image_id": str(row.image_id),
                "phash_int": int(ph_int),
                "mtime": float(row.mtime or 0.0),
                "width": int(row.width or 0),
                "height": int(row.height or 0),
                "size_bytes": int(row.size_bytes or 0),
            }
            items.append(record)
            image_meta[str(row.image_id)] = record

        if not items:
            self._logger.info("phash_and_duplicates_noop", extra={})
            return

        items.sort(key=lambda item: item["mtime"], reverse=True)
        phash_map: dict[str, ImageMeta] = {str(item["image_id"]): item for item in items}

        dirty_ids = {
            row.image_id
            for row in primary_session.execute(
                select(Image.image_id).where(
                    and_(
                        Image.status == "active",
                        Image.phash.is_not(None),
                        Image.phash_algo == PHASH_ALGO,
                        Image.near_duplicate_dirty.is_(True),
                    )
                )
            )
        }

        existing_pairs = {
            (
                row.anchor_image_id,
                row.duplicate_image_id,
            )
            for row in primary_session.execute(select(ImageNearDuplicate)).scalars()
        }

        if not existing_pairs:
            dirty_ids = set(phash_map.keys())

        if not dirty_ids:
            self._logger.info("phash_and_duplicates_skip", extra={"reason": "no_dirty_ids"})
            return

        # Database engines enforce limits on the number of bound variables in a
        # single statement (commonly around 1k). Large libraries can easily exceed
        # this when deleting by IN (...) for both anchor and duplicate columns, so
        # chunk the deletes to stay under that limit while keeping behavior the
        # same. Near-duplicate edges are persisted in the primary database only.
        dirty_id_list = list(dirty_ids)
        if dirty_id_list:
            chunk_size = 400
            for offset in range(0, len(dirty_id_list), chunk_size):
                chunk = dirty_id_list[offset : offset + chunk_size]
                primary_session.execute(
                    delete(ImageNearDuplicate).where(
                        or_(
                            ImageNearDuplicate.anchor_image_id.in_(chunk),
                            ImageNearDuplicate.duplicate_image_id.in_(chunk),
                        )
                    )
                )

        pairs_to_add: set[tuple[str, str]] = set()
        pair_count = 0

        for dirty_id in dirty_ids:
            dirty_info = phash_map[dirty_id]
            dirty_int = int(dirty_info["phash_int"])
            dirty_mtime = float(dirty_info["mtime"])

            for other_id, other_info in phash_map.items():
                if other_id == dirty_id:
                    continue
                distance = int((dirty_int ^ int(other_info["phash_int"])).bit_count())
                if distance > threshold:
                    continue

                other_mtime = float(other_info["mtime"])
                if (dirty_mtime > other_mtime) or (dirty_mtime == other_mtime and dirty_id < other_id):
                    anchor_id, duplicate_id = dirty_id, other_id
                else:
                    anchor_id, duplicate_id = other_id, dirty_id

                pair = (anchor_id, duplicate_id)
                if pair in pairs_to_add:
                    continue
                pairs_to_add.add(pair)

                primary_session.add(
                    ImageNearDuplicate(
                        anchor_image_id=anchor_id,
                        duplicate_image_id=duplicate_id,
                        phash_distance=distance,
                        created_at=now,
                    )
                )
                pair_count += 1
                if pair_count % 1000 == 0:
                    primary_session.commit()

        primary_session.commit()

        # Clear dirty flags now that near-duplicate relationships are rebuilt.
        primary_session.execute(
            update(Image).where(Image.near_duplicate_dirty.is_(True)).values(near_duplicate_dirty=False)
        )
        primary_session.commit()

        groups_created, memberships_created = self._rebuild_duplicate_groups(primary_session, image_meta)

        self._logger.info(
            "phash_and_duplicates_complete",
            extra={
                "phash_updated": phash_updated,
                "near_duplicate_pairs": pair_count,
                "threshold": threshold,
                "duplicate_groups": groups_created,
                "duplicate_memberships": memberships_created,
            },
        )

    def _rebuild_duplicate_groups(self, primary_session: Session, image_meta: dict[str, ImageMeta]) -> tuple[int, int]:
        primary_session.execute(delete(ImageNearDuplicateMembership))
        primary_session.execute(delete(ImageNearDuplicateGroup))
        pair_rows = primary_session.execute(
            select(ImageNearDuplicate.anchor_image_id, ImageNearDuplicate.duplicate_image_id)
        ).all()
        if not pair_rows:
            primary_session.commit()
            return 0, 0

        graph: dict[str, set[str]] = defaultdict(set)
        for row in pair_rows:
            anchor_id = row.anchor_image_id
            duplicate_id = row.duplicate_image_id
            graph[anchor_id].add(duplicate_id)
            graph[duplicate_id].add(anchor_id)

        visited: set[str] = set()
        groups_created = 0
        memberships_created = 0
        method = f"{PHASH_ALGO}_v1"
        timestamp = time.time()

        for node in graph:
            if node in visited:
                continue

            component: list[str] = []
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(graph.get(current, []))

            if len(component) < 2:
                continue

            canonical_id = self._select_canonical_image(component, image_meta)
            if canonical_id is None:
                continue

            group = ImageNearDuplicateGroup(
                canonical_image_id=canonical_id,
                method=method,
                created_at=timestamp,
            )
            primary_session.add(group)
            primary_session.flush()

            canonical_meta = image_meta.get(canonical_id)
            if canonical_meta is None:
                continue

            for member_id in component:
                member_meta = image_meta.get(member_id)
                if member_meta is None:
                    continue
                distance = self._phash_distance(
                    int(member_meta["phash_int"]),
                    int(canonical_meta["phash_int"]),
                )
                primary_session.add(
                    ImageNearDuplicateMembership(
                        group_id=group.id,
                        image_id=member_id,
                        is_canonical=member_id == canonical_id,
                        distance=float(distance),
                    )
                )
                memberships_created += 1

            groups_created += 1

        primary_session.commit()
        return groups_created, memberships_created

    @staticmethod
    def _select_canonical_image(members: Sequence[str], image_meta: dict[str, ImageMeta]) -> str | None:
        best_id: str | None = None
        best_key: tuple[int, int, str] | None = None
        for image_id in members:
            meta = image_meta.get(image_id)
            if meta is None:
                continue
            width = int(meta.get("width", 0))
            height = int(meta.get("height", 0))
            area = width * height
            size_bytes = int(meta.get("size_bytes", 0))
            key = (area, size_bytes, image_id)
            if best_key is None or key > best_key:
                best_key = key
                best_id = image_id
        return best_id

    @staticmethod
    def _phash_distance(lhs: int, rhs: int) -> int:
        return int((int(lhs) ^ int(rhs)).bit_count())

    def _load_non_canonical_image_ids(self, primary_session: Session) -> set[str]:
        if not self._settings.pipeline.skip_duplicates_for_heavy_models:
            return set()
        rows = primary_session.execute(
            select(ImageNearDuplicateMembership.image_id).where(ImageNearDuplicateMembership.is_canonical.is_(False))
        )
        return {row.image_id for row in rows}


__all__ = ["PreprocessingPipeline", "RunJournalRecord", "load_run_journal", "save_run_journal"]
