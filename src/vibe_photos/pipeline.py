"""High-level M1 preprocessing pipeline orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from torch import Tensor
from utils.logging import get_logger
from vibe_photos.cache_manifest import CACHE_FORMAT_VERSION, write_cache_manifest
from vibe_photos.classifier import SceneClassifierWithAttributes, build_scene_classifier
from vibe_photos.config import Settings, _normalize_siglip_label, load_settings
from vibe_photos.db import (
    Image,
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageRegion,
    ImageScene,
    open_primary_session,
    open_projection_session,
)
from vibe_photos.hasher import (
    CONTENT_HASH_ALGO,
    PHASH_ALGO,
    compute_content_hash,
    compute_perceptual_hash,
    hamming_distance_phash,
)
from vibe_photos.ml.models import get_blip_caption_model, get_siglip_embedding_model
from vibe_photos.scanner import FileInfo, scan_roots


LOGGER = get_logger(__name__)


@dataclass
class RunJournalRecord:
    """Lightweight checkpoint for resumable pipeline execution."""

    stage: str
    cursor_image_id: Optional[str]
    updated_at: float


def _extract_exif_and_gps(image, datetime_format: str = "raw") -> tuple[Optional[str], Optional[str], Optional[Dict[str, object]]]:
    """Extract EXIF datetime, camera model, and GPS coordinates from a PIL image.

    Returns a tuple of ``(exif_datetime, camera_model, gps_payload)`` where the GPS payload,
    when present, contains ``latitude`` and ``longitude`` in decimal degrees and associated
    reference fields.
    """

    try:
        from PIL import ExifTags as _ExifTags
    except Exception:
        return None, None, None

    try:
        exif = image.getexif()
    except Exception:
        return None, None, None

    if not exif:
        return None, None, None

    exif_dict = dict(exif)
    exif_by_name: Dict[str, object] = {}
    for tag_id, value in exif_dict.items():
        name = _ExifTags.TAGS.get(tag_id, str(tag_id))
        exif_by_name[name] = value

    exif_datetime: Optional[str] = None
    camera_model: Optional[str] = None

    raw_dt = exif_by_name.get("DateTimeOriginal") or exif_by_name.get("DateTime")
    if isinstance(raw_dt, str):
        if datetime_format == "iso":
            try:
                # Common EXIF format: "YYYY:MM:DD HH:MM:SS"
                dt = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
                exif_datetime = dt.isoformat(timespec="seconds")
            except Exception:
                exif_datetime = raw_dt
        else:
            exif_datetime = raw_dt

    model = exif_by_name.get("Model")
    make = exif_by_name.get("Make")
    if isinstance(model, str) and isinstance(make, str):
        camera_model = f"{make} {model}".strip()
    elif isinstance(model, str):
        camera_model = model
    elif isinstance(make, str):
        camera_model = make

    gps_info = exif_by_name.get("GPSInfo")
    if not gps_info or not isinstance(gps_info, dict):
        return exif_datetime, camera_model, None

    gps_tags: Dict[str, object] = {}
    for key, value in gps_info.items():
        name = _ExifTags.GPSTAGS.get(key, str(key))
        gps_tags[name] = value

    lat = gps_tags.get("GPSLatitude")
    lat_ref = gps_tags.get("GPSLatitudeRef")
    lon = gps_tags.get("GPSLongitude")
    lon_ref = gps_tags.get("GPSLongitudeRef")

    def _to_degrees(value) -> Optional[float]:
        try:
            d, m, s = value
        except Exception:
            return None
        try:
            d_val = float(d)
            m_val = float(m)
            s_val = float(s)
        except Exception:
            return None
        return d_val + (m_val / 60.0) + (s_val / 3600.0)

    if not (lat and lon and isinstance(lat_ref, str) and isinstance(lon_ref, str)):
        return exif_datetime, camera_model, None

    lat_deg = _to_degrees(lat)
    lon_deg = _to_degrees(lon)
    if lat_deg is None or lon_deg is None:
        return exif_datetime, camera_model, None

    if lat_ref.upper() == "S":
        lat_deg = -lat_deg
    if lon_ref.upper() == "W":
        lon_deg = -lon_deg

    gps_payload: Dict[str, object] = {
        "latitude": lat_deg,
        "longitude": lon_deg,
        "latitude_ref": lat_ref,
        "longitude_ref": lon_ref,
    }
    return exif_datetime, camera_model, gps_payload


def _siglip_label_group(
    label: str,
    label_groups: Dict[str, Sequence[str]] | None = None,
    canonical_map: Dict[str, str] | None = None,
) -> Optional[str]:
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


def load_run_journal(cache_root: Path) -> Optional[RunJournalRecord]:
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


class PreprocessingPipeline:
    """Orchestrates the M1 preprocessing steps for one or more album roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with application settings.

        Args:
            settings: Optional pre-loaded Settings instance. When omitted,
                configuration is loaded from ``config/settings.yaml``.
        """

        self._settings = settings or load_settings()
        self._logger = get_logger(__name__, extra={"component": "preprocess"})
        self._cache_root: Optional[Path] = None
        self._journal: Optional[RunJournalRecord] = None

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

    def run(self, roots: Sequence[Path], primary_db_path: Path, projection_db_path: Path) -> None:
        """Run the preprocessing pipeline for the given album roots.

        The initial implementation focuses on wiring and logging. Individual
        stages are implemented as stubs that can be expanded in dedicated
        iterations.

        Args:
            roots: Album root directories to scan.
            primary_db_path: Path to the primary operational database.
            projection_db_path: Path to the projection database for model outputs.
        """

        cache_root = projection_db_path.parent
        self._cache_root = cache_root
        # Ensure cache manifest exists and reflects current settings; this also
        # provides a single place to bump cache format versions in future.
        write_cache_manifest(cache_root, self._settings)
        self._logger.info(
            "pipeline_start",
            extra={
                "roots": [str(root) for root in roots],
                "primary_db": str(primary_db_path),
                "projection_db": str(projection_db_path),
            },
        )

        journal = load_run_journal(cache_root)
        self._journal = journal
        if journal is not None:
            self._logger.info(
                "pipeline_resume",
                extra={"stage": journal.stage, "cursor_image_id": journal.cursor_image_id},
            )

        with open_primary_session(primary_db_path) as primary_session, open_projection_session(projection_db_path) as projection_session:
            stage_plan: List[tuple[str, Callable[[], None], bool]] = [
                (
                    "scan_and_hash",
                    lambda: self._run_scan_and_hash(roots, primary_session),
                    True,
                ),
                (
                    "phash_and_duplicates",
                    lambda: self._run_perceptual_hashing_and_duplicates(primary_session, projection_session),
                    True,
                ),
                ("thumbnails", lambda: self._run_thumbnails(primary_session), True),
                (
                    "embeddings_and_captions",
                    lambda: self._run_embeddings_and_captions(primary_session, projection_session),
                    False,
                ),
                ("scene_classification", lambda: self._run_scene_classification(primary_session, projection_session), False),
            ]

            if self._settings.pipeline.run_detection and self._settings.models.detection.enabled:
                stage_plan.append(
                    (
                        "region_detection",
                        lambda: self._run_region_detection_and_reranking(primary_session),
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

    def _run_scan_and_hash(self, roots: Sequence[Path], primary_session) -> None:
        """Scan album roots and populate the images table with content hashes."""

        self._logger.info("scan_and_hash_start", extra={})
        files: List[FileInfo] = list(scan_roots(roots))
        self._logger.info("scan_and_hash_files_discovered", extra={"file_count": len(files)})

        path_to_image_id: Dict[str, str] = {}
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
                    primary_session.add(old_image)

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
                    )
                )

            path_to_image_id[path_str] = new_image_id

        primary_session.commit()

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

            existing_paths: List[str] = []
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
        self._logger.info("scan_and_hash_complete", extra={"file_count": len(files)})

    def _run_thumbnails(self, primary_session) -> None:
        """Generate web-friendly thumbnails for active images under cache/images/thumbnails/."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        thumbnails_dir = cache_root / "images" / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info("thumbnails_start", extra={"thumbnails_dir": str(thumbnails_dir)})

        rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active")
        )

        created = 0
        skipped = 0

        try:
            from PIL import Image as _ImageThumb
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error("thumbnail_pil_import_error", extra={"error": str(exc)})
            return

        max_side = max(1, int(self._settings.pipeline.thumbnail_size))
        quality = int(self._settings.pipeline.thumbnail_quality)

        for row in rows:
            image_id = row.image_id
            path_str = row.primary_path
            thumb_path = thumbnails_dir / f"{image_id}.jpg"

            if thumb_path.exists():
                skipped += 1
                continue

            try:
                image = _ImageThumb.open(path_str).convert("RGB")
            except Exception as exc:
                self._logger.error(
                    "thumbnail_image_open_error",
                    extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                )
                continue

            try:
                max_size = (max_side, max_side)
                resample_attr = getattr(_ImageThumb, "Resampling", None)
                if resample_attr is not None:
                    resample = resample_attr.LANCZOS
                else:
                    resample = _ImageThumb.LANCZOS

                image.thumbnail(max_size, resample=resample)
                image.save(thumb_path, format="JPEG", quality=quality)
                created += 1
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error(
                    "thumbnail_generate_error",
                    extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                )
                try:
                    if thumb_path.exists():
                        thumb_path.unlink()
                except OSError:
                    pass
                continue

        self._logger.info(
            "thumbnails_complete",
            extra={"created": created, "skipped_existing": skipped, "thumbnails_dir": str(thumbnails_dir)},
        )

    def _run_region_detection_and_reranking(self, primary_session, target_image_ids: Optional[Sequence[str]] = None) -> None:
        """Run optional object-level detection and prepare region metadata.

        When ``target_image_ids`` is provided, restrict processing to that subset of
        active images; otherwise, process all active images as in the full pipeline.
        """

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
            Detection as RegionDetection,
            OwlVitDetector,
            build_owlvit_detector,
            detection_priority,
            filter_secondary_regions_by_priority,
            non_max_suppression,
        )
        from vibe_photos.ml.models import get_siglip_embedding_model

        cache_root = self._cache_root
        regions_dir = cache_root / "regions"
        regions_dir.mkdir(parents=True, exist_ok=True)

        if detection_cfg.backend != "owlvit":
            self._logger.info(
                "region_detection_backend_unsupported",
                extra={"backend": detection_cfg.backend},
            )
            return

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
        label_groups = siglip_labels_cfg.label_groups
        canonical_group_map: Dict[str, str] = {}
        for group_name, members in label_groups.items():
            for member in members:
                key = _normalize_siglip_label(member)
                if key and key not in canonical_group_map:
                    canonical_group_map[key] = group_name

        import numpy as np
        from PIL import Image as _Image

        active_rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active").order_by(Image.image_id)
        )
        paths_by_id: Dict[str, str] = {row.image_id: row.primary_path for row in active_rows}

        if target_image_ids is not None:
            allowed_ids = {str(image_id) for image_id in target_image_ids}
            paths_by_id = {image_id: path for image_id, path in paths_by_id.items() if image_id in allowed_ids}

        caption_cfg = self._settings.models.caption
        caption_model_name = caption_cfg.resolved_model_name()
        caption_rows = primary_session.execute(
            select(ImageCaption.image_id, ImageCaption.caption).where(ImageCaption.model_name == caption_model_name)
        )
        captions_by_id: Dict[str, str] = {row.image_id: row.caption for row in caption_rows}

        if not paths_by_id:
            self._logger.info("region_detection_noop", extra={})
            return

        process_ids = sorted(paths_by_id.keys())

        if self._settings.pipeline.skip_duplicates_for_heavy_models:
            duplicate_rows = primary_session.execute(select(ImageNearDuplicate.duplicate_image_id))
            duplicate_ids = {row.duplicate_image_id for row in duplicate_rows}
            process_ids = [image_id for image_id in process_ids if image_id not in duplicate_ids]

        # Pre-compute text embeddings for candidate labels once.
        import torch

        text_inputs = siglip_processor(text=candidate_labels, padding=True, return_tensors="pt")
        text_inputs = text_inputs.to(siglip_device)
        with torch.no_grad():
            text_emb = siglip_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        candidate_label_groups = [
            _siglip_label_group(label, canonical_map=canonical_group_map) for label in candidate_labels
        ]
        num_candidate_labels = len(candidate_labels)
        base_min_prob = max(0.0, float(detection_cfg.siglip_min_prob))
        min_margin = max(0.0, float(detection_cfg.siglip_min_margin))

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
                detections: List[RegionDetection] = detector.detect(image=image, prompts=candidate_labels)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error(
                    "region_detection_backend_error",
                    extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                )
                continue

            detections = non_max_suppression(detections, iou_threshold=nms_iou_threshold)

            # Remove any existing regions for this image_id to keep results consistent with the current model.
            primary_session.execute(
                delete(ImageRegion).where(ImageRegion.image_id == image_id)
            )

            width, height = image.size

            # Prepare SigLIP inputs for region crops.
            region_images: List["_Image.Image"] = []
            region_indices: List[int] = []
            priorities: List[float] = []

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
                pairs = list(zip(detections, priorities))
                pairs.sort(key=lambda pair: pair[1], reverse=True)
                detections, priorities = zip(*pairs)
                detections = list(detections)
                priorities = list(priorities)

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

            refined_labels: Dict[int, str] = {}
            refined_scores: Dict[int, float] = {}

            if region_images:
                region_inputs = siglip_processor(images=region_images, return_tensors="pt")
                region_inputs = region_inputs.to(siglip_device)

                with torch.no_grad():
                    region_emb = siglip_model.get_image_features(**region_inputs)

                region_emb = region_emb / region_emb.norm(dim=-1, keepdim=True)
                logits = region_emb @ text_emb.T
                probs = torch.softmax(logits, dim=-1)
                probs_cpu = probs.detach().cpu().numpy()

                for idx, region_index in enumerate(region_indices):
                    row = probs_cpu[idx]

                    det_label = detections[region_index].label
                    det_group = _siglip_label_group(det_label, canonical_map=canonical_group_map)

                    if det_group is None:
                        # When the detector label cannot be mapped to a semantic group,
                        # skip SigLIP refinement to avoid cross-group overrides.
                        continue

                    group_indices = [
                        label_index
                        for label_index, group_name in enumerate(candidate_label_groups)
                        if group_name == det_group
                    ]

                    if not group_indices:
                        # No candidate labels in the same group; keep detector-only result.
                        continue

                    scores = [(label_index, float(row[label_index])) for label_index in group_indices]
                    scores.sort(key=lambda item: item[1], reverse=True)

                    best_pos, best_prob = scores[0]
                    second_prob = scores[1][1] if len(scores) > 1 else 0.0

                    group_size = len(group_indices)
                    uniform_prob = 1.0 / float(group_size) if group_size > 0 else 0.0
                    min_prob = max(base_min_prob, uniform_prob + 0.05)

                    if best_prob < min_prob or (best_prob - second_prob) < min_margin:
                        continue

                    best_label = candidate_labels[best_pos]
                    refined_labels[region_index] = best_label
                    refined_scores[region_index] = best_prob

            # Persist regions to SQLite and cache JSON.
            region_payloads = []

            for new_index, det in enumerate(detections):
                refined_label = refined_labels.get(new_index)
                refined_score = refined_scores.get(new_index)

                primary_session.add(
                    ImageRegion(
                        image_id=image_id,
                        region_index=new_index,
                        x_min=float(det.bbox.x_min),
                        y_min=float(det.bbox.y_min),
                        x_max=float(det.bbox.x_max),
                        y_max=float(det.bbox.y_max),
                        detector_label=det.label,
                        detector_score=float(det.score),
                        refined_label=refined_label,
                        refined_score=refined_score,
                        backend=det.backend,
                        model_name=det.model_name,
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
                        "refined_label": refined_label,
                        "refined_score": refined_score,
                    }
                )

            primary_session.commit()
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

        self._logger.info(
            "region_detection_complete",
            extra={"regions_created": updated_rows},
        )

    def _run_scene_classification(self, primary_session, _projection_session) -> None:
        """Run lightweight scene classification based on cached embeddings."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._logger.info("scene_classification_start", extra={})

        cache_root = self._cache_root

        classifier: SceneClassifierWithAttributes = build_scene_classifier(self._settings)
        classifier_version = classifier.classifier_version

        embedding_model_name = self._settings.models.embedding.resolved_model_name()

        existing_versions = {
            row.image_id: row.classifier_version
            for row in primary_session.execute(select(ImageScene.image_id, ImageScene.classifier_version))
        }

        embedding_rows = primary_session.execute(
            select(ImageEmbedding.image_id, ImageEmbedding.embedding_path).where(ImageEmbedding.model_name == embedding_model_name)
        )
        embedding_path_by_id = {row.image_id: row.embedding_path for row in embedding_rows}

        candidate_ids = [
            row.image_id
            for row in primary_session.execute(
                select(Image.image_id).where(Image.status == "active").order_by(Image.image_id)
            )
        ]

        target_ids: List[str] = []
        for image_id in candidate_ids:
            if image_id not in embedding_path_by_id:
                continue
            if image_id not in existing_versions:
                target_ids.append(image_id)
                continue
            if existing_versions[image_id] != classifier_version:
                target_ids.append(image_id)

        if self._journal is not None and self._journal.stage == "scene_classification" and self._journal.cursor_image_id:
            cursor_id = self._journal.cursor_image_id
            before = len(target_ids)
            target_ids = [image_id for image_id in target_ids if image_id > cursor_id]
            self._logger.info(
                "scene_classification_resume_from_cursor",
                extra={"cursor_image_id": cursor_id, "skipped": before - len(target_ids)},
            )

        if not target_ids:
            self._logger.info("scene_classification_noop", extra={})
            return

        import numpy as np
        import torch

        detections_dir = cache_root / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

        batch_size = max(1, self._settings.models.embedding.batch_size)
        updated_rows = 0

        for batch_start in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[batch_start : batch_start + batch_size]
            embeddings: List[Tensor] = []
            valid_ids: List[str] = []

            for image_id in batch_ids:
                rel_path = embedding_path_by_id[image_id]
                emb_path = cache_root / "embeddings" / rel_path
                try:
                    vec = np.load(emb_path)
                except Exception as exc:
                    self._logger.error(
                        "scene_embedding_load_error",
                        extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
                    )
                    continue

                try:
                    tensor = torch.from_numpy(vec).float()
                except Exception as exc:
                    self._logger.error(
                        "scene_embedding_tensor_error",
                        extra={"image_id": image_id, "error": str(exc)},
                    )
                    continue

                embeddings.append(tensor)
                valid_ids.append(image_id)

            if not embeddings:
                continue

            attributes_list = classifier.classify_batch(embeddings)

            now = time.time()
            for image_id, attributes in zip(valid_ids, attributes_list):
                stmt = sqlite_insert(ImageScene).values(
                    image_id=image_id,
                    scene_type=attributes.scene_type,
                    scene_confidence=attributes.scene_confidence,
                    has_text=bool(attributes.has_text),
                    has_person=bool(attributes.has_person),
                    is_screenshot=bool(attributes.is_screenshot),
                    is_document=bool(attributes.is_document),
                    classifier_name=attributes.classifier_name,
                    classifier_version=attributes.classifier_version,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[ImageScene.image_id],
                    set_={
                        "scene_type": stmt.excluded.scene_type,
                        "scene_confidence": stmt.excluded.scene_confidence,
                        "has_text": stmt.excluded.has_text,
                        "has_person": stmt.excluded.has_person,
                        "is_screenshot": stmt.excluded.is_screenshot,
                        "is_document": stmt.excluded.is_document,
                        "classifier_name": stmt.excluded.classifier_name,
                        "classifier_version": stmt.excluded.classifier_version,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                primary_session.execute(stmt)

                detection_payload = {
                    "image_id": image_id,
                    "scene_type": attributes.scene_type,
                    "scene_confidence": attributes.scene_confidence,
                    "has_text": attributes.has_text,
                    "has_person": attributes.has_person,
                    "is_screenshot": attributes.is_screenshot,
                    "is_document": attributes.is_document,
                    "classifier_name": attributes.classifier_name,
                    "classifier_version": attributes.classifier_version,
                    "updated_at": now,
                }
                cache_path = detections_dir / f"{image_id}.json"
                cache_path.write_text(json.dumps(detection_payload), encoding="utf-8")
                updated_rows += 1

            save_run_journal(
                cache_root,
                RunJournalRecord(stage="scene_classification", cursor_image_id=batch_ids[-1], updated_at=time.time()),
            )

        primary_session.commit()
        self._logger.info("scene_classification_complete", extra={"updated_rows": updated_rows})

    def _run_embeddings_and_captions(self, primary_session, _projection_session) -> None:
        """Compute SigLIP embeddings and BLIP captions for canonical images."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._logger.info("embeddings_and_captions_start", extra={})

        cache_root = self._cache_root

        embedding_cfg = self._settings.models.embedding
        caption_cfg = self._settings.models.caption

        embedding_model_name = embedding_cfg.resolved_model_name()
        caption_model_name = caption_cfg.resolved_model_name()

        siglip_processor, siglip_model, siglip_device = get_siglip_embedding_model(settings=self._settings)
        blip_processor, blip_model, blip_device = get_blip_caption_model(settings=self._settings)

        active_rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(Image.status == "active").order_by(Image.image_id)
        )
        paths_by_id: Dict[str, str] = {row.image_id: row.primary_path for row in active_rows}

        if not paths_by_id:
            self._logger.info("embeddings_and_captions_noop", extra={})
            return

        process_ids = sorted(paths_by_id.keys())

        if self._settings.pipeline.skip_duplicates_for_heavy_models:
            duplicate_rows = primary_session.execute(select(ImageNearDuplicate.duplicate_image_id))
            duplicate_ids = {row.duplicate_image_id for row in duplicate_rows}
            process_ids = [image_id for image_id in process_ids if image_id not in duplicate_ids]

        if self._journal is not None and self._journal.stage == "embeddings_and_captions" and self._journal.cursor_image_id:
            cursor_id = self._journal.cursor_image_id
            before = len(process_ids)
            process_ids = [image_id for image_id in process_ids if image_id > cursor_id]
            self._logger.info(
                "embeddings_resume_from_cursor",
                extra={"cursor_image_id": cursor_id, "skipped": before - len(process_ids)},
            )

        existing_embedding_ids = {
            row.image_id
            for row in primary_session.execute(select(ImageEmbedding.image_id).where(ImageEmbedding.model_name == embedding_model_name))
        }

        existing_caption_ids = {
            row.image_id
            for row in primary_session.execute(select(ImageCaption.image_id).where(ImageCaption.model_name == caption_model_name))
        }

        embedding_targets = [image_id for image_id in process_ids if image_id not in existing_embedding_ids]
        caption_targets = [image_id for image_id in process_ids if image_id not in existing_caption_ids]

        embeddings_dir = cache_root / "embeddings"
        captions_dir = cache_root / "captions"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import torch

        now = time.time()

        embedding_batch_size = max(1, embedding_cfg.batch_size)
        total_embeddings = 0

        for batch_start in range(0, len(embedding_targets), embedding_batch_size):
            batch_ids = embedding_targets[batch_start : batch_start + embedding_batch_size]
            images: List["Image.Image"] = []
            valid_ids: List[str] = []

            from PIL import Image as _Image

            for image_id in batch_ids:
                path_str = paths_by_id[image_id]
                try:
                    img = _Image.open(path_str).convert("RGB")
                except Exception as exc:
                    self._logger.error(
                        "embedding_image_open_error",
                        extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                    )
                    continue

                images.append(img)
                valid_ids.append(image_id)

            if not images:
                continue

            inputs = siglip_processor(images=images, return_tensors="pt")
            inputs = inputs.to(siglip_device)

            with torch.no_grad():
                features = siglip_model.get_image_features(**inputs)

            for idx, image_id in enumerate(valid_ids):
                emb = features[idx]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                vec = emb.detach().cpu().numpy().astype(np.float32)

                rel_path = f"{image_id}.npy"
                emb_path = embeddings_dir / rel_path
                np.save(emb_path, vec)

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

                stmt = sqlite_insert(ImageEmbedding).values(
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
                total_embeddings += 1

            primary_session.commit()
            if batch_ids:
                save_run_journal(
                    cache_root,
                    RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=batch_ids[-1], updated_at=time.time()),
                )

        caption_batch_size = max(1, caption_cfg.batch_size)
        total_captions = 0

        from PIL import Image as _ImageCaption

        for batch_start in range(0, len(caption_targets), caption_batch_size):
            batch_ids = caption_targets[batch_start : batch_start + caption_batch_size]
            images: List["_ImageCaption.Image"] = []
            valid_ids: List[str] = []

            for image_id in batch_ids:
                path_str = paths_by_id[image_id]
                try:
                    img = _ImageCaption.open(path_str).convert("RGB")
                except Exception as exc:
                    self._logger.error(
                        "caption_image_open_error",
                        extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                    )
                    continue

                images.append(img)
                valid_ids.append(image_id)

            if not images:
                continue

            inputs = blip_processor(images=images, return_tensors="pt")
            inputs = inputs.to(blip_device)

            with torch.no_grad():
                generated_ids = blip_model.generate(**inputs)

            for idx, image_id in enumerate(valid_ids):
                token_ids = generated_ids[idx]
                caption_text = blip_processor.decode(token_ids, skip_special_tokens=True)

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

                stmt = sqlite_insert(ImageCaption).values(
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
                total_captions += 1

            primary_session.commit()

            if batch_ids:
                save_run_journal(
                    cache_root,
                    RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=batch_ids[-1], updated_at=time.time()),
                )

        primary_session.commit()
        self._logger.info(
            "embeddings_and_captions_complete",
            extra={"embeddings_created": total_embeddings, "captions_created": total_captions},
        )

    def process_embedding_task(self, primary_session, image_id: str) -> None:
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

        from PIL import Image as _Image
        import numpy as np
        import torch

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

        stmt = sqlite_insert(ImageEmbedding).values(
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

    def process_caption_task(self, primary_session, image_id: str) -> None:
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

        from PIL import Image as _ImageCaption
        import torch

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

        stmt = sqlite_insert(ImageCaption).values(
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

    def process_region_detection_task(self, primary_session, image_id: str) -> None:
        """Run region detection for a single image via the detection pipeline."""

        if not self._settings.pipeline.run_detection or not self._settings.models.detection.enabled:
            return

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._run_region_detection_and_reranking(primary_session, target_image_ids=[image_id])

    def _run_perceptual_hashing_and_duplicates(self, primary_session, _projection_session) -> None:
        """Compute perceptual hashes and rebuild near-duplicate groups."""

        self._logger.info("phash_and_duplicates_start", extra={})

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        cache_root = self._cache_root
        metadata_dir = cache_root / "images" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()

        rows = primary_session.execute(
            select(Image.image_id, Image.primary_path).where(
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
        )

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

            exif_datetime: Optional[str]
            camera_model: Optional[str]
            gps_payload: Optional[Dict[str, object]]

            try:
                exif_datetime, camera_model, gps_payload = _extract_exif_and_gps(
                    image, datetime_format=self._settings.pipeline.exif_datetime_format
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("exif_parse_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                exif_datetime = None
                camera_model = None
                gps_payload = None

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

            metadata_payload = {
                "image_id": image_id,
                "primary_path": path_str,
                "exif_datetime": exif_datetime,
                "camera_model": camera_model,
                "gps": gps_payload,
                "updated_at": now,
            }
            try:
                metadata_path = metadata_dir / f"{image_id}.json"
                metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
            except OSError as exc:
                self._logger.error(
                    "exif_metadata_write_error",
                    extra={"image_id": image_id, "path": str(metadata_dir), "error": str(exc)},
                )

        primary_session.commit()
        self._logger.info("phash_update_complete", extra={"updated_count": phash_updated})

        rows = primary_session.execute(
            select(Image.image_id, Image.phash, Image.mtime).where(
                and_(Image.status == "active", Image.phash.is_not(None), Image.phash_algo == PHASH_ALGO)
            )
        )

        threshold = self._settings.pipeline.phash_hamming_threshold
        if threshold <= 0:
            threshold = 12

        # Collect pHash-bearing images with mtime for anchor selection.
        items: List[Dict[str, object]] = []
        for row in rows:
            if not row.phash:
                continue
            try:
                ph_int = int(row.phash, 16)
            except (TypeError, ValueError):
                continue
            items.append({"image_id": row.image_id, "phash_int": ph_int, "mtime": float(row.mtime or 0.0)})

        if not items:
            near_pairs: List[tuple[str, str, int]] = []
        else:
            # Newest-first order for anchor selection.
            items.sort(key=lambda item: item["mtime"], reverse=True)

            ids: List[str] = [str(item["image_id"]) for item in items]
            ph_ints: List[int] = [int(item["phash_int"]) for item in items]

            near_pairs = []
            used: set[str] = set()
            n = len(ids)
            for i in range(n):
                anchor_id = ids[i]
                if anchor_id in used:
                    continue
                anchor_int = ph_ints[i]
                used.add(anchor_id)
                for j in range(i + 1, n):
                    candidate_id = ids[j]
                    if candidate_id in used:
                        continue
                    distance = int((anchor_int ^ ph_ints[j]).bit_count())
                    if distance <= threshold:
                        near_pairs.append((anchor_id, candidate_id, distance))
                        used.add(candidate_id)

        primary_session.execute(delete(ImageNearDuplicate))
        for anchor_id, duplicate_id, distance in near_pairs:
            primary_session.add(
                ImageNearDuplicate(
                    anchor_image_id=anchor_id,
                    duplicate_image_id=duplicate_id,
                    phash_distance=distance,
                    created_at=now,
                )
            )
        primary_session.commit()

        self._logger.info(
            "phash_and_duplicates_complete",
            extra={"phash_updated": phash_updated, "near_duplicate_pairs": len(near_pairs), "threshold": threshold},
        )


__all__ = ["PreprocessingPipeline", "RunJournalRecord", "load_run_journal", "save_run_journal"]
