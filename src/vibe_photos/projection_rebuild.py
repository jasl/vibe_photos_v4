"""Helpers to rebuild projection tables from on-disk cache artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.cache_manifest import CACHE_FORMAT_VERSION, load_cache_manifest
from vibe_photos.db import ImageCaption, ImageEmbedding, ImageRegion, ImageScene


LOGGER = get_logger(__name__)


def _cache_trusted(cache_root: Path) -> bool:
    """Return True if the cache manifest matches the current format version."""

    manifest = load_cache_manifest(cache_root)
    if manifest is None:
        # No manifest yet; treat cache as best-effort trusted.
        return True
    return manifest.cache_format_version == CACHE_FORMAT_VERSION


def rebuild_embeddings_from_cache(cache_root: Path, session: Session) -> None:
    """Rebuild ImageEmbedding rows from ``cache/embeddings`` artifacts."""

    embeddings_dir = cache_root / "embeddings"
    if not embeddings_dir.exists():
        LOGGER.info("rebuild_embeddings_noop", extra={"reason": "missing_embeddings_dir", "path": str(embeddings_dir)})
        return

    if not _cache_trusted(cache_root):
        LOGGER.info(
            "rebuild_embeddings_skip_untrusted_cache",
            extra={"cache_root": str(cache_root), "cache_format_version": CACHE_FORMAT_VERSION},
        )
        return

    session.execute(delete(ImageEmbedding))
    session.commit()

    total = 0
    for path in sorted(embeddings_dir.glob("*.npy")):
        stem = path.stem
        image_id = stem

        meta_path = embeddings_dir / f"{stem}.json"
        if not meta_path.exists():
            LOGGER.warning(
                "embedding_meta_missing",
                extra={"image_id": image_id, "npy_path": str(path), "meta_path": str(meta_path)},
            )
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error(
                "embedding_meta_load_error",
                extra={"image_id": image_id, "path": str(meta_path), "error": str(exc)},
            )
            continue

        model_name = str(meta.get("model_name") or "")
        model_backend = str(meta.get("model_backend") or "")
        embedding_dim_raw = meta.get("embedding_dim")
        try:
            embedding_dim = int(embedding_dim_raw)
        except (TypeError, ValueError):
            LOGGER.error(
                "embedding_meta_dim_error",
                extra={"image_id": image_id, "path": str(meta_path), "value": embedding_dim_raw},
            )
            continue

        rel_path = path.name

        try:
            vec = np.load(path)
        except Exception as exc:
            LOGGER.error(
                "embedding_npy_load_error",
                extra={"image_id": image_id, "path": str(path), "error": str(exc)},
            )
            continue

        if vec.shape[-1] != embedding_dim:
            LOGGER.error(
                "embedding_dim_mismatch",
                extra={"image_id": image_id, "path": str(path), "expected": embedding_dim, "actual": int(vec.shape[-1])},
            )
            continue

        updated_at_raw = meta.get("updated_at")
        try:
            updated_at = float(updated_at_raw)
        except (TypeError, ValueError):
            updated_at = 0.0

        session.add(
            ImageEmbedding(
                image_id=image_id,
                model_name=model_name,
                embedding_path=rel_path,
                embedding_dim=embedding_dim,
                model_backend=model_backend,
                updated_at=updated_at,
            )
        )
        total += 1

    session.commit()
    LOGGER.info("rebuild_embeddings_complete", extra={"rows": total})


def rebuild_captions_from_cache(cache_root: Path, session: Session) -> None:
    """Rebuild ImageCaption rows from ``cache/captions`` artifacts."""

    captions_dir = cache_root / "captions"
    if not captions_dir.exists():
        LOGGER.info("rebuild_captions_noop", extra={"reason": "missing_captions_dir", "path": str(captions_dir)})
        return

    if not _cache_trusted(cache_root):
        LOGGER.info(
            "rebuild_captions_skip_untrusted_cache",
            extra={"cache_root": str(cache_root), "cache_format_version": CACHE_FORMAT_VERSION},
        )
        return

    session.execute(delete(ImageCaption))
    session.commit()

    total = 0
    for path in sorted(captions_dir.glob("*.json")):
        try:
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error(
                "caption_payload_load_error",
                extra={"path": str(path), "error": str(exc)},
            )
            continue

        image_id = str(payload.get("image_id") or path.stem)
        caption = payload.get("caption")
        if not isinstance(caption, str) or not caption:
            continue

        model_name = str(payload.get("model_name") or "")
        model_backend = str(payload.get("model_backend") or "")
        updated_raw = payload.get("updated_at")
        try:
            updated_at = float(updated_raw)
        except (TypeError, ValueError):
            updated_at = 0.0

        session.add(
            ImageCaption(
                image_id=image_id,
                model_name=model_name,
                caption=caption,
                model_backend=model_backend,
                updated_at=updated_at,
            )
        )
        total += 1

    session.commit()
    LOGGER.info("rebuild_captions_complete", extra={"rows": total})


def rebuild_scenes_from_cache(cache_root: Path, session: Session) -> None:
    """Rebuild ImageScene rows from ``cache/detections`` classifier outputs."""

    detections_dir = cache_root / "detections"
    if not detections_dir.exists():
        LOGGER.info("rebuild_scenes_noop", extra={"reason": "missing_detections_dir", "path": str(detections_dir)})
        return

    if not _cache_trusted(cache_root):
        LOGGER.info(
            "rebuild_scenes_skip_untrusted_cache",
            extra={"cache_root": str(cache_root), "cache_format_version": CACHE_FORMAT_VERSION},
        )
        return

    session.execute(delete(ImageScene))
    session.commit()

    total = 0
    for path in sorted(detections_dir.glob("*.json")):
        try:
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error("scene_payload_load_error", extra={"path": str(path), "error": str(exc)})
            continue

        image_id = str(payload.get("image_id") or path.stem)
        scene_type = payload.get("scene_type")
        if not isinstance(scene_type, str) or not scene_type:
            continue

        scene_conf = payload.get("scene_confidence")
        try:
            scene_confidence = float(scene_conf) if scene_conf is not None else None
        except (TypeError, ValueError):
            scene_confidence = None

        has_text = bool(payload.get("has_text"))
        has_person = bool(payload.get("has_person"))
        is_screenshot = bool(payload.get("is_screenshot"))
        is_document = bool(payload.get("is_document"))
        classifier_name = str(payload.get("classifier_name") or "")
        classifier_version = str(payload.get("classifier_version") or "")
        updated_raw = payload.get("updated_at")
        try:
            updated_at = float(updated_raw)
        except (TypeError, ValueError):
            updated_at = 0.0

        session.add(
            ImageScene(
                image_id=image_id,
                scene_type=scene_type,
                scene_confidence=scene_confidence,
                has_text=has_text,
                has_person=has_person,
                is_screenshot=is_screenshot,
                is_document=is_document,
                classifier_name=classifier_name,
                classifier_version=classifier_version,
                updated_at=updated_at,
            )
        )
        total += 1

    session.commit()
    LOGGER.info("rebuild_scenes_complete", extra={"rows": total})


def rebuild_regions_from_cache(cache_root: Path, session: Session) -> None:
    """Rebuild ImageRegion rows from ``cache/regions`` artifacts."""

    regions_dir = cache_root / "regions"
    if not regions_dir.exists():
        LOGGER.info("rebuild_regions_noop", extra={"reason": "missing_regions_dir", "path": str(regions_dir)})
        return

    if not _cache_trusted(cache_root):
        LOGGER.info(
            "rebuild_regions_skip_untrusted_cache",
            extra={"cache_root": str(cache_root), "cache_format_version": CACHE_FORMAT_VERSION},
        )
        return

    session.execute(delete(ImageRegion))
    session.commit()

    total = 0
    for path in sorted(regions_dir.glob("*.json")):
        try:
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error("region_payload_load_error", extra={"path": str(path), "error": str(exc)})
            continue

        image_id = str(payload.get("image_id") or path.stem)
        detections: List[Dict[str, Any]] = payload.get("detections") or []
        backend = str(payload.get("backend") or "")
        model_name = str(payload.get("model_name") or "")
        updated_raw = payload.get("updated_at")
        try:
            updated_at = float(updated_raw)
        except (TypeError, ValueError):
            updated_at = 0.0

        for index, det in enumerate(detections):
            bbox = det.get("bbox") or {}

            def _as_float(key: str) -> float:
                val = bbox.get(key)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            x_min = _as_float("x_min")
            y_min = _as_float("y_min")
            x_max = _as_float("x_max")
            y_max = _as_float("y_max")

            detector_label = str(det.get("detector_label") or "")
            detector_score_raw = det.get("detector_score")
            try:
                detector_score = float(detector_score_raw)
            except (TypeError, ValueError):
                detector_score = 0.0

            refined_label_raw = det.get("refined_label")
            refined_label = str(refined_label_raw) if refined_label_raw is not None else None
            refined_score_raw = det.get("refined_score")
            try:
                refined_score = float(refined_score_raw) if refined_score_raw is not None else None
            except (TypeError, ValueError):
                refined_score = None

            session.add(
                ImageRegion(
                    image_id=image_id,
                    region_index=index,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    detector_label=detector_label,
                    detector_score=detector_score,
                    refined_label=refined_label,
                    refined_score=refined_score,
                    backend=backend,
                    model_name=model_name,
                    updated_at=updated_at,
                )
            )
            total += 1

    session.commit()
    LOGGER.info("rebuild_regions_complete", extra={"rows": total})


__all__ = [
    "rebuild_embeddings_from_cache",
    "rebuild_captions_from_cache",
    "rebuild_scenes_from_cache",
    "rebuild_regions_from_cache",
]

