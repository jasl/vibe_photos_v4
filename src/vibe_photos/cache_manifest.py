"""Helpers for cache manifest and format versioning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logging import get_logger
from vibe_photos.config import Settings


LOGGER = get_logger(__name__)

# Bump this when cache payload formats change in a non-backwards compatible way.
CACHE_FORMAT_VERSION: int = 1


@dataclass
class CacheManifest:
    """Lightweight description of the cache layout and version."""

    cache_format_version: int
    created_at: float
    models: Dict[str, Any]
    pipeline: Dict[str, Any]


def _manifest_path(cache_root: Path) -> Path:
    return cache_root / "manifest.json"


def load_cache_manifest(cache_root: Path) -> Optional[CacheManifest]:
    """Load the cache manifest from ``cache/manifest.json`` if present."""

    path = _manifest_path(cache_root)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("cache_manifest_load_error", extra={"path": str(path), "error": str(exc)})
        return None

    try:
        version = int(raw.get("cache_format_version"))
    except (TypeError, ValueError):
        version = CACHE_FORMAT_VERSION

    created_raw = raw.get("created_at")
    try:
        created_at = float(created_raw)
    except (TypeError, ValueError):
        created_at = time.time()

    models = raw.get("models") or {}
    pipeline = raw.get("pipeline") or {}

    return CacheManifest(cache_format_version=version, created_at=created_at, models=models, pipeline=pipeline)


def write_cache_manifest(cache_root: Path, settings: Settings) -> CacheManifest:
    """Create or update the cache manifest to reflect current settings."""

    cache_root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(cache_root)

    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "created_at": time.time(),
        "models": {
            "embedding": {
                "backend": settings.models.embedding.backend,
                "model_name": settings.models.embedding.resolved_model_name(),
                "batch_size": settings.models.embedding.batch_size,
            },
            "caption": {
                "backend": settings.models.caption.backend,
                "model_name": settings.models.caption.resolved_model_name(),
                "batch_size": settings.models.caption.batch_size,
            },
            "detection": {
                "enabled": settings.models.detection.enabled,
                "backend": settings.models.detection.backend,
                "model_name": settings.models.detection.model_name,
                "max_regions_per_image": settings.models.detection.max_regions_per_image,
                "score_threshold": settings.models.detection.score_threshold,
            },
        },
        "pipeline": {
            "thumbnail_size": settings.pipeline.thumbnail_size,
            "thumbnail_quality": settings.pipeline.thumbnail_quality,
            "phash_hamming_threshold": settings.pipeline.phash_hamming_threshold,
            "run_detection": settings.pipeline.run_detection,
            "skip_duplicates_for_heavy_models": settings.pipeline.skip_duplicates_for_heavy_models,
        },
    }

    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("cache_manifest_write_error", extra={"path": str(path), "error": str(exc)})

    return CacheManifest(
        cache_format_version=CACHE_FORMAT_VERSION,
        created_at=payload["created_at"],
        models=payload["models"],
        pipeline=payload["pipeline"],
    )


__all__ = ["CACHE_FORMAT_VERSION", "CacheManifest", "load_cache_manifest", "write_cache_manifest"]

