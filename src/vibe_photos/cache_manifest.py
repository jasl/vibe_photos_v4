"""Helpers for cache manifest and format versioning."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from utils.logging import get_logger
from vibe_photos.config import Settings

LOGGER = get_logger(__name__)

# Bump this when cache payload formats change in a non-backwards compatible way.
CACHE_FORMAT_VERSION: int = 1

_CACHE_SUBDIRS = [
    "artifacts",
    "regions",
    "label_text_prototypes",
]
_CACHE_FILES = ["manifest.json", "run_journal.json"]


@dataclass
class CacheManifest:
    """Lightweight description of the cache layout and version."""

    cache_format_version: int
    created_at: float
    models: dict[str, Any]
    pipeline: dict[str, Any]


def _manifest_path(cache_root: Path) -> Path:
    return cache_root / "manifest.json"


def _build_manifest_payload(settings: Settings, created_at: float | None = None) -> dict[str, Any]:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "created_at": created_at or time.time(),
        "models": {
            "embedding": {
                "backend": settings.models.embedding.backend,
                "model_name": settings.models.embedding.resolved_model_name(),
            },
            "caption": {
                "backend": settings.models.caption.backend,
                "model_name": settings.models.caption.resolved_model_name(),
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
            "thumbnail_size_small": settings.pipeline.thumbnail_size_small,
            "thumbnail_size_large": settings.pipeline.thumbnail_size_large,
            "thumbnail_quality": settings.pipeline.thumbnail_quality,
            "phash_hamming_threshold": settings.pipeline.phash_hamming_threshold,
            "run_detection": settings.pipeline.run_detection,
            "skip_duplicates_for_heavy_models": settings.pipeline.skip_duplicates_for_heavy_models,
        },
    }


def _payloads_match(existing: CacheManifest, expected: dict[str, Any]) -> bool:
    if existing.cache_format_version != expected.get("cache_format_version"):
        return False

    def _normalize(block: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(block, sort_keys=True)))

    existing_models = _normalize(existing.models)
    expected_models = _normalize(expected.get("models", {}))
    existing_pipeline = _normalize(existing.pipeline)
    expected_pipeline = _normalize(expected.get("pipeline", {}))
    return existing_models == expected_models and existing_pipeline == expected_pipeline


def load_cache_manifest(cache_root: Path) -> CacheManifest | None:
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


def clear_cache_artifacts(cache_root: Path) -> None:
    """Remove generated cache artifacts when the manifest changes."""

    for subdir in _CACHE_SUBDIRS:
        path = cache_root / subdir
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    for filename in _CACHE_FILES:
        path = cache_root / filename
        if path.exists():
            try:
                path.unlink()
            except OSError:
                LOGGER.warning("cache_file_cleanup_failed", extra={"path": str(path)})


def ensure_cache_manifest(cache_root: Path, settings: Settings) -> CacheManifest:
    """Validate cache manifest; clear caches and rewrite on mismatch."""

    cache_root.mkdir(parents=True, exist_ok=True)

    existing = load_cache_manifest(cache_root)
    expected_payload = _build_manifest_payload(settings)

    if existing is None:
        LOGGER.info("cache_manifest_missing_create", extra={"cache_root": str(cache_root)})
        return write_cache_manifest(cache_root, settings)

    if not _payloads_match(existing, expected_payload):
        LOGGER.info(
            "cache_manifest_mismatch_reset",
            extra={"cache_root": str(cache_root), "existing_version": existing.cache_format_version},
        )
        clear_cache_artifacts(cache_root)
        return write_cache_manifest(cache_root, settings)

    return existing


def write_cache_manifest(cache_root: Path, settings: Settings) -> CacheManifest:
    """Create or update the cache manifest to reflect current settings."""

    cache_root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(cache_root)

    payload = _build_manifest_payload(settings)

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


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CacheManifest",
    "clear_cache_artifacts",
    "ensure_cache_manifest",
    "load_cache_manifest",
    "write_cache_manifest",
]
