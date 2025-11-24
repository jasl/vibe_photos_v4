"""Filesystem cache helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_cache_root(target: str | Path) -> Path:
    """Resolve a filesystem cache root, enforcing directory-only inputs."""

    raw = str(target).strip()
    if not raw:
        raise ValueError("cache target cannot be empty")

    if "://" in raw:
        raise ValueError("cache roots must be filesystem paths; URLs are not supported")

    path = Path(raw).resolve()
    if path.suffix == ".db":
        raise ValueError("cache root must be a directory, not a database file path")

    return path


__all__ = ["resolve_cache_root"]
