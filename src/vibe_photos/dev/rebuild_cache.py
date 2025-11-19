"""CLI entrypoint to rebuild projection tables from cache artifacts."""

from __future__ import annotations

from pathlib import Path

import typer

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import open_projection_session, reset_projection_tables
from vibe_photos.projection_rebuild import (
    rebuild_captions_from_cache,
    rebuild_embeddings_from_cache,
    rebuild_regions_from_cache,
    rebuild_scenes_from_cache,
)


LOGGER = get_logger(__name__)


def main(
    cache_root: Path = typer.Option(
        Path("cache"),
        "--cache-root",
        help="Root directory for cache artifacts (embeddings, captions, detections, regions).",
    ),
    cache_db: Path = typer.Option(
        Path("cache/index.db"),
        "--cache-db",
        help="SQLite database path whose projection tables should be rebuilt from cache.",
    ),
    reset_db: bool = typer.Option(
        True,
        "--reset-db/--no-reset-db",
        help="Whether to clear existing projection tables before rebuilding.",
    ),
) -> None:
    """Rebuild projection tables (embeddings, captions, scenes, regions) from cache."""

    settings: Settings = load_settings()

    LOGGER.info(
        "rebuild_cache_start",
        extra={"cache_root": str(cache_root), "cache_db": str(cache_db), "reset_db": reset_db},
    )

    with open_projection_session(cache_db) as session:
        if reset_db:
            reset_projection_tables(session)

        rebuild_embeddings_from_cache(cache_root, session)
        rebuild_captions_from_cache(cache_root, session)
        rebuild_scenes_from_cache(cache_root, session)
        rebuild_regions_from_cache(cache_root, session)

    LOGGER.info("rebuild_cache_complete", extra={"cache_root": str(cache_root), "cache_db": str(cache_db)})


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]

