"""CLI to invalidate cache artifacts by stage."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import typer
from sqlalchemy import delete
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.cache_helpers import resolve_cache_root
from vibe_photos.cache_manifest import clear_cache_artifacts
from vibe_photos.config import load_settings
from vibe_photos.db import (
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageScene,
    Region,
    RegionEmbedding,
    open_primary_session,
)

LOGGER = get_logger(__name__)

Stage = Literal["embeddings", "captions", "scenes", "regions", "duplicates", "all"]


def _remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _invalidate_db_tables(session: Session, stages: set[Stage]) -> None:
    if "embeddings" in stages or "all" in stages:
        session.execute(delete(ImageEmbedding))
    if "captions" in stages or "all" in stages:
        session.execute(delete(ImageCaption))
    if "scenes" in stages or "all" in stages:
        session.execute(delete(ImageScene))
    if "regions" in stages or "all" in stages:
        session.execute(delete(Region))
        session.execute(delete(RegionEmbedding))
    if "duplicates" in stages or "all" in stages:
        session.execute(delete(ImageNearDuplicate))
    session.commit()


def _invalidate_cache_dirs(cache_root: Path, stages: set[Stage]) -> None:
    if "embeddings" in stages or "all" in stages:
        _remove_path(cache_root / "embeddings")
    if "captions" in stages or "all" in stages:
        _remove_path(cache_root / "captions")
    if "regions" in stages or "all" in stages:
        _remove_path(cache_root / "regions")
    if "duplicates" in stages or "all" in stages:
        # No dedicated dir; rely on DB cleanup.
        return


def main(
    cache_root: Path | None = typer.Option(
        None,
        "--cache-root",
        help="Cache root directory; defaults to cache.root from settings.yaml.",
    ),
    stages: Iterable[Stage] = typer.Option(
        ["all"],
        "--stage",
        "-s",
        help="Stages to invalidate. Defaults to all.",
    ),
        full_reset: bool = typer.Option(
        False,
        "--full-reset",
        help="Remove all cache artifacts regardless of stage selection.",
    ),
) -> None:
    """Invalidate cached artifacts for selected stages."""

    settings = load_settings()
    if cache_root is None:
        cache_root = resolve_cache_root(settings.cache.root)
    cache_root = cache_root.resolve()
    raw_stages = set(stages or ["all"])
    selected: set[Stage] = {cast(Stage, stage) for stage in raw_stages}

    if full_reset:
        clear_cache_artifacts(cache_root)
        LOGGER.info("cache_full_reset", extra={"cache_root": str(cache_root)})
        return

    db_target = settings.databases.primary_url
    with open_primary_session(db_target) as session:
        _invalidate_db_tables(session, selected)

    _invalidate_cache_dirs(cache_root, selected)
    LOGGER.info("cache_stages_invalidated", extra={"stages": sorted(selected), "cache_root": str(cache_root)})


if __name__ == "__main__":
    typer.run(main)
