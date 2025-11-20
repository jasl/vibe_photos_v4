"""CLI to invalidate cache artifacts by stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Literal

import typer
from sqlalchemy import delete

from utils.logging import get_logger
from vibe_photos.cache_manifest import clear_cache_artifacts
from vibe_photos.db import (
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageRegion,
    ImageScene,
    open_projection_session,
)


LOGGER = get_logger(__name__)

Stage = Literal["embeddings", "captions", "scenes", "regions", "duplicates", "all"]


def _remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _invalidate_db_tables(session, stages: set[Stage]) -> None:
    if "embeddings" in stages or "all" in stages:
        session.execute(delete(ImageEmbedding))
    if "captions" in stages or "all" in stages:
        session.execute(delete(ImageCaption))
    if "scenes" in stages or "all" in stages:
        session.execute(delete(ImageScene))
    if "regions" in stages or "all" in stages:
        session.execute(delete(ImageRegion))
    if "duplicates" in stages or "all" in stages:
        session.execute(delete(ImageNearDuplicate))
    session.commit()


def _invalidate_cache_dirs(cache_root: Path, stages: set[Stage]) -> None:
    if "embeddings" in stages or "all" in stages:
        _remove_path(cache_root / "embeddings")
    if "captions" in stages or "all" in stages:
        _remove_path(cache_root / "captions")
    if "scenes" in stages or "all" in stages:
        _remove_path(cache_root / "detections")
    if "regions" in stages or "all" in stages:
        _remove_path(cache_root / "regions")
    if "duplicates" in stages or "all" in stages:
        # No dedicated dir; rely on DB cleanup.
        return


def main(
    cache_root: Path = typer.Option(Path("cache"), help="Cache root containing cache/index.db."),
    stages: Iterable[Stage] = typer.Option(
        ["all"],
        "--stage",
        "-s",
        help="Stages to invalidate. Defaults to all.",
    ),
    full_reset: bool = typer.Option(
        False,
        "--full-reset",
        help="Remove all cache artifacts and cache/index.db regardless of stage selection.",
    ),
) -> None:
    """Invalidate cached artifacts for selected stages."""

    cache_root = cache_root.resolve()
    selected = set(stages or ["all"])

    if full_reset:
        clear_cache_artifacts(cache_root)
        LOGGER.info("cache_full_reset", extra={"cache_root": str(cache_root)})
        return

    db_path = cache_root / "index.db"
    with open_projection_session(db_path) as session:
        _invalidate_db_tables(session, selected)

    _invalidate_cache_dirs(cache_root, selected)
    LOGGER.info("cache_stages_invalidated", extra={"stages": sorted(selected), "cache_root": str(cache_root)})


if __name__ == "__main__":
    typer.run(main)

