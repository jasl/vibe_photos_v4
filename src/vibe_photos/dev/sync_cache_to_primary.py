"""CLI to copy cache projection tables into the primary DB for API/UI use."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import typer
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from utils.logging import get_logger
from vibe_photos.db import (
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageScene,
    open_primary_session,
    open_projection_session,
)

LOGGER = get_logger(__name__)

TableName = Literal["embeddings", "captions", "scenes", "duplicates", "all"]


def _sync_embeddings(src_session, dst_session) -> int:
    rows = src_session.execute(select(ImageEmbedding)).scalars()
    count = 0
    for row in rows:
        stmt = sqlite_insert(ImageEmbedding).values(
            image_id=row.image_id,
            model_name=row.model_name,
            embedding_path=row.embedding_path,
            embedding_dim=row.embedding_dim,
            model_backend=row.model_backend,
            updated_at=row.updated_at,
        ).on_conflict_do_update(
            index_elements=[ImageEmbedding.image_id, ImageEmbedding.model_name],
            set_={
                "embedding_path": row.embedding_path,
                "embedding_dim": row.embedding_dim,
                "model_backend": row.model_backend,
                "updated_at": row.updated_at,
            },
        )
        dst_session.execute(stmt)
        count += 1
    dst_session.commit()
    return count


def _sync_captions(src_session, dst_session) -> int:
    rows = src_session.execute(select(ImageCaption)).scalars()
    count = 0
    for row in rows:
        stmt = sqlite_insert(ImageCaption).values(
            image_id=row.image_id,
            model_name=row.model_name,
            caption=row.caption,
            model_backend=row.model_backend,
            updated_at=row.updated_at,
        ).on_conflict_do_update(
            index_elements=[ImageCaption.image_id, ImageCaption.model_name],
            set_={
                "caption": row.caption,
                "model_backend": row.model_backend,
                "updated_at": row.updated_at,
            },
        )
        dst_session.execute(stmt)
        count += 1
    dst_session.commit()
    return count


def _sync_scenes(src_session, dst_session) -> int:
    rows = src_session.execute(select(ImageScene)).scalars()
    count = 0
    for row in rows:
        stmt = sqlite_insert(ImageScene).values(
            image_id=row.image_id,
            scene_type=row.scene_type,
            scene_confidence=row.scene_confidence,
            has_text=row.has_text,
            has_person=row.has_person,
            is_screenshot=row.is_screenshot,
            is_document=row.is_document,
            classifier_name=row.classifier_name,
            classifier_version=row.classifier_version,
            updated_at=row.updated_at,
        ).on_conflict_do_update(
            index_elements=[ImageScene.image_id],
            set_={
                "scene_type": row.scene_type,
                "scene_confidence": row.scene_confidence,
                "has_text": row.has_text,
                "has_person": row.has_person,
                "is_screenshot": row.is_screenshot,
                "is_document": row.is_document,
                "classifier_name": row.classifier_name,
                "classifier_version": row.classifier_version,
                "updated_at": row.updated_at,
            },
        )
        dst_session.execute(stmt)
        count += 1
    dst_session.commit()
    return count


def _sync_duplicates(src_session, dst_session) -> int:
    rows = src_session.execute(select(ImageNearDuplicate)).scalars()
    count = 0
    for row in rows:
        stmt = sqlite_insert(ImageNearDuplicate).values(
            anchor_image_id=row.anchor_image_id,
            duplicate_image_id=row.duplicate_image_id,
            phash_distance=row.phash_distance,
            created_at=row.created_at,
        ).on_conflict_do_update(
            index_elements=[ImageNearDuplicate.anchor_image_id, ImageNearDuplicate.duplicate_image_id],
            set_={
                "phash_distance": row.phash_distance,
                "created_at": row.created_at,
            },
        )
        dst_session.execute(stmt)
        count += 1
    dst_session.commit()
    return count


def main(
    cache_db: Path = typer.Option(Path("cache/index.db"), help="Cache DB path (projection)."),
    db: Path = typer.Option(Path("data/index.db"), help="Primary DB path."),
    tables: Iterable[TableName] = typer.Option(
        ["all"], "--table", "-t", help="Tables to sync: embeddings, captions, scenes, duplicates, or all."
    ),
) -> None:
    """Copy cache projection tables into the primary DB."""

    selected = set(tables or ["all"])
    with open_projection_session(cache_db) as src, open_primary_session(db) as dst:
        total = 0
        if "embeddings" in selected or "all" in selected:
            total += _sync_embeddings(src, dst)
        if "captions" in selected or "all" in selected:
            total += _sync_captions(src, dst)
        if "scenes" in selected or "all" in selected:
            total += _sync_scenes(src, dst)
        if "duplicates" in selected or "all" in selected:
            total += _sync_duplicates(src, dst)

    LOGGER.info("cache_sync_complete", extra={"tables": sorted(selected), "rows_synced": total, "cache_db": str(cache_db), "db": str(db)})


if __name__ == "__main__":
    typer.run(main)
