"""CLI to copy cache tables into the primary DB for API/UI use."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.config import load_settings
from vibe_photos.db import (
    ImageCaption,
    ImageEmbedding,
    ImageNearDuplicate,
    ImageScene,
    dialect_insert,
    open_cache_session,
    open_primary_session,
    sqlite_path_from_target,
)

LOGGER = get_logger(__name__)

TableName = Literal["embeddings", "captions", "scenes", "duplicates", "all"]


def _sync_embeddings(src_session: Session, dst_session: Session) -> int:
    rows = src_session.execute(select(ImageEmbedding)).scalars()
    count = 0
    for row in rows:
        stmt = dialect_insert(dst_session, ImageEmbedding).values(
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


def _sync_captions(src_session: Session, dst_session: Session) -> int:
    rows = src_session.execute(select(ImageCaption)).scalars()
    count = 0
    for row in rows:
        stmt = dialect_insert(dst_session, ImageCaption).values(
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


def _sync_scenes(src_session: Session, dst_session: Session) -> int:
    rows = src_session.execute(select(ImageScene)).scalars()
    count = 0
    for row in rows:
        stmt = dialect_insert(dst_session, ImageScene).values(
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


def _sync_duplicates(src_session: Session, dst_session: Session) -> int:
    rows = src_session.execute(select(ImageNearDuplicate)).scalars()
    count = 0
    for row in rows:
        stmt = dialect_insert(dst_session, ImageNearDuplicate).values(
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
    cache_db: str | None = typer.Option(
        None,
        "--cache-db",
        help="Cache database URL or path. Defaults to databases.cache_url in settings.yaml.",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="Primary database URL or path. Defaults to databases.primary_url in settings.yaml.",
    ),
    tables: Iterable[TableName] = typer.Option(
        ["all"], "--table", "-t", help="Tables to sync: embeddings, captions, scenes, duplicates, or all."
    ),
) -> None:
    """Copy cache tables into the primary DB."""

    settings = load_settings()
    cache_target = cache_db or settings.databases.cache_url
    cache_path = sqlite_path_from_target(cache_target)
    primary_target = db or settings.databases.primary_url

    selected = set(tables or ["all"])
    with open_cache_session(cache_path) as src, open_primary_session(primary_target) as dst:
        total = 0
        if "embeddings" in selected or "all" in selected:
            total += _sync_embeddings(src, dst)
        if "captions" in selected or "all" in selected:
            total += _sync_captions(src, dst)
        if "scenes" in selected or "all" in selected:
            total += _sync_scenes(src, dst)
        if "duplicates" in selected or "all" in selected:
            total += _sync_duplicates(src, dst)

    LOGGER.info(
        "cache_sync_complete",
        extra={"tables": sorted(selected), "rows_synced": total, "cache_db": str(cache_path), "db": str(primary_target)},
    )


if __name__ == "__main__":
    typer.run(main)
