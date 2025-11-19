"""CLI to enqueue heavy-model preprocessing tasks into the SQLite queue.

This helper scans the primary database for images that are missing SigLIP
embeddings and BLIP captions for the current configuration and populates the
``preprocess_task`` table with ``embedding`` and ``caption`` tasks.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from sqlalchemy import and_, delete, select

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import Image, ImageCaption, ImageEmbedding, ImageNearDuplicate, PreprocessTask, open_primary_session


LOGGER = get_logger(__name__, extra={"component": "enqueue_heavy"})


def main(
    root_db: Path = typer.Option(
        Path("data/index.db"),
        "--db",
        help="Path to the primary operational SQLite database.",
    ),
) -> None:
    """Enqueue embedding and caption tasks for images missing heavy-model outputs."""

    settings: Settings = load_settings()

    embedding_cfg = settings.models.embedding
    caption_cfg = settings.models.caption

    embedding_model_name = embedding_cfg.resolved_model_name()
    caption_model_name = caption_cfg.resolved_model_name()

    LOGGER.info(
        "enqueue_heavy_start",
        extra={
            "db": str(root_db),
            "embedding_model": embedding_model_name,
            "caption_model": caption_model_name,
        },
    )

    now = time.time()

    with open_primary_session(root_db) as session:
        # Clear any stale heavy-model tasks so we can rebuild the queue cleanly.
        session.execute(
            delete(PreprocessTask).where(
                PreprocessTask.task_type.in_(("embedding", "caption")),
            )
        )
        session.commit()

        active_rows = session.execute(
            select(Image.image_id).where(Image.status == "active").order_by(Image.image_id)
        )
        process_ids = [row.image_id for row in active_rows]

        if settings.pipeline.skip_duplicates_for_heavy_models:
            dup_rows = session.execute(select(ImageNearDuplicate.duplicate_image_id))
            duplicate_ids = {row.duplicate_image_id for row in dup_rows}
            process_ids = [image_id for image_id in process_ids if image_id not in duplicate_ids]

        existing_embedding_ids = {
            row.image_id
            for row in session.execute(
                select(ImageEmbedding.image_id).where(ImageEmbedding.model_name == embedding_model_name)
            )
        }
        existing_caption_ids = {
            row.image_id
            for row in session.execute(
                select(ImageCaption.image_id).where(ImageCaption.model_name == caption_model_name)
            )
        }

        embedding_targets = [image_id for image_id in process_ids if image_id not in existing_embedding_ids]
        caption_targets = [image_id for image_id in process_ids if image_id not in existing_caption_ids]

        enqueued = 0

        for image_id in embedding_targets:
            session.add(
                PreprocessTask(
                    image_id=image_id,
                    task_type="embedding",
                    priority=20,
                    status="pending",
                    retry_count=0,
                    max_retries=3,
                    error_message=None,
                    created_at=now,
                    started_at=None,
                    completed_at=None,
                )
            )
            enqueued += 1

        for image_id in caption_targets:
            session.add(
                PreprocessTask(
                    image_id=image_id,
                    task_type="caption",
                    priority=30,
                    status="pending",
                    retry_count=0,
                    max_retries=3,
                    error_message=None,
                    created_at=now,
                    started_at=None,
                    completed_at=None,
                )
            )
            enqueued += 1

        session.commit()

    LOGGER.info(
        "enqueue_heavy_complete",
        extra={"db": str(root_db), "tasks_enqueued": enqueued},
    )


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]

