"""CLI to enqueue region-detection tasks into the SQLite queue.

This helper scans the primary database for active images that are missing
`image_region` rows for the current detection configuration and have BLIP
captions available, then inserts `detection` tasks into the `preprocess_task`
table. Detection tasks can be processed by `vibe_photos.dev.worker`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from sqlalchemy import delete, select

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import (
    Image,
    ImageCaption,
    ImageNearDuplicate,
    ImageRegion,
    PreprocessTask,
    open_primary_session,
)


LOGGER = get_logger(__name__, extra={"component": "enqueue_detection"})


def main(
    db: Path = typer.Option(
        Path("data/index.db"),
        "--db",
        help="Path to the primary operational SQLite database.",
    ),
) -> None:
    """Enqueue region-detection tasks for images missing `image_region` rows."""

    settings: Settings = load_settings()

    if not settings.pipeline.run_detection or not settings.models.detection.enabled:
        LOGGER.info(
            "enqueue_detection_disabled",
            extra={
                "run_detection": bool(settings.pipeline.run_detection),
                "detection_enabled": bool(settings.models.detection.enabled),
            },
        )
        return

    detection_cfg = settings.models.detection
    caption_cfg = settings.models.caption
    caption_model_name = caption_cfg.resolved_model_name()

    LOGGER.info(
        "enqueue_detection_start",
        extra={
            "db": str(db),
            "detection_backend": detection_cfg.backend,
            "detection_model": detection_cfg.model_name,
            "caption_model": caption_model_name,
        },
    )

    with open_primary_session(db) as session:
        # Clear any stale detection tasks so we can rebuild the queue cleanly.
        session.execute(
            delete(PreprocessTask).where(
                PreprocessTask.task_type == "detection",
            )
        )
        session.commit()

        active_rows = session.execute(
            select(Image.image_id).where(Image.status == "active").order_by(Image.image_id)
        )
        candidate_ids: List[str] = [row.image_id for row in active_rows]

        if settings.pipeline.skip_duplicates_for_heavy_models:
            dup_rows = session.execute(select(ImageNearDuplicate.duplicate_image_id))
            duplicate_ids = {row.duplicate_image_id for row in dup_rows}
            candidate_ids = [image_id for image_id in candidate_ids if image_id not in duplicate_ids]

        existing_region_ids = {
            row.image_id
            for row in session.execute(select(ImageRegion.image_id).distinct())
        }
        caption_ids = {
            row.image_id
            for row in session.execute(
                select(ImageCaption.image_id).where(ImageCaption.model_name == caption_model_name)
            )
        }

        detection_targets = [
            image_id
            for image_id in candidate_ids
            if image_id not in existing_region_ids and image_id in caption_ids
        ]

        if not detection_targets:
            LOGGER.info("enqueue_detection_noop", extra={"db": str(db)})
            return

        now = __import__("time").time()
        enqueued = 0

        for image_id in detection_targets:
            session.add(
                PreprocessTask(
                    image_id=image_id,
                    task_type="detection",
                    priority=40,
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
        "enqueue_detection_complete",
        extra={"db": str(db), "tasks_enqueued": enqueued},
    )


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]

