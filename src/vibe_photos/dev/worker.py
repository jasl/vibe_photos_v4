"""CLI workers to process queued preprocessing tasks concurrently.

This tool pulls tasks from the ``preprocess_task`` table and runs per-image
embedding and caption computation using the shared M1 models.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import typer
from sqlalchemy import and_, select, update

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import PreprocessTask, open_primary_session
from vibe_photos.pipeline import PreprocessingPipeline


LOGGER = get_logger(__name__, extra={"component": "worker"})


def _reset_inflight_tasks(db_path: Path) -> None:
    """Reset tasks stuck in processing state back to pending."""

    with open_primary_session(db_path) as session:
        session.execute(
            update(PreprocessTask)
            .where(PreprocessTask.status == "processing")
            .values(status="pending", started_at=None)
        )
        session.commit()


def _claim_tasks(
    db_path: Path,
    task_type_filter: Optional[str],
    batch_size: int,
    lock: threading.Lock,
) -> List[Tuple[int, str, str]]:
    """Select a small batch of pending tasks and mark them as processing.

    Returns a list of ``(task_id, image_id, task_type)`` tuples.
    """

    with lock:
        with open_primary_session(db_path) as session:

            query = select(PreprocessTask).where(PreprocessTask.status == "pending")
            if task_type_filter:
                query = query.where(PreprocessTask.task_type == task_type_filter)
            query = query.order_by(PreprocessTask.priority, PreprocessTask.id).limit(batch_size)

            rows: Sequence[PreprocessTask] = session.execute(query).scalars().all()
            if not rows:
                return []

            now = time.time()
            claimed: List[Tuple[int, str, str]] = []
            for task in rows:
                task.status = "processing"
                task.started_at = now
                session.add(task)
                claimed.append((task.id, task.image_id, task.task_type))

            session.commit()

    return claimed


def _mark_task_completed(db_path: Path, task_id: int) -> None:
    with open_primary_session(db_path) as session:
        session.execute(
            update(PreprocessTask)
            .where(PreprocessTask.id == task_id)
            .values(status="completed", completed_at=time.time(), error_message=None)
        )
        session.commit()


def _mark_task_failed(db_path: Path, task_id: int, error_message: str) -> None:
    with open_primary_session(db_path) as session:
        row = session.get(PreprocessTask, task_id)
        if row is None:
            return

        retry_count = int(row.retry_count or 0) + 1
        status = "failed" if retry_count >= int(row.max_retries or 3) else "pending"

        row.retry_count = retry_count
        row.status = status
        row.error_message = error_message
        row.completed_at = time.time()
        session.add(row)
        session.commit()


def main(
    db: Path = typer.Option(
        Path("data/index.db"),
        "--db",
        help="Path to the primary operational SQLite database.",
    ),
    cache_root: Path = typer.Option(
        Path("cache"),
        "--cache-root",
        help="Root directory where cache artifacts (embeddings, captions) are stored.",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        min=1,
        help="Number of worker threads to process tasks concurrently.",
    ),
    task_type: Optional[str] = typer.Option(
        None,
        "--task-type",
        help="Optional task type filter (for example 'embedding' or 'caption').",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        min=1,
        help="Maximum number of tasks to claim per worker iteration.",
    ),
) -> None:
    """Run concurrent workers that process heavy-model preprocessing tasks."""

    settings: Settings = load_settings()

    LOGGER.info(
        "worker_start",
        extra={
            "db": str(db),
            "cache_root": str(cache_root),
            "workers": workers,
            "task_type": task_type or "any",
        },
    )

    _reset_inflight_tasks(db)

    lock = threading.Lock()

    def _worker_loop(worker_id: int) -> None:
        pipeline = PreprocessingPipeline(settings=settings)
        pipeline._cache_root = cache_root  # type: ignore[attr-defined]

        while True:
            claimed = _claim_tasks(db, task_type, batch_size, lock)
            if not claimed:
                LOGGER.info("worker_idle", extra={"worker_id": worker_id})
                return

            for task_id, image_id, ttype in claimed:
                try:
                    with open_primary_session(db) as session:
                        if ttype == "embedding":
                            pipeline.process_embedding_task(session, image_id)
                        elif ttype == "caption":
                            pipeline.process_caption_task(session, image_id)
                        else:
                            LOGGER.info(
                                "worker_skip_task_type",
                                extra={"worker_id": worker_id, "task_id": task_id, "task_type": ttype},
                            )
                    _mark_task_completed(db, task_id)
                except Exception as exc:  # pragma: no cover - defensive
                    msg = str(exc)
                    LOGGER.error(
                        "worker_task_error",
                        extra={"worker_id": worker_id, "task_id": task_id, "task_type": ttype, "error": msg},
                    )
                    _mark_task_failed(db, task_id, msg)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for idx in range(max(1, workers)):
            executor.submit(_worker_loop, idx)

    # Emit a final summary of remaining tasks.
    with open_primary_session(db) as session:
        pending = session.execute(
            select(PreprocessTask).where(PreprocessTask.status == "pending")
        ).scalars().all()
        processing = session.execute(
            select(PreprocessTask).where(PreprocessTask.status == "processing")
        ).scalars().all()
        failed = session.execute(
            select(PreprocessTask).where(and_(PreprocessTask.status == "failed"))
        ).scalars().all()

    LOGGER.info(
        "worker_complete",
        extra={
            "pending_tasks": len(pending),
            "processing_tasks": len(processing),
            "failed_tasks": len(failed),
        },
    )


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]

