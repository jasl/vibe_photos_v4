"""Scan directories and enqueue Celery tasks for discovered images."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import typer
from sqlalchemy import select

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import Image, open_primary_session
from vibe_photos.hasher import CONTENT_HASH_ALGO, compute_content_hash
from vibe_photos.scanner import FileInfo, scan_roots
from vibe_photos.task_queue import post_process, pre_process, process

LOGGER = get_logger(__name__, extra={"component": "enqueue_celery"})


def _existing_path_map(session) -> dict[str, str]:
    mapping: dict[str, str] = {}
    rows = session.execute(select(Image.image_id, Image.all_paths)).mappings()
    for row in rows:
        raw_paths = row["all_paths"]
        try:
            paths = json.loads(raw_paths) if raw_paths else []
        except json.JSONDecodeError:
            paths = []
        for stored_path in paths:
            mapping[str(stored_path)] = row["image_id"]
    return mapping


def _ingest_files(session, files: Sequence[FileInfo]) -> list[str]:
    """Upsert :class:`Image` rows for scanned files and return their IDs."""

    total_files = len(files)
    LOGGER.info("celery_ingest_files_discovered", extra={"file_count": total_files})

    progress_interval = max(1, total_files // 20) if total_files else 0

    path_to_image_id = _existing_path_map(session)
    processed: list[str] = []
    now = time.time()

    for index, file_info in enumerate(files, start=1):
        path = file_info.path.resolve()
        path_str = str(path)
        image_id = compute_content_hash(path)
        old_image_id = path_to_image_id.get(path_str)

        if old_image_id is not None and old_image_id != image_id:
            old_image = session.get(Image, old_image_id)
            if old_image is not None:
                try:
                    old_paths = json.loads(old_image.all_paths) if old_image.all_paths else []
                except json.JSONDecodeError:
                    old_paths = []
                old_paths = [p for p in old_paths if p != path_str]
                old_image.all_paths = json.dumps(old_paths)
                if old_image.primary_path == path_str:
                    old_image.primary_path = old_paths[0] if old_paths else ""
                if not old_paths:
                    old_image.status = "deleted"
                old_image.updated_at = now
                session.add(old_image)

            path_to_image_id.pop(path_str, None)

        existing = session.get(Image, image_id)

        if existing is not None:
            try:
                paths = json.loads(existing.all_paths) if existing.all_paths else []
            except json.JSONDecodeError:
                paths = []

            if path_str not in paths:
                paths.append(path_str)

            existing.primary_path = existing.primary_path or path_str
            existing.all_paths = json.dumps(paths)
            existing.size_bytes = file_info.size_bytes
            existing.mtime = file_info.mtime
            existing.hash_algo = CONTENT_HASH_ALGO
            existing.status = "active" if existing.status == "deleted" else existing.status
            existing.updated_at = now
            session.add(existing)
        else:
            session.add(
                Image(
                    image_id=image_id,
                    primary_path=path_str,
                    all_paths=json.dumps([path_str]),
                    size_bytes=file_info.size_bytes,
                    mtime=file_info.mtime,
                    width=None,
                    height=None,
                    exif_datetime=None,
                    camera_model=None,
                    hash_algo=CONTENT_HASH_ALGO,
                    phash=None,
                    phash_algo=None,
                    phash_updated_at=None,
                    created_at=now,
                    updated_at=now,
                    status="active",
                    error_message=None,
                    schema_version=1,
                )
            )

        path_to_image_id[path_str] = image_id
        processed.append(image_id)

        if progress_interval and (index % progress_interval == 0 or index == total_files):
            percent = round(index * 100.0 / max(total_files, 1), 1)
            LOGGER.info(
                "celery_ingest_progress %s/%s (%.1f%%)",
                index,
                total_files,
                percent,
                extra={"processed": index, "total": total_files, "percent": percent},
            )

    session.commit()
    return processed


def _enqueue_targets(image_ids: Sequence[str], task: str) -> None:
    total_images = len(image_ids)
    progress_interval = max(1, total_images // 20) if total_images else 0

    queued = 0

    for index, image_id in enumerate(image_ids, start=1):
        if task == "pre_process":
            pre_process.delay(image_id)
        elif task == "process":
            process.delay(image_id)
        elif task == "post_process":
            post_process.delay(image_id)
        else:  # pragma: no cover - defensive
            LOGGER.error("celery_enqueue_unknown_task", extra={"task": task})
            break
        queued += 1

        if progress_interval and (index % progress_interval == 0 or index == total_images):
            percent = round(index * 100.0 / max(total_images, 1), 1)
            LOGGER.info(
                "celery_enqueue_progress %s/%s (%.1f%%)",
                index,
                total_images,
                percent,
                extra={
                    "processed": index,
                    "total": total_images,
                    "percent": percent,
                    "task": task,
                    "queued": queued,
                },
            )

    LOGGER.info(
        "celery_enqueue_complete",
        extra={"task": task, "queued": queued},
    )


def main(
    roots: list[Path] = typer.Argument(..., help="One or more directories to scan for images."),
    db: Path = typer.Option(Path("data/index.db"), help="Path to the primary SQLite database."),
    task: str = typer.Option(
        "process",
        "--task",
        "-t",
        help="Task to enqueue: pre_process, process, or post_process. Defaults to process.",
    ),
) -> None:
    """Scan directories, persist images, and enqueue Celery tasks."""

    task = task.lower().strip()
    if task not in {"pre_process", "process", "post_process"}:
        raise typer.BadParameter("task must be one of: pre_process, process, post_process")

    settings: Settings = load_settings()
    LOGGER.info(
        "celery_enqueue_start",
        extra={
            "roots": [str(root) for root in roots],
            "db": str(db),
            "task": task,
            "queues": {
                "pre_process": settings.queues.preprocess_queue,
                "process": settings.queues.main_queue,
                "post_process": settings.queues.post_process_queue,
            },
        },
    )

    files: list[FileInfo] = list(scan_roots(roots))
    if not files:
        LOGGER.warning("celery_enqueue_no_files", extra={"roots": [str(root) for root in roots]})
        return

    with open_primary_session(db) as session:
        image_ids = _ingest_files(session, files)

    _enqueue_targets(image_ids, task)


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]
