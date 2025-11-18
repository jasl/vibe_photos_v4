"""High-level M1 preprocessing pipeline orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from torch import Tensor
from utils.logging import get_logger
from vibe_photos.classifier import build_scene_classifier, classify_image_embeddings
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import open_primary_db, open_projection_db
from vibe_photos.hasher import CONTENT_HASH_ALGO, compute_content_hash
from vibe_photos.scanner import FileInfo, scan_roots


LOGGER = get_logger(__name__)


@dataclass
class RunJournalRecord:
    """Lightweight checkpoint for resumable pipeline execution."""

    stage: str
    cursor_image_id: Optional[str]
    updated_at: float


def load_run_journal(cache_root: Path) -> Optional[RunJournalRecord]:
    """Load the run journal from ``cache/run_journal.json`` if it exists."""

    journal_path = cache_root / "run_journal.json"
    if not journal_path.exists():
        return None

    try:
        data = json.loads(journal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("run_journal_load_error", extra={"path": str(journal_path), "error": str(exc)})
        return None

    stage = str(data.get("stage") or "")
    cursor = data.get("cursor_image_id")
    updated_raw = data.get("updated_at")

    try:
        updated_at = float(updated_raw)
    except (TypeError, ValueError):
        updated_at = time.time()

    if not stage:
        return None

    return RunJournalRecord(stage=stage, cursor_image_id=str(cursor) if cursor is not None else None, updated_at=updated_at)


def save_run_journal(cache_root: Path, record: RunJournalRecord) -> None:
    """Persist the run journal to ``cache/run_journal.json``."""

    cache_root.mkdir(parents=True, exist_ok=True)
    journal_path = cache_root / "run_journal.json"
    payload = {
        "stage": record.stage,
        "cursor_image_id": record.cursor_image_id,
        "updated_at": record.updated_at,
    }
    journal_path.write_text(json.dumps(payload), encoding="utf-8")


class PreprocessingPipeline:
    """Orchestrates the M1 preprocessing steps for one or more album roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with application settings.

        Args:
            settings: Optional pre-loaded Settings instance. When omitted,
                configuration is loaded from ``config/settings.yaml``.
        """

        self._settings = settings or load_settings()
        self._logger = get_logger(__name__, extra={"component": "preprocess"})

    def run(self, roots: Sequence[Path], primary_db_path: Path, projection_db_path: Path) -> None:
        """Run the preprocessing pipeline for the given album roots.

        The initial implementation focuses on wiring and logging. Individual
        stages are implemented as stubs that can be expanded in dedicated
        iterations.

        Args:
            roots: Album root directories to scan.
            primary_db_path: Path to the primary operational database.
            projection_db_path: Path to the projection database for model outputs.
        """

        cache_root = projection_db_path.parent
        self._logger.info(
            "pipeline_start",
            extra={
                "roots": [str(root) for root in roots],
                "primary_db": str(primary_db_path),
                "projection_db": str(projection_db_path),
            },
        )

        journal = load_run_journal(cache_root)
        if journal is not None:
            self._logger.info(
                "pipeline_resume",
                extra={"stage": journal.stage, "cursor_image_id": journal.cursor_image_id},
            )

        with open_primary_db(primary_db_path) as primary_conn, open_projection_db(projection_db_path) as projection_conn:
            self._run_scan_and_hash(roots, primary_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="scan_and_hash", cursor_image_id=None, updated_at=time.time()))

            self._run_scene_classification(primary_conn, projection_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="scene_classification", cursor_image_id=None, updated_at=time.time()))

            self._run_embeddings_and_captions(primary_conn, projection_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=None, updated_at=time.time()))

            self._run_perceptual_hashing_and_duplicates(primary_conn, projection_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="phash_and_duplicates", cursor_image_id=None, updated_at=time.time()))

        self._logger.info("pipeline_complete", extra={})

    def _run_scan_and_hash(self, roots: Sequence[Path], primary_conn) -> None:
        """Scan album roots and populate the images table with content hashes.

        This stage is responsible for maintaining the ``images`` table in the
        primary database:

        - Insert rows for new content hashes discovered under the configured roots.
        - Merge paths for existing hashes, keeping one canonical ``primary_path``.
        - Handle file content changes by moving a path from the old ``image_id``
          to the new one.
        - Update deletion status when all recorded paths for an ``image_id``
          disappear from disk.
        """

        self._logger.info("scan_and_hash_start", extra={})
        files: List[FileInfo] = list(scan_roots(roots))
        self._logger.info("scan_and_hash_files_discovered", extra={"file_count": len(files)})

        cursor = primary_conn.cursor()

        # Build a mapping from path -> image_id based on existing rows.
        path_to_image_id: Dict[str, str] = {}
        cursor.execute("SELECT image_id, all_paths FROM images")
        for row in cursor.fetchall():
            raw_paths = row["all_paths"]
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []
            for stored_path in paths:
                path_to_image_id[str(stored_path)] = row["image_id"]

        now = time.time()

        # Scan files and upsert image records.
        for file_info in files:
            path = file_info.path.resolve()
            path_str = str(path)
            new_image_id = compute_content_hash(path)
            old_image_id = path_to_image_id.get(path_str)

            # If the file existed before but its content hash changed, detach it
            # from the previous image row.
            if old_image_id is not None and old_image_id != new_image_id:
                cursor.execute("SELECT all_paths, primary_path, status FROM images WHERE image_id = ?", (old_image_id,))
                old_row = cursor.fetchone()
                if old_row is not None:
                    try:
                        old_paths = json.loads(old_row["all_paths"]) if old_row["all_paths"] else []
                    except json.JSONDecodeError:
                        old_paths = []

                    new_paths = [p for p in old_paths if p != path_str]
                    status = old_row["status"]
                    primary_path = old_row["primary_path"]

                    if new_paths:
                        if primary_path not in new_paths:
                            primary_path = new_paths[0]
                        # If the row was previously marked deleted but still has
                        # remaining paths, restore it to active.
                        if status == "deleted":
                            status = "active"
                    else:
                        # All paths removed for this image_id; mark as deleted.
                        status = "deleted"

                    cursor.execute(
                        "UPDATE images SET all_paths = ?, primary_path = ?, status = ?, updated_at = ? WHERE image_id = ?",
                        (json.dumps(new_paths), primary_path, status, now, old_image_id),
                    )

                path_to_image_id.pop(path_str, None)
                old_image_id = None

            # Upsert the row for the new image_id.
            cursor.execute(
                "SELECT image_id, primary_path, all_paths, size_bytes, mtime, status FROM images WHERE image_id = ?",
                (new_image_id,),
            )
            existing = cursor.fetchone()

            if existing is not None:
                try:
                    paths = json.loads(existing["all_paths"]) if existing["all_paths"] else []
                except json.JSONDecodeError:
                    paths = []

                if path_str not in paths:
                    paths.append(path_str)

                primary_path = existing["primary_path"] or path_str
                status = existing["status"] or "active"
                if status == "deleted":
                    status = "active"

                cursor.execute(
                    """
                    UPDATE images
                    SET primary_path = ?, all_paths = ?, size_bytes = ?, mtime = ?, hash_algo = ?, status = ?, updated_at = ?
                    WHERE image_id = ?
                    """,
                    (
                        primary_path,
                        json.dumps(paths),
                        file_info.size_bytes,
                        file_info.mtime,
                        CONTENT_HASH_ALGO,
                        status,
                        now,
                        new_image_id,
                    ),
                )
            else:
                all_paths_json = json.dumps([path_str])
                cursor.execute(
                    """
                    INSERT INTO images (
                        image_id,
                        primary_path,
                        all_paths,
                        size_bytes,
                        mtime,
                        width,
                        height,
                        exif_datetime,
                        camera_model,
                        hash_algo,
                        phash,
                        phash_algo,
                        phash_updated_at,
                        created_at,
                        updated_at,
                        status,
                        error_message,
                        schema_version
                    ) VALUES (
                        ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, ?, ?, ?, NULL, ?
                    )
                    """,
                    (
                        new_image_id,
                        path_str,
                        all_paths_json,
                        file_info.size_bytes,
                        file_info.mtime,
                        CONTENT_HASH_ALGO,
                        now,
                        now,
                        "active",
                        1,
                    ),
                )

            path_to_image_id[path_str] = new_image_id

        primary_conn.commit()

        # Second pass: remove paths that no longer exist on disk and mark rows
        # as deleted when all paths disappear.
        now = time.time()
        cursor.execute("SELECT image_id, all_paths, primary_path, status FROM images")
        for row in cursor.fetchall():
            raw_paths = row["all_paths"]
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []

            if not paths:
                continue

            existing_paths: List[str] = []
            for stored_path in paths:
                stored_path_str = str(stored_path)
                try:
                    if Path(stored_path_str).exists():
                        existing_paths.append(stored_path_str)
                except OSError:
                    # Treat paths that raise OS errors as missing.
                    continue

            if existing_paths == paths:
                continue

            primary_path = row["primary_path"]
            status = row["status"]

            if existing_paths:
                if primary_path not in existing_paths:
                    primary_path = existing_paths[0]
                if status == "deleted":
                    status = "active"
            else:
                status = "deleted"

            cursor.execute(
                "UPDATE images SET all_paths = ?, primary_path = ?, status = ?, updated_at = ? WHERE image_id = ?",
                (json.dumps(existing_paths), primary_path, status, now, row["image_id"]),
            )

        primary_conn.commit()
        self._logger.info("scan_and_hash_complete", extra={"file_count": len(files)})

    def _run_scene_classification(self, primary_conn, projection_conn) -> None:
        """Placeholder for lightweight scene classification."""

        self._logger.info("scene_classification_start", extra={})
        classifier, classifier_name, classifier_version = build_scene_classifier(self._settings)
        embeddings: List[Tensor] = []
        _ = classify_image_embeddings(
            classifier=classifier,
            embeddings=embeddings,
            classifier_name=classifier_name,
            classifier_version=classifier_version,
        )
        self._logger.info("scene_classification_stub", extra={})

    def _run_embeddings_and_captions(self, primary_conn, projection_conn) -> None:
        """Placeholder for SigLIP embeddings and BLIP captions."""

        self._logger.info("embeddings_and_captions_start", extra={})
        self._logger.info("embeddings_and_captions_stub", extra={})

    def _run_perceptual_hashing_and_duplicates(self, primary_conn, projection_conn) -> None:
        """Placeholder for perceptual hashing and near-duplicate grouping."""

        self._logger.info("phash_and_duplicates_start", extra={})
        self._logger.info("phash_and_duplicates_stub", extra={})


__all__ = ["PreprocessingPipeline", "RunJournalRecord", "load_run_journal", "save_run_journal"]
