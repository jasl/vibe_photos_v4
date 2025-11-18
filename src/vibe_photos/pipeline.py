"""High-level M1 preprocessing pipeline orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from torch import Tensor
from utils.logging import get_logger
from vibe_photos.classifier import build_scene_classifier, classify_image_embeddings
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import open_primary_db, open_projection_db
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
        """Scan album roots and populate the images table with content hashes."""

        self._logger.info("scan_and_hash_start", extra={})
        files: List[FileInfo] = list(scan_roots(roots))
        self._logger.info("scan_and_hash_stub", extra={"file_count": len(files)})

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
