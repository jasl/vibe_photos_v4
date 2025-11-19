from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

site_packages = Path(__file__).resolve().parents[1] / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
sys.path.append(str(site_packages))

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vibe_photos.config import Settings
from vibe_photos.pipeline import PreprocessingPipeline, RunJournalRecord, save_run_journal


@contextmanager
def _dummy_session(*_args, **_kwargs):
    yield object()


def _stub_stage(recorder: List[str], name: str):
    def _stage(*_args, **_kwargs):
        recorder.append(name)

    return _stage


def test_pipeline_skips_completed_and_resumes_next_stage(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    projection_db = cache_root / "index.db"
    primary_db = tmp_path / "data" / "index.db"
    primary_db.parent.mkdir(parents=True, exist_ok=True)

    save_run_journal(cache_root, RunJournalRecord(stage="thumbnails", cursor_image_id=None, updated_at=0.0))

    settings = Settings()
    pipeline = PreprocessingPipeline(settings=settings)

    monkeypatch.setattr("vibe_photos.pipeline.open_primary_session", _dummy_session)
    monkeypatch.setattr("vibe_photos.pipeline.open_projection_session", _dummy_session)

    executed: List[str] = []
    pipeline._run_scan_and_hash = _stub_stage(executed, "scan_and_hash")  # type: ignore[assignment]
    pipeline._run_perceptual_hashing_and_duplicates = _stub_stage(executed, "phash_and_duplicates")  # type: ignore[assignment]
    pipeline._run_thumbnails = _stub_stage(executed, "thumbnails")  # type: ignore[assignment]
    pipeline._run_embeddings_and_captions = _stub_stage(executed, "embeddings_and_captions")  # type: ignore[assignment]
    pipeline._run_scene_classification = _stub_stage(executed, "scene_classification")  # type: ignore[assignment]

    pipeline.run(roots=[Path("/tmp/album")], primary_db_path=primary_db, projection_db_path=projection_db)

    assert executed == ["embeddings_and_captions", "scene_classification"]


def test_pipeline_resumes_stage_from_cursor(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    projection_db = cache_root / "index.db"
    primary_db = tmp_path / "data" / "index.db"
    primary_db.parent.mkdir(parents=True, exist_ok=True)

    save_run_journal(cache_root, RunJournalRecord(stage="embeddings_and_captions", cursor_image_id="img_002", updated_at=0.0))

    settings = Settings()
    pipeline = PreprocessingPipeline(settings=settings)

    monkeypatch.setattr("vibe_photos.pipeline.open_primary_session", _dummy_session)
    monkeypatch.setattr("vibe_photos.pipeline.open_projection_session", _dummy_session)

    executed: List[str] = []
    pipeline._run_scan_and_hash = _stub_stage(executed, "scan_and_hash")  # type: ignore[assignment]
    pipeline._run_perceptual_hashing_and_duplicates = _stub_stage(executed, "phash_and_duplicates")  # type: ignore[assignment]
    pipeline._run_thumbnails = _stub_stage(executed, "thumbnails")  # type: ignore[assignment]
    pipeline._run_embeddings_and_captions = _stub_stage(executed, "embeddings_and_captions")  # type: ignore[assignment]
    pipeline._run_scene_classification = _stub_stage(executed, "scene_classification")  # type: ignore[assignment]

    pipeline.run(roots=[Path("/tmp/album")], primary_db_path=primary_db, projection_db_path=projection_db)

    assert executed == ["embeddings_and_captions", "scene_classification"]
