"""Unit tests for database helper utilities."""

from __future__ import annotations

from sqlalchemy.engine.url import make_url

from vibe_photos.db_helpers import normalize_database_url


def test_normalize_database_url_accepts_postgres_url() -> None:
    url = "postgresql+psycopg://postgres:secret@127.0.0.1:5432/vibe"

    assert normalize_database_url(url) == str(make_url(url))

