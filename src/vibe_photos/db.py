"""SQLite helpers and schema initialization for the M1 pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from utils.logging import get_logger


LOGGER = get_logger(__name__)


PRIMARY_DB_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS images (
      image_id        TEXT PRIMARY KEY,
      primary_path    TEXT NOT NULL,
      all_paths       TEXT NOT NULL,
      size_bytes      INTEGER NOT NULL,
      mtime           REAL NOT NULL,
      width           INTEGER,
      height          INTEGER,
      exif_datetime   TEXT,
      camera_model    TEXT,
      hash_algo       TEXT NOT NULL,
      phash           TEXT,
      phash_algo      TEXT,
      phash_updated_at REAL,
      created_at      REAL NOT NULL,
      updated_at      REAL NOT NULL,
      status          TEXT NOT NULL,
      error_message   TEXT,
      schema_version  INTEGER NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_images_primary_path ON images(primary_path);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
    """,
    """
    CREATE TABLE IF NOT EXISTS image_scene (
      image_id           TEXT PRIMARY KEY,
      scene_type         TEXT NOT NULL,
      scene_confidence   REAL,
      has_text           INTEGER NOT NULL,
      has_person         INTEGER NOT NULL,
      is_screenshot      INTEGER NOT NULL,
      is_document        INTEGER NOT NULL,
      classifier_name    TEXT NOT NULL,
      classifier_version TEXT NOT NULL,
      updated_at         REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_scene_scene_type ON image_scene(scene_type);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_scene_has_text ON image_scene(has_text);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_scene_has_person ON image_scene(has_person);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_scene_is_screenshot ON image_scene(is_screenshot);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_scene_is_document ON image_scene(is_document);
    """,
    """
    CREATE TABLE IF NOT EXISTS image_embedding (
      image_id       TEXT NOT NULL,
      model_name     TEXT NOT NULL,
      embedding_path TEXT NOT NULL,
      embedding_dim  INTEGER NOT NULL,
      model_backend  TEXT NOT NULL,
      updated_at     REAL NOT NULL,
      PRIMARY KEY (image_id, model_name)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_embedding_model_name ON image_embedding(model_name);
    """,
    """
    CREATE TABLE IF NOT EXISTS image_caption (
      image_id      TEXT NOT NULL,
      model_name    TEXT NOT NULL,
      caption       TEXT NOT NULL,
      model_backend TEXT NOT NULL,
      updated_at    REAL NOT NULL,
      PRIMARY KEY (image_id, model_name)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_caption_model_name ON image_caption(model_name);
    """,
    """
    CREATE TABLE IF NOT EXISTS image_near_duplicate (
      anchor_image_id    TEXT NOT NULL,
      duplicate_image_id TEXT NOT NULL,
      phash_distance     INTEGER NOT NULL,
      created_at         REAL NOT NULL,
      PRIMARY KEY (anchor_image_id, duplicate_image_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_image_near_duplicate_duplicate ON image_near_duplicate(duplicate_image_id);
    """,
)


PROJECTION_DB_SCHEMA_STATEMENTS: tuple[str, ...] = PRIMARY_DB_SCHEMA_STATEMENTS


def _ensure_parent_directory(path: Path) -> None:
    """Ensure that the parent directory of a database path exists."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("db_parent_directory_error", extra={"path": str(path), "error": str(exc)})
        raise


def _apply_schema(connection: sqlite3.Connection, statements: Iterable[str]) -> None:
    """Apply a sequence of SQL statements to a SQLite connection."""

    cursor = connection.cursor()
    for statement in statements:
        cursor.execute(statement)
    connection.commit()


def open_primary_db(path: Path) -> sqlite3.Connection:
    """Open the primary SQLite database and ensure the M1 schema exists.

    Args:
        path: Filesystem path to the primary database, typically under ``data/``.

    Returns:
        An open SQLite connection with the schema initialized.
    """

    _ensure_parent_directory(path)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    _apply_schema(connection, PRIMARY_DB_SCHEMA_STATEMENTS)
    return connection


def open_projection_db(path: Path) -> sqlite3.Connection:
    """Open the projection SQLite database and ensure the M1 schema exists.

    Args:
        path: Filesystem path to the projection database, typically under ``cache/``.

    Returns:
        An open SQLite connection with the schema initialized.
    """

    _ensure_parent_directory(path)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    _apply_schema(connection, PROJECTION_DB_SCHEMA_STATEMENTS)
    return connection


__all__ = ["open_primary_db", "open_projection_db"]
