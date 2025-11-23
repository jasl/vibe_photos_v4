"""Initialize SQLite schemas for primary (data) and cache databases.

This script complements the M2 blueprint by creating:
- Existing ORM tables via :mod:`vibe_photos.db`
- New label/cluster/region schemas (not yet in the ORM)

Run via ``uv run python scripts/init_databases.py`` or through ``init_project.sh``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

# Ensure src/ is on sys.path so we can import shared logging and DB helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.logging import get_logger  # noqa: E402
from vibe_photos.db import open_primary_session, open_projection_session  # noqa: E402

LOGGER = get_logger(__name__)


DATA_TABLE_STATEMENTS: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS labels (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      key TEXT NOT NULL UNIQUE,
      display_name TEXT NOT NULL,
      level TEXT NOT NULL,
      parent_id INTEGER NULL,
      icon TEXT NULL,
      is_active INTEGER NOT NULL DEFAULT 1,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      FOREIGN KEY(parent_id) REFERENCES labels(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_aliases (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      label_id INTEGER NOT NULL,
      alias_text TEXT NOT NULL,
      language TEXT NULL,
      is_preferred INTEGER NOT NULL DEFAULT 0,
      FOREIGN KEY(label_id) REFERENCES labels(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_assignment (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      target_type TEXT NOT NULL,
      target_id TEXT NOT NULL,
      label_id INTEGER NOT NULL,
      source TEXT NOT NULL,
      label_space_ver TEXT NOT NULL,
      score REAL NOT NULL,
      extra_json TEXT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      UNIQUE(target_type, target_id, label_id, source, label_space_ver),
      FOREIGN KEY(label_id) REFERENCES labels(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS image_similarity_cluster (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      key TEXT NOT NULL UNIQUE,
      method TEXT NOT NULL,
      params_json TEXT NOT NULL,
      created_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster_membership (
      cluster_id INTEGER NOT NULL,
      target_type TEXT NOT NULL,
      target_id TEXT NOT NULL,
      distance REAL NOT NULL,
      is_center INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY(cluster_id, target_type, target_id),
      FOREIGN KEY(cluster_id) REFERENCES image_similarity_cluster(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS image_near_duplicate_group (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      canonical_image_id TEXT NOT NULL,
      method TEXT NOT NULL,
      created_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS image_near_duplicate_membership (
      group_id INTEGER NOT NULL,
      image_id TEXT NOT NULL,
      is_canonical INTEGER NOT NULL DEFAULT 0,
      distance REAL NOT NULL,
      PRIMARY KEY(group_id, image_id),
      FOREIGN KEY(group_id) REFERENCES image_near_duplicate_group(id)
    );
    """,
)


CACHE_TABLE_STATEMENTS: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS regions (
      id TEXT PRIMARY KEY,
      image_id TEXT NOT NULL,
      x_min REAL NOT NULL,
      y_min REAL NOT NULL,
      x_max REAL NOT NULL,
      y_max REAL NOT NULL,
      detector TEXT NOT NULL,
      raw_label TEXT NULL,
      raw_score REAL NOT NULL,
      created_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS region_embedding (
      region_id TEXT NOT NULL,
      model_name TEXT NOT NULL,
      embedding_path TEXT NOT NULL,
      embedding_dim INTEGER NOT NULL,
      backend TEXT NOT NULL,
      updated_at REAL NOT NULL,
      PRIMARY KEY (region_id, model_name),
      FOREIGN KEY(region_id) REFERENCES regions(id)
    );
    """,
)


def _ensure_parents(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - simple filesystem guard
        LOGGER.error("init_db_parent_error", extra={"path": str(path), "error": str(exc)})
        raise


def _run_statements(path: Path, statements: Iterable[str]) -> None:
    _ensure_parents(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        cur = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()


def _init_primary_db(path: Path) -> None:
    # Create existing ORM tables.
    session = open_primary_session(path)
    session.close()
    # Create additional label/cluster/near-duplicate tables.
    _run_statements(path, DATA_TABLE_STATEMENTS)
    LOGGER.info("init_primary_db_ok", extra={"path": str(path)})


def _init_cache_db(path: Path) -> None:
    session = open_projection_session(path)
    session.close()
    _run_statements(path, CACHE_TABLE_STATEMENTS)
    LOGGER.info("init_cache_db_ok", extra={"path": str(path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize primary and cache SQLite schemas.")
    parser.add_argument("--data-db", type=Path, default=Path("data/index.db"), help="Path to primary data DB.")
    parser.add_argument("--cache-db", type=Path, default=Path("cache/index.db"), help="Path to cache DB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _init_primary_db(args.data_db)
    _init_cache_db(args.cache_db)
    LOGGER.info(
        "init_databases_complete",
        extra={"data_db": str(args.data_db), "cache_db": str(args.cache_db)},
    )


if __name__ == "__main__":
    main()
