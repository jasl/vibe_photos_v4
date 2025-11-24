"""Shared helpers for database URLs and dialect-aware operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session


def normalize_database_url(target: str | Path) -> str:
    """Normalize database URLs, permitting sqlite paths for local/test flows."""

    raw = str(target).strip()
    if not raw:
        raise ValueError("database target cannot be empty")

    if "://" not in raw:
        path = Path(raw).expanduser().resolve()
        return f"sqlite:///{path}"

    url = make_url(raw)
    if url.drivername.startswith("postgresql"):
        return str(url)

    if url.drivername.startswith("sqlite"):
        if url.database:
            return str(url)
        raise ValueError("sqlite URLs must include a database path or :memory:")

    raise ValueError(f"Unsupported database dialect {url.drivername!r}; expected postgresql or sqlite")


def resolve_cache_root(target: str | Path) -> Path:
    """Resolve a filesystem cache root, ignoring legacy SQLite URL formats."""

    raw = str(target).strip()
    if not raw:
        raise ValueError("cache target cannot be empty")

    if "://" in raw:
        raise ValueError("cache roots must be filesystem paths; URLs are not supported")

    path = Path(raw).resolve()
    if path.suffix == ".db":
        raise ValueError("cache root must be a directory, not a legacy SQLite file path")

    return path


def dialect_insert(session: Session, table: Any) -> Any:
    """Return a dialect-aware INSERT statement supporting ON CONFLICT."""

    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Session is not bound to an engine.")

    name = bind.dialect.name
    if name.startswith("postgresql"):
        return pg_insert(table)
    if name.startswith("sqlite"):
        return sqlite_insert(table)

    raise NotImplementedError(f"Unsupported dialect for upsert: {name}")


__all__ = [
    "normalize_database_url",
    "resolve_cache_root",
    "dialect_insert",
]
