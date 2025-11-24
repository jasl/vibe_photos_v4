"""Shared helpers for database URLs and dialect-aware operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session


def normalize_database_url(target: str | Path) -> str:
    """Normalize database URL or path inputs to absolute URLs."""

    if isinstance(target, Path):
        return f"sqlite:///{target.resolve()}"

    raw = str(target).strip()
    if not raw:
        raise ValueError("database target cannot be empty")

    if "://" not in raw:
        return f"sqlite:///{Path(raw).resolve()}"

    url = make_url(raw)
    if url.drivername.startswith("sqlite"):
        database = url.database or ""
        if database not in {":memory:", ""}:
            db_path = Path(database)
            if not db_path.is_absolute():
                db_path = (Path.cwd() / db_path).resolve()
            url = url.set(database=str(db_path))
        return str(url)

    return raw


def sqlite_path_from_target(target: str | Path) -> Path:
    """Return an absolute filesystem path for a SQLite target."""

    if isinstance(target, Path):
        return target.resolve()

    raw = str(target).strip()
    if "://" not in raw:
        return Path(raw).resolve()

    url = make_url(raw)
    if not url.drivername.startswith("sqlite"):
        raise ValueError(f"Expected sqlite URL, received {raw!r}")

    database = url.database or ""
    if database == ":memory:":
        raise ValueError("in-memory SQLite URLs are not supported for cache roots")

    db_path = Path(database)
    if not db_path.is_absolute():
        db_path = (Path.cwd() / db_path).resolve()
    return db_path


def dialect_insert(session: Session, table: Any) -> Any:
    """Return a dialect-aware INSERT statement supporting ON CONFLICT."""

    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Session is not bound to an engine.")

    name = bind.dialect.name
    if name == "sqlite":
        return sqlite_insert(table)
    if name.startswith("postgresql"):
        return pg_insert(table)
    raise NotImplementedError(f"Unsupported dialect for upsert: {name}")


def normalize_cache_target(target: str | Path) -> str:
    """Return a cache DB target string, accepting cache roots or legacy DB paths."""

    if isinstance(target, Path):
        base = target
    else:
        raw = str(target).strip()
        if not raw:
            raise ValueError("cache target cannot be empty")
        if "://" in raw:
            return raw
        base = Path(raw)

    resolved = base.resolve()
    if resolved.suffix == ".db":
        return str(resolved)
    return str((resolved / "index.db").resolve())


__all__ = [
    "normalize_database_url",
    "sqlite_path_from_target",
    "normalize_cache_target",
    "dialect_insert",
]
