"""Shared helpers for database URLs and dialect-aware operations."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session


def normalize_database_url(target: str) -> str:
    """Normalize PostgreSQL database URLs and reject unsupported inputs."""

    raw = target.strip()
    if not raw:
        raise ValueError("database target cannot be empty")

    if "://" not in raw:
        raise ValueError("database target must include a driver scheme such as postgresql+psycopg://")

    url = make_url(raw)
    if url.drivername.startswith("postgresql") and url.database:
        return str(raw)

    raise ValueError(f"Unsupported database dialect {url.drivername!r}; expected postgresql with an explicit database name")


def dialect_insert(session: Session, table: Any) -> Any:
    """Return a dialect-aware INSERT statement supporting ON CONFLICT."""

    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Session is not bound to an engine.")

    name = bind.dialect.name
    if name.startswith("postgresql"):
        return pg_insert(table)

    raise NotImplementedError(f"Unsupported dialect for upsert: {name}")


__all__ = [
    "normalize_database_url",
    "dialect_insert",
]
