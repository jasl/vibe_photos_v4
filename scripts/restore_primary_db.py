"""Restore a primary PostgreSQL database from a dump file."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine.url import URL, make_url  # noqa: E402

from utils.logging import get_logger  # noqa: E402
from vibe_photos.config import load_settings  # noqa: E402

LOGGER = get_logger(__name__)
DEFAULT_DUMP_PATH = PROJECT_ROOT / "tmp" / "primary_db.dump"


def _normalize_postgres_url(raw: str | URL) -> URL:
    url = raw if isinstance(raw, URL) else make_url(str(raw))
    if not url.drivername.startswith("postgresql"):
        raise ValueError(f"Primary database must be PostgreSQL, received {url.drivername!r}")
    return url.set(drivername="postgresql")


def _mask_url(url: URL) -> str:
    return url.set(password="***").render_as_string(hide_password=False)


def _pg_cli_dsn(url: URL) -> str:
    dsn = url.set(password=None)
    return dsn.render_as_string(hide_password=False)


def _build_pg_env(url: URL) -> dict[str, str]:
    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password
    return env


def _ensure_binary_available(binary: str) -> None:
    if which(binary) is None:
        LOGGER.error("pg_restore_missing", extra={"binary": binary})
        raise SystemExit(f"Required binary not found on PATH: {binary}")


def _validate_identifier(identifier: str, label: str) -> None:
    if not identifier:
        raise ValueError(f"{label} cannot be empty.")
    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
        raise ValueError(f"{label} must contain only letters, numbers, or underscores: {identifier!r}")


def _ensure_database(url: URL, drop_existing: bool, maintenance_db: str) -> None:
    """Create or recreate the target database using a maintenance connection.

    Uses SQLAlchemy Core because CREATE/DROP DATABASE cannot be expressed via ORM models.
    """

    database = url.database
    if not database:
        raise ValueError("Target URL must include a database name.")

    _validate_identifier(database, "Database name")
    if url.username:
        _validate_identifier(url.username, "Database owner")

    admin_url = url.set(database=maintenance_db, drivername="postgresql+psycopg")
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)

    with engine.connect() as conn:
        if drop_existing:
            LOGGER.info(
                "drop_database",
                extra={"database": database, "maintenance_db": maintenance_db, "target": _mask_url(url)},
            )
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": database},
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database}"'))

        existing = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": database})
        exists = existing.scalar_one_or_none() is not None
        if not exists:
            owner_clause = f' OWNER "{url.username}"' if url.username else ""
            conn.execute(text(f'CREATE DATABASE "{database}"{owner_clause}'))
            LOGGER.info(
                "create_database",
                extra={"database": database, "owner": url.username, "maintenance_db": maintenance_db},
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore the primary PostgreSQL database from a dump.")
    parser.add_argument(
        "--data-db",
        type=str,
        default=None,
        help="Target primary database URL. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--dump-file",
        type=str,
        default=str(DEFAULT_DUMP_PATH),
        help="Path to the dump produced by scripts/dump_primary_db.py (default: tmp/primary_db.dump).",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the database before restoring. Terminates active connections.",
    )
    parser.add_argument(
        "--maintenance-db",
        type=str,
        default="postgres",
        help="Database used for administrative commands when dropping/creating (default: postgres).",
    )
    parser.add_argument(
        "--pg-restore-bin",
        type=str,
        default="pg_restore",
        help="Path to the pg_restore executable (default: pg_restore from PATH).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for pg_restore. Matches the dump's format (custom/tar/directory).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip --clean/--if-exists so existing objects remain (default behavior cleans).",
    )
    return parser.parse_args()


def restore_database(url: URL, dump_path: Path, jobs: int | None, clean: bool, binary: str) -> None:
    env = _build_pg_env(url)
    dsn = _pg_cli_dsn(url)

    cmd = [binary, "--no-owner", "--dbname", dsn]
    if clean:
        cmd.extend(["--clean", "--if-exists"])
    if jobs:
        cmd.extend(["--jobs", str(jobs)])
    cmd.append(str(dump_path))

    LOGGER.info(
        "pg_restore_start",
        extra={"target": _mask_url(url), "dump": str(dump_path), "clean": clean, "jobs": jobs},
    )
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        LOGGER.error(
            "pg_restore_failed",
            extra={
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                "stdout": result.stdout.strip(),
            },
        )
        raise SystemExit(result.returncode)

    LOGGER.info(
        "pg_restore_complete",
        extra={"target": _mask_url(url), "dump": str(dump_path), "clean": clean, "jobs": jobs},
    )


def main() -> None:
    args = parse_args()
    settings = load_settings()
    target = args.data_db or settings.databases.primary_url
    url = _normalize_postgres_url(target)
    dump_path = Path(args.dump_file).expanduser().resolve()
    if not dump_path.exists():
        raise SystemExit(f"Dump file not found: {dump_path}")

    _ensure_binary_available(args.pg_restore_bin)
    _ensure_database(url, args.drop_existing, args.maintenance_db)
    restore_database(url, dump_path, args.jobs, not args.skip_clean, args.pg_restore_bin)


if __name__ == "__main__":
    main()
