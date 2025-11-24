"""Dump the primary PostgreSQL database to a file."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from sqlalchemy.engine.url import URL, make_url  # noqa: E402

from utils.logging import get_logger  # noqa: E402
from vibe_photos.config import load_settings  # noqa: E402

LOGGER = get_logger(__name__)
DEFAULT_DUMP_PATH = PROJECT_ROOT / "tmp" / "primary_db.dump"


def _normalize_postgres_url(raw: str | URL) -> URL:
    url = raw if isinstance(raw, URL) else make_url(str(raw))
    if not url.drivername.startswith("postgresql"):
        raise ValueError(f"Primary database must be PostgreSQL, received {url.drivername!r}")
    # Strip driver-specific suffix for CLI tools while keeping credentials.
    return url.set(drivername="postgresql")


def _mask_url(url: URL) -> str:
    return url.set(password="***").render_as_string(hide_password=False)


def _pg_cli_dsn(url: URL) -> str:
    """Build a DSN string suitable for pg_dump/pg_restore."""

    dsn = url.set(password=None)
    return dsn.render_as_string(hide_password=False)


def _build_pg_env(url: URL) -> dict[str, str]:
    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password
    return env


def _ensure_binary_available(binary: str) -> None:
    if which(binary) is None:
        LOGGER.error("pg_dump_missing", extra={"binary": binary})
        raise SystemExit(f"Required binary not found on PATH: {binary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump the primary PostgreSQL database to a file.")
    parser.add_argument(
        "--data-db",
        type=str,
        default=None,
        help="Primary PostgreSQL database URL. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DUMP_PATH),
        help="Destination file for the dump (default: tmp/primary_db.dump).",
    )
    parser.add_argument(
        "--format",
        choices=["custom", "plain", "directory", "tar"],
        default="custom",
        help="pg_dump output format. 'custom' works best with pg_restore (default).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for pg_dump (not supported for plain format).",
    )
    parser.add_argument(
        "--pg-dump-bin",
        type=str,
        default="pg_dump",
        help="Path to the pg_dump executable (default: pg_dump from PATH).",
    )
    return parser.parse_args()


def dump_database(url: URL, output_path: Path, fmt: str, jobs: int | None, binary: str) -> None:
    if fmt == "plain" and jobs:
        raise SystemExit("--jobs is not supported when using plain format dumps.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_pg_env(url)
    dsn = _pg_cli_dsn(url)

    cmd = [binary, "--format", fmt, "--no-owner", "--file", str(output_path)]
    if jobs:
        cmd.extend(["--jobs", str(jobs)])
    cmd.append(dsn)

    LOGGER.info(
        "pg_dump_start",
        extra={"target": _mask_url(url), "output": str(output_path), "format": fmt, "jobs": jobs},
    )
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        LOGGER.error(
            "pg_dump_failed",
            extra={
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                "stdout": result.stdout.strip(),
            },
        )
        raise SystemExit(result.returncode)

    size_bytes = output_path.stat().st_size
    LOGGER.info(
        "pg_dump_complete",
        extra={"output": str(output_path), "format": fmt, "jobs": jobs, "size_bytes": size_bytes},
    )


def main() -> None:
    args = parse_args()
    settings = load_settings()
    target = args.data_db or settings.databases.primary_url
    url = _normalize_postgres_url(target)
    output_path = Path(args.output).expanduser().resolve()

    _ensure_binary_available(args.pg_dump_bin)
    dump_database(url, output_path, args.format, args.jobs, args.pg_dump_bin)


if __name__ == "__main__":
    main()
