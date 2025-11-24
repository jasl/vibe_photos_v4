"""Initialize the primary database schema and ensure the cache root exists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path so we can import shared logging and DB helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.logging import get_logger  # noqa: E402
from vibe_photos.config import load_settings  # noqa: E402
from vibe_photos.db import open_primary_session  # noqa: E402
from vibe_photos.db_helpers import normalize_cache_target  # noqa: E402

LOGGER = get_logger(__name__)


def _init_primary_db(target: str | Path) -> None:
    session = open_primary_session(target)
    session.close()
    LOGGER.info("init_primary_db_ok", extra={"target": str(target)})


def _init_cache_db(target: str | Path) -> None:
    session = open_primary_session(target)
    session.close()
    LOGGER.info("init_cache_db_ok", extra={"target": str(target)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the primary database schema and cache root.")
    parser.add_argument(
        "--data-db",
        type=str,
        default=None,
        help="Primary database URL or path. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        "--cache-db",
        dest="cache_root",
        type=str,
        default=None,
        help="Cache root URL or path. Defaults to databases.cache_url in settings.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    primary_target = args.data_db or settings.databases.primary_url
    cache_input = args.cache_root or settings.databases.cache_url
    cache_target = normalize_cache_target(cache_input)
    _init_primary_db(primary_target)
    _init_cache_db(cache_target)
    LOGGER.info(
        "init_databases_complete",
        extra={"primary": str(primary_target), "cache": str(cache_target)},
    )


if __name__ == "__main__":
    main()
