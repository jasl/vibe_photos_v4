#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "${ROOT_DIR}/config/settings.yaml" ]]; then
  echo "Creating config/settings.yaml from example..."
  cp "${ROOT_DIR}/config/settings.example.yaml" "${ROOT_DIR}/config/settings.yaml"
fi

echo "Initializing SQLite databases..."
uv run python "${ROOT_DIR}/scripts/init_databases.py" --data-db "${ROOT_DIR}/data/index.db" --cache-db "${ROOT_DIR}/cache/index.db"

echo "Done."
