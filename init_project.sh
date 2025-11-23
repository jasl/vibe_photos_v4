#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "${ROOT_DIR}/config/settings.yaml" ]]; then
  echo "Creating config/settings.yaml from example..."
  cp "${ROOT_DIR}/config/settings.example.yaml" "${ROOT_DIR}/config/settings.yaml"
fi

echo "Initializing databases..."
uv run python "${ROOT_DIR}/scripts/init_databases.py"

# Seed base labels (scene/attr/object) idempotently.
echo "Seeding labels into primary database..."
uv run python -m vibe_photos.labels.seed_labels

# Build object text prototypes once so downstream passes can run immediately.
if [[ "${BUILD_OBJECT_PROTOTYPES:-1}" -eq 1 ]]; then
  echo "Building object label prototypes..."
  uv run python -m vibe_photos.labels.build_object_prototypes --output-name object_v1
fi

echo "Done."
