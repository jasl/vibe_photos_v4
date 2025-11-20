Vibe Photos — M1 Preprocessing Quickstart
========================================

This document describes how to run the M1 preprocessing pipeline and related
tools (debug UI and cache rebuild) on a local machine.

Environment Setup
-----------------

- Requirements:
  - Python 3.12
  - `uv` package manager
- Recommended one-time setup:
  - Create and activate the virtual environment:
    - `uv venv --python 3.12`
    - `source .venv/bin/activate`
  - Install dependencies:
    - `uv sync`
  - Initialize project configuration:
    - `./init_project.sh`
    - This will create `config/settings.yaml` with default model and pipeline
      settings; adjust album roots, batch sizes, and device as needed.
- Optional but recommended environment variables to avoid repeated downloads:
  - `TRANSFORMERS_CACHE=./models/transformers`
  - `PADDLEOCR_HOME=./models/paddleocr`

Pre-downloading Models
----------------------

Fetch the configured Hugging Face checkpoints ahead of running the pipeline to avoid first-request downloads:

- `uv run python -m vibe_photos.dev.download_models`
  - Pass `--include-detection` to download the optional OWL-ViT weights even when `models.detection.enabled` is false.
  - Pass `--skip-detection` to omit detection downloads when they are not needed.

GPU Acceleration (CUDA)
-----------------------

On Linux or Windows machines with a compatible NVIDIA GPU and CUDA 13.0:

- Ensure the GPU environment is set up (NVIDIA driver + CUDA runtime) and run `uv sync` on that machine so PyTorch with CUDA is installed from the configured `pytorch-cuda` index.
- In `config/settings.yaml`, set:
  - `models.embedding.device: "cuda"`
  - `models.caption.device: "cuda"`
  - `models.detection.device: "cuda"` (if detection is enabled)
- Alternatively, pass `--device cuda` to the preprocessing CLI or workers to override the configured device at runtime.

Running the M1 Preprocessing Pipeline
-------------------------------------

The main entrypoint for M1 is a Typer-based CLI that scans album roots,
populates the primary SQLite database, and writes cache artifacts for
embeddings, captions, thumbnails, and detections.

Basic usage:

- `uv run python -m vibe_photos.dev.preprocess --root <album_root> --db data/index.db --cache-db cache/index.db`

Key options:

- `--root`: one or more album root directories to scan (may be passed multiple times).
- `--db`: path to the primary operational SQLite database (default `data/index.db`).
- `--cache-db`: path to the projection SQLite database (default `cache/index.db`).
- `--batch-size`: override the model batch size from `config/settings.yaml`.
- `--device`: override the model device (for example `cpu`, `cuda`, or `mps`).

What the pipeline does in M1:

- Scans the specified roots and registers images in `images` with stable content
  hashes (`image_id`) and file metadata.
- Computes perceptual hashes (`phash`) and builds near-duplicate groups in
  `image_near_duplicate`.
- Generates JPEG thumbnails under `cache/images/thumbnails/` and writes EXIF and
  GPS metadata JSON under `cache/images/metadata/`.
- Runs SigLIP embeddings and BLIP captions for canonical images and stores:
  - Embedding vectors under `cache/embeddings/<image_id>.npy`.
  - Embedding metadata under `cache/embeddings/<image_id>.json`.
  - Caption payloads under `cache/captions/<image_id>.json`.
  - Corresponding projection rows in `image_embedding` and `image_caption`.
- Computes lightweight scene classification attributes and caches them under
  `cache/detections/<image_id>.json` as well as in `image_scene`.
- Optionally runs OWL-ViT based region detection (when enabled in settings) and
  writes region metadata under `cache/regions/<image_id>.json` and
  `image_region`.
- Maintains a run journal at `cache/run_journal.json` so that interrupted runs
  can resume without reprocessing completed batches when models are re-used.

Debug UI for M1 Outputs (Flask)
-------------------------------

After running the preprocessing pipeline, you can inspect results through a
simple Flask-based debug UI.

Start the UI:

- `FLASK_APP=vibe_photos.webui uv run flask run`

By default the UI connects to `data/index.db` in the project root. Once the
server is running, open the following in a browser:

- `http://127.0.0.1:5000/`

The UI provides:

- A paginated grid of images with thumbnails, scene type, and boolean flags
  (`has_text`, `has_person`, `is_screenshot`, `is_document`).
- Filters for scene type, presence of text/people, screenshots/documents, and
  optional region labels.
- Controls to hide or focus on near-duplicates based on `image_near_duplicate`.
- A detail view for each image that shows:
  - Original file path, size, dimensions, and status.
  - EXIF datetime, camera model, and GPS coordinates (when available).
  - BLIP caption and classifier attributes.
  - Near-duplicate neighbors and region detections (if detection is enabled).

Rebuilding Projection Tables from Cache (`rebuild_cache`)
---------------------------------------------------------

Projection tables in `cache/index.db` are considered disposable and can be
rebuilt from cache artifacts under `cache/`. A dedicated CLI entrypoint is
provided for this purpose.

Basic usage:

- `uv run python -m vibe_photos.dev.rebuild_cache --cache-root cache --cache-db cache/index.db`

Options:

- `--cache-root`: root directory for cache artifacts (default `cache`).
- `--cache-db`: SQLite database whose projection tables should be rebuilt
  (default `cache/index.db`).
- `--reset-db/--no-reset-db`:
  - When `--reset-db` (default), all projection tables (`image_scene`,
    `image_embedding`, `image_caption`, `image_near_duplicate`, `image_region`)
    are cleared before rebuilding.
  - Use `--no-reset-db` when you want to rebuild only missing rows and preserve
    any existing data.

What `rebuild_cache` currently restores:

- Embeddings: from `cache/embeddings/<image_id>.npy` and
  `cache/embeddings/<image_id>.json` into `image_embedding`.
- Captions: from `cache/captions/<image_id>.json` into `image_caption`.
- Scene attributes: from `cache/detections/<image_id>.json` into `image_scene`.
- Regions: from `cache/regions/<image_id>.json` into `image_region`.

Local Single-Process Debugging
------------------------------

`vibe_photos.dev.preprocess` remains a single-threaded entrypoint intended for
quick debugging runs:

- `uv run python -m vibe_photos.dev.preprocess --root <album_root> --db data/index.db --cache-db cache/index.db`

Use this path when you need to validate configuration or model changes locally
without bringing up external services.

Redis/Celery Task Queues (Preprocess/Main/Enhancement)
------------------------------------------------------

A Celery application (`vibe_photos.task_queue`) is available when you need
durable queues, explicit routing, or backfill orchestration. It defines three
queues:

- ``preprocess`` for cacheable artifacts (thumbnails, EXIF, hashes, embeddings,
  detections, captions).
- ``main`` for cache-first classification, clustering, and search-index jobs
  that only read from cached artifacts.
- ``enhancement`` for optional OCR/cloud-model passes with stricter
  concurrency.

Key usage notes:

- Configure broker/backend and queue names via ``config/settings.yaml`` under
  ``queues`` and ``enhancement``.
- Start workers with ``celery -A vibe_photos.task_queue worker -Q preprocess``
  (or substitute ``main`` / ``enhancement``); concurrency defaults are derived
  from the same settings block.
- Enqueue batches from a filesystem scan with ``uv run python -m
  vibe_photos.dev.enqueue_celery <root1> [<root2> ...]``. The helper inserts
  missing ``images`` rows, then enqueues preprocessing tasks, and can also
  enqueue main-stage and enhancement work via flags.
- Enqueue work with ``process_image.delay(<image_id>)`` or
  ``run_main_stage.delay(<image_id>)``. Tasks are idempotent because artifact
  keys combine model names and parameter hashes.
- Enhancement tasks reuse cached embeddings and write versioned manifests so
  they can be enabled per-tenant without blocking the main flow.

Cache Versioning
----------------

The cache layer is versioned via a manifest file:

- `cache/manifest.json` is written on each pipeline run and includes:
  - `cache_format_version`.
  - Effective model names and backends for embeddings, captions, and detection.
  - Key pipeline settings such as thumbnail size, quality, and pHash threshold.

Going forward:

- When `cache_format_version` changes, older cache directories may be treated as
  untrusted for projection rebuild and certain artifacts may be skipped during
  `rebuild_cache`. In those cases, re-running the preprocessing pipeline is the
  recommended way to regenerate compatible cache data.

Notes and Limitations
---------------------

- The primary database at `data/index.db` is the source of truth for images and
  model outputs. The projection database at `cache/index.db` is rebuildable from
  cache and is safe to discard.
- All paths in this document are relative to the project root; commands should
  be executed from the repository root with the virtual environment activated.
