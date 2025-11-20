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

Running the M1 Processing Pipeline (Single-Process)
---------------------------------------------------

For local runs, the recommended and simplest path is a single-process Typer CLI:

- `vibe_photos.dev.process` — scans album roots, populates the primary SQLite database,
  and writes all versioned cache artifacts for embeddings, captions, thumbnails, detections,
  regions, and scene labels.

Basic single-process run:

- `uv run python -m vibe_photos.dev.process --root <album_root> --db data/index.db --cache-db cache/index.db`

Key options:

- `--root`: one or more album root directories to scan (may be passed multiple times).
- `--db`: path to the primary operational SQLite database (default `data/index.db`).
- `--cache-db`: path to the projection SQLite database (default `cache/index.db`).
- `--batch-size`: override the model batch size from `config/settings.yaml`.
- `--device`: override the model device (for example `cpu`, `cuda`, or `mps`).
- `--image-path`: process a single image into the projection cache using shared preprocessing steps.
- `--image-id`: optional explicit `image_id` when `--image-path` is used; defaults to the content hash.

What a single-process run does in M1:

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
- Computes lightweight scene classification attributes and stores them in
  `image_scene` (with any auxiliary JSON cached under `cache/detections/`).
- Optionally runs OWL-ViT based region detection (when enabled in settings) and
  writes region metadata under `cache/regions/<image_id>.json` and
  `image_region`.
- Writes a cache manifest (`cache/manifest.json`) that records the cache format
  version plus the effective model/backends used for embeddings, captions, and
  detection so caches can be trusted across runs.
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

Local Single-Process Debugging
------------------------------

There is no separate debug-only entrypoint. Use
`vibe_photos.dev.process` for both full runs and local validation; the
versioned cache manifest plus the run journal keep repeated passes fast
and resumable, even when you re-run the command over the same roots.

Redis/Celery Task Queues (Optional)
-----------------------------------

You do not need Celery for basic M1 usage. The queue-backed path is only
required when you want durable task queues, explicit routing, or backfill
orchestration.

The Celery application (`vibe_photos.task_queue`) defines three queues:

- ``preprocess`` for cacheable artifacts (thumbnails, EXIF, hashes, embeddings,
  detections, captions).
- ``main`` for cache-first classification, clustering, and search-index jobs
  that only read from cached artifacts.
- ``enhancement`` for optional OCR/cloud-model passes with stricter
  concurrency.

To use Celery queues:

1. Configure broker/backend and queue names via ``config/settings.yaml`` under
   ``queues`` and ``enhancement``.
2. Start one or more workers. For local development, a single worker that
   consumes all queues is often sufficient:
   - `uv run celery -A vibe_photos.task_queue worker -Q preprocess,main,enhancement -l info`
   - For finer control, run separate workers per queue and tune concurrency via
     Celery options or the values in `settings.queues`.
3. Enqueue batches from a filesystem scan with:
   - `uv run python -m vibe_photos.dev.enqueue_celery <root1> [<root2> ...] [--enqueue-main] [--enqueue-enhancement]`
   - The helper inserts missing `images` rows into the primary database, then
     enqueues preprocessing tasks, and optionally main-stage and enhancement
     work based on the flags.
4. For ad-hoc enqueueing in Python, use:
   - `process_image.delay(<image_id>)`
   - `run_main_stage.delay(<image_id>)`
   - `run_enhancement.delay(<image_id>)`

Tasks are idempotent because artifact keys combine model names and parameter
hashes. Enhancement tasks reuse cached embeddings and write versioned manifests
so they can be enabled per-tenant without blocking the main flow.

Cache Versioning
----------------

The cache layer is versioned via a manifest file:

- `cache/manifest.json` is written on each pipeline run and includes:
  - `cache_format_version`.
  - Effective model names and backends for embeddings, captions, and detection.
  - Key pipeline settings such as thumbnail size, quality, and pHash threshold.

Going forward:

- When `cache_format_version` changes, older cache directories may be treated as
  untrusted. In those cases, re-running the preprocessing pipeline is the
  recommended way to regenerate compatible cache data and projection tables.

Notes and Limitations
---------------------

- The primary database at `data/index.db` is the source of truth for images and
  model outputs. The projection database at `cache/index.db` is rebuildable from
  cache and is safe to discard.
- All paths in this document are relative to the project root; commands should
  be executed from the repository root with the virtual environment activated.
