Vibe Photos — Preprocessing & Label Layer Quickstart
====================================================

This document describes how to run the current preprocessing + label pipeline
(SigLIP/BLIP features, region embeddings, label layer, clustering) and related
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

Running the Pipeline (Single-Process)
------------------------------------

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

`cache/index.db` is a cache of *pre_process* outputs (embeddings, captions, detections, scenes) and is safe to discard; `process` stages read
from this cache and may copy into the primary DB when required.
Serve/UI/API must read from `data/index.db` only; use the sync helper (below) to copy projection tables from cache into primary when needed.

What a single-process run does:

- Scans the specified roots and registers images in `images` with stable content
  hashes (`image_id`) and file metadata.
- Computes perceptual hashes (`phash`) and builds near-duplicate groups:
  - pHash is recomputed when missing or algorithm changes.
  - Near-duplicate pairs are incremental: new or content-changed images delete their old pairs and recompute against all active images; if the table is empty, a full pass runs once. Results are written to `image_near_duplicate` in both cache and primary DBs.
- Generates JPEG thumbnails under `cache/images/thumbnails/` and writes EXIF and
  GPS metadata JSON under `cache/images/metadata/`.
- Runs SigLIP embeddings and BLIP captions for canonical images and stores:
  - Embedding vectors under `cache/embeddings/<image_id>.npy`.
  - Embedding metadata under `cache/embeddings/<image_id>.json`.
  - Caption payloads under `cache/captions/<image_id>.json`.
  - Corresponding projection rows in `image_embedding` and `image_caption` (stored in `cache/index.db`; downstream steps may copy to `data/index.db` if needed).
- Computes lightweight scene classification attributes and mirrors them into the
  label layer while retaining the legacy `image_scene` projection (rows written
  to `cache/index.db`).
- Optionally runs OWL-ViT based region detection (when enabled in settings) and
  writes region metadata under `cache/regions/<image_id>.json` plus feature-only
  `regions` / `region_embedding` projections in `cache/index.db`. Object / cluster
  labels are produced later via label passes, not in the detection step.
- Object label pass respects `object.zero_shot.scene_whitelist` and optional
  `scene_fallback_labels` to avoid noisy predictions on screenshots/documents.
- Writes a cache manifest (`cache/manifest.json`) that records the cache format
  version plus the effective model/backends used for embeddings, captions, and
  detection so caches can be trusted across runs.
- Maintains a run journal at `cache/run_journal.json` so that interrupted runs
  can resume without reprocessing completed batches when models are re-used.

Redis/Celery Task Queues (Optional)
-----------------------------------

You do not need Celery for basic M1 usage. The queue-backed path is only
required when you want durable task queues, explicit routing, or backfill
orchestration.

The Celery application (`vibe_photos.task_queue`) defines three queues:

- ``pre_process`` for cacheable artifacts (thumbnails, EXIF, hashes, embeddings,
  detections, captions).
- ``process`` for cache-first classification, clustering, and search-index jobs
  that only read from cached artifacts.
- ``post_process`` for optional OCR/cloud-model passes with stricter
  concurrency.

To use Celery queues:

1. Configure broker/backend and queue names via ``config/settings.yaml`` under
   ``queues`` and ``post_process`` (which configures the ``post_process`` stage).
2. Start one or more workers. For local development, a single worker that
   consumes all queues is often sufficient:
   - `uv run celery -A vibe_photos.task_queue worker -Q pre_process,process,post_process -l info`
   - For finer control, run separate workers per queue and tune concurrency via
     Celery options or the values in `settings.queues`.
3. Enqueue batches from a filesystem scan with:
   - `uv run python -m vibe_photos.dev.enqueue_celery <root1> [<root2> ...] [--task pre_process|process|post_process]`
   - Defaults to `--task process`; process will auto-run pre_process when cache artifacts are missing.
   - The helper inserts missing `images` rows into the primary database, then enqueues the chosen task for each image.
4. For ad-hoc enqueueing in Python, use:
   - `pre_process.delay(<image_id>)`
   - `process.delay(<image_id>)`
   - `post_process.delay(<image_id>)`

Tasks are idempotent because artifact keys combine model names and parameter
hashes. Post-process tasks reuse cached embeddings and write versioned manifests
so they can be enabled per-tenant without blocking the main flow.

To clear pending Celery tasks (for example after changing configuration), you can purge queues from the broker:

- Purge all queues for this app (dangerous – removes all pending tasks):  
  - `uv run celery -A vibe_photos.task_queue purge -f`
- Purge only the default queues used by this project:  
  - `uv run celery -A vibe_photos.task_queue purge -f -Q pre_process,process,post_process`

Object Label Noise Controls (M2)
--------------------------------

Use the object label layer to suppress or merge noisy predictions without changing model features:

- `config/settings.yaml` → `object.blacklist`: label keys to drop when writing region/image assignments.
- `config/settings.yaml` → `object.remap`: map noisy keys to canonical ones (for example `object.remap: {"object.test.old": "object.test.new"}`).
- Both apply inside the zero-shot object pass and the aggregated image labels; detection stays feature-only.
- Tune these after inspecting sample outputs to remove “butter / water / generic-food” style labels before large runs.

Debug UI for Pipeline Outputs (Flask)
-------------------------------------

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
  an object label key (uses label assignments, not raw detector labels).
- Controls to hide or focus on near-duplicates based on `image_near_duplicate`.
- A detail view for each image that shows:
  - Original file path, size, dimensions, and status.
  - EXIF datetime, camera model, and GPS coordinates (when available).
  - BLIP caption and classifier attributes.
  - Object and cluster labels (top-ranked per image) and region-level object/cluster labels when detection is enabled.
  - Near-duplicate neighbors and region detections (if detection is enabled).

Cache Versioning
----------------

The cache layer is versioned via a manifest file:

- `cache/manifest.json` is written on each pipeline run and includes:
  - `cache_format_version`.
  - Effective model names and backends for embeddings, captions, and detection.
  - Key pipeline settings such as thumbnail size, quality, and pHash threshold.
  - Settings that only affect throughput (for example batch sizes) are intentionally
    omitted so they do not spuriously invalidate caches.

Going forward:

- When the manifest contents change (for example model swap or cache format bump),
  the pipeline clears cache artifacts and `cache/index.db` before writing a fresh
  manifest, forcing regeneration on the next run.
- When `cache_format_version` changes, older cache directories are treated as
  untrusted. Re-run the preprocessing pipeline to regenerate compatible cache data
  and projection tables.

Notes and Limitations
---------------------

- `cache/index.db` is the cache DB for pre_process outputs and can be safely
  discarded; cached model outputs should be considered authoritative there, and
  `process` stages copy needed subsets into `data/index.db` for API/UI use.
- Use `uv run python -m vibe_photos.dev.clear_cache --stage <...>` to invalidate
  specific cache stages (or `--full-reset` to clear all caches and cache/index.db).
- When a file’s content hash changes, its cache artifacts and near-duplicate pairs in `cache/index.db` are invalidated and rebuilt on the next run; primary DB rows remain intact for auditability.
- All paths in this document are relative to the project root; commands should
  be executed from the repository root with the virtual environment activated.
