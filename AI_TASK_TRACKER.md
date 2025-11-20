# AI Task Tracker — Vibe Photos Phase Final

This file tracks high-level implementation tasks and their status for the Phase Final architecture.

## Milestones

- [ ] M1 — Preprocessing & Feature Extraction (Local, No Containers)
- [ ] M2 — Search & Tools (SQLite + Local Services)
- [ ] M3 — Database & Deployment Upgrade (PostgreSQL + pgvector + docker-compose)
- [ ] M4 — Learning & Personalization

## Current Tasks (Outline)

### M1 — Preprocessing & Feature Extraction

- [x] Set up `uv` environment and base dependencies. (Project is pinned to Python 3.12 with `uv` metadata in `pyproject.toml`.)
- [x] Implement shared logging and configuration modules. (See `src/utils/logging.py` and `src/vibe_photos/config.py`.)
- [x] Define initial SQLite schema for photos, metadata, and model outputs. (See ORM models in `src/vibe_photos/db.py`.)
- [x] Implement photo scanning and registration for local folders/NAS mounts. (Implemented in `src/vibe_photos/scanner.py` and `_run_scan_and_hash` in `src/vibe_photos/pipeline.py`.)
- [x] Normalize images and generate thumbnails / web-friendly versions. (A preprocessing stage now writes configurable JPEG thumbnails (default 256×256 small, 1024×1024 large) to `cache/images/thumbnails/`, keyed by `image_id`; `/thumbnail/<image_id>` reads from the cache with a fallback to originals. Full normalized copies under `cache/images/processed/` remain future work.)
- [x] Extract EXIF, capture time, GPS (when present), and file timestamps. (EXIF datetime and camera model are populated on `images` rows during preprocessing; a metadata sidecar is written to `cache/images/metadata/<image_id>.json` with parsed GPS coordinates when present, and the Flask UI displays these fields.)
- [x] Compute file hashes and perceptual hashes; record near-duplicate relationships. (Content hashes and pHash-based near-duplicate groups are implemented in `src/vibe_photos/hasher.py` and `_run_perceptual_hashing_and_duplicates` in `src/vibe_photos/pipeline.py`.)
- [x] Integrate SigLIP embeddings and BLIP captions; cache results. (Implemented in `_run_embeddings_and_captions` with NPY/JSON caches under `cache/` and projections in SQLite.)
- [x] (Optional) Integrate Grounding DINO / OWL-ViT detection and SigLIP re-ranking. (OWL-ViT + SigLIP region re-ranking and JSON caches are implemented; enabled via `models.detection.enabled` and `pipeline.run_detection` settings.)
- [x] Consolidated preprocessing orchestration onto Celery task queues; the legacy SQLite `preprocess_task` queue and associated enqueue/worker CLIs have been removed in favor of `vibe_photos.task_queue` and `vibe_photos.dev.enqueue_celery`.
- [x] Add a Celery-backed enqueue helper to scan directories and push preprocessing/main/enhancement jobs to dedicated queues (`vibe_photos.dev.enqueue_celery`).
- [x] Define a stable, versioned on-disk format for preprocessing caches under `cache/` that is decoupled from the database schema. (A cache manifest (`cache/manifest.json`) and per-image JSON sidecars now version embeddings, captions, detections, and regions; projection tables are populated during pipeline runs and rely on the manifest for trust.)
- [x] Build a simple Flask-based debug UI to list canonical photos and show per-photo preprocessing details and similar images. (Implemented in `src/vibe_photos/webui/__init__.py` with templates under `src/vibe_photos/webui/templates`.)
- [x] Standardize database access on SQLAlchemy ORM/Core models and prohibit new raw SQL usage in pipeline and web UI.

#### M1 — Extras beyond the original plan

- OWL-ViT region detection with SigLIP-based label re-ranking is fully wired into the pipeline, with results stored in the `image_region` table and JSON caches under `cache/regions/`. SigLIP re-ranking now applies confidence and margin thresholds and only refines labels within the same semantic group as the original detector label; if a detector label cannot be grouped confidently, the region keeps only the detector output (no cross-category overrides such as random `AirPods` on food regions).
- A standalone SigLIP+BLIP helper (`SiglipBlipDetector` in `src/vibe_photos/ml/siglip_blip.py`) is available for ad-hoc zero-shot classification + captioning outside the main pipeline.
- The Flask debug UI exposes additional filters such as duplicate-hiding, near-duplicate facets, and region-label filtering on top of the planned scene/attribute filters.
- Region detection now includes: (1) class-agnostic NMS to merge highly overlapping boxes; (2) a configurable priority heuristic combining detector score, normalized box area, and distance to image center; (3) caption-based fallback regions for cases where detection misses an obvious primary object (e.g., drinks in front of background food); and (4) priority-aware filtering of low-value secondary regions so the database and UI focus on a small number of high-quality boxes per image. Priority is not stored in the database but is recomputed in the Web UI using the same heuristic for transparency and tuning.
- Shared preprocessing steps live in `src/vibe_photos/preprocessing.py` and back both Celery workers and the `src/vibe_photos/dev/process.py --image-path` single-image helper for local runs.

#### M1 — Known PoC / placeholder areas

- Full image normalization and storage under `cache/images/processed/` is not yet implemented; only thumbnails are generated in the preprocessing pipeline today.
- EXIF and GPS metadata are parsed during preprocessing and surfaced in the debug UI, but the on-disk metadata format is minimal and may evolve as later milestones add richer EXIF/sidecar handling.
- The preprocessing pipeline is resumable via a JSON run journal in `cache/run_journal.json`; it now skips completed stages and resumes batch cursors. Celery (`vibe_photos.task_queue`) is available for durable preprocessing/main/enhancement workers, while the single-process loop remains the default local entrypoint.
- The projection database (`cache/index.db`) is populated alongside cache artifacts; cache validity is gated by the manifest version rather than by a separate rebuild path.
 - Caption-aware primary-region fallback in the detection stage assumes that BLIP captions have already been computed and written to `image_caption` for any image that runs detection. Future incremental “detection-only” entry points must either preserve this ordering (captions first) or gracefully disable/adjust caption-based fallbacks to avoid surprising gaps in primary regions.

#### Future technical improvements (beyond M1)

- Optimize pHash-based near-duplicate grouping to reduce worst-case O(N²) behavior (for example via bucketing or approximate nearest-neighbor strategies) once library scale and performance bottlenecks are better understood.
- Introduce a configurable label blacklist / remapping layer for region and image-level labels so low-information or noisy nouns (for example `butter`, `water`) can be suppressed or mapped to coarser categories during SigLIP refinement and caption-based fallbacks. The blacklist should live in configuration (alongside SigLIP label groups) and be applied as a post-processing step, without changing underlying model logits or shared label dictionaries.

### M2 — Search & Tools

- [ ] Implement FastAPI endpoints backed by SQLite for search and inspection.
- [ ] Implement CLI commands for preprocessing, search, and exports.
- [ ] Build a minimal Streamlit UI for search and result browsing.

### M3 — PostgreSQL + pgvector + docker-compose

- [ ] Design PostgreSQL schema and migrations.
- [ ] Implement data migration from SQLite (where feasible).
- [ ] Define `docker-compose` stack (API, workers, DB, Redis, UI).

### M4 — Learning & Personalization

- [ ] Implement few-shot learning pipeline using real embeddings.
- [ ] Build annotation UI for corrections and batch actions.
- [ ] Wire feedback into ranking and prototype updates.

Update this file as tasks progress so future coding AIs and human developers can quickly understand the current implementation state.
