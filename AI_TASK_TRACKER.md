# AI Task Tracker — Vibe Photos Phase Final

This file tracks high-level implementation tasks and their status for the Phase Final architecture.

## Milestones

- [x] M1 — Preprocessing & Feature Extraction (Local, No Containers)
- [ ] M2 — Perception Quality & Labeling
- [ ] M3 — Search & Tools (PostgreSQL + pgvector + docker-compose)
- [ ] M4 — Learning & Personalization

## Current Tasks (Outline)

### M1 — Preprocessing & Feature Extraction

- [x] Set up `uv` environment and base dependencies. (Project is pinned to Python 3.12 with `uv` metadata in `pyproject.toml`.)
- [x] Implement shared logging and configuration modules. (See `src/utils/logging.py` and `src/vibe_photos/config.py`.)
- [x] Define initial PostgreSQL schema for photos, metadata, and model outputs. (See ORM models in `src/vibe_photos/db.py`; legacy file-backed database paths are no longer supported.)
- [x] Implement photo scanning and registration for local folders/NAS mounts. (Implemented in `src/vibe_photos/scanner.py` and `_run_scan_and_hash` in `src/vibe_photos/pipeline.py`.)
- [x] Normalize images and generate thumbnails / web-friendly versions. (A preprocessing stage now writes configurable JPEG thumbnails (default 256×256 small, 1024×1024 large) to artifact-managed paths such as `cache/artifacts/<image_id>/thumbnail_large/<hash>/thumbnail_1024.jpg`, keyed by `image_id`; `/thumbnail/<image_id>` reads from the recorded artifact path with a fallback to originals. Full normalized copies under `cache/images/processed/` remain future work.)
- [x] Extract EXIF, capture time, GPS (when present), and file timestamps. (EXIF datetime, camera model, and GPS coordinates are now written to both the primary `images` table and a metadata artifact under `cache/artifacts/<image_id>/metadata/`, so downstream services and the Web UI no longer read `cache/images/metadata` directly.)
- [x] Compute file hashes and perceptual hashes; record near-duplicate relationships. (Content hashes and pHash-based near-duplicate groups are implemented in `src/vibe_photos/hasher.py` and `_run_perceptual_hashing_and_duplicates` in `src/vibe_photos/pipeline.py`; pHash is recomputed only when missing/algorithm changes, and near-duplicate pairs are incremental—dirty images drop their old pairs and recompute against active images, full pass only when the table is empty.)
- [x] Integrate SigLIP embeddings and BLIP captions; cache results. (Implemented in `_run_embeddings_and_captions` with NPY/JSON caches under `cache/` and metadata persisted in the primary database.)
- [x] (Optional) Integrate Grounding DINO / OWL-ViT detection and SigLIP re-ranking. (OWL-ViT + SigLIP region re-ranking and JSON caches are implemented; enabled via `models.detection.enabled` and `pipeline.run_detection` settings.)
- [x] Consolidated preprocessing orchestration onto Celery task queues; the legacy file-backed `preprocess_task` queue and associated enqueue/worker CLIs have been removed in favor of `vibe_photos.task_queue` and `vibe_photos.dev.enqueue_celery`.
- [x] Add a Celery-backed enqueue helper to scan directories and push `pre_process`/`process`/`post_process` jobs to dedicated queues (`vibe_photos.dev.enqueue_celery`).
- [x] Define a stable, versioned on-disk format for preprocessing caches under `cache/` that is decoupled from the database schema. (A cache manifest (`cache/manifest.json`) and per-image JSON sidecars now version embeddings, captions, and region payloads; cache tables are populated during pipeline runs and rely on the manifest for trust.)
- [x] Build a simple Flask-based debug UI to list canonical photos and show per-photo preprocessing details and similar images. (Implemented in `src/vibe_photos/webui/__init__.py` with templates under `src/vibe_photos/webui/templates`.)
- [x] Standardize database access on SQLAlchemy ORM/Core models and prohibit new raw SQL usage in pipeline and web UI.

#### M1 — Extras beyond the original plan

- OWL-ViT region detection with SigLIP-based label re-ranking is fully wired into the pipeline, with results stored in the feature-only `regions`/`region_embedding` tables in the primary database and JSON caches under `cache/regions/`. SigLIP re-ranking now applies confidence and margin thresholds and only refines labels within the same semantic group as the original detector label; if a detector label cannot be grouped confidently, the region keeps only the detector output (no cross-category overrides such as random `AirPods` on food regions).
- A standalone SigLIP+BLIP helper (`SiglipBlipDetector` in `src/vibe_photos/ml/siglip_blip.py`) is available for ad-hoc zero-shot classification + captioning outside the main pipeline.
- The Flask debug UI exposes additional filters such as duplicate-hiding, near-duplicate facets, and region-label filtering on top of the planned scene/attribute filters.
- Region detection now includes: (1) class-agnostic NMS to merge highly overlapping boxes; (2) a configurable priority heuristic combining detector score, normalized box area, and distance to image center; (3) caption-based fallback regions for cases where detection misses an obvious primary object (e.g., drinks in front of background food); and (4) priority-aware filtering of low-value secondary regions so the database and UI focus on a small number of high-quality boxes per image. Priority is not stored in the database but is recomputed in the Web UI using the same heuristic for transparency and tuning.
- Shared preprocessing steps live in `src/vibe_photos/preprocessing.py` and back both Celery workers and the `src/vibe_photos/dev/process.py --image-path` single-image helper for local runs.
- Caption-based primary-region fallback now derives keywords from `models.siglip_labels.label_groups` by default (`models.detection.caption_primary_use_label_groups`), eliminating the duplicate `caption_primary_keywords` block while still allowing overrides when needed.
- pHash recomputation now orders rows and uses skip-locked row locks on PostgreSQL to keep concurrent Celery workers from deadlocking when updating `images`.

#### M1 — Known PoC / placeholder areas

- Full image normalization and storage under `cache/images/processed/` is not yet implemented; only thumbnails are generated in the preprocessing pipeline today.
- EXIF and GPS metadata are parsed during preprocessing and surfaced in the debug UI, but the on-disk metadata format is minimal and may evolve as later milestones add richer EXIF/sidecar handling.
- The preprocessing pipeline is resumable via a JSON run journal in `cache/run_journal.json`; it now skips completed stages and resumes batch cursors. Celery (`vibe_photos.task_queue`) is available for durable `pre_process`/`process`/`post_process` workers, while the single-process loop remains the default local entrypoint.
- Cache validity is gated by the manifest version rather than by a standalone cache database; cache roots are filesystem directories only.
- Cache root resolution now lives in `vibe_photos.cache_helpers.resolve_cache_root` (moved out of `vibe_photos.db_helpers` to drop the DB coupling).
- Caption-aware primary-region fallback in the detection stage assumes that BLIP captions have already been computed and written to `image_caption` for any image that runs detection. Future incremental “detection-only” entry points must either preserve this ordering (captions first) or gracefully disable/adjust caption-based fallbacks to avoid surprising gaps in primary regions.

#### Future technical improvements (beyond M1)

- Optimize pHash-based near-duplicate grouping to reduce worst-case O(N²) behavior (for example via bucketing or approximate nearest-neighbor strategies) once library scale and performance bottlenecks are better understood.
- Introduce a configurable label blacklist / remapping layer for region and image-level labels so low-information or noisy nouns (for example `butter`, `water`) can be suppressed or mapped to coarser categories during SigLIP refinement and caption-based fallbacks. The blacklist should live in configuration (alongside SigLIP label groups) and be applied as a post-processing step, without changing underlying model logits or shared label dictionaries.

### M2 — Perception Quality & Labeling

- [x] Introduce M2 label layer schema + seeds (scene/attr/object) and build SigLIP text prototypes (`uv run python -m vibe_photos.labels.build_object_prototypes`).
- [x] Refactor detection pass to feature-only: write `regions` + `region_embedding` in the primary database, drop in-pass label refinement.
- [x] Add region zero-shot object label pass (CLI) that writes `label_assignment` for regions and aggregated images with configurable score/margin.
- [ ] Improve SigLIP label dictionaries and grouping to reduce manual maintenance of `settings.models.siglip_labels` and cover common creator scenarios (electronics, food, documents, screenshots).
- [ ] Tighten detection thresholds and add label blacklist/remapping to suppress low-information or noisy nouns in region and image-level labels.
- [x] Run targeted evaluations on a labeled subset (≈1k photos) to measure coarse category accuracy, object/product recall, and failure patterns; capture findings in `blueprints/phase_final/knowledge/lessons_learned.md`. (Latest: scene 75.1% acc; attr P/R — has_person 0.94/0.71, has_text 0.73/0.77, is_document 0.91/0.20, is_screenshot 1.00/0.04, has_animal ~0; object top‑1 44% with coverage gaps → thresholds/scene whitelist widened in settings.)
- [x] Add lightweight tools (CLI or notebooks) to inspect per-label distributions and confusion cases, wired against the current PostgreSQL + cache stack. (See `python -m vibe_photos.eval.labels --gt samples/ground_truth.json`.)

### M3 — Search & Tools (PostgreSQL + pgvector + docker-compose)

- [ ] Design PostgreSQL schema and migrations based on the M1/M2 cache tables and Phase Final specs.
- [ ] Implement search and inspection APIs backed by PostgreSQL + pgvector (hybrid text + vector + filters).
- [ ] Define a `docker-compose` stack (API, workers, DB, Redis, UI) suitable for PC/NAS deployment and wire the existing preprocessing pipeline into this stack.
- [x] Add CLI utilities to dump and restore the primary PostgreSQL database (`scripts/dump_primary_db.py`, `scripts/restore_primary_db.py`).

### M4 — Learning & Personalization

- [ ] Implement few-shot learning pipeline using real embeddings.
- [ ] Build annotation UI for corrections and batch actions.
- [ ] Wire feedback into ranking and prototype updates.

Update this file as tasks progress so future coding AIs and human developers can quickly understand the current implementation state.
