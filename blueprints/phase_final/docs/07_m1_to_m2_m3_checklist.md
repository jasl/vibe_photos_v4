# M1 → M2/M3 Transition Checklist (Draft)

This draft checklist summarizes the main engineering steps to move from the implemented M1 preprocessing pipeline to M2 (Perception Quality & Labeling) and M3 (Search & Tools on PostgreSQL + pgvector + docker-compose), from the Phase Final point of view.

> Operational note: PostgreSQL + pgvector is the only primary database target in current code. Any mentions of file-backed databases (for example `data/index.db`) are historical and should not be treated as deployable options; the filesystem `cache/` remains the rebuildable artifact source.

## 1. Pre-M2 Readiness (Grounded on M1)

- [ ] Confirm M1 data quality:
  - [ ] Run the M1 pipeline on a representative subset (≥10k photos) using `vibe_photos.dev.process` and verify cache/journal behavior.
  - [ ] Manually inspect a sample in the Flask debug UI (`vibe_photos.webui`) for:
    - [ ] EXIF/time/GPS correctness.
    - [ ] Coarse categories (scene attributes) being reasonable.
    - [ ] Captions and near-duplicate groups roughly matching human expectations.
- [ ] Lock down cache format:
  - [ ] Ensure `cache/manifest.json` records the final `cache_format_version`, model identifiers, and key pipeline settings.
  - [ ] Document how to treat old cache directories when the manifest version changes.
- [ ] Validate model plumbing:
  - [ ] Confirm `docs/MODELS.md` matches `config/settings.yaml` and `src/vibe_photos/ml/model_presets.py`.
  - [ ] Verify `vibe_photos.dev.download_models` can pre-download all configured checkpoints for offline use.

## 2. M2 — Perception Quality & Labeling

Goal: Improve recognition quality and label hygiene on top of the existing PostgreSQL + cache stack, before investing in full search tooling.

- [ ] SigLIP label dictionaries and grouping:
  - [ ] Review and refine `settings.models.siglip_labels` to:
    - [ ] Reduce manual maintenance cost for common creator scenarios (electronics, food, documents, screenshots, products).
    - [ ] Improve grouping and coverage for both coarse categories and product‑level hints.
  - [ ] Design a label blacklist/remapping mechanism (in config or code) to suppress low‑information or noisy labels and map them to more useful categories.
- [ ] Detection + SigLIP/BLIP integration:
  - [ ] Tune OWL‑ViT + SigLIP thresholds (`siglip_min_prob`, `siglip_min_margin`, region priority heuristics) using real‑world samples.
  - [ ] Validate caption‑aware primary region selection behavior and adjust configuration so that “main objects” are consistently captured.
- [ ] Evaluation and reporting:
  - [ ] Build a small labeled evaluation set (≈1k photos) with:
    - [ ] Coarse category labels.
    - [ ] Key object/product annotations for a handful of target classes.
  - [ ] Add scripts/CLIs to:
    - [ ] Dump per‑image label/caption summaries for review.
    - [ ] Compute simple metrics (precision/recall or confusion matrices) over this set.
  - [ ] Summarize findings and recommended defaults in `docs/MODELS.md` and `blueprints/phase_final/knowledge/lessons_learned.md`.

## 3. M3 — Search & Tools (PostgreSQL + pgvector + docker-compose)

Goal: Promote the architecture to its Phase Final form with PostgreSQL + pgvector, a single docker-compose stack, and production search/tools built on top of the improved perception outputs.

- [ ] Design PostgreSQL schema:
  - [ ] Translate the canonical schema (`blueprints/phase_final/specs/database_schema.sql`) to PostgreSQL, including pgvector columns for embeddings.
- [ ] Ensure tables map cleanly from the current cache tables (M1/M2) to PostgreSQL entities.
  - [ ] Decide where to store:
    - [ ] Photo records and regions.
    - [ ] Embeddings and captions.
    - [ ] User-facing annotations and collections (even if initially empty).
- [ ] Wire pgvector search:
  - [ ] Create pgvector indexes for image embeddings (and optional region embeddings).
  - [ ] Mirror M2 search semantics:
    - [ ] Hybrid scoring using pgvector + PostgreSQL full-text search.
    - [ ] Coarse category and object filters.
- [ ] Containerize the stack:
  - [ ] Write a `docker-compose` file that defines:
    - [ ] PostgreSQL + pgvector.
    - [ ] Redis.
    - [ ] API service.
    - [ ] Background worker service (Celery).
    - [ ] UI service (Streamlit).
  - [ ] Ensure volumes mount:
    - [ ] Original photo libraries.
    - [ ] `models/`, `cache/`, and `data/`.
  - [ ] Provide `.env.example` and short setup docs for local and NAS-style environments.
- [ ] Queue & worker alignment:
  - [ ] Align the existing Celery tasks (`vibe_photos.task_queue`) with PostgreSQL-backed state.
  - [ ] Decide which tasks remain legacy-backed (if any) vs. fully migrated to PostgreSQL.
- [ ] Operational readiness:
  - [ ] Add basic health checks for API, workers, and DB.
  - [ ] Define backup/restore procedures for PostgreSQL (docs + scripts).
  - [ ] Capture a “first-run” checklist for end users (mounting paths, downloading models, starting the stack).

## 4. Open Questions for Future Discussion

This draft intentionally leaves a few decisions open for later design reviews:

- How much of the M2 API should be preserved unchanged in M3 vs. evolved with PostgreSQL features?
- Whether to prefer “rebuild from cache + manifest” or “migrate DB rows” as the primary upgrade path for large libraries.
- How aggressively to use region embeddings (vs. image-level only) in early M3 search.

Update this checklist as design decisions are recorded in `decisions/AI_DECISION_RECORD.md` and as M2/M3 implementation progresses.
