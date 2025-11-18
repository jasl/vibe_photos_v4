# AI Task Tracker — Vibe Photos Phase Final

This file tracks high-level implementation tasks and their status for the Phase Final architecture.

## Milestones

- [ ] M1 — Preprocessing & Feature Extraction (Local, No Containers)
- [ ] M2 — Search & Tools (SQLite + Local Services)
- [ ] M3 — Database & Deployment Upgrade (PostgreSQL + pgvector + docker-compose)
- [ ] M4 — Learning & Personalization

## Current Tasks (Outline)

### M1 — Preprocessing & Feature Extraction

- [ ] Set up `uv` environment and base dependencies.
- [ ] Implement shared logging and configuration modules.
- [ ] Define initial SQLite schema for photos, metadata, and model outputs.
- [ ] Implement photo scanning and registration for local folders/NAS mounts.
- [ ] Normalize images and generate thumbnails / web-friendly versions.
- [ ] Extract EXIF, capture time, GPS (when present), and file timestamps.
- [ ] Compute file hashes and perceptual hashes; record near-duplicate relationships.
- [ ] Integrate SigLIP embeddings and BLIP captions; cache results.
- [ ] (Optional) Integrate Grounding DINO / OWL-ViT detection and SigLIP re-ranking.
- [ ] Implement a resumable, concurrent preprocessing pipeline (queue + workers).
- [ ] Define a stable, versioned on-disk format for preprocessing caches under `cache/` that is decoupled from the database schema.
- [ ] Build a simple Flask-based debug UI to list canonical photos and show per-photo preprocessing details and similar images.

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
