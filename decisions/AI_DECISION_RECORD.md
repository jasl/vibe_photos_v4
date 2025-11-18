# AI Decision Record — Vibe Photos Phase Final

This file tracks major product and technical decisions so future coding AIs and human developers can understand why the system looks the way it does.

## ADR-001 — Local PC / NAS Deployment via docker-compose

- **Context:** Primary users are self-media creators managing large local photo libraries (30k–100k photos). They prefer keeping data on their own machines or home servers.
- **Decision:** Phase Final targets single-host deployments on personal computers and NAS, orchestrated by `docker-compose`. Kubernetes and multi-cluster deployments are out of scope.
- **Consequences:**
  - Simpler installation and updates for end users.
  - Architecture must assume limited horizontal scaling; focus on vertical optimization.

## ADR-002 — PostgreSQL + pgvector as Primary DB and Vector Store

- **Context:** The system needs hybrid search over metadata, captions, labels, and embeddings at local scale.
- **Decision:** Use PostgreSQL + pgvector as the single system of record for both relational data and vector embeddings.
- **Consequences:**
  - Simplified operations, backups, and restores for users.
  - All search and analytics logic is expressed in SQL + pgvector functions.
  - Future external vector services (Faiss, Qdrant, etc.) remain optional and out of scope for Phase Final.

## ADR-003 — Detection-First Perception with Grounding DINO / OWL-ViT + SigLIP/BLIP

- **Context:** Manual tagging is infeasible for 30k+ photos. Existing album apps handle faces well but struggle with objects/products/food/documents.
- **Decision:** The perception stack must use an open-vocabulary detection pipeline (Grounding DINO or OWL-ViT) with SigLIP re-ranking and BLIP captions as core components.
- **Consequences:**
  - Better coverage of “物” (objects/products/food/documents) without manual labels.
  - Higher compute cost, mitigated by asynchronous processing and optional GPU use.

## ADR-004 — OCR as Pluggable, Optional Component

- **Context:** Initial experiments showed limited benefit from OCR relative to cost, but future document/screenshot scenarios may need it.
- **Decision:** Design the system with an OCR abstraction, but do not require PaddleOCR or any specific OCR engine for the first release.
- **Consequences:**
  - Simplifies initial installation and model downloads.
  - Allows plugging in PaddleOCR or other OCR models later without schema changes.

## ADR-005 — Staged Rollout: Preprocessing → Search → PostgreSQL/pgvector → Learning

- **Context:** Previous attempts tried to build the full architecture at once and became difficult to debug.
- **Decision:** Implement Phase Final in stages:
  1. Preprocessing and feature extraction with SQLite (including SigLIP/BLIP and, where feasible, detection).
  2. Search and tools built on top of the preprocessed SQLite data (CLI, API, minimal UI).
  3. Database and deployment upgrade to PostgreSQL + pgvector and docker-compose.
  4. Learning and few-shot personalization.
- **Consequences:**
  - Each stage is individually testable and shippable.
  - Feedback can shape later milestones without rewriting everything.

Add new ADR entries here as decisions are made or revised.

## ADR-006 — Stable Preprocessing Cache Separate from Database Schema

- **Context:** Early milestones (M1–M2) will iterate quickly on database schemas while the cost of recomputing model outputs over tens of thousands of photos is high.
- **Decision:** Treat preprocessing outputs (normalized images, hashes, EXIF/GPS, model outputs) as a stable, versioned cache stored under `cache/` in a format decoupled from any specific database schema. Databases (SQLite, then PostgreSQL + pgvector) are projections over this cache and can be rebuilt when schemas change.
- **Consequences:**
  - Expensive model inference can be reused across milestones as long as model and pipeline versions remain compatible.
  - Schema migrations become simpler: drop and rebuild DBs from caches when appropriate.
  - Incremental processing logic can rely on cache metadata (hashes, timestamps, versions) to determine which photos need work.
