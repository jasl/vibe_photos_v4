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
  1. Preprocessing and feature extraction with the local pipeline (including SigLIP/BLIP and, where feasible, detection).
  2. Search and tools built on top of the preprocessed data (CLI, API, minimal UI).
  3. Database and deployment upgrade to PostgreSQL + pgvector and docker-compose.
  4. Learning and few-shot personalization.
- **Consequences:**
  - Each stage is individually testable and shippable.
  - Feedback can shape later milestones without rewriting everything.

Add new ADR entries here as decisions are made or revised.

## ADR-006 — Stable Preprocessing Cache Separate from Database Schema

- **Context:** Early milestones (M1–M2) will iterate quickly on database schemas while the cost of recomputing model outputs over tens of thousands of photos is high.
- **Decision:** Treat preprocessing outputs (normalized images, hashes, EXIF/GPS, model outputs) as a stable, versioned cache stored under `cache/` in a format decoupled from any specific database schema. Databases (PostgreSQL + pgvector) act as cache layers over this data and can be rebuilt when schemas change.
- **Consequences:**
  - Expensive model inference can be reused across milestones as long as model and pipeline versions remain compatible.
  - Schema migrations become simpler: drop and rebuild DBs from caches when appropriate.
  - Incremental processing logic can rely on cache metadata (hashes, timestamps, versions) to determine which photos need work.

## ADR-007 — Updated Milestones: M2 as Perception Quality, M3 as Search & Tools on PostgreSQL/pgvector

- **Context:** Early Phase Final documents described M2 as “Search & Tools on the legacy stack” and M3 as “Database & Deployment Upgrade”. In practice, M1 has already delivered a substantial preprocessing pipeline plus a basic debug UI, while real‑world usage surfaced that recognition quality (labels, detection, SigLIP prompts) is the main bottleneck before building full search tooling.
- **Decision:** Reframe the milestones as:
  1. M1 — Preprocessing & Feature Extraction (implemented in this repository).
  2. M2 — Perception Quality & Labeling: focus on improving SigLIP label dictionaries/grouping, detection + SigLIP/BLIP integration, and evaluation tooling on top of the existing PostgreSQL + cache stack.
  3. M3 — Search & Tools (PostgreSQL + pgvector + docker-compose): implement production search/tools on top of PostgreSQL + pgvector and containerize the stack for PC/NAS deployment.
  4. M4 — Learning & Personalization: keep few‑shot learning and feedback loops as the final stage.
- **Consequences:**
- M2 work is centered on improving recognition quality (reducing manual `siglip_labels` maintenance and noisy object labels) while reusing the existing M1 pipeline and cache tables.
  - Search APIs, UIs, and PostgreSQL/pgvector migration are concentrated in M3, aligning infrastructure work with a clear, user‑visible “Search & Tools” milestone.
  - Documentation and task trackers (AI_TASK_TRACKER, Phase Final docs) treat the new milestone definitions as the source of truth going forward.
