# Queue & Task Management — Local Worker Topology

This document describes how ingestion and analysis tasks are orchestrated on a single PC or NAS using a queue and worker system.

## 1. Selected Stack

- **Task orchestrator:** Celery (or an equivalent Python task queue).
- **Broker & result backend:** Redis.
- **Deployment:** All components run inside the `docker-compose` stack, typically on the same host as PostgreSQL.

### 1.1 Queues

Define dedicated queues for different workloads:

- `high_priority` — UX‑critical tasks:
  - Single‑image ingestion when the user explicitly requests analysis.
  - Quick re‑analysis after manual corrections.
- `default` — Standard ingestion:
  - Bulk folder scans and initial processing for 30k+ photos.
- `learning` — Few‑shot and model update jobs:
  - Prototype recalculation.
  - Evaluation runs.
- `maintenance` — Background maintenance:
  - Reindexing (pgvector).
  - Cleanup (pruning caches, dropping old embeddings).

Each queue can have its own worker concurrency configuration.

## 2. Task Patterns

- **Sequential workflows (chains):**
  - `scan → preprocess → detect → caption → embed → persist` executed as a Celery chain.
  - Guarantees that each step processes outputs from the previous one.

- **Parallel batch processing (groups/chords):**
  - Use Celery groups to:
    - Process independent photos in parallel batches.
    - Aggregate results for reporting or stats computation.

- **Retries & backoff:**
  - Configure retry policies for:
    - Model loading issues (temporary).
    - Transient IO errors.
  - Exponential backoff with a cap to avoid overloading the system.

## 3. Operational Notes for Local Deployments

- **Concurrency limits:**
  - Distinguish between:
    - CPU‑bound tasks (image decoding, EXIF, light transforms).
    - GPU‑bound tasks (detector, SigLIP, BLIP).
  - On GPU machines:
    - Restrict GPU‑heavy tasks to a small number of workers.
  - On CPU‑only machines:
    - Limit overall concurrency to avoid thrashing; allow configuration via environment variables.

- **Locks:**
  - Use Redis‑based locks for:
    - Exclusive jobs (e.g. vector index rebuild).
    - Model upgrade workflows (embedding recomputation).

- **Progress reporting:**
  - Store job status in PostgreSQL:
    - Total processed photos.
    - Pending, in‑progress, completed, and failed counts.
  - UI/CLI polls an API for ingestion and processing progress.

- **Graceful shutdown:**
  - Workers should:
    - Acknowledge tasks only when processing begins.
    - Complete current tasks before termination where possible.
    - Persist partial progress in long‑running jobs.

Document any queue topology changes and operational learnings in `AI_DECISION_RECORD.md` and update monitoring dashboards accordingly.
