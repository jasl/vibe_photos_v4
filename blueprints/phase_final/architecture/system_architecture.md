# Phase Final System Architecture — Local PC / NAS

This document provides a high‑level view of the system architecture for Phase Final, optimized for deployment on personal computers and NAS via `docker-compose`.

## 1. Layered View

1. **Interface Layer**
   - Streamlit‑based web UI for search, browsing, and annotation.
   - FastAPI endpoints powering the UI, CLI, and any automation scripts.
   - Minimal CLI for ingestion control and scripted searches.

2. **Service Layer**
   - **Perception services:**
     - Open‑vocabulary detection (Grounding DINO / OWL‑ViT).
     - SigLIP classification and embeddings, including a small, high‑precision set of coarse categories (e.g. food, electronics, document, screenshot, people, landscape, other).
     - BLIP captioning.
     - Optional OCR (pluggable, off by default).
   - **Search & ranking service:**
     - Combines PostgreSQL FTS and pgvector similarity.
     - Applies ranking logic and result explanation.
   - **Learning & annotation service:**
     - Handles corrections, few‑shot learning, and batch operations.
   - **Task orchestration:**
     - Celery workers (or equivalent) consuming jobs from Redis.

3. **Persistence Layer**
   - PostgreSQL + pgvector as the single source of truth for:
     - Photos, detections, captions, embeddings, annotations, collections, jobs.
   - Redis as task broker and optional cache.
   - Host filesystem / NAS:
     - Original photos (mounted read‑only).
     - Normalized images, thumbnails, and other caches (mounted read/write).

All components run on a single host using `docker-compose`, but the design allows scaling individual containers (e.g. spawning more workers) if hardware resources permit.

## 2. Key Flows

### 2.1 Ingestion & Analysis

1. **Registration**
   - User configures one or more root folders via the UI/CLI.
   - API scans the filesystem, registers new files in PostgreSQL, and enqueues ingestion tasks.

2. **Pre‑processing**
   - Worker loads each image, validates it, and generates:
     - Normalized JPEGs.
     - Thumbnails (e.g. 512×512).
   - Results stored under `cache/` with references in the DB.

3. **Perception**
   - Detection pipeline (Grounding DINO / OWL‑ViT) runs with configured prompts.
   - SigLIP:
     - Re‑ranks detected boxes.
     - Produces image embeddings.
   - BLIP generates captions.

4. **Persistence & Status**
   - Worker writes all perception outputs into PostgreSQL tables and updates:
     - Processing status (`pending`, `processing`, `completed`, `failed`).
     - Review flags (`needs_review`) when confidence thresholds are not met.

5. **Resilience**
   - Jobs are idempotent; restarting the stack or worker should not corrupt state.
   - Failed jobs are logged and retried with backoff; persistent failures are surfaced in the UI.

### 2.2 Search & Browsing

1. **Query Handling**
   - UI/CLI sends search requests to FastAPI with:
     - Text query.
     - Filters (time range, coarse category, collections, favorites, etc.).
   - FastAPI:
     - Embeds the query text using SigLIP.
     - Constructs SQL + pgvector queries.

2. **Retrieval & Ranking**
   - PostgreSQL:
     - Performs FTS on captions, labels, and optional OCR.
     - Performs vector search using pgvector.
   - Service layer fuses scores and returns a ranked list of photo IDs.

3. **Result Enrichment**
   - API fetches:
     - Photo metadata (paths, timestamps).
     - Key labels, captions, and detected objects.
     - Preview URLs or paths for thumbnails.

4. **Presentation**
   - UI shows:
     - Grid of images with core labels and caption snippets.
     - Detail view with bounding boxes and label explanations.

### 2.3 Learning & Annotation

1. **Human Feedback**
   - On each photo, users can:
     - Accept or reject AI labels.
     - Override primary category or product name.
     - Create or assign to collections.

2. **Few‑Shot Learning**
   - Users can initiate a “teach new product” flow:
     - Select example photos.
     - Provide a product name and optional metadata.
   - The system:
     - Computes prototype embeddings.
     - Stores them in a registry used by detection prompts and ranking.

3. **Batch Operations**
   - Recognizer identifies similar photos based on embeddings.
   - Users apply labels or actions to entire groups.

## 3. Supporting Services

- **Monitoring (optional in early phases)**
  - Prometheus scrapes:
    - API latency and error rates.
    - Worker throughput and failure counts.
  - Grafana dashboards visualize:
    - Ingestion backlog.
    - Search performance.
    - Model inference time distributions.

- **Feature Flags**
  - Controlled toggles for:
    - Heavy detection vs. SigLIP‑only fallback.
    - OCR on/off.
    - Experimental ranking strategies.

- **Audit Logging**
  - Logs:
    - All automated label changes (with model versions and scores).
    - All human overrides and batch operations.
  - Enables:
    - Rollbacks for problematic changes.
    - Analysis of where models need improvement.

Use this overview together with the detailed subsystem documents (`vector_db_and_model_versioning.md`, `queue_and_task_management.md`) and the docs in `../docs/` when designing or modifying the system.
