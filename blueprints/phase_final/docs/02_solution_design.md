# Phase Final Solution Design — Local‑First, Detection‑First Architecture

This document translates the requirements into a concrete architecture that runs entirely on a user’s own machine or NAS, orchestrated by `docker-compose`.

## 1. High‑Level Architecture

- **Deployment:** Single `docker-compose` stack, designed for:
  - macOS desktops/laptops.
  - Windows / Linux PCs (UMA + optional Nvidia/AMD GPU).
  - Home servers / NAS that can run Docker.
- **Core components:**
  - **Web UI:** Streamlit (or similar) for search, browsing, and annotation.
  - **API Layer:** FastAPI service exposing ingestion, search, and annotation APIs.
  - **Background Workers:** Celery workers (or an equivalent background task system) for ingestion and batch analysis.
  - **Database:** PostgreSQL + pgvector as the single source of truth for metadata and embeddings.
  - **Cache/Queue:** Redis as task broker and caching layer.
  - **Object Storage:** Local filesystem or NAS share mounted into containers for original photos and thumbnails.

The architecture is **local‑first**: once models are downloaded and the stack is up, all functionality works offline.

## 2. Perception Stack — Detection First, Re‑ranking Second

### 2.1 Open‑Vocabulary Detection (Grounding DINO / OWL‑ViT)

- **Responsibility:** Given an image and a list of text prompts, detect objects and return bounding boxes with scores.
- **Prompts:**
  - Generic: `["phone", "smartphone", "laptop", "computer", "monitor", "tablet", "camera", "food", "pizza", "burger", "document", "screenshot", "people", "person", "keyboard", "headphones", ...]`
  - User‑specific: dynamic prompts derived from few‑shot products and recent labels.
- **Outputs stored:**
  - Bounding boxes with normalized coordinates.
  - Raw text labels from the detector.
  - Detector confidence scores and model metadata.

### 2.2 SigLIP — Classification & Embeddings

- **Responsibilities:**
  - Re‑score detected objects by:
    - Cropping box regions.
    - Computing SigLIP similarity against candidate labels.
  - Produce:
    - Image‑level labels (coarse and fine).
    - Image embeddings for vector search (pgvector).
  - Optional: embeddings for object crops, if needed for fine‑grained retrieval.

### 2.2.1 Coarse Category Strategy

- **Motivation:**
  - Coarse categories (e.g. food, electronics, document, screenshot, people, landscape, other) are a primary way for users to narrow search.
  - They must be **high precision**; ambiguous cases should fall back to `other` rather than being mis‑classified.
- **Whole‑image classification:**
  - Use SigLIP image embeddings to perform a small‑head classification over a fixed coarse category set.
  - Compute scores for all coarse categories and apply a conservative confidence threshold.
  - If no category passes the threshold, assign `other`.
- **Primary category decision:**
  - Use whole‑image coarse category classification as the default `primary_category` for each photo.
  - Allow the detection pipeline to override the primary category only when:
    - A single category is strongly dominant across regions (many boxes, high confidence), and
    - It is consistent with the whole‑image classifier scores.
- **Usage in search & UI:**
  - Expose coarse categories as a first‑class filter in the UI and API.
  - Use them to:
    - Gate heavy detection for CPU‑only environments (e.g. enable detection primarily for electronics, food, document, screenshot).
    - Decide when to attempt OCR (e.g. document/screenshot only, see OCR notes).

### 2.3 BLIP — Captions & Narrative Context

- **Responsibilities:**
  - Generate a short caption per image.
  - Optionally generate alternate captions for:
    - “Product focus” (e.g. “a silver MacBook Pro on a wooden desk”).
    - “Food focus” (e.g. “a close‑up of a slice of pepperoni pizza”).
- **Usage:**
  - Store captions in PostgreSQL for full‑text search.
  - Use captions in ranking, particularly when the query is descriptive rather than named entities.

### 2.4 Pluggable OCR (Optional)

- OCR is not required for the first release but the design includes:
  - An abstract OCR interface used by the ingestion pipeline.
  - A disabled default implementation (no external dependency).
  - The ability to plug in PaddleOCR in a future release without schema changes.

### 2.5 Few‑Shot Personalization

- **Responsibilities:**
  - Accept user‑provided example images for a product or category.
  - Use DINOv2/SigLIP embeddings to build a prototype vector (mean embedding).
  - At query/ingestion time:
    - Compute similarity to prototypes.
    - Promote products above generic categories when confidence is high.

## 3. Core Workflows

### 3.1 Ingestion & Analysis (Fully Asynchronous)

1. **Scan & Register**
   - User points the system at one or more root folders.
   - API/CLI scans the filesystem:
     - Creates “photo” records with basic metadata (path, filename, content hash, perceptual hash, timestamps) following the canonical M1 hashing scheme (`xxhash64-v1` for content, `phash64-v1` for perceptual hash).
     - Schedules tasks into the processing queue.
   - Incremental behavior:
     - Detects new or modified files based on content hashes and timestamps.
     - Uses preprocessing cache metadata (versioned by model/pipeline) to determine whether to reuse or recompute results.

2. **Thumbnail & Normalization**
   - Worker loads original images, generates normalized JPEGs and thumbnails.
   - Stores normalized images and thumbnails under `cache/` or a mounted volume.

3. **Detection & Captioning**
   - Worker runs Grounding DINO / OWL‑ViT detection with base prompt set + user‑specific prompts.
   - Worker runs SigLIP classification and embedding computation.
   - Worker runs BLIP caption generation.

4. **Persistence**
   - All results are written to:
     - Stable, versioned preprocessing caches under `cache/` (JSON/NPY or similar) that are independent of DB schema. These caches store **raw model outputs** (embeddings, detections, captions, OCR blocks), not downstream classifications.
     - The active search database (SQLite in early milestones, PostgreSQL + pgvector in Phase Final) as a projection over those caches. Photo‑level classifications such as coarse categories (`primary_category`) are computed from cached embeddings/detections as part of pipeline logic and persisted only in the database.
   - This separation allows:
     - Fast rebuilds of search DBs when schemas change across milestones.
     - Reuse of expensive model outputs as long as model and pipeline versions remain compatible, while allowing classification/ranking logic to evolve and be recomputed without rerunning heavy models.

5. **Status Tracking**
   - Photo records track processing status (`pending`, `processing`, `completed`, `failed`) and flags (`needs_review`).
   - UI/API can query progress and show ingestion statistics.

### 3.2 Hybrid Search

1. **Query Parsing**
   - FastAPI layer receives natural‑language queries + optional filters (time, category, collections).
   - The query is converted into:
     - A text search term for full‑text search.
     - A dense embedding (via SigLIP) for vector search.

2. **Candidate Retrieval**
   - PostgreSQL + pgvector returns candidates using:
     - Vector similarity on image embeddings.
     - Full‑text search on captions, labels, (optional OCR).
     - Structured filters on categories, time, favorites, collections.

3. **Ranking & Explanation**
   - Combine scores (e.g. Reciprocal Rank Fusion or weighted sum).
   - Surface explanations per result:
     - Matched objects and labels.
     - Matched words in captions/OCR.
     - Relevant time and collection membership.

### 3.3 Annotation Assistance & Learning

1. **Human‑in‑the‑loop feedback**
   - UI presents:
     - AI‑predicted primary label and product name.
     - Alternative suggestions (top‑K labels, similar products).
   - User actions:
     - Accept / reject / override labels.
     - Apply labels to similar photos (batch action).

2. **Learning loop**
   - The system logs:
     - Corrections vs. predictions.
     - Frequently used overrides.
   - Few‑shot component:
     - Updates prototype vectors based on confirmed labels.
     - Feeds prototypes back into detection prompts and ranking.

## 4. Component Responsibilities

- **Web UI**
  - Search bar + filters.
  - Results grid with infinite scroll.
  - Detail panel showing detections, captions, labels, and EXIF/time.
  - Annotation tools and collection management.

- **FastAPI Service**
  - Exposes REST endpoints for:
    - Ingestion (add paths, check status).
    - Search (text + filters + pagination).
    - Annotations (apply labels, batch operations).
  - Auth is optional/minimal (local single‑user assumptions).

- **Background Workers**
  - Execute the ingestion and analysis pipeline.
  - Handle retries with exponential backoff for transient failures.
  - Support scheduling long‑running operations (re‑embedding, migrations).

- **PostgreSQL + pgvector**
  - Store all persistent entities:
    - Photos, detections, captions, OCR blocks, embeddings.
    - Collections, search history, annotations, jobs.
  - Provide hybrid SQL + vector search with indexes tuned for local scale (≤100k photos).

- **Redis**
  - Task queue broker for background jobs.
  - Cache for hot search results and UI state where beneficial.

- **Filesystem / NAS**
  - Store originals under user‑controlled paths.
  - Mount volumes into containers for:
    - Read‑only access to originals.
    - Read/write access for caches and logs.

## 5. Data Model Highlights

While details live in `../specs/database_schema.sql`, the conceptual model includes:

- `photos` — core photo records with filesystem metadata, processing status, and primary AI labels.
- `photo_regions` — object‑level detections with bounding boxes, detector labels, and re‑ranked labels.
- `photo_embeddings` — image and (optionally) region embeddings with model/version metadata.
- `captions` — BLIP (and future) captions associated with photos.
- `annotations` — human labels, corrections, and training flags.
- `collections` / `collection_photos` — saved searches and hand‑curated groups.
- `jobs` / `processing_queue` — background pipeline state and telemetry.

## 6. Evolution Path

To reduce risk, implementation can proceed in stages while keeping the Phase Final design as the target:

1. **M1 — Preprocessing & Feature Extraction** — Single‑machine, SQLite‑backed pipeline:
   - High‑throughput preprocessing (normalization, thumbnails, EXIF/GPS, hashes, perceptual hashes).
   - Initial perception models (SigLIP/BLIP and, where feasible, detection).
   - Versioned caches under `cache/` and a lightweight SQLite DB.
2. **M2 — Search & Tools** — Local services on top of SQLite:
   - FastAPI endpoints for search and inspection.
   - CLI commands and a minimal UI for browsing and debugging.
3. **M3 — Database & Deployment Upgrade** — PostgreSQL + pgvector with `docker-compose`:
   - Migrate from SQLite to PostgreSQL/pgvector.
   - Containerized API, workers, DB, Redis, and UI.
4. **M4 — Learning & Personalization** — Few‑shot and corrections loop:
   - Teach the system about niche products via few‑shot examples.
   - Leverage feedback to improve ranking and suggestions.

Use this design together with `01_requirements.md`, `03_technical_choices.md`, and the architecture notes when implementing Phase Final.
