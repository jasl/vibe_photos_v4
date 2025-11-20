# Phase Final Technical Choices — Local Deployment Digest

This document records the main technical decisions for the Phase Final architecture focused on personal computers and NAS deployments.

## 1. Guiding Principles

- **Local‑first:** All core functionality must run fully on the user’s own machine or NAS, orchestrated by `docker-compose`.
- **Detection‑first:** Open‑vocabulary detection (Grounding DINO / OWL‑ViT) plus SigLIP/BLIP re‑ranking is a **must‑have**, not a future optimization.
- **Model quality over minimal size:** Target modern Macs and UMA PCs with reasonable RAM/VRAM; larger models are acceptable if they materially improve search quality.
- **CPU compatible, GPU enhanced:** The stack must run on CPU‑only machines, but should automatically take advantage of GPUs if available.
- **Staged rollout:** Implementation proceeds in stages (SQLite prototype → PostgreSQL/pgvector), while keeping the final stack stable.

## 2. Runtime Stack

| Layer       | Choice                            | Notes |
|------------|------------------------------------|-------|
| Orchestration | `docker-compose`               | Single‑host stack for PC/NAS deployments. |
| Web UI     | Streamlit (or similar)            | Simple, Python‑native UI for search and annotation. |
| API        | FastAPI                            | Async, typed, OpenAPI; also used by the UI and CLI. |
| Task Queue | Celery + Redis (or compatible)     | Handles ingestion/analysis tasks and scheduled jobs. |
| Database   | PostgreSQL 16 + pgvector           | Unified store for metadata + vector embeddings. |
| Cache      | Redis                              | Broker + cache for hot data. |
| Storage    | Local filesystem / NAS             | Original photos, normalized copies, thumbnails. |
| Monitoring | Prometheus + Grafana (optional)    | Metrics and dashboards; optional in early MVP. |

Kubernetes is **explicitly out of scope** for Phase Final; if needed later, manifests will be treated as a separate project.

## 3. Model Strategy

### 3.1 Core Models

- **Open‑vocabulary detection (必选):**
  - Grounding DINO or OWL‑ViT for object detection with text prompts.
  - Deployed with configurations for:
    - Base prompt sets (electronics, food, documents, screenshots, people, landscape).
    - Dynamic prompts from few‑shot products and user history.

- **Image classification / embeddings:**
  - SigLIP (e.g. `google/siglip2-base-patch16-224`) as the default model.
  - Option to offer a larger variant for higher quality on capable hardware.
  - Used to:
    - Re‑rank detector boxes.
    - Generate image embeddings (vector search).
    - Embed text queries for retrieval.

- **Captioning:**
  - BLIP (e.g. `Salesforce/blip-image-captioning-base`) as baseline caption model.
  - Optional BLIP‑large configuration for better captions when GPU resources are available.

- **Coarse categories:**
  - Use SigLIP image embeddings to drive a small, fixed coarse category head (e.g. food, electronics, document, screenshot, people, landscape, other).
  - Prioritize **high precision**; ambiguous photos should be classified as `other` rather than mis‑labeled.
  - Use coarse categories to:
    - Drive primary category selection.
    - Gate heavy detection and OCR work, especially on CPU‑only deployments (e.g. detection primarily for electronics/food/document/screenshot, optional or disabled for landscapes/portraits).

### 3.2 Optional/Pluggable Models

- **OCR:**
  - PaddleOCR is **not required by default**; the system only depends on an abstract OCR interface.
  - When enabled, PaddleOCR (Chinese/English) or a similar engine can be plugged in without schema changes.

- **Few‑shot embeddings:**
  - DINOv2 (or SigLIP embeddings) for prototype‑based few‑shot classifiers.
  - Used to learn niche devices/products via a small set of labeled examples.

## 4. Search & Storage Strategy

- **Hybrid Search:**
  - PostgreSQL full‑text search (FTS) over:
    - BLIP captions.
    - Normalized labels from detection and classification.
    - Optional OCR text.
  - pgvector for similarity search over:
    - Image embeddings (entire photo).
    - Optionally, region embeddings.
  - Final ranking via:
    - Reciprocal Rank Fusion or weighted scoring combining FTS + vector similarity + structured signals.

- **Database Layout:**
  - A normalized schema (see `../specs/database_schema.sql`) with:
    - Separate tables for photos, detections, captions, embeddings, annotations, collections, and jobs.
    - Versioned model metadata on embeddings and detection outputs.

- **File Storage:**
  - Originals kept on host filesystem / NAS; mounted read‑only into containers.
  - Normalized copies, thumbnails, and caches stored under dedicated directories (`cache/`, `data/`), also mounted into containers.
  - Preprocessing caches follow a stable, versioned on‑disk format (e.g. JSON/NPY) that is decoupled from the search database schema so that databases can be rebuilt quickly as the schema evolves across milestones.

## 5. Environments & Configurations

- **Development / Test:**
  - Prefer running services directly on the host using:
    - Python 3.12 managed via `uv` (`uv venv`, `uv sync`, `uv run`).
    - SQLite for early milestones (M1–M2).
  - Use `docker-compose` only when testing near‑final deployment (M3+).
  - GPU usage:
    - Optional; when present, used by detector and models.

- **User Deployment (PC / NAS):**
  - Install Docker / Docker Desktop.
  - Download model weights into a dedicated `models/` directory, pointed to via:
    - `TRANSFORMERS_CACHE` for Hugging Face models.
    - Future OCR caches as needed.
  - Launch stack with a single `docker-compose up -d` command.

## 6. Configuration & Extensibility

- **Configuration:**
  - Centralized YAML or environment‑variable based configuration for:
    - Paths to photo roots.
    - Model choices (base vs large, CPU vs GPU).
    - Batch sizes and concurrency limits.
  - Per‑user overrides stored in the database where appropriate (e.g. preferences about default categories).

- **Extensibility:**
  - Detection pipeline pluggable via:
    - Model registry entries.
    - Detector abstraction allowing future models (e.g. newer OWL‑ViT variants).
  - Search logic configurable:
    - Weights for FTS vs vector similarity.
    - Thresholds for highlighting “confident” vs “needs review” results.

- **UI technology across milestones:**
  - M1 uses a simple Flask‑based debug UI to inspect preprocessing results and similar‑image groups locally.
  - Later milestones converge on FastAPI + Streamlit as the main API/UI stack for end users.

Refer to `decisions/AI_DECISION_RECORD.md` for chronological decision logs and deviations from this document.
