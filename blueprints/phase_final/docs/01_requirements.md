# Phase Final Requirements — Local Photo Intelligence for Creators

This document captures the product requirements for the “Phase Final” generation of Vibe Photos, aligned with the new constraint that everything runs on a user’s own machine or NAS via `docker-compose`.

## 1. Users & Context

- Primary audience: Chinese self-media creators managing **30k–100k local photos** for reviews, tutorials, and lifestyle content.
- Storage: local disks, external drives, or **home/NAS servers**; users are privacy‑sensitive and do not want to push full‑resolution photos to the cloud.
- Typical workflows:
  - Preparing a video / article about a specific **device**（e.g. iPhone 15、MacBook Pro、机械键盘）and needing all relevant shots.
  - Writing content about **food / travel / documents**, e.g. “世界各地的披萨做法”, “某款显示器的拆解过程”.
  - Digging out **old screenshots or scanned documents** (发票、保修卡、快递单、PPT、聊天记录截图) as evidence.
- Pain points:
  - Hard to retrieve “物” (objects/products/food/documents) based only on memory.
  - Existing album apps do people/faces well, but object and product search is weak or opaque.
  - Manual tagging is impossible at 30k+ photos; any workflow that relies on human labeling is unacceptable.

## 2. Product Vision

Vibe Photos should act as a **local, privacy‑preserving photo search engine** focused on “物”：

- Users describe what they remember (e.g. “那次在东京吃的厚底披萨”, “老款 MacBook Pro 撕开底壳的照片”, “带发票的 iPhone 二手交易图”).
- The system returns:
  - The **most relevant photos** containing the requested object/product/food/document.
  - Clear hints about **why** each photo was matched (detected objects, captions, time, similar past results).
- Everything runs on the user’s own machine or NAS, orchestrated by `docker-compose`; no mandatory cloud component.

## 3. Functional Requirements

### 3.1 Perception — Understand Objects, Products, and Scenes

1. **Open‑vocabulary object detection (必选)**  
   - Use a **Grounding DINO / OWL‑ViT** style pipeline to detect objects with free‑form text prompts.  
   - Support detecting both generic classes (“电脑”, “手机”, “文件”, “食物”) and **specific products** (“iPhone 15 Pro”, “MacBook Pro 14”, “HHKB Pro2”).
   - Save both **bounding boxes** and **normalized text labels** into the database.

2. **Image‑level labels and captions**  
   - Use **SigLIP** for zero‑shot classification and image embeddings.  
   - Use **BLIP** for natural‑language captions.  
   - Combine detection and captions to produce:
     - Coarse categories: electronics / food / document / screenshot / people / landscape …
     - Fine‑grained object names and product hints.

3. **Re‑ranking and consistency**  
   - Re‑rank detected objects with SigLIP scores to reduce hallucinations and noise.  
   - Ensure that the final photo record contains:
     - Top‑N image‑level labels.
     - Top‑N object‑level labels (with regions).
     - At most one **primary category** and a short caption.

4. **OCR (可选插件，不是必选依赖)**  
   - The system must be designed so that OCR is **pluggable** via a clear interface.  
   - Default build does **not** require PaddleOCR, but the schema and codepaths allow:
     - Dropping in PaddleOCR or other engines later.
     - Enabling/disabling OCR per deployment.

5. **Few‑shot personalization (加分项)**  
   - Allow users to **teach** the system about rare devices or niche products (e.g. 某款示波器、某个定制键盘).
   - Use DINOv2/SigLIP embeddings + prototype vectors for **few‑shot detection and re‑ranking**.
   - Prioritize product‑level recognition that improves with more feedback over time.

### 3.2 Search & Discovery

1. **Natural‑language search**  
   - Users can search with free text (Chinese or bilingual):
     - “木质桌面上的老款银色 MacBook Pro”
     - “披萨和啤酒的同框特写”
     - “带机身序列号和发票的 iPhone 二手交易照片”
   - System combines:
     - Full‑text search over captions/labels/optional OCR.
     - Vector similarity search (SigLIP embeddings) via pgvector.
     - Structured filters (time, folder/source, favorite flag).

2. **Hybrid filters & faceting**
   - Filter by:
     - Time ranges (by taken/imported date).
     - Coarse categories (electronics / food / document / screenshot / people / landscape …).
     - “Contains specific object/product” (e.g. has “pizza” object, has “iPhone 15 Pro”).
   - Show facets to help users narrow down quickly (e.g. top brands, categories, years).

3. **Similarity & grouping**
   - Given a photo, show **similar photos** (same scene, same product, same dish).
   - Support **group view** for:
     - Similar electronics sets (某一代 MacBook/iPhone 的全套照片).
     - Similar food shots (同款披萨/拉面).
     - Deduplication of near‑identical images.

4. **Export & integration**
   - Allow users to:
     - Bulk export selected photos’ paths into a text/JSON list for downstream tools.
     - Save search results as “collections” for later reuse.

### 3.3 Operations & Workflow

1. **Bulk ingestion & incremental updates**
   - Users can point the system to one or more root folders (本机相册、NAS 共享目录).  
   - The system:
     - Scans and registers new photos.
     - Schedules them for **asynchronous** analysis.
     - Tracks progress and status per photo.
   - Incremental processing:
     - Detects new or modified files via hashes and timestamps.
     - Reuses cached preprocessing results when models and pipeline versions have not changed.
     - Supports fast rebuilds of search databases from preprocessing caches when schemas evolve across milestones.

2. **Fully asynchronous processing (非实时)**
   - No hard latency requirement for individual images.  
   - The ingestion pipeline must be able to:
     - Process large batches (30k+ photos) in the background.
     - Resume gracefully after restarts.
     - Throttle or pause on resource constraints (CPU/GPU/IO).

3. **Dedupe and quality control**
   - Detect and mark:
     - Exact duplicates (same file hash or perceptual hash).
     - Near duplicates (cropped, resized, slightly edited).
   - Allow users to bulk hide or group duplicates.
   - Preprocessing must:
     - Compute robust fingerprints (e.g. perceptual hashes) for every image.
     - Record similarity relationships so near‑duplicate groups are inspectable.
     - Skip redundant heavy model inference for near‑identical images where safe.

4. **Human‑in‑the‑loop learning**
   - Capture user corrections and confirmations:
     - Accepting/rejecting predicted labels.
     - Overriding primary category or product name.
   - Use these signals to:
     - Re‑rank similar future photos.
     - Improve few‑shot prototypes.

### 3.4 Interfaces

1. **Web UI (Streamlit or similar)**
   - Run inside the `docker-compose` stack.
   - Provide:
     - Search bar with filters.
     - Grid view with key labels and captions.
     - Detail panel showing detected objects, labels, and metadata.
     - Simple annotation tools (approve/override labels, mark favorites, batch select).

2. **API (FastAPI)**
   - Provide programmatic access for scripting / automation:
     - Enqueue ingestion for a path.
     - Query photos by text/filters.
     - Fetch detection/annotation details.

3. **CLI**
   - Minimal commands:
     - `scan` / `ingest` / `status`.
     - `search` with textual queries and filters.

## 4. Quality & Performance Targets

- Accuracy:
  - ≥85% for common coarse categories（electronics/food/document/screenshot/people/landscape）.
  - ≥60% recall for niche products when the user has given at least a few labeled examples (few‑shot), with human review in the loop.
- Throughput:
  - Ingestion pipeline processes **≥5–10 images/second** on a typical modern desktop (multi‑core CPU, optional GPU), averaged over long runs.
- Latency:
  - Search latency **≤500 ms** for 50k–100k assets on local PostgreSQL + pgvector, assuming embeddings are precomputed.
- UX:
  - From starting a search to selecting usable photos **≤3 minutes** in common cases, saving at least **30 minutes/day** compared to manual browsing.

## 5. Constraints & Non‑Goals

1. **Deployment & Infrastructure**
   - Must run via **`docker-compose`** on:
     - macOS desktops/laptops.
     - Windows / Linux PCs (with UMA + Nvidia/AMD GPUs optional).
     - Home NAS or small home servers capable of running Docker.
   - **Kubernetes is out of scope** for Phase Final; keep manifests only as optional future work if ever needed.

2. **Hardware**
   - Optimized for modern CPUs and optional GPUs with **no strict VRAM limit** (UMA/Nvidia/AMD); we can choose moderately large models where quality benefits justify the cost.
   - System must still function in **CPU‑only** mode, albeit slower.

3. **Privacy & Connectivity**
   - No requirement for public cloud; all core features must work offline after initial model downloads.
   - Optional integration with external services (e.g. remote backups) must be strictly opt‑in.

4. **Out‑of‑scope features**
   - Social sharing, commenting, or complex collaboration features.
   - Heavy image editing or batch transformation (beyond light thumbnailing and normalization).
   - Full, unsupervised auto‑labeling with zero human review; the system is intentionally **assistive**.

Use this requirements document together with `02_solution_design.md` and architecture notes to guide design and implementation.
