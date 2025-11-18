# M1 Preprocessing & Lightweight Classification Blueprint

Status: draft (implements reviewed guidance for the first milestone)

## 1. Background and Goals
- Local library: ~30,000 photos spanning landscapes, portraits, electronics, screenshots, documents, and food.
- Long-term goal: enable fast, on-device object-centric search (e.g., `iPhone`, `pizza`, `MacBook Pro`, `document`, `screenshot`).
- M1 goal: ship stable preprocessing—**scanning + hashing + lightweight scene classification + QA Web UI**—as the foundation for later tagging, embeddings, OCR, and search.

## 2. Scope
### In Scope (M1)
1. **File scanning & content hashing**
   - Recursively scan one or more configured roots.
   - Filter non-image files.
   - Compute a content hash as the stable `image_id` and detect new/modified/deleted files.

2. **Lightweight pre-classification**
   - Produce per-image attributes such as:
     - `scene_type`: `PORTRAIT / GROUP / FOOD / ELECTRONICS / DOCUMENT / SCREENSHOT / LANDSCAPE / OTHER`
     - `has_text`, `has_person`, `is_screenshot`, `is_document`
   - Models load once; support batch inference.

3. **Flask Web UI for QA**
   - Filter images by the above attributes.
   - Paginated grid with thumbnails and metadata.
   - Detail view shows full-size image and pre-classification output.

### Out of Scope (M1)
- Object-level tagging (`pizza`, `iPhone`, `MacBook Pro`), embeddings, similarity search.
- OCR and model-based text parsing.
- Complex UI, multi-user workflows, model training, or ensembles.

## 3. Architecture Overview
```
[Album roots]
   |  (scan)
   v
[Hasher]  ---> [images table]
   |  (classify)
   v
[Scene classifier] ---> [image_scene table]
   |  (inspect)
   v
[Flask Web UI]
```

Run modes:
- CLI preprocessing entry point (scan + classify; configurable batch size/device).
- Flask app for manual inspection during development.

## 4. Data Model (SQLite)
### `images` table
Tracks stable identity and file metadata.

```sql
CREATE TABLE IF NOT EXISTS images (
  image_id        TEXT PRIMARY KEY,
  primary_path    TEXT NOT NULL,
  all_paths       TEXT NOT NULL,  -- JSON array
  size_bytes      INTEGER NOT NULL,
  mtime           REAL NOT NULL,
  width           INTEGER,
  height          INTEGER,
  exif_datetime   TEXT,
  camera_model    TEXT,
  hash_algo       TEXT NOT NULL,
  created_at      REAL NOT NULL,
  updated_at      REAL NOT NULL,
  status          TEXT NOT NULL,  -- active / deleted / error
  error_message   TEXT,
  schema_version  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_primary_path ON images(primary_path);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
```

### `image_scene` table
Stores lightweight pre-classification outputs.

```sql
CREATE TABLE IF NOT EXISTS image_scene (
  image_id           TEXT PRIMARY KEY,
  scene_type         TEXT NOT NULL,
  scene_confidence   REAL,
  has_text           INTEGER NOT NULL,
  has_person         INTEGER NOT NULL,
  is_screenshot      INTEGER NOT NULL,
  is_document        INTEGER NOT NULL,
  classifier_name    TEXT NOT NULL,
  classifier_version TEXT NOT NULL,
  updated_at         REAL NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_scene_scene_type ON image_scene(scene_type);
CREATE INDEX IF NOT EXISTS idx_image_scene_has_text ON image_scene(has_text);
CREATE INDEX IF NOT EXISTS idx_image_scene_has_person ON image_scene(has_person);
CREATE INDEX IF NOT EXISTS idx_image_scene_is_screenshot ON image_scene(is_screenshot);
CREATE INDEX IF NOT EXISTS idx_image_scene_is_document ON image_scene(is_document);
```

## 5. Pipeline Design
### 5.1 Scanner
- Recursively walk configured roots with an image extension allowlist (`.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.gif`; configurable).
- Emit `FileInfo` objects: `path`, `size_bytes`, `mtime`.

### 5.2 Hasher
- Stream files to compute `image_id` (`sha1` or `xxhash`).
- Upsert into `images`:
  - Insert new rows for unseen hashes.
  - Merge `all_paths` for duplicates; refresh metadata for changed files.
- Mark missing paths as `deleted` unless other paths still reference the hash.

### 5.3 Lightweight Scene Classification
- Batch process active images missing `image_scene` rows or outdated `classifier_version`.
- Suggested approach:
  - CLIP zero-shot prompts for scene types and text-heavy cues.
  - Simple face/person detection or CLIP prompts for `has_person`.
  - CLIP or heuristic prompts for `has_text`, `is_screenshot`, `is_document`.
- **Model initialization happens once per process**; no per-image reloads.

### 5.4 Error Handling
- Corrupt/unreadable images do not halt the run; record `status="error"` and `error_message`.
- Re-running the pipeline skips unchanged images and only processes new/modified assets.

### 5.5 Incremental & Resumable Execution
- Use `image_id` as the stable unit of work so **scan, hash, and classify are idempotent**.
- Track per-stage versions/timestamps (e.g., `classifier_version`) to identify stale rows without reprocessing everything.
- Maintain a lightweight run journal (batch cursor + stage name + checkpoint time) so an interrupted run can **resume from the last completed batch** instead of restarting the full dataset.
- Prefer deterministic batching (e.g., ORDER BY `image_id`) to make retries repeatable and diff-friendly in logs.

## 6. Flask Web UI
Routes:
1. `GET /` → redirect to `/images`.
2. `GET /images`: list with filters (`scene_type`, `has_text`, `has_person`, `is_screenshot`, `is_document`, `page`, `page_size`).
3. `GET /image/<image_id>`: detail page with image preview and metadata.
4. (Optional) `GET /thumbnail/<image_id>`: serve or generate thumbnails; for M1, browser scaling or static routing is acceptable.

UI expectations:
- Filters at the top of the list page; grid layout (4–6 per row depending on CSS).
- Each card shows thumbnail + `scene_type` + boolean flags.
- Detail page shows primary path, size, mtime, EXIF when available, and classifier metadata.

## 7. Suggested Package Layout
```
project_root/
  app/
    config.py
    db.py
    scanner.py
    hasher.py
    classifier.py
    pipeline.py
    webui/
      views.py
      templates/
        base.html
        images.html
        image_detail.html
      static/
        css/
        js/
  scripts/
    preprocess.py
  data/
    index.db
  docs/
    m1_spec.md (this file)
```

## 8. Acceptance Criteria
### Functional
- `uv run python -m scripts.preprocess --root <album> --db data/index.db --batch-size 16 --device cpu|cuda`:
  - Populates `images` with active assets and hashes.
  - Populates `image_scene` for active images; records errors without stopping.
- Re-running skips unchanged images; updates new/modified assets; flags missing paths appropriately.
- Interrupted runs can resume from recorded checkpoints/batch cursors without reprocessing completed work.
- Flask UI (`FLASK_APP=app.webui uv run flask run`) loads list and detail pages with working filters and pagination.

### Performance & Stability
- Models load once; batches reuse loaded weights.
- Pipeline continues after single-image failures.
- Reasonable throughput for ~30k images (exact numbers to be captured after first full run).

### Evolvability Hooks for M2+
- Use `schema_version`, `classifier_name`, and `classifier_version` to detect stale rows and re-run when models change.
- Scene and `has_text` outputs gate future OCR runs; cache layer can later store embeddings, logits, detections, and OCR text keyed by `image_id`.

## 9. Search-Relevant Expectations (Preview for Future Phases)
- Each photo should eventually emit 1–3 high-quality coarse + mid-level labels to support text and embedding search.
- Electronics/food/document classifiers should be prioritized for accuracy.
- Offline batch processing with local indexes is sufficient for the target dataset scale; no cloud GPU dependency is required.
