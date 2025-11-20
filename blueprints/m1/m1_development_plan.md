# M1 Preprocessing & Lightweight Classification Blueprint

Status: ready for implementation (implements reviewed guidance for the first milestone)

## 1. Background and Goals
- Local library: ~30,000 photos spanning landscapes, portraits, electronics, screenshots, documents, and food.
- Long-term goal: enable fast, on-device object-centric search (e.g., `iPhone`, `pizza`, `MacBook Pro`, `document`, `screenshot`).
- M1 goal: ship stable preprocessing—**scanning + hashing + lightweight scene classification + SigLIP embeddings + BLIP captions + QA Web UI**—as the foundation for later tagging, OCR, and search.

## 2. Scope
### In Scope (M1)
1. **File scanning, content hashing & perceptual hashing**
   - Recursively scan one or more configured roots.
   - Filter non-image files.
   - Compute a content hash as the stable `image_id` and detect new/modified/deleted files.
   - Compute a perceptual hash (`phash`) for active, decodable images and persist it on the `images` rows.

2. **Lightweight pre-classification**
   - Produce per-image attributes such as:
     - `scene_type`: `PEOPLE / FOOD / ELECTRONICS / DOCUMENT / SCREENSHOT / LANDSCAPE / OTHER`
     - `has_text`, `has_person`, `is_screenshot`, `is_document`
   - Models load once; support batch inference.

3. **Flask Web UI for QA**
   - Filter images by the above attributes.
   - Paginated grid with thumbnails and metadata.
   - Detail view shows full-size image and pre-classification output.

4. **SigLIP embeddings & BLIP captions**
   - Compute SigLIP embeddings and BLIP captions for canonical images.
   - Persist primary results under `cache/embeddings/` and `cache/captions/`, with SQLite projections for inspection and lightweight search.

5. **Near-duplicate grouping**
   - Use perceptual hashes (`phash`) and a fixed Hamming-distance threshold to group visually similar images into near-duplicate sets.
   - Persist near-duplicate relationships into `image_near_duplicate` in `cache/index.db` for later cleanup and UX flows.

### Out of Scope (M1)
- Object-level tagging (`pizza`, `iPhone`, `MacBook Pro`), similarity search and ranking.
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
- CLI processing entry point (`uv run python -m vibe_photos.dev.process --root <album> --db data/index.db [--cache-db cache/index.db] [--batch-size ... --device ...]`) for scan + hash + classify + embeddings/captions; batch size/device are read from `config/settings.yaml` with CLI flags acting as overrides when provided.
- Flask app for manual inspection during development.
- A cache manifest under `cache/manifest.json` versions model/config settings; preprocessing stages trust cache artifacts only when the manifest matches the active `cache_format_version`, and a run journal in `cache/run_journal.json` supports resumable single-process runs.

## 4. Data Model (SQLite)
M1 uses two SQLite databases:
- A **primary operational database** at `data/index.db` for canonical image metadata, model outputs, near-duplicate relationships, and run state. This database is the only one that production-facing services (CLI, Web UI, future APIs) MUST read and write.
- A **projection database** at `cache/index.db` that MAY mirror the same schema for experimentation or ad-hoc analysis. It is considered disposable and is repopulated during preprocessing runs when the cache manifest matches the current `cache_format_version`; production services SHOULD NOT depend on it.

### Hash Strategy: Content vs Perceptual
M1 uses two distinct hash types:
- **Content hash (`image_id`)**
  - Purpose: identify byte-for-byte identical files and serve as the stable primary key.
  - Properties: highly sensitive—any change in file bytes (compression, crop, watermark) yields a new hash.
  - Algorithm: 64-bit xxHash (`xxhash64-v1`) computed over the raw file byte stream and stored as a 16-character hexadecimal string.
  - Uses:
    - Stable identity (`image_id` primary key).
    - Exact deduplication across multiple file paths.
    - Change detection: when the hash changes, treat the asset as a new image.
- **Perceptual hash (`phash`)**
  - Purpose: approximate visual similarity and tolerate minor edits (resize, crop, exposure tweaks).
  - Properties: robust to small pixel-level changes; diverges when content/composition meaningfully changes.
  - Uses in M1:
    - Compute and store `phash` for active, decodable images.
    - Drive near-duplicate grouping via Hamming-distance thresholds and canonical representative selection.
    - Enable future similarity grouping, “keep one representative” flows, and “potential duplicate” UI.

Mental model:
- `image_id` → “Are these files exactly the same bits?”
- `phash`   → “Do these photos look roughly the same?”

### `images` table (primary DB: `data/index.db`)
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
  phash           TEXT,           -- e.g. 64-bit perceptual hash as 16-char hex
  phash_algo      TEXT,           -- e.g. "phash64-v2"
  phash_updated_at REAL,          -- last computation time (POSIX timestamp)
  created_at      REAL NOT NULL,
  updated_at      REAL NOT NULL,
  status          TEXT NOT NULL,  -- active / deleted / error
  error_message   TEXT,
  schema_version  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_primary_path ON images(primary_path);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
```

Perceptual hash expectations in M1:
- For all rows with `status = "active"` whose images can be decoded, `phash`, `phash_algo`, and `phash_updated_at` MUST be populated.
- When an image cannot be decoded, `phash` MAY remain `NULL`, and the failure should be recorded via `status` and `error_message`.

Deletion semantics in M1:
- `images.status = "deleted"` is used only when **all** recorded paths for an `image_id` have disappeared from disk.
- When some paths disappear but at least one valid path remains, the row stays `status = "active"` and the missing paths are removed from `all_paths`.
- This ensures that a logical image remains active as long as it is present under any configured root.

### `image_scene` table (projection DB: `cache/index.db`)
Stores lightweight pre-classification outputs as a projection over cached detection files under `cache/detections/`.

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
  updated_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_image_scene_scene_type ON image_scene(scene_type);
CREATE INDEX IF NOT EXISTS idx_image_scene_has_text ON image_scene(has_text);
CREATE INDEX IF NOT EXISTS idx_image_scene_has_person ON image_scene(has_person);
CREATE INDEX IF NOT EXISTS idx_image_scene_is_screenshot ON image_scene(is_screenshot);
CREATE INDEX IF NOT EXISTS idx_image_scene_is_document ON image_scene(is_document);
```

### `image_embedding` table (projection DB: `cache/index.db`)
Stores metadata for per-image SigLIP embeddings as a projection over cache files under `cache/embeddings/`; the schema allows multiple embeddings per image keyed by `model_name`.

```sql
CREATE TABLE IF NOT EXISTS image_embedding (
  image_id       TEXT NOT NULL,
  model_name     TEXT NOT NULL,
  embedding_path TEXT NOT NULL,  -- path under cache/embeddings
  embedding_dim  INTEGER NOT NULL,
  model_backend  TEXT NOT NULL,
  updated_at     REAL NOT NULL,
  PRIMARY KEY (image_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_image_embedding_model_name ON image_embedding(model_name);
```

### `image_caption` table (projection DB: `cache/index.db`)
Stores BLIP caption outputs as a projection over cache files under `cache/captions/`; the schema allows multiple captions per image keyed by `model_name`.

```sql
CREATE TABLE IF NOT EXISTS image_caption (
  image_id      TEXT NOT NULL,
  model_name    TEXT NOT NULL,
  caption       TEXT NOT NULL,
  model_backend TEXT NOT NULL,
  updated_at    REAL NOT NULL,
  PRIMARY KEY (image_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_image_caption_model_name ON image_caption(model_name);
```

### `image_near_duplicate` table (projection DB: `cache/index.db`)
Stores near-duplicate relationships derived from `images.phash`. This table is fully rebuildable from the `images` table and the pHash algorithm.

```sql
CREATE TABLE IF NOT EXISTS image_near_duplicate (
  anchor_image_id    TEXT NOT NULL,
  duplicate_image_id TEXT NOT NULL,
  phash_distance     INTEGER NOT NULL,
  created_at         REAL NOT NULL,
  PRIMARY KEY (anchor_image_id, duplicate_image_id)
);

CREATE INDEX IF NOT EXISTS idx_image_near_duplicate_duplicate
  ON image_near_duplicate(duplicate_image_id);
```

By convention:
- `anchor_image_id` identifies the canonical representative of a near-duplicate group (current implementation picks the newer `mtime`, ties by `image_id`).
- `duplicate_image_id` is another image assigned to that group where the Hamming distance between `phash` values is within the configured threshold (current default `<= 16`; override via `pipeline.phash_hamming_threshold`).

## 5. Pipeline Design
All model and pipeline parameters (model IDs, devices, batch sizes, duplicate handling, detection toggles) are read from `config/settings.yaml` via `vibe_photos.config.load_settings()`. The CLI MAY provide overrides for common options (for example `--batch-size`, `--device`), and these take precedence over config values when supplied.

### Configuration Overview (M1)
- Configuration is loaded from `config/settings.yaml`; if the file is missing or malformed, `vibe_photos.config.load_settings()` falls back to safe defaults.
- M1 expects the following fields (see also `config/settings.example.yaml`):
  - `models.embedding`: SigLIP embedding model configuration (backend, `model_name` or `preset`, `device`, `batch_size`).
  - `models.caption`: BLIP captioning model configuration (backend, `model_name` or `preset`, `device`, `batch_size`).
  - `models.detection`: optional detection model; the current repository defaults to OWL-ViT enabled (`enabled: true`) with `pipeline.run_detection: true`. Set both to `false` for CPU-only runs or when you want to skip detection.
  - `models.ocr`: OCR configuration; for M1, `enabled` MUST remain `false` so preprocessing works without OCR.
  - `pipeline.run_detection`: whether to run detection in the preprocessing pipeline; default is `true` in this branch.
  - `pipeline.skip_duplicates_for_heavy_models`: when `true`, SigLIP embeddings and BLIP captions are computed only for canonical representatives of duplicate/near-duplicate groups.
  - `pipeline.phash_hamming_threshold`: pHash near-duplicate threshold; this branch uses `16` by default (instead of the original `12`), with gating logic keyed off `duplicate_image_id` rows.
- A minimal M1 configuration (after copying `settings.example.yaml` to `settings.yaml`) might look like:

```yaml
models:
  embedding:
    backend: siglip
    model_name: google/siglip2-base-patch16-224
    device: auto
    batch_size: 16

  caption:
    backend: blip
    model_name: Salesforce/blip-image-captioning-base
    device: auto
    batch_size: 4

  detection:
    enabled: true
    backend: owlvit
    model_name: google/owlvit-base-patch32
    device: auto
    max_regions_per_image: 5
    score_threshold: 0.2

pipeline:
  run_detection: true
  skip_duplicates_for_heavy_models: true
  phash_hamming_threshold: 16
```

### 5.0 CLI Parameters
- `--root`: one or more album root directories to scan (required; may be passed multiple times).
- `--db`: path to the primary operational SQLite database (optional, defaults to `data/index.db`).
- `--cache-db`: path to the projection/cache SQLite database for model outputs (optional, defaults to `cache/index.db`).
- `--batch-size`: override the batch size from `config/settings.yaml` for model inference.
- `--device`: override the device from `config/settings.yaml` (for example `cpu`, `cuda`, `mps`).

### 5.1 Scanner
- Recursively walk configured roots with an image extension allowlist (`.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.gif`; configurable).
- Emit `FileInfo` objects: `path`, `size_bytes`, `mtime`.

### 5.2 Hasher
- Stream files to compute `image_id` using a 64-bit xxHash (`xxhash64-v1`) over the raw file bytes, emitting a 16-character hexadecimal string.
- Upsert into `images`:
  - Insert new rows for unseen hashes.
  - Merge `all_paths` for duplicates; refresh metadata for changed files.
- When paths disappear from disk, remove them from `all_paths`; mark a row as `status = "deleted"` only when **all** recorded paths for that `image_id` are missing.

### 5.3 Lightweight Scene Classification
- Batch process active images missing `image_scene` rows or outdated `classifier_version`.
- Use the SigLIP embedding model configured via `config/settings.yaml` (`Settings.models.embedding`) and loaded via `src/vibe_photos/ml/models.py` singletons; inference uses the `transformers` `AutoProcessor` / `AutoModel` APIs.
- Scene classifier metadata:
  - In M1, `classifier_name` SHOULD be the resolved SigLIP model identifier (for example `google/siglip2-base-patch16-224`).
  - `classifier_version` SHOULD append a simple version tag (for example `google/siglip2-base-patch16-224-v1`) so stale rows can be detected and refreshed.
- Scene types and prompts (zero-shot over SigLIP embeddings):
  - Define the following scene types and English prompts:
    - `PEOPLE` → `"a portrait or group photo of people"`
    - `FOOD` → `"a photo of food"`
    - `ELECTRONICS` → `"a photo of electronic devices"`
    - `DOCUMENT` → `"a photo of a document or ID card"`
    - `SCREENSHOT` → `"a screenshot of a computer or phone user interface"`
    - `LANDSCAPE` → `"a landscape photo of nature or city"`
    - `OTHER` → `"a photo of something else"`
  - At initialization time, encode all scene prompts once using the SigLIP text encoder, L2-normalize the resulting embeddings, and reuse them for all batches.
  - For each image batch:
    - Encode images with the SigLIP image encoder and L2-normalize the embeddings.
    - Compute cosine similarities between image embeddings and the 8 scene prompt embeddings.
    - Apply a softmax over the 8 similarities to obtain a probability distribution.
    - Set `scene_type` to the argmax class and `scene_confidence` to the corresponding softmax probability.
- Boolean attributes (`has_text`, `has_person`, `is_screenshot`, `is_document`) via prompt pairs:
  - For each attribute, define a positive and negative prompt and a fixed decision threshold (M1 default: `0.05` on the similarity difference):
    - `has_person`:
      - Positive: `"a photo with one or more people"`
      - Negative: `"a photo without any people"`
    - `has_text`:
      - Positive: `"an image with a lot of text"`
      - Negative: `"an image without any text"`
    - `is_screenshot`:
      - Positive: `"a screenshot of a computer or phone user interface"`
      - Negative: `"a regular photo taken by a camera"`
    - `is_document`:
      - Positive: `"a photo of a document or ID card"`
      - Negative: `"a normal photo that is not a document"`
  - Encode all positive/negative prompts once at initialization, L2-normalize, and reuse for all batches.
  - For each attribute and each image embedding:
    - Compute `sim_pos = cosine(image, positive_prompt)` and `sim_neg = cosine(image, negative_prompt)`.
    - Set the attribute to `True` when `(sim_pos - sim_neg) > threshold` and `False` otherwise.
- **Model initialization happens once per process**; no per-image reloads.
- Scene classification outputs SHOULD also be cached under `cache/detections/<image_id>.json` (including `classifier_name`, `classifier_version`, and timestamps) so SQLite tables can be rebuilt from cache artifacts.

### 5.4 Embeddings & Captions (SigLIP + BLIP)
- Batch process active, non-deleted images to compute:
  - SigLIP image embeddings for each canonical `image_id`.
  - BLIP image captions for each canonical `image_id`.
- Use `Settings.models.embedding` and `Settings.models.caption` from `config/settings.yaml` to resolve concrete model names, with CLI flags (for example `--batch-size`, `--device`) acting as overrides at runtime.
- Load models once per process via `src/vibe_photos/ml/models.py` singletons and reuse them across batches.
- Write embeddings and captions primarily to cache:
  - Embeddings under `cache/embeddings/<image_id>.npy` (embedding vector + model metadata and preprocessing version).
  - Captions under `cache/captions/<image_id>.json` (caption text + model metadata + timestamps).
- Project these cached results into the projection SQLite database (`cache/index.db`) using the `image_embedding` and `image_caption` tables to support lightweight search and inspection; projection tables are written during preprocessing when the cache manifest matches the active format version.
- When `skip_duplicates_for_heavy_models=True` in `Settings.pipeline`, only run embeddings and captions for canonical representatives of duplicate/near-duplicate groups:
  - Exact duplicates (identical `image_id`) share one canonical representative.
  - Near-duplicates discovered via `image_near_duplicate` (anchor/duplicate relationships) reuse the anchor image's embeddings and captions.
  - In a fresh full run, it is preferable to compute pHash + near-duplicate groups before heavy models; in incremental runs, previously computed near-duplicate groups MAY be reused to gate heavy models.

### 5.5 Perceptual Hashing (pHash) & Near-Duplicate Grouping
- Responsibility: compute a robust perceptual hash for active images and persist it on the `images` rows.
- Inputs:
  - `image_id` (or corresponding primary paths) for `status = "active"` images in the `images` table.
- Processing:
  - Load images from `images.primary_path` using a standard imaging library.
  - Compute a 64-bit perceptual hash (`phash64-v2`) per image using a DCT-based algorithm:
    - Convert to grayscale and resize to a fixed resolution of 32×32 pixels.
    - Apply 2D DCT (type II) over the 32×32 grayscale matrix.
    - Take the top-left 8×8 low-frequency block (64 coefficients, including the DC term).
    - Compute the median of these 64 coefficients.
    - Set a bit to 1 if `coeff > median`, else 0, producing a 64-bit value.
    - Store as a 16-character hexadecimal string (`phash`), with `phash_algo = "phash64-v2"`.
  - Write back `phash`, `phash_algo`, and `phash_updated_at` in the `images` table.
- Implementation (current code path):
  - Uses the above DCT-based pHash algorithm and sets `phash_algo = "phash64-v2"`.
  - Near-duplicate grouping runs incrementally against “dirty” images (new or content-changed) and compares each dirty image against all active pHashes without bucketing. Items are sorted by `mtime` descending; anchors are chosen by newer `mtime`, tied by `image_id`. When the table is empty, the first run compares all pairs. This keeps heavy-model gating deterministic for a fixed set of mtimes but can change anchors if mtimes change.
  - Heavy models skip any `duplicate_image_id` found in `image_near_duplicate`; the `anchor_image_id` is treated as the canonical representative for embeddings/captions when `skip_duplicates_for_heavy_models` is `true`.
- Incremental behavior:
  - When `image_id` (content hash) changes, treat the asset as a new image and recompute `phash`.
  - When only `primary_path` changes but `image_id` remains stable, do not recompute `phash`.
  - When the configured algorithm version changes (for example from `"phash64-v2"` to `"phash64-v3"`), select rows where `phash_algo != current_algo` and recompute.
- Near-duplicate grouping:
  - Define near-duplicates using the Hamming distance between 64-bit `phash` values: two active images are considered near-duplicates when `distance(phash_a, phash_b) <= 16` (repository default; override via `pipeline.phash_hamming_threshold`).
  - The current implementation rebuilds pairs for dirty images by comparing against all active pHashes (no prefix buckets). Anchors are selected by newer `mtime` (tie-break on `image_id`); pairs are written to both `cache/index.db` and `data/index.db` and are incremental unless the table is empty.
  - Heavy-model gating relies on `duplicate_image_id` membership; change detection clears prior pairs for dirty IDs before recomputing.
  - Bucketing by high pHash bits remains a future optimization if scale/perf requires it.

### 5.6 Error Handling
- Corrupt/unreadable images do not halt the run; record `status="error"` and `error_message`.
- Re-running the pipeline skips unchanged images and only processes new/modified assets.

### 5.7 Incremental & Resumable Execution
- Use `image_id` as the stable unit of work so **scan, hash, and classify are idempotent**.
- Track per-stage versions/timestamps (e.g., `classifier_version`) to identify stale rows without reprocessing everything.
- Maintain a lightweight run journal so an interrupted run can **resume from the last completed batch** instead of restarting the full dataset:
  - Store the journal as a small JSON file at `cache/run_journal.json` containing at least the stage name, last completed batch cursor (for example last processed `image_id`), and checkpoint time.
  - Example shape:
    - `{"stage": "embeddings", "cursor_image_id": "abcd1234...", "updated_at": 1710000000.0}`.
  - The journal is best-effort and MAY be discarded without affecting correctness; it only exists to improve ergonomics on large runs.
- Prefer deterministic batching (e.g., ORDER BY `image_id`) to make retries repeatable and diff-friendly in logs.
- Use the shared application logger writing to `log/` (no `print()`), and include per-batch checkpoint metadata in the logs (stage name, batch cursor, counts).

## 6. Flask Web UI
Routes:
1. `GET /` → redirect to `/images`.
2. `GET /images`: list with filters (`scene_type`, `has_text`, `has_person`, `is_screenshot`, `is_document`, `page`, `page_size`).
3. `GET /image/<image_id>`: detail page with image preview and metadata.
4. (Optional) `GET /thumbnail/<image_id>`: serve or generate thumbnails; for M1, browser scaling or static routing is acceptable.

UI expectations:
- Filters at the top of the list page; grid layout (4–6 per row depending on CSS).
- Each card shows thumbnail + `scene_type` + boolean flags.
- Detail page shows a logical path (for example, a path relative to the configured album root or a user-friendly label), size, mtime, EXIF when available, and classifier metadata; it MUST NOT expose absolute filesystem paths or `file://` URIs.
- All image content (originals and thumbnails) is served via Flask routes keyed by `image_id` (such as `/image/<image_id>` or `/thumbnail/<image_id>`); templates SHOULD NOT embed raw OS paths.
- If thumbnail caching is implemented, write generated thumbnails to `cache/images/thumbnails/` and have `GET /thumbnail/<image_id>` read from there.

## 7. Suggested Package Layout & Entrypoints
```
project_root/
  src/
    vibe_photos/
      config.py
      db.py
      scanner.py
      hasher.py
      classifier.py
      pipeline.py
      webui/
        __init__.py
        views.py
        templates/
          base.html
          images.html
          image_detail.html
        static/
          css/
          js/
      dev/
        preprocess.py  # M1 CLI entrypoint
  tools/
    # Optional helpers; can host additional CLI entrypoints if needed
  data/
    index.db
  blueprints/
    m1/
      m1_development_plan.md  # this blueprint
```

- M1 processing CLI: `uv run python -m vibe_photos.dev.process --root <album> --db data/index.db [--cache-db cache/index.db] [--batch-size ... --device ...]`, using `config/settings.yaml` as the source of defaults with CLI flags overriding when provided.
- When `--cache-db` is omitted, the preprocessing CLI uses `cache/index.db` as the default projection/cache database path.
- Flask Web UI: `FLASK_APP=vibe_photos.webui uv run flask run` for manual inspection during development.

## 8. Acceptance Criteria
### Functional
- `uv run python -m vibe_photos.dev.process --root <album> --db data/index.db --cache-db cache/index.db --batch-size 16 --device cpu|cuda`:
  - Populates `images` with active assets, content hashes (`image_id`), and perceptual hashes (`phash`) for all decodable images.
  - Populates `image_scene` for active images and writes corresponding cache files under `cache/detections/`; records errors without stopping.
  - Computes near-duplicate groups based on `phash` Hamming distance (threshold `<= 16` by default) and persists them into `image_near_duplicate` in both cache and primary databases.
  - Computes near-duplicate groups based on `phash` Hamming distance (threshold `<= 16` by default) and persists them into `image_near_duplicate` in both `cache/index.db` and `data/index.db`; anchors are chosen by `mtime` and tie-broken by `image_id`.
  - Computes SigLIP embeddings and BLIP captions for canonical images and persists them under `cache/embeddings/` and `cache/captions/`, with projections into `image_embedding` and `image_caption`.
- Preprocessing respects `config/settings.yaml` defaults for model IDs, devices, batch sizes, and pipeline flags, with CLI arguments overriding config values when supplied.
- Re-running skips unchanged images; updates new/modified assets; flags missing paths appropriately.
- Interrupted runs can resume from recorded checkpoints/batch cursors without reprocessing completed work.
- Flask UI (`FLASK_APP=vibe_photos.webui uv run flask run`) loads list and detail pages with working filters and pagination, serving thumbnails from `cache/images/thumbnails/` when available. By default, detection is enabled; disable it via configuration for CPU-only environments.

### Performance & Stability
- Models load once; batches reuse loaded weights.
- Pipeline continues after single-image failures.
- Reasonable throughput for ~30k images (exact numbers to be captured after first full run).
 - All runs emit structured logs to `log/` (via the shared logger) including per-batch checkpoint metadata.

### Evolvability Hooks for M2+
- Use `schema_version`, `classifier_name`, and `classifier_version` to detect stale rows and re-run when models change.
- Scene, `has_text`, embeddings, and captions outputs gate future OCR runs and search; the cache layer stores embeddings, logits, detections, and OCR text keyed by `image_id`.

## 9. Search-Relevant Expectations (Preview for Future Phases)
- Each photo should eventually emit 1–3 high-quality coarse + mid-level labels to support text and embedding search.
- Electronics/food/document classifiers should be prioritized for accuracy.
- Offline batch processing with local indexes is sufficient for the target dataset scale; no cloud GPU dependency is required.
