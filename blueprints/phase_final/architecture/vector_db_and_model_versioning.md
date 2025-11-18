# Vector DB & Model Versioning — Local Deployment Strategy

This document details how embeddings and models are managed when running PostgreSQL + pgvector on a personal computer or NAS.

## 1. Vector Store Strategy

- **Single store:** Use PostgreSQL + pgvector to co‑locate:
  - Photo metadata (paths, timestamps, categories).
  - Embeddings for photos (and optionally regions).
- **Indexes:**
  - Use pgvector indexes (e.g. HNSW or IVFFlat depending on PostgreSQL/pgvector version and hardware).
  - Index tuning optimized for:
    - Dataset sizes of 30k–100k photos.
    - Latency targets of ≤500 ms per search on a single machine.
- **Tables:**
  - `photo_embeddings` table (or vector columns on `photos`) with:
    - `photo_id` (FK to photos).
    - `embedding_model` (e.g. `siglip2-base-patch16-224`).
    - `embedding_version` (semantic version string).
    - `embedding` (vector type).
    - `created_at` / `updated_at`.
  - Optional `region_embeddings` table for region‑level embeddings when needed.

## 2. Model Registry & Versioning

- **Registry table:** `model_registry` records all deployed models:
  - `id`, `name` (e.g. `siglip2-base-patch16-224`).
  - `type` (`detector`, `embedding`, `captioning`, `ocr`, `few_shot`).
  - `version` (semantic version `vX.Y.Z`).
  - `artifact_path` (relative path in `models/`).
  - `config` (JSON with dimensions, parameters, etc.).
  - `created_at`, `deprecated_at`.

- **Tracked models:**
  - Detector models (Grounding DINO / OWL‑ViT).
  - Embedding/classification models (SigLIP variants).
  - Captioning models (BLIP variants).
  - Optional OCR models (PaddleOCR or others).
  - Few‑shot prototype models (DINOv2, SigLIP‑based).

- **Compatibility:**
  - Each embedding record references the specific model/version that produced it.
  - When swapping models, new embeddings are written alongside existing ones for safe migration/rollback.

## 3. Updating Models and Embeddings

When updating one of the core models (e.g. SigLIP, BLIP, detector):

1. **Register new model:**
   - Insert into `model_registry` with status `staging`.
   - Download and verify artifacts into `models/`.

2. **Background embedding/detection job:**
   - Celery workers run a long‑running job that:
     - Computes embeddings or detections for all photos (or a sampled subset).
     - Stores outputs in:
       - New table (`photo_embeddings_new`) or
       - New `embedding_version` entries alongside existing ones.

3. **Validation:**
   - Evaluate:
     - Recall/precision on curated evaluation sets (electronics, food, documents, screenshots).
     - Search latency and resource usage.
   - Record results in research notes and/or the decision log.

4. **Promotion:**
   - Mark the new model as `active` in `model_registry`.
   - Update application configuration to use the new model for:
     - New ingestions.
     - Query embeddings.

5. **Cleanup (optional):**
   - Keep older embeddings and models for rollback until the user opts to reclaim disk space.
   - Provide maintenance commands to drop deprecated versions if desired.

### Update Workflow Summary

```text
register new model
→ background job computes new embeddings/detections
→ validate quality and latency
→ mark model active in registry
→ optionally drop old versions when safe
```

## 4. Local‑First Considerations

- **Disk usage:**
  - Embeddings and model artifacts can be large; all are stored under user‑controlled paths (`models/`, `data/`).
  - Provide tools to:
    - Estimate space usage per model/version.
    - Prune old embeddings and disabled models.

- **Performance tuning:**
  - On machines without GPUs:
    - Prefer more compact models or reduced batch sizes.
    - Allow users to trade off performance vs. accuracy in settings.
  - On machines with GPUs:
    - Enable GPU‑accelerated inference for detectors and embeddings.
    - Keep pgvector on CPU; rely on indexes to meet latency targets.

- **Resilience:**
  - Background jobs must be restartable:
    - Track progress via job tables.
    - Resume after crashes or shutdowns.

Document any deviations from this strategy in `AI_DECISION_RECORD.md` for future maintainers.
