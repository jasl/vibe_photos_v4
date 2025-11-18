# Model Integration Guidelines (for Coding AI)

These notes refine `AI_CODING_STANDARDS.md` for model integration. They
are written for coding AIs implementing Phase Final pipelines.

## 1. Transformers API Only

- Use `AutoProcessor`, `AutoModel`, and `AutoModelForSeq2SeqLM`.
- Do **not** use legacy CLIP or BLIP pipelines such as:
  - `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`
  - `pipeline("image-classification", model=...)`
- Always pass `return_tensors="pt"` and move tensors, models, and
  processor outputs to the configured `device`.

## 2. Singletons per Process

- All models (SigLIP, BLIP, detector, future OCR) MUST be loaded once
  per process and reused for all images.
- Implement model loaders (for example under `src/vibe_photos/ml/models.py`)
  that cache instances in module-level variables or a simple registry.
- Expose typed accessors (for example `get_siglip_model()`) rather than
  reloading inside loops or per-request code.

## 3. Batching

- M1 must support batching:
  - SigLIP: batch images up to `models.embedding.batch_size`.
  - BLIP: batch images up to `models.caption.batch_size`.
- Detection batching is optional in M1. Correctness is more important
  than throughput in the first implementation.
- Keep batch sizes configurable in `settings.yaml`; never hard-code
  them in library code.

## 4. CPU-First, GPU-Enhanced

- The code MUST run on CPU-only machines.
- If `torch.cuda.is_available()` or Apple MPS is available, use it but
  never depend on GPU-only features.
- Device selection should follow a simple helper (for example
  `get_torch_device()`), and all tensors and models must be moved
  consistently.

## 5. Caching and Database Separation

- All model outputs (embeddings, captions, detections) MUST be stored:
  - In versioned JSON or NPY files under `cache/` as the primary source
    of truth.
  - In a SQLite database as a projection over those caches for query
    and indexing.
- The database schema is allowed to evolve; cache formats must remain
  stable across migrations so databases can be rebuilt quickly.
- Cache artifacts must include:
  - Model name and backend.
  - Preprocessing or pipeline version.
  - Timestamps for creation and updates.

## 6. Config-Driven Model Choices

- Never hard-code model IDs, batch sizes, or device choices in the
  library code.
- Always read from `config/settings.yaml`, parsed in a central
  configuration module (for example `src/vibe_photos/config.py`).
- Toggle detection, OCR, and heavy models via configuration flags:
  library code should honor `enabled` / `run_detection` style options
  instead of embedding conditional logic in multiple places.

