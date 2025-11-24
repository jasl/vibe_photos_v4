# Models — Phase Final (Current Defaults)

This document specifies the concrete model choices for the Phase Final pipeline
(foundation built in M1, extended in M2 with the label layer) and reflects the
current implementation in this repository.

The pipeline MUST provide:

- SigLIP image embeddings (full image) for search and coarse categories.
- BLIP captions (one short caption per image).
- Optional open-vocabulary detection (OWL-ViT) where hardware permits.

Detection is optional and can be enabled when hardware permits; the pipeline is
structured so detectors can be plugged in without refactoring the rest of the
pipeline.

## 1. Embeddings and Zero-Shot Classification (SigLIP)

**Purpose**

- Global image embeddings for vector search.
- Zero-shot classification for coarse categories (electronics, food, document, etc.).
- Re-ranking detection regions.

**Default model (implemented)**

- `google/siglip2-base-patch16-224`

This matches the blueprints and current implementation:

- Configuration defaults in `config/settings.yaml` / `config/settings.example.yaml`.
- Singleton loaders in `src/vibe_photos/ml/models.py`.
- The `SiglipBlipDetector` helper in `src/vibe_photos/ml/siglip_blip.py`.
- The coarse-category smoke test in `tools/test_coarse_categories.py`.

**High-quality option (later milestones)**

- `google/siglip2-large-patch16-384` or a similar higher-capacity SigLIP2 variant
  when available and practical.

**Implementation notes (for Coding AI)**

- Use the **Transformers** API (no legacy CLIP pipeline):

  ```python
  from transformers import AutoProcessor, AutoModel

  processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
  model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
  ```

- Do **not** use deprecated `CLIPModel` / `CLIPProcessor` here.

- Compute image features once per image:

  ```python
  inputs = processor(images=image, return_tensors="pt").to(device)
  with torch.no_grad():
      img_emb = model.get_image_features(**inputs)
  ```

- Normalize embeddings to unit length for cosine similarity:

  ```python
  img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
  ```

The model name MUST be configurable via `settings.yaml` under
`models.embedding.model_name`; callers SHOULD prefer the typed helpers in
`src/vibe_photos/config.py` and `src/vibe_photos/ml/models.py` instead of
constructing models manually.

## 2. Captioning (BLIP)

**Purpose**

- One short, human-readable caption per image.
- Used for:
  - Quick visual understanding in the UI.
  - Full-text search over captions.

**Default model (implemented)**

- `Salesforce/blip-image-captioning-base`

**Optional high-quality model**

- `Salesforce/blip-image-captioning-large` (GPU strongly recommended).

**Implementation notes**

M1 already wires BLIP via a shared loader; Coding AIs should reuse the existing
helpers and treat the snippet below as conceptual guidance, not a separate
implementation.

- Use the official BLIP captioning model via Transformers:

  ```python
  from transformers import AutoProcessor, AutoModelForSeq2SeqLM

  blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  blip_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/blip-image-captioning-base")

  inputs = blip_processor(images=image, return_tensors="pt").to(device)
  with torch.no_grad():
      ids = blip_model.generate(**inputs, max_length=50)
  caption = blip_processor.decode(ids[0], skip_special_tokens=True)
  ```

- Keep captions short (≤ 50 tokens) and neutral in tone.

- Store caption text in PostgreSQL using the `ImageCaption` ORM
  (`src/vibe_photos/db.py`) and mirror it into JSON caches under `cache/captions/`
  for rebuilds (see `_run_embeddings_and_captions` in `src/vibe_photos/pipeline.py`).

## 3. Open-Vocabulary Detection (Optional in M1)

**Purpose**

- Detect "things" (phone, laptop, pizza, document, MacBook Pro 14, etc.).
- Provide bounding boxes and initial labels for later re-ranking with SigLIP.

**M1 baseline detector (implemented, optional)**

For M1, detection is **optional** but already integrated behind configuration
flags. The current implementation uses **OWL-ViT** as a pragmatic starting
point:

- Backend: `owlvit`
- Model: `google/owlvit-base-patch32`

Detection is controlled by:

- `models.detection.enabled` and related fields in `config/settings.yaml`.
- `pipeline.run_detection` in `config/settings.yaml`.

Later milestones may introduce **Grounding DINO** (for example
`GroundingDINO_SwinT_OGC`) as the default when a GPU is available. The
detector MUST be pluggable via configuration so swapping backends does not
change the rest of the pipeline.

**Interface (for Coding AI)**

Define a detector abstraction (Python protocol or interface):

```python
from typing import Protocol

from PIL import Image


class Detector(Protocol):
    def detect(self, image: Image.Image, prompts: list[str]) -> list["Detection"]:
        ...
```

Where `Detection` includes:

- `bbox` (normalized coordinates).
- `label` (text from prompt).
- `score` (detector confidence).
- `model_name` and `backend` metadata.

Detections should be stored under:

- JSON cache file per photo under `cache/regions/` (for rebuilds).
- The feature-only `regions` + `region_embedding` tables in the primary database
  via the ORM models in `src/vibe_photos/db.py`.

## 4. OCR (Out of Scope for M1)

M1 **must not** depend on OCR. The pipeline should be designed with an
abstract OCR interface, but no OCR engine is required to run.

When implemented in later milestones, PaddleOCR (Chinese and English)
is a good candidate, but this will be decided in a future AI decision
record.

## 5. Few-Shot Embeddings (Later Milestones)

M1 only needs to ensure that image embeddings are stored with model
metadata so they can be reused later for few-shot learning.

Few-shot logic itself is a later milestone (M4). The embedding schema in this
repository already includes:

- `ImageEmbedding.model_name` and `ImageEmbedding.model_backend` (see
  `src/vibe_photos/db.py`).
- A `cache_format_version` field in `cache/manifest.json` that captures
  significant preprocessing or normalization changes across runs.

Future milestones MAY extend this with explicit version tags (for example
`siglip2-base-patch16-224-v1`) once real-world migrations are better
understood.
