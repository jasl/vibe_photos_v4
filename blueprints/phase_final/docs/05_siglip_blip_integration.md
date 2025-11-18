# Detection + SigLIP + BLIP Integration — Coding AI Notes

This note describes how Grounding DINO / OWL‑ViT, SigLIP, and BLIP work together in the perception pipeline.

## 1. Rationale

- **Goal:** Automatically understand “what is in the photo” at both **object level** (boxes) and **image level** (scene/product summary), without manual labels.
- **Detection first:** Grounding DINO / OWL‑ViT provide open‑vocabulary detection:
  - They detect objects using text prompts (“phone”, “pizza”, “document”, “MacBook Pro 14”).
  - This yields explicit regions for each “thing” in the frame.
- **Re‑ranking and embeddings:** SigLIP:
  - Re‑scores detected objects to reduce hallucinations.
  - Produces image embeddings for vector search.
- **Narrative context:** BLIP:
  - Produces natural‑language captions that are easy for users to interpret.
  - Serves as an additional signal for full‑text search and ranking.

Together, these components produce rich, structured metadata for every photo, enabling powerful “按物找图”.

## 2. High‑Level Flow

For each image:

1. **Load models and processors (startup)**
   - Grounding DINO / OWL‑ViT detection model + tokenizer.
   - SigLIP model + processor.
   - BLIP caption model + processor.

2. **Detection**
   - Construct a list of prompts:
     - Core categories: electronics, food, documents, screenshots, people, landscapes, etc.
     - Product prompts from the few‑shot registry (e.g. “HHKB Professional Hybrid Type‑S”, “Keysight DSO‑X 3024T”).
   - Run Grounding DINO / OWL‑ViT:
     - Get bounding boxes, associated prompt texts, and detector scores.

3. **SigLIP re‑ranking & embeddings**
   - For each detected box:
     - Crop the region.
     - Run SigLIP with a candidate label set (including generic + product labels).
     - Produce a refined score distribution.
   - For the full image:
     - Compute an image embedding using SigLIP.
     - Store the embedding in pgvector for later search.

4. **BLIP captioning**
   - Run BLIP once per image:
     - Produce a short, descriptive caption.
   - Optionally condition prompts to focus on:
     - Products (“a silver laptop on a wooden desk”).
     - Food (“a plate of pizza with cheese and tomatoes”).

5. **Persistence**
   - Store in the database:
     - Photo record with primary category and confidence.
     - Detected regions with bounding boxes and merged labels/scores.
     - SigLIP image embedding (and optionally region embeddings), including coarse category classification outputs.
     - BLIP captions.

## 3. Implementation Sketch

The actual implementation must follow project coding standards, but the flow conceptually looks like:

```python
def analyze_image(image: Image.Image, prompts: list[str], labels: list[str]) -> AnalyzedPhoto:
    # 1) Open-vocabulary detection
    detections = detector.detect(image, prompts=prompts)

    # 2) SigLIP re-ranking and embeddings
    img_inputs = siglip_processor(images=image, return_tensors="pt").to(device)
    img_emb = siglip_model.get_image_features(**img_inputs)

    region_results = []
    for det in detections:
        crop = image.crop(det.bbox.to_tuple())
        inputs = siglip_processor(
            text=labels,
            images=crop,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            probs = torch.softmax(outputs.logits_per_image[0], dim=-1)
        top_scores = sorted(zip(labels, probs.tolist()), key=lambda pair: pair[1], reverse=True)
        region_results.append((det, top_scores))

    # 3) BLIP caption
    blip_inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(**blip_inputs, max_length=50)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    # 4) Construct domain object to persist
    return build_analyzed_photo(img_emb, region_results, caption)
```

> Note: Actual code should handle batching, device placement, error handling, and logging. The purpose here is to illustrate **data flow**, not final APIs.

### 3.1 Coarse Categories via SigLIP Zero-Shot

Coarse categories (food, electronics, screenshot, document, people, landscape, other) can be implemented as a small zero-shot
classifier on top of existing SigLIP image embeddings, without any manual labels.

**Configuration example (YAML):**

```yaml
categories:
  - id: food
    zh: 食物
    prompts:
      - "a photo of food"
      - "a plate of cooked food"
      - "a restaurant meal"
  - id: electronics
    zh: 电子产品
    prompts:
      - "an electronic device"
      - "a laptop on a desk"
      - "a smartphone on a table"
  - id: screenshot
    zh: 截图
    prompts:
      - "a computer screenshot"
      - "a phone screenshot"
  - id: document
    zh: 文档
    prompts:
      - "a photo of a document"
      - "a photo of a receipt or contract"
  - id: people
    zh: 人像
    prompts:
      - "a portrait photo of a person"
      - "a group photo of people"
  - id: landscape
    zh: 风景
    prompts:
      - "a landscape photo of nature"
      - "a cityscape photo"
  - id: other
    zh: 其他
    prompts: []  # fallback only, not scored directly
```

**Classifier sketch (Python-style):**

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor


@dataclass
class CoarseCategory:
    id: str
    zh: str
    prompts: List[str]


class CoarseCategoryClassifier:
    def __init__(
        self,
        siglip_model,
        siglip_processor,
        categories: List[CoarseCategory],
        threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.model = siglip_model
        self.processor = siglip_processor
        self.categories = categories
        self.threshold = threshold
        self.device = device
        self.text_embs: Dict[str, Tensor] = {}
        self._prepare_text_embeddings()

    def _prepare_text_embeddings(self) -> None:
        """Encode all prompts once at startup."""
        self.model.eval()
        for category in self.categories:
            if not category.prompts:
                continue
            inputs = self.processor(
                text=category.prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            self.text_embs[category.id] = text_emb

    def classify_from_image_embedding(self, img_emb: Tensor) -> Tuple[str, Dict[str, float]]:
        """Return primary coarse category and per-category scores."""
        if img_emb.ndim == 2:
            img_emb = img_emb[0]
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        scores: Dict[str, float] = {}
        for category in self.categories:
            text_emb = self.text_embs.get(category.id)
            if text_emb is None:
                continue
            logits = img_emb @ text_emb.T
            scores[category.id] = float(logits.max().item())

        best_id = "other"
        best_score = -1.0
        for category_id, score in scores.items():
            if score > best_score:
                best_id = category_id
                best_score = score

        if best_score < self.threshold:
            best_id = "other"

        scores.setdefault("other", 0.0)
        return best_id, scores
```

**Integration into the main pipeline:**

```python
def analyze_image(image: Image.Image, prompts: list[str], labels: list[str]) -> AnalyzedPhoto:
    detections = detector.detect(image, prompts=prompts)

    img_inputs = siglip_processor(images=image, return_tensors="pt").to(device)
    img_emb = siglip_model.get_image_features(**img_inputs)

    primary_category_id, coarse_scores = coarse_classifier.classify_from_image_embedding(img_emb)

    # ... region re-ranking and BLIP caption as above ...

    return AnalyzedPhoto(
        image_path=...,
        primary_category=primary_category_id,
        coarse_category_scores=coarse_scores,
        siglip_embedding=img_emb.cpu().numpy(),
        detections=region_results,
        caption=caption,
    )
```

> API calls to `get_image_features` / `get_text_features` are conceptual; adapt them to the specific SigLIP model class and version in use.

## 4. Practical Tips

- **Caching:**
  - Cache model weights in `models/` and reuse model instances across tasks.
  - For large batches, process images in mini‑batches to balance throughput and memory.
- **Prompts & labels:**
  - Maintain:
    - A global base label set (electronics, food, documents, etc.).
    - Per‑user product prompts from few‑shot learning and recent annotations.
    - A shared **label dictionary** that:
      - Assigns each canonical label an ID (e.g. `pizza`, `laptop`, `macbook_pro`).
      - Stores English model labels plus Chinese display names and aliases (e.g. `pizza` ↔ `披萨`/`比萨`).
      - Normalizes synonyms before persistence so that detector/SigLIP outputs map to the same canonical ID.
  - Regularly prune or consolidate rarely used prompts to keep inference efficient.
  - Use the label dictionary for:
    - Building detection prompt lists.
    - SigLIP candidate label sets.
    - Query normalization (mapping queries like “披萨” to `pizza`).

- **Region constraints & filtering:**
  - Limit the number of stored regions per image (e.g. top N by SigLIP‑refined score) to avoid noisy, low‑value boxes.
  - Apply a minimum confidence threshold after SigLIP re‑ranking; discard regions below this threshold.
  - Prefer fewer, higher‑quality detections over many uncertain ones, especially for search and UI explanations.
- **Performance:**
  - Prefer GPU when available for detector and SigLIP/BLIP.
  - On CPU‑only machines:
    - Use smaller models.
    - Increase batch size carefully to maintain throughput.
    - Allow users to throttle concurrency.

Record any concrete implementation decisions and model choices in `AI_DECISION_RECORD.md` as the system evolves.
