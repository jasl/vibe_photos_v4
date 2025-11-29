## Qwen → SigLIP Teacher–Student Labeling Workflow

This note documents how we currently generate labeling “truth”, distill it into the
runtime SigLIP models, and validate progress. It should help future iterations
when we swap in other Qwen variants (e.g. Qwen3-VL 8B) or continue to curate the
dataset.

### 1. Teacher data pipeline

1. **Annotate with Qwen**  
   - Run `uv run python tools/qwen_vl_annotate_batch.py --root ... --output tmp/qwen_vl_raw_annotations.jsonl --resume`.  
   - This runs on the external GPU box (currently Qwen3-VL 32B). Each record is keyed by `image_id`.
2. **Convert to ground truth schema**  
   - `uv run python tools/qwen_to_ground_truth.py --input tmp/qwen_vl_raw_annotations.jsonl --output tmp/ground_truth_auto.jsonl`.  
   - Maps Qwen scene/attribute/object enums into M2 label keys (`scene.*`, `attr.*`, `object.*`).
3. **Sample for human audit**  
   - `uv run python tools/sample_ground_truth_for_review.py --input tmp/ground_truth_auto.jsonl --output tmp/ground_truth_human.json --size 800`.  
   - Manually fix the subset and save as `tmp/ground_truth_human.audited.json`.
4. **Evaluate teacher quality**  
   - `uv run python tools/evaluate_auto_vs_human.py --human tmp/ground_truth_human.audited.json --auto tmp/ground_truth_auto.jsonl`.  
   - Confirms Qwen is trustworthy (currently scene/attributes/objects all align closely).

### 2. Student (SigLIP) training

1. **Scene head**  
   - `uv run python tools/train_scene_head_from_qwen.py --gt tmp/ground_truth_auto.jsonl --output models/scene_head_from_qwen.pt`.  
   - Consumes SigLIP embeddings (`image_embedding` table) and trains a linear softmax head over `scene.*`.
2. **Attribute head**  
   - `uv run python tools/train_attribute_head_from_qwen.py --gt tmp/ground_truth_auto.jsonl --output models/attribute_head_from_qwen.pt`.  
   - Trains a multi-label BCE head for `attr.has_person`, `attr.has_text`, `attr.has_animal`, etc.  
   - “Rare / deterministic” attributes (`is_document`, `is_screenshot`) are derived directly from scene labels (see next section).
3. **(Future) Object head**  
   - Not implemented yet, but we will follow the same pattern, focusing on high-value `object.*` keys (monitor/keyboard/phone/food/etc.).

### 3. Runtime integration

`src/vibe_photos/classifier.py` loads the trained heads if the `.pt` files exist:

- Scene predictions come from `LearnedSceneHeadAdapter` (otherwise fall back to the
  zero-shot coarse classifier).
- Attributes use `LearnedAttributeHead` when available. For document/screenshot we
  override with explicit rules:
  - `is_document = ("scene.document" in assigned scene labels)`
  - `is_screenshot = ("scene.screenshot" in assigned scene labels)`
- Untouched attributes still have the prompt-based margins for diagnostics until
  everything is migrated.

After deploying new heads, rerun the pipeline (`uv run python -m
vibe_photos.dev.process ...`) so that scene/attribute label assignments refresh,
then validate via `uv run python -m vibe_photos.eval.labels --gt
tmp/ground_truth_human.audited.json`.

### 4. Ongoing dataset maintenance

1. Periodically rerun the teacher steps to refresh `ground_truth_auto.jsonl`.
2. Maintain/expand the human-audited subset; treat it as the canonical eval set.
3. Any time the teacher or mappings improve, retrain student heads and redeploy.
4. When hardware allows, swap the teacher (e.g. Qwen3-VL 8B). The rest of the
   pipeline stays unchanged—only the teacher annotations + retraining steps need
   rerunning.

This separation keeps runtime inference lightweight while ensuring we can
continuously improve accuracy through better weak labels and head retraining.

