## Qwen3-VL Assisted Evaluation Workflow (M2)

This document describes how to use an external Qwen3-VL server to generate
weak labels for your photo library and turn them into evaluation ground truth
for the M2 label layer.

The flow has three main steps:

1. Run Qwen3-VL on an external GPU server to annotate images.
2. Convert the JSONL annotations into `ground_truth_auto.jsonl`.
3. Sample a subset for manual correction and run the existing M2 eval CLI.

The design follows the blueprint in `blueprints/m2/m2_perception_and_labels_blueprint.md`
and the high-level plan in `use.plan.md`.

---

### 1. Generate Qwen3-VL Annotations on the Server

On the H100 (or other GPU) server you should have:

- A local deployment of Qwen3-VL accessible via an **OpenAI-compatible** HTTP API,
  e.g.:
  - `QWEN_BASE_URL=http://localhost:8000/v1`
  - Supports `POST /v1/chat/completions`
  - Supports `image_url` messages with `data:image/...;base64,...` payloads.
- A copy of this repository with the same photo files as on your local machine
  (copied via `rsync` or similar, **without** recompression).

Then on the server:

```bash
export QWEN_BASE_URL="http://localhost:8000/v1"
export QWEN_API_KEY="dummy-key-or-real-key"
export QWEN_MODEL="Qwen/Qwen3-VL-32B-Instruct"

# Dry run on a small subset
uv run python tools/qwen_vl_annotate_batch.py \
  --root /mnt/photos \
  --output /mnt/cache/qwen_vl_raw_annotations.jsonl \
  --limit 200

# Full run with resume support
uv run python tools/qwen_vl_annotate_batch.py \
  --root /mnt/photos \
  --output /mnt/cache/qwen_vl_raw_annotations.jsonl \
  --resume
```

The script:

- Uses the same `compute_content_hash` as the M2 pipeline to compute
  `images.image_id`.
- Calls Qwen3-VL once per image with a constrained JSON-only prompt.
- Appends one JSON object per line to the JSONL output file:

```jsonc
{
  "image_id": "8f14e45fceea167a5a36dedd4bea2543",
  "server_path": "/mnt/photos/2024/IMG_1234.JPG",
  "image_path": "/mnt/photos/2024/IMG_1234.JPG",
  "rel_path": "2024/IMG_1234.JPG",
  "annotation": {
    "scene": "product",
    "has_person": false,
    "is_document": false,
    "is_screenshot": false,
    "objects": [
      {
        "name_free": "mechanical keyboard",
        "coarse_type": "keyboard",
        "is_main": true,
        "confidence": 0.92
      }
    ]
  }
}
```

You should then copy the JSONL file back to your local machine, for example:

```bash
rsync -avz user@server:/mnt/cache/qwen_vl_raw_annotations.jsonl tmp/qwen_vl_raw_annotations.jsonl
```

---

### 2. Convert JSONL to M2 Ground Truth Format

On your local machine (where the M2 code lives) use:

```bash
uv run python tools/qwen_to_ground_truth.py \
  --input tmp/qwen_vl_raw_annotations.jsonl \
  --output tmp/ground_truth_auto.jsonl \
  --min-object-confidence 0.7
```

This script:

- Reads each JSONL record from Qwen (`image_id`, `annotation`).
- Maps Qwen's `scene` string to M2 label keys:
  - `"product" → "scene.product"`
  - `"food" → "scene.food"`
  - `"people" / "selfie" → "scene.people"`
  - `"animals" → "scene.animals"`
  - `"document" → "scene.document"`
  - `"screenshot" → "scene.screenshot"`
  - Anything unexpected → `"scene.other"`.
- Maps booleans to attribute keys:
  - `has_person → attr.has_person`
  - `has_animal → attr.has_animal`
  - `is_document → attr.is_document`
  - `is_screenshot → attr.is_screenshot`
  - (For now `attr.has_text` is not inferred from Qwen.)
- Maps `objects[*].coarse_type` to seeded object labels in the DB, only when:
  - `is_main` is `True` (unless `--include-non-main` is passed), and
  - `confidence >= --min-object-confidence` (default 0.7).

Examples of coarse_type → object label mapping:

- `"keyboard" → "object.electronics.peripheral.keyboard"`
- `"mouse" → "object.electronics.peripheral.mouse"`
- `"monitor" → "object.electronics.display.monitor"`
- `"phone" → "object.electronics.phone"`
- `"laptop" → "object.electronics.laptop"`
- `"tablet" → "object.electronics.tablet"`
- `"food" → "object.food"`
- `"drink" → "object.drink"`
- `"dessert" → "object.food.dessert"`

The output is a JSONL file where each line looks like:

```jsonc
{
  "image_id": "8f14e45fceea167a5a36dedd4bea2543",
  "scene": ["scene.product"],
  "attributes": {
    "attr.has_person": false,
    "attr.is_document": false,
    "attr.is_screenshot": false
  },
  "objects": ["object.electronics.peripheral.keyboard"]
}
```

This format is directly consumable by `vibe_photos.eval.labels` (the eval CLI
will treat it as JSONL if top-level parsing fails).

---

### 3. Sample Records for Manual Correction

To build a high-quality evaluation subset, sample a few hundred to ~1500
records and check them manually:

```bash
uv run python tools/sample_ground_truth_for_review.py \
  --input tmp/ground_truth_auto.jsonl \
  --output tmp/ground_truth_human.json \
  --size 1000 \
  --seed 42
```

This script:

- Loads the auto ground truth (JSON or JSONL).
- Randomly samples `--size` records (bounded by total count).
- Writes a pretty-printed JSON list to `ground_truth_human.json`.

You can then open `ground_truth_human.json` in your editor and:

- Fix obvious scene mistakes (`scene.food` vs `scene.product`, etc.).
- Toggle attributes `attr.has_person`, `attr.is_document`, `attr.is_screenshot`.
- Add or remove object label keys in `objects`.

The important guarantee is that **`image_id` stays unchanged** so that
evaluation uses the same IDs as the M2 databases.

---

### 4. Run M2 Evaluation CLI

Once you have a human-corrected subset:

```bash
uv run python -m vibe_photos.eval.labels --gt tmp/ground_truth_human.json
```

This will print:

- Scene accuracy and confusion-style stats.
- Attribute precision/recall/F1 per attribute key.
- Object-label top-1 / top-3 hit rates.

You can optionally evaluate directly on the auto labels for quick iteration:

```bash
uv run python -m vibe_photos.eval.labels --gt tmp/ground_truth_auto.jsonl
```

The absolute numbers will be influenced by Qwen's own error rate, but deltas
between different M2 configurations (e.g., object label dictionaries, thresholds,
or classifier versions) are still useful.

---

### 5. Typical Iteration Loop

1. Run Qwen3-VL annotations once and keep the JSONL as a cache.
2. Convert to ground_truth_auto.jsonl with updated mapping/thresholds.
3. Re-sample a small human set when mappings change significantly.
4. Run `vibe_photos.eval.labels` on the human subset for honest metrics.
5. Use the large auto set to sanity-check that improvements generalize, even if
   the labels are noisy.

This lets you iterate quickly on label dictionaries and thresholds while still
anchoring the M2 pipeline to a relatively small, high-quality human-labeled
evaluation set.


---

### 6. Distill Qwen labels into lightweight SigLIP heads

Once `ground_truth_auto.jsonl` and the audited subset look good, use them as
teacher data for the on-device SigLIP student. The goal is to keep large models
off the target hardware while still benefitting from their judgments.

1. **Train scene head (teacher → student)**

   ```bash
   uv run python tools/train_scene_head_from_qwen.py \
     --gt tmp/ground_truth_auto.jsonl \
     --output models/scene_head_from_qwen.pt
   ```

   - Reads SigLIP embeddings from the database (`image_embedding` table).
   - Learns a linear softmax head that maps embeddings to `scene.*`.
   - `src/vibe_photos/classifier.py` auto-loads the `.pt` file if present.

2. **Train attribute head**

   ```bash
   uv run python tools/train_attribute_head_from_qwen.py \
     --gt tmp/ground_truth_auto.jsonl \
     --output models/attribute_head_from_qwen.pt
   ```

   - Produces a BCE-with-logits multi-label head for `attr.has_person`,
     `attr.has_text`, `attr.has_animal`, etc.
   - The classifier prioritizes this head when available, but still retains
     the original prompt-based margins for logging/analysis.

3. **Rules for rare attributes**

   - `is_document` and `is_screenshot` are derived directly from the scene
     label (`scene.document`, `scene.screenshot`). This keeps them consistent
     with the taxonomy and avoids sparse-head instability.
   - The learned attribute head still emits logits/margins so you can inspect
     them or add overrides later.

4. **Evaluate against human truth**

   ```bash
   uv run python -m vibe_photos.eval.labels \
     --gt tmp/ground_truth_human.audited.json
   ```

   - Confirms end-to-end quality of the student heads before deployment.
   - Use the same command after every retrain to track drift.

5. **Future extensibility**

   - You can retrain the heads whenever a new teacher (e.g., Qwen3-VL 8B) or
     better human labels become available—no runtime changes required.
   - The same pattern can be extended to object labels by adding
     `tools/train_object_head_from_qwen.py` and wiring a `LearnedObjectHead`
     into the label pass when needed.





