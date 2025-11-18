# Review Alignment Updates

This note incorporates the latest review guidance into the Phase Final blueprint so downstream contributors understand the revised priorities and pipeline expectations.

## Retrieval Targets
- Each image must surface **1–3 high-quality coarse + mid-level labels** (e.g., `food` + `pizza`).
- Support **text search + embedding similarity** so queries like "pizza" or "iPhone" return relevant shots even without captions.
- Ensure strong classifiers for **electronics / food / documents** to unlock the authoring workflow.

## Model Responsibilities
- **Scene type classification (mandatory):** predict `{people, landscape, food, electronics, document/screenshot, other}` to gate later steps.
- **Multi-label object/concept tagging (mandatory):** use VLM/CLIP-style zero-shot labels; emit English tags then map to Chinese synonyms as needed.
- **Captions (optional for M1):** reserve for selected albums; prioritize correctness over verbosity.
- **Detection assistance:** for cluttered scenes (e.g., a table of dishes), run lightweight detection and label each region, merging global + regional tags.

## OCR Trigger Strategy
- Add a cheap pre-check (`has_text`, screenshot heuristics, or text-area ratio) before running OCR.
- Run OCR only when:
  - Scene type is `document / screenshot / electronics packaging`, or
  - Detected text area exceeds a threshold.
- Expect a 70%–90% cost reduction by avoiding unnecessary OCR passes.

## Pipeline Structure (Offline Batch)
1. **Scan & hash:** compute stable content hashes (`image_id`) for caching and deduping.
2. **Lightweight pre-classification:** output coarse scene types and `has_text` to decide downstream work.
3. **Main inference:**
   - Compute global embeddings.
   - Perform zero-shot tagging over a curated label list.
   - (Optional) Run region detection for multi-object scenes and merge deduped labels.
4. **OCR (gated):** execute only for eligible images per the trigger rules.
5. **Incremental + resumable runs:**
   - Treat `image_id` (content hash) as the stable unit of work to skip previously processed files.
   - Persist per-stage timestamps/versions so a rerun only touches missing or stale rows and can safely resume after interruptions.
6. **Cache vs DB separation:**
   - **Cache layer:** per-image embeddings, raw label scores/logits, detection boxes, raw OCR text keyed by `image_hash`.
   - **Database layer:** normalized tags, extracted models (e.g., `iPhone 14 Pro`), vector index references; rebuilt easily when models change.

## Processing Topology
- **Single init per process:** load models once and reuse across batches.
- **CPU path:** single process with threaded/async workers pulling from a queue.
- **GPU path:** prefer an inference server process that holds all models; workers send image paths via local RPC/HTTP.

## M1 Focus
- Ship **scene classification + tagging**; run captions sparingly.
- Avoid heavy UI—CLI or minimal Web UI suffices.
- Defer training/ensembles; rely on zero-/few-shot setups and cached results for rebuilds.

Use this document alongside the existing requirements/solution design files to guide Phase Final implementation and keep alignment with the reviewed plan.
