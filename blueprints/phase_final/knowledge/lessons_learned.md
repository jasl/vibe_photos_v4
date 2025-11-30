# Lessons Learned — Coding AI Snapshot

This file captures key insights from previous iterations and the updated Phase Final blueprint.

## 1. Product Insights

- Focusing on **self‑media creators** with 30k–100k local photos gives clearer requirements than trying to serve every user type.
- Users care most about finding **objects and products** (“iPhone”, “MacBook”, “披萨”, “屏幕截图中的某个文档”), not just faces or generic scenes.
- AI is most valuable as an **assistant**:
  - High‑confidence predictions can auto‑apply.
  - Medium‑confidence predictions should be suggested for review.
  - Low‑confidence cases should defer to manual decisions.
- **Batch operations** (apply a confirmed label to similar assets) deliver the largest productivity gains.

## 2. Engineering Takeaways

- Start simple:
  - First validate usefulness with **local prototypes (SigLIP/BLIP)** on a small subset of photos.
  - Only after confirming value, adopt heavier models and PostgreSQL + pgvector.
- Favor a single, well‑factored preprocessing pipeline (as in M1) over parallel PoCs; reuse the same codepath from CLIs, queues, and UIs to keep behavior consistent.
- Avoid over‑engineering:
  - Direct, well‑tested implementations beat complex abstractions that are hard to debug.
  - Keep services in a single `docker-compose` stack for Phase Final; defer multi‑cluster or multi‑tenant concerns.
- **Detection‑first perception**:
  - Grounding DINO / OWL‑ViT + SigLIP/BLIP re‑ranking is necessary to reliably find “物” at scale.
  - Design the perception pipeline to be pluggable so models can evolve without rewriting the rest of the system.
- Cache aggressively:
  - Thumbnails, detections, captions, embeddings, and (future) OCR results should be cached under `cache/` and reused across runs.
- Treat caches as durable artifacts and databases as cache layers; updating schemas should prefer rebuilds from cache (using a manifest version) over ad-hoc migration scripts when feasible.

## 3. Process Improvements

- Ship in **small, verifiable increments**:
  - MVP first, then detection, then PostgreSQL/pgvector, then learning.
  - Avoid “big bang” rewrites that are hard to debug.
- Collect structured feedback:
  - Maintain examples of searches that work well and those that fail.
  - Use these to guide model choices and ranking tweaks.
- Document as you go:
  - Update `decisions/AI_DECISION_RECORD.md` when making or revising important choices.
  - Keep `AI_TASK_TRACKER.md` in sync with planned milestones and completed work so future coding AIs start with accurate context.

## 4. M2 Evaluation Snapshot (2025-11-30)

- Dataset: 1,000-image human-audited ground truth (`tmp/ground_truth_human.audited.json`).
- Scene accuracy (label layer): **75.1%**.
- Attributes (label layer, SigLIP heads):
  - `has_person` P≈0.94 R≈0.71 (prev thresholds too strict).
  - `has_text`  P≈0.73 R≈0.77.
  - `is_document` P≈0.91 R≈0.20 → lowered threshold to 0.20.
  - `is_screenshot` P≈1.00 R≈0.04 → lowered threshold to 0.18.
  - `has_animal` nearly zero coverage → threshold relaxed to 0.18 pending more data.
- Object labels (label layer, zero-shot): top-1/top-3/top-5 = **44.0%** on 637 labeled images; 335/357 error samples had **no predictions** → coverage is the main issue.
  - Actions taken: widened `object.zero_shot.scene_whitelist` (add people/animals/document/screenshot/snapshot), lowered `score_min` to 0.25 and `margin_min` to 0.05 to increase recall.
  - Next: inspect `tmp/eval_object_errors.jsonl` after rerun to decide blacklist/remap tweaks and label-group additions.
