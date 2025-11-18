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
  - First validate usefulness with **SQLite + SigLIP/BLIP** on a small subset of photos.
  - Only after confirming value, adopt heavier models and PostgreSQL + pgvector.
- Avoid over‑engineering:
  - Direct, well‑tested implementations beat complex abstractions that are hard to debug.
  - Keep services in a single `docker-compose` stack for Phase Final; defer multi‑cluster or multi‑tenant concerns.
- **Detection‑first perception**:
  - Grounding DINO / OWL‑ViT + SigLIP/BLIP re‑ranking is necessary to reliably find “物” at scale.
  - Design the perception pipeline to be pluggable so models can evolve without rewriting the rest of the system.
- Cache aggressively:
  - Thumbnails, detections, captions, embeddings, and (future) OCR results should be cached under `cache/` and reused across runs.

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
