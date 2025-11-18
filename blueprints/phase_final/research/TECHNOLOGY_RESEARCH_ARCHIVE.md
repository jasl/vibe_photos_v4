# Technology Research Archive — Coding AI Summary

This archive condenses evaluations run while selecting the Phase Final stack. Final decisions live in `AI_DECISION_RECORD.md`; use the notes below for historical context.

## Vector Stores
- **pgvector** chosen for current scale (<100k embeddings) due to transactional consistency and unified SQL/vector queries.
- **Faiss** reserved for future acceleration if datasets exceed ~1M embeddings or latency >200 ms despite tuning.
- **Other options** (Qdrant, Pinecone, Milvus) rejected as overkill for local-first deployment.

## Search Engines
- PostgreSQL FTS satisfies MVP requirements; Elasticsearch/MeiliSearch remain optional upgrades if advanced text relevance becomes necessary.

## Task Queues
- Celery + Redis offers mature retry semantics, scheduling, and monitoring. RQ/Dramatiq were evaluated but lacked needed features at scale.

## Model Comparisons
- SigLIP + BLIP outperformed RTMDet/CLIP combinations in multilingual accuracy and maintainability.
- PaddleOCR selected for Chinese/English text extraction; alternatives lagged in accuracy or performance.

Keep this file updated when new experiments run so future coding AIs understand prior analysis before re-evaluating the stack.
