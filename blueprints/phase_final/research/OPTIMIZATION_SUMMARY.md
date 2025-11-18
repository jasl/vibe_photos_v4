# Optimization Summary — Coding AI Notes

- Celery + Redis are used for task orchestration with prioritized queues; Prometheus/Grafana (optional) monitor throughput and failures.
- Vector storage is standardized on PostgreSQL + pgvector with appropriate indexes; Faiss or other vector engines are reserved for future scale if required.
- Detection + SigLIP/BLIP are treated as core perception components; optimization should preserve accuracy while improving throughput.
- The product focus remains: image ingestion, understanding, and search — avoid expanding into non‑core domains at the cost of stability.

Use this summary as a reminder of past tuning decisions and revisit the underlying docs if additional optimization work begins.
