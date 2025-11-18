# Vector DB Clarification — Local‑First Decision

**Decision:** PostgreSQL + pgvector is the primary and sufficient vector store for Phase Final, targeting local PC/NAS deployments with up to ~100k photos.

## 1. Reasons for PostgreSQL + pgvector

- **Single system of record:**
  - Photo metadata, captions, labels, and embeddings all live in one transactional database.
  - Simplifies backups, restores, and migrations for end‑users.

- **Hybrid query capability:**
  - PostgreSQL full‑text search + pgvector similarity in a single query.
  - Easy to combine with structured filters (time, collections, favorites).

- **Operational simplicity:**
  - A single PostgreSQL container in `docker-compose`.
  - No need for external vector services or separate clusters.

- **Scale fit:**
  - Target dataset: 30k–100k photos per user.
  - With appropriate indexes, pgvector can handle these sizes comfortably on a single machine.

## 2. When to Reconsider

Introduce an external or specialized vector service only if:

- Photo/embedding count consistently exceeds **~1M** on user deployments.
- Search latency remains **>200 ms** for typical queries despite:
  - Index tuning.
  - Reasonable hardware (desktop‑class CPU, SSD).
- Users require advanced features beyond pgvector’s scope, such as:
  - Cross‑user multi‑tenant search.
  - Online learning and very high query concurrency.

In such scenarios, options like Faiss, Qdrant, or other vector services can be considered, but they are **explicitly out of scope** for Phase Final.

## 3. Guidance for Tuning

- Start with HNSW (where available) or IVFFlat indexes tuned for:
  - The expected number of embeddings.
  - The hardware’s memory and CPU characteristics.
- Log and monitor:
  - Query latencies.
  - Index build times.
  - Disk space used by embeddings and indexes.

Record performance findings in `research/OPTIMIZATION_SUMMARY.md` and update this note if the strategy changes.
