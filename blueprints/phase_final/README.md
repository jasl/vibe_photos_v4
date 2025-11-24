# Phase Final Blueprint — Coding AI Overview

Phase Final defines the production-ready architecture for Vibe Photos once MVP validation succeeds.

## Folder Guide
- `docs/` — Requirements, solution design, technical choices, implementation roadmap, and review alignment notes.
- `architecture/` — System diagrams and deep dives (vector DB, queues, etc.).
- `knowledge/` — Lessons learned to inform future iterations.
- `research/` — Experiment logs and cleanup reports.
- `HANDOVER.md` — Checklist for transferring ownership into production teams.

## Core Themes
- PostgreSQL + pgvector is the standard stack for scalable search.
- Celery + Redis orchestrate ingestion and background jobs.
- Streamlit remains the single UI stack; production work focuses on hardening it.
- Monitoring/observability stack (Prometheus/Grafana) introduced.

Before implementing Phase Final work, read the docs in order `01_requirements.md` → `02_solution_design.md` → `03_technical_choices.md` → `04_implementation_guide.md`, then consult the architecture notes for subsystem specifics.

## Migration Notes
- `docker-compose.yml` currently only ships a Redis stub; it will be expanded to add PostgreSQL + pgvector, the FastAPI service, background workers, and the Streamlit UI as Phase Final work lands.
- Pre-process outputs live in `cache/` and are mirrored into `data/index.db`; plan to import these cached artifacts into PostgreSQL during the first pgvector migration so M1 runs can seed Phase Final indexes without recomputing models. Content-hash changes invalidate cache-side artifacts and near-duplicate pairs for that image and trigger incremental recompute; primary rows remain unchanged for auditability.
- Near duplicates are incremental: when new images arrive or content changes, existing pairs involving those images are dropped and recomputed against all active images; if the table is empty, a full pass runs once.
- Upcoming steps: define the pgvector schema that mirrors M1 caches, add container images/services to the compose stack, and wire ingestion/search APIs to read from PostgreSQL instead of legacy prototypes.
