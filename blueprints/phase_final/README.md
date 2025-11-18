# Phase Final Blueprint — Coding AI Overview

Phase Final defines the production-ready architecture for Vibe Photos once MVP validation succeeds.

## Folder Guide
- `docs/` — Requirements, solution design, technical choices, implementation roadmap.
- `architecture/` — System diagrams and deep dives (vector DB, queues, etc.).
- `knowledge/` — Lessons learned to inform future iterations.
- `research/` — Experiment logs and cleanup reports.
- `HANDOVER.md` — Checklist for transferring ownership into production teams.

## Core Themes
- PostgreSQL + pgvector replaces SQLite for scalable search.
- Celery + Redis orchestrate ingestion and background jobs.
- Streamlit remains the single UI stack; production work focuses on hardening it.
- Monitoring/observability stack (Prometheus/Grafana) introduced.

Before implementing Phase Final work, read the docs in order `01_requirements.md` → `02_solution_design.md` → `03_technical_choices.md` → `04_implementation_guide.md`, then consult the architecture notes for subsystem specifics.
