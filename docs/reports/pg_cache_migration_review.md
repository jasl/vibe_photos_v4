# PostgreSQL and Cache Migration Audit

## Residual legacy database code paths

Remaining follow-ups have been addressed by enforcing PostgreSQL-only URL normalization and dropping the legacy cache sentinel.
- The database engine factory previously retained file-backed-only behaviors (PRAGMA setup, legacy `connect_args`, and dialect-specific exports). With PostgreSQL as the only target, these branches can be isolated to shims or removed to simplify the engine bootstrap. 【F:src/vibe_photos/db.py†L375-L458】【F:src/vibe_photos/db.py†L493-L506】

## Documentation alignment gaps

- Coding notes now reflect the PostgreSQL-primary + filesystem cache model; keep them synced when cache formats change. 【F:docs/AI_CODING_NOTES.md†L41-L62】
- Phase blueprints and checklists previously implied file-backed databases as an operational target. Add or keep call-outs that PostgreSQL + pgvector is the primary store and file-backed mentions are historical; a note has been added to the Phase Final implementation guide, but a full sweep of older milestones is still recommended. 【F:blueprints/m1/m1_development_plan.md†L65-L308】【F:blueprints/phase_final/docs/04_implementation_guide.md†L46-L312】

## Suggested next steps

1. Make PostgreSQL the default in `normalize_database_url` (plain paths or missing schemes should raise or require an explicit driver), and introduce a cache-root-specific helper that no longer relies on `.db` sentinels.
2. Deprecate or shim legacy-only branches in the engine factory (`_get_engine`) and helper exports; with PostgreSQL as the only supported database, delete PRAGMA wiring and narrow `dialect_insert` accordingly.
3. Sweep docs and blueprints to replace legacy-database-as-primary messaging with the new architecture (PostgreSQL primary DB + rebuildable filesystem cache). Align cache descriptions with the refactored helper names to reduce confusion for new contributors.
