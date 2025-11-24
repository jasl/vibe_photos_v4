# PostgreSQL and Cache Migration Audit

## Residual SQLite-oriented code paths

Remaining follow-ups have been addressed by enforcing PostgreSQL-only URL normalization and dropping the legacy SQLite cache sentinel.
- The database engine factory retains SQLite-only behaviors (PRAGMA setup, `connect_args` timeouts, and SQLite-specific exports in `__all__`). If SQLite is no longer supported, these branches can be isolated to legacy/shim paths or removed to simplify the engine bootstrap. 【F:src/vibe_photos/db.py†L375-L458】【F:src/vibe_photos/db.py†L493-L506】

## Documentation alignment gaps

- Coding notes still describe caching to SQLite as the canonical store, which conflicts with the current PostgreSQL-primary + filesystem cache design. Updating this section to reflect the Postgres + cache separation would keep guidance consistent. 【F:docs/AI_CODING_NOTES.md†L41-L62】
- Phase blueprints and checklists continue to position SQLite as the operational database (for example Phase Final and M2 documents), while the README and compose defaults now assume PostgreSQL. These should be refreshed to avoid signaling that SQLite is an active target. 【F:blueprints/m1/m1_development_plan.md†L65-L308】【F:blueprints/phase_final/docs/04_implementation_guide.md†L46-L312】

## Suggested next steps

1. Make PostgreSQL the default in `normalize_database_url` (plain paths or missing schemes should raise or require an explicit driver), and introduce a cache-root-specific helper that no longer relies on SQLite URLs or `.db` sentinels.
2. Deprecate or shim SQLite-only branches in the engine factory (`_get_engine`) and helper exports; if SQLite support is intentionally removed, delete PRAGMA wiring and narrow `dialect_insert` to PostgreSQL.
3. Sweep docs and blueprints to replace SQLite-as-primary messaging with the new architecture (PostgreSQL primary DB + rebuildable filesystem cache). Align cache descriptions with the refactored helper names to reduce confusion for new contributors.
