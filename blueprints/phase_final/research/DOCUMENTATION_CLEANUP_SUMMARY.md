# Documentation Cleanup Summary — Coding AI Note

- Active decisions should live in `decisions/AI_DECISION_RECORD.md` and the Phase Final docs; research files remain for historical context only.
- Vector store references are standardized to PostgreSQL + pgvector; Faiss and other services appear solely as future options when scale demands it.
- Docs are organized so implementers read requirements, design, and architecture guides first, then consult research archives if needed.

Maintain this separation when adding new materials:

- Decisions in `decisions/`.
- Implementation guides and specs in `blueprints/`.
- Exploratory notes here in `research/`.
