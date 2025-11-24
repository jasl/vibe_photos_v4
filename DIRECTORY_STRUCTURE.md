# Directory Protocol — Vibe Photos Coding AI

This document explains where every artifact lives and how coding AIs should interact with the filesystem during development and testing.

## 1. Operational Principles

- Phase 1 is disposable: you may wipe `data/`, `cache/`, `log/`, and `tmp/` whenever you need a clean run.
- Do not modify anything inside `samples/` or blueprint folders; they are treated as read-only specs.
- Model assets are large; download once into `models/` and reuse across runs.

## 2. Directory Breakdown

| Path | Access | Purpose | Notes |
|------|--------|---------|-------|
| `samples/` | Read-only | Canonical evaluation photos provided by stakeholders. | Never edit or commit changes. |
| `data/` | Read/Write | Runtime PostgreSQL exports/snapshots and incremental state archives. | Git-ignored. Safe to delete between runs. |
| `cache/` | Read/Write | Reusable artifacts: normalized images, thumbnails, detection JSON, OCR results, embeddings, perceptual hashes, ingestion queue files. | Share across phases to skip recomputation. |
| `log/` | Read/Write | Rotating logs (`*.log`, `*.log.*`). | Configure rotation (10 MB, keep 5). |
| `tmp/` | Read/Write | Short-lived temp files. | Clean freely. |
| `models/` | Read/Write | Downloaded model weights (SigLIP, BLIP, PaddleOCR, etc.). | Git-ignored; ensure `.gitkeep` if needed. |
| `blueprints/` | Read-only | Design documentation for all phases. | Update only when explicitly refining docs. |

## 3. Cache Keys & Structure

```
cache/
├── artifacts/             # Per-image artifacts (thumbnails, metadata JSON, etc.)
├── embeddings/            # Image + region embedding vectors (.npy + metadata)
│   └── regions/<model>/   # Region-level embeddings
├── captions/              # BLIP caption outputs (.json)
├── regions/               # Detection payloads (.json)
├── label_text_prototypes/ # Object-label prototype archives (.npz)
├── manifest.json          # Cache manifest (version + settings snapshot)
└── run_journal.json       # Resumable pipeline journal
```

- Use perceptual hash (`phash`) as the cache key to deduplicate identical content.
- Store metadata in JSON with ISO timestamps and version info.

## 4. Reset Recipes

```bash
# Full reset (wipe runtime artifacts)
rm -rf data/* cache/* log/* tmp/*

# Preserve expensive caches (keep processed assets & embeddings)
rm -rf data/* log/* tmp/*

# Targeted cleanup
rm -rf cache/captions/*
rm -rf cache/embeddings/*
rm -rf cache/regions/*
rm -rf cache/artifacts/*
rm -rf cache/label_text_prototypes/*
```

## 5. Model Storage Guidance

- Download weights into `models/` with transformers/paddle cache pointing to this directory (`TRANSFORMERS_CACHE`, `PADDLEOCR_HOME`).
- Keep a README in `models/` describing which checkpoints are present and their source URLs.
- Do not commit model binaries; ensure `.gitignore` rules remain intact.
