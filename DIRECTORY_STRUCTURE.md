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
| `data/` | Read/Write | Runtime SQLite DB (`vibe_photos.db`) and incremental state snapshots. | Git-ignored. Safe to delete between runs. |
| `cache/` | Read/Write | Reusable artifacts: normalized images, thumbnails, detection JSON, OCR results, embeddings, perceptual hashes, ingestion queue files. | Share across phases to skip recomputation. |
| `log/` | Read/Write | Rotating logs (`*.log`, `*.log.*`). | Configure rotation (10 MB, keep 5). |
| `tmp/` | Read/Write | Short-lived temp files. | Clean freely. |
| `models/` | Read/Write | Downloaded model weights (SigLIP, BLIP, PaddleOCR, etc.). | Git-ignored; ensure `.gitkeep` if needed. |
| `blueprints/` | Read-only | Design documentation for all phases. | Update only when explicitly refining docs. |

## 3. Cache Keys & Structure

```
cache/
├── images/
│   ├── processed/      # Normalized JPEG assets
│   └── thumbnails/     # 512x512 previews
├── detections/         # SigLIP classification results (.json)
├── captions/           # BLIP caption outputs (.json)
├── ocr/                # OCR text blocks (.json)
├── embeddings/         # Vector cache exports (SQLite dumps today, pgvector exports later)
├── ingestion_queue/    # Filesystem-backed task queue segments for the long-lived worker
└── hashes/             # Perceptual hash lookups
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
rm -rf cache/detections/*      # Remove classification cache
rm -rf cache/ocr/*             # Remove OCR cache
rm -rf cache/images/processed/*
```

## 5. Model Storage Guidance

- Download weights into `models/` with transformers/paddle cache pointing to this directory (`TRANSFORMERS_CACHE`, `PADDLEOCR_HOME`).
- Keep a README in `models/` describing which checkpoints are present and their source URLs.
- Do not commit model binaries; ensure `.gitignore` rules remain intact.
