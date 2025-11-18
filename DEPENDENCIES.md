# Dependency Manifest — Coding AI Reference

Use this manifest to validate that your environment matches the expected versions for Vibe Photos. The root `pyproject.toml` + `uv.lock` pair is the authoritative source; every other requirement list mirrors those files.

## 4. Installation Workflow

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync # install core + dev deps as defined in the lockfile
```

For GPU builds on NVIDIA hardware:

```bash
uv sync --all-extras
```

## 5. Model Artifacts
- SigLIP: `google/siglip2-base-patch16-224` (~400 MB)
- BLIP: `Salesforce/blip-image-captioning-base` (~990 MB)
- PaddleOCR Chinese package (~200 MB)

When adjustments are required, update `pyproject.toml`, run `uv lock`, then reflect the change in any mirrored requirement lists so downstream coding AIs inherit a consistent stack.
