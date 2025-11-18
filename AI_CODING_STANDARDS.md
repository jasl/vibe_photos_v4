# Coding Standards — Vibe Photos Coding AI Charter

Follow these standards verbatim. They encode the minimum quality bar expected from every coding AI committing to this repository.

## 1. Language Policy

- **Source code, comments, commits:** English only.
- **UI assets, configuration files, and code snippets inside documentation:** English only.
- **Runtime messages:** System logs in English. User-visible strings go through localization modules where relevant.

## 2. File Layout & Imports

```python
"""Module summary in one sentence."""

# Standard library imports
from pathlib import Path
from typing import Iterable

# Third-party imports
from fastapi import APIRouter

# Internal imports
from vibe_photos.core.detector import detect

DEFAULT_BATCH_SIZE = 16
```

- Order imports: standard library → third-party → local modules.
- No wildcard imports.
- Keep module docstrings concise and informative.

## 3. Programming Patterns

- Prefer pure functions; introduce classes only when state must persist.
- Use `dataclass` or `TypedDict` to structure responses; avoid anonymous dictionaries.
- Guard clause error handling:

  ```python
  if not image_path.exists():
      logger.warning("image_missing", extra={"path": str(image_path)})
      return Result.err("image_not_found")
  ```

- All public functions require type hints and docstrings describing arguments, return values, and side effects.

## 4. Logging & Telemetry

- Use the shared logger factory in `src/utils/logging.py`.
- Log records must include structured metadata (`extra={"asset_id": str(asset_id)}`).
- No print statements except inside CLI entrypoints.
- Emit correlation IDs for request-scoped operations (store in `contextvars`).

## 5. Testing Expectations

- Minimum coverage: 80% for `src/core` and `src/api` combined.
- Always write tests before or alongside implementations.
- Use pytest markers: `@pytest.mark.integration`, `@pytest.mark.performance`, etc.
- Provide CLI contract tests using Typer's `CliRunner`.

## 6. Async & Concurrency

- Prefer `async def` for IO-bound operations; wrap blocking calls in thread executors.
- Never block event loops with synchronous heavy work—delegate to worker pools.
- When mixing sync/async modules, expose both sync and async entrypoints where appropriate.

## 7. Data & Config Handling

- Sensitive data must be read from environment variables or secrets files (even if currently mocked).

## 8. Git Hygiene

- Commit messages follow `type(scope): summary` (e.g., `feat(detector): add confidence calibration`).
- Group related changes; do not mix refactors with feature work in one commit.
- Update documentation and changelog fragments in the same commit as the code when possible.

## 10. Anti-Patterns to Avoid

- Long functions (>80 lines). Break into smaller helpers.
- Hidden global state (use dependency injection or explicit parameters).
- Silent exception swallowing; always log and rethrow or return a typed error.
- Duplicated logic across CLI/API/UI—centralize in `src/core` services.

Respect these rules to keep the codebase predictable for every coding AI that follows you.
