"""CLI entrypoint for the M1 preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.pipeline import PreprocessingPipeline


LOGGER = get_logger(__name__)

app = typer.Typer(help="Run the Vibe Photos M1 preprocessing pipeline.")


def _apply_cli_overrides(settings: Settings, batch_size: Optional[int], device: Optional[str]) -> Settings:
    """Apply CLI overrides for batch size and device to the settings."""

    if batch_size is not None and batch_size > 0:
        settings.models.embedding.batch_size = batch_size
        settings.models.caption.batch_size = batch_size

    if device:
        settings.models.embedding.device = device
        settings.models.caption.device = device

    return settings


@app.command("run")
def run(
    root: list[Path] = typer.Option(
        ...,
        "--root",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Album root directory to scan. May be specified multiple times.",
    ),
    db: Path = typer.Option(
        Path("data/index.db"),
        "--db",
        help="Path to the primary operational SQLite database.",
    ),
    cache_db: Path = typer.Option(
        Path("cache/index.db"),
        "--cache-db",
        help="Path to the projection SQLite database for model outputs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override the model batch size configured in settings.yaml.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Override the model device from settings.yaml, for example cpu, cuda, or mps.",
    ),
) -> None:
    """Run the M1 preprocessing pipeline for one or more album roots."""

    settings = load_settings()
    settings = _apply_cli_overrides(settings, batch_size=batch_size, device=device)

    pipeline = PreprocessingPipeline(settings=settings)
    pipeline.run(roots=root, primary_db_path=db, projection_db_path=cache_db)


def main() -> None:
    """Entrypoint used when invoking the module as a script."""

    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]

