"""CLI entrypoint for the M1 processing pipeline.

This single-process CLI scans album roots, populates the primary database,
and writes all versioned cache artifacts. It is the recommended entrypoint
for local runs of M1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from utils.logging import get_logger
from vibe_photos.artifact_store import ArtifactManager
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import open_projection_session
from vibe_photos.hasher import compute_content_hash
from vibe_photos.ml.siglip_blip import SiglipBlipDetector
from vibe_photos.pipeline import PreprocessingPipeline
from vibe_photos.preprocessing import ensure_preprocessing_artifacts


LOGGER = get_logger(__name__)


def _apply_cli_overrides(settings: Settings, batch_size: Optional[int], device: Optional[str]) -> Settings:
    """Apply CLI overrides for batch size and device to the settings."""

    if batch_size is not None and batch_size > 0:
        settings.models.embedding.batch_size = batch_size
        settings.models.caption.batch_size = batch_size

    if device:
        settings.models.embedding.device = device
        settings.models.caption.device = device

    return settings


def ensure_artifacts_for_image(
    image_path: Path,
    projection_db: Path,
    settings: Settings,
    *,
    image_id: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> str:
    """Create preprocessing artifacts for a single image using shared steps."""

    resolved_id = image_id or compute_content_hash(image_path)
    root = artifact_root or projection_db.parent / "artifacts"
    detector = SiglipBlipDetector(settings=settings)

    with open_projection_session(projection_db) as projection_session:
        manager = ArtifactManager(session=projection_session, root=root)
        image = Image.open(image_path).convert("RGB")
        ensure_preprocessing_artifacts(
            image_id=resolved_id,
            image=image,
            image_path=image_path,
            settings=settings,
            manager=manager,
            detector=detector,
        )

    return resolved_id


def main(
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
    image_path: Optional[Path] = typer.Option(
        None,
        "--image-path",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        help="Process a single image into the projection cache using shared preprocessing steps.",
    ),
    image_id: Optional[str] = typer.Option(
        None,
        "--image-id",
        help="Optional image_id to use when --image-path is set; defaults to the content hash.",
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
    """Run the M1 processing pipeline for one or more album roots."""

    settings = load_settings()
    settings = _apply_cli_overrides(settings, batch_size=batch_size, device=device)

    if image_path:
        resolved_id = ensure_artifacts_for_image(
            image_path=image_path,
            projection_db=cache_db,
            settings=settings,
            image_id=image_id,
        )
        LOGGER.info(
            "single_image_process_complete",
            extra={"image_id": resolved_id, "image_path": str(image_path), "projection_db": str(cache_db)},
        )
        return

    pipeline = PreprocessingPipeline(settings=settings)
    pipeline.run(roots=root, primary_db_path=db, projection_db_path=cache_db)


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main", "ensure_artifacts_for_image"]

