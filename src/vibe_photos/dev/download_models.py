"""CLI to pre-download configured model checkpoints for offline runs."""

from __future__ import annotations

import typer
from huggingface_hub import snapshot_download

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings

LOGGER = get_logger(__name__)


def _resolve_models(settings: Settings, include_detection: bool) -> list[str]:
    """Return a sorted list of unique model identifiers to download."""

    model_names = {
        settings.models.embedding.resolved_model_name(),
        settings.models.caption.resolved_model_name(),
    }

    detection_cfg = settings.models.detection
    if include_detection and detection_cfg.model_name:
        model_names.add(detection_cfg.model_name)

    return sorted(model_names)


def _download_model(repo_id: str) -> None:
    """Download a single model repository to the local Hugging Face cache."""

    LOGGER.info("model_download_start", extra={"model_name": repo_id})
    cache_path = snapshot_download(repo_id=repo_id)
    LOGGER.info("model_download_complete", extra={"model_name": repo_id, "cache_path": cache_path})


def main(
    include_detection: bool | None = typer.Option(
        None,
        "--include-detection/--skip-detection",
        help=(
            "Whether to download detection model weights. Defaults to"
            " config.models.detection.enabled."
        ),
    ),
) -> None:
    """Download configured Hugging Face model checkpoints ahead of pipeline runs."""

    settings = load_settings()
    detection_cfg = settings.models.detection
    download_detection = detection_cfg.enabled if include_detection is None else include_detection

    LOGGER.info(
        "model_download_prepare",
        extra={
            "embedding_model": settings.models.embedding.resolved_model_name(),
            "caption_model": settings.models.caption.resolved_model_name(),
            "include_detection": download_detection,
            "detection_model": detection_cfg.model_name if download_detection else None,
        },
    )

    model_names = _resolve_models(settings, include_detection=download_detection)
    for model_name in model_names:
        _download_model(model_name)

    LOGGER.info("model_download_finished", extra={"downloaded_models": model_names})


if __name__ == "__main__":
    typer.run(main)


__all__ = ["main"]
