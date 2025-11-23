"""Shared thumbnail helpers for pipeline and task queue consumers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.Image import Resampling

from utils.logging import get_logger

LOGGER = get_logger(__name__, extra={"component": "thumbnailing"})


def _get_resample_filter() -> Resampling:
    """Return the preferred resample filter compatible with the current Pillow."""

    return Resampling.LANCZOS


def build_thumbnail_image(image: Image.Image, max_side: int) -> Image.Image:
    """Produce a resized copy of an image constrained to ``max_side`` pixels."""

    safe_side = max(1, int(max_side))
    resized = image.copy()
    resized.thumbnail((safe_side, safe_side), resample=_get_resample_filter())
    return resized


def save_thumbnail(image: Image.Image, output_path: Path, max_side: int, quality: int) -> None:
    """Resize ``image`` and persist it to ``output_path`` with the given quality."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized = build_thumbnail_image(image, max_side)

    try:
        resized.save(output_path, format="JPEG", quality=quality)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error(
            "thumbnail_save_error",
            extra={"path": str(output_path), "max_side": max_side, "quality": quality, "error": str(exc)},
        )
        raise


__all__ = ["build_thumbnail_image", "save_thumbnail"]
