"""Lightweight unit tests for pipeline helper functions.

These cover pure, dependency-free helpers to guard refactors and typing changes
without requiring heavy model downloads or database access.
"""

from __future__ import annotations

from PIL import Image

from vibe_photos.pipeline import ImageMeta, PreprocessingPipeline, _extract_exif_and_gps


def test_phash_distance_matches_bitcount() -> None:
    """_phash_distance should equal the XOR bitcount between two hashes."""

    lhs = int("a3f", 16)  # 0b101000111111
    rhs = int("a0f", 16)  # 0b101000001111
    expected = (lhs ^ rhs).bit_count()

    assert PreprocessingPipeline._phash_distance(lhs, rhs) == expected


def test_select_canonical_image_prefers_area_then_size_then_id() -> None:
    """Canonical image selection favors larger area, then size_bytes, then id lexicographically."""

    meta: dict[str, ImageMeta] = {
        "a": {"image_id": "a", "phash_int": 1, "mtime": 0.0, "width": 100, "height": 100, "size_bytes": 500},
        "b": {"image_id": "b", "phash_int": 2, "mtime": 0.0, "width": 120, "height": 90, "size_bytes": 600},
        "c": {"image_id": "c", "phash_int": 3, "mtime": 0.0, "width": 120, "height": 90, "size_bytes": 700},
    }

    # Same area for b and c; c wins due to larger size_bytes.
    canonical = PreprocessingPipeline._select_canonical_image(["a", "b", "c"], meta)

    assert canonical == "c"


def test_extract_exif_and_gps_handles_missing_exif() -> None:
    """Images without EXIF should return (None, None, None) without raising."""

    image = Image.new("RGB", (10, 10), color="red")

    exif_dt, camera_model, gps = _extract_exif_and_gps(image)

    assert exif_dt is None
    assert camera_model is None
    assert gps is None

