"""Content and perceptual hashing helpers for images."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import xxhash
from PIL import Image

from utils.logging import get_logger

LOGGER = get_logger(__name__)

CONTENT_HASH_ALGO: Final[str] = "xxhash64-v1"
PHASH_ALGO: Final[str] = "phash64-v2"

_PHASH_SIZE: Final[int] = 32
_PHASH_REDUCED_SIZE: Final[int] = 8
_PHASH_DCT_MATRIX: np.ndarray | None = None


def _get_dct_matrix(size: int) -> np.ndarray:
    """Return a cached DCT-II transform matrix of the given size."""

    global _PHASH_DCT_MATRIX
    if _PHASH_DCT_MATRIX is not None and _PHASH_DCT_MATRIX.shape == (size, size):
        return _PHASH_DCT_MATRIX

    n = np.arange(size, dtype=np.float64)
    k = n[:, None]
    factor = np.pi / size

    mat = np.cos((2.0 * n + 1.0) * k * factor)
    mat[0, :] *= np.sqrt(1.0 / size)
    mat[1:, :] *= np.sqrt(2.0 / size)

    _PHASH_DCT_MATRIX = mat
    return mat


def compute_content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the 64-bit content hash for a file.

    The implementation uses ``xxhash.xxh64`` over the raw file bytes and
    returns a 16-character hexadecimal string, matching the M1 blueprint.

    Args:
        path: Path to the file whose content should be hashed.
        chunk_size: Size of the read buffer in bytes.

    Returns:
        Content hash as a 16-character lowercase hexadecimal string.
    """

    hasher = xxhash.xxh64()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    digest = hasher.intdigest()
    return f"{digest:016x}"


def compute_perceptual_hash(image: Image.Image) -> str:
    """Compute the 64-bit perceptual hash for an image.

    The implementation follows the M1 blueprint:

    - Convert to grayscale and resize to 32×32 pixels.
    - Apply a 2D DCT (type II) over the 32×32 matrix.
    - Take the top-left 8×8 low-frequency block (including the DC term).
    - Compute the median of these 64 coefficients.
    - Set a bit to 1 when ``coeff > median``, else 0, yielding a 64-bit value
      encoded as a 16-character hexadecimal string.

    Args:
        image: PIL Image instance to hash.

    Returns:
        Perceptual hash as a 16-character lowercase hexadecimal string.
    """

    # Normalize to grayscale 32×32.
    resample = getattr(Image, "Resampling", Image).LANCZOS
    gray = image.convert("L").resize((_PHASH_SIZE, _PHASH_SIZE), resample=resample)
    pixels = np.asarray(gray, dtype=np.float64)

    dct_mat = _get_dct_matrix(_PHASH_SIZE)
    # 2D DCT: C * A * C^T
    dct_rows = dct_mat @ pixels
    dct = dct_rows @ dct_mat.T

    low_freq = dct[:_PHASH_REDUCED_SIZE, :_PHASH_REDUCED_SIZE]
    median = np.median(low_freq)

    bits = (low_freq > median).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return f"{value:016x}"


def hamming_distance_phash(a_hex: str, b_hex: str) -> int:
    """Compute the Hamming distance between two 64-bit pHash values."""

    try:
        a_int = int(a_hex, 16)
        b_int = int(b_hex, 16)
    except ValueError:
        LOGGER.error("phash_hex_parse_error", extra={"a": a_hex, "b": b_hex})
        return 64

    return int((a_int ^ b_int).bit_count())


__all__ = ["CONTENT_HASH_ALGO", "PHASH_ALGO", "compute_content_hash", "compute_perceptual_hash", "hamming_distance_phash"]
