"""Content and perceptual hashing helpers for images."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import xxhash

from utils.logging import get_logger


LOGGER = get_logger(__name__)

CONTENT_HASH_ALGO: Final[str] = "xxhash64-v1"


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


def compute_perceptual_hash_placeholder() -> str:
    """Placeholder for the DCT-based perceptual hash implementation.

    The M1 blueprint specifies a 64-bit pHash computed over a 32×32 grayscale
    image using a 2D DCT. The concrete implementation is intentionally left
    for a dedicated iteration so that it can be validated and benchmarked
    in isolation.
    """

    raise NotImplementedError("Perceptual hashing is not implemented yet.")


__all__ = ["CONTENT_HASH_ALGO", "compute_content_hash", "compute_perceptual_hash_placeholder"]

