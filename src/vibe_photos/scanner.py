"""Filesystem scanner for album roots."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".webp",
        ".gif",
    }
)


@dataclass(frozen=True)
class FileInfo:
    """Lightweight file metadata for scanning results."""

    path: Path
    size_bytes: int
    mtime: float


def scan_roots(roots: Sequence[Path], extensions: frozenset[str] | None = None) -> Iterator[FileInfo]:
    """Recursively scan album roots and yield image file descriptors.

    Args:
        roots: Album root directories to scan.
        extensions: Allowed file extensions, lowercased and including the leading dot.
            When omitted, :data:`DEFAULT_IMAGE_EXTENSIONS` is used.

    Yields:
        FileInfo instances for each discovered file that matches the extension filter.
    """

    allowed = extensions or DEFAULT_IMAGE_EXTENSIONS

    for root in roots:
        if not root.exists() or not root.is_dir():
            LOGGER.warning("scan_root_missing", extra={"root": str(root)})
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in allowed:
                continue

            stat = path.stat()
            yield FileInfo(path=path, size_bytes=stat.st_size, mtime=stat.st_mtime)


__all__ = ["FileInfo", "DEFAULT_IMAGE_EXTENSIONS", "scan_roots"]

