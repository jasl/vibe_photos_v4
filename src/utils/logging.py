"""Shared logging configuration and logger factory."""

from __future__ import annotations

import logging
from typing import Any, Dict


def _configure_root_logger() -> None:
    """Configure the root logger with a basic handler if needed."""

    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def get_logger(name: str, extra: Dict[str, Any] | None = None) -> logging.LoggerAdapter:
    """Return a structured logger adapter for the given name.

    The first call configures a simple root handler. Callers can pass a base
    ``extra`` mapping that is attached to every log record emitted through the
    returned adapter.
    """

    _configure_root_logger()
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, extra or {})


__all__ = ["get_logger"]

