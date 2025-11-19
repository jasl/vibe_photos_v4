"""Shared logging configuration and logger factory."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_ROOT = _PROJECT_ROOT / "log"


class _StructuredFormatter(logging.Formatter):
    """Formatter that renders records as single-line JSON objects (JSONL-friendly)."""

    def format(self, record: logging.LogRecord) -> str:
        # Compute non-standard attributes added via ``extra=``.
        standard_keys = logging.makeLogRecord({}).__dict__.keys()
        extras: Dict[str, Any] = {
            key: value for key, value in record.__dict__.items() if key not in standard_keys and key != "stack_info"
        }

        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if extras:
            payload.update(extras)

        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            # Fallback to string representation when an extra field is not JSON-serializable.
            safe_payload: Dict[str, Any] = {
                key: (str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value)
                for key, value in payload.items()
            }
            return json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)


class _ConsoleFormatter(logging.Formatter):
    """Formatter for console output that renders ``extra`` fields as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        standard_keys = logging.makeLogRecord({}).__dict__.keys()
        ignore_keys = set(standard_keys) | {"stack_info", "asctime", "message"}
        extras: Dict[str, Any] = {key: value for key, value in record.__dict__.items() if key not in ignore_keys}

        if not extras:
            return base

        parts = [f"{key}={value!r}" for key, value in sorted(extras.items())]
        return f"{base} | " + " ".join(parts)


def _configure_root_logger() -> None:
    """Configure the root logger with console and file handlers if needed."""

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.INFO)

    # Human-readable console output without JSON blobs.
    console_handler = logging.StreamHandler()
    console_formatter = _ConsoleFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # Structured JSON-ish logs in a rotating file under log/.
    try:
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_ROOT / "vibe_photos.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = _StructuredFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)
    except Exception:
        # Do not fail the application if file logging cannot be initialized.
        pass


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
