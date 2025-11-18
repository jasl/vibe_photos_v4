"""Flask Web UI for inspecting M1 preprocessing outputs."""

from __future__ import annotations

from typing import Any

from flask import Flask, redirect, url_for

from utils.logging import get_logger


LOGGER = get_logger(__name__)

app = Flask(__name__)


@app.route("/")
def index() -> Any:
    """Redirect to the images list."""

    return redirect(url_for("images"))


@app.route("/images")
def images() -> str:
    """Placeholder route listing processed images."""

    return "Images list view is not implemented yet."


@app.route("/image/<image_id>")
def image_detail(image_id: str) -> str:
    """Placeholder route showing a single image detail by logical identifier."""

    return f"Detail view for image_id={image_id} is not implemented yet."


@app.route("/thumbnail/<image_id>")
def image_thumbnail(image_id: str) -> str:
    """Placeholder route serving thumbnails by logical identifier."""

    return f"Thumbnail for image_id={image_id} is not implemented yet."


__all__ = ["app"]
