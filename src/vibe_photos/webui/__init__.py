"""Flask Web UI for inspecting M1 preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from sqlalchemy import and_, func, select

from utils.logging import get_logger
from vibe_photos.db import Image, ImageCaption, ImageScene, open_primary_session


LOGGER = get_logger(__name__)

app = Flask(__name__)


def _get_primary_db_path() -> Path:
    return Path("data/index.db")


@app.route("/")
def index() -> Any:
    """Redirect to the images list."""

    return redirect(url_for("images"))


@app.route("/images")
def images() -> Any:
    """List images with basic filters and pagination."""

    scene_type = request.args.get("scene_type") or None
    has_text = request.args.get("has_text")
    has_person = request.args.get("has_person")
    is_screenshot = request.args.get("is_screenshot")
    is_document = request.args.get("is_document")

    page = max(1, int(request.args.get("page", "1")))
    page_size = max(1, min(64, int(request.args.get("page_size", "24"))))
    offset = (page - 1) * page_size

    filters = [Image.status == "active"]
    if scene_type:
        filters.append(ImageScene.scene_type == scene_type)

    def _flag_clause(value: Optional[str], column) -> None:
        if value is None or value == "":
            return
        if value.lower() in {"1", "true", "yes"}:
            filters.append(column.is_(True))
        elif value.lower() in {"0", "false", "no"}:
            filters.append(column.is_(False))

    _flag_clause(has_text, ImageScene.has_text)
    _flag_clause(has_person, ImageScene.has_person)
    _flag_clause(is_screenshot, ImageScene.is_screenshot)
    _flag_clause(is_document, ImageScene.is_document)

    primary_path = _get_primary_db_path()

    with open_primary_session(primary_path) as session:
        base_query = (
            select(ImageScene.image_id)
            .join(Image, Image.image_id == ImageScene.image_id)
            .where(and_(*filters))
        )
        total_stmt = select(func.count()).select_from(base_query.subquery())
        total = int(session.execute(total_stmt).scalar_one())

        query_stmt = (
            select(
                ImageScene.image_id,
                ImageScene.scene_type,
                ImageScene.has_text,
                ImageScene.has_person,
                ImageScene.is_screenshot,
                ImageScene.is_document,
            )
            .join(Image, Image.image_id == ImageScene.image_id)
            .where(and_(*filters))
            .order_by(ImageScene.image_id)
            .limit(page_size)
            .offset(offset)
        )
        rows = session.execute(query_stmt).all()

    items: List[Dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "image_id": row["image_id"],
                "scene_type": row["scene_type"],
                "has_text": bool(row["has_text"]),
                "has_person": bool(row["has_person"]),
                "is_screenshot": bool(row["is_screenshot"]),
                "is_document": bool(row["is_document"]),
            }
        )

    return render_template(
        "images.html",
        images=items,
        page=page,
        page_size=page_size,
        total=total,
        scene_type=scene_type or "",
        has_text=has_text or "",
        has_person=has_person or "",
        is_screenshot=is_screenshot or "",
        is_document=is_document or "",
    )


@app.route("/image/<image_id>")
def image_detail(image_id: str) -> Any:
    """Detail page for a single image, keyed by logical image identifier."""

    primary_path = _get_primary_db_path()

    with open_primary_session(primary_path) as session:
        image_row = session.get(Image, image_id)
        if image_row is None:
            abort(404)

        scene_row = session.execute(select(ImageScene).where(ImageScene.image_id == image_id)).scalar_one_or_none()

        caption_row = (
            session.execute(
                select(ImageCaption.caption, ImageCaption.model_name).where(ImageCaption.image_id == image_id).order_by(ImageCaption.model_name)
            )
        ).first()

    primary_path_str = image_row.primary_path
    logical_name = Path(primary_path_str).name

    scene_info: Dict[str, Any] | None = None
    if scene_row is not None:
        scene_info = {
            "scene_type": scene_row.scene_type,
            "scene_confidence": scene_row.scene_confidence,
            "has_text": bool(scene_row.has_text),
            "has_person": bool(scene_row.has_person),
            "is_screenshot": bool(scene_row.is_screenshot),
            "is_document": bool(scene_row.is_document),
            "classifier_name": scene_row.classifier_name,
            "classifier_version": scene_row.classifier_version,
        }

    caption_text: Optional[str] = None
    caption_model: Optional[str] = None
    if caption_row is not None:
        caption_text = caption_row.caption
        caption_model = caption_row.model_name

    return render_template(
        "image_detail.html",
        image_id=image_id,
        logical_name=logical_name,
        size_bytes=image_row.size_bytes,
        mtime=image_row.mtime,
        width=image_row.width,
        height=image_row.height,
        status=image_row.status,
        scene=scene_info,
        caption=caption_text,
        caption_model=caption_model,
        phash=image_row.phash,
        phash_algo=image_row.phash_algo,
    )


@app.route("/thumbnail/<image_id>")
def image_thumbnail(image_id: str) -> Any:
    """Serve an image thumbnail or original content by logical identifier."""

    primary_path = _get_primary_db_path()

    with open_primary_session(primary_path) as session:
        row = session.execute(
            select(Image.primary_path).where(and_(Image.image_id == image_id, Image.status == "active"))
        ).first()
        if row is None:
            abort(404)

        path_str = row.primary_path

    try:
        return send_file(path_str)
    except Exception as exc:
        LOGGER.error("thumbnail_send_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
        abort(404)


__all__ = ["app"]
