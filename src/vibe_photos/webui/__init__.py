"""Flask Web UI for inspecting M1 preprocessing outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from sqlalchemy import and_, func, or_, select

from utils.logging import get_logger
from vibe_photos.artifact_store import ArtifactSpec
from vibe_photos.config import load_settings
from vibe_photos.db import ArtifactRecord, Image, ImageCaption, ImageNearDuplicate, ImageRegion, ImageScene
from vibe_photos.db import open_primary_session, open_projection_session


LOGGER = get_logger(__name__)

app = Flask(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _PROJECT_ROOT / "data"
_CACHE_ROOT = _PROJECT_ROOT / "cache"


def _get_primary_db_path() -> Path:
    return _DATA_ROOT / "index.db"


def _get_projection_db_path() -> Path:
    return _DATA_ROOT / "projection.db"


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
    has_duplicates = request.args.get("has_duplicates")
    hide_duplicates = request.args.get("hide_duplicates")
    if not request.args:
        hide_duplicates = "1"
    region_label = request.args.get("region_label") or ""

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
        base_query = select(Image.image_id).where(Image.status == "active")
        if filters:
            base_query = base_query.join(ImageScene, ImageScene.image_id == Image.image_id, isouter=True).where(
                and_(*filters)
            )
        if region_label:
            base_query = (
                base_query.join(ImageRegion, ImageRegion.image_id == Image.image_id).where(
                    ImageRegion.refined_label == region_label
                )
            )
        if hide_duplicates and hide_duplicates.lower() in {"1", "true", "yes"}:
            dup_subq = select(ImageNearDuplicate.duplicate_image_id)
            base_query = base_query.where(~Image.image_id.in_(dup_subq))
        if has_duplicates and has_duplicates.lower() in {"1", "true", "yes"}:
            base_query = (
                base_query.join(
                    ImageNearDuplicate,
                    or_(
                        ImageNearDuplicate.anchor_image_id == Image.image_id,
                        ImageNearDuplicate.duplicate_image_id == Image.image_id,
                    ),
                ).distinct(ImageScene.image_id)
            )
        total_stmt = select(func.count()).select_from(base_query.subquery())
        total = int(session.execute(total_stmt).scalar_one())

        query_stmt = select(
            Image.image_id,
            ImageScene.scene_type,
            ImageScene.has_text,
            ImageScene.has_person,
            ImageScene.is_screenshot,
            ImageScene.is_document,
        ).join(ImageScene, ImageScene.image_id == Image.image_id, isouter=True).where(Image.status == "active")
        if filters:
            query_stmt = query_stmt.where(and_(*filters))
        if region_label:
            query_stmt = query_stmt.join(ImageRegion, ImageRegion.image_id == Image.image_id).where(
                ImageRegion.refined_label == region_label
            )
            query_stmt = query_stmt.distinct(Image.image_id)
        if hide_duplicates and hide_duplicates.lower() in {"1", "true", "yes"}:
            dup_subq = select(ImageNearDuplicate.duplicate_image_id)
            query_stmt = query_stmt.where(~Image.image_id.in_(dup_subq))
        if has_duplicates and has_duplicates.lower() in {"1", "true", "yes"}:
            query_stmt = query_stmt.join(
                ImageNearDuplicate,
                or_(
                    ImageNearDuplicate.anchor_image_id == Image.image_id,
                    ImageNearDuplicate.duplicate_image_id == Image.image_id,
                ),
            ).distinct(Image.image_id)
        query_stmt = query_stmt.order_by(Image.image_id).limit(page_size).offset(offset)
        rows = session.execute(query_stmt).mappings().all()

        dup_ids = {
            row.duplicate_image_id
            for row in session.execute(select(ImageNearDuplicate.duplicate_image_id))
        }

    items: List[Dict[str, Any]] = []
    for row in rows:
        image_id = row["image_id"]
        items.append(
            {
                "image_id": image_id,
                "scene_type": row["scene_type"],
                "has_text": bool(row["has_text"]),
                "has_person": bool(row["has_person"]),
                "is_screenshot": bool(row["is_screenshot"]),
                "is_document": bool(row["is_document"]),
                "is_duplicate": image_id in dup_ids,
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
        has_duplicates=has_duplicates or "",
        hide_duplicates=hide_duplicates or "",
        region_label=region_label,
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

        region_rows = (
            session.execute(
                select(ImageRegion).where(ImageRegion.image_id == image_id).order_by(ImageRegion.region_index)
            )
            .scalars()
            .all()
        )

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

    # Near-duplicate images (both as anchor and as duplicate).
    near_duplicates: List[Dict[str, Any]] = []
    with open_primary_session(primary_path) as session:
        anchor_rows = (
            session.execute(
                select(ImageNearDuplicate).where(ImageNearDuplicate.anchor_image_id == image_id)
            )
            .scalars()
            .all()
        )
        duplicate_rows = (
            session.execute(
                select(ImageNearDuplicate).where(ImageNearDuplicate.duplicate_image_id == image_id)
            )
            .scalars()
            .all()
        )

        related_ids: Dict[str, Dict[str, Any]] = {}

        for row in anchor_rows:
            other_id = row.duplicate_image_id
            related_ids[other_id] = {
                "image_id": other_id,
                "role": "anchor",
                "phash_distance": row.phash_distance,
                "created_at": row.created_at,
            }

        for row in duplicate_rows:
            other_id = row.anchor_image_id
            existing = related_ids.get(other_id)
            # Keep the smallest distance if multiple entries exist.
            if existing is not None and existing.get("phash_distance", 1e9) <= row.phash_distance:
                continue
            related_ids[other_id] = {
                "image_id": other_id,
                "role": "duplicate",
                "phash_distance": row.phash_distance,
                "created_at": row.created_at,
            }

        for other_id, meta in related_ids.items():
            other_image = session.get(Image, other_id)
            if other_image is None:
                continue
            meta["status"] = other_image.status
            meta["primary_path"] = other_image.primary_path
            near_duplicates.append(meta)

    near_duplicates.sort(key=lambda item: (item.get("phash_distance", 0), item["image_id"]))

    gps_latitude: float | None = None
    gps_longitude: float | None = None
    metadata_path = Path("cache") / "images" / "metadata" / f"{image_id}.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            gps_payload = payload.get("gps") or {}
            lat_val = gps_payload.get("latitude")
            lon_val = gps_payload.get("longitude")
            if isinstance(lat_val, (int, float)):
                gps_latitude = float(lat_val)
            if isinstance(lon_val, (int, float)):
                gps_longitude = float(lon_val)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error(
                "metadata_load_error",
                extra={"image_id": image_id, "path": str(metadata_path), "error": str(exc)},
            )

    regions: List[Dict[str, Any]] = []
    if region_rows:
        settings = load_settings()
        detection_cfg = settings.models.detection
        area_gamma = float(detection_cfg.primary_area_gamma)
        center_penalty = float(detection_cfg.primary_center_penalty)

        for region in region_rows:
            width = max(0.0, float(region.x_max) - float(region.x_min))
            height = max(0.0, float(region.y_max) - float(region.y_min))
            area = max(1e-6, width * height)

            cx = 0.5 * (float(region.x_min) + float(region.x_max))
            cy = 0.5 * (float(region.y_min) + float(region.y_max))
            distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

            area_weight = area**area_gamma
            center_weight = 1.0 - center_penalty * distance
            if center_weight < 0.0:
                center_weight = 0.0

            priority = float(region.detector_score) * area_weight * center_weight

            regions.append(
                {
                    "index": region.region_index,
                    "x_min": region.x_min,
                    "y_min": region.y_min,
                    "x_max": region.x_max,
                    "y_max": region.y_max,
                    "detector_label": region.detector_label,
                    "detector_score": region.detector_score,
                    "refined_label": region.refined_label,
                    "refined_score": region.refined_score,
                    "backend": region.backend,
                    "model_name": region.model_name,
                    "priority": priority,
                }
            )

    return render_template(
        "image_detail.html",
        image_id=image_id,
        logical_name=logical_name,
        size_bytes=image_row.size_bytes,
        mtime=image_row.mtime,
        width=image_row.width,
        height=image_row.height,
        status=image_row.status,
        exif_datetime=image_row.exif_datetime,
        camera_model=image_row.camera_model,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        scene=scene_info,
        caption=caption_text,
        caption_model=caption_model,
        near_duplicates=near_duplicates,
        regions=regions,
        phash=image_row.phash,
        phash_algo=image_row.phash_algo,
    )


@app.route("/thumbnail/<image_id>")
def image_thumbnail(image_id: str) -> Any:
    """Serve an image thumbnail or original content by logical identifier."""

    variant = request.args.get("variant") or "large"
    settings = load_settings()
    if variant == "small":
        thumb_spec = ArtifactSpec(
            artifact_type="thumbnail_small", model_name="pil", params={"size": min(256, settings.pipeline.thumbnail_size)}
        )
    else:
        thumb_spec = ArtifactSpec(
            artifact_type="thumbnail_large", model_name="pil", params={"size": max(1024, settings.pipeline.thumbnail_size)}
        )

    primary_path = _get_primary_db_path()
    projection_path = _get_projection_db_path()

    with open_primary_session(primary_path) as session:
        row = session.execute(
            select(Image.primary_path, Image.phash).where(and_(Image.image_id == image_id, Image.status == "active"))
        ).first()
        if row is None:
            abort(404)

        path_str = row.primary_path
        phash_hex = row.phash

    thumb_path: Path | None = None
    with open_projection_session(projection_path) as projection_session:
        artifact_row = projection_session.execute(
            select(ArtifactRecord.storage_path).where(
                ArtifactRecord.image_id == image_id,
                ArtifactRecord.artifact_type == thumb_spec.artifact_type,
                ArtifactRecord.version_key == thumb_spec.version_key,
                ArtifactRecord.status == "complete",
            )
        ).scalar_one_or_none()
        if artifact_row is not None:
            candidate = Path(artifact_row)
            if not candidate.is_absolute():
                candidate = _PROJECT_ROOT / candidate
            thumb_path = candidate if candidate.exists() else None

    if thumb_path is None:
        legacy_cache_path = _CACHE_ROOT / "images" / "thumbnails" / f"{image_id}.jpg"
        thumb_path = legacy_cache_path if legacy_cache_path.exists() else None

    if thumb_path is not None:
        try:
            return send_file(thumb_path)
        except FileNotFoundError as exc:
            LOGGER.warning(
                "thumbnail_file_missing",
                extra={"image_id": image_id, "path": str(thumb_path), "error": str(exc)},
            )
        except Exception as exc:
            LOGGER.error(
                "thumbnail_send_error",
                extra={"image_id": image_id, "path": str(thumb_path), "error": str(exc)},
            )

    # Backward-compatibility: fall back to legacy pHash-keyed thumbnails if present.
    if phash_hex:
        legacy_thumb_path = _CACHE_ROOT / "images" / "thumbnails" / f"{phash_hex}.jpg"
        if legacy_thumb_path.exists():
            try:
                return send_file(legacy_thumb_path)
            except FileNotFoundError as exc:
                LOGGER.warning(
                    "thumbnail_legacy_file_missing",
                    extra={"image_id": image_id, "path": str(legacy_thumb_path), "error": str(exc)},
                )
            except Exception as exc:
                LOGGER.error(
                    "thumbnail_send_error",
                    extra={"image_id": image_id, "path": str(legacy_thumb_path), "error": str(exc)},
                )

    try:
        return send_file(path_str)
    except FileNotFoundError as exc:
        LOGGER.warning("image_file_missing", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
        abort(404, description="Image content not found on disk. Run the preprocessing pipeline again or rescan your album roots.")
    except Exception as exc:
        LOGGER.error("thumbnail_send_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
        abort(500, description="Unexpected error while serving thumbnail.")


__all__ = ["app"]
