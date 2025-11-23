"""Flask Web UI for inspecting preprocessing + label-layer outputs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.artifact_store import ArtifactSpec
from vibe_photos.config import load_settings
from vibe_photos.db import (
    ArtifactRecord,
    Image,
    ImageCaption,
    ImageNearDuplicate,
    Label,
    LabelAssignment,
    Region,
    open_primary_session,
    open_projection_session,
    sqlite_path_from_target,
)
from vibe_photos.labels.scene_schema import (
    ATTRIBUTE_LABEL_KEYS,
    normalize_scene_filter,
    scene_type_from_label_key,
)

LOGGER = get_logger(__name__)

app = Flask(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _get_primary_db_target() -> str | Path:
    settings = load_settings()
    return settings.databases.primary_url


def _get_projection_db_path() -> Path:
    settings = load_settings()
    return sqlite_path_from_target(settings.databases.projection_url)


def _get_cache_root() -> Path:
    return _get_projection_db_path().parent


def _is_checked(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _load_label_ids(session: Session, keys: set[str]) -> dict[str, int]:
    key_list = [key for key in keys if key]
    if not key_list:
        return {}
    stmt = select(Label.key, Label.id).where(Label.key.in_(tuple(key_list)))
    return {row.key: row.id for row in session.execute(stmt)}


def _assignment_exists_clause(label_id: int, label_space: str) -> Any:
    return exists().where(
        and_(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id == Image.image_id,
            LabelAssignment.label_id == label_id,
            LabelAssignment.label_space_ver == label_space,
        )
    )


@app.route("/")
def index() -> Any:
    """Redirect to the images list."""

    return redirect(url_for("images"))


@app.route("/images")
def images() -> Any:
    """List images with basic filters and pagination."""

    settings = load_settings()
    scene_space = settings.label_spaces.scene_current

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

    region_filter_ids: set[str] | None = None

    primary_target = _get_primary_db_target()

    with open_primary_session(primary_target) as session:
        label_keys = set(ATTRIBUTE_LABEL_KEYS.values())
        scene_filter_key = normalize_scene_filter(scene_type)
        if scene_filter_key:
            label_keys.add(scene_filter_key)

        label_ids = _load_label_ids(session, label_keys)

        stmt = select(Image.image_id).where(Image.status == "active")

        scene_label_id = label_ids.get(scene_filter_key) if scene_filter_key else None
        if scene_label_id is not None:
            stmt = stmt.where(_assignment_exists_clause(scene_label_id, scene_space))

        attr_filters = [
            (has_person, ATTRIBUTE_LABEL_KEYS["has_person"]),
            (has_text, ATTRIBUTE_LABEL_KEYS["has_text"]),
            (is_screenshot, ATTRIBUTE_LABEL_KEYS["is_screenshot"]),
            (is_document, ATTRIBUTE_LABEL_KEYS["is_document"]),
        ]
        for param_value, label_key in attr_filters:
            if not _is_checked(param_value):
                continue
            label_id = label_ids.get(label_key)
            if label_id is None:
                continue
            stmt = stmt.where(_assignment_exists_clause(label_id, scene_space))

        obj_space = settings.label_spaces.object_current
        if region_label:
            region_targets = session.execute(
                select(LabelAssignment.target_id)
                .join(Label, Label.id == LabelAssignment.label_id)
                .where(
                    LabelAssignment.target_type == "region",
                    LabelAssignment.label_space_ver == obj_space,
                    Label.key == region_label,
                )
            ).scalars()
            region_filter_ids = {str(target_id).split("#", 1)[0] for target_id in region_targets}

        if region_filter_ids is not None:
            if region_filter_ids:
                stmt = stmt.where(Image.image_id.in_(region_filter_ids))
            else:
                return render_template(
                    "images.html",
                    images=[],
                    page=page,
                    page_size=page_size,
                    total=0,
                    scene_type=scene_type or "",
                    has_text=has_text or "",
                    has_person=has_person or "",
                    is_screenshot=is_screenshot or "",
                    is_document=is_document or "",
                    has_duplicates=has_duplicates or "",
                    hide_duplicates=hide_duplicates or "",
                    region_label=region_label,
                )
        if _is_checked(hide_duplicates):
            dup_subq = select(ImageNearDuplicate.duplicate_image_id)
            stmt = stmt.where(~Image.image_id.in_(dup_subq))
        if _is_checked(has_duplicates):
            stmt = stmt.join(
                ImageNearDuplicate,
                or_(
                    ImageNearDuplicate.anchor_image_id == Image.image_id,
                    ImageNearDuplicate.duplicate_image_id == Image.image_id,
                ),
            ).distinct(Image.image_id)

        total_stmt = select(func.count()).select_from(stmt.subquery())
        total = int(session.execute(total_stmt).scalar_one())

        query_stmt = stmt.order_by(Image.image_id).limit(page_size).offset(offset)
        rows: list[dict[str, Any]] = [dict(row) for row in session.execute(query_stmt).mappings().all()]

        dup_ids = set(session.execute(select(ImageNearDuplicate.duplicate_image_id)).scalars().all())

        image_ids = [str(row["image_id"]) for row in rows if row.get("image_id")]
        scenes_by_image: dict[str, dict[str, Any]] = {}
        attributes_by_image: dict[str, dict[str, bool]] = defaultdict(dict)
        objects_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        clusters_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

        if image_ids:
            label_rows = session.execute(
                select(
                    LabelAssignment.target_id,
                    Label.key,
                    Label.level,
                    LabelAssignment.score,
                    LabelAssignment.extra_json,
                    Label.display_name,
                    LabelAssignment.source,
                    LabelAssignment.label_space_ver,
                )
                .join(Label, Label.id == LabelAssignment.label_id)
                .where(
                    LabelAssignment.target_type == "image",
                    LabelAssignment.target_id.in_(image_ids),
                    LabelAssignment.label_space_ver.in_(
                        (
                            scene_space,
                            settings.label_spaces.object_current,
                            settings.label_spaces.cluster_current,
                        )
                    ),
                    Label.level.in_(["scene", "attribute", "object", "cluster"]),
                )
            ).all()

            for label_row in label_rows:
                target_id = label_row.target_id
                if label_row.level == "scene":
                    existing = scenes_by_image.get(target_id)
                    if existing is None or label_row.score > existing["score"]:
                        scenes_by_image[target_id] = {
                            "key": label_row.key,
                            "score": label_row.score,
                            "extra": label_row.extra_json,
                        }
                    continue

                if label_row.level == "attribute":
                    attributes_by_image[target_id][label_row.key] = True
                    continue

                if label_row.level == "object" and label_row.label_space_ver == settings.label_spaces.object_current:
                    objects_by_image[target_id].append(
                        {
                            "display_name": label_row.display_name,
                            "score": label_row.score,
                            "source": label_row.source,
                        }
                    )
                    continue

                if label_row.level == "cluster" and label_row.label_space_ver == settings.label_spaces.cluster_current:
                    clusters_by_image[target_id].append(
                        {
                            "display_name": label_row.display_name,
                            "score": label_row.score,
                        }
                    )

    items: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id:
            continue
        scene_entry = scenes_by_image.get(image_id)
        scene_label = scene_type_from_label_key(scene_entry["key"]) if scene_entry else None
        attr_flags = attributes_by_image.get(image_id, {})
        items.append(
            {
                "image_id": image_id,
                "scene_type": scene_label,
                "has_text": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["has_text"])),
                "has_person": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["has_person"])),
                "is_screenshot": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["is_screenshot"])),
                "is_document": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["is_document"])),
                "is_duplicate": image_id in dup_ids,
                "objects": sorted(objects_by_image.get(image_id, []), key=lambda entry: entry.get("score", 0.0), reverse=True)[:4],
                "clusters": sorted(clusters_by_image.get(image_id, []), key=lambda entry: entry.get("score", 0.0), reverse=True)[:3],
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

    primary_target = _get_primary_db_target()
    settings = load_settings()
    scene_space = settings.label_spaces.scene_current

    with open_primary_session(primary_target) as session:
        image_row = session.get(Image, image_id)
        if image_row is None:
            abort(404)

        caption_row = (
            session.execute(
                select(ImageCaption.caption, ImageCaption.model_name).where(ImageCaption.image_id == image_id).order_by(ImageCaption.model_name)
            )
        ).first()


    primary_path_str = image_row.primary_path
    logical_name = Path(primary_path_str).name

    region_rows: list[Region] = []
    projection_path = _get_projection_db_path()
    if projection_path.exists():
        with open_projection_session(projection_path) as projection_session:
            region_rows = list(
                projection_session.execute(select(Region).where(Region.image_id == image_id)).scalars().all()
            )
    region_ids = [region.id for region in region_rows]

    with open_primary_session(primary_target) as session:
        label_rows = session.execute(
            select(
                Label.key,
                Label.level,
                LabelAssignment.score,
                LabelAssignment.extra_json,
            )
            .join(Label, Label.id == LabelAssignment.label_id)
            .where(
                LabelAssignment.target_type == "image",
                LabelAssignment.target_id == image_id,
                LabelAssignment.label_space_ver == scene_space,
                Label.level.in_(["scene", "attribute"]),
            )
        ).all()

        image_object_rows = session.execute(
            select(
                Label.display_name,
                Label.key,
                LabelAssignment.score,
                LabelAssignment.source,
            )
            .join(Label, Label.id == LabelAssignment.label_id)
            .where(
                LabelAssignment.target_type == "image",
                LabelAssignment.target_id == image_id,
                LabelAssignment.label_space_ver == settings.label_spaces.object_current,
                Label.level == "object",
            )
            .order_by(LabelAssignment.score.desc())
        ).all()

        cluster_image_rows = session.execute(
            select(
                Label.display_name,
                Label.key,
                LabelAssignment.score,
                LabelAssignment.source,
            )
            .join(Label, Label.id == LabelAssignment.label_id)
            .where(
                LabelAssignment.target_type == "image",
                LabelAssignment.target_id == image_id,
                LabelAssignment.label_space_ver == settings.label_spaces.cluster_current,
                Label.level == "cluster",
            )
            .order_by(LabelAssignment.created_at.desc())
        ).all()

        region_object_rows: list[Any] = []
        region_cluster_rows: list[Any] = []
        if region_ids:
            region_object_rows = list(
                session.execute(
                select(
                    LabelAssignment.target_id,
                    Label.display_name,
                    LabelAssignment.score,
                )
                .join(Label, Label.id == LabelAssignment.label_id)
                .where(
                    LabelAssignment.target_type == "region",
                    LabelAssignment.target_id.in_(region_ids),
                    LabelAssignment.label_space_ver == settings.label_spaces.object_current,
                    Label.level == "object",
                )
                .order_by(LabelAssignment.score.desc())
            ).all())

            region_cluster_rows = list(
                session.execute(
                select(
                    LabelAssignment.target_id,
                    Label.display_name,
                    LabelAssignment.score,
                )
                .join(Label, Label.id == LabelAssignment.label_id)
                .where(
                    LabelAssignment.target_type == "region",
                    LabelAssignment.target_id.in_(region_ids),
                    LabelAssignment.label_space_ver == settings.label_spaces.cluster_current,
                    Label.level == "cluster",
                )
            ).all())

    scene_entry: dict[str, Any] | None = None
    attr_flags: dict[str, bool] = {}
    object_labels: list[dict[str, Any]] = [
        {
            "display_name": row.display_name,
            "key": row.key,
            "score": row.score,
            "source": row.source,
        }
        for row in image_object_rows
    ]
    cluster_labels: list[dict[str, Any]] = [
        {
            "display_name": row.display_name,
            "key": row.key,
            "score": row.score,
            "source": row.source,
        }
        for row in cluster_image_rows
    ]
    region_object_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    region_cluster_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in region_object_rows:
        region_object_map[row.target_id].append(
            {
                "display_name": row.display_name,
                "score": row.score,
            }
        )
    for row in region_cluster_rows:
        region_cluster_map[row.target_id].append(
            {
                "display_name": row.display_name,
                "score": row.score,
            }
        )
    for row in label_rows:
        if row.level == "scene":
            if scene_entry is None or row.score > scene_entry.get("score", 0.0):
                scene_entry = {"key": row.key, "score": row.score, "extra": row.extra_json}
            continue
        attr_flags[row.key] = True

    scene_info: dict[str, Any] | None = None
    if scene_entry is not None:
        extra_payload: dict[str, Any] = {}
        if scene_entry.get("extra"):
            try:
                extra_payload = json.loads(scene_entry["extra"])
            except Exception:
                extra_payload = {}

        scene_info = {
            "scene_type": scene_type_from_label_key(scene_entry["key"]),
            "scene_confidence": scene_entry["score"],
            "has_text": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["has_text"])),
            "has_person": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["has_person"])),
            "is_screenshot": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["is_screenshot"])),
            "is_document": bool(attr_flags.get(ATTRIBUTE_LABEL_KEYS["is_document"])),
            "classifier_name": extra_payload.get("classifier_name"),
            "classifier_version": extra_payload.get("classifier_version"),
        }
    caption_text: str | None = None
    caption_model: str | None = None
    if caption_row is not None:
        caption_text = caption_row.caption
        caption_model = caption_row.model_name

    # Near-duplicate images (both as anchor and as duplicate).
    near_duplicates: list[dict[str, Any]] = []
    with open_primary_session(primary_target) as session:
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

        related_ids: dict[str, dict[str, Any]] = {}

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
    metadata_path = _get_cache_root() / "images" / "metadata" / f"{image_id}.json"
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

    regions: list[dict[str, Any]] = []
    if region_rows:
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

            priority = float(region.raw_score) * area_weight * center_weight

            try:
                index = int(str(region.id).split("#")[-1])
            except Exception:
                index = 0

            regions.append(
                {
                    "index": index,
                    "x_min": region.x_min,
                    "y_min": region.y_min,
                    "x_max": region.x_max,
                    "y_max": region.y_max,
                    "detector_label": region.raw_label or "(raw)",
                    "detector_score": region.raw_score,
                    "backend": region.detector,
                    "priority": priority,
                    "object_labels": region_object_map.get(region.id, []),
                    "cluster_labels": region_cluster_map.get(region.id, []),
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
        object_labels=object_labels,
        cluster_labels=cluster_labels,
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
            artifact_type="thumbnail_small",
            model_name="pil",
            params={"size": settings.pipeline.thumbnail_size_small},
        )
    else:
        thumb_spec = ArtifactSpec(
            artifact_type="thumbnail_large",
            model_name="pil",
            params={"size": settings.pipeline.thumbnail_size_large},
        )

    primary_target = _get_primary_db_target()
    projection_path = _get_projection_db_path()
    cache_root = _get_cache_root()

    with open_primary_session(primary_target) as session:
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
        legacy_cache_path = cache_root / "images" / "thumbnails" / f"{image_id}.jpg"
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
        legacy_thumb_path = cache_root / "images" / "thumbnails" / f"{phash_hex}.jpg"
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
