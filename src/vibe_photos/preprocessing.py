"""Shared preprocessing steps for Celery tasks and CLI tooling."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from PIL import Image

from vibe_photos.artifact_store import (
    ArtifactManager,
    ArtifactResult,
    ArtifactSpec,
    hash_file,
)
from vibe_photos.config import Settings
from vibe_photos.db import ArtifactRecord
from vibe_photos.hasher import (
    CONTENT_HASH_ALGO,
    compute_content_hash,
    compute_perceptual_hash,
)
from vibe_photos.ml.siglip_blip import SiglipBlipDetector
from vibe_photos.thumbnailing import save_thumbnail


def extract_exif_and_gps(
    image: Image.Image, datetime_format: str = "raw"
) -> tuple[str | None, str | None, dict[str, object] | None]:
    """Extract EXIF datetime, camera model, and GPS coordinates from a PIL image."""

    try:
        from PIL import ExifTags as _ExifTags
    except Exception:
        return None, None, None

    try:
        exif = image.getexif()
    except Exception:
        return None, None, None

    if not exif:
        return None, None, None

    exif_dict = dict(exif)
    exif_by_name: dict[str, object] = {}
    for tag_id, value in exif_dict.items():
        name = _ExifTags.TAGS.get(tag_id, str(tag_id))
        exif_by_name[name] = value

    exif_datetime: str | None = None
    camera_model: str | None = None

    raw_dt = exif_by_name.get("DateTimeOriginal") or exif_by_name.get("DateTime")
    if isinstance(raw_dt, str):
        if datetime_format == "iso":
            try:
                dt = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
                exif_datetime = dt.isoformat(timespec="seconds")
            except Exception:
                exif_datetime = raw_dt
        else:
            exif_datetime = raw_dt

    model = exif_by_name.get("Model")
    make = exif_by_name.get("Make")
    if isinstance(model, str) and isinstance(make, str):
        camera_model = f"{make} {model}".strip()
    elif isinstance(model, str):
        camera_model = model
    elif isinstance(make, str):
        camera_model = make

    gps_info = exif_by_name.get("GPSInfo")
    if not gps_info or not isinstance(gps_info, dict):
        return exif_datetime, camera_model, None

    gps_tags: dict[str, object] = {}
    for key, value in gps_info.items():
        name = _ExifTags.GPSTAGS.get(key, str(key))
        gps_tags[name] = value

    lat = gps_tags.get("GPSLatitude")
    lat_ref = gps_tags.get("GPSLatitudeRef")
    lon = gps_tags.get("GPSLongitude")
    lon_ref = gps_tags.get("GPSLongitudeRef")

    def _to_degrees(value: object) -> float | None:
        if not isinstance(value, Sequence):
            return None
        if len(value) < 3:
            return None
        d, m, s = value[0], value[1], value[2]
        try:
            d_val = float(d)
            m_val = float(m)
            s_val = float(s)
            return d_val + (m_val / 60.0) + (s_val / 3600.0)
        except Exception:
            return None

    latitude = _to_degrees(lat)
    longitude = _to_degrees(lon)
    if latitude is None or longitude is None:
        return exif_datetime, camera_model, None

    if isinstance(lat_ref, str) and lat_ref.upper() == "S":
        latitude = -latitude
    if isinstance(lon_ref, str) and lon_ref.upper() == "W":
        longitude = -longitude

    gps_payload = {
        "latitude": latitude,
        "latitude_ref": lat_ref,
        "longitude": longitude,
        "longitude_ref": lon_ref,
    }
    return exif_datetime, camera_model, gps_payload


def build_thumbnail(
    image: Image.Image,
    output_dir: Path,
    size: int,
    quality: int,
    *,
    name_suffix: str | None = None,
) -> ArtifactResult:
    """Write a thumbnail JPEG to ``output_dir`` and return its artifact metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    out_path = output_dir / f"thumbnail{suffix}_{size}.jpg"
    save_thumbnail(image, out_path, size, quality)
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def extract_exif_payload(image: Image.Image) -> dict:
    """Extract a JSON-serializable EXIF payload from a PIL image."""

    try:
        exif = image.getexif()
    except Exception:
        return {}

    payload: dict[str, object] = {}
    for tag_id, value in dict(exif).items():
        payload[str(tag_id)] = value if isinstance(value, (int, float, str)) else str(value)
    return payload


def build_exif_artifact(image: Image.Image, output_dir: Path, *, datetime_format: str) -> ArtifactResult:
    """Write EXIF metadata to ``exif.json`` and return its artifact metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = extract_exif_payload(image)
    out_path = output_dir / "exif.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path), payload_path=out_path)


def build_content_hash_artifact(image_path: Path, output_dir: Path) -> ArtifactResult:
    """Compute and persist the content hash for an image file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    value = compute_content_hash(image_path)
    out_path = output_dir / "content_hash.txt"
    out_path.write_text(value, encoding="utf-8")
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def build_phash_artifact(image: Image.Image, output_dir: Path) -> ArtifactResult:
    """Compute and persist the perceptual hash for an image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    value = compute_perceptual_hash(image)
    out_path = output_dir / "perceptual_hash.txt"
    out_path.write_text(value, encoding="utf-8")
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def build_embedding_artifact(
    detector: SiglipBlipDetector,
    image: Image.Image,
    output_dir: Path,
) -> ArtifactResult:
    """Run SigLIP once and persist the raw embedding + metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vec = detector.embed_image(image=image)
    emb_path = output_dir / "embedding.npy"
    meta_path = output_dir / "embedding_meta.json"
    import numpy as np

    np.save(emb_path, vec)
    meta = {"embedding_dim": int(vec.shape[-1]), "dtype": "float32"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return ArtifactResult(storage_path=emb_path, checksum=hash_file(emb_path), payload_path=meta_path)


def build_caption_artifact(detector: SiglipBlipDetector, image: Image.Image, output_dir: Path) -> ArtifactResult:
    """Generate a BLIP caption and persist it to ``caption.txt``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    caption = detector._generate_caption_with_blip(image=image)
    out_path = output_dir / "caption.txt"
    out_path.write_text(caption, encoding="utf-8")
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def build_metadata_artifact(metadata: dict[str, object], output_dir: Path) -> ArtifactResult:
    """Persist normalized metadata (EXIF, camera, GPS) to JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metadata.json"
    out_path.write_text(json.dumps(metadata), encoding="utf-8")
    return ArtifactResult(storage_path=out_path, checksum=hash_file(out_path))


def ensure_preprocessing_artifacts(
    *,
    image_id: str,
    image: Image.Image,
    image_path: Path,
    settings: Settings,
    manager: ArtifactManager,
    detector: SiglipBlipDetector,
) -> dict[str, ArtifactSpec | ArtifactRecord | dict[str, object]]:
    """Ensure all preprocessing artifacts exist for a single image and return key artifacts/specs."""

    metadata_exif_datetime, metadata_camera_model, metadata_gps = extract_exif_and_gps(
        image, datetime_format=settings.pipeline.exif_datetime_format
    )
    metadata_payload: dict[str, object] = {
        "image_id": image_id,
        "primary_path": str(image_path),
        "updated_at": time.time(),
    }
    if metadata_exif_datetime is not None:
        metadata_payload["exif_datetime"] = metadata_exif_datetime
    if metadata_camera_model is not None:
        metadata_payload["camera_model"] = metadata_camera_model
    if metadata_gps is not None:
        metadata_payload["gps"] = metadata_gps

    thumb_large = ArtifactSpec(
        artifact_type="thumbnail_large",
        model_name="pil",
        params={"size": settings.pipeline.thumbnail_size_large},
    )
    thumb_large_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=thumb_large,
        builder=lambda path: build_thumbnail(
            image,
            path,
            settings.pipeline.thumbnail_size_large,
            settings.pipeline.thumbnail_quality,
            name_suffix="large",
        ),
    )

    thumb_small = ArtifactSpec(
        artifact_type="thumbnail_small",
        model_name="pil",
        params={"size": settings.pipeline.thumbnail_size_small},
    )
    thumb_small_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=thumb_small,
        builder=lambda path: build_thumbnail(
            image,
            path,
            settings.pipeline.thumbnail_size_small,
            settings.pipeline.thumbnail_quality,
            name_suffix="small",
        ),
    )

    exif_spec = ArtifactSpec(
        artifact_type="exif",
        model_name="pil_exif",
        params={"datetime_format": settings.pipeline.exif_datetime_format},
    )
    exif_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=exif_spec,
        builder=lambda out_dir: build_exif_artifact(image, out_dir, datetime_format=settings.pipeline.exif_datetime_format),
    )

    metadata_spec = ArtifactSpec(
        artifact_type="metadata",
        model_name="exif_summary",
        params={"datetime_format": settings.pipeline.exif_datetime_format},
    )
    metadata_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=metadata_spec,
        builder=lambda out_dir: build_metadata_artifact(metadata_payload, out_dir),
    )

    hash_spec = ArtifactSpec(artifact_type="content_hash", model_name=CONTENT_HASH_ALGO, params={})
    content_hash_artifact = manager.ensure_artifact(
        image_id=image_id, spec=hash_spec, builder=lambda out_dir: build_content_hash_artifact(image_path, out_dir)
    )

    phash_spec = ArtifactSpec(artifact_type="perceptual_hash", model_name="phash", params={})
    phash_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=phash_spec,
        builder=lambda out_dir: build_phash_artifact(image, out_dir),
    )

    embed_spec = ArtifactSpec(
        artifact_type="embedding",
        model_name=settings.models.embedding.resolved_model_name(),
        params={},
    )
    embedding = manager.ensure_artifact(
        image_id=image_id,
        spec=embed_spec,
        builder=lambda out_dir: build_embedding_artifact(detector, image, out_dir),
        dependencies=[content_hash_artifact.id],
    )

    caption_spec = ArtifactSpec(
        artifact_type="caption",
        model_name=settings.models.caption.resolved_model_name(),
        params={},
    )
    caption_artifact = manager.ensure_artifact(
        image_id=image_id,
        spec=caption_spec,
        builder=lambda out_dir: build_caption_artifact(detector, image, out_dir),
        dependencies=[embedding.id],
    )

    return {
        "thumb_large_spec": thumb_large,
        "thumb_small_spec": thumb_small,
        "thumb_large_artifact": thumb_large_artifact,
        "thumb_small_artifact": thumb_small_artifact,
        "exif_spec": exif_spec,
        "exif_artifact": exif_artifact,
        "metadata_spec": metadata_spec,
        "metadata_artifact": metadata_artifact,
        "metadata_payload": metadata_payload,
        "content_hash_artifact": content_hash_artifact,
        "phash_artifact": phash_artifact,
        "embedding_artifact": embedding,
        "embedding_spec": embed_spec,
        "caption_spec": caption_spec,
        "caption_artifact": caption_artifact,
    }
