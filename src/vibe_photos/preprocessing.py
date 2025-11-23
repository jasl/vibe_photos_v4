"""Shared preprocessing steps for Celery tasks and CLI tooling."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vibe_photos.artifact_store import ArtifactManager, ArtifactResult, ArtifactSpec, hash_file
from vibe_photos.config import Settings
from vibe_photos.db import ArtifactRecord
from vibe_photos.hasher import CONTENT_HASH_ALGO, compute_content_hash, compute_perceptual_hash
from vibe_photos.ml.siglip_blip import SiglipBlipDetector
from vibe_photos.thumbnailing import save_thumbnail


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


def ensure_preprocessing_artifacts(
    *,
    image_id: str,
    image: Image.Image,
    image_path: Path,
    settings: Settings,
    manager: ArtifactManager,
    detector: SiglipBlipDetector,
) -> dict[str, ArtifactSpec | ArtifactRecord]:
    """Ensure all preprocessing artifacts exist for a single image and return key artifacts/specs."""

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
        params={"device": settings.models.embedding.device, "batch_size": settings.models.embedding.batch_size},
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
        params={"device": settings.models.caption.device},
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
        "content_hash_artifact": content_hash_artifact,
        "phash_artifact": phash_artifact,
        "embedding_artifact": embedding,
        "embedding_spec": embed_spec,
        "caption_spec": caption_spec,
        "caption_artifact": caption_artifact,
    }
