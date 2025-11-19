"""SQLAlchemy helpers, schema definitions, and session management."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from sqlalchemy import Boolean, Float, Index, Integer, String, Text, create_engine, delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from utils.logging import get_logger


LOGGER = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Image(Base):
    """Primary image metadata and processing state."""

    __tablename__ = "images"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    primary_path: Mapped[str] = mapped_column(String, nullable=False)
    all_paths: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mtime: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exif_datetime: Mapped[str | None] = mapped_column(String, nullable=True)
    camera_model: Mapped[str | None] = mapped_column(String, nullable=True)
    hash_algo: Mapped[str] = mapped_column(String, nullable=False)
    phash: Mapped[str | None] = mapped_column(String, nullable=True)
    phash_algo: Mapped[str | None] = mapped_column(String, nullable=True)
    phash_updated_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index("idx_images_primary_path", "primary_path"),
        Index("idx_images_status", "status"),
    )


class ImageScene(Base):
    """Scene classification outputs for an image."""

    __tablename__ = "image_scene"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    scene_type: Mapped[str] = mapped_column(String, nullable=False)
    scene_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_text: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_person: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_screenshot: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_document: Mapped[bool] = mapped_column(Boolean, nullable=False)
    classifier_name: Mapped[str] = mapped_column(String, nullable=False)
    classifier_version: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("idx_image_scene_scene_type", "scene_type"),
        Index("idx_image_scene_has_text", "has_text"),
        Index("idx_image_scene_has_person", "has_person"),
        Index("idx_image_scene_is_screenshot", "is_screenshot"),
        Index("idx_image_scene_is_document", "is_document"),
    )


class ImageEmbedding(Base):
    """Model embeddings for an image."""

    __tablename__ = "image_embedding"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_path: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_backend: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("idx_image_embedding_model_name", "model_name"),)


class ImageCaption(Base):
    """Caption text for an image and model variant."""

    __tablename__ = "image_caption"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    model_backend: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("idx_image_caption_model_name", "model_name"),)


class ImageNearDuplicate(Base):
    """Near-duplicate relationships computed from perceptual hashes."""

    __tablename__ = "image_near_duplicate"

    anchor_image_id: Mapped[str] = mapped_column(String, primary_key=True)
    duplicate_image_id: Mapped[str] = mapped_column(String, primary_key=True)
    phash_distance: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("idx_image_near_duplicate_duplicate", "duplicate_image_id"),)


class ImageRegion(Base):
    """Object-level detection regions for an image."""

    __tablename__ = "image_region"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    region_index: Mapped[int] = mapped_column(Integer, primary_key=True)

    x_min: Mapped[float] = mapped_column(Float, nullable=False)  # normalized [0, 1]
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)

    detector_label: Mapped[str] = mapped_column(String, nullable=False)
    detector_score: Mapped[float] = mapped_column(Float, nullable=False)

    refined_label: Mapped[str | None] = mapped_column(String, nullable=True)
    refined_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    backend: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("idx_image_region_image_id", "image_id"),
        Index("idx_image_region_detector_label", "detector_label"),
        Index("idx_image_region_refined_label", "refined_label"),
    )


class PreprocessTask(Base):
    """Queue of preprocessing tasks for concurrent workers.

    This lightweight table lives in SQLite alongside the primary or projection
    database and is safe to rebuild if needed. Tasks are idempotent: workers
    may safely retry failed tasks without affecting correctness.
    """

    __tablename__ = "preprocess_task"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    completed_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_preprocess_task_status_priority", "status", "priority"),
        Index("idx_preprocess_task_image_type", "image_id", "task_type"),
    )


_ENGINE_CACHE: Dict[Path, Engine] = {}


def _ensure_parent_directory(path: Path) -> None:
    """Ensure the parent directory for a database file exists."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("db_parent_directory_error", extra={"path": str(path), "error": str(exc)})
        raise


def _get_engine(path: Path) -> Engine:
    """Return a cached SQLite engine for the provided path, creating schema if needed."""

    resolved = path.resolve()
    engine = _ENGINE_CACHE.get(resolved)
    if engine is None:
        _ensure_parent_directory(resolved)
        engine = create_engine(f"sqlite:///{resolved}", future=True)
        Base.metadata.create_all(engine)
        _ENGINE_CACHE[resolved] = engine
    return engine


def open_primary_session(path: Path) -> Session:
    """Open a SQLAlchemy session for the primary database."""

    engine = _get_engine(path)
    return Session(engine, future=True)


def open_projection_session(path: Path) -> Session:
    """Open a SQLAlchemy session for the projection database."""

    engine = _get_engine(path)
    return Session(engine, future=True)


def reset_projection_tables(session: Session) -> None:
    """Remove projection tables for a clean rebuild when re-running the pipeline."""

    session.execute(delete(ImageScene))
    session.execute(delete(ImageEmbedding))
    session.execute(delete(ImageCaption))
    session.execute(delete(ImageNearDuplicate))
    session.execute(delete(ImageRegion))
    session.commit()


__all__ = [
    "Base",
    "Image",
    "ImageCaption",
    "ImageEmbedding",
    "ImageNearDuplicate",
    "ImageRegion",
    "ImageScene",
    "PreprocessTask",
    "open_primary_session",
    "open_projection_session",
    "reset_projection_tables",
]
