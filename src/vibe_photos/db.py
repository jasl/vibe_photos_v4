"""SQLAlchemy helpers, schema definitions, and session management."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
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


class ArtifactRecord(Base):
    """Versioned artifact metadata persisted to durable storage."""

    __tablename__ = "artifact"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    version_key: Mapped[str] = mapped_column(String, nullable=False)
    params_hash: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="complete")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("idx_artifact_identity", "image_id", "artifact_type", "version_key", unique=True),
        Index("idx_artifact_type", "artifact_type"),
    )


class ArtifactDependency(Base):
    """Lineage mapping between artifacts to support cache reuse."""

    __tablename__ = "artifact_dependency"

    artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    depends_on_artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    __table_args__ = (
        Index("idx_artifact_dependency_parent", "depends_on_artifact_id"),
    )


class ProcessResult(Base):
    """Versioned classification and clustering outputs bound to artifacts."""

    __tablename__ = "process_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False)
    result_type: Mapped[str] = mapped_column(String, nullable=False)
    version_key: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    source_artifact_ids: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("idx_main_stage_image", "image_id"),
        Index("idx_main_stage_type", "result_type"),
    )


class PostProcessResult(Base):
    """Resource-heavy annotations stored separately from main pipeline outputs."""

    __tablename__ = "post_process_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False)
    post_process_type: Mapped[str] = mapped_column(String, nullable=False)
    version_key: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    source_artifact_ids: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("idx_post_process_image", "image_id"),
        Index("idx_post_process_type", "post_process_type"),
    )


class Label(Base):
    """Unified labels covering scenes, objects, attributes, and clusters."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    level: Mapped[str] = mapped_column(String, nullable=False)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("labels.id"), nullable=True)
    icon: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("idx_labels_level_key", "level", "key"),)


class LabelAlias(Base):
    """Multilingual aliases used to build text prototypes for labels."""

    __tablename__ = "label_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label_id: Mapped[int] = mapped_column(Integer, ForeignKey("labels.id"), nullable=False)
    alias_text: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    is_preferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (Index("idx_label_aliases_label", "label_id"),)


class LabelAssignment(Base):
    """Label assignments for images or regions with source and version metadata."""

    __tablename__ = "label_assignment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_type: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    label_id: Mapped[int] = mapped_column(Integer, ForeignKey("labels.id"), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    label_space_ver: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "target_type",
            "target_id",
            "label_id",
            "source",
            "label_space_ver",
            name="uq_label_assignment_target_label",
        ),
        Index("idx_label_assignment_target", "target_type", "target_id", "label_space_ver"),
        Index("idx_label_assignment_label", "label_id", "label_space_ver", "source"),
    )


_ENGINE_CACHE: Dict[Path, Engine] = {}
_ENGINE_LOCK = Lock()


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
    if engine is not None:
        return engine

    with _ENGINE_LOCK:
        engine = _ENGINE_CACHE.get(resolved)
        if engine is not None:
            return engine

        _ensure_parent_directory(resolved)
        engine = create_engine(
            f"sqlite:///{resolved}",
            future=True,
            # Allow readers/writers to wait for a busy database instead of failing fast.
            connect_args={"timeout": 30.0},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:  # type: ignore[override]
            """Configure SQLite for better concurrent access."""

            cursor = dbapi_connection.cursor()
            try:
                # WAL mode lets readers proceed while a writer has a transaction open.
                cursor.execute("PRAGMA journal_mode=WAL")
                # Additional safeguard on the SQLite side; in milliseconds.
                cursor.execute("PRAGMA busy_timeout = 30000")
            finally:
                cursor.close()

        try:
            Base.metadata.create_all(engine)
        except OperationalError as exc:
            # In highly concurrent startup (e.g., multiple Celery workers) another
            # process may create tables between SQLite's existence check and the
            # CREATE TABLE statement. Treat "already exists" as a benign race.
            message = str(exc).lower()
            if "already exists" in message:
                LOGGER.info(
                    "db_create_all_table_exists_race",
                    extra={"path": str(resolved), "error": str(exc)},
                )
            else:
                raise

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
    "ArtifactDependency",
    "ArtifactRecord",
    "ProcessResult",
    "PostProcessResult",
    "Label",
    "LabelAlias",
    "LabelAssignment",
    "open_primary_session",
    "open_projection_session",
    "reset_projection_tables",
]
