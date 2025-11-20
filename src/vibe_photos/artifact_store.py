"""Artifact caching helpers for queue-backed workers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from utils.logging import get_logger

from vibe_photos.db import ArtifactDependency, ArtifactRecord


LOGGER = get_logger(__name__, extra={"component": "artifact_store"})


@dataclass(frozen=True)
class ArtifactSpec:
    """Identifies a cached artifact for an image."""

    artifact_type: str
    model_name: str
    params: Dict[str, object]

    @property
    def params_hash(self) -> str:
        serialized = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def version_key(self) -> str:
        return f"{self.model_name}:{self.params_hash}" if self.model_name else self.params_hash


@dataclass(frozen=True)
class ArtifactResult:
    """Result from building an artifact for durable storage."""

    storage_path: Path
    checksum: str
    payload_path: Optional[Path] = None


class ArtifactManager:
    """Persist and lookup cached artifacts with strong version keys."""

    def __init__(self, session: Session, root: Path) -> None:
        self._session = session
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def ensure_artifact(
        self,
        *,
        image_id: str,
        spec: ArtifactSpec,
        builder: Callable[[Path], ArtifactResult],
        dependencies: Optional[Iterable[int]] = None,
    ) -> ArtifactRecord:
        """Return an existing artifact or build and persist a new one."""

        existing = self._session.execute(
            select(ArtifactRecord).where(
                ArtifactRecord.image_id == image_id,
                ArtifactRecord.artifact_type == spec.artifact_type,
                ArtifactRecord.version_key == spec.version_key,
                ArtifactRecord.status == "complete",
            )
        ).scalar_one_or_none()

        if existing is not None:
            return existing

        artifact_dir = self._root / image_id / spec.artifact_type / spec.params_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        result = builder(artifact_dir)
        now = time.time()

        stmt = sqlite_insert(ArtifactRecord).values(
            image_id=image_id,
            artifact_type=spec.artifact_type,
            version_key=spec.version_key,
            params_hash=spec.params_hash,
            checksum=result.checksum,
            storage_path=str(result.storage_path),
            status="complete",
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ArtifactRecord.image_id, ArtifactRecord.artifact_type, ArtifactRecord.version_key],
            set_={
                "checksum": result.checksum,
                "storage_path": str(result.storage_path),
                "status": "complete",
                "error_message": None,
                "updated_at": now,
            },
        )
        self._session.execute(stmt)
        self._session.commit()

        row = self._session.execute(
            select(ArtifactRecord).where(
                ArtifactRecord.image_id == image_id,
                ArtifactRecord.artifact_type == spec.artifact_type,
                ArtifactRecord.version_key == spec.version_key,
            )
        ).scalar_one()

        if dependencies:
            self._record_dependencies(row.id, dependencies)

        LOGGER.info(
            "artifact_cached",
            extra={
                "image_id": image_id,
                "artifact_type": spec.artifact_type,
                "version_key": spec.version_key,
                "storage_path": str(result.storage_path),
            },
        )
        return row

    def _record_dependencies(self, artifact_id: int, dependencies: Iterable[int]) -> None:
        for dep in dependencies:
            stmt = sqlite_insert(ArtifactDependency).values(
                artifact_id=artifact_id, depends_on_artifact_id=int(dep)
            ).on_conflict_do_nothing()
            self._session.execute(stmt)
        self._session.commit()


def hash_file(path: Path) -> str:
    """Return a stable checksum for a stored artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ArtifactManager", "ArtifactResult", "ArtifactSpec", "hash_file"]

