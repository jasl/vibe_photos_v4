"""Repository helpers for label layer entities."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.db import Label, LabelAlias, LabelAssignment

LOGGER = get_logger(__name__, extra={"component": "label_repository"})
_ALLOWED_TARGET_TYPES: set[str] = {"image", "region"}


class LabelRepository:
    """Persist labels, aliases, and assignments via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_or_create_label(
        self,
        *,
        key: str,
        level: str,
        display_name: str,
        parent_key: str | None = None,
        icon: str | None = None,
    ) -> Label:
        """Return an existing label or create it if missing.

        The method is idempotent and will update display metadata when the stored
        row differs from the provided attributes.
        """

        parent_id = self._maybe_lookup_label_id(parent_key) if parent_key else None
        now = time.time()

        existing = self._get_label_by_key(key)
        if existing is None:
            label = Label(
                key=key,
                level=level,
                display_name=display_name,
                parent_id=parent_id,
                icon=icon,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            self._session.add(label)
            self._session.flush()
            LOGGER.info("label_created", extra={"key": key, "level": level})
            return label

        if existing.level != level:
            raise ValueError(
                f"Label {key!r} already exists with level {existing.level!r}, expected {level!r}."
            )

        updated = False
        if parent_id != existing.parent_id:
            existing.parent_id = parent_id
            updated = True
        if display_name != existing.display_name:
            existing.display_name = display_name
            updated = True
        if icon is not None and icon != existing.icon:
            existing.icon = icon
            updated = True

        if updated:
            existing.updated_at = now
            self._session.add(existing)
            self._session.flush()
            LOGGER.info("label_updated", extra={"key": key, "level": level})

        return existing

    def ensure_aliases(self, label: Label, aliases: Sequence[str], language: str | None = None) -> None:
        """Insert aliases that do not already exist for a label."""

        if not aliases:
            return

        if label.id is None:
            self._session.flush()
        if label.id is None:
            raise ValueError("Label must be persisted before adding aliases.")

        normalized = [alias.strip() for alias in aliases if alias and alias.strip()]
        if not normalized:
            return

        existing = self._get_existing_aliases(label.id, language=language)
        new_aliases = [alias for alias in normalized if alias not in existing]
        if not new_aliases:
            return

        rows = [
            LabelAlias(label_id=label.id, alias_text=alias, language=language, is_preferred=False)
            for alias in new_aliases
        ]
        self._session.add_all(rows)
        self._session.flush()
        LOGGER.info(
            "label_aliases_added",
            extra={"key": label.key, "language": language or "unknown", "count": len(new_aliases)},
        )

    def upsert_label_assignment(
        self,
        *,
        target_type: str,
        target_id: str,
        label: Label,
        source: str,
        label_space_ver: str,
        score: float,
        extra_json: str | None = None,
    ) -> LabelAssignment:
        """Create or update a label assignment for an image or region."""

        if target_type not in _ALLOWED_TARGET_TYPES:
            raise ValueError(f"Unsupported target_type {target_type!r}. Expected one of {_ALLOWED_TARGET_TYPES}.")
        if not target_id:
            raise ValueError("target_id must be provided.")
        if label.id is None:
            self._session.flush()
        if label.id is None:
            raise ValueError("Label must be persisted before writing assignments.")

        now = time.time()
        stmt = select(LabelAssignment).where(
            LabelAssignment.target_type == target_type,
            LabelAssignment.target_id == target_id,
            LabelAssignment.label_id == label.id,
            LabelAssignment.source == source,
            LabelAssignment.label_space_ver == label_space_ver,
        )
        assignment = self._session.execute(stmt).scalar_one_or_none()
        if assignment is None:
            assignment = LabelAssignment(
                target_type=target_type,
                target_id=target_id,
                label_id=label.id,
                source=source,
                label_space_ver=label_space_ver,
                score=score,
                extra_json=extra_json,
                created_at=now,
                updated_at=now,
            )
            self._session.add(assignment)
            self._session.flush()
            LOGGER.info(
                "label_assignment_created",
                extra={
                    "key": label.key,
                    "target_type": target_type,
                    "target_id": target_id,
                    "source": source,
                    "label_space_ver": label_space_ver,
                },
            )
            return assignment

        assignment.score = score
        assignment.extra_json = extra_json
        assignment.updated_at = now
        self._session.add(assignment)
        self._session.flush()
        LOGGER.info(
            "label_assignment_updated",
            extra={
                "key": label.key,
                "target_type": target_type,
                "target_id": target_id,
                "source": source,
                "label_space_ver": label_space_ver,
            },
        )
        return assignment

    def _get_label_by_key(self, key: str) -> Label | None:
        stmt = select(Label).where(Label.key == key)
        return self._session.execute(stmt).scalar_one_or_none()

    def _maybe_lookup_label_id(self, key: str | None) -> int | None:
        if key is None:
            return None

        label = self._get_label_by_key(key)
        if label is None:
            raise ValueError(f"Parent label {key!r} does not exist.")
        if label.id is None:
            self._session.flush()
        if label.id is None:
            raise ValueError(f"Parent label {key!r} must be persisted before use.")
        return label.id

    def _get_existing_aliases(self, label_id: int, *, language: str | None) -> set[str]:
        stmt = select(LabelAlias.alias_text).where(LabelAlias.label_id == label_id)
        if language is None:
            stmt = stmt.where(LabelAlias.language.is_(None))
        else:
            stmt = stmt.where(LabelAlias.language == language)

        results = self._session.execute(stmt).scalars().all()
        return set(results)
