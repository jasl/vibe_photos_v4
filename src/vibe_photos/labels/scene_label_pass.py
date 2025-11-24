"""Scene + attribute label pass writing into the unified label layer."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.classifier import (
    SceneAttributes,
    SceneClassifierWithAttributes,
    build_scene_classifier,
)
from vibe_photos.config import Settings
from vibe_photos.db import Image, ImageEmbedding, ImageScene, Label
from vibe_photos.db_helpers import dialect_insert
from vibe_photos.labels.repository import LabelRepository
from vibe_photos.labels.scene_schema import (
    ALL_SCENE_LABEL_KEYS,
    ATTRIBUTE_LABEL_KEYS,
    scene_label_key_from_type,
)

LOGGER = get_logger(__name__, extra={"component": "scene_label_pass"})


def _margin_to_probability(margin: float) -> float:
    """Map a margin (-inf, +inf) to [0, 1] via logistic transformation."""

    return float(1.0 / (1.0 + math.exp(-margin)))


def _load_embedding_rows(cache_session: Session, model_name: str) -> dict[str, str]:
    stmt = select(ImageEmbedding.image_id, ImageEmbedding.embedding_path).where(ImageEmbedding.model_name == model_name)
    rows = cache_session.execute(stmt).all()
    return {row.image_id: row.embedding_path for row in rows}


def _resolve_embedding_path(cache_root: Path, rel_path: str) -> Path:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return path_obj
    candidate = cache_root / path_obj
    if candidate.exists():
        return candidate
    return cache_root / "embeddings" / path_obj


def _load_active_image_ids(primary_session: Session) -> list[str]:
    stmt = select(Image.image_id).where(Image.status == "active").order_by(Image.image_id)
    return [row.image_id for row in primary_session.execute(stmt)]


def _upsert_image_scene(
    *,
    image_id: str,
    attributes: SceneAttributes,
    cache_session: Session,
    primary_session: Session,
) -> None:
    now = time.time()
    def _upsert(target_session: Session) -> None:
        stmt = dialect_insert(target_session, ImageScene).values(
            image_id=image_id,
            scene_type=attributes.scene_type,
            scene_confidence=attributes.scene_confidence,
            has_text=bool(attributes.has_text),
            has_person=bool(attributes.has_person),
            is_screenshot=bool(attributes.is_screenshot),
            is_document=bool(attributes.is_document),
            classifier_name=attributes.classifier_name,
            classifier_version=attributes.classifier_version,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ImageScene.image_id],
            set_={
                "scene_type": stmt.excluded.scene_type,
                "scene_confidence": stmt.excluded.scene_confidence,
                "has_text": stmt.excluded.has_text,
                "has_person": stmt.excluded.has_person,
                "is_screenshot": stmt.excluded.is_screenshot,
                "is_document": stmt.excluded.is_document,
                "classifier_name": stmt.excluded.classifier_name,
                "classifier_version": stmt.excluded.classifier_version,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        target_session.execute(stmt)

    _upsert(cache_session)
    if cache_session is not primary_session:
        _upsert(primary_session)


def _write_scene_assignment(
    *,
    repo: LabelRepository,
    scene_labels: dict[str, Label],
    attributes: SceneAttributes,
    image_id: str,
    label_space_ver: str,
) -> None:
    label_key = scene_label_key_from_type(attributes.scene_type)
    label = scene_labels[label_key]
    extra = json.dumps(
        {
            "classifier_name": attributes.classifier_name,
            "classifier_version": attributes.classifier_version,
            "scene_margin": attributes.scene_margin,
        }
    )
    score = float(max(0.0, min(1.0, attributes.scene_confidence or 0.0)))
    repo.upsert_label_assignment(
        target_type="image",
        target_id=image_id,
        label=label,
        source="classifier",
        label_space_ver=label_space_ver,
        score=score,
        extra_json=extra,
    )


def _write_attribute_assignments(
    *,
    repo: LabelRepository,
    attribute_labels: dict[str, Label],
    attributes: SceneAttributes,
    image_id: str,
    label_space_ver: str,
) -> None:
    entries = [
        ("has_person", attributes.has_person, attributes.has_person_margin),
        ("has_text", attributes.has_text, attributes.has_text_margin),
        ("is_screenshot", attributes.is_screenshot, attributes.is_screenshot_margin),
        ("is_document", attributes.is_document, attributes.is_document_margin),
    ]
    for attr_id, value, margin in entries:
        if not value:
            continue
        label = attribute_labels.get(attr_id)
        if label is None:
            continue
        margin_value = float(margin or 0.0)
        score = _margin_to_probability(margin_value)
        extra = json.dumps(
            {
                "classifier_name": attributes.classifier_name,
                "classifier_version": attributes.classifier_version,
                "margin": margin_value,
            }
        )
        repo.upsert_label_assignment(
            target_type="image",
            target_id=image_id,
            label=label,
            source="classifier",
            label_space_ver=label_space_ver,
            score=score,
            extra_json=extra,
        )


def run_scene_label_pass(
    *,
    primary_session: Session,
    cache_session: Session,
    settings: Settings,
    cache_root: Path,
    label_space_ver: str | None = None,
    start_after: str | None = None,
    update_cursor: Callable[[str | None], None] | None = None,
) -> int:
    """Compute scene + attribute labels and persist them to the label layer."""

    label_space = label_space_ver or settings.label_spaces.scene_current
    embedding_model = settings.models.embedding.resolved_model_name()

    embedding_paths = _load_embedding_rows(cache_session, embedding_model)
    if not embedding_paths:
        LOGGER.info("scene_label_pass_no_embeddings", extra={"model_name": embedding_model})
        return 0

    candidate_ids = _load_active_image_ids(primary_session)
    target_ids = [image_id for image_id in candidate_ids if image_id in embedding_paths]
    if start_after:
        target_ids = [image_id for image_id in target_ids if image_id > start_after]

    if not target_ids:
        LOGGER.info("scene_label_pass_no_targets", extra={})
        return 0

    classifier: SceneClassifierWithAttributes = build_scene_classifier(settings)
    repo = LabelRepository(primary_session)
    scene_labels = {key: repo.require_label(key) for key in ALL_SCENE_LABEL_KEYS}
    attribute_labels = {attr_id: repo.require_label(label_key) for attr_id, label_key in ATTRIBUTE_LABEL_KEYS.items()}

    batch_size = max(1, settings.models.embedding.batch_size)
    processed = 0
    total = len(target_ids)
    progress_interval = max(1, total // 20) if total else 0

    same_session = cache_session is primary_session

    for batch_start in range(0, total, batch_size):
        batch_ids = target_ids[batch_start : batch_start + batch_size]
        embeddings: list[torch.Tensor] = []
        valid_ids: list[str] = []

        for image_id in batch_ids:
            rel_path = embedding_paths.get(image_id)
            if rel_path is None:
                continue
            emb_path = _resolve_embedding_path(cache_root, rel_path)
            try:
                vec = np.load(emb_path)
            except Exception as exc:  # pragma: no cover - defensive log
                LOGGER.error(
                    "scene_label_pass_embedding_load_error",
                    extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
                )
                continue
            try:
                tensor = torch.from_numpy(vec).float()
            except Exception as exc:  # pragma: no cover - defensive log
                LOGGER.error(
                    "scene_label_pass_tensor_error",
                    extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
                )
                continue
            embeddings.append(tensor)
            valid_ids.append(image_id)

        if not embeddings:
            continue

        attributes_list: Iterable[SceneAttributes] = classifier.classify_batch(embeddings)

        for image_id, attrs in zip(valid_ids, attributes_list, strict=True):
            _write_scene_assignment(
                repo=repo,
                scene_labels=scene_labels,
                attributes=attrs,
                image_id=image_id,
                label_space_ver=label_space,
            )
            _write_attribute_assignments(
                repo=repo,
                attribute_labels=attribute_labels,
                attributes=attrs,
                image_id=image_id,
                label_space_ver=label_space,
            )
            _upsert_image_scene(
                image_id=image_id,
                attributes=attrs,
                cache_session=cache_session,
                primary_session=primary_session,
            )

        cache_session.commit()
        if not same_session:
            primary_session.commit()

        processed += len(valid_ids)
        if update_cursor and valid_ids:
            update_cursor(valid_ids[-1])

        if progress_interval:
            current = min(batch_start + batch_size, total)
            if current % progress_interval == 0 or current == total:
                percent = round(current * 100.0 / max(total, 1), 1)
                LOGGER.info(
                    "scene_label_pass_progress",
                    extra={
                        "processed": processed,
                        "total": total,
                        "percent": percent,
                    },
                )

    LOGGER.info("scene_label_pass_complete", extra={"processed": processed, "label_space": label_space})
    if update_cursor:
        update_cursor(None)
    return processed


__all__ = ["run_scene_label_pass"]
