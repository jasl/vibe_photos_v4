"""Propagate canonical labels to near-duplicate images."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import delete, select

from utils.logging import get_logger
from vibe_photos.db import ImageNearDuplicateGroup, ImageNearDuplicateMembership, Label, LabelAssignment
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"component": "duplicate_propagation"})


def _load_duplicate_map(primary_session) -> Dict[str, List[str]]:
    groups = {
        row.id: row.canonical_image_id
        for row in primary_session.execute(
            select(ImageNearDuplicateGroup.id, ImageNearDuplicateGroup.canonical_image_id)
        )
    }
    if not groups:
        return {}

    rows = primary_session.execute(
        select(
            ImageNearDuplicateMembership.group_id,
            ImageNearDuplicateMembership.image_id,
            ImageNearDuplicateMembership.is_canonical,
        )
    ).all()

    duplicates: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        canonical_id = groups.get(row.group_id)
        if canonical_id is None or row.is_canonical:
            continue
        duplicates[canonical_id].append(row.image_id)
    return duplicates


def propagate_duplicate_labels(primary_session, *, label_space_ver: str) -> int:
    """Copy canonical image labels to their near-duplicate siblings."""

    repo = LabelRepository(primary_session)
    duplicate_map = _load_duplicate_map(primary_session)

    primary_session.execute(
        delete(LabelAssignment).where(
            LabelAssignment.target_type == "image",
            LabelAssignment.label_space_ver == label_space_ver,
            LabelAssignment.source == "duplicate_propagated",
        )
    )

    if not duplicate_map:
        primary_session.commit()
        LOGGER.info(
            "duplicate_propagation_skip",
            extra={"label_space": label_space_ver, "reason": "no_duplicate_groups"},
        )
        return 0

    canonical_ids = list(duplicate_map.keys())
    assignment_rows = primary_session.execute(
        select(
            LabelAssignment.label_id,
            LabelAssignment.target_id,
            LabelAssignment.score,
            LabelAssignment.extra_json,
        )
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id.in_(canonical_ids),
            LabelAssignment.label_space_ver == label_space_ver,
            LabelAssignment.source != "duplicate_propagated",
        )
    ).all()

    if not assignment_rows:
        primary_session.commit()
        return 0

    label_ids = {row.label_id for row in assignment_rows}
    label_by_id = {
        label.id: label
        for label in primary_session.execute(select(Label).where(Label.id.in_(label_ids))).scalars()
    }

    propagated = 0
    for assignment in assignment_rows:
        label = label_by_id.get(assignment.label_id)
        if label is None:
            continue
        canonical_id = assignment.target_id
        targets = duplicate_map.get(canonical_id, [])
        if not targets:
            continue

        extra_payload = {}
        if assignment.extra_json:
            try:
                extra_payload = json.loads(assignment.extra_json)
            except Exception:
                extra_payload = {}
        extra_payload["duplicate_source_image_id"] = canonical_id
        extra_json = json.dumps(extra_payload)

        for duplicate_id in targets:
            repo.upsert_label_assignment(
                target_type="image",
                target_id=duplicate_id,
                label=label,
                source="duplicate_propagated",
                label_space_ver=label_space_ver,
                score=float(assignment.score or 0.0),
                extra_json=extra_json,
            )
            propagated += 1

    primary_session.commit()
    LOGGER.info(
        "duplicate_propagation_complete",
        extra={"label_space": label_space_ver, "assignments": propagated},
    )
    return propagated


__all__ = ["propagate_duplicate_labels"]


