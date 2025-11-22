"""Seed core labels (scene, attributes, objects) for M2 label layer.

This is intentionally opinionated for the prototype: correctness and M2-ready
structure are prioritized over backward compatibility with M1 schemas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.db import open_primary_session
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"command": "seed_labels"})


@dataclass(frozen=True)
class BaseLabelSpec:
    key: str
    display_name: str
    level: str
    parent: str | None = None
    aliases_en: Sequence[str] = ()
    aliases_zh: Sequence[str] = ()


SCENE_LABEL_SPECS: tuple[BaseLabelSpec, ...] = (
    BaseLabelSpec(key="scene", display_name="Scene", level="scene"),
    BaseLabelSpec(key="scene.landscape", display_name="Landscape", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.snapshot", display_name="Snapshot", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.people", display_name="People", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.food", display_name="Food", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.product", display_name="Product", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.document", display_name="Document", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.screenshot", display_name="Screenshot", level="scene", parent="scene"),
    BaseLabelSpec(key="scene.other", display_name="Other", level="scene", parent="scene"),
)


ATTRIBUTE_LABEL_SPECS: tuple[BaseLabelSpec, ...] = (
    BaseLabelSpec(key="attr", display_name="Attribute", level="attribute"),
    BaseLabelSpec(key="attr.has_person", display_name="Has Person", level="attribute", parent="attr"),
    BaseLabelSpec(key="attr.has_text", display_name="Has Text", level="attribute", parent="attr"),
    BaseLabelSpec(key="attr.is_document", display_name="Is Document", level="attribute", parent="attr"),
    BaseLabelSpec(key="attr.is_screenshot", display_name="Is Screenshot", level="attribute", parent="attr"),
)


def _seed_base_specs(repo: LabelRepository, specs: Iterable[BaseLabelSpec]) -> None:
    for spec in specs:
        label = repo.get_or_create_label(
            key=spec.key,
            level=spec.level,
            display_name=spec.display_name,
            parent_key=spec.parent,
        )
        repo.ensure_aliases(label, spec.aliases_en, language="en")
        repo.ensure_aliases(label, spec.aliases_zh, language="zh")


def seed_labels(session: Session) -> None:
    """Seed scene + attribute + object labels idempotently."""

    repo = LabelRepository(session)
    _seed_base_specs(repo, SCENE_LABEL_SPECS)
    _seed_base_specs(repo, ATTRIBUTE_LABEL_SPECS)

    # Reuse the existing object seeds to avoid drift.
    from vibe_photos.labels.seed_object_labels import seed_object_labels

    seed_object_labels(session)
    session.commit()
    LOGGER.info("seed_labels_complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed base labels into the primary database.")
    parser.add_argument("--db", type=Path, default=Path("data/index.db"), help="Path to primary data DB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open_primary_session(args.db) as session:
        seed_labels(session)


if __name__ == "__main__":
    main()
