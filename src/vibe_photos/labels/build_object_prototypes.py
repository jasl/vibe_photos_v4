"""Build SigLIP text prototypes for object labels (M2 label layer)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import (
    Label,
    LabelAlias,
    open_primary_session,
    sqlite_path_from_target,
)
from vibe_photos.ml.models import get_siglip_embedding_model

LOGGER = get_logger(__name__, extra={"command": "build_object_prototypes"})


def _collect_alias_texts(session: Session, label_id: int) -> dict[str | None, list[str]]:
    rows = session.execute(
        select(LabelAlias.alias_text, LabelAlias.language).where(LabelAlias.label_id == label_id)
    ).all()
    grouped: dict[str | None, list[str]] = {}
    for alias_text, language in rows:
        grouped.setdefault(language, []).append(alias_text)
    return grouped


def build_object_prototypes(
    *,
    session: Session,
    settings: Settings,
    cache_root: Path,
    output_name: str,
) -> Path:
    """Encode label aliases into SigLIP text prototypes and persist to cache."""

    siglip_processor, siglip_model, siglip_device = get_siglip_embedding_model(settings=settings)

    label_rows = session.execute(
        select(Label).where(Label.level == "object", Label.is_active.is_(True)).order_by(Label.id)
    ).scalars()

    label_ids: list[int] = []
    label_keys: list[str] = []
    prototype_vecs: list[np.ndarray] = []

    for label in label_rows:
        alias_map = _collect_alias_texts(session, label.id)
        candidates: list[str] = []
        candidates.extend(alias_map.get("en", []))
        candidates.extend(alias_map.get("zh", []))
        candidates.extend(alias_map.get(None, []))
        if not candidates:
            candidates.append(label.display_name)

        inputs = siglip_processor(text=candidates, padding=True, return_tensors="pt")
        inputs = inputs.to(siglip_device)
        with torch.no_grad():
            text_emb = siglip_model.get_text_features(**inputs)

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        proto = text_emb.mean(dim=0)
        proto = proto / proto.norm()

        label_ids.append(label.id)
        label_keys.append(label.key)
        prototype_vecs.append(proto.detach().cpu().numpy().astype(np.float32))

    if not prototype_vecs:
        raise RuntimeError("No object labels found; seed labels before building prototypes.")

    prototypes = np.stack(prototype_vecs, axis=0)
    output_dir = cache_root / "label_text_prototypes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.npz"

    np.savez(
        output_path,
        label_ids=np.asarray(label_ids, dtype=np.int64),
        label_keys=np.asarray(label_keys),
        prototypes=prototypes,
    )

    LOGGER.info(
        "build_object_prototypes_complete",
        extra={"count": len(label_ids), "output_path": str(output_path)},
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigLIP text prototypes for object labels.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Primary database URL or path. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root where prototypes will be stored. Defaults to the cache DB directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="object_v1",
        help="Name of the prototype file (produces <name>.npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    primary_target = args.db or settings.databases.primary_url
    if args.cache_root:
        cache_root = Path(args.cache_root)
    else:
        cache_root = sqlite_path_from_target(settings.databases.cache_url).parent

    with open_primary_session(primary_target) as session:
        build_object_prototypes(session=session, settings=settings, cache_root=cache_root, output_name=args.output_name)


if __name__ == "__main__":
    main()
