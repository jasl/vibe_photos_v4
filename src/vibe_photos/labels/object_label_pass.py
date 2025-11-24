"""Region-based zero-shot object labeling (M2 label layer)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import (
    Label,
    LabelAssignment,
    Region,
    RegionEmbedding,
    open_cache_session,
    open_primary_session,
    sqlite_path_from_target,
)
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"command": "object_label_pass"})


class RegionEmbeddingRow(NamedTuple):
    region_id: str
    embedding_path: str
    image_id: str


def _load_prototypes(cache_root: Path, prototype_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_root / "label_text_prototypes" / f"{prototype_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prototype file not found: {path}")

    data = np.load(path, allow_pickle=False)
    return data["label_ids"], data["prototypes"]


def _load_region_rows(cache_session: Session, embedding_model_name: str) -> list[RegionEmbeddingRow]:
    rows = cache_session.execute(
        select(RegionEmbedding.region_id, RegionEmbedding.embedding_path, Region.image_id)
        .join(Region, Region.id == RegionEmbedding.region_id)
        .where(RegionEmbedding.model_name == embedding_model_name)
    ).all()

    return [
        RegionEmbeddingRow(region_id=row.region_id, embedding_path=row.embedding_path, image_id=row.image_id)
        for row in rows
    ]


def _load_scene_labels(primary_session: Session, settings: Settings) -> dict[str, str]:
    """Return best scene label key per image_id for the active scene space."""

    stmt = (
        select(LabelAssignment.target_id, Label.key, LabelAssignment.score)
        .join(Label, Label.id == LabelAssignment.label_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.label_space_ver == settings.label_spaces.scene_current,
            Label.level == "scene",
        )
    )
    best: dict[str, tuple[str, float]] = {}
    for row in primary_session.execute(stmt):
        current = best.get(row.target_id)
        if current is None or float(row.score or 0.0) > current[1]:
            best[row.target_id] = (row.key, float(row.score or 0.0))
    return {image_id: key for image_id, (key, _) in best.items()}


def run_object_label_pass(
    *,
    primary_session: Session,
    cache_session: Session,
    settings: Settings,
    cache_root: Path,
    label_space_ver: str | None = None,
    prototype_name: str | None = None,
) -> None:
    """Compute zero-shot object labels for regions and aggregate to images."""

    label_space = label_space_ver or settings.label_spaces.object_current
    proto_name = prototype_name or settings.label_spaces.object_current

    # Clear previous auto-generated assignments for idempotent reruns.
    primary_session.execute(
        delete(LabelAssignment).where(
            LabelAssignment.label_space_ver == label_space,
            LabelAssignment.target_type.in_(("image", "region")),
            LabelAssignment.source.in_(("zero_shot", "aggregated", "duplicate_propagated")),
        )
    )
    primary_session.commit()

    label_ids, prototypes = _load_prototypes(cache_root, proto_name)
    if prototypes.ndim != 2:
        raise ValueError("Prototypes must have shape [L, D].")

    repo = LabelRepository(primary_session)
    labels = (
        primary_session.execute(select(Label).where(Label.id.in_(label_ids.tolist()))).scalars().all()
    )
    label_by_id = {label.id: label for label in labels if label.id is not None}

    missing = [int(label_id) for label_id in label_ids if int(label_id) not in label_by_id]
    if missing:
        raise RuntimeError(f"Label ids missing in DB: {missing}")

    label_idx_by_id = {int(label_id): idx for idx, label_id in enumerate(label_ids.tolist())}
    label_idx_by_key = {label_by_id[lid].key: label_idx_by_id[lid] for lid in label_by_id if lid in label_idx_by_id}

    blacklist_keys = {key.strip() for key in settings.object.blacklist if key}
    remap_targets: dict[str, Label] = {}
    for src_key, dst_key in (settings.object.remap or {}).items():
        if not src_key or not dst_key:
            continue
        try:
            remap_targets[src_key] = repo.require_label(dst_key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Remap target {dst_key!r} missing in labels table") from exc

    for mapped_label in remap_targets.values():
        if mapped_label.id is not None and mapped_label.id not in label_by_id:
            label_by_id[mapped_label.id] = mapped_label

    def _resolve_label(global_idx: int) -> tuple[Label | None, int]:
        """Return the mapped label (or None if blacklisted) and its id."""

        raw_label_id = int(label_ids[global_idx])
        label = label_by_id.get(raw_label_id)
        if label is None:
            return None, raw_label_id
        if label.key in blacklist_keys:
            return None, raw_label_id
        mapped = remap_targets.get(label.key)
        if mapped is not None:
            return mapped, int(mapped.id) if mapped.id is not None else raw_label_id
        return label, raw_label_id

    embedding_model_name = settings.models.embedding.resolved_model_name()
    emb_rows = _load_region_rows(cache_session, embedding_model_name)
    if not emb_rows:
        LOGGER.info("object_label_pass_no_regions", extra={})
        return

    scene_by_image = _load_scene_labels(primary_session, settings)

    score_min = float(settings.object.zero_shot.score_min)
    margin_min = float(settings.object.zero_shot.margin_min)
    top_k = int(settings.object.zero_shot.top_k)
    agg_min_regions = int(settings.object.aggregation.min_regions)
    agg_score_min = float(settings.object.aggregation.score_min)
    scene_whitelist = set(settings.object.zero_shot.scene_whitelist or [])
    scene_fallback = {k: list(v) for k, v in (settings.object.zero_shot.scene_fallback_labels or {}).items()}

    label_matrix = prototypes.astype(np.float32)
    image_best: dict[str, dict[int, float]] = defaultdict(dict)
    image_label_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    total_regions = len(emb_rows)
    progress_interval = max(1, total_regions // 20) if total_regions else 0

    for idx, row in enumerate(emb_rows, start=1):
        region_id = row.region_id
        embedding_rel = row.embedding_path
        image_id = row.image_id

        scene_label = scene_by_image.get(image_id)
        allowed_indices: list[int] | None = None
        if scene_label and scene_label in scene_whitelist:
            allowed_indices = None  # use all labels
        elif scene_label and scene_label in scene_fallback:
            allowed_indices = [
                label_idx_by_key[key]
                for key in scene_fallback[scene_label]
                if key in label_idx_by_key
            ]
            if not allowed_indices:
                continue
        else:
            continue  # skip scenes outside whitelist/fallback

        emb_path = cache_root / "embeddings" / embedding_rel
        try:
            region_vec = np.load(emb_path).astype(np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("object_label_pass_load_error", extra={"region_id": region_id, "path": str(emb_path), "error": str(exc)})
            continue

        if region_vec.ndim != 1:
            LOGGER.error("object_label_pass_vec_dim", extra={"region_id": region_id, "shape": region_vec.shape})
            continue

        # Normalize to cosine space.
        norm = np.linalg.norm(region_vec)
        if norm == 0.0:
            continue
        region_vec = region_vec / norm

        current_matrix = label_matrix if allowed_indices is None else label_matrix[allowed_indices]
        sims = current_matrix @ region_vec
        top_indices = sims.argsort()[::-1][: top_k or len(sims)]
        top_scores = sims[top_indices]

        if top_scores.size == 0:
            continue

        top1_score = float(top_scores[0])
        second_score = float(top_scores[1]) if top_scores.size > 1 else 0.0
        margin = top1_score - second_score

        accepted_indices = [i for i, score in zip(top_indices, top_scores, strict=True) if score >= score_min and score >= 0.0]
        if not accepted_indices or margin < margin_min:
            continue

        # Write region-level assignments for accepted labels (default top-k filtered by score).
        for rank, proto_idx in enumerate(accepted_indices):
            global_idx = allowed_indices[proto_idx] if allowed_indices is not None else int(proto_idx)
            resolved_label, target_label_id = _resolve_label(global_idx)
            if resolved_label is None:
                continue

            score_val = float(top_scores[rank])
            extra = json.dumps({"sim_rank": rank, "top1_score": top1_score, "margin": margin})

            repo.upsert_label_assignment(
                target_type="region",
                target_id=region_id,
                label=resolved_label,
                source="zero_shot",
                label_space_ver=label_space,
                score=score_val,
                extra_json=extra,
            )

            current_best = image_best[image_id].get(target_label_id, 0.0)
            if score_val > current_best:
                image_best[image_id][target_label_id] = score_val

            if score_val >= agg_score_min:
                image_label_counts[image_id][target_label_id] += 1

        if progress_interval and idx % progress_interval == 0:
            percent = round(idx * 100.0 / max(total_regions, 1), 1)
            LOGGER.info("object_label_pass_progress", extra={"processed": idx, "total": total_regions, "percent": percent})

    # Aggregate to image level: promote labels that appear on >= min regions.
    for image_id, label_scores in image_best.items():
        for label_id, score in label_scores.items():
            num_regions = image_label_counts[image_id].get(label_id, 0)
            if num_regions < agg_min_regions:
                continue

            label = label_by_id.get(label_id)
            if label is None:
                continue
            repo.upsert_label_assignment(
                target_type="image",
                target_id=image_id,
                label=label,
                source="aggregated",
                label_space_ver=label_space,
                score=score,
                extra_json=json.dumps({"regions": num_regions}),
            )

    primary_session.commit()
    LOGGER.info(
        "object_label_pass_complete",
        extra={"regions": total_regions, "images": len(image_best)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run region-based zero-shot object label pass.")
    parser.add_argument(
        "--data-db",
        type=str,
        default=None,
        help="Primary database URL or path. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default=None,
        help="Cache database URL or path. Defaults to databases.cache_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root containing embeddings. Defaults to the cache DB directory.",
    )
    parser.add_argument("--label-space", type=str, default="object_v1", help="Label space version for assignments.")
    parser.add_argument("--prototype", type=str, default="object_v1", help="Prototype file name (without .npz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    primary_target = args.data_db or settings.databases.primary_url
    cache_target = args.cache_db or settings.databases.cache_url
    if args.cache_root:
        cache_root = Path(args.cache_root)
    else:
        cache_root = sqlite_path_from_target(cache_target).parent

    with open_primary_session(primary_target) as primary_session, open_cache_session(cache_target) as cache_session:
        run_object_label_pass(
            primary_session=primary_session,
            cache_session=cache_session,
            settings=settings,
            cache_root=cache_root,
            label_space_ver=args.label_space,
            prototype_name=args.prototype,
        )


if __name__ == "__main__":
    main()
