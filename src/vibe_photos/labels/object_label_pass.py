"""Region-based zero-shot object labeling (M2 label layer)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sqlalchemy import select

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import Label, Region, RegionEmbedding, open_primary_session, open_projection_session
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"command": "object_label_pass"})


def _load_prototypes(cache_root: Path, prototype_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = cache_root / "label_text_prototypes" / f"{prototype_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prototype file not found: {path}")

    data = np.load(path, allow_pickle=False)
    return data["label_ids"], data["prototypes"]


def _load_region_rows(projection_session, embedding_model_name: str) -> Iterable[Tuple[str, str, str]]:
    rows = projection_session.execute(
        select(RegionEmbedding.region_id, RegionEmbedding.embedding_path, Region.image_id)
        .join(Region, Region.id == RegionEmbedding.region_id)
        .where(RegionEmbedding.model_name == embedding_model_name)
    )
    return rows.all()


def run_object_label_pass(
    *,
    primary_session,
    projection_session,
    settings: Settings,
    cache_root: Path,
    label_space_ver: str | None = None,
    prototype_name: str | None = None,
) -> None:
    """Compute zero-shot object labels for regions and aggregate to images."""

    label_space = label_space_ver or settings.label_spaces.object_current
    proto_name = prototype_name or settings.label_spaces.object_current

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

    embedding_model_name = settings.models.embedding.resolved_model_name()
    emb_rows = _load_region_rows(projection_session, embedding_model_name)
    if not emb_rows:
        LOGGER.info("object_label_pass_no_regions", extra={})
        return

    score_min = float(settings.object.zero_shot.score_min)
    margin_min = float(settings.object.zero_shot.margin_min)
    top_k = int(settings.object.zero_shot.top_k)
    agg_min_regions = int(settings.object.aggregation.min_regions)
    agg_score_min = float(settings.object.aggregation.score_min)

    label_matrix = prototypes.astype(np.float32)
    image_best: Dict[str, Dict[int, float]] = defaultdict(dict)
    image_label_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    total_regions = len(emb_rows)
    progress_interval = max(1, total_regions // 20) if total_regions else 0

    for idx, row in enumerate(emb_rows, start=1):
        region_id = row.region_id
        embedding_rel = row.embedding_path
        image_id = row.image_id

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

        sims = label_matrix @ region_vec
        top_indices = sims.argsort()[::-1][: top_k or len(sims)]
        top_scores = sims[top_indices]

        if top_scores.size == 0:
            continue

        top1_score = float(top_scores[0])
        second_score = float(top_scores[1]) if top_scores.size > 1 else 0.0
        margin = top1_score - second_score

        accepted_indices = [i for i, score in zip(top_indices, top_scores) if score >= score_min and score >= 0.0]
        if not accepted_indices or margin < margin_min:
            continue

        # Write region-level assignments for accepted labels (default top-k filtered by score).
        for rank, proto_idx in enumerate(accepted_indices):
            label_id = int(label_ids[proto_idx])
            label = label_by_id[label_id]
            score_val = float(top_scores[rank])
            extra = json.dumps({"sim_rank": rank, "top1_score": top1_score, "margin": margin})

            repo.upsert_label_assignment(
                target_type="region",
                target_id=region_id,
                label=label,
                source="zero_shot",
                label_space_ver=label_space,
                score=score_val,
                extra_json=extra,
            )

            current_best = image_best[image_id].get(label_id, 0.0)
            if score_val > current_best:
                image_best[image_id][label_id] = score_val

            if score_val >= agg_score_min:
                image_label_counts[image_id][label_id] += 1

        if progress_interval and idx % progress_interval == 0:
            percent = round(idx * 100.0 / max(total_regions, 1), 1)
            LOGGER.info("object_label_pass_progress", extra={"processed": idx, "total": total_regions, "percent": percent})

    # Aggregate to image level: promote labels that appear on >= min regions.
    for image_id, label_scores in image_best.items():
        for label_id, score in label_scores.items():
            num_regions = image_label_counts[image_id].get(label_id, 0)
            if num_regions < agg_min_regions:
                continue

            label = label_by_id[label_id]
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
    parser.add_argument("--data-db", type=Path, default=Path("data/index.db"), help="Primary DB path (labels target).")
    parser.add_argument("--cache-db", type=Path, default=Path("cache/index.db"), help="Cache DB path (regions source).")
    parser.add_argument("--cache-root", type=Path, default=Path("cache"), help="Cache root containing embeddings.")
    parser.add_argument("--label-space", type=str, default="object_v1", help="Label space version for assignments.")
    parser.add_argument("--prototype", type=str, default="object_v1", help="Prototype file name (without .npz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    with open_primary_session(args.data_db) as primary_session, open_projection_session(args.cache_db) as projection_session:
        run_object_label_pass(
            primary_session=primary_session,
            projection_session=projection_session,
            settings=settings,
            cache_root=args.cache_root,
            label_space_ver=args.label_space,
            prototype_name=args.prototype,
        )


if __name__ == "__main__":
    main()
