"""Region-based zero-shot object labeling (M2 label layer)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.cache_helpers import resolve_cache_root
from vibe_photos.config import Settings, _project_root, load_settings
from vibe_photos.db import (
    ImageEmbedding,
    Label,
    LabelAssignment,
    Region,
    RegionEmbedding,
    open_primary_session,
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


def _resolve_region_embedding_path(cache_root: Path, rel_path: str) -> Path:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return path_obj
    return cache_root / path_obj


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


def _load_image_embedding_rows(cache_session: Session, model_name: str) -> dict[str, str]:
    """Return mapping image_id -> embedding_path for image-level SigLIP embeddings."""

    rows = cache_session.execute(
        select(ImageEmbedding.image_id, ImageEmbedding.embedding_path).where(
            ImageEmbedding.model_name == model_name
        )
    ).all()
    return {row.image_id: row.embedding_path for row in rows}


def _default_object_head_path() -> Path:
    """Return the default path where the Qwen-trained object head is stored."""

    return _project_root() / "models" / "object_head_from_qwen.pt"


def _load_learned_object_head() -> tuple[torch.nn.Linear, list[str]] | None:
    """Load a trained object head from disk if available.

    Returns ``None`` when the file is missing or invalid, in which case callers
    should fall back to zero-shot only.
    """

    path = _default_object_head_path()
    if not path.exists():
        LOGGER.info("object_head_missing", extra={"path": str(path)})
        return None

    try:
        payload: dict[str, Any] = torch.load(path, map_location="cpu")  # type: ignore[name-defined]
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("object_head_load_error", extra={"path": str(path), "error": str(exc)})
        return None

    label_keys = payload.get("label_keys")
    state = payload.get("state")
    if not isinstance(label_keys, list) or not label_keys or not isinstance(state, dict):
        LOGGER.error(
            "object_head_payload_invalid",
            extra={"path": str(path), "keys": list(payload.keys())},
        )
        return None

    input_dim = state.get("input_dim")
    num_labels = state.get("num_labels")
    state_dict = state.get("state_dict")
    if not isinstance(input_dim, int) or not isinstance(num_labels, int) or not isinstance(state_dict, dict):
        LOGGER.error(
            "object_head_state_invalid",
            extra={
                "path": str(path),
                "has_input_dim": isinstance(input_dim, int),
                "has_num_labels": isinstance(num_labels, int),
                "has_state_dict": isinstance(state_dict, dict),
            },
        )
        return None

    try:
        linear = torch.nn.Linear(input_dim, num_labels)
        linear.load_state_dict(state_dict)
        linear.eval()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("object_head_build_error", extra={"path": str(path), "error": str(exc)})
        return None

    LOGGER.info(
        "object_head_loaded",
        extra={
            "path": str(path),
            "num_labels": num_labels,
        },
    )
    return linear, [str(key) for key in label_keys]


def run_object_label_pass(
    *,
    primary_session: Session,
    cache_session: Session,
    settings: Settings,
    cache_root: Path,
    label_space_ver: str | None = None,
    prototype_name: str | None = None,
) -> None:
    """Compute object labels for regions and images.

    This pass combines:

    - Region-level zero-shot labeling using SigLIP text prototypes.
    - Image-level aggregation of region labels.
    - Optional image-level object head predictions distilled from Qwen
      (when ``models/object_head_from_qwen.pt`` is present).
    """

    label_space = label_space_ver or settings.label_spaces.object_current
    proto_name = prototype_name or settings.label_spaces.object_current

    # Clear previous auto-generated assignments for idempotent reruns.
    primary_session.execute(
        delete(LabelAssignment).where(
            LabelAssignment.label_space_ver == label_space,
            LabelAssignment.target_type.in_(("image", "region")),
            LabelAssignment.source.in_(("zero_shot", "aggregated", "duplicate_propagated", "classifier")),
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

        emb_path = _resolve_region_embedding_path(cache_root, embedding_rel)
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

    # Optionally run the learned image-level object head as an additional
    # source of object labels. This uses SigLIP image embeddings and writes
    # assignments with ``source='classifier'``.
    learned_head = _load_learned_object_head()
    if learned_head is not None:
        linear, head_label_keys = learned_head
        device = next(linear.parameters()).device

        # Map head label keys to Label rows in the DB.
        head_labels: dict[str, Label] = {}
        for key in head_label_keys:
            try:
                label = repo.require_label(key)
            except ValueError:
                LOGGER.error("object_head_label_missing", extra={"label_key": key})
                continue
            head_labels[key] = label

        if head_labels:
            embedding_paths = _load_image_embedding_rows(cache_session, embedding_model_name)
            allowed_scenes = set(scene_whitelist) | set(scene_fallback.keys())

            for image_id, rel_path in embedding_paths.items():
                scene_label = scene_by_image.get(image_id)
                if allowed_scenes and scene_label not in allowed_scenes:
                    continue

                emb_path = _resolve_region_embedding_path(cache_root, rel_path)
                try:
                    vec = np.load(emb_path).astype(np.float32)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.error(
                        "object_head_embedding_load_error",
                        extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
                    )
                    continue

                if vec.ndim != 1:
                    LOGGER.error(
                        "object_head_invalid_embedding_shape",
                        extra={"image_id": image_id, "path": str(emb_path), "shape": tuple(vec.shape)},
                    )
                    continue

                emb_tensor = torch.from_numpy(vec).float().to(device)
                if emb_tensor.ndim != 1:
                    LOGGER.error(
                        "object_head_invalid_embedding_tensor_shape",
                        extra={"image_id": image_id, "shape": tuple(emb_tensor.shape)},
                    )
                    continue

                with torch.no_grad():
                    logits = linear(emb_tensor)
                    probs = torch.sigmoid(logits)

                probs_np = probs.detach().cpu().numpy()
                indices = probs_np.argsort()[::-1]
                max_k = max(top_k or 0, 5)
                indices = indices[:max_k]

                for rank, idx in enumerate(indices):
                    prob = float(probs_np[idx])
                    # Simple default: only promote reasonably confident labels.
                    if prob < 0.5:
                        continue
                    label_key = head_label_keys[idx]
                    label = head_labels.get(label_key)
                    if label is None:
                        continue

                    extra = json.dumps(
                        {
                            "head_rank": rank,
                            "head_prob": prob,
                        }
                    )
                    repo.upsert_label_assignment(
                        target_type="image",
                        target_id=image_id,
                        label=label,
                        source="classifier",
                        label_space_ver=label_space,
                        score=prob,
                        extra_json=extra,
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
        help="Primary PostgreSQL database URL. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        dest="cache_root",
        help="Cache root directory containing embeddings. Defaults to cache.root.",
    )
    parser.add_argument("--label-space", type=str, default="object_v1", help="Label space version for assignments.")
    parser.add_argument("--prototype", type=str, default="object_v1", help="Prototype file name (without .npz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    primary_target = args.data_db or settings.databases.primary_url
    cache_target_input = args.cache_root or settings.cache.root
    cache_root = resolve_cache_root(cache_target_input)

    with open_primary_session(primary_target) as primary_session:
        cache_session = primary_session
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
