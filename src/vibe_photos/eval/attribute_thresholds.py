"""CLI to sweep attribute head thresholds against human-audited ground truth.

This tool focuses on the Qwen-trained SigLIP **attribute head** and a small
human-audited eval set in the M2 ground-truth schema. It:

- Loads `ground_truth_human.audited.json` (or any compatible JSON/JSONL file).
- Finds the corresponding SigLIP embeddings from the primary database.
- Runs the learned attribute head to get logits/probabilities per attribute.
- Sweeps thresholds to report precision/recall/F1 for each attribute.

The typical usage is:

    uv run python -m vibe_photos.eval.attribute_thresholds \
      --gt tmp/ground_truth_human.audited.json

You can optionally target specific attributes and/or override the DB/cache:

    uv run python -m vibe_photos.eval.attribute_thresholds \
      --gt tmp/ground_truth_human.audited.json \
      --attr has_person --attr has_text \
      --target-precision 0.9
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from vibe_photos.cache_helpers import resolve_cache_root
from vibe_photos.config import Settings, _project_root, load_settings
from vibe_photos.db import ImageEmbedding, open_primary_session
from vibe_photos.labels.scene_schema import ATTRIBUTE_LABEL_KEYS


app = typer.Typer(add_completion=False)


_DEFAULT_TARGET_ATTR_IDS: tuple[str, ...] = ("has_person", "has_text", "has_animal")


@dataclass
class ThresholdMetrics:
    """Binary classification metrics at a particular threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _load_gt_records(path: Path) -> list[dict[str, Any]]:
    """Load ground-truth-style records from JSON or JSONL."""

    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                obj = json.loads(line_stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        return records

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]
    return []


def _build_gt_index(
    records: Sequence[dict[str, Any]],
    target_attr_ids: Sequence[str],
) -> dict[str, dict[str, bool]]:
    """Return mapping image_id -> {attr_id -> bool} for requested attributes.

    Records without a usable ``image_id`` or without any of the requested
    attributes are skipped. Attributes with missing/non-bool values are treated
    as "not labeled" and omitted from the per-image mapping.
    """

    label_key_by_attr_id: dict[str, str] = {attr_id: ATTRIBUTE_LABEL_KEYS[attr_id] for attr_id in target_attr_ids}
    index: dict[str, dict[str, bool]] = {}

    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            continue

        attrs_payload = record.get("attributes") or {}
        if not isinstance(attrs_payload, dict):
            continue

        per_image: dict[str, bool] = {}
        for attr_id, label_key in label_key_by_attr_id.items():
            raw_value = attrs_payload.get(label_key)
            if isinstance(raw_value, bool):
                per_image[attr_id] = bool(raw_value)

        if per_image:
            index[image_id] = per_image

    return index


def _resolve_embedding_path(cache_root: Path, rel_path: str) -> Path:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return path_obj
    return cache_root / path_obj


def _load_embeddings_for_ids(
    *,
    session: Session,
    settings: Settings,
    cache_root: Path,
    image_ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Load SigLIP embeddings for the given image_ids into memory."""

    ids = {image_id for image_id in image_ids if image_id}
    if not ids:
        return {}

    model_name = settings.models.embedding.resolved_model_name()
    stmt = (
        select(ImageEmbedding.image_id, ImageEmbedding.embedding_path)
        .where(ImageEmbedding.model_name == model_name, ImageEmbedding.image_id.in_(sorted(ids)))
    )
    rows = session.execute(stmt).all()

    path_by_id = {row.image_id: row.embedding_path for row in rows}
    if not path_by_id:
        return {}

    embeddings: dict[str, np.ndarray] = {}
    for image_id, rel_path in path_by_id.items():
        emb_path = _resolve_embedding_path(cache_root, rel_path)
        try:
            vec = np.load(emb_path)
        except Exception:
            continue
        if vec.ndim != 1:
            continue
        embeddings[image_id] = vec.astype(np.float32)

    return embeddings


def _default_attribute_head_path() -> Path:
    """Return the default path where the Qwen-trained attribute head is stored."""

    return _project_root() / "models" / "attribute_head_from_qwen.pt"


def _load_attribute_head(settings: Settings) -> tuple[torch.nn.Linear, list[str]]:
    """Load the trained attribute head weights for offline evaluation.

    Returns:
        A tuple ``(linear_layer, attr_label_keys)`` where:

        - ``linear_layer`` is a ``torch.nn.Linear`` taking SigLIP embeddings
          and producing one logit per attribute.
        - ``attr_label_keys`` is the list of attribute label keys
          (e.g. ``"attr.has_person"``) in the same order as the logits.
    """

    path = _default_attribute_head_path()
    if not path.exists():
        raise RuntimeError(
            f"Attribute head not found at {path}. "
            "Train it first via tools/train_attribute_head_from_qwen.py.",
        )

    payload: dict[str, Any] = torch.load(path, map_location="cpu")
    attr_keys = payload.get("attr_keys")
    state = payload.get("state")
    if not isinstance(attr_keys, list) or not attr_keys or not isinstance(state, dict):
        raise RuntimeError(f"Invalid attribute head payload at {path}: expected 'attr_keys' and 'state'.")

    input_dim = state.get("input_dim")
    num_attrs = state.get("num_attrs")
    state_dict = state.get("state_dict")
    if not isinstance(input_dim, int) or not isinstance(num_attrs, int) or not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Invalid attribute head state at {path}: "
            f"input_dim={input_dim!r}, num_attrs={num_attrs!r}, state_dict={type(state_dict)!r}",
        )

    linear = torch.nn.Linear(input_dim, num_attrs)
    linear.load_state_dict(state_dict)
    linear.eval()

    label_keys: list[str] = [str(key) for key in attr_keys]
    return linear, label_keys


def _sigmoid(x: float) -> float:
    """Numerically stable logistic transform."""

    # Guard extreme values to avoid overflow in exp().
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _collect_attribute_scores(
    *,
    linear: torch.nn.Linear,
    attr_label_keys: list[str],
    gt_by_image: dict[str, dict[str, bool]],
    embeddings: dict[str, np.ndarray],
    target_attr_ids: Sequence[str],
) -> dict[str, list[tuple[float, bool]]]:
    """Return per-attribute lists of (probability, ground_truth_bool)."""

    label_key_by_attr_id: dict[str, str] = {attr_id: ATTRIBUTE_LABEL_KEYS[attr_id] for attr_id in target_attr_ids}
    idx_by_label_key: dict[str, int] = {str(label_key): idx for idx, label_key in enumerate(attr_label_keys)}

    result: dict[str, list[tuple[float, bool]]] = {attr_id: [] for attr_id in target_attr_ids}

    for image_id, gt_attrs in gt_by_image.items():
        vec = embeddings.get(image_id)
        if vec is None:
            continue

        try:
            emb_tensor = torch.from_numpy(vec).float()
        except Exception:
            continue
        if emb_tensor.ndim != 1:
            continue

        with torch.no_grad():
            logits = linear(emb_tensor)

        for attr_id in target_attr_ids:
            gt_value = gt_attrs.get(attr_id)
            if gt_value is None:
                continue

            label_key = label_key_by_attr_id[attr_id]
            idx = idx_by_label_key.get(label_key)
            if idx is None or idx >= logits.shape[0]:
                continue

            logit = float(logits[idx].item())
            prob = _sigmoid(logit)
            result[attr_id].append((prob, bool(gt_value)))

    return result


def _sweep_thresholds(
    pairs: Sequence[tuple[float, bool]],
    *,
    max_thresholds: int = 101,
) -> list[ThresholdMetrics]:
    """Compute metrics across a range of thresholds for a single attribute."""

    if not pairs:
        return []

    probs = sorted({prob for prob, _ in pairs})
    if not probs:
        return []

    if len(probs) > max_thresholds:
        # Sample thresholds approximately uniformly across the observed range.
        indices = np.linspace(0, len(probs) - 1, max_thresholds, dtype=int)
        unique_indices = sorted(set(int(idx) for idx in indices))
        thresholds = [probs[idx] for idx in unique_indices]
    else:
        thresholds = probs

    metrics: list[ThresholdMetrics] = []
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0

        for prob, label in pairs:
            predicted = prob >= threshold
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif not predicted and label:
                fn += 1

        denom_prec = tp + fp
        denom_rec = tp + fn
        precision = tp / denom_prec if denom_prec else 0.0
        recall = tp / denom_rec if denom_rec else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics.append(
            ThresholdMetrics(
                threshold=float(threshold),
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )

    return metrics


def _print_attribute_report(
    *,
    attr_id: str,
    pairs: Sequence[tuple[float, bool]],
    metrics: Sequence[ThresholdMetrics],
    target_precision: float,
) -> None:
    total = len(pairs)
    num_pos = sum(1 for _, label in pairs if label)
    num_neg = total - num_pos

    print(f"\n=== Attribute: {attr_id} ===")
    print(f"Samples: total={total}, positive={num_pos}, negative={num_neg}")
    if not metrics:
        print("No metrics (no usable samples).")
        return

    best_f1 = max(metrics, key=lambda m: m.f1)
    print(
        "Best F1: "
        f"thr={best_f1.threshold:.3f}  P={best_f1.precision:.3f}  "
        f"R={best_f1.recall:.3f}  F1={best_f1.f1:.3f}  "
        f"(TP={best_f1.tp}, FP={best_f1.fp}, FN={best_f1.fn})"
    )

    candidates = [m for m in metrics if m.precision >= target_precision]
    if candidates:
        best_prec = max(candidates, key=lambda m: (m.recall, m.f1))
        print(
            f"Best P≥{target_precision:.2f}: "
            f"thr={best_prec.threshold:.3f}  P={best_prec.precision:.3f}  "
            f"R={best_prec.recall:.3f}  F1={best_prec.f1:.3f}  "
            f"(TP={best_prec.tp}, FP={best_prec.fp}, FN={best_prec.fn})"
        )
    else:
        print(f"No threshold reaches precision ≥ {target_precision:.2f}.")

    # Also print a few reference thresholds near current defaults (0.5, 0.7, 0.9).
    ref_points = [0.5, 0.7, 0.9]
    print("Reference thresholds:")
    for ref in ref_points:
        closest = min(metrics, key=lambda m: abs(m.threshold - ref))
        print(
            f"  ~{ref:.2f}: thr={closest.threshold:.3f}  "
            f"P={closest.precision:.3f}  R={closest.recall:.3f}  F1={closest.f1:.3f}  "
            f"(TP={closest.tp}, FP={closest.fp}, FN={closest.fn})"
        )


@app.command()
def main(
    gt: Path = typer.Option(
        ...,
        "--gt",
        help="Path to human-audited ground truth JSON/JSONL "
        "(e.g. tmp/ground_truth_human.audited.json).",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="Primary PostgreSQL URL. Defaults to databases.primary_url in settings.yaml.",
    ),
    cache_root: str | None = typer.Option(
        None,
        "--cache-root",
        help="Cache root directory containing SigLIP embeddings. Defaults to cache.root.",
    ),
    attr: list[str] = typer.Option(
        None,
        "--attr",
        help=(
            "Attribute ids to tune (e.g. --attr has_person --attr has_text). "
            "Defaults to has_person, has_text, has_animal."
        ),
    ),
    target_precision: float = typer.Option(
        0.9,
        "--target-precision",
        min=0.0,
        max=1.0,
        help="Precision target used when recommending a high-precision threshold.",
    ),
    max_thresholds: int = typer.Option(
        101,
        "--max-thresholds",
        min=10,
        max=1000,
        help="Maximum number of distinct thresholds to evaluate per attribute.",
    ),
) -> None:
    """Sweep attribute head thresholds and report per-attribute P/R/F1 curves."""

    records = _load_gt_records(gt)
    if not records:
        print(f"No ground truth records loaded from {gt}")
        raise typer.Exit(code=1)

    settings = load_settings()
    primary_url = db or settings.databases.primary_url
    cache_root_input = cache_root or settings.cache.root
    cache_root_path = resolve_cache_root(cache_root_input)

    target_attr_ids: list[str] = list(attr) if attr else list(_DEFAULT_TARGET_ATTR_IDS)
    unknown = [name for name in target_attr_ids if name not in ATTRIBUTE_LABEL_KEYS]
    if unknown:
        valid = ", ".join(sorted(ATTRIBUTE_LABEL_KEYS.keys()))
        raise typer.BadParameter(f"Unknown attribute ids: {unknown}. Valid options: {valid}")

    gt_by_image = _build_gt_index(records, target_attr_ids)
    if not gt_by_image:
        print("No usable attribute labels found in ground truth for the requested attributes.")
        raise typer.Exit(code=1)

    print(
        f"Loaded {len(records)} ground-truth records, "
        f"{len(gt_by_image)} images have at least one labeled target attribute.",
    )

    with open_primary_session(primary_url) as session:
        embeddings = _load_embeddings_for_ids(
            session=session,
            settings=settings,
            cache_root=cache_root_path,
            image_ids=gt_by_image.keys(),
        )
        if not embeddings:
            print(
                "No SigLIP embeddings found for the labeled images. "
                "Make sure the pipeline has written image embeddings to the database.",
            )
            raise typer.Exit(code=1)

        linear, attr_label_keys = _load_attribute_head(settings)
        per_attr_pairs = _collect_attribute_scores(
            linear=linear,
            attr_label_keys=attr_label_keys,
            gt_by_image=gt_by_image,
            embeddings=embeddings,
            target_attr_ids=target_attr_ids,
        )

    print(
        f"Collected attribute probabilities for "
        f"{sum(len(pairs) for pairs in per_attr_pairs.values())} (image, attribute) pairs.",
    )

    for attr_id in target_attr_ids:
        pairs = per_attr_pairs.get(attr_id, [])
        metrics = _sweep_thresholds(pairs, max_thresholds=max_thresholds)
        _print_attribute_report(
            attr_id=attr_id,
            pairs=pairs,
            metrics=metrics,
            target_precision=target_precision,
        )


if __name__ == "__main__":
    app()


