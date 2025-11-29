"""CLI to evaluate the Qwen-distilled object head against ground truth.

This tool evaluates the **object head** trained by
``tools/train_object_head_from_qwen.py`` directly on SigLIP image embeddings
and a small ground-truth set in the M2 schema. It bypasses the label layer and
object label pass so you can quickly answer:

- Given a set of canonical object labels per image, how often does the head
  put one of them in its top-1 / top-k predictions?

Typical usage:

    uv run python -m vibe_photos.eval.object_head \\
      --gt tmp/ground_truth_human.audited.json

You can override the head path or database/cache targets if needed:

    uv run python -m vibe_photos.eval.object_head \\
      --gt tmp/ground_truth_human.audited.json \\
      --head-path models/object_head_from_qwen.pt
"""

from __future__ import annotations

import json
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


app = typer.Typer(add_completion=False)


@dataclass
class ObjectHeadEvalResult:
    total: int
    top1_hits: int
    top3_hits: int
    top5_hits: int


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


def _build_gt_objects_index(records: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """Return mapping image_id -> set[object_label_key] from GT records."""

    index: dict[str, set[str]] = {}
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            continue

        objs = record.get("objects") or []
        if not isinstance(objs, list):
            continue

        keys = {str(obj) for obj in objs if isinstance(obj, str) and obj}
        # We only evaluate images that have at least one GT object label.
        if keys:
            index[image_id] = keys

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


def _default_object_head_path() -> Path:
    """Return the default path where the Qwen-trained object head is stored."""

    return _project_root() / "models" / "object_head_from_qwen.pt"


def _load_object_head(head_path: Path | None = None) -> tuple[torch.nn.Linear, list[str]]:
    """Load the trained object head weights for evaluation.

    Returns:
        A tuple ``(linear_layer, label_keys)`` where:

        - ``linear_layer`` is a ``torch.nn.Linear`` taking SigLIP embeddings
          and producing one logit per object label.
        - ``label_keys`` is the list of object label keys
          (e.g. ``"object.electronics.laptop"``) in the same order as logits.
    """

    path = head_path or _default_object_head_path()
    if not path.exists():
        raise RuntimeError(
            f"Object head not found at {path}. "
            "Train it first via tools/train_object_head_from_qwen.py.",
        )

    payload: dict[str, Any] = torch.load(path, map_location="cpu")
    label_keys = payload.get("label_keys")
    state = payload.get("state")
    if not isinstance(label_keys, list) or not label_keys or not isinstance(state, dict):
        raise RuntimeError(f"Invalid object head payload at {path}: expected 'label_keys' and 'state'.")

    input_dim = state.get("input_dim")
    num_labels = state.get("num_labels")
    state_dict = state.get("state_dict")
    if not isinstance(input_dim, int) or not isinstance(num_labels, int) or not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Invalid object head state at {path}: "
            f"input_dim={input_dim!r}, num_labels={num_labels!r}, state_dict={type(state_dict)!r}",
        )

    linear = torch.nn.Linear(input_dim, num_labels)
    linear.load_state_dict(state_dict)
    linear.eval()

    keys: list[str] = [str(key) for key in label_keys]
    return linear, keys


def _evaluate_object_head(
    *,
    linear: torch.nn.Linear,
    label_keys: list[str],
    gt_objects: dict[str, set[str]],
    embeddings: dict[str, np.ndarray],
    top_k: int,
) -> ObjectHeadEvalResult:
    """Compute top-k object head metrics."""

    total = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    for image_id, gt_set in gt_objects.items():
        if not gt_set:
            continue
        vec = embeddings.get(image_id)
        if vec is None:
            continue

        emb_tensor = torch.from_numpy(vec).float()
        if emb_tensor.ndim != 1:
            continue

        with torch.no_grad():
            logits = linear(emb_tensor)

        probs = torch.sigmoid(logits)
        probs_np = probs.detach().cpu().numpy()
        indices = probs_np.argsort()[::-1]

        sorted_labels = [label_keys[idx] for idx in indices[: max(top_k, 5)]]

        total += 1
        top1 = sorted_labels[:1]
        top3 = sorted_labels[:3]
        top5 = sorted_labels[:5]

        if any(label in gt_set for label in top1):
            top1_hits += 1
        if any(label in gt_set for label in top3):
            top3_hits += 1
        if any(label in gt_set for label in top5):
            top5_hits += 1

    return ObjectHeadEvalResult(
        total=total,
        top1_hits=top1_hits,
        top3_hits=top3_hits,
        top5_hits=top5_hits,
    )


@app.command()
def main(
    gt: Path = typer.Option(
        ...,
        "--gt",
        help="Path to ground truth JSON/JSONL (e.g. tmp/ground_truth_human.audited.json).",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="Primary PostgreSQL database URL. Defaults to databases.primary_url in settings.yaml.",
    ),
    cache_root: str | None = typer.Option(
        None,
        "--cache-root",
        help="Cache root directory where embeddings are stored. Defaults to cache.root.",
    ),
    head_path: Path | None = typer.Option(
        None,
        "--head-path",
        help="Path to the trained object head (.pt). Defaults to models/object_head_from_qwen.pt.",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        min=1,
        max=10,
        help="Max top-k used for evaluation (top-1, top-3, top-5 are always reported where applicable).",
    ),
) -> None:
    """Evaluate the Qwen-distilled object head on a ground-truth subset."""

    records = _load_gt_records(gt)
    if not records:
        typer.echo(f"No ground truth records loaded from {gt}")
        raise typer.Exit(code=1)

    gt_objects = _build_gt_objects_index(records)
    if not gt_objects:
        typer.echo("No records with non-empty objects in ground truth; nothing to evaluate.")
        raise typer.Exit(code=1)

    settings = load_settings()
    primary_url = db or settings.databases.primary_url
    cache_root_input = cache_root or settings.cache.root
    cache_root_path = resolve_cache_root(cache_root_input)

    typer.echo(
        f"Loaded {len(records)} GT records, "
        f"{len(gt_objects)} images have non-empty object labels.",
    )

    with open_primary_session(primary_url) as session:
        embeddings = _load_embeddings_for_ids(
            session=session,
            settings=settings,
            cache_root=cache_root_path,
            image_ids=gt_objects.keys(),
        )

    if not embeddings:
        typer.echo(
            "No SigLIP embeddings found for the labeled images. "
            "Make sure the pipeline has written image embeddings to the database.",
        )
        raise typer.Exit(code=1)

    linear, label_keys = _load_object_head(head_path)
    result = _evaluate_object_head(
        linear=linear,
        label_keys=label_keys,
        gt_objects=gt_objects,
        embeddings=embeddings,
        top_k=top_k,
    )

    if result.total == 0:
        typer.echo("No overlapping images between GT and embeddings; nothing to evaluate.")
        raise typer.Exit(code=1)

    total = result.total
    top1 = result.top1_hits / total
    top3 = result.top3_hits / total
    top5 = result.top5_hits / total

    typer.echo("\n=== Object head (image-level) ===")
    typer.echo(f"Total images with GT objects and embeddings: {total}")
    typer.echo(f"Top-1 accuracy: {result.top1_hits}/{total} = {top1:.3f}")
    typer.echo(f"Top-3 accuracy: {result.top3_hits}/{total} = {top3:.3f}")
    typer.echo(f"Top-5 accuracy: {result.top5_hits}/{total} = {top5:.3f}")


if __name__ == "__main__":
    app()


