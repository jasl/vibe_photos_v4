"""CLI to dump object label errors for detailed analysis.

This tool compares **pipeline object labels** in the label layer against a
ground-truth file in the M2 schema and writes JSONL dumps for further
inspection or GPT-based analysis.

It uses the same evaluation convention as :mod:`vibe_photos.eval.labels`:

- Ground truth format:

  {
    "image_id": "...",
    "objects": ["object.electronics.laptop", "object.food.pizza"]
  }

- Predictions are read from the label layer:

  - `target_type='image'`
  - `label_space_ver = settings.label_spaces.object_current`
  - `labels.level='object'`
  - **all sources** (e.g. `aggregated`, `classifier`, `manual`, `duplicate_propagated`)

Typical usage:

    uv run python -m vibe_photos.eval.object_errors \\
      --gt tmp/ground_truth_human.audited.json \\
      --output-dir tmp

This will write a JSONL file `tmp/eval_object_errors.jsonl` with entries like:

  {
    "image_id": "...",
    "gt_objects": [...],
    "pred_topk": [...],
    "hit_top1": false,
    "hit_top3": false,
    "hit_top5": true
  }

You can then bucket or sample from this file to inspect systematic failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from vibe_photos.config import Settings, load_settings
from vibe_photos.db import Label, LabelAssignment, open_primary_session


app = typer.Typer(add_completion=False)


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


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _build_gt_objects_index(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return mapping image_id -> list[object_label_key] from GT records."""

    index: dict[str, list[str]] = {}
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            continue
        objs = _as_str_list(record.get("objects"))
        if objs:
            index[image_id] = objs
    return index


def _predict_objects_from_label_layer(
    session: Session,
    settings: Settings,
    image_id: str,
    *,
    max_k: int,
) -> list[str]:
    """Return top-k object label keys from the label layer for a single image."""

    rows = session.execute(
        select(Label.key, LabelAssignment.score)
        .select_from(LabelAssignment)
        .join(Label, Label.id == LabelAssignment.label_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id == image_id,
            LabelAssignment.label_space_ver == settings.label_spaces.object_current,
            Label.level == "object",
        )
        .order_by(LabelAssignment.score.desc())
    ).all()

    if not rows:
        return []

    keys: list[str] = [str(row.key) for row in rows]
    if max_k > 0 and len(keys) > max_k:
        return keys[:max_k]
    return keys


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
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        help="Directory to write eval_object_errors.jsonl (default: current directory).",
    ),
    max_k: int = typer.Option(
        5,
        "--max-k",
        min=1,
        max=20,
        help="Maximum number of predicted object labels to consider per image (default: 5).",
    ),
) -> None:
    """Dump object prediction errors from the label layer for analysis."""

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

    typer.echo(
        f"Loaded {len(records)} GT records, "
        f"{len(gt_objects)} images have non-empty objects.",
    )

    with open_primary_session(primary_url) as session:
        common_ids = sorted(
            {
                image_id
                for image_id in gt_objects.keys()
                if _predict_objects_from_label_layer(session, settings, image_id, max_k=1) is not None
            }
        )

        if not common_ids:
            typer.echo("No overlapping images between ground truth and label layer; nothing to evaluate.")
            raise typer.Exit(code=1)

        total = 0
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        errors: list[dict[str, Any]] = []

        for image_id in common_ids:
            gt_list = gt_objects.get(image_id, [])
            if not gt_list:
                continue
            gt_set = set(gt_list)

            preds = _predict_objects_from_label_layer(session, settings, image_id, max_k=max_k)
            if preds is None:
                continue

            total += 1
            top1 = preds[:1]
            top3 = preds[:3]
            top5 = preds[:5]

            hit1 = any(label in gt_set for label in top1)
            hit3 = any(label in gt_set for label in top3)
            hit5 = any(label in gt_set for label in top5)

            if hit1:
                top1_hits += 1
            if hit3:
                top3_hits += 1
            if hit5:
                top5_hits += 1

            if not hit3:
                errors.append(
                    {
                        "image_id": image_id,
                        "gt_objects": gt_list,
                        "pred_topk": preds,
                        "hit_top1": hit1,
                        "hit_top3": hit3,
                        "hit_top5": hit5,
                    }
                )

    if total == 0:
        typer.echo("No usable images for evaluation; nothing to write.")
        raise typer.Exit(code=1)

    top1 = top1_hits / total
    top3 = top3_hits / total
    top5 = top5_hits / total

    typer.echo("\n=== Object labels from label layer ===")
    typer.echo(f"Total images with GT objects considered: {total}")
    typer.echo(f"Top-1 accuracy: {top1_hits}/{total} = {top1:.3f}")
    typer.echo(f"Top-3 accuracy: {top3_hits}/{total} = {top3:.3f}")
    typer.echo(f"Top-5 accuracy: {top5_hits}/{total} = {top5:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    err_path = output_dir / "eval_object_errors.jsonl"
    with err_path.open("w", encoding="utf-8") as handle:
        for err in errors:
            handle.write(json.dumps(err, ensure_ascii=False) + "\n")

    typer.echo(f"\nWrote {len(errors)} error samples (hit_top3 == false) to {err_path}")


if __name__ == "__main__":
    app()


