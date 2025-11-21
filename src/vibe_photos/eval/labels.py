"""CLI to evaluate label assignments against a ground-truth subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import typer
from sqlalchemy import select

from vibe_photos.config import load_settings
from vibe_photos.db import Label, LabelAssignment, open_primary_session
from vibe_photos.labels.scene_schema import ATTRIBUTE_LABEL_KEYS

app = typer.Typer(add_completion=False)


@app.command()
def main(
    gt: Path = typer.Option(..., "--gt", help="Path to ground truth JSON or JSONL file."),
    db: Path = typer.Option(Path("data/index.db"), "--db", help="Path to data/index.db."),
) -> None:
    """Evaluate scene/attribute/object labels."""

    records = _load_ground_truth(gt)
    if not records:
        typer.echo("No ground truth records found.")
        raise typer.Exit(code=1)

    settings = load_settings()
    with open_primary_session(db) as session:
        metrics = _compute_metrics(session, settings, records)

    _print_metrics(metrics)


def _load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
    except json.JSONDecodeError:
        # Try JSONL
        lines: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                lines.append(obj)
        return lines
    return []


def _compute_metrics(session, settings, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scene_total = 0
    scene_correct = 0

    attr_metrics: Dict[str, Dict[str, int]] = {
        key: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for key in ATTRIBUTE_LABEL_KEYS.values()
    }

    object_total = 0
    object_top1 = 0
    object_top3 = 0

    for record in records:
        image_id = record.get("image_id")
        if not image_id:
            continue

        gt_scene = set(_as_list(record.get("scene")))
        predicted_scene = _predict_scene(session, image_id, settings)
        if gt_scene:
            scene_total += 1
            if predicted_scene and predicted_scene in gt_scene:
                scene_correct += 1

        gt_attrs = record.get("attributes") or {}
        predicted_attrs = _predict_attributes(session, image_id, settings)
        for attr_key, stats in attr_metrics.items():
            if attr_key not in gt_attrs:
                continue
            expected = bool(gt_attrs[attr_key])
            predicted = attr_key in predicted_attrs
            if predicted and expected:
                stats["tp"] += 1
            elif predicted and not expected:
                stats["fp"] += 1
            elif not predicted and expected:
                stats["fn"] += 1
            else:
                stats["tn"] += 1

        gt_objects = set(_as_list(record.get("objects")))
        predicted_objects = _predict_objects(session, image_id, settings)
        if gt_objects:
            object_total += 1
            if predicted_objects:
                if any(obj in gt_objects for obj in predicted_objects[:1]):
                    object_top1 += 1
                if any(obj in gt_objects for obj in predicted_objects[:3]):
                    object_top3 += 1

    return {
        "scene": {"total": scene_total, "correct": scene_correct},
        "attributes": attr_metrics,
        "objects": {
            "total": object_total,
            "top1": object_top1,
            "top3": object_top3,
        },
    }


def _predict_scene(session, image_id: str, settings) -> str | None:
    rows = session.execute(
        select(Label.key, LabelAssignment.score)
        .join(Label, Label.id == LabelAssignment.label_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id == image_id,
            LabelAssignment.label_space_ver == settings.label_spaces.scene_current,
            Label.level == "scene",
        )
        .order_by(LabelAssignment.score.desc())
    ).all()
    if not rows:
        return None
    return rows[0].key


def _predict_attributes(session, image_id: str, settings) -> set[str]:
    rows = session.execute(
        select(Label.key)
        .join(Label, Label.id == LabelAssignment.label_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id == image_id,
            LabelAssignment.label_space_ver == settings.label_spaces.scene_current,
            Label.level == "attribute",
        )
    ).all()
    return {row.key for row in rows}


def _predict_objects(session, image_id: str, settings) -> List[str]:
    rows = session.execute(
        select(Label.key, LabelAssignment.score)
        .join(Label, Label.id == LabelAssignment.label_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.target_id == image_id,
            LabelAssignment.label_space_ver == settings.label_spaces.object_current,
            Label.level == "object",
        )
        .order_by(LabelAssignment.score.desc())
    ).all()
    return [row.key for row in rows]


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def _print_metrics(metrics: Dict[str, Any]) -> None:
    scene = metrics["scene"]
    scene_total = max(1, scene["total"])
    scene_acc = scene["correct"] / scene_total
    typer.echo(f"Scene accuracy: {scene['correct']}/{scene_total} ({scene_acc * 100:.2f}%)")

    typer.echo("\nAttributes:")
    for attr_key, stats in metrics["attributes"].items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        typer.echo(
            f"  {attr_key}: precision={prec:.2f} recall={rec:.2f} "
            f"(tp={tp} fp={fp} fn={fn})"
        )

    objs = metrics["objects"]
    total = max(1, objs["total"])
    top1 = objs["top1"] / total
    top3 = objs["top3"] / total
    typer.echo(
        f"\nObject labels: top-1 {objs['top1']}/{total} ({top1 * 100:.2f}%), "
        f"top-3 {objs['top3']}/{total} ({top3 * 100:.2f}%)"
    )


if __name__ == "__main__":
    app()


