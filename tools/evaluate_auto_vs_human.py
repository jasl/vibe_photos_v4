#!/usr/bin/env python3
"""Compare Qwen-generated auto ground truth against human-audited ground truth.

This script is intended to answer the question:

- Given `ground_truth_auto.jsonl` (from `tools/qwen_to_ground_truth.py`) and
  `ground_truth_human.audited.json` (manually corrected subset sampled via
  `tools/sample_ground_truth_for_review.py`),
- How well do the auto labels match the human ones?

It works entirely in the **M2 ground truth schema**:

{
  "image_id": "7871af0b9fea560c",
  "scene": ["scene.product"],
  "attributes": {
    "attr.has_person": false,
    "attr.has_text": true,
    "attr.has_animal": false,
    "attr.is_document": false,
    "attr.is_screenshot": false
  },
  "objects": [
    "object.electronics.laptop",
    "object.electronics.peripheral.keyboard"
  ]
}

Outputs:

- Scene accuracy + a simple confusion table (human → auto).
- Per-attribute precision / recall / F1 (treating each attr key as a binary label).
- "Main object" label top‑1 accuracy, where we approximate the main object as
  the **first entry in `objects`** when present.
- JSONL files with mismatched examples for further inspection or GPT analysis:
  - `eval_scene_errors.jsonl`
  - `eval_attr_<short_attr_name>_errors.jsonl`
  - `eval_main_object_errors.jsonl`
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibe_photos.labels.scene_schema import ALL_ATTRIBUTE_LABEL_KEYS, ATTRIBUTE_LABEL_KEYS


@dataclass(frozen=True)
class Sample:
    image_id: str
    scene: str | None
    attrs: dict[str, bool | None]
    objects: list[str]
    main_object_label: str | None


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


ATTR_LABEL_KEYS: list[str] = list(ALL_ATTRIBUTE_LABEL_KEYS)
ATTR_SHORT_BY_KEY: dict[str, str] = {value: key for key, value in ATTRIBUTE_LABEL_KEYS.items()}


def _load_gt_records(path: Path) -> list[dict[str, Any]]:
    """Load ground-truth style records from JSON or JSONL."""

    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: JSONL
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


def _extract_scene(record: dict[str, Any]) -> str | None:
    labels = _as_str_list(record.get("scene"))
    return labels[0] if labels else None


def _extract_attrs(record: dict[str, Any]) -> dict[str, bool | None]:
    payload = record.get("attributes") or {}
    result: dict[str, bool | None] = {}
    if not isinstance(payload, dict):
        return result
    for key in ATTR_LABEL_KEYS:
        value = payload.get(key)
        result[key] = bool(value) if isinstance(value, bool) else None
    return result


def _extract_objects(record: dict[str, Any]) -> list[str]:
    objs = record.get("objects") or []
    if not isinstance(objs, list):
        return []
    return [str(obj) for obj in objs if isinstance(obj, str)]


def _extract_main_object_label(objects: list[str]) -> str | None:
    return objects[0] if objects else None


def load_samples(path: Path) -> dict[str, Sample]:
    """Load ground-truth-like records and index them by image_id."""

    records = _load_gt_records(path)
    samples: dict[str, Sample] = {}
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            continue

        scene = _extract_scene(record)
        attrs = _extract_attrs(record)
        objects = _extract_objects(record)
        main_object_label = _extract_main_object_label(objects)

        samples[image_id] = Sample(
            image_id=image_id,
            scene=scene,
            attrs=attrs,
            objects=objects,
            main_object_label=main_object_label,
        )
    return samples


def evaluate(
    human: dict[str, Sample],
    auto: dict[str, Sample],
    *,
    output_dir: Path,
) -> None:
    """Compute metrics and write error JSONL files."""

    common_ids = sorted(set(human.keys()) & set(auto.keys()))
    total = len(common_ids)
    print(f"Found {total} images present in both human and auto ground truth.")
    if total == 0:
        return

    # Scene metrics
    scene_correct = 0
    scene_confusion: Counter[tuple[str, str]] = Counter()
    scene_errors: list[dict[str, Any]] = []

    # Attribute metrics
    attr_metrics: dict[str, BinaryMetrics] = {key: BinaryMetrics() for key in ATTR_LABEL_KEYS}
    attr_errors: dict[str, list[dict[str, Any]]] = {key: [] for key in ATTR_LABEL_KEYS}

    # Main object metrics
    main_obj_total = 0
    main_obj_correct = 0
    main_obj_confusion: Counter[tuple[str, str]] = Counter()
    main_obj_errors: list[dict[str, Any]] = []

    for image_id in common_ids:
        h = human[image_id]
        a = auto[image_id]

        # --- Scene ---
        if h.scene is not None:
            gt_scene = h.scene
            pred_scene = a.scene
            if pred_scene == gt_scene:
                scene_correct += 1
            scene_confusion[(gt_scene, pred_scene or "<none>")] += 1
            if pred_scene != gt_scene:
                scene_errors.append(
                    {
                        "image_id": image_id,
                        "human_scene": gt_scene,
                        "auto_scene": pred_scene,
                    }
                )

        # --- Attributes ---
        for attr_key, metrics in attr_metrics.items():
            gt_val = h.attrs.get(attr_key)
            pred_val = a.attrs.get(attr_key)
            if gt_val is None or pred_val is None:
                continue

            if gt_val and pred_val:
                metrics.tp += 1
            elif not gt_val and pred_val:
                metrics.fp += 1
            elif gt_val and not pred_val:
                metrics.fn += 1

            if gt_val != pred_val:
                attr_errors[attr_key].append(
                    {
                        "image_id": image_id,
                        "attr_key": attr_key,
                        "attr_name": ATTR_SHORT_BY_KEY.get(attr_key, attr_key),
                        "human": gt_val,
                        "auto": pred_val,
                    }
                )

        # --- Main object label (approximate) ---
        if h.main_object_label is not None:
            gt_obj = h.main_object_label
            pred_obj = a.main_object_label
            main_obj_total += 1
            if pred_obj == gt_obj:
                main_obj_correct += 1
            main_obj_confusion[(gt_obj, pred_obj or "<none>")] += 1
            if pred_obj != gt_obj:
                main_obj_errors.append(
                    {
                        "image_id": image_id,
                        "human_main_object": gt_obj,
                        "auto_main_object": pred_obj,
                        "human_objects": h.objects,
                        "auto_objects": a.objects,
                    }
                )

    # ----- Print summary metrics -----
    print("\n=== Scene ===")
    scene_acc = scene_correct / total if total else 0.0
    print(f"Scene accuracy: {scene_correct}/{total} = {scene_acc:.3f}")
    print("Scene confusion (human → auto, top 20):")
    for (gt, pred), count in sorted(scene_confusion.items(), key=lambda x: -x[1])[:20]:
        print(f"  {gt:18s} → {pred:18s}: {count}")

    print("\n=== Attributes (per key) ===")
    for attr_key, metrics in attr_metrics.items():
        short = ATTR_SHORT_BY_KEY.get(attr_key, attr_key)
        print(
            f"{short:15s}  P={metrics.precision:.3f}  R={metrics.recall:.3f}  "
            f"F1={metrics.f1:.3f}  (TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn})"
        )

    print("\n=== Main object label (approx. first objects[0]) ===")
    if main_obj_total:
        main_acc = main_obj_correct / main_obj_total
        print(
            f"Main object top-1 accuracy: "
            f"{main_obj_correct}/{main_obj_total} = {main_acc:.3f}"
        )
        print("Main object confusion (human → auto, top 20):")
        for (gt, pred), count in sorted(main_obj_confusion.items(), key=lambda x: -x[1])[:20]:
            print(f"  {gt:32s} → {pred:32s}: {count}")
    else:
        print("No records with non-empty human objects; skipping main-object metrics.")

    # ----- Write error JSONL files -----
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_err_path = output_dir / "eval_scene_errors.jsonl"
    with scene_err_path.open("w", encoding="utf-8") as handle:
        for err in scene_errors:
            handle.write(json.dumps(err, ensure_ascii=False) + "\n")

    for attr_key, errs in attr_errors.items():
        short = ATTR_SHORT_BY_KEY.get(attr_key, attr_key)
        attr_path = output_dir / f"eval_attr_{short}_errors.jsonl"
        with attr_path.open("w", encoding="utf-8") as handle:
            for err in errs:
                handle.write(json.dumps(err, ensure_ascii=False) + "\n")

    main_err_path = output_dir / "eval_main_object_errors.jsonl"
    with main_err_path.open("w", encoding="utf-8") as handle:
        for err in main_obj_errors:
            handle.write(json.dumps(err, ensure_ascii=False) + "\n")

    print("\nError samples written to:")
    print(f"  {scene_err_path}")
    for attr_key in ATTR_LABEL_KEYS:
        short = ATTR_SHORT_BY_KEY.get(attr_key, attr_key)
        print(f"  {output_dir / f'eval_attr_{short}_errors.jsonl'}")
    print(f"  {main_err_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen auto ground truth (ground_truth_auto.jsonl) "
            "against human-audited ground truth."
        )
    )
    parser.add_argument(
        "--human",
        type=Path,
        required=True,
        help="Path to human-audited ground truth JSON (e.g. tmp/ground_truth_human.audited.json).",
    )
    parser.add_argument(
        "--auto",
        type=Path,
        required=True,
        help="Path to auto ground truth JSON/JSONL (e.g. tmp/ground_truth_auto.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write eval_*.jsonl error dumps (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    human_samples = load_samples(args.human)
    auto_samples = load_samples(args.auto)

    if not human_samples:
        print(f"No human ground truth records loaded from {args.human}")
        return 1
    if not auto_samples:
        print(f"No auto ground truth records loaded from {args.auto}")
        return 1

    evaluate(human_samples, auto_samples, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


