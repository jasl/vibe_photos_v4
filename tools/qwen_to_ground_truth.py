#!/usr/bin/env python3
"""Convert Qwen3-VL JSONL annotations into M2 evaluation ground truth.

This script is intended to run on the local machine (where data/index.db lives)
after copying the JSONL produced by ``tools/qwen_vl_annotate_batch.py`` from
the H100 server.

Input JSONL (one record per line), as produced by qwen_vl_annotate_batch:

{
  "image_id": "8f14e45fceea167a5a36dedd4bea2543",
  "server_path": "/mnt/photos/2024/IMG_1234.JPG",
  "image_path": "/mnt/photos/2024/IMG_1234.JPG",
  "rel_path": "2024/IMG_1234.JPG",
  "annotation": {
    "scene": "product",
    "has_person": false,
    "has_text": false,
    "is_document": false,
    "is_screenshot": false,
    "objects": [
      {
        "name_free": "mechanical keyboard",
        "coarse_type": "keyboard",
        "is_main": true,
        "confidence": 0.92
      }
    ]
  }
}

Output JSONL (one ground-truth record per line) matching the expectations of
``vibe_photos.eval.labels``:

{
  "image_id": "8f14e45fceea167a5a36dedd4bea2543",
  "scene": ["scene.product"],
  "attributes": {
    "attr.has_person": false,
    "attr.has_text": false,
    "attr.is_document": false,
    "attr.is_screenshot": false
  },
  "objects": ["object.electronics.peripheral.keyboard"]
}
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SceneMapping = dict[str, str]
AttrMapping = dict[str, str]
ObjectMapping = dict[str, str]


def _default_scene_mapping() -> SceneMapping:
    return {
        "landscape": "scene.landscape",
        "snapshot": "scene.snapshot",
        "people": "scene.people",
        "selfie": "scene.people",
        "food": "scene.food",
        "product": "scene.product",
        "document": "scene.document",
        "screenshot": "scene.screenshot",
        "other": "scene.other",
    }


def _default_attr_mapping() -> AttrMapping:
    return {
        "has_person": "attr.has_person",
        "has_text": "attr.has_text",
        "is_document": "attr.is_document",
        "is_screenshot": "attr.is_screenshot",
    }


def _default_object_mapping() -> ObjectMapping:
    # Map Qwen coarse_type enums to M2 object label keys seeded in
    # src/vibe_photos/labels/seed_object_labels.py. Many-to-one mappings are fine
    # here; this is only for eval / weak supervision.
    return {
        "electronics": "object.electronics",
        "phone": "object.electronics.phone",
        "laptop": "object.electronics.laptop",
        "tablet": "object.electronics.tablet",
        "computer_case": "object.electronics",
        "monitor": "object.electronics.display.monitor",
        "keyboard": "object.electronics.peripheral.keyboard",
        "mouse": "object.electronics.peripheral.mouse",
        "earphones": "object.electronics.audio.earbuds",
        "headphones": "object.electronics.audio.headphones",
        "camera": "object.electronics.camera",
        "game_console": "object.electronics.console",
        "controller": "object.electronics.console.controller",
        "food": "object.food",
        "drink": "object.drink",
        "dessert": "object.food.dessert",
        "paper_document": "object.document.paper",
        "screen": "object.electronics.display.monitor",
        "packaging": "object.dailystuff",
        "card": "object.document.bank_card",
        "id_card": "object.document.id_card",
    }


def _iter_input_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                rec = json.loads(line_stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _map_scene(scene_raw: Any, scene_map: SceneMapping) -> list[str]:
    if not isinstance(scene_raw, str):
        return []
    key = scene_map.get(scene_raw.strip().lower())
    if not key:
        return ["scene.other"]
    return [key]


def _map_attributes(annotation: dict[str, Any], attr_map: AttrMapping) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for src_key, dst_key in attr_map.items():
        value = annotation.get(src_key)
        if isinstance(value, bool):
            result[dst_key] = bool(value)
    return result


def _map_objects(
    objects: Any,
    object_map: ObjectMapping,
    *,
    min_confidence: float,
    only_main: bool,
) -> list[str]:
    if not isinstance(objects, list):
        return []

    labels: set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        coarse_type = obj.get("coarse_type")
        if not isinstance(coarse_type, str):
            continue
        coarse_type_norm = coarse_type.strip().lower()

        if only_main and not bool(obj.get("is_main", False)):
            continue

        confidence_raw = obj.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue

        label_key = object_map.get(coarse_type_norm)
        if not label_key:
            continue
        labels.add(label_key)

    return sorted(labels)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-VL JSONL annotations into M2 ground_truth.json(l)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSONL file produced by tools/qwen_vl_annotate_batch.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for ground truth JSONL (one record per line).",
    )
    parser.add_argument(
        "--min-object-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for mapping an object coarse_type (default: 0.7).",
    )
    parser.add_argument(
        "--include-non-main",
        action="store_true",
        help="Include objects even when is_main is False (default: only use is_main=True).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    scene_map = _default_scene_mapping()
    attr_map = _default_attr_mapping()
    object_map = _default_object_mapping()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    converted = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in _iter_input_records(input_path):
            image_id = record.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                continue
            if image_id in seen_ids:
                continue

            annotation = record.get("annotation") or {}
            if not isinstance(annotation, dict):
                continue

            scene_labels = _map_scene(annotation.get("scene"), scene_map)
            attr_payload = _map_attributes(annotation, attr_map)
            object_labels = _map_objects(
                annotation.get("objects"),
                object_map,
                min_confidence=float(args.min_object_confidence),
                only_main=not bool(args.include_non_main),
            )

            gt_entry: dict[str, Any] = {
                "image_id": image_id,
                "scene": scene_labels,
                "attributes": attr_payload,
                "objects": object_labels,
            }
            out.write(json.dumps(gt_entry, ensure_ascii=False) + "\n")
            converted += 1
            seen_ids.add(image_id)

    print(f"Converted {converted} Qwen records into ground truth entries at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





