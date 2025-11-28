#!/usr/bin/env python3
"""Sample a subset of ground-truth records for manual review/correction.

This helper is designed for Step 4 of the Qwen3-VL evaluation workflow:

- Given an auto-generated ground_truth JSON/JSONL file (typically produced by
  tools/qwen_to_ground_truth.py),
- Randomly sample N entries,
- Write them to a new JSON file as a list of objects for easier manual editing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat as JSONL.
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a subset of ground-truth records for manual review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to auto-generated ground truth (JSON or JSONL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for sampled ground truth JSON (list of objects).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=800,
        help="Number of samples to draw (default: 800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    records = _load_records(input_path)
    if not records:
        raise SystemExit(f"No valid records found in {input_path}")

    sample_size = max(1, min(int(args.size), len(records)))
    random.seed(int(args.seed))
    sampled = random.sample(records, sample_size) if sample_size < len(records) else records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Sampled {len(sampled)} records out of {len(records)} from {input_path} into {output_path} "
        f"(seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
