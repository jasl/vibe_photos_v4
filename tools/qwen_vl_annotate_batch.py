#!/usr/bin/env python3
"""Batch annotate images with Qwen3-VL via an OpenAI-compatible endpoint.

This script is intended to run on the external H100 server where Qwen3-VL is
hosted. It does **not** touch the local primary database. Instead it:

- Computes the same content hash used by the M2 pipeline (images.image_id).
- Calls a Qwen3-VL endpoint to obtain a coarse JSON annotation per image.
- Writes one JSON object per line to a JSONL file that can later be converted
  into M2's ground_truth.json format on the local machine.

Each JSONL line looks like:

{
  "image_id": "8f14e45fceea167a5a36dedd4bea2543",
  "server_path": "/mnt/photos/2024/IMG_1234.JPG",
  "annotation": {
    "scene": "product",
    "has_person": false,
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

You can resume long runs with --resume; the script will skip records whose
image_id is already present in the output file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageOps

from vibe_photos.hasher import compute_content_hash


@dataclass(frozen=True)
class QwenConfig:
    """Configuration for calling a Qwen3-VL OpenAI-compatible endpoint."""

    base_url: str = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.environ.get("QWEN_API_KEY", "qwen-placeholder-key")
    model: str = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")

    request_timeout: float = float(os.environ.get("QWEN_REQUEST_TIMEOUT", "120"))
    max_retries: int = int(os.environ.get("QWEN_MAX_RETRIES", "3"))
    retry_backoff: float = float(os.environ.get("QWEN_RETRY_BACKOFF", "5.0"))


@dataclass(frozen=True)
class ImagePayloadOptions:
    """Controls how images are resized/compressed before hitting Qwen."""

    max_edge_px: int | None = 2048
    target_format: str | None = "PNG"
    jpeg_quality: int = 88
    strip_metadata: bool = True

    def requires_processing(self) -> bool:
        return any((self.max_edge_px is not None, self.target_format is not None, self.strip_metadata))


SYSTEM_PROMPT = r"""
你是一个给个人相册做自动标注的助手，只负责“粗分类”和“主要物体列表”。

要求：
- 你不知道的就回答 "unknown"，禁止胡乱猜品牌和型号。
- 你只看图像本身，不要幻想图外信息。
- 输出必须是**合法的 JSON 对象**，不要包含任何多余文字。

scene 字段只能从下面的枚举中选择一个：
["landscape", "snapshot", "people", "selfie", "food", "product", "document", "screenshot", "other"]

coarse_type 字段只能从下面的枚举中选择一个：
[
  "electronics", "circuit_board“, "disk", "charger",
  "phone", "laptop", "tablet", "mini_pc", "computer_case"
  "monitor", "keyboard", "mouse", "earphones", "headphones", "hub",
  "camera", "game_console", "controller",
  "food", "drink", "dessert",
  "paper_document", "screen", "packaging", "book",
  "card", "id_card",
  "other_object", "unknown"
]

输出 JSON 结构必须是下面这样（不要省略任何字段，可以把 objects 设为空列表）：
{
  "scene": "<one of scene enums>",
  "has_person": true or false,
  "has_text": true or false,
  "is_document": true or false,
  "is_screenshot": true or false,
  "objects": [
    {
      "name_free": "<自由描述，可以是中文或英文，例如 'mechanical keyboard'>",
      "coarse_type": "<one of coarse_type enums>",
      "is_main": true or false,
      "confidence": 0.0 到 1.0 之间的数字
    },
    ...
  ]
}

注意：
- 如果图中没有明显物体，可以让 objects 为空列表。
- 如果不确定某个字段，请尽量使用 false 或 "unknown"，而不是瞎猜。
"""


def iter_images(root: Path, exts: Sequence[str] | None = None) -> Iterable[Path]:
    """Yield image files under a root directory."""

    if exts is None:
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts:
            yield path


def _guess_mime_from_suffix(suffix: str) -> str:
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in (".bmp", ".dib"):
        return "image/bmp"
    if suffix in (".tiff", ".tif"):
        return "image/tiff"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def _prepare_image_payload_bytes(path: Path, opts: ImagePayloadOptions) -> tuple[bytes, str]:
    """Resize/compress an image and return (bytes, mime-type)."""

    with Image.open(path) as img:
        image = ImageOps.exif_transpose(img)
        target_format = (opts.target_format or (image.format or "JPEG")).upper()
        working = image.copy()

        if opts.max_edge_px is not None and max(working.size) > opts.max_edge_px:
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            working.thumbnail((opts.max_edge_px, opts.max_edge_px), resample=resample)

        if target_format in {"JPEG", "WEBP"} and working.mode not in ("RGB", "L"):
            working = working.convert("RGB")

        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        if target_format == "JPEG":
            save_kwargs["quality"] = opts.jpeg_quality
            save_kwargs["optimize"] = True
            save_kwargs["progressive"] = True
        if opts.strip_metadata:
            working.info.pop("exif", None)
        working.save(buffer, format=target_format, **save_kwargs)
        mime = Image.MIME.get(target_format, f"image/{target_format.lower()}")
    return buffer.getvalue(), mime


def encode_image_to_data_url(path: Path, payload_opts: ImagePayloadOptions | None = None) -> str:
    """Encode an image file as a data URL suitable for OpenAI image_url content."""

    if payload_opts is None or not payload_opts.requires_processing():
        data = path.read_bytes()
        mime = _guess_mime_from_suffix(path.suffix.lower())
    else:
        data, mime = _prepare_image_payload_bytes(path, payload_opts)

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_processed_ids(output: Path) -> set[str]:
    """Load already processed image_ids from an existing JSONL file."""

    processed: set[str] = set()
    if not output.exists():
        return processed

    with output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            image_id = rec.get("image_id")
            if isinstance(image_id, str):
                processed.add(image_id)
    return processed


class QwenVLAnnotator:
    """Thin wrapper around an OpenAI-compatible Qwen3-VL endpoint."""

    def __init__(self, cfg: QwenConfig, image_opts: ImagePayloadOptions | None = None) -> None:
        self._cfg = cfg
        self._image_opts = image_opts or ImagePayloadOptions()
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def _build_messages(self, image_data_url: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请按照要求分析这张图片，并只输出 JSON：",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ]

    def annotate_image(self, image_path: Path) -> dict[str, Any]:
        """Call Qwen3-VL to obtain a JSON annotation for a single image."""

        data_url = encode_image_to_data_url(image_path, self._image_opts)
        messages = self._build_messages(data_url)

        last_err: BaseException | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    timeout=self._cfg.request_timeout,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # pragma: no cover - defensive
                last_err = exc
                sys.stderr.write(
                    f"[WARN] Qwen annotation failed (attempt {attempt}/{self._cfg.max_retries}) "
                    f"for {image_path}: {exc}\n"
                )
                if attempt < self._cfg.max_retries:
                    time.sleep(self._cfg.retry_backoff)
                else:
                    break

        raise RuntimeError(f"Qwen annotation failed after {self._cfg.max_retries} attempts: {last_err}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-annotate images with Qwen3-VL (OpenAI-compatible) and emit JSONL."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing images to annotate (scanned recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to JSONL output file (one record per line).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of new images to annotate (for dry runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip images whose image_id already exists in the output JSONL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for Qwen calls (default: 4).",
    )
    parser.add_argument(
        "--max-edge-px",
        type=int,
        default=2048,
        help="Resize images so the longest edge is at most this many pixels before encoding (0 disables).",
    )
    parser.add_argument(
        "--payload-format",
        choices=("jpeg", "png", "webp", "keep"),
        default="png",
        help="Image format used for payloads (use 'keep' to preserve original format, default: png).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=88,
        help="JPEG quality (1-95) when --payload-format=jpeg (default: 88).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Preserve EXIF metadata in payloads (default strips metadata).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = QwenConfig()
    max_edge = args.max_edge_px if args.max_edge_px > 0 else None
    target_format = None if args.payload_format == "keep" else args.payload_format.upper()
    jpeg_quality = max(1, min(95, args.jpeg_quality))
    image_opts = ImagePayloadOptions(
        max_edge_px=max_edge,
        target_format=target_format,
        jpeg_quality=jpeg_quality,
        strip_metadata=not args.keep_metadata,
    )
    annotator = QwenVLAnnotator(cfg, image_opts=image_opts)

    root = args.root
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        sys.stderr.write(f"[ERROR] Image root does not exist: {root}\n")
        return 1

    processed_ids: set[str] = set()
    if args.resume:
        processed_ids = load_processed_ids(output)
        sys.stderr.write(f"[INFO] Resume enabled; {len(processed_ids)} image_ids already present in {output}\n")

    total_seen = 0
    tasks: list[tuple[str, Path]] = []

    for image_path in iter_images(root):
        total_seen += 1
        image_id = compute_content_hash(image_path)
        if image_id in processed_ids:
            continue
        tasks.append((image_id, image_path))
        if args.limit is not None and len(tasks) >= args.limit:
            break

    if not tasks:
        sys.stderr.write("[INFO] No new images to annotate (all already present in output).\n")
        return 0

    workers = max(1, int(args.workers))
    sys.stderr.write(f"[INFO] Starting Qwen annotation with {workers} worker threads over {len(tasks)} images.\n")

    total_new = 0

    def _worker(image_id: str, image_path: Path) -> tuple[str, Path, dict[str, Any]]:
        annotation = annotator.annotate_image(image_path)
        return image_id, image_path, annotation

    with output.open("a", encoding="utf-8") as f_out, ThreadPoolExecutor(max_workers=workers) as executor:
        total_tasks = len(tasks)
        completed = 0

        for batch_start in range(0, total_tasks, workers):
            batch = tasks[batch_start : batch_start + workers]
            future_to_task = {
                executor.submit(_worker, image_id, image_path): (image_id, image_path) for image_id, image_path in batch
            }

            for future in as_completed(future_to_task):
                image_id, image_path = future_to_task[future]
                try:
                    _, resolved_path, annotation = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    sys.stderr.write(f"[WARN] Annotation failed for {image_path}: {exc}\n")
                    continue

                record = {
                    "image_id": image_id,
                    # "server_path": str(resolved_path.resolve()),
                    # "image_path": str(resolved_path.resolve()),
                    "rel_path": str(resolved_path.relative_to(root)),
                    "annotation": annotation,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                processed_ids.add(image_id)
                total_new += 1
                completed += 1

                if total_new and total_new % 50 == 0:
                    sys.stderr.write(
                        f"[INFO] New annotations: {total_new} images "
                        f"(total visited {total_seen}, completed {completed}/{total_tasks})\n"
                    )

    sys.stderr.write(
        f"[DONE] Visited {total_seen} images, submitted {len(tasks)} for annotation, "
        f"wrote {total_new} new records to {output}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


