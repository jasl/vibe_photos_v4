#!/usr/bin/env python3
"""CLI tool to smoke-test coarse categories using SigLIP image embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from vibe_photos.config import load_settings
from vibe_photos.ml import DEFAULT_COARSE_CATEGORIES, build_siglip_coarse_classifier


RAW_SIGLIP_LABELS: List[str] = [
    "phone",
    "smartphone",
    "iPhone",
    "Android phone",
    "computer",
    "laptop",
    "MacBook",
    "tablet",
    "iPad",
    "headphones",
    "AirPods",
    "camera",
    "food",
    "pizza",
    "burger",
    "sushi",
    "noodles",
    "dessert",
    "cake",
    "document",
    "book",
    "notes",
    "person",
    "people",
    "landscape",
    "architecture",
    "building",
    "animal",
    "pet",
]


def iter_image_paths(root: Path) -> Iterable[Path]:
    """Yield image paths under the given root directory."""

    for extension in (".jpg", ".jpeg", ".png"):
        yield from root.rglob(f"*{extension}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Smoke-test coarse categories with SigLIP.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("samples"),
        help="Directory containing test images (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "SigLIP model name or path. When omitted, the value from "
            "config/settings.yaml (models.embedding.model_name) is used, "
            "falling back to 'google/siglip2-base-patch16-224' if not configured."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help=(
            "Minimum margin between top-1 and top-2 coarse category probabilities required to select a non-'other' "
            "category (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=0.15,
        help=(
            "Minimum top-1 coarse category probability required to select a non-'other' category "
            "(default: %(default)s)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def classify_images(
    image_paths: Sequence[Path],
    model_name: str,
    threshold: float,
    score_min: float,
) -> None:
    """Run coarse category classification on a sequence of images and print results."""

    if not image_paths:
        print("No images found for testing.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] Using device: {device}")
    print(f"[setup] Loading SigLIP model and processor: {model_name} (this may take a while)...")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print("[setup] Models loaded, preparing coarse category classifier...")

    classifier = build_siglip_coarse_classifier(
        siglip_model=model,
        siglip_processor=processor,
        categories=DEFAULT_COARSE_CATEGORIES,
        threshold=threshold,
        score_min=score_min,
        device=device,
    )

    raw_label_embeddings: torch.Tensor | None = None
    if RAW_SIGLIP_LABELS:
        print("[setup] Encoding raw SigLIP label prompts for reference...")
        text_inputs = processor(text=RAW_SIGLIP_LABELS, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_label_embeddings = model.get_text_features(**text_inputs)
        raw_label_embeddings = raw_label_embeddings / raw_label_embeddings.norm(dim=-1, keepdim=True)

    print(f"[run] Found {len(image_paths)} images. Starting classification...\n")

    for image_path in sorted(image_paths):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_embedding = model.get_image_features(**inputs)

        primary_category, scores = classifier.classify_from_image_embedding(image_embedding)
        top_scores: List[Tuple[str, float]] = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:3]

        print(f"{image_path}:")
        print(f"  primary: {primary_category}")
        for label, score in top_scores:
            print(f"  - {label}: {score:.3f}")

        if raw_label_embeddings is not None:
            # Compute raw SigLIP similarities against a fixed label set.
            emb = image_embedding[0] if image_embedding.ndim == 2 else image_embedding
            emb = emb / emb.norm()
            sims = emb @ raw_label_embeddings.T  # shape: (num_labels,)
            raw_scores: Dict[str, float] = {
                label: float(score) for label, score in zip(RAW_SIGLIP_LABELS, sims.tolist())
            }
            top_raw = sorted(raw_scores.items(), key=lambda pair: pair[1], reverse=True)[:5]
            print("  raw SigLIP top-5:")
            for label, score in top_raw:
                print(f"    - {label}: {score:.3f}")

        print()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for manual coarse category testing."""

    args = parse_args(argv)
    images_dir: Path = args.images_dir

    if args.model_name is not None:
        model_name = args.model_name
    else:
        settings = load_settings()
        model_name = settings.models.embedding.resolved_model_name()

    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Images directory not found or not a directory: {images_dir}")
        return 1

    image_paths = list(iter_image_paths(images_dir))
    classify_images(
        image_paths=image_paths,
        model_name=model_name,
        threshold=args.threshold,
        score_min=args.score_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
