#!/usr/bin/env python3
"""Minimal SigLIP-based detector for prototype experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


@dataclass
class DetectionResult:
    """Container for detection outcomes."""

    category: str
    confidence: float
    alternatives: List[Dict[str, float]]
    metadata: Dict[str, object]


class SimpleDetector:
    """Lightweight multilingual zero-shot classifier built on SigLIP."""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224") -> None:
        print(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.categories: Dict[str, List[str]] = {
            "general": [
                "Electronics",
                "Food",
                "Documents",
                "Landscape",
                "People",
                "Animals",
                "Architecture",
                "Other",
            ],
            "electronics": [
                "Smartphone",
                "Laptop",
                "Tablet",
                "Camera",
                "Headphones",
                "Watch",
                "Monitor",
                "Keyboard",
            ],
            "food": [
                "Pizza",
                "Burger",
                "Noodles",
                "Rice",
                "Dessert",
                "Fruit",
                "Beverage",
                "Vegetables",
            ],
            "specific_products": [
                "iPhone",
                "MacBook",
                "iPad",
                "AirPods",
                "Samsung Galaxy",
                "ThinkPad",
                "Surface",
            ],
        }

    def detect(self, image_path: Path, category_set: str = "general", threshold: float = 0.3) -> DetectionResult:
        """Run zero-shot classification over a predefined label set."""
        image = Image.open(image_path).convert("RGB")
        categories = self.categories.get(category_set, self.categories["general"])

        inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        confidences = probs[0].tolist()
        results = sorted(zip(categories, confidences), key=lambda item: item[1], reverse=True)

        main_category, main_confidence = results[0]
        alternatives = [
            {"category": cat, "confidence": conf}
            for cat, conf in results[1:4]
        ]

        metadata = {
            "image_size": image.size,
            "threshold_met": main_confidence >= threshold,
            "needs_review": main_confidence < 0.5,
            "model": "siglip2-base-patch16-224",
        }

        return DetectionResult(
            category=main_category,
            confidence=main_confidence,
            alternatives=alternatives,
            metadata=metadata,
        )

    def detect_hierarchical(self, image_path: Path) -> Dict[str, Dict[str, float]]:
        """Perform hierarchical classification from coarse to fine labels."""
        general_result = self.detect(image_path, "general")

        results: Dict[str, Dict[str, float]] = {
            "level1": {
                "category": general_result.category,
                "confidence": general_result.confidence,
            }
        }

        if general_result.category == "Electronics" and general_result.confidence > 0.5:
            electronic_result = self.detect(image_path, "electronics")
            results["level2"] = {
                "category": electronic_result.category,
                "confidence": electronic_result.confidence,
            }

            if electronic_result.category in {"Smartphone", "Laptop", "Tablet"}:
                product_result = self.detect(image_path, "specific_products")
                results["level3"] = {
                    "category": product_result.category,
                    "confidence": product_result.confidence,
                }

        elif general_result.category == "Food" and general_result.confidence > 0.5:
            food_result = self.detect(image_path, "food")
            results["level2"] = {
                "category": food_result.category,
                "confidence": food_result.confidence,
            }

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python simple_detector.py /path/to/image.jpg")

    detector = SimpleDetector()
    output = detector.detect_hierarchical(Path(sys.argv[1]))
    print(output)
