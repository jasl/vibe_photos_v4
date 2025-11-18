#!/usr/bin/env python3
"""Proof-of-concept detector using SigLIP for classification and BLIP for captions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)


@dataclass
class DetectionResult:
    """Structured response for SigLIP + BLIP analysis."""

    image_path: str
    siglip_scores: Dict[str, float]
    blip_caption: str
    detected_objects: List[str]
    confidence: float
    metadata: Dict[str, Any]


class SigLIPBLIPDetector:
    """Multilingual zero-shot detector combining SigLIP classification and BLIP captions."""

    def __init__(
        self,
        device: str = "cpu",
        siglip_model: str = "google/siglip2-base-patch16-224",
        blip_model: str = "Salesforce/blip-image-captioning-base",
    ) -> None:
        self.device = device

        print(f"Loading SigLIP model: {siglip_model}")
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model)
        self.siglip_model = AutoModel.from_pretrained(siglip_model).to(device)

        print(f"Loading BLIP model: {blip_model}")
        self.blip_processor = BlipProcessor.from_pretrained(blip_model)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model).to(device)

        print(f"Models ready on device: {self.device}")

    def detect(
        self,
        image_path: Path,
        candidate_labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        generate_caption: bool = True,
    ) -> DetectionResult:
        """Run zero-shot classification and optional captioning on a single image."""
        image = Image.open(image_path).convert("RGB")

        if candidate_labels is None:
            candidate_labels = [
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

        siglip_results = self._classify_with_siglip(image, candidate_labels)
        caption = self._generate_caption_with_blip(image) if generate_caption else ""

        detected_objects = [label for label, score in siglip_results.items() if score >= confidence_threshold]

        top_scores = sorted(siglip_results.values(), reverse=True)[:3]
        overall_confidence = float(np.mean(top_scores)) if top_scores else 0.0

        return DetectionResult(
            image_path=str(image_path),
            siglip_scores=siglip_results,
            blip_caption=caption,
            detected_objects=detected_objects,
            confidence=overall_confidence,
            metadata={
                "model": "SigLIP+BLIP",
                "siglip_model": self.siglip_processor.name_or_path,
                "blip_model": self.blip_processor.name_or_path,
                "device": self.device,
            },
        )

    def _classify_with_siglip(self, image: Image.Image, labels: List[str]) -> Dict[str, float]:
        """Compute zero-shot classification scores via SigLIP."""
        inputs = self.siglip_processor(text=labels, images=image, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.siglip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)

        return {label: float(prob) for label, prob in zip(labels, probs[0].cpu().numpy())}

    def _generate_caption_with_blip(self, image: Image.Image) -> str:
        """Generate a natural language caption using BLIP."""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def batch_detect(self, image_paths: List[Path], batch_size: int = 4, **kwargs: Any) -> List[DetectionResult]:
        """Run detection over a list of images."""
        results: List[DetectionResult] = []

        for index in range(0, len(image_paths), batch_size):
            batch = image_paths[index : index + batch_size]
            for path in batch:
                try:
                    result = self.detect(path, **kwargs)
                    results.append(result)
                    print(f"Processed: {path.name}")
                except Exception as error:  # noqa: BLE001
                    print(f"Failed on {path}: {error}")

        return results

    def analyze_for_specific_product(
        self, image_path: Path, product_keywords: List[str]
    ) -> Tuple[bool, float, str]:
        """Check whether an image likely contains a specific product."""
        result = self.detect(image_path, candidate_labels=product_keywords)

        max_score = max(result.siglip_scores.values()) if result.siglip_scores else 0.0
        contains_product = max_score > 0.5

        if contains_product:
            label, score = max(result.siglip_scores.items(), key=lambda item: item[1])
            description = f"Detected: {label} (confidence: {score:.2%})"
            if result.blip_caption:
                description += f"\nCaption: {result.blip_caption}"
        else:
            description = "No matching product detected"

        return contains_product, max_score, description


def demo() -> None:
    """Demonstrate SigLIP + BLIP detection capabilities."""
    print("=== SigLIP + BLIP detector demo ===\n")

    detector = SigLIPBLIPDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")

    test_images = [
        Path("./sample_data/iphone.jpg"),
        Path("./sample_data/pizza.jpg"),
        Path("./sample_data/document.pdf"),
    ]

    if not any(path.exists() for path in test_images):
        print("\nSample images not found. Displaying mocked output...\n")
        mock_results = [
            {
                "image": "iphone.jpg",
                "top_detections": {"iPhone": 0.92, "phone": 0.88, "smartphone": 0.87},
                "caption": "a close up of a cell phone on a table",
                "confidence": 0.90,
            },
            {
                "image": "pizza.jpg",
                "top_detections": {"pizza": 0.95, "food": 0.90, "dish": 0.82},
                "caption": "a pizza with cheese and vegetables on it",
                "confidence": 0.92,
            },
            {
                "image": "document.pdf",
                "top_detections": {"document": 0.85, "paper": 0.83, "notes": 0.80},
                "caption": "a piece of paper with text on it",
                "confidence": 0.83,
            },
        ]

        for result in mock_results:
            print(f"Image: {result['image']}")
            print("Primary detections:")
            for label, score in result["top_detections"].items():
                print(f"  - {label}: {score:.2%}")
            print(f"Caption: {result['caption']}")
            print(f"Overall confidence: {result['confidence']:.2%}")
            print("-" * 50)

    else:
        print("\nRunning detection...\n")
        results = detector.batch_detect(test_images)

        for result in results:
            print(f"Image: {result.image_path}")
            print(f"Detected objects: {', '.join(result.detected_objects)}")
            print(f"Caption: {result.blip_caption}")
            print(f"Overall confidence: {result.confidence:.2%}")

            top_5 = sorted(result.siglip_scores.items(), key=lambda item: item[1], reverse=True)[:5]
            print("Top-5 classes:")
            for label, score in top_5:
                print(f"  - {label}: {score:.2%}")
            print("-" * 50)

    print("\n=== Targeted product detection ===")
    print("Query keywords: ['iPhone 15', 'iPhone 14', 'iPhone', 'Apple phone']")
    print("Mock result: Detected iPhone (confidence 92%)")
    print("Caption: a close up of an iPhone on a wooden table")

    comparison = {
        "SigLIP+BLIP advantages": [
            "No heavy dependencies (no mmcv stack)",
            "Multilingual zero-shot support",
            "Natural-language captioning",
            "Semantic understanding beyond fixed classes",
        ],
        "RTMDet drawbacks": [
            "Requires legacy mmcv dependency",
            "Limited to predefined COCO classes",
            "No caption generation",
            "No multilingual capabilities",
        ],
    }
    print("\n=== SigLIP + BLIP vs RTMDet ===")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    demo()
