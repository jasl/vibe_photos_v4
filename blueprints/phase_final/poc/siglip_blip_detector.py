#!/usr/bin/env python3
"""Proof-of-concept detector using SigLIP for classification and BLIP for captions."""

from __future__ import annotations

import json
from pathlib import Path

from vibe_photos.ml.siglip_blip import SiglipBlipDetectionResult, SiglipBlipDetector


def demo() -> None:
    """Demonstrate SigLIP + BLIP detection capabilities."""
    print("=== SigLIP + BLIP detector demo ===\n")

    detector = SiglipBlipDetector()

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
        default_labels = [
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
            "tapas",
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

        results: list[SiglipBlipDetectionResult] = []
        for image_path in test_images:
            try:
                result = detector.detect(image_path=image_path, candidate_labels=default_labels, confidence_threshold=0.3)
                results.append(result)
                print(f"Processed: {image_path.name}")
            except Exception as error:  # noqa: BLE001
                print(f"Failed on {image_path}: {error}")

        for result in results:
            print(f"Image: {result.image_path}")
            print(f"Detected objects: {', '.join(result.detected_labels)}")
            print(f"Caption: {result.caption or ''}")
            print(f"Overall confidence: {result.confidence:.2%}")

            top_5 = sorted(result.label_scores.items(), key=lambda item: item[1], reverse=True)[:5]
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
