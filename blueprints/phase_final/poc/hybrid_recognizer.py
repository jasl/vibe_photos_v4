#!/usr/bin/env python3
"""Demonstration of balancing automation with human review."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ConfidenceLevel(Enum):
    """Confidence buckets used to route actions."""

    HIGH = "high"  # > 0.8 — auto-accept
    MEDIUM = "medium"  # 0.5-0.8 — request confirmation
    LOW = "low"  # < 0.5 — manual annotation required


@dataclass
class RecognitionResult:
    """Aggregate state for a recognized asset."""

    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    ai_category: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_suggestions: List[Dict[str, object]] = field(default_factory=list)

    human_label: Optional[str] = None
    human_verified: bool = False

    confidence_level: Optional[ConfidenceLevel] = None
    action_taken: Optional[str] = None
    needs_review: bool = True

    used_for_training: bool = False
    similar_images: List[str] = field(default_factory=list)


class HybridRecognizer:
    """Hybrid recognizer combining AI ranking with human validation."""

    def __init__(self) -> None:
        self.thresholds = {
            "auto_accept": 0.8,
            "suggest": 0.5,
            "reject": 0.3,
        }

        self.learned_patterns: Dict[str, int] = {}
        self.annotation_history: List[Dict[str, object]] = []
        self.user_preferences = {
            "common_labels": ["iPhone", "MacBook", "Pizza", "Screenshot"],
            "recent_labels": [],
            "label_shortcuts": {
                "1": "iPhone",
                "2": "MacBook",
                "3": "Pizza",
                "4": "Document",
            },
        }

    def recognize(self, image_path: str, ai_prediction: Tuple[str, float]) -> RecognitionResult:
        """Run the hybrid recognition flow for a single image."""
        category, confidence = ai_prediction

        result = RecognitionResult(image_path=image_path, ai_category=category, ai_confidence=confidence)
        result.confidence_level = self._get_confidence_level(confidence)

        if result.confidence_level == ConfidenceLevel.HIGH:
            result.action_taken = "auto_accepted"
            result.human_label = category
            result.needs_review = False
            result.human_verified = True
            print(f"✅ Auto-accepted: {category} ({confidence:.1%})")

        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            result.action_taken = "suggested"
            result.ai_suggestions = self._generate_suggestions(image_path, category)
            result.needs_review = True
            print(f"💡 AI suggestion: {category} ({confidence:.1%})")
            print(f"   Alternatives: {result.ai_suggestions}")

        else:
            result.action_taken = "manual_required"
            result.needs_review = True
            print(f"❓ Manual review needed (AI guess: {category} — {confidence:.1%})")

        result.similar_images = self._find_similar_images(image_path)
        if result.similar_images:
            print(f"📷 Found {len(result.similar_images)} similar images")

        return result

    def apply_human_annotation(self, result: RecognitionResult, human_label: str) -> RecognitionResult:
        """Apply a human decision to the recognition result."""
        result.human_label = human_label
        result.human_verified = True
        result.needs_review = False

        self.annotation_history.append(
            {
                "image": result.image_path,
                "ai_prediction": result.ai_category,
                "ai_confidence": result.ai_confidence,
                "human_label": human_label,
                "timestamp": result.timestamp,
            }
        )

        if human_label not in self.user_preferences["recent_labels"]:
            self.user_preferences["recent_labels"].insert(0, human_label)
            self.user_preferences["recent_labels"] = self.user_preferences["recent_labels"][:10]

        if human_label != result.ai_category:
            result.used_for_training = True
            self._learn_from_correction(result)

        print(f"✏️ Human label applied: {human_label}")
        return result

    def batch_apply(self, primary_result: RecognitionResult, similar_images: List[str]) -> List[RecognitionResult]:
        """Propagate a confirmed label to similar assets."""
        if not primary_result.human_verified:
            print("⚠️ Primary image is not verified; skip batch propagation.")
            return []

        results: List[RecognitionResult] = []
        for image in similar_images:
            batch_result = RecognitionResult(
                image_path=image,
                ai_category=primary_result.ai_category,
                ai_confidence=0.95,
                human_label=primary_result.human_label,
                human_verified=True,
                needs_review=False,
                action_taken="batch_applied",
            )
            results.append(batch_result)

        print(f"🎯 Applied label '{primary_result.human_label}' to {len(results)} similar images")
        return results

    def generate_annotation_ui(self, result: RecognitionResult) -> Dict[str, object]:
        """Produce UI hints for a hypothetical annotation console."""
        ui_data: Dict[str, object] = {
            "image": result.image_path,
            "ai_prediction": {
                "category": result.ai_category,
                "confidence": result.ai_confidence,
                "level": result.confidence_level.value if result.confidence_level else None,
            },
            "suggestions": [],
            "shortcuts": self.user_preferences["label_shortcuts"],
            "recent_labels": self.user_preferences["recent_labels"],
            "similar_count": len(result.similar_images),
            "actions": [],
        }

        if result.confidence_level == ConfidenceLevel.HIGH:
            ui_data["actions"] = [
                {"key": "Space", "action": "Confirm", "primary": True},
                {"key": "X", "action": "Skip"},
                {"key": "E", "action": "Edit"},
            ]
            ui_data["message"] = f"AI is confident this is: {result.ai_category}"

        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            ui_data["suggestions"] = [
                result.ai_category,
                *[suggestion["label"] for suggestion in result.ai_suggestions[:3]],
            ]
            ui_data["actions"] = [
                {"key": "1-4", "action": "Choose suggestion"},
                {"key": "T", "action": "Enter custom label"},
                {"key": "X", "action": "Skip"},
            ]
            ui_data["message"] = "AI is unsure. Please confirm or adjust."

        else:
            ui_data["actions"] = [
                {"key": "1-9", "action": "Use quick label"},
                {"key": "T", "action": "Enter label"},
                {"key": "X", "action": "Mark unknown"},
            ]
            ui_data["message"] = "AI cannot classify this image. Manual input required."

        if result.similar_images:
            ui_data["batch_option"] = {
                "available": True,
                "count": len(result.similar_images),
                "key": "G",
                "action": "Apply to similar images",
            }

        return ui_data

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map a confidence score to a routing bucket."""
        if confidence >= self.thresholds["auto_accept"]:
            return ConfidenceLevel.HIGH
        if confidence >= self.thresholds["suggest"]:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _generate_suggestions(self, image_path: str, primary_category: str) -> List[Dict[str, object]]:
        """Craft alternate suggestions for human review."""
        suggestions: List[Dict[str, object]] = []

        if self.user_preferences["recent_labels"]:
            suggestions.append(
                {
                    "label": self.user_preferences["recent_labels"][0],
                    "reason": "recently_used",
                    "score": 0.7,
                }
            )

        similar_categories = {
            "Smartphone": ["iPhone", "Samsung", "Android phone"],
            "Laptop": ["MacBook", "ThinkPad", "Notebook"],
            "Food": ["Pizza", "Burger", "Noodles"],
        }

        if primary_category in similar_categories:
            for cat in similar_categories[primary_category][:2]:
                suggestions.append({"label": cat, "reason": "similar_category", "score": 0.6})

        return suggestions

    def _find_similar_images(self, image_path: str) -> List[str]:
        """Mock similar-image lookup using randomness."""
        import random

        if random.random() > 0.5:
            count = random.randint(1, 10)
            return [f"similar_{index}.jpg" for index in range(count)]
        return []

    def _learn_from_correction(self, result: RecognitionResult) -> None:
        """Update in-memory pattern store based on corrections."""
        key = f"{result.ai_category}->{result.human_label}"
        self.learned_patterns.setdefault(key, 0)
        self.learned_patterns[key] += 1

        print(f"🧠 Learning pattern: {key} (seen {self.learned_patterns[key]} times)")

        if self.learned_patterns[key] >= 5:
            print("💡 Frequent correction detected. Consider retraining the model.")

    def get_statistics(self) -> Dict[str, object]:
        """Summarize annotation metrics."""
        total = len(self.annotation_history)
        if total == 0:
            return {"message": "No annotations yet."}

        correct = sum(1 for item in self.annotation_history if item["ai_prediction"] == item["human_label"])
        accuracy = correct / total if total else 0

        return {
            "total_annotations": total,
            "ai_correct": correct,
            "ai_accuracy": accuracy,
            "learned_patterns": len(self.learned_patterns),
            "common_corrections": sorted(
                self.learned_patterns.items(), key=lambda pair: pair[1], reverse=True
            )[:5],
        }


def demo() -> None:
    """Demonstrate the hybrid workflow end-to-end."""
    print("=== Hybrid recognizer demo ===\n")

    recognizer = HybridRecognizer()

    test_cases = [
        ("photo1.jpg", ("iPhone", 0.92)),
        ("photo2.jpg", ("Smartphone", 0.65)),
        ("photo3.jpg", ("Unknown object", 0.25)),
        ("photo4.jpg", ("MacBook", 0.88)),
        ("photo5.jpg", ("Laptop", 0.55)),
    ]

    results: List[RecognitionResult] = []

    for image_path, ai_prediction in test_cases:
        print(f"\n--- Processing: {image_path} ---")
        result = recognizer.recognize(image_path, ai_prediction)

        if result.needs_review:
            ui_data = recognizer.generate_annotation_ui(result)
            print(f"UI prompt: {ui_data['message']}")

            if result.confidence_level == ConfidenceLevel.MEDIUM:
                human_label = result.ai_category
            else:
                human_label = "Professional equipment"

            result = recognizer.apply_human_annotation(result, human_label)

            if result.similar_images:
                batch_results = recognizer.batch_apply(result, result.similar_images)
                results.extend(batch_results)

        results.append(result)

    print("\n=== Statistics ===")
    stats = recognizer.get_statistics()
    print(json.dumps(stats, indent=2))

    output = {
        "results": [
            {
                "image": item.image_path,
                "ai_category": item.ai_category,
                "ai_confidence": item.ai_confidence,
                "human_label": item.human_label,
                "action": item.action_taken,
                "verified": item.human_verified,
            }
            for item in results
        ],
        "statistics": stats,
    }

    with open("hybrid_recognition_results.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("\n✅ Saved results to hybrid_recognition_results.json")


if __name__ == "__main__":
    demo()
