#!/usr/bin/env python3
"""Few-shot learner proof of concept for specialized products."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProductPrototype:
    """Prototype representation of a learned product."""

    name: str
    category: str
    feature_vector: np.ndarray
    sample_count: int
    threshold: float
    created_at: str
    accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize metadata for reporting."""
        return {
            "name": self.name,
            "category": self.category,
            "sample_count": self.sample_count,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "accuracy": self.accuracy,
        }


class FewShotLearner:
    """Few-shot learner that identifies products with limited samples."""

    def __init__(self, feature_dim: int = 512) -> None:
        self.feature_dim = feature_dim
        self.prototypes: Dict[str, ProductPrototype] = {}
        self.min_samples = 3
        self.max_samples = 20
        self.feature_extractor = self._mock_feature_extractor
        print(f"Few-shot learner ready (feature dimension: {feature_dim})")

    def learn_new_product(self, product_name: str, category: str, sample_images: List[str]) -> ProductPrototype:
        """Create a prototype from a handful of labeled samples."""
        n_samples = len(sample_images)
        if n_samples < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples; received {n_samples}.")

        if n_samples > self.max_samples:
            print(f"Sample count exceeds {self.max_samples}; truncating to first {self.max_samples} items.")
            sample_images = sample_images[: self.max_samples]

        print(f"\nLearning product: {product_name}")
        print(f"Category: {category}")
        print(f"Samples: {len(sample_images)}")

        features = []
        for img_path in sample_images:
            feature = self.feature_extractor(img_path)
            features.append(feature)
            print(f"  ✓ Extracted features from {img_path}")

        features_array = np.array(features)
        prototype_vector = np.mean(features_array, axis=0)
        print(f"Prototype vector computed (shape: {prototype_vector.shape})")

        threshold = self._calculate_threshold(features_array, prototype_vector)
        print(f"Acceptance threshold: {threshold:.4f}")

        prototype = ProductPrototype(
            name=product_name,
            category=category,
            feature_vector=prototype_vector,
            sample_count=len(sample_images),
            threshold=threshold,
            created_at=datetime.now().isoformat(),
        )

        accuracy = self._validate_prototype(prototype, features_array)
        prototype.accuracy = accuracy
        print(f"Self-validation accuracy: {accuracy:.1%}")

        self.prototypes[product_name] = prototype
        print(f"✅ Learned product: {product_name}")
        return prototype

    def recognize(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Recognize the most likely learned products in an image."""
        if not self.prototypes:
            return [("No learned products", 0.0)]

        feature = self.feature_extractor(image_path)

        similarities: List[Tuple[str, float]] = []
        for product_name, prototype in self.prototypes.items():
            similarity = self._compute_similarity(feature, prototype.feature_vector)
            if similarity >= prototype.threshold:
                similarities.append((product_name, similarity))

        similarities.sort(key=lambda item: item[1], reverse=True)
        if not similarities:
            return [("No matches", 0.0)]

        return similarities[:top_k]

    def update_product(self, product_name: str, new_samples: List[str]) -> None:
        """Incrementally update an existing prototype with new samples."""
        if product_name not in self.prototypes:
            raise ValueError(f"Product {product_name} is not trained yet.")

        prototype = self.prototypes[product_name]
        print(f"\nUpdating product: {product_name}")
        print(f"Current samples: {prototype.sample_count}")
        print(f"New samples: {len(new_samples)}")

        new_features = [self.feature_extractor(path) for path in new_samples]
        new_features_array = np.array(new_features)

        old_weight = prototype.sample_count
        new_weight = len(new_samples)
        total_weight = old_weight + new_weight

        updated_vector = (
            prototype.feature_vector * old_weight + np.mean(new_features_array, axis=0) * new_weight
        ) / total_weight

        prototype.feature_vector = updated_vector
        prototype.sample_count = total_weight

        simulated_old = np.tile(prototype.feature_vector, (old_weight, 1))
        all_features = np.vstack([simulated_old, new_features_array])
        prototype.threshold = self._calculate_threshold(all_features, updated_vector)

        print(f"✅ Product updated. Total samples: {prototype.sample_count}")

    def remove_product(self, product_name: str) -> None:
        """Remove a learned product prototype."""
        if product_name in self.prototypes:
            del self.prototypes[product_name]
            print(f"Removed product: {product_name}")
        else:
            print(f"Product not found: {product_name}")

    def list_products(self) -> List[Dict[str, object]]:
        """List all learned products."""
        items: List[Dict[str, object]] = []
        for name, prototype in self.prototypes.items():
            items.append(
                {
                    "name": name,
                    "category": prototype.category,
                    "samples": prototype.sample_count,
                    "accuracy": prototype.accuracy,
                    "created": prototype.created_at,
                }
            )
        return items

    def save_model(self, path: str) -> None:
        """Persist prototypes to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as handle:
            pickle.dump(self.prototypes, handle)
        print(f"Model saved to: {save_path}")

    def load_model(self, path: str) -> None:
        """Load previously saved prototypes."""
        with open(path, "rb") as handle:
            self.prototypes = pickle.load(handle)
        print(f"Loaded {len(self.prototypes)} products from disk")

    def _mock_feature_extractor(self, image_path: str) -> np.ndarray:
        """Mock feature extractor (replace with DINOv2/SigLIP in production)."""
        np.random.seed(hash(image_path) % 2**32)
        base_feature = np.random.randn(self.feature_dim)

        lower_name = image_path.lower()
        if "iphone" in lower_name:
            base_feature[:10] += 2.0
        elif "macbook" in lower_name:
            base_feature[10:20] += 2.0
        elif "pizza" in lower_name:
            base_feature[20:30] += 2.0

        return base_feature / np.linalg.norm(base_feature)

    def _compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Compute cosine similarity in [0, 1]."""
        similarity = np.dot(feature1, feature2)
        return (similarity + 1.0) / 2.0

    def _calculate_threshold(self, features: np.ndarray, prototype: np.ndarray) -> float:
        """Derive an acceptance threshold based on feature dispersion."""
        similarities = np.array([self._compute_similarity(feat, prototype) for feat in features])
        threshold = np.mean(similarities) - np.std(similarities)
        return max(0.3, min(0.9, threshold))

    def _validate_prototype(self, prototype: ProductPrototype, features: np.ndarray) -> float:
        """Validate prototype accuracy against training features."""
        correct = 0
        for feature in features:
            similarity = self._compute_similarity(feature, prototype.feature_vector)
            if similarity >= prototype.threshold:
                correct += 1
        return correct / len(features)


def demo() -> None:
    """Demonstrate the few-shot learning pipeline."""
    print("=== Few-shot learner demo ===\n")

    learner = FewShotLearner(feature_dim=128)

    print("\nScenario 1: Learn a professional oscilloscope")
    oscilloscope_samples = [
        "keysight_dso_1.jpg",
        "keysight_dso_2.jpg",
        "keysight_dso_3.jpg",
        "keysight_dso_4.jpg",
        "keysight_dso_5.jpg",
    ]
    learner.learn_new_product("Keysight DSO-X 3024T", "Test equipment", oscilloscope_samples)

    print("\nScenario 2: Learn a rare mechanical keyboard")
    keyboard_samples = [
        "custom_keyboard_1.jpg",
        "custom_keyboard_2.jpg",
        "custom_keyboard_3.jpg",
    ]
    learner.learn_new_product("HHKB Professional Hybrid Type-S", "Keyboard", keyboard_samples)

    print("\nScenario 3: Learn a Neapolitan pizza")
    pizza_samples = [
        "neapolitan_pizza_1.jpg",
        "neapolitan_pizza_2.jpg",
        "neapolitan_pizza_3.jpg",
        "neapolitan_pizza_4.jpg",
    ]
    learner.learn_new_product("Neapolitan Margherita Pizza", "Cuisine", pizza_samples)

    print("\n=== Recognition tests ===")
    test_images = [
        "keysight_dso_test.jpg",
        "random_keyboard.jpg",
        "neapolitan_pizza_new.jpg",
        "unknown_device.jpg",
    ]

    for img in test_images:
        print(f"\nTest image: {img}")
        for product, similarity in learner.recognize(img, top_k=2):
            print(f"  - {product}: {similarity:.1%}")

    print("\n=== Incremental learning ===")
    learner.update_product("Keysight DSO-X 3024T", ["keysight_dso_6.jpg", "keysight_dso_7.jpg"])

    print("\n=== Learned products ===")
    products = learner.list_products()
    for product in products:
        print(f"- {product['name']}")
        print(f"  Category: {product['category']}")
        print(f"  Samples: {product['samples']}")
        if product["accuracy"] is not None:
            print(f"  Accuracy: {product['accuracy']:.1%}")

    learner.save_model("few_shot_model.pkl")

    stats = {
        "total_products": len(learner.prototypes),
        "products": products,
        "feature_dim": learner.feature_dim,
        "min_samples": learner.min_samples,
    }

    with open("few_shot_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("\n✅ Demo complete!")
    print("- Model saved to: few_shot_model.pkl")
    print("- Stats saved to: few_shot_stats.json")


if __name__ == "__main__":
    demo()
