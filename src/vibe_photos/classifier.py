"""Lightweight scene classification and attribute prediction for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from utils.logging import get_logger
from vibe_photos.config import Settings
from vibe_photos.ml.coarse_categories import CoarseCategoryClassifier, build_siglip_coarse_classifier
from vibe_photos.ml.models import get_siglip_embedding_model


LOGGER = get_logger(__name__)


@dataclass
class SceneAttributes:
    """Final per-image scene attributes produced by the classifier."""

    scene_type: str
    scene_confidence: float
    scene_margin: float

    has_text: bool
    has_text_margin: float

    has_person: bool
    has_person_margin: float

    is_screenshot: bool
    is_screenshot_margin: float

    is_document: bool
    is_document_margin: float

    classifier_name: str
    classifier_version: str


@dataclass(frozen=True)
class AttributePromptConfig:
    """Configuration for a boolean attribute predicted using text prompts."""

    id: str
    positive_prompt: str
    negative_prompt: str
    threshold: float


@dataclass
class SceneClassifierWithAttributes:
    """Wrapper tying together coarse scene classification and attributes.

    The classifier expects SigLIP image embeddings as input. It does not own the
    SigLIP model itself; the embedding model is managed by callers (for example,
    the preprocessing pipeline).
    """

    coarse_classifier: CoarseCategoryClassifier
    attribute_embeddings: Dict[str, Tuple[Tensor, Tensor]]
    attribute_thresholds: Dict[str, float]
    classifier_name: str
    classifier_version: str

    def _normalize_embedding(self, embedding: Tensor) -> Tensor:
        """Normalize an embedding to shape ``(D,)`` with L2 norm equal to 1."""

        if embedding.ndim == 2:
            if embedding.shape[0] != 1:
                raise ValueError(f"embedding must have shape (D,) or (1, D), got {tuple(embedding.shape)}")
            emb = embedding[0]
        else:
            emb = embedding

        norm = emb.norm()
        if float(norm) == 0.0:
            raise ValueError("embedding must have non-zero norm")

        return emb / norm

    def _predict_boolean_with_margin(self, attribute_id: str, image_embedding: Tensor) -> Tuple[bool, float]:
        """Predict a boolean attribute and return its value and margin.

        The margin is defined as::

            margin = sim(positive_prompt) - sim(negative_prompt)
        """

        pair = self.attribute_embeddings.get(attribute_id)
        if pair is None:
            return False, 0.0

        pos_emb, neg_emb = pair

        device = image_embedding.device
        pos_emb = pos_emb.to(device)
        neg_emb = neg_emb.to(device)

        sim_pos = float(torch.matmul(image_embedding, pos_emb))
        sim_neg = float(torch.matmul(image_embedding, neg_emb))
        margin = sim_pos - sim_neg

        threshold = self.attribute_thresholds.get(attribute_id, 0.0)
        value = margin > threshold
        return value, margin

    @torch.no_grad()
    def classify_batch(self, embeddings: Iterable[Tensor]) -> List[SceneAttributes]:
        """Classify a batch of image embeddings into scene attributes."""

        results: List[SceneAttributes] = []

        for embedding in embeddings:
            img_emb = self._normalize_embedding(embedding)

            primary_category, scores = self.coarse_classifier.classify_from_image_embedding(img_emb)
            scene_type = primary_category.upper()

            scene_prob = float(scores.get(primary_category, 0.0))
            other_probs = [value for key, value in scores.items() if key != primary_category]
            second_prob = max(other_probs) if other_probs else 0.0
            scene_margin = scene_prob - second_prob

            has_person, has_person_margin = self._predict_boolean_with_margin("has_person", img_emb)
            has_text, has_text_margin = self._predict_boolean_with_margin("has_text", img_emb)
            is_screenshot, is_screenshot_margin = self._predict_boolean_with_margin("is_screenshot", img_emb)
            is_document, is_document_margin = self._predict_boolean_with_margin("is_document", img_emb)

            if not has_text:
                is_screenshot = False
                is_screenshot_margin = 0.0
                is_document = False
                is_document_margin = 0.0

            screenshot_like = {"SCREENSHOT", "DOCUMENT"}
            if scene_type not in screenshot_like:
                is_screenshot = False
                is_screenshot_margin = 0.0

            document_like = {"DOCUMENT"}
            if scene_type not in document_like:
                is_document = False
                is_document_margin = 0.0

            results.append(
                SceneAttributes(
                    scene_type=scene_type,
                    scene_confidence=scene_prob,
                    scene_margin=scene_margin,
                    has_text=has_text,
                    has_text_margin=has_text_margin,
                    has_person=has_person,
                    has_person_margin=has_person_margin,
                    is_screenshot=is_screenshot,
                    is_screenshot_margin=is_screenshot_margin,
                    is_document=is_document,
                    is_document_margin=is_document_margin,
                    classifier_name=self.classifier_name,
                    classifier_version=self.classifier_version,
                )
            )

        return results


def build_scene_classifier(settings: Settings) -> SceneClassifierWithAttributes:
    """Create a scene classifier wired to the configured SigLIP model."""

    processor, model, device = get_siglip_embedding_model(settings=settings)
    coarse_classifier: CoarseCategoryClassifier = build_siglip_coarse_classifier(
        siglip_model=model,
        siglip_processor=processor,
        device=device,
    )

    model_name = settings.models.embedding.resolved_model_name()
    classifier_name = model_name
    classifier_version = f"{model_name}-v1"

    LOGGER.info(
        "scene_classifier_build_start",
        extra={
            "classifier_name": classifier_name,
            "classifier_version": classifier_version,
            "device": str(device),
        },
    )

    attribute_configs: Sequence[AttributePromptConfig] = [
        AttributePromptConfig(
            id="has_person",
            positive_prompt="a photo with one or more people",
            negative_prompt="a photo without any people",
            threshold=0.05,
        ),
        AttributePromptConfig(
            id="has_text",
            positive_prompt="an image with a lot of text",
            negative_prompt="an image without any text",
            threshold=0.05,
        ),
        AttributePromptConfig(
            id="is_screenshot",
            positive_prompt=(
                "a full-screen screenshot of a computer or phone user interface with windows and icons and text"
            ),
            negative_prompt="a natural photo taken by a camera, not a screenshot",
            threshold=0.18,
        ),
        AttributePromptConfig(
            id="is_document",
            positive_prompt="a photo of a paper document or ID card on a table",
            negative_prompt="a normal photo that is not a document or ID",
            threshold=0.14,
        ),
    ]

    attribute_embeddings: Dict[str, Tuple[Tensor, Tensor]] = {}
    attribute_thresholds: Dict[str, float] = {}

    for config in attribute_configs:
        inputs = processor(
            text=[config.positive_prompt, config.negative_prompt],
            padding=True,
            return_tensors="pt",
        )
        if device is not None:
            inputs = inputs.to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        norms = text_features.norm(dim=-1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        normalized = text_features / norms

        pos_emb = normalized[0]
        neg_emb = normalized[1]

        attribute_embeddings[config.id] = (pos_emb, neg_emb)
        attribute_thresholds[config.id] = config.threshold

    LOGGER.info(
        "scene_classifier_build_done",
        extra={
            "classifier_name": classifier_name,
            "classifier_version": classifier_version,
            "num_attributes": len(attribute_embeddings),
        },
    )

    return SceneClassifierWithAttributes(
        coarse_classifier=coarse_classifier,
        attribute_embeddings=attribute_embeddings,
        attribute_thresholds=attribute_thresholds,
        classifier_name=classifier_name,
        classifier_version=classifier_version,
    )


__all__ = [
    "SceneAttributes",
    "AttributePromptConfig",
    "SceneClassifierWithAttributes",
    "build_scene_classifier",
]
