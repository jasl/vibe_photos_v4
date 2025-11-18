"""Lightweight scene classification and attribute prediction for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from torch import Tensor

from utils.logging import get_logger
from vibe_photos.config import Settings
from vibe_photos.ml.coarse_categories import CoarseCategoryClassifier, build_siglip_coarse_classifier
from vibe_photos.ml.models import get_siglip_embedding_model


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SceneAttributes:
    """Predicted scene type and boolean attributes for an image."""

    scene_type: str
    scene_confidence: float
    has_text: bool
    has_person: bool
    is_screenshot: bool
    is_document: bool
    classifier_name: str
    classifier_version: str


def build_scene_classifier(settings: Settings) -> Tuple[CoarseCategoryClassifier, str, str]:
    """Create a coarse scene classifier wired to the configured SigLIP model.

    Args:
        settings: Loaded application settings containing model configuration.

    Returns:
        A tuple of the classifier instance, classifier name, and classifier version.
    """

    processor, model, device = get_siglip_embedding_model(settings=settings)
    classifier = build_siglip_coarse_classifier(
        siglip_model=model,
        siglip_processor=processor,
        device=device,
    )
    model_name = settings.models.embedding.resolved_model_name()
    classifier_name = model_name
    classifier_version = f"{model_name}-v1"
    return classifier, classifier_name, classifier_version


def classify_image_embeddings(
    classifier: CoarseCategoryClassifier,
    embeddings: Iterable[Tensor],
    classifier_name: str,
    classifier_version: str,
) -> List[SceneAttributes]:
    """Classify a batch of image embeddings into scene attributes.

    The initial implementation focuses on coarse ``scene_type`` and confidence
    using the zero-shot SigLIP-based classifier. Boolean attributes such as
    ``has_text`` and ``has_person`` are left for future iterations and are
    currently derived from the primary coarse category.

    Args:
        classifier: Configured coarse category classifier.
        embeddings: Iterable of image embeddings with shape ``(embedding_dim,)``.
        classifier_name: Logical classifier name to record in outputs.
        classifier_version: Logical classifier version string.

    Returns:
        List of SceneAttributes, one per input embedding.
    """

    results: List[SceneAttributes] = []

    for embedding in embeddings:
        primary_category, scores = classifier.classify_from_image_embedding(embedding)
        confidence = float(scores.get(primary_category, 0.0))

        has_person = primary_category == "people"
        is_screenshot = primary_category == "screenshot"
        is_document = primary_category == "document"
        has_text = is_screenshot or is_document

        attributes = SceneAttributes(
            scene_type=primary_category.upper(),
            scene_confidence=confidence,
            has_text=has_text,
            has_person=has_person,
            is_screenshot=is_screenshot,
            is_document=is_document,
            classifier_name=classifier_name,
            classifier_version=classifier_version,
        )
        results.append(attributes)

    return results


__all__ = ["SceneAttributes", "build_scene_classifier", "classify_image_embeddings"]

