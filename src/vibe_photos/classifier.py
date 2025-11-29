"""Lightweight scene classification and attribute prediction for the pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from utils.logging import get_logger
from vibe_photos.config import Settings
from vibe_photos.labels.scene_schema import ATTRIBUTE_LABEL_KEYS, scene_type_from_label_key
from vibe_photos.ml.coarse_categories import (
    CoarseCategoryClassifier,
    build_siglip_coarse_classifier,
)
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

    has_animal: bool
    has_animal_margin: float

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

    coarse_classifier: "CoarseSceneClassifier"
    attribute_embeddings: dict[str, tuple[Tensor, Tensor]]
    attribute_thresholds: dict[str, float]
    classifier_name: str
    classifier_version: str
    attribute_head: "AttributeHead | None" = None

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

        return cast(Tensor, emb / norm)

    def _predict_boolean_with_margin(self, attribute_id: str, image_embedding: Tensor) -> tuple[bool, float]:
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
    def classify_batch(self, embeddings: Iterable[Tensor]) -> list[SceneAttributes]:
        """Classify a batch of image embeddings into scene attributes."""

        results: list[SceneAttributes] = []

        for embedding in embeddings:
            img_emb = self._normalize_embedding(embedding)

            primary_category, scores = self.coarse_classifier.classify_from_image_embedding(img_emb)
            scene_type = primary_category.upper()

            scene_prob = float(scores.get(primary_category, 0.0))
            other_probs = [value for key, value in scores.items() if key != primary_category]
            second_prob = max(other_probs) if other_probs else 0.0
            scene_margin = scene_prob - second_prob

            if self.attribute_head is not None:
                attr_map = self.attribute_head.predict(img_emb)
                has_person, has_person_margin = attr_map.get("has_person", (False, 0.0))
                has_animal, has_animal_margin = attr_map.get("has_animal", (False, 0.0))
                has_text, has_text_margin = attr_map.get("has_text", (False, 0.0))
                _, is_screenshot_margin = attr_map.get("is_screenshot", (False, 0.0))
                _, is_document_margin = attr_map.get("is_document", (False, 0.0))
            else:
                has_person, has_person_margin = self._predict_boolean_with_margin("has_person", img_emb)
                has_animal, has_animal_margin = self._predict_boolean_with_margin("has_animal", img_emb)
                has_text, has_text_margin = self._predict_boolean_with_margin("has_text", img_emb)
                _, is_screenshot_margin = self._predict_boolean_with_margin("is_screenshot", img_emb)
                _, is_document_margin = self._predict_boolean_with_margin("is_document", img_emb)

            is_screenshot = scene_type == "SCREENSHOT"
            is_document = scene_type == "DOCUMENT"

            results.append(
                SceneAttributes(
                    scene_type=scene_type,
                    scene_confidence=scene_prob,
                    scene_margin=scene_margin,
                    has_text=has_text,
                    has_text_margin=has_text_margin,
                    has_person=has_person,
                    has_person_margin=has_person_margin,
                    has_animal=has_animal,
                    has_animal_margin=has_animal_margin,
                    is_screenshot=is_screenshot,
                    is_screenshot_margin=is_screenshot_margin,
                    is_document=is_document,
                    is_document_margin=is_document_margin,
                    classifier_name=self.classifier_name,
                    classifier_version=self.classifier_version,
                )
            )

        return results


class CoarseSceneClassifier(Protocol):
    """Protocol for coarse scene classifiers."""

    def classify_from_image_embedding(self, image_embedding: Tensor) -> tuple[str, dict[str, float]]:
        """Return a primary category identifier and per-category scores."""
        ...


class AttributeHead(Protocol):
    """Protocol for learned attribute heads."""

    def predict(self, image_embedding: Tensor) -> dict[str, tuple[bool, float]]:
        """Return mapping from attribute id to (value, margin)."""
        ...


class LearnedSceneHeadAdapter:
    """Coarse scene classifier backed by a trained linear head over embeddings.

    This helper exposes the same :meth:`classify_from_image_embedding` interface
    as :class:`CoarseCategoryClassifier` so it can be plugged into
    :class:`SceneClassifierWithAttributes`.
    """

    def __init__(self, *, linear: torch.nn.Linear, scene_labels: list[str]) -> None:
        self._linear = linear
        self._linear.eval()
        self._scene_labels = list(scene_labels)
        self._device = next(self._linear.parameters()).device

    @torch.no_grad()
    def classify_from_image_embedding(self, image_embedding: Tensor) -> tuple[str, dict[str, float]]:
        """Return (primary_category_id, scores) using the learned head.

        - ``primary_category_id`` is a lowercase coarse scene type
          (e.g., ``"food"`` or ``"people"``).
        - ``scores`` maps coarse types to probabilities in ``[0.0, 1.0]``.
        """

        if image_embedding.ndim == 2:
            if image_embedding.shape[0] != 1:
                raise ValueError(f"image_embedding must have shape (D,) or (1, D), got {tuple(image_embedding.shape)}")
            emb = image_embedding[0]
        else:
            emb = image_embedding

        if emb.ndim != 1:
            raise ValueError(f"image_embedding must have shape (D,) or (1, D), got {tuple(image_embedding.shape)}")

        emb = emb.to(self._device)
        logits = self._linear(emb)
        probs = torch.softmax(logits, dim=-1)
        probs_cpu = probs.detach().cpu()

        scores: dict[str, float] = {}
        for idx, prob in enumerate(probs_cpu):
            if idx >= len(self._scene_labels):
                break
            scene_key = self._scene_labels[idx]
            scene_type = scene_type_from_label_key(scene_key) or "OTHER"
            category_id = scene_type.lower()
            prob_value = float(prob.item())
            existing = scores.get(category_id)
            scores[category_id] = prob_value if existing is None else max(existing, prob_value)

        if not scores:
            # Fallback to OTHER when something goes wrong.
            return "other", {"other": 1.0}

        primary_category = max(scores.items(), key=lambda item: item[1])[0]
        if "other" not in scores:
            scores.setdefault("other", 0.0)

        return primary_category, scores


def _project_root() -> Path:
    """Best-effort detection of the repository root for model discovery."""

    return Path(__file__).resolve().parents[2]


def _default_scene_head_path() -> Path:
    """Return the default path where the Qwen-trained scene head is stored."""

    return _project_root() / "models" / "scene_head_from_qwen.pt"


def _default_attribute_head_path() -> Path:
    """Return the default path where the Qwen-trained attribute head is stored."""

    return _project_root() / "models" / "attribute_head_from_qwen.pt"


def _load_learned_scene_head() -> LearnedSceneHeadAdapter | None:
    """Load a trained scene head from disk if available.

    Returns ``None`` when the file is missing or invalid, in which case callers
    should fall back to the zero-shot coarse category classifier.
    """

    path = _default_scene_head_path()
    if not path.exists():
        return None

    try:
        payload: dict[str, Any] = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("scene_head_load_error", extra={"path": str(path), "error": str(exc)})
        return None

    scene_labels = payload.get("scene_labels")
    state = payload.get("state")
    if not isinstance(scene_labels, list) or not scene_labels or not isinstance(state, dict):
        LOGGER.error(
            "scene_head_payload_invalid",
            extra={"path": str(path), "keys": list(payload.keys())},
        )
        return None

    input_dim = state.get("input_dim")
    num_classes = state.get("num_classes")
    state_dict = state.get("state_dict")
    if not isinstance(input_dim, int) or not isinstance(num_classes, int) or not isinstance(state_dict, dict):
        LOGGER.error(
            "scene_head_state_invalid",
            extra={
                "path": str(path),
                "has_input_dim": isinstance(input_dim, int),
                "has_num_classes": isinstance(num_classes, int),
                "has_state_dict": isinstance(state_dict, dict),
            },
        )
        return None

    try:
        linear = torch.nn.Linear(input_dim, num_classes)
        linear.load_state_dict(state_dict)
        linear.eval()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("scene_head_build_error", extra={"path": str(path), "error": str(exc)})
        return None

    LOGGER.info(
        "scene_head_loaded",
        extra={
            "path": str(path),
            "num_classes": num_classes,
        },
    )
    return LearnedSceneHeadAdapter(linear=linear, scene_labels=[str(label) for label in scene_labels])


class LearnedAttributeHead:
    """Attribute head backed by a trained linear model over embeddings."""

    def __init__(
        self,
        *,
        linear: torch.nn.Linear,
        attr_label_keys: list[str],
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._linear = linear
        self._linear.eval()
        self._device = next(self._linear.parameters()).device

        reverse_map: dict[str, str] = {label_key: attr_id for attr_id, label_key in ATTRIBUTE_LABEL_KEYS.items()}
        attr_ids: list[str] = []
        for key in attr_label_keys:
            attr_id = reverse_map.get(str(key))
            if attr_id is not None:
                attr_ids.append(attr_id)
        self._attr_ids = attr_ids
        base_thresholds = thresholds or {}
        self._thresholds = {attr_id: float(base_thresholds.get(attr_id, 0.5)) for attr_id in attr_ids}

    @torch.no_grad()
    def predict(self, image_embedding: Tensor) -> dict[str, tuple[bool, float]]:
        """Return mapping from attribute id to (bool value, margin ~= logit)."""

        if image_embedding.ndim == 2:
            if image_embedding.shape[0] != 1:
                raise ValueError(f"image_embedding must have shape (D,) or (1, D), got {tuple(image_embedding.shape)}")
            emb = image_embedding[0]
        else:
            emb = image_embedding

        if emb.ndim != 1:
            raise ValueError(f"image_embedding must have shape (D,) or (1, D), got {tuple(image_embedding.shape)}")

        emb = emb.to(self._device)
        logits = self._linear(emb)
        probs = torch.sigmoid(logits)

        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()

        result: dict[str, tuple[bool, float]] = {}
        for idx, (logit, prob) in enumerate(zip(logits_cpu, probs_cpu)):
            if idx >= len(self._attr_ids):
                break
            attr_id = self._attr_ids[idx]
            margin = float(logit.item())
            threshold = self._thresholds.get(attr_id, 0.5)
            value = bool(prob.item() >= threshold)
            result[attr_id] = (value, margin)
        return result


def _load_learned_attribute_head(settings: Settings) -> LearnedAttributeHead | None:
    """Load a trained attribute head from disk if available."""

    path = _default_attribute_head_path()
    if not path.exists():
        return None

    try:
        payload: dict[str, Any] = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("attr_head_load_error", extra={"path": str(path), "error": str(exc)})
        return None

    attr_keys = payload.get("attr_keys")
    state = payload.get("state")
    if not isinstance(attr_keys, list) or not attr_keys or not isinstance(state, dict):
        LOGGER.error(
            "attr_head_payload_invalid",
            extra={"path": str(path), "keys": list(payload.keys())},
        )
        return None

    input_dim = state.get("input_dim")
    num_attrs = state.get("num_attrs")
    state_dict = state.get("state_dict")
    if not isinstance(input_dim, int) or not isinstance(num_attrs, int) or not isinstance(state_dict, dict):
        LOGGER.error(
            "attr_head_state_invalid",
            extra={
                "path": str(path),
                "has_input_dim": isinstance(input_dim, int),
                "has_num_attrs": isinstance(num_attrs, int),
                "has_state_dict": isinstance(state_dict, dict),
            },
        )
        return None

    try:
        linear = torch.nn.Linear(input_dim, num_attrs)
        linear.load_state_dict(state_dict)
        linear.eval()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("attr_head_build_error", extra={"path": str(path), "error": str(exc)})
        return None

    LOGGER.info(
        "attr_head_loaded",
        extra={
            "path": str(path),
            "num_attrs": num_attrs,
        },
    )
    thresholds = settings.attributes.head_thresholds
    return LearnedAttributeHead(
        linear=linear,
        attr_label_keys=[str(key) for key in attr_keys],
        thresholds=thresholds,
    )


def build_scene_classifier(settings: Settings) -> SceneClassifierWithAttributes:
    """Create a scene classifier wired to the configured SigLIP model."""

    processor, model, device = get_siglip_embedding_model(settings=settings)
    model_name = settings.models.embedding.resolved_model_name()

    learned_head = _load_learned_scene_head()
    if learned_head is not None:
        coarse_classifier = learned_head
        classifier_name = f"{model_name}-qwen_head"
        classifier_version = f"{model_name}-qwen_head-v1"
        LOGGER.info(
            "scene_classifier_using_learned_head",
            extra={
                "classifier_name": classifier_name,
                "classifier_version": classifier_version,
                "head_path": str(_default_scene_head_path()),
                "device": str(device),
            },
        )
    else:
        coarse_classifier: CoarseSceneClassifier = build_siglip_coarse_classifier(
            siglip_model=model,
            siglip_processor=processor,
            device=device,
        )
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
            id="has_animal",
            positive_prompt="a photo with one or more animals or pets, such as a cat or a dog",
            negative_prompt="a photo without any animals or pets",
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

    attribute_embeddings: dict[str, tuple[Tensor, Tensor]] = {}
    attribute_thresholds: dict[str, float] = {}

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

    attribute_head = _load_learned_attribute_head(settings)
    if attribute_head is not None:
        LOGGER.info(
            "scene_classifier_using_learned_attribute_head",
            extra={
                "classifier_name": classifier_name,
                "classifier_version": classifier_version,
                "head_path": str(_default_attribute_head_path()),
            },
        )

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
        attribute_head=attribute_head,
    )


__all__ = [
    "SceneAttributes",
    "AttributePromptConfig",
    "SceneClassifierWithAttributes",
    "build_scene_classifier",
]
