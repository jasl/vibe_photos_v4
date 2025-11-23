"""Zero-shot coarse category classification built on top of image embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CoarseCategory:
    """Configuration for a single coarse category."""

    id: str
    display_name: str
    prompts: Sequence[str]


class CoarseCategoryClassifier:
    """Zero-shot coarse category classifier based on text and image embeddings.

    This helper is model-agnostic: callers provide a function that turns a sequence of
    prompt strings into a 2D tensor of text embeddings. The classifier only handles
    normalization and cosine-similarity style scoring against a single image embedding
    and converts them into probability-like scores.
    """

    def __init__(
        self,
        categories: Sequence[CoarseCategory],
        encode_text: Callable[[Sequence[str]], Tensor],
        threshold: float = 0.05,
        score_min: float = 0.30,
        fallback_category_id: str = "other",
    ) -> None:
        """Initialize the classifier.

        Args:
            categories: Coarse category definitions to score against.
            encode_text: Function that encodes a sequence of prompt strings into
                a tensor of shape ``(num_prompts, embedding_dim)``.
            threshold: Minimum margin (difference between the top-1 and top-2
                probabilities) required for a category to be selected as
                primary. Below this, the fallback category is used when
                available.
            score_min: Minimum top-1 probability required for a category to be
                considered confident. If the top-1 probability falls below this
                value, the fallback category is used.
            fallback_category_id: Category identifier to use as a fallback when
                confidence is low. The category does not need to have any
                prompts; it is treated as a catch-all bucket.
        """
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if score_min < 0.0:
            raise ValueError("score_min must be non-negative")

        self._categories: list[CoarseCategory] = list(categories)
        self._encode_text: Callable[[Sequence[str]], Tensor] = encode_text
        self._threshold: float = threshold
        self._score_min: float = score_min
        self._fallback_category_id: str = fallback_category_id
        self._text_embeddings: dict[str, Tensor] = {}

        self._prepare_text_embeddings()

    @property
    def categories(self) -> Sequence[CoarseCategory]:
        """Return the configured coarse categories."""

        return tuple(self._categories)

    @property
    def threshold(self) -> float:
        """Return the minimum probability margin required for a category to become primary."""

        return self._threshold

    @property
    def score_min(self) -> float:
        """Return the minimum top-1 probability required for a confident prediction."""

        return self._score_min

    def _prepare_text_embeddings(self) -> None:
        """Encode and normalize all category prompts once at startup."""

        for category in self._categories:
            if not category.prompts:
                continue

            text_embeddings = self._encode_text(list(category.prompts))
            if text_embeddings.ndim != 2:
                raise ValueError("encode_text must return a 2D tensor of shape (num_prompts, embedding_dim)")

            norms = text_embeddings.norm(dim=-1, keepdim=True)
            # Avoid division by zero for degenerate embeddings.
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            normalized = text_embeddings / norms

            self._text_embeddings[category.id] = normalized

    def classify_from_image_embedding(self, image_embedding: Tensor) -> tuple[str, dict[str, float]]:
        """Classify an image embedding into a primary coarse category.

        Args:
            image_embedding: Image embedding tensor of shape ``(embedding_dim,)``
                or ``(1, embedding_dim)``.

        Returns:
            A tuple of:
                - primary_category_id: identifier of the selected coarse category.
                - scores: mapping from category identifier to probability in [0.0, 1.0].
        """
        if image_embedding.ndim == 2:
            if image_embedding.shape[0] != 1:
                raise ValueError("image_embedding must have shape (embedding_dim,) or (1, embedding_dim)")
            image_embedding = image_embedding[0]

        if image_embedding.ndim != 1:
            raise ValueError("image_embedding must have shape (embedding_dim,) or (1, embedding_dim)")

        norm = image_embedding.norm()
        if norm == 0:
            raise ValueError("image_embedding must have non-zero norm")

        normalized_image = image_embedding / norm

        scores: dict[str, float] = {}
        scored_category_ids: list[str] = []

        for category in self._categories:
            text_embeddings = self._text_embeddings.get(category.id)
            if text_embeddings is None:
                # Category without prompts, or not prepared for scoring.
                continue

            # (embedding_dim,) @ (embedding_dim, num_prompts) -> (num_prompts,)
            similarities = normalized_image @ text_embeddings.T
            max_score = float(similarities.max().item())
            scores[category.id] = max_score
            scored_category_ids.append(category.id)

        if not scored_category_ids:
            primary_category_id = self._fallback_category_id
            if self._fallback_category_id not in scores:
                scores[self._fallback_category_id] = 0.0
            return primary_category_id, scores

        raw_scores = torch.tensor([scores[cid] for cid in scored_category_ids], dtype=normalized_image.dtype)
        probs = torch.softmax(raw_scores, dim=-1)
        probs_cpu = probs.detach().cpu()

        for idx, category_id in enumerate(scored_category_ids):
            scores[category_id] = float(probs_cpu[idx])

        num_categories = probs.shape[0]
        if num_categories == 1:
            best_index = 0
            best_prob = float(probs_cpu[0])
            second_prob = 0.0
        else:
            top_values, top_indices = torch.topk(probs, k=2)
            best_prob = float(top_values[0])
            second_prob = float(top_values[1])
            best_index = int(top_indices[0])

        best_category_id = scored_category_ids[best_index]
        margin = best_prob - second_prob

        primary_category_id = best_category_id
        if primary_category_id != self._fallback_category_id:
            if best_prob < self._score_min and margin < self._threshold:
                LOGGER.debug(
                    "coarse_category_fallback_other",
                    extra={
                        "top1_category": best_category_id,
                        "top1_prob": best_prob,
                        "top2_prob": second_prob,
                        "margin": margin,
                    },
                )
                primary_category_id = self._fallback_category_id

        if self._fallback_category_id not in scores:
            scores[self._fallback_category_id] = 0.0

        return primary_category_id, scores


DEFAULT_COARSE_CATEGORIES: list[CoarseCategory] = [
    CoarseCategory(
        id="food",
        display_name="Food",
        prompts=[
            "a photo of food",
            "a plate of cooked food",
            "a restaurant meal",
        ],
    ),
    CoarseCategory(
        id="electronics",
        display_name="Electronics",
        prompts=[
            "an electronic device",
            "a laptop on a desk",
            "a smartphone on a table",
        ],
    ),
    CoarseCategory(
        id="screenshot",
        display_name="Screenshot",
        prompts=[
            "a computer screenshot",
            "a phone screenshot",
        ],
    ),
    CoarseCategory(
        id="document",
        display_name="Document",
        prompts=[
            "a photo of a document",
            "a photo of a receipt or contract",
        ],
    ),
    CoarseCategory(
        id="people",
        display_name="People",
        prompts=[
            "a portrait photo of a person",
            "a group photo of people",
        ],
    ),
    CoarseCategory(
        id="landscape",
        display_name="Landscape",
        prompts=[
            "a landscape photo of nature",
            "a cityscape photo",
        ],
    ),
    CoarseCategory(
        id="other",
        display_name="Other",
        prompts=[],
    ),
]


def build_siglip_coarse_classifier(
    siglip_model: Any,
    siglip_processor: Any,
    categories: Sequence[CoarseCategory] | None = None,
    threshold: float = 0.01,
    score_min: float = 0.15,
    device: str | torch.device | None = None,
) -> CoarseCategoryClassifier:
    """Create a coarse category classifier wired to a SigLIP-style model.

    This helper assumes the underlying model exposes a CLIP-like API with a
    ``get_text_features`` method. If the SigLIP version in use does not provide
    this interface, callers should provide a custom ``encode_text`` function and
    construct :class:`CoarseCategoryClassifier` directly instead.

    Args:
        siglip_model: Text/image model instance, typically loaded via
            ``transformers.AutoModel.from_pretrained`` or a SigLIP-specific
            class.
        siglip_processor: Paired processor (tokenizer + image processor) for
            the model.
        categories: Optional custom coarse categories. When omitted,
            :data:`DEFAULT_COARSE_CATEGORIES` is used.
        threshold: Minimum margin between top-1 and top-2 probabilities
            required for a category to be selected as primary.
        score_min: Minimum top-1 probability required for a category to be
            considered confident.
        device: Optional device string or torch device on which to perform text
            encoding. When ``None``, the processor outputs are left on their
            default device.

    Returns:
        A configured :class:`CoarseCategoryClassifier` instance.
    """

    def encode_text(prompts: Sequence[str]) -> Tensor:
        if not prompts:
            return torch.empty(0, 0)

        inputs = siglip_processor(text=list(prompts), padding=True, return_tensors="pt")
        if device is not None:
            inputs = inputs.to(device)

        with torch.no_grad():
            text_embeddings = cast(Tensor, siglip_model.get_text_features(**inputs).detach().cpu())

        return text_embeddings

    return CoarseCategoryClassifier(
        categories=categories or DEFAULT_COARSE_CATEGORIES,
        encode_text=encode_text,
        threshold=threshold,
        score_min=score_min,
    )


__all__ = [
    "CoarseCategory",
    "CoarseCategoryClassifier",
    "DEFAULT_COARSE_CATEGORIES",
    "build_siglip_coarse_classifier",
]
