"""SigLIP + BLIP zero-shot image analysis helpers.

These utilities reuse the shared model loaders from :mod:`vibe_photos.ml.models`
and provide a small, typed API for running multilingual zero-shot
classification and captioning on full images.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from utils.logging import get_logger
from vibe_photos.config import Settings, load_settings
from vibe_photos.ml.models import get_blip_caption_model, get_siglip_embedding_model

LOGGER = get_logger(__name__, extra={"component": "siglip_blip"})


@dataclass
class SiglipBlipDetectionResult:
    """Structured result for SigLIP + BLIP analysis of a single image."""

    image_path: Path
    label_scores: dict[str, float]
    detected_labels: list[str]
    caption: str | None
    confidence: float


class SiglipBlipDetector:
    """Zero-shot image-level detector powered by SigLIP and BLIP.

    This helper performs:

    - Zero-shot classification over a caller-provided label set using SigLIP
      text and image encoders with cosine-similarity + softmax.
    - Optional BLIP captioning to produce a short natural-language summary.

    The detector reuses the shared SigLIP and BLIP singletons configured via
    :class:`vibe_photos.config.Settings` to avoid repeated model loads.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the detector from application settings.

        Args:
            settings: Optional pre-loaded settings instance. When omitted,
                configuration is loaded from ``config/settings.yaml``.
        """

        self._settings = settings or load_settings()

        self._siglip_processor, self._siglip_model, self._siglip_device = get_siglip_embedding_model(
            settings=self._settings
        )
        self._blip_processor, self._blip_model, self._blip_device = get_blip_caption_model(settings=self._settings)

        LOGGER.info(
            "siglip_blip_detector_init",
            extra={
                "siglip_device": str(self._siglip_device),
                "blip_device": str(self._blip_device),
            },
        )

    def detect(
        self,
        image_path: Path,
        candidate_labels: Sequence[str],
        confidence_threshold: float = 0.3,
        generate_caption: bool = True,
    ) -> SiglipBlipDetectionResult:
        """Run zero-shot classification and optional captioning on a single image.

        Args:
            image_path: Path to the image file to analyze.
            candidate_labels: Human-readable labels to score using SigLIP.
            confidence_threshold: Minimum probability for a label to be
                included in ``detected_labels``.
            generate_caption: When ``True``, run BLIP captioning.

        Returns:
            A :class:`SiglipBlipDetectionResult` instance containing scores,
            detected labels, and an optional caption.
        """

        image = Image.open(image_path).convert("RGB")

        scores = self._classify_with_siglip(image=image, labels=list(candidate_labels))
        caption: str | None = self._generate_caption_with_blip(image=image) if generate_caption else None

        detected = [label for label, score in scores.items() if score >= confidence_threshold]
        top_scores = sorted(scores.values(), reverse=True)[:3]
        confidence = float(np.mean(top_scores)) if top_scores else 0.0

        return SiglipBlipDetectionResult(
            image_path=image_path,
            label_scores=scores,
            detected_labels=detected,
            caption=caption,
            confidence=confidence,
        )

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        """Return a normalized SigLIP embedding for a single image."""

        image_inputs = self._siglip_processor(images=image, return_tensors="pt")
        image_inputs = image_inputs.to(self._siglip_device)

        with torch.no_grad():
            image_emb: Tensor = self._siglip_model.get_image_features(**image_inputs)

        emb = image_emb[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return cast(NDArray[np.float32], emb.detach().cpu().numpy().astype(np.float32))

    def _classify_with_siglip(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        """Compute zero-shot classification scores via SigLIP."""

        if not labels:
            return {}

        image_inputs = self._siglip_processor(images=image, return_tensors="pt")
        text_inputs = self._siglip_processor(text=labels, padding=True, return_tensors="pt")

        image_inputs = image_inputs.to(self._siglip_device)
        text_inputs = text_inputs.to(self._siglip_device)

        with torch.no_grad():
            image_emb: Tensor = self._siglip_model.get_image_features(**image_inputs)
            text_emb: Tensor = self._siglip_model.get_text_features(**text_inputs)

        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        logits = image_emb @ text_emb.T  # (1, num_labels)
        probs = torch.softmax(logits[0], dim=-1)
        probs_cpu = probs.detach().cpu().numpy().tolist()

        return {label: float(prob) for label, prob in zip(labels, probs_cpu, strict=True)}

    def _generate_caption_with_blip(self, image: Image.Image) -> str:
        """Generate a natural language caption using BLIP."""

        inputs = self._blip_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self._blip_device)

        with torch.no_grad():
            generated_ids = self._blip_model.generate(**inputs, max_new_tokens=50)

        return str(self._blip_processor.decode(generated_ids[0], skip_special_tokens=True))


__all__ = ["SiglipBlipDetector", "SiglipBlipDetectionResult"]
