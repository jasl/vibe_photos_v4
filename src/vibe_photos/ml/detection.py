"""Open-vocabulary detection backends (OWL-ViT, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

import torch
from PIL import Image
from torch import Tensor
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from utils.logging import get_logger
from vibe_photos.config import DetectionModelConfig, Settings, load_settings
from vibe_photos.ml.models import _select_device


LOGGER = get_logger(__name__, extra={"component": "detection"})


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box with normalized coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Detection:
    """Single detection result from an open-vocabulary detector."""

    bbox: BoundingBox
    label: str
    score: float
    backend: str
    model_name: str


class Detector(Protocol):
    """Protocol for open-vocabulary detectors."""

    def detect(self, image: Image.Image, prompts: Sequence[str]) -> List[Detection]:
        """Run detection on a single image given text prompts."""


class OwlVitDetector:
    """OWL-ViT based open-vocabulary detector."""

    def __init__(self, config: DetectionModelConfig, device: torch.device) -> None:
        """Initialize the OWL-ViT detector from configuration."""

        self._config = config
        self._device = device

        model_name = config.model_name
        self._processor = OwlViTProcessor.from_pretrained(model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)

        self._backend = config.backend
        self._model_name = model_name

        LOGGER.info(
            "owlvit_detector_init",
            extra={
                "model_name": model_name,
                "backend": self._backend,
                "device": str(device),
                "score_threshold": float(config.score_threshold),
                "max_regions_per_image": int(config.max_regions_per_image),
            },
        )

    def detect(self, image: Image.Image, prompts: Sequence[str]) -> List[Detection]:
        """Run OWL-ViT detection for a single image."""

        if not prompts:
            return []

        text_batch = [list(prompts)]
        image_batch = [image]

        inputs = self._processor(text=text_batch, images=image_batch, return_tensors="pt")
        inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        width, height = image.size
        target_sizes = torch.tensor([[height, width]], device=self._device)

        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(self._config.score_threshold),
        )
        result = results[0]

        boxes: Tensor = result["boxes"]
        scores: Tensor = result["scores"]
        labels: Tensor = result["labels"]

        detections: List[Detection] = []

        for box, score, label_idx in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box.tolist()

            x_min_norm = max(0.0, min(1.0, x_min / float(width)))
            y_min_norm = max(0.0, min(1.0, y_min / float(height)))
            x_max_norm = max(0.0, min(1.0, x_max / float(width)))
            y_max_norm = max(0.0, min(1.0, y_max / float(height)))

            index = int(label_idx)
            label = prompts[index] if 0 <= index < len(prompts) else str(index)

            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x_min=x_min_norm,
                        y_min=y_min_norm,
                        x_max=x_max_norm,
                        y_max=y_max_norm,
                    ),
                    label=label,
                    score=float(score),
                    backend=self._backend,
                    model_name=self._model_name,
                )
            )

        if self._config.max_regions_per_image > 0 and len(detections) > self._config.max_regions_per_image:
            detections.sort(key=lambda det: det.score, reverse=True)
            detections = detections[: int(self._config.max_regions_per_image)]

        return detections


def build_owlvit_detector(settings: Settings | None = None) -> OwlVitDetector:
    """Create an OWL-ViT detector from application settings."""

    resolved_settings = settings or load_settings()
    cfg = resolved_settings.models.detection

    if not cfg.enabled:
        raise RuntimeError("Detection is disabled in settings (models.detection.enabled is False).")

    if cfg.backend != "owlvit":
        raise ValueError(f"Unsupported detection backend for build_owlvit_detector: {cfg.backend!r}")

    device = _select_device(cfg.device)
    return OwlVitDetector(config=cfg, device=device)


__all__ = ["BoundingBox", "Detection", "Detector", "OwlVitDetector", "build_owlvit_detector"]
