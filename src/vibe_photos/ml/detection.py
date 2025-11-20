"""Open-vocabulary detection backends (OWL-ViT, etc.)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor, device as TorchDevice
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from utils.logging import get_logger
from vibe_photos.config import DetectionModelConfig, Settings, load_settings
from vibe_photos.ml.models import _select_device


LOGGER = get_logger(__name__, extra={"component": "detection"})


_OWL_PROCESSOR: OwlViTProcessor | None = None
_OWL_MODEL: OwlViTForObjectDetection | None = None
_OWL_DEVICE: TorchDevice | None = None
_OWL_MODEL_NAME: str | None = None


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


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute intersection-over-union (IoU) for normalized bounding boxes."""

    ax1, ay1, ax2, ay2 = a.x_min, a.y_min, a.x_max, a.y_max
    bx1, by1, bx2, by2 = b.x_min, b.y_min, b.x_max, b.y_max

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def non_max_suppression(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    """Class-agnostic non-max suppression.

    For highly overlapping boxes, keep only the highest-score detection.
    """

    if not detections:
        return []

    if iou_threshold <= 0.0:
        return list(detections)

    sorted_dets = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: List[Detection] = []

    for det in sorted_dets:
        duplicate = False
        for kept_det in kept:
            if iou(det.bbox, kept_det.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


def detection_priority(det: Detection, *, area_gamma: float = 0.3, center_penalty: float = 0.6) -> float:
    """Compute a heuristic priority score for a detection.

    Higher when the region is larger, closer to the image center, and has a higher detector score.
    """

    bbox = det.bbox
    width = max(0.0, bbox.x_max - bbox.x_min)
    height = max(0.0, bbox.y_max - bbox.y_min)
    area = max(1e-6, width * height)

    cx = 0.5 * (bbox.x_min + bbox.x_max)
    cy = 0.5 * (bbox.y_min + bbox.y_max)
    distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    area_weight = area**area_gamma
    center_weight = 1.0 - center_penalty * distance
    if center_weight < 0.0:
        center_weight = 0.0

    return det.score * area_weight * center_weight


def filter_secondary_regions_by_priority(
    detections: Sequence[Detection],
    priorities: Sequence[float],
    *,
    max_regions: int,
    secondary_min_priority: float,
    secondary_min_relative_to_primary: float,
) -> Tuple[List[Detection], List[float]]:
    """Filter low-priority secondary regions after sorting by priority.

    Assumes detections and priorities are aligned and already sorted from highest to lowest priority.
    Always keeps the primary region and retains secondary regions that meet both absolute and
    relative thresholds, up to ``max_regions`` total.
    """

    if not detections:
        return [], []

    if len(detections) != len(priorities):
        raise ValueError("detections and priorities must have the same length")

    primary_priority = float(priorities[0])

    kept_detections: List[Detection] = [detections[0]]
    kept_priorities: List[float] = [primary_priority]

    if max_regions <= 1:
        return kept_detections, kept_priorities

    for det, priority in zip(detections[1:], priorities[1:]):
        if len(kept_detections) >= max_regions:
            break

        if priority < secondary_min_priority:
            continue

        if primary_priority > 0.0:
            relative = priority / primary_priority
            if relative < secondary_min_relative_to_primary:
                continue

        kept_detections.append(det)
        kept_priorities.append(priority)

    return kept_detections, kept_priorities


def _load_owlvit(config: DetectionModelConfig) -> tuple[OwlViTProcessor, OwlViTForObjectDetection, TorchDevice]:
    """Load (or reuse) the OWL-ViT processor/model for the current process."""

    model_name = config.model_name
    device = _select_device(config.device)

    global _OWL_PROCESSOR, _OWL_MODEL, _OWL_DEVICE, _OWL_MODEL_NAME
    if (
        _OWL_PROCESSOR is not None
        and _OWL_MODEL is not None
        and _OWL_DEVICE is not None
        and _OWL_MODEL_NAME == model_name
        and _OWL_DEVICE == device
    ):
        return _OWL_PROCESSOR, _OWL_MODEL, _OWL_DEVICE

    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    _OWL_PROCESSOR = processor
    _OWL_MODEL = model
    _OWL_DEVICE = device
    _OWL_MODEL_NAME = model_name

    LOGGER.info(
        "owlvit_model_loaded",
        extra={
            "model_name": model_name,
            "device": str(device),
        },
    )

    return processor, model, device


class OwlVitDetector:
    """OWL-ViT based open-vocabulary detector."""

    def __init__(
        self,
        config: DetectionModelConfig,
        device: TorchDevice,
        *,
        processor: OwlViTProcessor | None = None,
        model: OwlViTForObjectDetection | None = None,
    ) -> None:
        """Initialize the OWL-ViT detector from configuration."""

        self._config = config
        self._device = device

        model_name = config.model_name
        self._processor = processor or OwlViTProcessor.from_pretrained(model_name)
        self._model = (model or OwlViTForObjectDetection.from_pretrained(model_name)).to(device)
        self._model.eval()

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

        return detections


def build_owlvit_detector(settings: Settings | None = None) -> OwlVitDetector:
    """Create an OWL-ViT detector from application settings."""

    resolved_settings = settings or load_settings()
    cfg = resolved_settings.models.detection

    if not cfg.enabled:
        raise RuntimeError("Detection is disabled in settings (models.detection.enabled is False).")

    if cfg.backend != "owlvit":
        raise ValueError(f"Unsupported detection backend for build_owlvit_detector: {cfg.backend!r}")

    processor, model, device = _load_owlvit(cfg)
    return OwlVitDetector(config=cfg, device=device, processor=processor, model=model)


__all__ = [
    "BoundingBox",
    "Detection",
    "Detector",
    "OwlVitDetector",
    "build_owlvit_detector",
    "iou",
    "non_max_suppression",
    "detection_priority",
    "filter_secondary_regions_by_priority",
]
