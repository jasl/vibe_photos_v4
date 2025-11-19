"""Configuration loader and typed settings for the Vibe Photos application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import inflect
import yaml

from vibe_photos.ml.model_presets import (
    BLIP_IMAGE_CAPTIONING_BASE,
    SIGLIP2_BASE_PATCH16_224,
    BLIP_PRESETS,
    SIGLIP_PRESETS,
)


@dataclass
class EmbeddingModelConfig:
    """Configuration for the embedding model (SigLIP)."""

    backend: str = "siglip"
    model_name: str = SIGLIP2_BASE_PATCH16_224
    preset: Optional[str] = None
    device: str = "auto"
    batch_size: int = 16

    def resolved_model_name(self) -> str:
        """Return the concrete model name to load for embeddings.

        Resolution order:
        1. If ``preset`` is set, resolve via :data:`SIGLIP_PRESETS`.
        2. Otherwise, use ``model_name``.
        3. Fallback to the default SigLIP2 base checkpoint.
        """
        if self.preset:
            preset_name = SIGLIP_PRESETS.get(self.preset)
            if preset_name is None:
                raise ValueError(f"Unsupported SigLIP preset: {self.preset!r}")
            return preset_name

        if self.model_name:
            return self.model_name

        return SIGLIP2_BASE_PATCH16_224


@dataclass
class CaptionModelConfig:
    """Configuration for the captioning model (BLIP)."""

    backend: str = "blip"
    model_name: str = BLIP_IMAGE_CAPTIONING_BASE
    preset: Optional[str] = None
    device: str = "auto"
    batch_size: int = 4

    def resolved_model_name(self) -> str:
        """Return the concrete model name to load for captioning.

        Resolution order:
        1. If ``preset`` is set, resolve via :data:`BLIP_PRESETS`.
        2. Otherwise, use ``model_name``.
        3. Fallback to the default BLIP base checkpoint.
        """
        if self.preset:
            preset_name = BLIP_PRESETS.get(self.preset)
            if preset_name is None:
                raise ValueError(f"Unsupported BLIP preset: {self.preset!r}")
            return preset_name

        if self.model_name:
            return self.model_name

        return BLIP_IMAGE_CAPTIONING_BASE


@dataclass
class DetectionModelConfig:
    """Configuration for the optional detection model."""

    enabled: bool = False
    backend: str = "owlvit"
    model_name: str = "google/owlvit-base-patch32"
    device: str = "auto"
    max_regions_per_image: int = 10
    score_threshold: float = 0.25
    siglip_min_prob: float = 0.15
    siglip_min_margin: float = 0.05
    nms_iou_threshold: float = 0.8
    primary_area_gamma: float = 0.3
    primary_center_penalty: float = 0.6
    caption_primary_enabled: bool = True
    caption_primary_min_priority: float = 0.02
    caption_primary_box_margin_x: float = 0.2
    caption_primary_box_margin_y: float = 0.1
    caption_primary_keywords: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    secondary_min_priority: float = 0.03
    secondary_min_relative_to_primary: float = 0.3


_INFLECT_ENGINE: Any | None = None


def _get_inflect_engine() -> Any:
    global _INFLECT_ENGINE
    if _INFLECT_ENGINE is None:
        _INFLECT_ENGINE = inflect.engine()
    return _INFLECT_ENGINE


def _normalize_siglip_label(label: str) -> str:
    """Normalize a label for grouping by case and plurality."""

    text = str(label).strip().lower()
    if not text:
        return text

    engine = _get_inflect_engine()
    singular = engine.singular_noun(text)
    if singular:
        text = singular

    return text


@dataclass
class SiglipLabelConfig:
    """Configuration for SigLIP label and category prompts."""

    label_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "food": [
                "food",
                "tapas",
                "pizza",
                "burger",
                "sushi",
                "noodles",
                "dessert",
                "cake",
            ],
            "electronics": [
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
            ],
            "document": [
                "document",
                "book",
                "notes",
            ],
            "person": [
                "person",
                "people",
            ],
            "scene": [
                "landscape",
                "architecture",
                "building",
            ],
            "animal": [
                "animal",
                "pet",
            ],
            "specific_products": [
                "iPhone",
                "MacBook",
                "iPad",
                "AirPods",
                "Samsung Galaxy",
                "ThinkPad",
                "Surface",
            ],
        }
    )

    @property
    def candidate_labels(self) -> List[str]:
        """Flatten label groups into a unique list of candidate labels."""

        labels: List[str] = []
        seen: set[str] = set()

        for group_labels in self.label_groups.values():
            for label in group_labels:
                text = str(label).strip()
                if not text:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                labels.append(text)

        return labels

    @property
    def simple_detector_categories(self) -> Dict[str, List[str]]:
        """Derive SimpleDetector label sets from label groups.

        - ``general``: pluralized, human-friendly group names + ``Other``.
        - per-group sets: title-cased labels for each group key.
        """

        categories: Dict[str, List[str]] = {}
        general: List[str] = []

        for raw_group_name, group_labels in self.label_groups.items():
            key = str(raw_group_name)
            base_name = key.replace("_", " ")
            general.append(base_name.title())

            fine_labels: List[str] = []
            for label in group_labels:
                fine_labels.append(self._format_label_for_display(str(label)))
            categories[key] = fine_labels

        general.append("Other")
        categories["general"] = general

        return categories

    @staticmethod
    def _format_label_for_display(label: str) -> str:
        text = label.strip()
        if not text:
            return text

        # Preserve labels that already contain uppercase characters (likely brand names).
        if any(ch.isupper() for ch in text):
            return text

        return text.title()


@dataclass
class OcrConfig:
    """Configuration for optional OCR."""

    enabled: bool = False


@dataclass
class ModelsConfig:
    """Bundle of all model-related configuration."""

    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    caption: CaptionModelConfig = field(default_factory=CaptionModelConfig)
    detection: DetectionModelConfig = field(default_factory=DetectionModelConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    siglip_labels: SiglipLabelConfig = field(default_factory=SiglipLabelConfig)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing and inference pipeline."""

    run_detection: bool = False
    skip_duplicates_for_heavy_models: bool = True
    phash_hamming_threshold: int = 12
    thumbnail_size: int = 512
    thumbnail_quality: int = 85
    exif_datetime_format: str = "raw"


@dataclass
class Settings:
    """Top-level application settings."""

    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def load_settings(settings_path: Path | None = None) -> Settings:
    """Load application settings from a YAML file, falling back to defaults.

    The loader is deliberately defensive: if the file is missing or malformed,
    it returns a :class:`Settings` instance populated with default values.
    """
    path = settings_path or Path("config/settings.yaml")
    settings = Settings()

    if not path.exists() or not path.is_file():
        return settings

    raw: Any
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    if not isinstance(raw, dict):
        return settings

    models_raw = _as_dict(raw.get("models"))
    embedding_raw = _as_dict(models_raw.get("embedding"))
    caption_raw = _as_dict(models_raw.get("caption"))
    detection_raw = _as_dict(models_raw.get("detection"))
    ocr_raw = _as_dict(models_raw.get("ocr"))
    siglip_labels_raw = _as_dict(models_raw.get("siglip_labels"))

    embedding_cfg = settings.models.embedding
    if isinstance(embedding_raw.get("backend"), str):
        embedding_cfg.backend = embedding_raw["backend"]
    if isinstance(embedding_raw.get("model_name"), str):
        embedding_cfg.model_name = embedding_raw["model_name"]
    if isinstance(embedding_raw.get("preset"), str):
        embedding_cfg.preset = embedding_raw["preset"]
    if isinstance(embedding_raw.get("device"), str):
        embedding_cfg.device = embedding_raw["device"]
    if isinstance(embedding_raw.get("batch_size"), int):
        embedding_cfg.batch_size = embedding_raw["batch_size"]

    caption_cfg = settings.models.caption
    if isinstance(caption_raw.get("backend"), str):
        caption_cfg.backend = caption_raw["backend"]
    if isinstance(caption_raw.get("model_name"), str):
        caption_cfg.model_name = caption_raw["model_name"]
    if isinstance(caption_raw.get("preset"), str):
        caption_cfg.preset = caption_raw["preset"]
    if isinstance(caption_raw.get("device"), str):
        caption_cfg.device = caption_raw["device"]
    if isinstance(caption_raw.get("batch_size"), int):
        caption_cfg.batch_size = caption_raw["batch_size"]

    detection_cfg = settings.models.detection
    if isinstance(detection_raw.get("enabled"), bool):
        detection_cfg.enabled = detection_raw["enabled"]
    if isinstance(detection_raw.get("backend"), str):
        detection_cfg.backend = detection_raw["backend"]
    if isinstance(detection_raw.get("model_name"), str):
        detection_cfg.model_name = detection_raw["model_name"]
    if isinstance(detection_raw.get("device"), str):
        detection_cfg.device = detection_raw["device"]
    if isinstance(detection_raw.get("max_regions_per_image"), int):
        detection_cfg.max_regions_per_image = detection_raw["max_regions_per_image"]
    if isinstance(detection_raw.get("score_threshold"), (int, float)):
        detection_cfg.score_threshold = float(detection_raw["score_threshold"])
    if isinstance(detection_raw.get("siglip_min_prob"), (int, float)):
        detection_cfg.siglip_min_prob = float(detection_raw["siglip_min_prob"])
    if isinstance(detection_raw.get("siglip_min_margin"), (int, float)):
        detection_cfg.siglip_min_margin = float(detection_raw["siglip_min_margin"])
    if isinstance(detection_raw.get("nms_iou_threshold"), (int, float)):
        detection_cfg.nms_iou_threshold = float(detection_raw["nms_iou_threshold"])
    if isinstance(detection_raw.get("primary_area_gamma"), (int, float)):
        detection_cfg.primary_area_gamma = float(detection_raw["primary_area_gamma"])
    if isinstance(detection_raw.get("primary_center_penalty"), (int, float)):
        detection_cfg.primary_center_penalty = float(detection_raw["primary_center_penalty"])
    if isinstance(detection_raw.get("caption_primary_enabled"), bool):
        detection_cfg.caption_primary_enabled = detection_raw["caption_primary_enabled"]
    if isinstance(detection_raw.get("caption_primary_min_priority"), (int, float)):
        detection_cfg.caption_primary_min_priority = float(detection_raw["caption_primary_min_priority"])
    if isinstance(detection_raw.get("caption_primary_box_margin_x"), (int, float)):
        detection_cfg.caption_primary_box_margin_x = float(detection_raw["caption_primary_box_margin_x"])
    if isinstance(detection_raw.get("caption_primary_box_margin_y"), (int, float)):
        detection_cfg.caption_primary_box_margin_y = float(detection_raw["caption_primary_box_margin_y"])

    caption_keywords_raw = _as_dict(detection_raw.get("caption_primary_keywords"))
    if caption_keywords_raw:
        parsed_keywords: Dict[str, Dict[str, Any]] = {}
        for group_name, group_cfg in caption_keywords_raw.items():
            if not isinstance(group_cfg, dict):
                continue
            entry: Dict[str, Any] = {}
            label_value = group_cfg.get("label")
            if isinstance(label_value, str):
                entry["label"] = label_value
            keywords_value = group_cfg.get("keywords")
            if isinstance(keywords_value, list):
                entry["keywords"] = [str(keyword) for keyword in keywords_value]
            if entry:
                parsed_keywords[str(group_name)] = entry
        if parsed_keywords:
            detection_cfg.caption_primary_keywords = parsed_keywords
    if isinstance(detection_raw.get("secondary_min_priority"), (int, float)):
        detection_cfg.secondary_min_priority = float(detection_raw["secondary_min_priority"])
    if isinstance(detection_raw.get("secondary_min_relative_to_primary"), (int, float)):
        detection_cfg.secondary_min_relative_to_primary = float(
            detection_raw["secondary_min_relative_to_primary"]
        )

    ocr_cfg = settings.models.ocr
    if isinstance(ocr_raw.get("enabled"), bool):
        ocr_cfg.enabled = ocr_raw["enabled"]

    siglip_labels_cfg = settings.models.siglip_labels
    label_groups_raw = _as_dict(siglip_labels_raw.get("label_groups"))
    if label_groups_raw:
        parsed_label_groups: Dict[str, List[str]] = {}
        for group_name, labels in label_groups_raw.items():
            if isinstance(labels, list):
                parsed_label_groups[str(group_name)] = [str(label) for label in labels]
        if parsed_label_groups:
            siglip_labels_cfg.label_groups = parsed_label_groups

    pipeline_raw = _as_dict(raw.get("pipeline"))
    pipeline_cfg = settings.pipeline
    if isinstance(pipeline_raw.get("run_detection"), bool):
        pipeline_cfg.run_detection = pipeline_raw["run_detection"]
    if isinstance(pipeline_raw.get("skip_duplicates_for_heavy_models"), bool):
        pipeline_cfg.skip_duplicates_for_heavy_models = pipeline_raw["skip_duplicates_for_heavy_models"]
    if isinstance(pipeline_raw.get("phash_hamming_threshold"), int):
        pipeline_cfg.phash_hamming_threshold = pipeline_raw["phash_hamming_threshold"]
    if isinstance(pipeline_raw.get("thumbnail_size"), int):
        pipeline_cfg.thumbnail_size = pipeline_raw["thumbnail_size"]
    if isinstance(pipeline_raw.get("thumbnail_quality"), int):
        pipeline_cfg.thumbnail_quality = pipeline_raw["thumbnail_quality"]
    if isinstance(pipeline_raw.get("exif_datetime_format"), str):
        pipeline_cfg.exif_datetime_format = pipeline_raw["exif_datetime_format"]

    return settings


def get_embedding_model_name(settings: Settings | None = None) -> str:
    """Convenience helper to resolve the embedding model name."""

    cfg = (settings or Settings()).models.embedding
    return cfg.resolved_model_name()


def get_caption_model_name(settings: Settings | None = None) -> str:
    """Convenience helper to resolve the captioning model name."""

    cfg = (settings or Settings()).models.caption
    return cfg.resolved_model_name()


__all__ = [
    "EmbeddingModelConfig",
    "CaptionModelConfig",
    "DetectionModelConfig",
    "SiglipLabelConfig",
    "OcrConfig",
    "ModelsConfig",
    "PipelineConfig",
    "Settings",
    "load_settings",
    "get_embedding_model_name",
    "get_caption_model_name",
]
