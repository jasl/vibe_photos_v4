"""Configuration loader and typed settings for the Vibe Photos application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import inflect
import yaml

from vibe_photos.ml.model_presets import (
    BLIP_IMAGE_CAPTIONING_BASE,
    BLIP_PRESETS,
    SIGLIP2_BASE_PATCH16_224,
    SIGLIP_PRESETS,
)


@dataclass
class EmbeddingModelConfig:
    """Configuration for the embedding model (SigLIP)."""

    backend: str = "siglip"
    model_name: str = SIGLIP2_BASE_PATCH16_224
    preset: str | None = None
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
    preset: str | None = None
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
    nms_iou_threshold: float = 0.8
    primary_area_gamma: float = 0.3
    primary_center_penalty: float = 0.6
    caption_primary_enabled: bool = True
    caption_primary_min_priority: float = 0.02
    caption_primary_box_margin_x: float = 0.2
    caption_primary_box_margin_y: float = 0.1
    caption_primary_keywords: dict[str, dict[str, Any]] = field(default_factory=dict)
    caption_primary_use_label_groups: bool = True
    secondary_min_priority: float = 0.03
    secondary_min_relative_to_primary: float = 0.3


_INFLECT_ENGINE: Any | None = None


def _project_root() -> Path:
    """Best-effort detection of the repository root for config discovery."""

    module_path = Path(__file__).resolve()
    try:
        return module_path.parents[2]
    except IndexError:  # pragma: no cover - defensive fallback
        return module_path.parent


def _build_default_settings_paths() -> list[Path]:
    """Return candidate settings paths ordered by preference."""

    cwd_candidate = (Path.cwd() / "config" / "settings.yaml").resolve()
    repo_candidate = (_project_root() / "config" / "settings.yaml").resolve()

    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in (cwd_candidate, repo_candidate):
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


_DEFAULT_SETTINGS_PATHS = _build_default_settings_paths()


def _resolve_settings_path(settings_path: Path | str | None) -> Path:
    """Determine which settings file to load, honoring overrides."""

    if settings_path:
        return Path(settings_path).expanduser().resolve()

    env_override = os.getenv("VIBE_PHOTOS_SETTINGS")
    if env_override:
        return Path(env_override).expanduser().resolve()

    for candidate in _DEFAULT_SETTINGS_PATHS:
        if candidate.exists():
            return candidate
    return _DEFAULT_SETTINGS_PATHS[0]


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


def _derive_caption_keywords_from_label_groups(label_groups: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Build caption fallback keywords from SigLIP label groups.

    Each label becomes its own keyword-driven entry so captions can fall back to the
    most specific token that SigLIP already knows about.
    """

    derived: dict[str, dict[str, Any]] = {}
    seen_labels: set[str] = set()

    for labels in label_groups.values():
        for label in labels:
            text = str(label).strip()
            if not text:
                continue
            normalized = text.lower()
            if normalized in seen_labels:
                continue
            seen_labels.add(normalized)
            derived[text] = {"label": text, "keywords": [normalized]}

    return derived


@dataclass
class SiglipLabelConfig:
    """Configuration for SigLIP label and category prompts."""

    label_groups: dict[str, list[str]] = field(
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
    def candidate_labels(self) -> list[str]:
        """Flatten label groups into a unique list of candidate labels."""

        labels: list[str] = []
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
    def simple_detector_categories(self) -> dict[str, list[str]]:
        """Derive SimpleDetector label sets from label groups.

        - ``general``: pluralized, human-friendly group names + ``Other``.
        - per-group sets: title-cased labels for each group key.
        """

        categories: dict[str, list[str]] = {}
        general: list[str] = []

        for raw_group_name, group_labels in self.label_groups.items():
            key = str(raw_group_name)
            base_name = key.replace("_", " ")
            general.append(base_name.title())

            fine_labels: list[str] = []
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
class ObjectZeroShotConfig:
    """Thresholds and knobs for region-based zero-shot object labeling."""

    top_k: int = 5
    score_min: float = 0.32
    margin_min: float = 0.08
    scene_whitelist: list[str] = field(default_factory=lambda: ["scene.product", "scene.food"])
    scene_fallback_labels: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ObjectAggregationConfig:
    """Aggregation from regions to image-level object labels."""

    min_regions: int = 1
    score_min: float = 0.32


@dataclass
class ObjectConfig:
    """Top-level object labeling config bundle."""

    zero_shot: ObjectZeroShotConfig = field(default_factory=ObjectZeroShotConfig)
    aggregation: ObjectAggregationConfig = field(default_factory=ObjectAggregationConfig)
    blacklist: list[str] = field(default_factory=list)
    remap: dict[str, str] = field(default_factory=dict)


@dataclass
class ClusterLevelConfig:
    """Clustering hyperparameters for a specific target type."""

    k: int = 20
    sim_threshold: float = 0.78
    min_size: int = 3


@dataclass
class ClusterConfig:
    """Configuration bundle for image/region clustering."""

    image: ClusterLevelConfig = field(default_factory=ClusterLevelConfig)
    region: ClusterLevelConfig = field(default_factory=ClusterLevelConfig)


@dataclass
class DatabaseConfig:
    """Database connection targets for primary and cache stores."""

    primary_url: str = "sqlite:///data/index.db"
    cache_url: str = "sqlite:///cache/index.db"


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
    run_cluster: bool = False
    skip_duplicates_for_heavy_models: bool = True
    phash_hamming_threshold: int = 12
    thumbnail_size_small: int = 256
    thumbnail_size_large: int = 1024
    thumbnail_quality: int = 85
    exif_datetime_format: str = "raw"


@dataclass
class QueueConfig:
    """Celery worker and queue configuration."""

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    preprocess_queue: str = "pre_process"
    main_queue: str = "process"
    post_process_queue: str = "post_process"
    default_concurrency: int = 2
    post_process_concurrency: int = 1
    backfill_batch_size: int = 128


@dataclass
class MainProcessingConfig:
    """Configurable thresholds for cache-first classification and clustering."""

    classification_threshold: float = 0.35
    caption_confidence_threshold: float = 0.2
    region_min_score: float = 0.25
    cluster_phash_distance: int = 8
    search_index_shard_size: int = 5000


@dataclass
class PostProcessConfig:
    """Optional post-process analyses such as OCR or cloud models."""

    enable_ocr: bool = False
    enable_cloud_models: bool = False
    queue_name: str = "post_process"
    max_concurrency: int = 1


@dataclass
class LabelSpacesConfig:
    """Active label space versions for scene/object/cluster labels."""

    scene_current: str = "scene_v1"
    object_current: str = "object_v1"
    cluster_current: str = "cluster_v1"


@dataclass
class Settings:
    """Top-level application settings."""

    databases: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    main_processing: MainProcessingConfig = field(default_factory=MainProcessingConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    label_spaces: LabelSpacesConfig = field(default_factory=LabelSpacesConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def load_settings(settings_path: Path | None = None) -> Settings:
    """Load application settings from a YAML file, falling back to defaults.

    The loader is deliberately defensive: if the file is missing or malformed,
    it returns a :class:`Settings` instance populated with default values.
    """
    path = _resolve_settings_path(settings_path)
    settings = Settings()

    if not path.exists() or not path.is_file():
        return settings

    raw: Any
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    if not isinstance(raw, dict):
        return settings

    databases_raw = _as_dict(raw.get("databases"))
    db_cfg = settings.databases
    if isinstance(databases_raw.get("primary_url"), str):
        db_cfg.primary_url = databases_raw["primary_url"]
    cache_value = databases_raw.get("cache_url")
    if isinstance(cache_value, str):
        db_cfg.cache_url = cache_value

    models_raw = _as_dict(raw.get("models"))
    embedding_raw = _as_dict(models_raw.get("embedding"))
    caption_raw = _as_dict(models_raw.get("caption"))
    detection_raw = _as_dict(models_raw.get("detection"))
    ocr_raw = _as_dict(models_raw.get("ocr"))
    siglip_labels_raw = _as_dict(models_raw.get("siglip_labels"))
    label_spaces_raw = _as_dict(raw.get("label_spaces"))
    object_raw = _as_dict(raw.get("object"))

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
    if isinstance(detection_raw.get("caption_primary_use_label_groups"), bool):
        detection_cfg.caption_primary_use_label_groups = detection_raw["caption_primary_use_label_groups"]

    caption_keywords_raw = _as_dict(detection_raw.get("caption_primary_keywords"))
    if caption_keywords_raw:
        parsed_keywords: dict[str, dict[str, Any]] = {}
        for group_name, group_cfg in caption_keywords_raw.items():
            if not isinstance(group_cfg, dict):
                continue
            entry: dict[str, Any] = {}
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
        parsed_label_groups: dict[str, list[str]] = {}
        for group_name, labels in label_groups_raw.items():
            if isinstance(labels, list):
                parsed_label_groups[str(group_name)] = [str(label) for label in labels]
        if parsed_label_groups:
            siglip_labels_cfg.label_groups = parsed_label_groups

    if detection_cfg.caption_primary_use_label_groups and siglip_labels_cfg.label_groups:
        derived_caption_keywords = _derive_caption_keywords_from_label_groups(siglip_labels_cfg.label_groups)
        merged_caption_keywords = {**derived_caption_keywords, **detection_cfg.caption_primary_keywords}
        detection_cfg.caption_primary_keywords = merged_caption_keywords

    pipeline_raw = _as_dict(raw.get("pipeline"))
    pipeline_cfg = settings.pipeline
    if isinstance(pipeline_raw.get("run_detection"), bool):
        pipeline_cfg.run_detection = pipeline_raw["run_detection"]
    if isinstance(pipeline_raw.get("run_cluster"), bool):
        pipeline_cfg.run_cluster = pipeline_raw["run_cluster"]
    if isinstance(pipeline_raw.get("skip_duplicates_for_heavy_models"), bool):
        pipeline_cfg.skip_duplicates_for_heavy_models = pipeline_raw["skip_duplicates_for_heavy_models"]
    if isinstance(pipeline_raw.get("phash_hamming_threshold"), int):
        pipeline_cfg.phash_hamming_threshold = pipeline_raw["phash_hamming_threshold"]
    if isinstance(pipeline_raw.get("thumbnail_size_small"), int):
        pipeline_cfg.thumbnail_size_small = pipeline_raw["thumbnail_size_small"]
    if isinstance(pipeline_raw.get("thumbnail_size_large"), int):
        pipeline_cfg.thumbnail_size_large = pipeline_raw["thumbnail_size_large"]
    elif isinstance(pipeline_raw.get("thumbnail_size"), int):
        legacy_size = pipeline_raw["thumbnail_size"]
        pipeline_cfg.thumbnail_size_small = min(256, legacy_size)
        pipeline_cfg.thumbnail_size_large = max(1024, legacy_size)
    if isinstance(pipeline_raw.get("thumbnail_quality"), int):
        pipeline_cfg.thumbnail_quality = pipeline_raw["thumbnail_quality"]
    if isinstance(pipeline_raw.get("exif_datetime_format"), str):
        pipeline_cfg.exif_datetime_format = pipeline_raw["exif_datetime_format"]

    queue_raw = _as_dict(raw.get("queues"))
    queue_cfg = settings.queues
    if isinstance(queue_raw.get("broker_url"), str):
        queue_cfg.broker_url = queue_raw["broker_url"]
    if isinstance(queue_raw.get("result_backend"), str):
        queue_cfg.result_backend = queue_raw["result_backend"]
    if isinstance(queue_raw.get("preprocess_queue"), str):
        queue_cfg.preprocess_queue = queue_raw["preprocess_queue"]
    if isinstance(queue_raw.get("main_queue"), str):
        queue_cfg.main_queue = queue_raw["main_queue"]
    if isinstance(queue_raw.get("post_process_queue"), str):
        queue_cfg.post_process_queue = queue_raw["post_process_queue"]
    if isinstance(queue_raw.get("default_concurrency"), int):
        queue_cfg.default_concurrency = queue_raw["default_concurrency"]
    if isinstance(queue_raw.get("post_process_concurrency"), int):
        queue_cfg.post_process_concurrency = queue_raw["post_process_concurrency"]
    if isinstance(queue_raw.get("backfill_batch_size"), int):
        queue_cfg.backfill_batch_size = queue_raw["backfill_batch_size"]

    main_stage_raw = _as_dict(raw.get("main_processing"))
    main_stage_cfg = settings.main_processing
    if isinstance(main_stage_raw.get("classification_threshold"), (int, float)):
        main_stage_cfg.classification_threshold = float(main_stage_raw["classification_threshold"])
    if isinstance(main_stage_raw.get("caption_confidence_threshold"), (int, float)):
        main_stage_cfg.caption_confidence_threshold = float(
            main_stage_raw["caption_confidence_threshold"]
        )
    if isinstance(main_stage_raw.get("region_min_score"), (int, float)):
        main_stage_cfg.region_min_score = float(main_stage_raw["region_min_score"])
    if isinstance(main_stage_raw.get("cluster_phash_distance"), int):
        main_stage_cfg.cluster_phash_distance = main_stage_raw["cluster_phash_distance"]
    if isinstance(main_stage_raw.get("search_index_shard_size"), int):
        main_stage_cfg.search_index_shard_size = main_stage_raw["search_index_shard_size"]

    post_process_raw = _as_dict(raw.get("post_process"))
    post_process_cfg = settings.post_process
    if isinstance(post_process_raw.get("enable_ocr"), bool):
        post_process_cfg.enable_ocr = post_process_raw["enable_ocr"]
    if isinstance(post_process_raw.get("enable_cloud_models"), bool):
        post_process_cfg.enable_cloud_models = post_process_raw["enable_cloud_models"]
    if isinstance(post_process_raw.get("queue_name"), str):
        post_process_cfg.queue_name = post_process_raw["queue_name"]
    if isinstance(post_process_raw.get("max_concurrency"), int):
        post_process_cfg.max_concurrency = post_process_raw["max_concurrency"]

    label_spaces_cfg = settings.label_spaces
    if isinstance(label_spaces_raw.get("scene_current"), str):
        label_spaces_cfg.scene_current = label_spaces_raw["scene_current"]
    if isinstance(label_spaces_raw.get("object_current"), str):
        label_spaces_cfg.object_current = label_spaces_raw["object_current"]
    if isinstance(label_spaces_raw.get("cluster_current"), str):
        label_spaces_cfg.cluster_current = label_spaces_raw["cluster_current"]

    object_cfg = settings.object
    zero_shot_raw = _as_dict(object_raw.get("zero_shot"))
    aggregation_raw = _as_dict(object_raw.get("aggregation"))
    if isinstance(zero_shot_raw.get("top_k"), int):
        object_cfg.zero_shot.top_k = zero_shot_raw["top_k"]
    if isinstance(zero_shot_raw.get("score_min"), (int, float)):
        object_cfg.zero_shot.score_min = float(zero_shot_raw["score_min"])
    if isinstance(zero_shot_raw.get("margin_min"), (int, float)):
        object_cfg.zero_shot.margin_min = float(zero_shot_raw["margin_min"])
    if isinstance(zero_shot_raw.get("scene_whitelist"), list):
        object_cfg.zero_shot.scene_whitelist = [str(item) for item in zero_shot_raw["scene_whitelist"] if str(item)]
    scene_fallback_raw = _as_dict(zero_shot_raw.get("scene_fallback_labels"))
    if scene_fallback_raw:
        parsed: dict[str, list[str]] = {}
        for scene_key, label_list in scene_fallback_raw.items():
            if not isinstance(label_list, list):
                continue
            parsed[str(scene_key)] = [str(label) for label in label_list if str(label)]
        if parsed:
            object_cfg.zero_shot.scene_fallback_labels = parsed

    blacklist_raw = object_raw.get("blacklist")
    if isinstance(blacklist_raw, list):
        object_cfg.blacklist = [str(item) for item in blacklist_raw if str(item)]

    remap_raw = _as_dict(object_raw.get("remap"))
    if remap_raw:
        object_cfg.remap = {str(src): str(dst) for src, dst in remap_raw.items() if str(dst)}

    if isinstance(aggregation_raw.get("min_regions"), int):
        object_cfg.aggregation.min_regions = aggregation_raw["min_regions"]
    if isinstance(aggregation_raw.get("score_min"), (int, float)):
        object_cfg.aggregation.score_min = float(aggregation_raw["score_min"])

    cluster_raw = _as_dict(raw.get("cluster"))
    cluster_cfg = settings.cluster
    image_cluster_raw = _as_dict(cluster_raw.get("image"))
    region_cluster_raw = _as_dict(cluster_raw.get("region"))
    if isinstance(image_cluster_raw.get("k"), int):
        cluster_cfg.image.k = image_cluster_raw["k"]
    if isinstance(image_cluster_raw.get("sim_threshold"), (int, float)):
        cluster_cfg.image.sim_threshold = float(image_cluster_raw["sim_threshold"])
    if isinstance(image_cluster_raw.get("min_size"), int):
        cluster_cfg.image.min_size = image_cluster_raw["min_size"]
    if isinstance(region_cluster_raw.get("k"), int):
        cluster_cfg.region.k = region_cluster_raw["k"]
    if isinstance(region_cluster_raw.get("sim_threshold"), (int, float)):
        cluster_cfg.region.sim_threshold = float(region_cluster_raw["sim_threshold"])
    if isinstance(region_cluster_raw.get("min_size"), int):
        cluster_cfg.region.min_size = region_cluster_raw["min_size"]

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
    "DatabaseConfig",
    "EmbeddingModelConfig",
    "CaptionModelConfig",
    "DetectionModelConfig",
    "SiglipLabelConfig",
    "OcrConfig",
    "ModelsConfig",
    "PipelineConfig",
    "QueueConfig",
    "MainProcessingConfig",
    "PostProcessConfig",
    "Settings",
    "load_settings",
    "get_embedding_model_name",
    "get_caption_model_name",
]
