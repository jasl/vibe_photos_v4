"""Shared model loaders and singletons for SigLIP and BLIP.

These helpers ensure that heavy models are loaded once per process and
configured according to :mod:`vibe_photos.config` settings.
"""

from __future__ import annotations

import torch
from torch import device as TorchDevice
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, PreTrainedModel

from vibe_photos.config import CaptionModelConfig, EmbeddingModelConfig, Settings, load_settings


def _select_device(config_device: str = "auto") -> TorchDevice:
    """Select a torch device based on configuration, preferring CPU-safe fallbacks.

    The resolution strategy is:

    - ``auto``: CUDA → MPS → CPU.
    - Explicit values (``cuda``, ``mps``, ``cpu``): use when available, otherwise fall back to CPU.
    """

    normalized = (config_device or "auto").lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")

    # Fallback: unknown or unavailable device, use CPU to remain robust on all machines.
    return torch.device("cpu")


_SIGLIP_PROCESSOR: AutoProcessor | None = None
_SIGLIP_MODEL: PreTrainedModel | None = None
_SIGLIP_DEVICE: TorchDevice | None = None
_SIGLIP_MODEL_NAME: str | None = None

_BLIP_PROCESSOR: AutoProcessor | None = None
_BLIP_MODEL: PreTrainedModel | None = None
_BLIP_DEVICE: TorchDevice | None = None
_BLIP_MODEL_NAME: str | None = None


def _load_siglip(config: EmbeddingModelConfig) -> tuple[AutoProcessor, PreTrainedModel, TorchDevice]:
    model_name = config.resolved_model_name()
    device = _select_device(config.device)

    global _SIGLIP_PROCESSOR, _SIGLIP_MODEL, _SIGLIP_DEVICE, _SIGLIP_MODEL_NAME
    if (
        _SIGLIP_PROCESSOR is not None
        and _SIGLIP_MODEL is not None
        and _SIGLIP_DEVICE is not None
        and _SIGLIP_MODEL_NAME == model_name
        and _SIGLIP_DEVICE == device
    ):
        return _SIGLIP_PROCESSOR, _SIGLIP_MODEL, _SIGLIP_DEVICE

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    _SIGLIP_PROCESSOR = processor
    _SIGLIP_MODEL = model
    _SIGLIP_DEVICE = device
    _SIGLIP_MODEL_NAME = model_name

    return processor, model, device


def _load_blip_caption(config: CaptionModelConfig) -> tuple[AutoProcessor, PreTrainedModel, TorchDevice]:
    model_name = config.resolved_model_name()
    device = _select_device(config.device)

    global _BLIP_PROCESSOR, _BLIP_MODEL, _BLIP_DEVICE, _BLIP_MODEL_NAME
    if (
        _BLIP_PROCESSOR is not None
        and _BLIP_MODEL is not None
        and _BLIP_DEVICE is not None
        and _BLIP_MODEL_NAME == model_name
        and _BLIP_DEVICE == device
    ):
        return _BLIP_PROCESSOR, _BLIP_MODEL, _BLIP_DEVICE

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)
    model.eval()

    _BLIP_PROCESSOR = processor
    _BLIP_MODEL = model
    _BLIP_DEVICE = device
    _BLIP_MODEL_NAME = model_name

    return processor, model, device


def get_siglip_embedding_model(settings: Settings | None = None) -> tuple[AutoProcessor, PreTrainedModel, TorchDevice]:
    """Return the shared SigLIP processor, model, and device for embeddings.

    The ``settings`` argument is optional; when omitted, configuration is
    loaded from ``config/settings.yaml`` using :func:`load_settings`.
    """

    cfg = (settings or load_settings()).models.embedding
    return _load_siglip(cfg)


def get_blip_caption_model(
    settings: Settings | None = None,
) -> tuple[AutoProcessor, PreTrainedModel, TorchDevice]:
    """Return the shared BLIP processor, model, and device for captioning."""

    cfg = (settings or load_settings()).models.caption
    return _load_blip_caption(cfg)


__all__ = [
    "get_siglip_embedding_model",
    "get_blip_caption_model",
]
