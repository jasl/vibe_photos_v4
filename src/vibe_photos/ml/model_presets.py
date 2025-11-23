"""Canonical model identifiers and presets used by the ML stack.

These constants centralize the Hugging Face model names and provide
named presets so that configuration files do not need to repeat raw
checkpoint identifiers everywhere.
"""

from __future__ import annotations

SIGLIP2_BASE_PATCH16_224 = "google/siglip2-base-patch16-224"
SIGLIP2_LARGE_PATCH16_384 = "google/siglip2-large-patch16-384"

BLIP_IMAGE_CAPTIONING_BASE = "Salesforce/blip-image-captioning-base"
BLIP_IMAGE_CAPTIONING_LARGE = "Salesforce/blip-image-captioning-large"

SIGLIP_PRESETS: dict[str, str] = {
    # Default embedding model for the current pipeline.
    "m1_default": SIGLIP2_BASE_PATCH16_224,
    # Higher-quality, higher-cost SigLIP2 variant for capable hardware.
    "hq_384": SIGLIP2_LARGE_PATCH16_384,
}

BLIP_PRESETS: dict[str, str] = {
    # Default captioning model for the current pipeline.
    "caption_base": BLIP_IMAGE_CAPTIONING_BASE,
    # Optional larger captioning model when a GPU is available.
    "caption_large": BLIP_IMAGE_CAPTIONING_LARGE,
}

__all__ = [
    "SIGLIP2_BASE_PATCH16_224",
    "SIGLIP2_LARGE_PATCH16_384",
    "BLIP_IMAGE_CAPTIONING_BASE",
    "BLIP_IMAGE_CAPTIONING_LARGE",
    "SIGLIP_PRESETS",
    "BLIP_PRESETS",
]
