"""Canonical model identifiers and presets used by the ML stack.

These constants centralize the Hugging Face model names and provide
named presets so that configuration files do not need to repeat raw
checkpoint identifiers everywhere.
"""

from __future__ import annotations

from typing import Dict

SIGLIP2_BASE_PATCH16_224 = "google/siglip2-base-patch16-224"
SIGLIP2_LARGE_PATCH16_384 = "google/siglip2-large-patch16-384"

BLIP_IMAGE_CAPTIONING_BASE = "Salesforce/blip-image-captioning-base"
BLIP_IMAGE_CAPTIONING_LARGE = "Salesforce/blip-image-captioning-large"

SIGLIP_PRESETS: Dict[str, str] = {
    # Default embedding model for M1.
    "m1_default": SIGLIP2_BASE_PATCH16_224,
    # Higher-quality, higher-cost SigLIP2 variant for capable hardware.
    "hq_384": SIGLIP2_LARGE_PATCH16_384,
}

BLIP_PRESETS: Dict[str, str] = {
    # Default captioning model for M1.
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

