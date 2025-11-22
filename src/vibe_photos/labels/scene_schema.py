"""Mappings between classifier outputs and label layer keys for scenes/attributes."""

from __future__ import annotations

from typing import Dict

# Scene classifier emits uppercase coarse categories. Map them to label keys.
SCENE_KEY_BY_TYPE: Dict[str, str] = {
    "LANDSCAPE": "scene.landscape",
    "SNAPSHOT": "scene.snapshot",
    "PEOPLE": "scene.people",
    "FOOD": "scene.food",
    "PRODUCT": "scene.product",
    "DOCUMENT": "scene.document",
    "SCREENSHOT": "scene.screenshot",
    "OTHER": "scene.other",
}

# Attribute identifiers → label keys.
ATTRIBUTE_LABEL_KEYS: Dict[str, str] = {
    "has_person": "attr.has_person",
    "has_text": "attr.has_text",
    "is_document": "attr.is_document",
    "is_screenshot": "attr.is_screenshot",
}

SCENE_TYPE_BY_KEY: Dict[str, str] = {value: key for key, value in SCENE_KEY_BY_TYPE.items()}
ALL_SCENE_LABEL_KEYS = tuple(SCENE_TYPE_BY_KEY.keys())
ALL_ATTRIBUTE_LABEL_KEYS = tuple(ATTRIBUTE_LABEL_KEYS.values())


def scene_label_key_from_type(scene_type: str | None) -> str:
    """Return the label key for a classifier scene type, defaulting to OTHER."""

    if not scene_type:
        return SCENE_KEY_BY_TYPE["OTHER"]

    normalized = scene_type.strip().upper()
    return SCENE_KEY_BY_TYPE.get(normalized, SCENE_KEY_BY_TYPE["OTHER"])


def scene_type_from_label_key(label_key: str | None) -> str | None:
    """Convert a label key back to a coarse scene type (uppercase)."""

    if not label_key:
        return None
    return SCENE_TYPE_BY_KEY.get(label_key)


def normalize_scene_filter(value: str | None) -> str | None:
    """Normalize user-provided scene filter (type or label key) to a label key."""

    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    if text.startswith("scene."):
        return text

    return scene_label_key_from_type(text)


__all__ = [
    "SCENE_KEY_BY_TYPE",
    "SCENE_TYPE_BY_KEY",
    "ALL_SCENE_LABEL_KEYS",
    "ALL_ATTRIBUTE_LABEL_KEYS",
    "ATTRIBUTE_LABEL_KEYS",
    "scene_label_key_from_type",
    "scene_type_from_label_key",
    "normalize_scene_filter",
]


