"""Tests for BLIP caption model loading and generation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

site_packages = (
    Path(__file__).resolve().parents[1]
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
sys.path.append(str(site_packages))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vibe_photos.config import CaptionModelConfig
from vibe_photos.ml import models


def test_blip_caption_uses_seq2seq_and_generates(monkeypatch):
    """Ensure BLIP caption loader wires the Seq2Seq model and generation pipeline."""

    models._BLIP_PROCESSOR = None
    models._BLIP_MODEL = None
    models._BLIP_DEVICE = None
    models._BLIP_MODEL_NAME = None

    captured: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self):
            captured["processor"] = self

        def __call__(self, images, return_tensors="pt"):
            captured["call"] = {"images": images, "return_tensors": return_tensors}
            return {"pixel_values": "pixels"}

        def decode(self, token_ids, skip_special_tokens=False):
            captured["decode"] = {
                "token_ids": token_ids,
                "skip_special_tokens": skip_special_tokens,
            }
            return "stub caption"

    class FakeModel:
        def __init__(self):
            captured["model"] = self
            self.received_device = None

        @classmethod
        def from_pretrained(cls, name):
            captured["model_name"] = name
            return cls()

        def to(self, device):
            self.received_device = device
            return self

        def eval(self):
            captured["eval_called"] = True

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(name, use_fast=True):
            captured["processor_name"] = name
            captured["use_fast"] = use_fast
            return FakeProcessor()

    monkeypatch.setattr(models, "AutoProcessor", FakeProcessorLoader)
    monkeypatch.setattr(models, "AutoModelForSeq2SeqLM", FakeModel)

    config = CaptionModelConfig(model_name="dummy-model", device="cpu")
    processor, model, device = models._load_blip_caption(config)

    assert isinstance(model, FakeModel)
    assert device.type == "cpu"
    assert captured["processor_name"] == "dummy-model"
    assert captured["use_fast"] is True
    assert captured["model_name"] == "dummy-model"
    assert captured["eval_called"] is True
    assert model.received_device.type == "cpu"

    inputs = processor(images="image-bytes", return_tensors="pt")
    generated_ids = model.generate(**inputs)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    assert caption == "stub caption"
    assert captured["generate_kwargs"] == inputs
    assert captured["decode"]["skip_special_tokens"] is True
