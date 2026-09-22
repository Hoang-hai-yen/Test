"""Unit tests for FGCLIPFeatureExtractor, RadioFeatureExtractor, and their
build_feature_extractor() factory dispatch -- see FeatureExtractorConfig.
model's own docstring (aero_eyes/config.py). Monkeypatches the network-
loading calls (transformers/torch.hub) so this runs without downloading any
real model weights (same convention as test_features_ensemble.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aero_eyes.config import AeroEyesConfig
from aero_eyes.models import features as features_mod


# ---------------------------------------------------------------------------
# FG-CLIP
# ---------------------------------------------------------------------------

class _FakeCausalLMForImages:
    """Stand-in for FG-CLIP's AutoModelForCausalLM -- exposes just the
    get_image_features() surface FGCLIPFeatureExtractor.extract() calls."""

    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_image_features(self, pixel_values):
        return torch.ones(pixel_values.shape[0], self._dim)


class _FakeImageProcessor:
    def preprocess(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


def test_fgclip_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown FG-CLIP variant"):
        features_mod.FGCLIPFeatureExtractor(variant="huge")


def test_fgclip_base_dim_and_normalized_output(monkeypatch):
    monkeypatch.setattr(
        features_mod.FGCLIPFeatureExtractor, "_load",
        lambda self, variant: (_FakeCausalLMForImages(512), _FakeImageProcessor()),
    )
    ext = features_mod.FGCLIPFeatureExtractor(variant="base")
    assert ext._dim() == 512
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 512)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_fgclip_large_dim():
    assert features_mod.FGCLIPFeatureExtractor._DIMS["large"] == 768


def test_fgclip_extract_empty_returns_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(
        features_mod.FGCLIPFeatureExtractor, "_load",
        lambda self, variant: (_FakeCausalLMForImages(512), _FakeImageProcessor()),
    )
    ext = features_mod.FGCLIPFeatureExtractor(variant="base")
    out = ext.extract([])
    assert out.shape == (0, 512)


# ---------------------------------------------------------------------------
# RADIO / C-RADIO
# ---------------------------------------------------------------------------

class _FakeRadioModel:
    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_nearest_supported_resolution(self, h, w):
        return (h, w)

    def __call__(self, batch):
        return torch.ones(batch.shape[0], self._dim), None


def test_radio_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown RADIO variant"):
        features_mod.RadioFeatureExtractor(variant="c-radio_v99-x")


def test_radio_warns_on_non_commercial_variant(monkeypatch, caplog):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeRadioModel(768))
    with caplog.at_level("WARNING"):
        ext = features_mod.RadioFeatureExtractor(variant="radio-b")
    assert "non-commercial" in caplog.text
    assert ext._dim() == 768


def test_radio_c_variant_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeRadioModel(768))
    with caplog.at_level("WARNING"):
        features_mod.RadioFeatureExtractor(variant="c-radio_v3-b")
    assert "non-commercial" not in caplog.text


def test_radio_extract_normalizes_and_probes_dim(monkeypatch):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeRadioModel(768))
    ext = features_mod.RadioFeatureExtractor(variant="c-radio_v3-b")
    assert ext._dim() == 768
    out = ext.extract([np.zeros((32, 32, 3), dtype=np.uint8)])
    assert out.shape == (1, 768)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_radio_extract_empty_returns_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeRadioModel(768))
    ext = features_mod.RadioFeatureExtractor(variant="c-radio_v3-b")
    out = ext.extract([])
    assert out.shape == (0, 768)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatches_to_fgclip(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "FGCLIPFeatureExtractor",
        lambda *a, **k: calls.append((a, k)) or _FakeCausalLMForImages(512),
    )
    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "fgclip"
    cfg.stage1.feature_extractor.fgclip_variant = "large"

    features_mod.build_feature_extractor(cfg)

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["variant"] == "large"


def test_factory_dispatches_to_radio(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "RadioFeatureExtractor",
        lambda *a, **k: calls.append((a, k)) or _FakeRadioModel(768),
    )
    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "radio"
    cfg.stage1.feature_extractor.radio_variant = "c-radio_v4-h"

    features_mod.build_feature_extractor(cfg)

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["variant"] == "c-radio_v4-h"
    assert kwargs["image_size"] == cfg.stage1.feature_extractor.image_size
