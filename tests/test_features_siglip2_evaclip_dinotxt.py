"""Unit tests for Siglip2FeatureExtractor, EVACLIPFeatureExtractor,
DinoTxtFeatureExtractor, and their build_feature_extractor() factory
dispatch -- see FeatureExtractorConfig.model's own docstring
(aero_eyes/config.py). Monkeypatches the network-loading calls
(transformers/open_clip/torch.hub) so this runs without downloading any
real model weights (same convention as test_features_fgclip_radio.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aero_eyes.config import AeroEyesConfig
from aero_eyes.models import features as features_mod


# ---------------------------------------------------------------------------
# SigLIP2
# ---------------------------------------------------------------------------

class _FakeSiglip2Model:
    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_image_features(self, **inputs):
        pixel_values = inputs["pixel_values"]
        return torch.ones(pixel_values.shape[0], self._dim)


class _FakeSiglip2Processor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


def test_siglip2_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown SigLIP2 variant"):
        features_mod.Siglip2FeatureExtractor(variant="giant")


def test_siglip2_extract_normalizes_and_probes_dim(monkeypatch):
    monkeypatch.setattr(
        features_mod.Siglip2FeatureExtractor, "_load",
        lambda self, variant: (_FakeSiglip2Model(768), _FakeSiglip2Processor()),
    )
    ext = features_mod.Siglip2FeatureExtractor(variant="base")
    assert ext._dim() == 768
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 768)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_siglip2_extract_empty_returns_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(
        features_mod.Siglip2FeatureExtractor, "_load",
        lambda self, variant: (_FakeSiglip2Model(1152), _FakeSiglip2Processor()),
    )
    ext = features_mod.Siglip2FeatureExtractor(variant="so400m")
    out = ext.extract([])
    assert out.shape == (0, 1152)


class _FakeModelOutputWithPooling:
    """Stand-in for transformers.modeling_outputs.BaseModelOutputWithPooling
    -- confirmed in practice (not just theoretical) that at least one real
    transformers/checkpoint combination makes model.get_image_features()
    return this instead of the documented projected embedding Tensor."""
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output
        self.last_hidden_state = pooler_output


class _FakeSiglip2ModelReturningModelOutput:
    """get_image_features() returns a BaseModelOutputWithPooling-like
    object instead of a plain Tensor -- the real-world failure mode
    Siglip2FeatureExtractor._get_image_features's own fallback handles."""
    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_image_features(self, **inputs):
        pixel_values = inputs["pixel_values"]
        return _FakeModelOutputWithPooling(torch.ones(pixel_values.shape[0], self._dim))


class _FakeSiglip2ModelReturningUnusableOutput:
    """get_image_features() returns something with neither .shape nor
    .pooler_output -- must raise a clear, actionable TypeError instead of
    an opaque AttributeError deep inside numpy/torch."""
    def eval(self):
        return self

    def to(self, device):
        return self

    def get_image_features(self, **inputs):
        return object()


def test_siglip2_falls_back_to_pooler_output_when_get_image_features_returns_model_output(monkeypatch):
    """Regression test for the real bug: on the affected transformers
    version, get_image_features() returned BaseModelOutputWithPooling
    (raising 'BaseModelOutputWithPooling has no attribute shape' at
    feats.shape[-1]) instead of a Tensor. The fallback must transparently
    recover the per-image embedding from .pooler_output."""
    monkeypatch.setattr(
        features_mod.Siglip2FeatureExtractor, "_load",
        lambda self, variant: (_FakeSiglip2ModelReturningModelOutput(768), _FakeSiglip2Processor()),
    )
    ext = features_mod.Siglip2FeatureExtractor(variant="base")
    assert ext._dim() == 768
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 768)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_siglip2_raises_clear_error_when_get_image_features_output_is_unusable(monkeypatch):
    monkeypatch.setattr(
        features_mod.Siglip2FeatureExtractor, "_load",
        lambda self, variant: (_FakeSiglip2ModelReturningUnusableOutput(), _FakeSiglip2Processor()),
    )
    with pytest.raises(TypeError, match="get_image_features\\(\\) returned"):
        features_mod.Siglip2FeatureExtractor(variant="base")


# ---------------------------------------------------------------------------
# EVA02-CLIP
# ---------------------------------------------------------------------------

class _FakeEvaClipModel:
    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def encode_image(self, batch):
        return torch.ones(batch.shape[0], self._dim)


def _fake_preprocess(pil_img):
    return torch.zeros(3, 224, 224)


def test_evaclip_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown EVA-CLIP variant"):
        features_mod.EVACLIPFeatureExtractor(variant="large")


def test_evaclip_extract_normalizes_and_probes_dim(monkeypatch):
    monkeypatch.setattr(
        features_mod.EVACLIPFeatureExtractor, "_load",
        lambda self: (_FakeEvaClipModel(512), _fake_preprocess),
    )
    ext = features_mod.EVACLIPFeatureExtractor(variant="base")
    assert ext._dim() == 512
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 512)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_evaclip_extract_empty_returns_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(
        features_mod.EVACLIPFeatureExtractor, "_load",
        lambda self: (_FakeEvaClipModel(512), _fake_preprocess),
    )
    ext = features_mod.EVACLIPFeatureExtractor(variant="base")
    out = ext.extract([])
    assert out.shape == (0, 512)


def test_evaclip_load_raises_clear_error_without_open_clip(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "open_clip":
            raise ImportError("simulated: open_clip_torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="open_clip_torch not installed"):
        features_mod.EVACLIPFeatureExtractor(variant="base")


# ---------------------------------------------------------------------------
# dino.txt
# ---------------------------------------------------------------------------

class _FakeDinoTxtModel:
    def __init__(self, dim: int):
        self._dim = dim

    def eval(self):
        return self

    def to(self, device):
        return self

    def encode_image(self, batch, normalize=True):
        n = batch.shape[0]
        feats = torch.ones(n, self._dim)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def test_dinotxt_handles_bare_model_return(monkeypatch):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeDinoTxtModel(1024))
    ext = features_mod.DinoTxtFeatureExtractor()
    assert ext._dim() == 1024
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 1024)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_dinotxt_handles_tuple_return(monkeypatch):
    monkeypatch.setattr(
        features_mod.torch.hub, "load",
        lambda *a, **k: (_FakeDinoTxtModel(1024), object()),  # (model, tokenizer)
    )
    ext = features_mod.DinoTxtFeatureExtractor()
    assert ext._dim() == 1024


def test_dinotxt_extract_empty_returns_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(features_mod.torch.hub, "load", lambda *a, **k: _FakeDinoTxtModel(1024))
    ext = features_mod.DinoTxtFeatureExtractor()
    out = ext.extract([])
    assert out.shape == (0, 1024)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatches_to_siglip2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "Siglip2FeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeSiglip2Model(768),
    )
    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "siglip2"
    cfg.stage1.feature_extractor.siglip2_variant = "so400m"

    features_mod.build_feature_extractor(cfg)

    assert len(calls) == 1
    assert calls[0]["variant"] == "so400m"


def test_factory_dispatches_to_evaclip(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "EVACLIPFeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeEvaClipModel(512),
    )
    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "evaclip"

    features_mod.build_feature_extractor(cfg)

    assert len(calls) == 1
    assert calls[0]["variant"] == "base"


def test_factory_dispatches_to_dinotxt(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "DinoTxtFeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeDinoTxtModel(1024),
    )
    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "dinotxt"

    features_mod.build_feature_extractor(cfg)

    assert len(calls) == 1
    assert calls[0]["image_size"] == cfg.stage1.feature_extractor.image_size
