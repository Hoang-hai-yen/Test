"""Unit tests for pooling="multiscale_attn" (DINOv2FeatureExtractor,
DINOv3FeatureExtractor) -- the ViT analog of DAVE's detect-and-verify
backbone (DAVE/models/backbone.py concatenates ResNet layer2+3+4 conv
features instead of a single global vector), see FeatureExtractorConfig.
dinov2_pooling/dinov3_pooling's own docstring (aero_eyes/config.py).

Covers the pure-math helpers directly (no model weights needed) plus the
extractor classes' wiring with fakes standing in for the HF model/processor
(same monkeypatch convention as test_features_ensemble.py -- no real
download in this test suite).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aero_eyes.models import features as features_mod


# ---------------------------------------------------------------------------
# _select_multiscale_layers
# ---------------------------------------------------------------------------

def test_select_multiscale_layers_vit_base_depth():
    # ViT-B/14 and ViT-B/16 both have 12 transformer blocks.
    assert features_mod._select_multiscale_layers(12) == [6, 9, 12]


def test_select_multiscale_layers_vit_large_depth():
    assert features_mod._select_multiscale_layers(24) == [12, 18, 24]


def test_select_multiscale_layers_always_includes_final_layer():
    for n in (1, 2, 3, 4, 9, 12, 24, 40):
        layers = features_mod._select_multiscale_layers(n)
        assert layers[-1] == n
        assert all(1 <= l <= n for l in layers)
        assert layers == sorted(set(layers))
        assert len(layers) <= 3


# ---------------------------------------------------------------------------
# _patch_grid_count
# ---------------------------------------------------------------------------

def test_patch_grid_count_exact():
    pixel_values = torch.zeros(2, 3, 224, 224)
    assert features_mod._patch_grid_count(pixel_values, patch_size=14) == 16 * 16
    assert features_mod._patch_grid_count(pixel_values, patch_size=16) == 14 * 14


def test_patch_grid_count_rejects_misaligned_size():
    pixel_values = torch.zeros(1, 3, 225, 225)
    with pytest.raises(RuntimeError, match="not divisible by"):
        features_mod._patch_grid_count(pixel_values, patch_size=14)


# ---------------------------------------------------------------------------
# _multiscale_attn_pool
# ---------------------------------------------------------------------------

def _make_hidden_and_attn(batch: int, seq: int, dim: int, num_layers: int, heads: int = 2):
    """num_layers hidden_states entries (index 0 = embeddings, matching the
    real hidden_states[1:] = after each block convention), each layer's own
    uniform-attention tensor."""
    hidden_states = tuple(torch.randn(batch, seq, dim) for _ in range(num_layers + 1))
    attentions = tuple(torch.full((batch, heads, seq, seq), 1.0 / seq) for _ in range(num_layers))
    return hidden_states, attentions


def test_multiscale_attn_pool_output_shape():
    batch, seq, dim, num_layers = 2, 1 + 9, 8, 4  # 1 CLS + 9 patches (3x3 grid)
    hidden_states, attentions = _make_hidden_and_attn(batch, seq, dim, num_layers)
    scale_layers = features_mod._select_multiscale_layers(num_layers)  # [2, 3, 4]
    pooled = features_mod._multiscale_attn_pool(hidden_states, attentions, scale_layers, num_patches=9)
    assert pooled.shape == (batch, dim * len(scale_layers))


def test_multiscale_attn_pool_handles_register_tokens():
    # CLS + 4 registers + 9 patches = seq 14; num_patches=9 must still work.
    batch, dim, num_layers = 1, 5, 2
    seq = 1 + 4 + 9
    hidden_states, attentions = _make_hidden_and_attn(batch, seq, dim, num_layers)
    scale_layers = [1, 2]
    pooled = features_mod._multiscale_attn_pool(hidden_states, attentions, scale_layers, num_patches=9)
    assert pooled.shape == (batch, dim * 2)


def test_multiscale_attn_pool_concentrated_attention_matches_that_patch():
    # If the CLS token attends ENTIRELY to one specific patch, the weighted
    # pool for that layer must equal that patch's own feature vector.
    batch, dim, num_layers, num_patches = 1, 6, 1, 4
    seq = 1 + num_patches  # no registers
    hidden_states = (torch.randn(batch, seq, dim), torch.randn(batch, seq, dim))
    attn = torch.zeros(batch, 1, seq, seq)
    target_patch = 2  # 0-indexed among patches
    attn[:, :, 0, 1 + target_patch] = 1.0  # CLS (row 0) attends fully to this one patch
    attentions = (attn,)
    pooled = features_mod._multiscale_attn_pool(hidden_states, attentions, [1], num_patches)
    expected = hidden_states[1][:, 1 + target_patch, :]
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_multiscale_attn_pool_raises_on_missing_attentions():
    hidden_states, _ = _make_hidden_and_attn(1, 5, 4, 1)
    with pytest.raises(RuntimeError, match="output_attentions=True"):
        features_mod._multiscale_attn_pool(hidden_states, None, [1], num_patches=4)
    with pytest.raises(RuntimeError, match="output_attentions=True"):
        features_mod._multiscale_attn_pool(hidden_states, (None,), [1], num_patches=4)


def test_multiscale_attn_pool_raises_on_bad_num_patches():
    hidden_states, attentions = _make_hidden_and_attn(1, 5, 4, 1)  # seq=5
    with pytest.raises(RuntimeError, match="leaves no room for a CLS token"):
        features_mod._multiscale_attn_pool(hidden_states, attentions, [1], num_patches=5)


# ---------------------------------------------------------------------------
# DINOv2FeatureExtractor(pooling=...) wiring
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class _FakeHFModel:
    def __init__(self, dim: int, num_layers: int, attn_implementation=None):
        self._dim = dim
        self.config = _FakeConfig(num_layers)
        self.attn_implementation = attn_implementation

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, pixel_values=None, output_hidden_states=False, output_attentions=False):
        batch = pixel_values.shape[0]
        h, w = pixel_values.shape[-2], pixel_values.shape[-1]
        num_patches = (h // 14) * (w // 14)
        seq = 1 + num_patches
        hidden_states = tuple(
            torch.randn(batch, seq, self._dim) for _ in range(self.config.num_hidden_layers + 1)
        )
        attentions = tuple(
            torch.full((batch, 1, seq, seq), 1.0 / seq) for _ in range(self.config.num_hidden_layers)
        )

        class _Out:
            pass
        out = _Out()
        out.hidden_states = hidden_states if output_hidden_states else None
        out.attentions = attentions if output_attentions else None
        out.last_hidden_state = hidden_states[-1]
        out.pooler_output = hidden_states[-1][:, 0]
        return out


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


def test_dinov2_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="Unknown DINOv2 pooling"):
        features_mod.DINOv2FeatureExtractor(pooling="bogus")


def test_dinov2_multiscale_attn_uses_hf_backend_and_dim(monkeypatch):
    monkeypatch.setattr(
        features_mod.DINOv2FeatureExtractor, "_load_hf",
        lambda self, variant, eager_attn=False: (_FakeHFModel(768, 12), _FakeProcessor()),
    )
    ext = features_mod.DINOv2FeatureExtractor(variant="vitb14", pooling="multiscale_attn")
    assert ext._scale_layers == [6, 9, 12]
    assert ext._dim() == 768 * 3
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 768 * 3)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_dinov2_cls_pooling_unaffected(monkeypatch):
    class _FakeHubModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, x):
            return torch.ones(x.shape[0], 768)

    monkeypatch.setattr(
        features_mod.torch.hub, "load", lambda *a, **k: _FakeHubModel(),
    )
    ext = features_mod.DINOv2FeatureExtractor(variant="vitb14")  # pooling defaults to "cls"
    assert ext.pooling == "cls"
    assert ext._dim() == 768
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 768)


# ---------------------------------------------------------------------------
# DINOv3FeatureExtractor(pooling=...) wiring
# ---------------------------------------------------------------------------

def test_dinov3_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="Unknown DINOv3 pooling"):
        features_mod.DINOv3FeatureExtractor(pooling="bogus")


def test_dinov3_multiscale_attn_rejects_kaggle_source():
    with pytest.raises(ValueError, match="requires source='huggingface'"):
        features_mod.DINOv3FeatureExtractor(
            source="kaggle", kaggle_model_id="whoever/dinov3", pooling="multiscale_attn",
        )


def test_dinov3_multiscale_attn_uses_eager_attention_and_dim(monkeypatch):
    load_calls = []

    def fake_load_hf(self, variant, pretrain_dataset, eager_attn=False):
        load_calls.append(eager_attn)
        return _FakeHFModel(768, 12), _FakeProcessor()

    monkeypatch.setattr(features_mod.DINOv3FeatureExtractor, "_load_huggingface", fake_load_hf)

    ext = features_mod.DINOv3FeatureExtractor(variant="vitb16", pooling="multiscale_attn")
    assert load_calls == [True]
    assert ext._scale_layers == [6, 9, 12]
    assert ext._dim() == 768 * 3
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 768 * 3)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_dinov3_cls_pooling_does_not_request_eager_attention(monkeypatch):
    load_calls = []

    def fake_load_hf(self, variant, pretrain_dataset, eager_attn=False):
        load_calls.append(eager_attn)
        return _FakeHFModel(768, 12), _FakeProcessor()

    monkeypatch.setattr(features_mod.DINOv3FeatureExtractor, "_load_huggingface", fake_load_hf)

    ext = features_mod.DINOv3FeatureExtractor(variant="vitb16")  # pooling defaults to "cls"
    assert load_calls == [False]
    assert ext._dim() == 768


# ---------------------------------------------------------------------------
# Factory / Ensemble pass-through
# ---------------------------------------------------------------------------

class _FakeBuiltExtractor:
    """Stands in for a fully-constructed *FeatureExtractor (not the HF model
    inside one) -- just enough surface (_dim) for EnsembleFeatureExtractor's
    own __init__ (which logs self._dim()) and build_feature_extractor's
    optional wrapper checks not to blow up."""

    def __init__(self, dim: int = 768):
        self._d = dim

    def _dim(self) -> int:
        return self._d


def test_ensemble_passes_pooling_through_to_dinov2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "DINOv2FeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeBuiltExtractor(768),
    )
    monkeypatch.setattr(features_mod, "CLIPFeatureExtractor", lambda *a, **k: _FakeBuiltExtractor(512))

    features_mod.EnsembleFeatureExtractor(dinov2_pooling="multiscale_attn")
    assert len(calls) == 1
    assert calls[0]["pooling"] == "multiscale_attn"


def test_factory_passes_dinov2_pooling(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "DINOv2FeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeBuiltExtractor(768),
    )
    from aero_eyes.config import AeroEyesConfig

    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "dinov2"
    cfg.stage1.feature_extractor.dinov2_pooling = "multiscale_attn"
    features_mod.build_feature_extractor(cfg)
    assert calls[0]["pooling"] == "multiscale_attn"


def test_factory_passes_dinov3_pooling(monkeypatch):
    calls = []
    monkeypatch.setattr(
        features_mod, "DINOv3FeatureExtractor",
        lambda *a, **k: calls.append(k) or _FakeBuiltExtractor(768),
    )
    from aero_eyes.config import AeroEyesConfig

    cfg = AeroEyesConfig()
    cfg.stage1.feature_extractor.model = "dinov3"
    cfg.stage1.feature_extractor.dinov3_pooling = "multiscale_attn"
    features_mod.build_feature_extractor(cfg)
    assert calls[0]["pooling"] == "multiscale_attn"
