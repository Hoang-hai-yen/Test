"""Unit test for EnsembleFeatureExtractor's dino_model dispatch (DINOv2 vs
DINOv3 + CLIP concatenation) -- see FeatureExtractorConfig.
ensemble_dino_model's own docstring (aero_eyes/config.py). Monkeypatches
the underlying DINOv2/DINOv3/CLIP extractor classes so this runs without
downloading any real model weights (no existing test in this project
exercises aero_eyes/models/features.py directly, for the same reason).
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.models import features as features_mod


class _FakeExtractor:
    def __init__(self, dim: int, *args, **kwargs):
        self._d = dim
        self.init_args = args
        self.init_kwargs = kwargs

    def extract(self, images, batch_size: int = 16, preprocess_mode=None) -> np.ndarray:
        return np.ones((len(images), self._d), dtype=np.float32)

    def _dim(self) -> int:
        return self._d


def test_ensemble_defaults_to_dinov2(monkeypatch):
    dinov2_calls = []
    monkeypatch.setattr(
        features_mod, "DINOv2FeatureExtractor",
        lambda *a, **k: dinov2_calls.append((a, k)) or _FakeExtractor(768),
    )
    monkeypatch.setattr(features_mod, "CLIPFeatureExtractor", lambda *a, **k: _FakeExtractor(512))
    monkeypatch.setattr(
        features_mod, "DINOv3FeatureExtractor",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("DINOv3 must not be constructed for dino_model='dinov2'")),
    )

    ext = features_mod.EnsembleFeatureExtractor()
    assert len(dinov2_calls) == 1
    assert ext._dim() == 768 + 512


def test_ensemble_dinov3_dispatches_to_dinov3_not_dinov2(monkeypatch):
    dinov3_calls = []
    monkeypatch.setattr(
        features_mod, "DINOv3FeatureExtractor",
        lambda *a, **k: dinov3_calls.append((a, k)) or _FakeExtractor(1024),
    )
    monkeypatch.setattr(features_mod, "CLIPFeatureExtractor", lambda *a, **k: _FakeExtractor(512))
    monkeypatch.setattr(
        features_mod, "DINOv2FeatureExtractor",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("DINOv2 must not be constructed for dino_model='dinov3'")),
    )

    ext = features_mod.EnsembleFeatureExtractor(
        dino_model="dinov3", dinov3_variant="vitl16", dinov3_pretrain_dataset="sat493m",
    )
    assert len(dinov3_calls) == 1
    _, kwargs = dinov3_calls[0]
    assert kwargs["variant"] == "vitl16"
    assert kwargs["pretrain_dataset"] == "sat493m"
    assert ext._dim() == 1024 + 512


def test_ensemble_extract_concatenates_and_normalizes(monkeypatch):
    monkeypatch.setattr(features_mod, "DINOv2FeatureExtractor", lambda *a, **k: _FakeExtractor(4))
    monkeypatch.setattr(features_mod, "CLIPFeatureExtractor", lambda *a, **k: _FakeExtractor(3))

    ext = features_mod.EnsembleFeatureExtractor()
    out = ext.extract([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert out.shape == (1, 7)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_ensemble_rejects_unknown_dino_model():
    with pytest.raises(ValueError, match="Unknown ensemble dino_model"):
        features_mod.EnsembleFeatureExtractor(dino_model="siglip")
