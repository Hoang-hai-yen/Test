"""Unit tests for DaveVerificationExtractor -- DAVE's (arXiv:2404.16622) OWN
verify-stage embedding (ResNet50+SWaV backbone + learned `feat_comp`
projection from verification.pth), wrapped to satisfy this project's
feature-extractor interface. See aero_eyes.models.dave_verification's own
module docstring and DaveVerificationConfig (aero_eyes/config.py) for the
full rationale.

Never downloads the real SWaV checkpoint or needs a real verification.pth --
aero_eyes.models._dave_vendor.Backbone/Feature_Transform (copied verbatim
from DAVE under its own MIT license -- see that module's own header) are
monkeypatched with tiny fakes before constructing DaveVerificationExtractor,
so these tests only exercise THIS project's own wrapper logic (checkpoint
key extraction, interface contract), not DAVE's real architecture (trusted
as-is, copied unmodified).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import aero_eyes.models.dave_verification as dv
from aero_eyes.types import Box


class _FakeBackbone(torch.nn.Module):
    """Stands in for _dave_vendor.Backbone -- same constructor signature,
    but a trivial conv + fixed-stride pool instead of a real ResNet50+SWaV,
    so no download and a tiny, fast forward pass."""

    def __init__(self, name, pretrained, dilation, reduction, swav, requires_grad):
        super().__init__()
        self.reduction = reduction
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        return F.avg_pool2d(feat, kernel_size=self.reduction, stride=self.reduction)


class _FakeFeatureTransform(torch.nn.Module):
    """Stands in for _dave_vendor.Feature_Transform -- trivial conv +
    flatten instead of the real ConvBlock1 arithmetic, purely to exercise
    the loading/interface contract."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(8, 5, kernel_size=1)
        self.flat = torch.nn.Flatten()

    def forward(self, x):
        return self.flat(self.proj(x))


@pytest.fixture
def fake_dave_classes(monkeypatch):
    """Swaps the real Backbone/Feature_Transform (imported into
    dave_verification's own namespace) for the fakes above, and returns a
    verification.pth-shaped checkpoint path built from the FAKE
    Feature_Transform's own real state_dict (prefixed "feat_comp.",
    matching how DAVE/main.py itself extracts it from a real checkpoint).
    """
    monkeypatch.setattr(dv, "Backbone", _FakeBackbone)
    monkeypatch.setattr(dv, "Feature_Transform", _FakeFeatureTransform)

    def _make_weights(tmp_path) -> str:
        real_feat_comp = _FakeFeatureTransform()
        checkpoint = {"model": {f"feat_comp.{k}": v for k, v in real_feat_comp.state_dict().items()}}
        weights_path = tmp_path / "verification.pth"
        torch.save(checkpoint, weights_path)
        return str(weights_path)

    return _make_weights


def test_missing_weights_path_raises(fake_dave_classes, tmp_path):
    with pytest.raises(FileNotFoundError):
        dv.DaveVerificationExtractor(
            weights_path=str(tmp_path / "missing_verification.pth"),
            device="cpu", image_size=32, reduction=4, kernel_dim=3,
        )


def test_checkpoint_without_feat_comp_keys_raises(fake_dave_classes, tmp_path):
    bad_path = tmp_path / "bad_verification.pth"
    torch.save({"model": {"some_other_module.weight": torch.zeros(3)}}, bad_path)
    with pytest.raises(ValueError, match="feat_comp"):
        dv.DaveVerificationExtractor(
            weights_path=str(bad_path),
            device="cpu", image_size=32, reduction=4, kernel_dim=3,
        )


def _build_extractor(fake_dave_classes, tmp_path):
    weights_path = fake_dave_classes(tmp_path)
    return dv.DaveVerificationExtractor(
        weights_path=weights_path,
        device="cpu", image_size=32, reduction=4, kernel_dim=3,
    )


def test_extract_returns_l2_normalized_expected_shape(fake_dave_classes, tmp_path):
    extractor = _build_extractor(fake_dave_classes, tmp_path)
    images = [np.zeros((40, 40, 3), dtype=np.uint8), np.full((50, 60, 3), 255, dtype=np.uint8)]

    out = extractor.extract(images)

    assert out.shape == (2, extractor._dim())
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_empty_list_returns_correct_shape(fake_dave_classes, tmp_path):
    extractor = _build_extractor(fake_dave_classes, tmp_path)
    out = extractor.extract([])
    assert out.shape == (0, extractor._dim())


def test_extract_crops_returns_l2_normalized_per_box(fake_dave_classes, tmp_path):
    extractor = _build_extractor(fake_dave_classes, tmp_path)
    frame = np.random.randint(0, 255, size=(80, 100, 3), dtype=np.uint8)
    boxes = [Box(10, 10, 40, 40), Box(50, 20, 90, 70), Box(0, 0, 100, 80)]

    out = extractor.extract_crops(frame, boxes)

    assert out.shape == (3, extractor._dim())
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_crops_empty_boxes_returns_correct_shape(fake_dave_classes, tmp_path):
    extractor = _build_extractor(fake_dave_classes, tmp_path)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = extractor.extract_crops(frame, [])
    assert out.shape == (0, extractor._dim())


def test_extract_crops_respects_batch_size(fake_dave_classes, tmp_path):
    """batch_size only chunks how many RoIs go through feat_comp per forward
    call -- output must be identical regardless of chunk size."""
    extractor = _build_extractor(fake_dave_classes, tmp_path)
    frame = np.random.randint(0, 255, size=(80, 100, 3), dtype=np.uint8)
    boxes = [Box(x, x, x + 20, x + 20) for x in range(0, 60, 10)]

    out_full = extractor.extract_crops(frame, boxes, batch_size=100)
    out_chunked = extractor.extract_crops(frame, boxes, batch_size=2)

    np.testing.assert_allclose(out_full, out_chunked, atol=1e-5)
