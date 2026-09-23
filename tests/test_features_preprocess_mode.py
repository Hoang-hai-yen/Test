"""Unit tests for stage1.feature_extractor.preprocess_mode -- "stretch"
(default, unchanged) vs "resize_then_crop" (DINOv2's own documented eval
protocol / the DINOv3 paper's own instance-retrieval eval protocol:
resize the shorter side preserving aspect ratio, then center-crop).

Covers the pure _resize_shorter_side_then_center_crop primitive,
_preprocess_dino's mode dispatch, and DINOv3FeatureExtractor._processor_
kwargs() (the per-call override for the HuggingFace AutoImageProcessor
path -- verified against facebook/dinov3-vitb16-pretrain-lvd1689m's own
preprocessor_config.json, which defaults to a direct stretch-to-square,
NOT resize-then-crop, despite that being the DINOv3 PAPER's own protocol).
"""
from __future__ import annotations

import types

import numpy as np
import pytest
from PIL import Image

from aero_eyes.models import features as features_mod
from aero_eyes.models.features import (
    DINOv3FeatureExtractor,
    _preprocess_dino,
    _resize_and_pad_to_square,
    _resize_shorter_side_then_center_crop,
)


# ---------------------------------------------------------------------------
# _resize_shorter_side_then_center_crop: pure primitive
# ---------------------------------------------------------------------------

def test_output_is_always_exactly_size_by_size():
    for w, h in [(400, 100), (100, 400), (224, 224), (50, 50), (1000, 999)]:
        img = Image.new("RGB", (w, h))
        out = _resize_shorter_side_then_center_crop(img, 224)
        assert out.size == (224, 224), f"failed for input size ({w},{h})"


def test_upscales_when_shorter_side_already_below_target():
    """A 50x50 image must still be scaled UP (never padded) to reach the
    224x224 crop -- BICUBIC upscaling, not a zero-padded canvas."""
    img = Image.new("RGB", (50, 50), color=(200, 100, 50))
    out = _resize_shorter_side_then_center_crop(img, 224)
    assert out.size == (224, 224)
    # Center pixel should still reflect the original solid color (no black
    # padding bleeding into the crop).
    cx, cy = out.size[0] // 2, out.size[1] // 2
    r, g, b = out.getpixel((cx, cy))
    assert abs(r - 200) < 30 and abs(g - 100) < 30 and abs(b - 50) < 30


def test_wide_image_crops_from_the_horizontal_center():
    """A wide image with a distinct left/right/center color band -- the
    crop must come from the horizontal CENTER after the shorter (height)
    side is resized to the target."""
    w, h = 600, 200
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    px = img.load()
    third = w // 3
    for x in range(w):
        for y in range(h):
            if x < third:
                px[x, y] = (255, 0, 0)      # left band: red
            elif x < 2 * third:
                px[x, y] = (0, 255, 0)      # center band: green
            else:
                px[x, y] = (0, 0, 255)      # right band: blue

    out = _resize_shorter_side_then_center_crop(img, 224)
    center = out.getpixel((112, 112))
    assert center == (0, 255, 0), f"expected center-band green, got {center}"


def test_preserves_aspect_ratio_of_content_unlike_naive_stretch():
    """A perfect circle drawn on a wide (non-square) canvas must still look
    round after resize_then_crop (aspect-preserving), whereas a naive
    square stretch would visibly distort it into an ellipse. Checked via
    the bounding extent of a solid-color disc along each axis."""
    from PIL import ImageDraw

    w, h = 400, 200
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = 80
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))

    out = _resize_shorter_side_then_center_crop(img, 224)
    arr = np.array(out.convert("L"))
    ys, xs = np.where(arr > 128)
    width_extent = xs.max() - xs.min()
    height_extent = ys.max() - ys.min()
    # A true circle stays round: width and height extents of the disc
    # should match within a small tolerance after aspect-preserving resize.
    assert abs(width_extent - height_extent) <= 3, (width_extent, height_extent)


# ---------------------------------------------------------------------------
# _preprocess_dino: mode dispatch
# ---------------------------------------------------------------------------

def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def test_stretch_mode_is_the_unchanged_default_behavior():
    img_pil = Image.new("RGB", (400, 100), color=(10, 20, 30))
    img_bgr = _to_bgr(img_pil)

    out_default = _preprocess_dino(img_bgr, image_size=224)
    out_explicit_stretch = _preprocess_dino(img_bgr, image_size=224, mode="stretch")
    assert out_default.shape == (3, 224, 224)
    assert np.allclose(out_default.numpy(), out_explicit_stretch.numpy())


def test_resize_then_crop_mode_produces_a_different_tensor_for_nonsquare_input():
    img_pil = Image.new("RGB", (400, 100), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([180, 0, 220, 100], fill=(255, 255, 255))  # a thin vertical stripe
    img_bgr = _to_bgr(img_pil)

    out_stretch = _preprocess_dino(img_bgr, image_size=224, mode="stretch")
    out_crop = _preprocess_dino(img_bgr, image_size=224, mode="resize_then_crop")
    assert out_stretch.shape == out_crop.shape == (3, 224, 224)
    assert not np.allclose(out_stretch.numpy(), out_crop.numpy())


def test_resize_then_crop_is_a_noop_for_already_square_input():
    """For a square input, resize_then_crop's resize(256)+crop(224) and a
    plain resize(224) end up sampling the SAME central region -- not
    bit-identical (different resize pass), but should be very close."""
    img_pil = Image.new("RGB", (224, 224))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    for i in range(0, 224, 16):
        draw.line([(i, 0), (i, 224)], fill=(i % 256, 0, 0))
    img_bgr = _to_bgr(img_pil)

    out_stretch = _preprocess_dino(img_bgr, image_size=224, mode="stretch")
    out_crop = _preprocess_dino(img_bgr, image_size=224, mode="resize_then_crop")
    assert np.abs(out_stretch.numpy() - out_crop.numpy()).mean() < 0.5


# ---------------------------------------------------------------------------
# DINOv3FeatureExtractor._processor_kwargs (HuggingFace path override)
# ---------------------------------------------------------------------------

def _make_dinov3_extractor(preprocess_mode: str, image_size: int = 224) -> DINOv3FeatureExtractor:
    """object.__new__ scaffolding (no real model/weights loaded) -- only
    the attributes _processor_kwargs() itself touches."""
    ext = object.__new__(DINOv3FeatureExtractor)
    ext.preprocess_mode = preprocess_mode
    ext.image_size = image_size
    return ext


def test_processor_kwargs_empty_for_stretch_mode():
    ext = _make_dinov3_extractor("stretch")
    assert ext._processor_kwargs("stretch") == {}


def test_processor_kwargs_empty_for_pad_to_square_mode():
    """pad_to_square always bypasses self.processor() entirely (see
    extract()) -- _processor_kwargs() is never even called with this mode
    in practice, but must still degrade to a harmless {} rather than
    erroring if it somehow were."""
    ext = _make_dinov3_extractor("pad_to_square")
    assert ext._processor_kwargs("pad_to_square") == {}


def test_processor_kwargs_overrides_for_resize_then_crop():
    ext = _make_dinov3_extractor("stretch", image_size=224)  # instance mode irrelevant -- mode arg wins
    kwargs = ext._processor_kwargs("resize_then_crop")
    assert kwargs["do_center_crop"] is True
    assert kwargs["crop_size"] == {"height": 224, "width": 224}
    assert kwargs["size"] == {"shortest_edge": 256}  # round(224 * 256/224) == 256


def test_processor_kwargs_scales_resize_size_with_image_size():
    ext = _make_dinov3_extractor("stretch", image_size=112)
    kwargs = ext._processor_kwargs("resize_then_crop")
    assert kwargs["crop_size"] == {"height": 112, "width": 112}
    assert kwargs["size"] == {"shortest_edge": round(112 * 256 / 224)}


# ---------------------------------------------------------------------------
# _resize_and_pad_to_square: pure primitive
# ---------------------------------------------------------------------------

def test_pad_to_square_output_is_always_exactly_size_by_size():
    for w, h in [(400, 100), (100, 400), (224, 224), (50, 50)]:
        img = Image.new("RGB", (w, h))
        out = _resize_and_pad_to_square(img, 224)
        assert out.size == (224, 224), f"failed for input size ({w},{h})"


def test_pad_to_square_never_discards_content_unlike_center_crop():
    """A thin vertical stripe near one EDGE of a wide image must survive
    pad_to_square (content preserved, just padded) but would be CUT OFF by
    resize_then_crop's genuine center-crop for a wide-enough image."""
    w, h = 600, 100
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 20, h], fill=(255, 255, 255))  # stripe at the far LEFT edge

    padded = _resize_and_pad_to_square(img, 224)
    cropped = _resize_shorter_side_then_center_crop(img, 224)

    padded_has_white = np.array(padded.convert("L")).max() > 200
    cropped_has_white = np.array(cropped.convert("L")).max() > 200
    assert padded_has_white, "pad_to_square must preserve the edge stripe"
    assert not cropped_has_white, "resize_then_crop should have cropped away the far-edge stripe"


def test_pad_to_square_fills_with_the_image_own_mean_color():
    img = Image.new("RGB", (400, 100), color=(10, 200, 30))
    out = _resize_and_pad_to_square(img, 224)
    # Corner pixels are padding (image is wider than tall, padded top/bottom).
    corner = out.getpixel((0, 0))
    assert corner == (10, 200, 30)


# ---------------------------------------------------------------------------
# ref vs candidate preprocess_mode differentiation
# ---------------------------------------------------------------------------

def test_dinov2_constructor_inherits_candidate_mode_when_unset(monkeypatch):
    import aero_eyes.models.features as features_mod

    captured = {}

    def fake_load(self, variant):
        captured["called"] = True
        class _M:
            def eval(self_): return self_
            def to(self_, device): return self_
        return _M()

    monkeypatch.setattr(features_mod.DINOv2FeatureExtractor, "_load", fake_load)
    ext = features_mod.DINOv2FeatureExtractor(preprocess_mode="resize_then_crop")
    assert ext.preprocess_mode == "resize_then_crop"
    assert ext.candidate_preprocess_mode == "resize_then_crop"  # inherited, not set separately

    ext2 = features_mod.DINOv2FeatureExtractor(
        preprocess_mode="resize_then_crop", candidate_preprocess_mode="stretch",
    )
    assert ext2.preprocess_mode == "resize_then_crop"
    assert ext2.candidate_preprocess_mode == "stretch"  # explicit override, independent of ref mode


def test_extract_crops_uses_candidate_mode_not_ref_mode(monkeypatch):
    """The core behavioral guarantee: extract_crops() (candidate path) must
    use candidate_preprocess_mode, NOT preprocess_mode (the ref/default
    path), when the two differ."""
    import aero_eyes.models.features as features_mod

    def fake_load(self, variant):
        class _M:
            def eval(self_): return self_
            def to(self_, device): return self_
        return _M()

    monkeypatch.setattr(features_mod.DINOv2FeatureExtractor, "_load", fake_load)
    ext = features_mod.DINOv2FeatureExtractor(
        preprocess_mode="stretch", candidate_preprocess_mode="resize_then_crop",
    )

    captured_modes = []
    real_extract = features_mod.DINOv2FeatureExtractor.extract

    def spy_extract(self, images, batch_size=16, preprocess_mode=None):
        captured_modes.append(preprocess_mode)
        return np.zeros((len(images), 1), dtype=np.float32)

    monkeypatch.setattr(features_mod.DINOv2FeatureExtractor, "extract", spy_extract)
    from aero_eyes.types import Box

    ext.extract_crops(np.zeros((50, 50, 3), dtype=np.uint8), [Box(0, 0, 10, 10)])
    assert captured_modes == ["resize_then_crop"]


# ---------------------------------------------------------------------------
# preprocess_mode flows through build_feature_extractor's constructor args
# ---------------------------------------------------------------------------

class _FakeExtractor:
    def __init__(self, dim: int, *args, **kwargs):
        self._d = dim

    def extract(self, images, batch_size: int = 16) -> np.ndarray:
        return np.ones((len(images), self._d), dtype=np.float32)

    def _dim(self) -> int:
        return self._d


def test_ensemble_forwards_preprocess_mode_to_dinov2(monkeypatch):
    captured_kwargs = {}
    monkeypatch.setattr(
        features_mod, "DINOv2FeatureExtractor",
        lambda *a, **k: captured_kwargs.update(k) or _FakeExtractor(768),
    )
    monkeypatch.setattr(features_mod, "CLIPFeatureExtractor", lambda *a, **k: _FakeExtractor(512))

    features_mod.EnsembleFeatureExtractor(dino_model="dinov2", preprocess_mode="resize_then_crop")
    assert captured_kwargs.get("preprocess_mode") == "resize_then_crop"


# ---------------------------------------------------------------------------
# DINOv3 pixel_values / forward_cls (shared by extract() and LoRA training)
# ---------------------------------------------------------------------------

class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors, **kwargs):
        import torch
        self.calls.append(kwargs)
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


def _hf_extractor(mode: str):
    ext = object.__new__(DINOv3FeatureExtractor)
    ext.source, ext.preprocess_mode, ext.image_size = "huggingface", mode, 224
    ext.processor = _FakeProcessor()
    return ext


def test_pixel_values_uses_processor_with_overrides_for_resize_then_crop():
    ext = _hf_extractor("stretch")
    pv = ext.pixel_values([np.zeros((30, 60, 3), np.uint8)], mode="resize_then_crop")
    assert pv.shape == (1, 3, 224, 224)
    assert ext.processor.calls[0]["do_center_crop"] is True


def test_pixel_values_bypasses_processor_for_pad_to_square():
    ext = _hf_extractor("pad_to_square")
    pv = ext.pixel_values([np.zeros((30, 60, 3), np.uint8)])
    assert pv.shape == (1, 3, 224, 224)
    assert ext.processor.calls == []


def test_forward_cls_returns_the_pooler_output_and_is_differentiable():
    import torch

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, pixel_values):
            return types.SimpleNamespace(pooler_output=pixel_values.flatten(1) * self.w)

    ext = object.__new__(DINOv3FeatureExtractor)
    ext.source, ext.model = "huggingface", _M()
    out = ext.forward_cls(torch.ones(2, 3))
    out.sum().backward()
    assert out.shape == (2, 3) and ext.model.w.grad is not None
