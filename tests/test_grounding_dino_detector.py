"""Unit tests for GroundingDinoDetector (aero_eyes.models.grounding_dino_detector)
and stage123_gdino.resolve_text_prompt -- exercise the pure box-filtering
logic (NMS/min_box_area/top-K) and prompt-precedence rules WITHOUT needing
the real transformers Grounding DINO checkpoint downloaded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.config import AeroEyesConfig
from aero_eyes.models import grounding_dino_detector as gd_mod
from aero_eyes.stages.stage123_gdino import resolve_text_prompt


# ---------------------------------------------------------------------------
# _normalize_prompt
# ---------------------------------------------------------------------------

def test_normalize_prompt_lowercases_and_adds_trailing_period():
    assert gd_mod._normalize_prompt("Red Backpack") == "red backpack."


def test_normalize_prompt_does_not_double_period():
    assert gd_mod._normalize_prompt("red backpack.") == "red backpack."


def test_normalize_prompt_does_not_split_multi_phrase():
    assert gd_mod._normalize_prompt("black backpack. red backpack") == "black backpack. red backpack."


# ---------------------------------------------------------------------------
# GroundingDinoDetector construction / box filtering (no real model needed)
# ---------------------------------------------------------------------------

class _FakeStage123GDinoCfg:
    """A plain stand-in for Stage123GDinoConfig with an out-of-range variant --
    pydantic's own Literal["tiny", "base"] validation already blocks this
    through the real config loader (test_normalize_prompt_* etc. below never
    hit this path), so this test targets GroundingDinoDetector's own runtime
    guard directly: a drift-guard against _HF_MAP and the Literal list ever
    getting out of sync, not something reachable via a real loaded config."""
    variant = "huge"


class _FakeCfgForVariantCheck:
    stage123_gdino = _FakeStage123GDinoCfg()

    def device(self):
        return "cpu"


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown stage123_gdino.variant"):
        gd_mod.GroundingDinoDetector(_FakeCfgForVariantCheck())


def _make_detector(
    nms_iou: float = 0.5, topk_per_keyframe: int = 5,
    min_box_area_enabled: bool = False, min_box_area: int = 24,
) -> "gd_mod.GroundingDinoDetector":
    """Bypasses __init__ (which needs a real HF checkpoint download) --
    only sets the attributes detect_frame's own filtering logic touches,
    same pattern as tests/test_geco2_min_box_area.py::_make_detector."""
    det = object.__new__(gd_mod.GroundingDinoDetector)
    det.box_threshold = 0.35
    det.text_threshold = 0.25
    det.nms_iou = nms_iou
    det.topk_per_keyframe = topk_per_keyframe
    det.min_box_area_enabled = min_box_area_enabled
    det.min_box_area = min_box_area
    det.device = "cpu"
    return det


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self, boxes, scores):
        self._boxes = boxes
        self._scores = scores

    def __call__(self, images, text, return_tensors="pt"):
        return _FakeBatch(input_ids=torch.zeros(1, 1, dtype=torch.long))

    def post_process_grounded_object_detection(
        self, outputs, input_ids, box_threshold, text_threshold, target_sizes,
    ):
        return [{"boxes": torch.tensor(self._boxes, dtype=torch.float32),
                 "scores": torch.tensor(self._scores, dtype=torch.float32)}]


class _FakeModel:
    def __call__(self, **kwargs):
        return object()


def _wire_fake(det: "gd_mod.GroundingDinoDetector", boxes, scores) -> None:
    det.model = _FakeModel()
    det.processor = _FakeProcessor(boxes, scores)


def test_detect_frame_filters_by_min_box_area():
    det = _make_detector(min_box_area_enabled=True, min_box_area=100, nms_iou=0.99)
    # box 0: 5x5=25px^2 (below floor, dropped); box 1: 20x20=400px^2 (kept)
    _wire_fake(det, boxes=[[0, 0, 5, 5], [50, 50, 70, 70]], scores=[0.9, 0.8])
    out = det.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8), "an object")
    assert len(out) == 1
    assert out[0].area() == pytest.approx(400.0)


def test_detect_frame_applies_nms():
    det = _make_detector(nms_iou=0.5, topk_per_keyframe=10)
    # Two heavily-overlapping boxes for the same object -- NMS should keep only the higher-score one.
    _wire_fake(det, boxes=[[10, 10, 50, 50], [12, 12, 52, 52]], scores=[0.9, 0.85])
    out = det.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8), "an object")
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.9)


def test_detect_frame_applies_topk():
    det = _make_detector(nms_iou=0.0, topk_per_keyframe=2)
    # 3 non-overlapping boxes, nms_iou=0.0 keeps all as candidates -- top-K caps at 2.
    _wire_fake(
        det,
        boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        scores=[0.5, 0.9, 0.7],
    )
    out = det.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8), "an object")
    assert len(out) == 2
    assert [b.score for b in out] == pytest.approx([0.9, 0.7])


def test_detect_frame_empty_when_no_boxes():
    det = _make_detector()
    _wire_fake(det, boxes=np.zeros((0, 4)), scores=[])
    out = det.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8), "an object")
    assert out == []


# ---------------------------------------------------------------------------
# resolve_text_prompt precedence
# ---------------------------------------------------------------------------

def _cfg(data_root, default_text_prompt: str = "", text_prompts: dict | None = None) -> AeroEyesConfig:
    return AeroEyesConfig.model_validate({
        "data": {"data_root": str(data_root)},
        "stage123_gdino": {
            "default_text_prompt": default_text_prompt,
            "text_prompts": text_prompts or {},
        },
    })


def test_resolve_text_prompt_prefers_sample_prompt_file(tmp_path):
    sample_dir = tmp_path / "Sample_0"
    sample_dir.mkdir()
    (sample_dir / "prompt.txt").write_text("a blue drone", encoding="utf-8")
    cfg = _cfg(tmp_path, default_text_prompt="fallback", text_prompts={"Sample_0": "map entry"})
    assert resolve_text_prompt(cfg, "Sample_0") == "a blue drone"


def test_resolve_text_prompt_falls_back_to_map_entry(tmp_path):
    cfg = _cfg(tmp_path, default_text_prompt="fallback", text_prompts={"Sample_0": "map entry"})
    assert resolve_text_prompt(cfg, "Sample_0") == "map entry"


def test_resolve_text_prompt_falls_back_to_default(tmp_path):
    cfg = _cfg(tmp_path, default_text_prompt="fallback")
    assert resolve_text_prompt(cfg, "Sample_0") == "fallback"


def test_resolve_text_prompt_raises_when_nothing_set(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="no text prompt for sample"):
        resolve_text_prompt(cfg, "Sample_0")


def test_resolve_text_prompt_falls_through_empty_prompt_file(tmp_path):
    sample_dir = tmp_path / "Sample_0"
    sample_dir.mkdir()
    (sample_dir / "prompt.txt").write_text("   ", encoding="utf-8")
    cfg = _cfg(tmp_path, default_text_prompt="fallback")
    assert resolve_text_prompt(cfg, "Sample_0") == "fallback"
