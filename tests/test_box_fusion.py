from pathlib import Path
from types import SimpleNamespace

import pytest

from aero_eyes.config import CandidateFusionConfig, Geco2CosineRescoreConfig
from aero_eyes.stages import stage123_geco2
from aero_eyes.types import Box
from aero_eyes.utils.box_fusion import box_iomin, fuse_overlapping_boxes
from tests.test_geco2_second_pass import _FakeDetector, _FakeExtractor, _frame, _patch_frame_iterator

# a motorbike detected as three overlapping PART boxes, plus one unrelated far box
WHEEL = Box(100, 150, 160, 210, score=0.6)
BODY = Box(120, 100, 260, 200, score=0.9)
HANDLE = Box(240, 90, 280, 150, score=0.5)
FAR = Box(500, 500, 540, 540, score=0.8)
PARTS = [BODY, FAR, WHEEL, HANDLE]          # best-score-first is NOT required of the input


def _cfg(**kw):
    return CandidateFusionConfig(enabled=True, **kw)


def test_disabled_by_default_and_a_noop():
    assert Geco2CosineRescoreConfig().candidate_fusion.enabled is False
    assert fuse_overlapping_boxes(PARTS, CandidateFusionConfig()) == PARTS


def test_iomin_is_one_for_nested_boxes_while_iou_is_low():
    from aero_eyes.utils.geometry import box_iou

    big, small = Box(0, 0, 100, 100), Box(20, 20, 50, 50)
    assert box_iomin(big, small) == 1.0
    assert box_iou(big, small) == pytest.approx(0.09)


def test_union_turns_parts_into_the_enclosing_box_and_appends_it():
    out = fuse_overlapping_boxes(PARTS, _cfg(containment_thresh=0.4))
    assert out[:4] == PARTS                                       # originals kept, in order
    fused = out[4]
    assert (fused.x1, fused.y1, fused.x2, fused.y2) == (100, 90, 280, 210)
    assert fused.score == 0.9                                     # best member score
    assert len(out) == 5                                          # FAR has no partner -> no fused box for it


def test_union_can_replace_the_members_instead():
    out = fuse_overlapping_boxes(PARTS, _cfg(containment_thresh=0.4, keep_originals=False))
    assert [round(b.score, 2) for b in out] == [0.9, 0.8]         # fused (0.9) then FAR (0.8), best first
    assert (out[0].x1, out[0].x2) == (100, 280)


def test_containment_threshold_decides_who_is_linked():
    # HANDLE overlaps BODY by only 41.7% of its own area: linked at 0.4, not at 0.5
    at05 = fuse_overlapping_boxes(PARTS, _cfg(containment_thresh=0.5))
    assert any((b.x1, b.x2) == (100, 260) for b in at05[4:])      # wheel + body only


def test_union_refuses_a_chain_that_spans_too_much():
    chain = [Box(50 * k, 0, 50 * k + 100, 10, score=0.5) for k in range(5)]     # each links only its neighbours
    assert len(fuse_overlapping_boxes(chain, _cfg(max_union_area_ratio=4.0))) == 6     # union is 3x the largest: fused
    assert fuse_overlapping_boxes(chain, _cfg(max_union_area_ratio=2.5)) == chain      # ...too big: left alone


def test_min_boxes_leaves_small_clusters_alone():
    assert fuse_overlapping_boxes([WHEEL, BODY], _cfg(containment_thresh=0.4, min_boxes=3)) == [WHEEL, BODY]


def test_wbf_averages_near_duplicates_by_score():
    a, b = Box(0, 0, 100, 100, score=0.8), Box(10, 0, 110, 100, score=0.4)
    out = fuse_overlapping_boxes([a, b, FAR], _cfg(mode="wbf", iou_thresh=0.3, keep_originals=False))
    fused = next(x for x in out if x is not FAR)
    assert fused.x1 == pytest.approx(10 / 3) and fused.x2 == pytest.approx(310 / 3)
    assert fused.score == pytest.approx(0.6)
    assert len(out) == 2


def test_wbf_does_not_merge_nested_parts_into_a_whole():
    big, small = Box(0, 0, 100, 100, score=0.9), Box(20, 20, 50, 50, score=0.6)   # IoU 0.09
    assert fuse_overlapping_boxes([big, small], _cfg(mode="wbf", iou_thresh=0.3)) == [big, small]


def test_peak_contrast_is_kept_only_when_every_member_has_one():
    a, b = Box(0, 0, 100, 100, score=0.9, peak_contrast=2.0), Box(20, 20, 50, 50, score=0.6, peak_contrast=5.0)
    assert fuse_overlapping_boxes([a, b], _cfg())[-1].peak_contrast == 5.0
    c = Box(20, 20, 50, 50, score=0.6)
    assert fuse_overlapping_boxes([a, c], _cfg())[-1].peak_contrast is None


def test_candidate_pass_fuses_before_extracting_features(monkeypatch):
    _patch_frame_iterator(monkeypatch, {10: _frame(10)})
    extractor = _FakeExtractor(dim=3)
    cfg = SimpleNamespace(
        stage2=SimpleNamespace(candidate=SimpleNamespace(feature_crop_pad=0.1)),
        runtime=SimpleNamespace(batch_size=8),
    )
    cands = stage123_geco2._run_geco2_candidate_pass(
        _FakeDetector({10: [BODY, WHEEL, HANDLE]}), extractor, Path("/nonexistent.mp4"), {10}, lambda: "P",
        color_sig=None, cpf_cfg=None, cfg=cfg, fusion_cfg=_cfg(containment_thresh=0.4),
    )
    assert len(cands[10]) == 4 and extractor.calls == [(10, 4)]      # 3 parts + 1 fused box, all embedded
