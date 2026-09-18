"""Unit tests for GeCo2DynamicPrototypeTracker -- stage123_geco2.
dynamic_prototype's online/incremental exemplar-token update, consecutive-
hit confirmation, and cosine cross-check (both "feature_extractor" and
"hiera" sources). Fakes GeCo2Detector.encode_exemplars and the
feature_extractor cross-check so this runs without the real GECO2 repo,
weights, or GPU."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models.geco2_detector import GeCo2DynamicPrototypeTracker
from aero_eyes.types import Box


def _make_cfg(accuracy_mode="cheap_boosters", multi_reference_embedding=True, multi_ref_pooling="mean", **dp_overrides):
    dp_defaults = dict(
        enabled=True,
        max_tokens=3,
        min_consecutive_hits=2,
        consecutive_hits_iou=0.5,
        cross_check_source="feature_extractor",
        cross_check_threshold=0.5,
    )
    dp_defaults.update(dp_overrides)
    return SimpleNamespace(
        stage123_geco2=SimpleNamespace(dynamic_prototype=SimpleNamespace(**dp_defaults)),
        stage1=SimpleNamespace(prototype=SimpleNamespace(cache_name="prototype.npz")),
        stage2=SimpleNamespace(candidate=SimpleNamespace(feature_crop_pad=0.1)),
        runtime=SimpleNamespace(batch_size=16),
        accuracy=SimpleNamespace(
            mode=accuracy_mode,
            cheap_boosters=SimpleNamespace(
                multi_reference_embedding=multi_reference_embedding,
                multi_ref_pooling=multi_ref_pooling,
            ),
        ),
    )


def _token_set(dim: int = 4, value: float = 1.0, n_tokens: int = 1) -> dict:
    t = torch.full((1, n_tokens, dim), value)
    return {"main": t.clone(), "l1": t.clone(), "l2": t.clone()}


class _FakeDetector:
    """Records encode_exemplars calls; returns a token whose value encodes
    which call it was, so tests can tell tokens apart."""

    def __init__(self, use_shape_token: bool = False):
        self.use_shape_token = use_shape_token
        self.calls: list[tuple] = []
        self._next_value = 10.0

    def encode_exemplars(self, ref_images_bgr, ref_boxes):
        self.calls.append((ref_images_bgr, ref_boxes))
        tok = _token_set(value=self._next_value)
        self._next_value += 1.0
        return tok


def _make_tracker(cfg, detector=None, base_prototype=None, tmp_path=None) -> GeCo2DynamicPrototypeTracker:
    from pathlib import Path

    detector = detector or _FakeDetector()
    base_prototype = base_prototype or _token_set(value=0.0)
    work_dir = Path(tmp_path) if tmp_path is not None else Path("/nonexistent")
    return GeCo2DynamicPrototypeTracker(cfg, detector, base_prototype, work_dir, "sample_x")


def test_effective_prototype_is_base_when_nothing_accepted():
    cfg = _make_cfg()
    base = _token_set(value=0.0)
    tracker = _make_tracker(cfg, base_prototype=base)
    assert tracker.effective_prototype() is base


def test_offer_noop_when_disabled():
    cfg = _make_cfg(enabled=False)
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    box = Box(10, 10, 20, 20, score=0.9)
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box)
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box)
    assert detector.calls == []
    assert tracker.effective_prototype() is tracker.base_prototype


def test_offer_requires_consecutive_hits_before_encoding():
    """Below min_consecutive_hits, encode_exemplars must not even be called
    -- the whole point is avoiding the extra backbone pass for unconfirmed
    candidates."""
    cfg = _make_cfg(min_consecutive_hits=3)
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    box = Box(10, 10, 20, 20, score=0.9)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    tracker.offer(frame, box)
    tracker.offer(frame, box)
    assert detector.calls == [], "should not encode until the 3rd consecutive agreeing hit"

    # Wire a trivial cross-check that always passes, to isolate this test
    # to the consecutive-hit gate alone.
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 1.0

    tracker.offer(frame, box)
    assert len(detector.calls) == 1, "3rd consecutive agreeing hit should trigger encode_exemplars"


def test_offer_rejects_candidate_below_cross_check_threshold():
    cfg = _make_cfg(min_consecutive_hits=1, cross_check_threshold=0.9)
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 0.5  # below threshold

    box = Box(10, 10, 20, 20, score=0.9)
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box)

    assert len(detector.calls) == 1, "encode_exemplars still runs (needed for the similarity check itself)"
    assert tracker.effective_prototype() is tracker.base_prototype, "low-similarity candidate must not be appended"


def test_offer_precomputed_feature_skips_extractor_call():
    """When the caller already has this box's stage1.feature_extractor
    embedding (e.g. run_stage12_geco2_candidates embeds every surviving
    candidate anyway), offer(precomputed_feature=...) must use it directly
    instead of calling the (here deliberately unset) extractor."""
    cfg = _make_cfg(min_consecutive_hits=1, cross_check_threshold=0.5)
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = np.array([1.0, 0.0])
    # _cross_extractor stays None -- if the precomputed-feature shortcut
    # didn't work, _feature_extractor_similarity would try to lazily build
    # a real one and blow up (no real model/prototype.npz available here).

    box = Box(10, 10, 20, 20, score=0.9)
    feat = np.array([1.0, 0.0])  # cosine sim = 1.0 with the fake prototype above
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box, precomputed_feature=feat)

    assert tracker._cross_extractor is None, "must not have tried to lazily build a real extractor"
    eff = tracker.effective_prototype()
    assert eff["main"].shape[1] == 2, "candidate should have been accepted (sim=1.0 >= 0.5)"


def test_offer_accepts_and_appends_when_both_gates_pass():
    cfg = _make_cfg(min_consecutive_hits=1, cross_check_threshold=0.5)
    detector = _FakeDetector()
    base = _token_set(value=0.0)
    tracker = _make_tracker(cfg, detector=detector, base_prototype=base)
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 0.8  # above threshold

    box = Box(10, 10, 20, 20, score=0.9)
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box)

    eff = tracker.effective_prototype()
    assert eff is not base
    assert eff["main"].shape[1] == 2  # base (1 token) + 1 appended
    assert eff["main"][0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]   # original ref token, untouched
    assert eff["main"][0, 1].tolist() == [10.0, 10.0, 10.0, 10.0]  # newly appended token


def test_offer_evicts_oldest_appended_token_never_the_original():
    cfg = _make_cfg(min_consecutive_hits=1, cross_check_threshold=0.0, max_tokens=2)
    detector = _FakeDetector()
    base = _token_set(value=0.0)
    tracker = _make_tracker(cfg, detector=detector, base_prototype=base)
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 1.0
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    box = Box(10, 10, 20, 20, score=0.9)

    for _ in range(3):
        tracker.offer(frame, box)  # min_consecutive_hits=1 -> each call confirms immediately

    eff = tracker.effective_prototype()
    # max_tokens=2 -> only the 2 MOST RECENT appended tokens survive (values 11, 12 -- the
    # first appended, value 10, was evicted), plus the original base token untouched.
    assert eff["main"].shape[1] == 3  # base + 2 dynamic
    values = sorted(eff["main"][:, i, 0].item() for i in range(3))
    assert values == [0.0, 11.0, 12.0]


def test_feature_extractor_similarity_uses_single_prototype_when_multi_ref_off():
    cfg = _make_cfg(multi_reference_embedding=False)
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = np.array([1.0, 0.0])
    tracker._cross_per_ref_features = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]

    box = Box(10, 10, 20, 20, score=0.9)
    feat = np.array([1.0, 0.0])
    sim = tracker._feature_extractor_similarity(np.zeros((10, 10, 3), dtype=np.uint8), box, precomputed_feature=feat)
    assert sim == pytest.approx(1.0), "multi_reference_embedding=False must score against the fused prototype only"


def test_feature_extractor_similarity_pools_per_ref_with_mean():
    cfg = _make_cfg(multi_reference_embedding=True, multi_ref_pooling="mean")
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = np.array([1.0, 0.0])  # would give sim=1.0 if wrongly used instead of per-ref
    # feat matches ref[0] perfectly (sim=1.0), is orthogonal to ref[1] and ref[2] (sim=0.0).
    tracker._cross_per_ref_features = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]

    box = Box(10, 10, 20, 20, score=0.9)
    feat = np.array([1.0, 0.0])
    sim = tracker._feature_extractor_similarity(np.zeros((10, 10, 3), dtype=np.uint8), box, precomputed_feature=feat)
    assert sim == pytest.approx(1.0 / 3.0), "mean pooling across the 3 per-ref cosine scores, not the fused vector"


def test_feature_extractor_similarity_pools_per_ref_with_max():
    cfg = _make_cfg(multi_reference_embedding=True, multi_ref_pooling="max")
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = np.array([0.0, 1.0])  # would give sim=0.0 if wrongly used instead of per-ref
    tracker._cross_per_ref_features = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]

    box = Box(10, 10, 20, 20, score=0.9)
    feat = np.array([1.0, 0.0])
    sim = tracker._feature_extractor_similarity(np.zeros((10, 10, 3), dtype=np.uint8), box, precomputed_feature=feat)
    assert sim == pytest.approx(1.0), "max pooling should surface the single best-matching ref's score"


def test_offer_precomputed_feature_multi_ref_pooling_gates_acceptance():
    """End-to-end through offer(): a candidate that only matches ONE of the
    3 per-ref vectors well should still be accepted under max pooling even
    though its score against the fused prototype would fail threshold."""
    cfg = _make_cfg(
        multi_reference_embedding=True, multi_ref_pooling="max",
        min_consecutive_hits=1, cross_check_threshold=0.9,
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = np.array([0.0, 1.0])  # fused vector: sim=0.0, would fail threshold
    tracker._cross_per_ref_features = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]

    box = Box(10, 10, 20, 20, score=0.9)
    feat = np.array([1.0, 0.0])
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box, precomputed_feature=feat)

    eff = tracker.effective_prototype()
    assert eff is not tracker.base_prototype, "max-pooled per-ref score (1.0) should clear threshold=0.9"


def test_offer_counters_track_where_candidates_stall():
    """Diagnostic counters (log_summary's data source) must distinguish
    the 3 failure points -- never offered, never confirmed, and confirmed
    but cross-check rejected -- since offer()'s own early returns/debug
    logs give no visibility into which one a zero-token run stalled at."""
    cfg = _make_cfg(min_consecutive_hits=2, cross_check_threshold=0.9)
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 0.5  # < 0.9
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    box = Box(10, 10, 20, 20, score=0.9)

    tracker.offer(frame, box)  # 1st hit -- not yet confirmed (min_consecutive_hits=2)
    assert tracker._n_offers == 1
    assert tracker._n_confirmed == 0

    tracker.offer(frame, box)  # 2nd agreeing hit -- confirmed, then cross-check rejects
    assert tracker._n_offers == 2
    assert tracker._n_confirmed == 1
    assert tracker._n_cross_check_rejected == 1
    assert tracker._n_appended == 0
    assert tracker.dynamic_token_count() == 0


def test_offer_counters_count_appended():
    cfg = _make_cfg(min_consecutive_hits=1, cross_check_threshold=0.5)
    tracker = _make_tracker(cfg)
    tracker._feature_extractor_similarity = lambda frame_bgr, box, precomputed_feature=None: 0.8
    box = Box(10, 10, 20, 20, score=0.9)

    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), box)

    assert tracker._n_offers == 1
    assert tracker._n_confirmed == 1
    assert tracker._n_appended == 1
    assert tracker._n_cross_check_rejected == 0


def test_offer_noop_when_disabled_does_not_increment_counters():
    cfg = _make_cfg(enabled=False)
    tracker = _make_tracker(cfg)
    tracker.offer(np.zeros((10, 10, 3), dtype=np.uint8), Box(10, 10, 20, 20, score=0.9))
    assert tracker._n_offers == 0


def test_log_summary_does_not_raise(caplog):
    cfg = _make_cfg()
    tracker = _make_tracker(cfg)
    tracker.log_summary()  # must not raise even with all counters at 0


def test_hiera_similarity_without_shape_token():
    cfg = _make_cfg(cross_check_source="hiera")
    detector = _FakeDetector(use_shape_token=False)
    base = {
        "main": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),  # 2 refs, 1 token each (no shape token)
        "l1": torch.zeros(1, 2, 2), "l2": torch.zeros(1, 2, 2),
    }
    tracker = _make_tracker(cfg, detector=detector, base_prototype=base)

    new_tokens = {"main": torch.tensor([[[1.0, 0.0]]]), "l1": torch.zeros(1, 1, 2), "l2": torch.zeros(1, 1, 2)}
    sim = tracker._hiera_similarity(new_tokens)
    # ref mean = normalize([1,0]+[0,1])/2 = [0.5,0.5] normalized; candidate = [1,0] normalized.
    expected = float(np.array([1.0, 0.0]) @ (np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5])))
    assert sim == pytest.approx(expected, abs=1e-5)


def test_hiera_similarity_skips_shape_tokens_when_enabled():
    """With use_shape_token=True, base_prototype's main tensor interleaves
    [exemplar, shape] per ref -- only the exemplar (even-index) tokens
    should contribute to the reference vector."""
    cfg = _make_cfg(cross_check_source="hiera")
    detector = _FakeDetector(use_shape_token=True)
    # 1 ref: [exemplar=(1,0), shape=(99,99)] -- shape token has a wildly
    # different value so the test fails loudly if it leaks into the mean.
    base = {
        "main": torch.tensor([[[1.0, 0.0], [99.0, 99.0]]]),
        "l1": torch.zeros(1, 2, 2), "l2": torch.zeros(1, 2, 2),
    }
    tracker = _make_tracker(cfg, detector=detector, base_prototype=base)

    new_tokens = {"main": torch.tensor([[[1.0, 0.0]]]), "l1": torch.zeros(1, 1, 2), "l2": torch.zeros(1, 1, 2)}
    sim = tracker._hiera_similarity(new_tokens)
    assert sim == pytest.approx(1.0, abs=1e-5)  # candidate == the (only) exemplar token exactly
