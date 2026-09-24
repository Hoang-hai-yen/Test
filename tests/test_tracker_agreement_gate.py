import pytest

from aero_eyes.types import Box
from aero_eyes.utils.detection_confirm import TrackerAgreementGate

TRACK = Box(100, 100, 150, 150)
NEAR = Box(105, 105, 155, 155)   # IoU ~0.68 with TRACK
FAR = Box(400, 400, 450, 450)    # IoU 0


def test_agreement_accepts_immediately_in_both_modes():
    for mode in ("conf_compare", "hits"):
        g = TrackerAgreementGate(0.3, mode, 2)
        assert g.judge(TRACK, NEAR, 0.1) == "accept"


def test_conf_compare_keeps_track_when_anchor_higher_or_equal():
    g = TrackerAgreementGate(0.3, "conf_compare", 2)
    g.anchored(0.8)
    assert g.judge(TRACK, FAR, 0.7) == "keep_track"
    assert g.judge(TRACK, FAR, 0.8) == "keep_track"  # tie -> existing track


def test_conf_compare_replaces_when_detection_higher():
    g = TrackerAgreementGate(0.3, "conf_compare", 2)
    g.anchored(0.6)
    assert g.judge(TRACK, FAR, 0.9) == "replace"


def test_conf_compare_unknown_anchor_lets_detection_win():
    g = TrackerAgreementGate(0.3, "conf_compare", 2)
    g.anchored(None)
    assert g.judge(TRACK, FAR, 0.05) == "replace"


def test_hits_keeps_until_required_consecutive_mismatches():
    g = TrackerAgreementGate(0.3, "hits", 2)
    assert g.judge(TRACK, FAR, 0.9) == "keep_track"
    assert g.judge(TRACK, FAR, 0.9) == "replace"
    assert g.judge(TRACK, FAR, 0.9) == "keep_track"  # streak restarted after replace


def test_hits_streak_broken_by_agreement_or_gap():
    g = TrackerAgreementGate(0.3, "hits", 2)
    assert g.judge(TRACK, FAR, 0.9) == "keep_track"
    assert g.judge(TRACK, NEAR, 0.9) == "accept"
    assert g.judge(TRACK, FAR, 0.9) == "keep_track"
    g.reset_streak()
    assert g.judge(TRACK, FAR, 0.9) == "keep_track"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        TrackerAgreementGate(0.3, "nope", 2)
