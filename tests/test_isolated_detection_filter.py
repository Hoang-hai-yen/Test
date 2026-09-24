import random
from types import SimpleNamespace

from aero_eyes.stages.stage3 import IsolatedKeyframeGate, find_isolated_keyframes


def _cfg(mode="offline", gap=2, conf=None):
    return SimpleNamespace(max_gap_intervals=gap, keep_conf_threshold=conf, mode=mode)


def test_examples():
    c = _cfg()
    assert find_isolated_keyframes({0: .5, 15: .5}, 8, c) == set()
    assert find_isolated_keyframes({0: .5, 40: .5}, 8, c) == {0, 40}
    assert find_isolated_keyframes({0: .5, 40: .5, 48: .5}, 8, c) == {0}
    assert find_isolated_keyframes({0: .5, 40: .9}, 8, _cfg(conf=.8)) == {0}


def test_online_matches_offline():
    rng = random.Random(0)
    for _ in range(200):
        frames = sorted(rng.sample(range(0, 400, 8), rng.randint(0, 20)))
        scores = {f: rng.random() for f in frames}
        conf = rng.choice([None, 0.7])
        gap = rng.randint(1, 4)
        assert find_isolated_keyframes(scores, 8, _cfg("offline", gap, conf)) == \
            find_isolated_keyframes(scores, 8, _cfg("online", gap, conf))


def test_gate_releases_lone_keyframe_after_max_gap():
    gate = IsolatedKeyframeGate(8, _cfg())
    assert gate.push(0, .5) == []
    assert gate.advance(8) == [] and gate.advance(16) == []
    assert gate.advance(24) == [(0, False)]  # 24 - 0 > 16, no detection arrived
