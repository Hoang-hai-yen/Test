"""Unit tests for aero_eyes.utils.ref_agreement.agreement_weights -- shared
BD-CSPN-style softmax reweighting used by accuracy.cheap_boosters.
multi_ref_pooling="agreement_weighted" (stage3.py's _pool_sims and
geco2_detector.py's _cosine_from_feature). Pure numpy, no torch needed.
"""
from __future__ import annotations

import numpy as np

from aero_eyes.utils.ref_agreement import agreement_weights


def test_agreement_weights_sum_to_one():
    refs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    w = agreement_weights(refs, epsilon=5.0)
    assert np.isclose(w.sum(), 1.0)


def test_agreement_weights_epsilon_zero_is_uniform():
    refs = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, -5.0]])
    w = agreement_weights(refs, epsilon=0.0)
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_agreement_weights_downweights_outlier():
    consensus = np.array([1.0, 0.0, 0.0])
    outlier = np.array([0.0, 1.0, 0.0])
    refs = np.stack([consensus, consensus * 0.95, outlier])
    w = agreement_weights(refs, epsilon=20.0)
    assert w[2] < w[0]
    assert w[2] < w[1]


def test_agreement_weights_identical_refs_uniform():
    refs = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
    w = agreement_weights(refs, epsilon=50.0)
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_agreement_weights_unnormalized_rows_still_correct():
    """Rows with very different norms but the SAME direction must still be
    treated as fully agreeing (normalized internally before comparing)."""
    refs = np.array([[1.0, 0.0], [100.0, 0.0], [0.0, 1.0]])
    w = agreement_weights(refs, epsilon=20.0)
    assert np.isclose(w[0], w[1], atol=1e-6)  # same direction -> same weight regardless of norm
    assert w[2] < w[0]
