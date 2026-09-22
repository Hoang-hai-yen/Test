"""BD-CSPN-style (Liu et al., ECCV 2020, arXiv:1911.10713 Eq. 5-6)
self-referential agreement weighting -- shared by
accuracy.cheap_boosters.multi_ref_pooling="agreement_weighted" (stage3.py's
_pool_sims and geco2_detector.py's _cosine_from_feature, weighting already-
computed per-reference SIMILARITY SCORES) so both call sites use exactly
the same weighting formula instead of drifting apart.

NOT the same code path as stage1.prototype.fusion="agreement_weighted"
(aero_eyes/stages/stage1.py::fuse_prototype), which weights raw EMBEDDINGS
before fusing them into one prototype vector and additionally combines the
result with a mask-area confidence weight that isn't available at this
(later, scoring-time) stage -- that one stays self-contained in stage1.py
rather than importing from here, since its own agreement direction is the
MASK-weighted mean, not the plain mean used below.
"""
from __future__ import annotations

import numpy as np


def agreement_weights(per_ref_array: np.ndarray, epsilon: float) -> np.ndarray:
    """Per-reference softmax weight from cosine agreement with the plain
    (unweighted) mean direction of ALL rows -- a reference whose own
    embedding disagrees with the consensus of the others gets a smaller
    weight. Rows need not be pre-normalized (normalized internally).
    Returns weights summing to 1, shape [num_refs].

    epsilon=0 makes every weight equal (reduces to a plain unweighted
    mean); higher epsilon downweights a disagreeing reference more
    aggressively (higher risk of overfitting to noise with very few refs).
    """
    row_norms = np.linalg.norm(per_ref_array, axis=1, keepdims=True).clip(min=1e-8)
    unit_refs = per_ref_array / row_norms
    mean_dir = unit_refs.mean(axis=0)
    mean_dir = mean_dir / max(np.linalg.norm(mean_dir), 1e-8)
    agreement = unit_refs @ mean_dir
    logits = epsilon * agreement
    weights = np.exp(logits - logits.max())  # numerically stable softmax
    return weights / weights.sum()
