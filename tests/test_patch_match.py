import torch
import torch.nn.functional as F

from aero_eyes.config import PatchMatchingConfig, Stage3Config
from aero_eyes.models.patch_match import chamfer_score, sinkhorn_score


def _tokens(n: int, d: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_identical_sets_score_one():
    a = _tokens(20)
    assert abs(chamfer_score(a, a) - 1.0) < 1e-5
    assert sinkhorn_score(a, a, epsilon=0.02, iters=200) > 0.95


def test_unrelated_sets_score_lower_than_identical():
    a, b = _tokens(20, seed=0), _tokens(20, seed=1)
    assert chamfer_score(a, b) < chamfer_score(a, a)
    assert sinkhorn_score(a, b) < sinkhorn_score(a, a, epsilon=0.05)


def test_detail_patches_unmatched_by_blank_patches():
    """A 'card' = blank patches + detail patches; a 'paper' = only blank
    patches. CLS-style averaging barely separates them; symmetric chamfer
    and OT must penalise the unmatched detail."""
    blank = _tokens(1, seed=3).repeat(30, 1)
    detail = _tokens(10, seed=4)
    card = torch.cat([blank, detail])
    paper = blank
    assert chamfer_score(card, card) - chamfer_score(card, paper) > 0.05
    assert sinkhorn_score(card, card) - sinkhorn_score(card, paper) > 0.05


def test_chamfer_asymmetric_ignores_extra_cand_detail():
    blank = _tokens(1, seed=3).repeat(30, 1)
    detail = _tokens(10, seed=4)
    card = torch.cat([blank, detail])
    # ref=paper, cand=card: every ref patch finds a blank match
    assert chamfer_score(blank, card, symmetric=False) > chamfer_score(blank, card, symmetric=True)


def test_patch_matching_disabled_by_default():
    assert Stage3Config().patch_matching == PatchMatchingConfig()
    assert Stage3Config().patch_matching.enabled is False
