"""Unit tests for scripts/train_lora_dinov3.py: pure helpers plus a tiny
end-to-end run_training() on a fake extractor (no real DINOv3 / videos)."""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from torch import nn

from aero_eyes.types import Box
from scripts.train_lora_dinov3 import (
    VideoData, augment_crop, degrade_ref, even_subsample, object_id, prototype_bg_loss,
    run_training, separation_metrics, split_candidates, split_videos,
)

IDS = ["Backpack_0", "Backpack_1", "Laptop_0", "Laptop_1", "Person1_0", "Person1_1"]


def test_object_id_strips_only_the_trailing_video_index():
    assert object_id("Laptop_1") == "Laptop"
    assert object_id("Person1_0") == "Person1"
    assert object_id("Water_Bottle_12") == "Water_Bottle"
    assert object_id("NoIndex") == "NoIndex"


def test_split_default_trains_on_everything():
    assert split_videos(IDS) == (IDS, [])


def test_split_by_suffix_shares_objects_between_train_and_val():
    train, val = split_videos(IDS, val_suffix="_1")
    assert val == ["Backpack_1", "Laptop_1", "Person1_1"]
    assert {object_id(v) for v in train} == {object_id(v) for v in val}


def test_split_by_object_holds_out_all_videos_of_that_object():
    train, val = split_videos(IDS, val_objects=["Laptop"])
    assert val == ["Laptop_0", "Laptop_1"]
    assert not any(object_id(v) == "Laptop" for v in train)


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        split_videos(IDS, val_objects=["Laptop"], val_suffix="_1")
    with pytest.raises(ValueError):
        split_videos(IDS, val_objects=["Nope"])
    with pytest.raises(ValueError):
        split_videos(IDS, val_suffix="_9")
    with pytest.raises(ValueError):  # would leave nothing to train on
        split_videos(["A_0", "A_1"], val_objects=["A"])


def test_split_candidates_by_iou_against_gt():
    gt = {1: Box(0, 0, 10, 10)}
    det = lambda b: types.SimpleNamespace(box=b)
    cands = {
        1: [det(Box(0, 0, 10, 10)),        # IoU 1.0  -> extra positive
            det(Box(100, 100, 110, 110)),  # IoU 0    -> negative
            det(Box(0, 0, 10, 20))],       # IoU 0.5  -> ambiguous, dropped
        2: [det(Box(5, 5, 9, 9))],         # frame without GT -> negative
    }
    neg, pos = split_candidates(cands, gt, neg_iou_max=0.1, pos_iou_min=0.6)
    assert [f for f, _ in pos] == [1]
    assert sorted(f for f, _ in neg) == [1, 2]


def test_even_subsample_spreads_across_the_sequence():
    out = even_subsample(list(range(100)), 5)
    assert out[0] == 0 and out[-1] == 99 and len(out) == 5
    assert even_subsample([1, 2], 5) == [1, 2]


def test_loss_is_lower_for_correct_labels_than_swapped_labels():
    protos = torch.eye(3)
    crops = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    bg = torch.tensor(0.0)
    good = prototype_bg_loss(crops, torch.tensor([0, 1]), protos, bg, tau=0.1)
    swapped = prototype_bg_loss(crops, torch.tensor([1, 0]), protos, bg, tau=0.1)
    assert good < swapped


def test_loss_rewards_pushing_clutter_away_from_every_prototype():
    protos = torch.eye(3)
    clutter = torch.tensor([[0.577, 0.577, 0.577]])   # equally (un)like all prototypes
    bg = torch.tensor(10.0)                            # strong background logit
    as_clutter = prototype_bg_loss(clutter, torch.tensor([3]), protos, bg, tau=0.1)
    as_target = prototype_bg_loss(clutter, torch.tensor([0]), protos, bg, tau=0.1)
    assert as_clutter < as_target


def test_loss_averages_target_and_clutter_groups_separately():
    protos = torch.eye(2)
    crops = torch.tensor([[1.0, 0], [1.0, 0], [1.0, 0], [0.6, 0.8]])
    labels = torch.tensor([0, 2, 2, 2])  # 1 target, 3 clutter (K=2)
    bg = torch.tensor(0.0)
    both = prototype_bg_loss(crops, labels, protos, bg, 0.1)
    t = prototype_bg_loss(crops[:1], labels[:1], protos, bg, 0.1)
    c = prototype_bg_loss(crops[1:], labels[1:], protos, bg, 0.1)
    assert both.item() == pytest.approx(((t + c) / 2).item(), rel=1e-5)


def test_separation_metrics_perfect_and_empty():
    m = separation_metrics(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert m["auroc"] == 1.0 and m["pos_cos"] > m["neg_cos"]
    assert np.isnan(separation_metrics(np.zeros(0), np.array([0.1]))["auroc"])


def test_augment_and_degrade_keep_shape_dtype():
    img = np.random.default_rng(0).integers(0, 255, (40, 30, 3), dtype=np.uint8)
    out = augment_crop(img, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert degrade_ref(img, 1.0) is img
    assert not np.array_equal(degrade_ref(img, 0.2), img)


# ---------------------------------------------------------------------------
# End to end on a tiny fake extractor
# ---------------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = _Attn(d)


class _Net(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, d)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Layer(d) for _ in range(2)])

    def forward(self, pv):
        x = self.proj(pv.flatten(1))
        for layer in self.encoder.layer:
            x = x + layer.attention.q_proj(x) + layer.attention.v_proj(x)
        return x


class _FakeExt:
    device = "cpu"
    preprocess_mode = candidate_preprocess_mode = "stretch"

    def __init__(self):
        torch.manual_seed(0)
        self.model = _Net()

    def pixel_values(self, images, mode=None):
        import cv2
        return torch.stack([
            torch.from_numpy(cv2.resize(im, (8, 8)).astype(np.float32) / 255.0).permute(2, 0, 1) for im in images
        ])

    def forward_cls(self, pv):
        return self.model(pv)


def _noisy(color, rng, noise):
    return np.clip(np.array(color, np.float32) + rng.normal(0, noise, (12, 12, 3)), 0, 255).astype(np.uint8)


def _video(vid, obj, color, rng):
    other_colors = [rng.integers(0, 255, 3) for _ in range(8)]
    return VideoData(
        video_id=vid, obj=obj,
        refs=[_noisy(color, rng, 1) for _ in range(3)],
        pos=[_noisy(color, rng, 15) for _ in range(20)],
        neg=[_noisy(c, rng, 15) for c in other_colors for _ in range(3)],
    )


def _args(tmp_path):
    return types.SimpleNamespace(
        seed=0, targets=["q_proj", "v_proj"], rank=4, alpha=8.0, last_n_blocks=None, bg_init_cos=0.3, tau=0.1,
        lr=5e-3, weight_decay=0.0, out_dir=str(tmp_path), epochs=3, steps_per_epoch=25, refs_per_object=3,
        pos_per_object=6, neg_per_object=6, ref_factors=[1.0], eval_ref_factor=1.0, micro_batch=16, amp=False,
    )


def test_run_training_reduces_loss_saves_checkpoint_and_reports_val(tmp_path):
    rng = np.random.default_rng(0)
    train = [_video("A_0", "A", (220, 30, 30), rng), _video("B_0", "B", (30, 200, 40), rng)]
    val = [_video("A_1", "A", (220, 30, 30), rng)]
    res = run_training(_FakeExt(), train, val, _args(tmp_path))
    losses = [h["train_loss"] for h in res["history"] if h["train_loss"] is not None]
    assert losses[-1] < losses[0]
    assert (tmp_path / "lora_last.pt").exists() and (tmp_path / "metrics.json").exists()
    assert res["history"][0]["epoch"] == 0 and "mean_auroc" in res["history"][0]
    assert 0.0 <= res["best_val_score"] <= 1.0


def test_run_training_needs_two_objects(tmp_path):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 2 objects"):
        run_training(_FakeExt(), [_video("A_0", "A", (220, 30, 30), rng)], [], _args(tmp_path))


def test_jitter_box_stays_a_correct_but_imperfect_detection():
    from aero_eyes.utils.geometry import box_iou
    from scripts.train_lora_dinov3 import jitter_box

    rng = np.random.default_rng(0)
    gt = Box(100, 100, 200, 180)
    jittered = [jitter_box(gt, rng, min_iou=0.5) for _ in range(50)]
    assert all(b is not None and box_iou(b, gt) >= 0.5 for b in jittered)
    assert any(b.x1 != gt.x1 or b.y2 != gt.y2 for b in jittered)      # actually perturbed
    assert jitter_box(gt, rng, min_iou=1.01) is None                   # impossible constraint


# ---------------------------------------------------------------------------
# Pushing clutter away: TPR@FPR, cosine hinge, hard-negative mining
# ---------------------------------------------------------------------------

def test_tpr_at_low_fpr_sees_the_clutter_tail_that_auroc_hides():
    pos = np.full(50, 0.9)
    neg = np.array([0.1] * 99 + [0.95])            # one clutter crop scores ABOVE every target
    m = separation_metrics(pos, neg)
    assert m["auroc"] > 0.98                        # AUROC still looks excellent
    assert m["tpr_fpr1pct"] == 1.0                  # 1 leaked clutter of 100 is allowed at FPR 1%
    assert m["tpr_fpr0p1pct"] == 0.0                # ...but zero leak allowed loses every target


def test_tpr_at_low_fpr_is_one_when_cleanly_separable():
    m = separation_metrics(np.array([0.8, 0.9]), np.array([0.1, 0.2, 0.3]))
    assert m["tpr_fpr1pct"] == 1.0 and m["tpr_fpr0p1pct"] == 1.0


def test_neg_margin_hinge_penalizes_clutter_close_to_its_own_prototype():
    protos = torch.eye(2)
    crops = torch.tensor([[1.0, 0.0]])             # cosine 1.0 with prototype 0
    labels = torch.tensor([2])                     # clutter (K=2)
    owner = torch.tensor([0])
    bg = torch.tensor(0.0)
    base = prototype_bg_loss(crops, labels, protos, bg, 0.1)
    with_margin = prototype_bg_loss(crops, labels, protos, bg, 0.1, owner=owner, neg_margin=0.2, margin_weight=2.0)
    assert with_margin.item() == pytest.approx(base.item() + 2.0 * (1.0 - 0.2), rel=1e-5)


def test_neg_margin_hinge_is_zero_for_clutter_already_below_the_margin():
    protos = torch.eye(2)
    crops = torch.tensor([[0.0, 1.0]])             # orthogonal to prototype 0 -> cosine 0
    labels, owner, bg = torch.tensor([2]), torch.tensor([0]), torch.tensor(0.0)
    base = prototype_bg_loss(crops, labels, protos, bg, 0.1)
    with_margin = prototype_bg_loss(crops, labels, protos, bg, 0.1, owner=owner, neg_margin=0.2)
    assert with_margin.item() == pytest.approx(base.item(), rel=1e-6)


def test_neg_margin_ignores_target_crops():
    protos = torch.eye(2)
    crops = torch.tensor([[1.0, 0.0]])
    labels, owner, bg = torch.tensor([0]), torch.tensor([0]), torch.tensor(0.0)   # a TARGET crop
    base = prototype_bg_loss(crops, labels, protos, bg, 0.1)
    with_margin = prototype_bg_loss(crops, labels, protos, bg, 0.1, owner=owner, neg_margin=0.0)
    assert with_margin.item() == pytest.approx(base.item(), rel=1e-6)


def test_mine_hard_negatives_keeps_the_clutter_most_similar_to_the_prototype():
    from scripts.train_lora_dinov3 import mine_hard_negatives, embed

    rng = np.random.default_rng(0)
    ext = _FakeExt()
    red = (220, 30, 30)
    proto = torch.nn.functional.normalize(embed(ext, [_noisy(red, rng, 1)] * 3, "stretch", 8, False).mean(0), dim=0)
    close = [_noisy((200, 50, 40), rng, 3) for _ in range(3)]        # reddish clutter -> hard
    far = [_noisy((30, 40, 210), rng, 3) for _ in range(6)]          # blue clutter -> easy
    picked = mine_hard_negatives(ext, far + close, proto, n=3, mode="stretch", micro_batch=8, amp=False)
    assert all(any(p is c for c in close) for p in picked)


def test_run_training_with_mining_margin_and_tpr_selection(tmp_path):
    rng = np.random.default_rng(0)
    train = [_video("A_0", "A", (220, 30, 30), rng), _video("B_0", "B", (30, 200, 40), rng)]
    val = [_video("A_1", "A", (220, 30, 30), rng)]
    args = _args(tmp_path)
    args.hard_neg_pool, args.neg_margin, args.neg_margin_weight, args.select_by = 12, 0.1, 1.0, "tpr_fpr1pct"
    res = run_training(_FakeExt(), train, val, args)
    assert "mean_tpr_fpr1pct" in res["history"][0] and "mean_tpr_fpr1pct" in res["history"][-1]
    assert 0.0 <= res["best_val_score"] <= 1.0
    assert (tmp_path / "lora_last.pt").exists()


# ---------------------------------------------------------------------------
# Triplet loss and patch-token (Chamfer/OT) scoring
# ---------------------------------------------------------------------------

def test_score_bg_loss_matches_prototype_bg_loss():
    from scripts.train_lora_dinov3 import score_bg_loss

    protos = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    crops = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    labels, owner, bg = torch.tensor([0, 1, 2, 3, 3]), torch.tensor([0, 1, 2, 0, 1]), torch.tensor(0.2)
    a = prototype_bg_loss(crops, labels, protos, bg, 0.1, owner=owner, neg_margin=0.1)
    b = score_bg_loss(crops @ protos.t(), labels, bg, 0.1, owner=owner, neg_margin=0.1)
    assert a.item() == pytest.approx(b.item(), rel=1e-6)


def test_triplet_zero_when_margin_satisfied_and_no_gradient():
    from scripts.train_lora_dinov3 import triplet_loss

    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.2, 0.2]], requires_grad=True)   # 2 targets + 1 clutter (owner 0)
    labels, owner = torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0])
    loss = triplet_loss(scores, labels, owner, margin=0.1)
    loss.backward()
    assert loss.item() == 0.0 and float(scores.grad.abs().sum()) == 0.0


def test_triplet_penalizes_clutter_scoring_near_the_target_and_hard_mining_picks_the_worst():
    from scripts.train_lora_dinov3 import triplet_loss

    # object 0: target 0.6; clutter at 0.55 (violates margin 0.1 by 0.05), 0.62 (by 0.12) and 0.1 (fine)
    scores = torch.tensor([[0.6, 0.0], [0.55, 0.0], [0.62, 0.0], [0.1, 0.0], [0.0, 0.9]])
    labels, owner = torch.tensor([0, 2, 2, 2, 1]), torch.tensor([0, 0, 0, 0, 1])
    allp = triplet_loss(scores, labels, owner, margin=0.1, mining="all")
    hard = triplet_loss(scores, labels, owner, margin=0.1, mining="hard")
    assert allp.item() == pytest.approx((0.05 + 0.12) / 2, abs=1e-6)     # mean over the 2 violating pairs
    assert hard.item() == pytest.approx(0.12, abs=1e-6)                 # only the hardest negative counts
    with pytest.raises(ValueError):
        triplet_loss(scores, labels, owner, mining="nope")


def test_triplet_without_negatives_is_a_zero_loss_that_can_backprop():
    from scripts.train_lora_dinov3 import triplet_loss

    scores = torch.tensor([[0.5, 0.5]], requires_grad=True)
    loss = triplet_loss(scores, torch.tensor([0]), torch.tensor([0]))
    loss.backward()
    assert loss.item() == 0.0


def test_patch_pair_scores_match_single_pair_functions_and_are_differentiable():
    from aero_eyes.models.patch_match import chamfer_score, patch_pair_scores, sinkhorn_score

    g = torch.Generator().manual_seed(0)
    cand = torch.nn.functional.normalize(torch.randn(3, 10, 16, generator=g), dim=-1).requires_grad_(True)
    ref = torch.nn.functional.normalize(torch.randn(2, 12, 16, generator=g), dim=-1)
    ch = patch_pair_scores(cand, ref, "chamfer", chunk=2)
    ot = patch_pair_scores(cand, ref, "ot", epsilon=0.05, iters=50, chunk=2)
    assert ch.shape == ot.shape == (3, 2)
    for c in range(3):
        for r in range(2):
            assert ch[c, r].item() == pytest.approx(chamfer_score(ref[r], cand[c].detach()), abs=1e-5)
            assert ot[c, r].item() == pytest.approx(sinkhorn_score(ref[r], cand[c].detach(), 0.05, 50), abs=1e-4)
    one_way = patch_pair_scores(cand, ref, "chamfer", symmetric=False)
    assert one_way[0, 0].item() == pytest.approx(chamfer_score(ref[0], cand[0].detach(), symmetric=False), abs=1e-5)
    (ch.sum() + ot.sum()).backward()
    assert cand.grad is not None and float(cand.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        patch_pair_scores(cand, ref, "nope")


class _FakePatchExt(_FakeExt):
    """_FakeExt plus forward_tokens: 4 'patches' = the four quadrants of the 8x8 input."""

    def forward_tokens(self, pv, layers=(-1,)):
        toks = []
        for rows in (slice(0, 4), slice(4, 8)):
            for cols in (slice(0, 4), slice(4, 8)):
                m = torch.zeros_like(pv)
                m[:, :, rows, cols] = 1.0
                toks.append(self.model(pv * m))
        tokens = torch.nn.functional.normalize(torch.stack(toks, dim=1), dim=-1)
        return self.model(pv), tokens


@pytest.mark.parametrize("score,method,loss", [
    ("patch", "chamfer", "ce"), ("patch", "ot", "triplet"), ("both", "chamfer", "triplet"),
])
def test_run_training_with_patch_scores(tmp_path, score, method, loss):
    rng = np.random.default_rng(0)
    train = [_video("A_0", "A", (220, 30, 30), rng), _video("B_0", "B", (30, 200, 40), rng)]
    val = [_video("A_1", "A", (220, 30, 30), rng)]
    args = _args(tmp_path)
    args.score, args.patch_method, args.loss = score, method, loss
    args.patch_layers, args.patch_ot_iters, args.cls_weight, args.triplet_margin = [-1], 10, 0.5, 0.05
    args.epochs, args.steps_per_epoch = 2, 6
    res = run_training(_FakePatchExt(), train, val, args)
    assert np.isfinite([h["train_loss"] for h in res["history"] if h["train_loss"] is not None]).all()
    assert 0.0 <= res["history"][0]["mean_auroc"] <= 1.0
    assert (tmp_path / "lora_last.pt").exists()


# ---------------------------------------------------------------------------
# Not comparing pad_to_square padding in patch mode
# ---------------------------------------------------------------------------

def test_pad_patch_mask_matches_the_actual_padding():
    from PIL import Image

    from aero_eyes.models.features import _resize_and_pad_to_square, pad_patch_mask

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))       # 300x200 (3:2)
    mask = pad_patch_mask(300, 200, 224, 16).reshape(14, 14)
    assert mask.sum() == 10 * 14                                    # rows 2..11 are content (34-186 px)
    assert not mask[:2].any() and not mask[12:].any() and mask[2:12].all()
    canvas = np.array(_resize_and_pad_to_square(img, 224))
    fill = canvas[0, 0]
    for row in (0, 1, 12, 13):                                      # fully-padded patch rows really are flat fill
        assert (canvas[row * 16:(row + 1) * 16] == fill).all()
    assert pad_patch_mask(300, 200, 224, 16, min_content_frac=0.7).sum() == 8 * 14   # edge patches (69% / 63%) drop out
    assert pad_patch_mask(224, 224, 224, 16).all()                  # a square image has no padding
    assert (pad_patch_mask(200, 300, 224, 16).reshape(14, 14).sum(1) == 10).all()             # portrait: pad on the sides


def test_masked_patch_scores_equal_scores_on_the_valid_patches_only_and_have_no_gradient_on_masked_ones():
    from aero_eyes.models.patch_match import chamfer_score, patch_pair_scores, sinkhorn_score

    g = torch.Generator().manual_seed(0)
    norm = lambda t: torch.nn.functional.normalize(t, dim=-1)
    cand = norm(torch.randn(2, 10, 16, generator=g)).requires_grad_(True)
    ref = norm(torch.randn(2, 12, 16, generator=g))
    cmask = torch.ones(2, 10, dtype=torch.bool); cmask[:, 6:] = False
    rmask = torch.ones(2, 12, dtype=torch.bool); rmask[0, 8:] = False
    for method, sym in (("chamfer", True), ("chamfer", False), ("ot", True)):
        got = patch_pair_scores(cand, ref, method, sym, 0.05, 50, chunk=1, cand_mask=cmask, ref_mask=rmask)
        for c in range(2):
            for r in range(2):
                cv, rv = cand[c][cmask[c]].detach(), ref[r][rmask[r]]
                want = sinkhorn_score(rv, cv, 0.05, 50) if method == "ot" else chamfer_score(rv, cv, sym)
                assert got[c, r].item() == pytest.approx(want, abs=1e-4), (method, sym, c, r)
    patch_pair_scores(cand, ref, "chamfer", cand_mask=cmask, ref_mask=rmask).sum().backward()
    assert float(cand.grad[:, 6:].abs().sum()) == 0.0 and float(cand.grad[:, :6].abs().sum()) > 0


class _FakeMaskExt(_FakePatchExt):
    """Adds pixel_values_and_mask: the 4th 'patch' (bottom-right quadrant) is padding."""

    def pixel_values_and_mask(self, images, mode=None, min_content_frac=0.5):
        mask = torch.ones(len(images), 4, dtype=torch.bool)
        mask[:, 3] = False
        return self.pixel_values(images, mode), mask


@pytest.mark.parametrize("method", ["chamfer", "ot"])
def test_run_training_masks_padding_when_asked(tmp_path, method):
    rng = np.random.default_rng(0)
    train = [_video("A_0", "A", (220, 30, 30), rng), _video("B_0", "B", (30, 200, 40), rng)]
    val = [_video("A_1", "A", (220, 30, 30), rng)]
    args = _args(tmp_path)
    args.score, args.patch_method, args.patch_ot_iters, args.epochs, args.steps_per_epoch = "patch", method, 8, 1, 4
    seen = []
    ext = _FakeMaskExt()
    orig = ext.pixel_values_and_mask
    ext.pixel_values_and_mask = lambda *a, **k: (seen.append(a[1]), orig(*a, **k))[1]
    args.patch_mask_padding = True
    res = run_training(ext, train, val, args)
    assert seen, "pixel_values_and_mask must be used when patch_mask_padding is on"
    assert np.isfinite(res["history"][-1]["train_loss"])
    seen.clear()
    args.patch_mask_padding = False
    run_training(_FakeMaskExt(), train, val, args)
    assert not seen


def test_patch_matching_config_masks_padding_by_default():
    from aero_eyes.config import PatchMatchingConfig

    c = PatchMatchingConfig()
    assert c.mask_padding is True and c.pad_min_content_frac == 0.5


def test_very_thin_crop_never_gets_an_empty_patch_mask():
    from aero_eyes.models.features import pad_patch_mask

    mask = pad_patch_mask(60, 3, 224, 16)          # 60x3 px crop -> ~11 px of content: under 50% of every patch
    assert mask.any()
    assert mask.sum() < mask.size                  # still mostly padding


def test_patch_scores_stay_finite_when_an_image_has_no_valid_patch():
    from aero_eyes.models.patch_match import patch_pair_scores

    g = torch.Generator().manual_seed(0)
    norm = lambda t: torch.nn.functional.normalize(t, dim=-1)
    cand, ref = norm(torch.randn(2, 10, 16, generator=g)), norm(torch.randn(2, 12, 16, generator=g))
    empty_c = torch.zeros(2, 10, dtype=torch.bool)
    empty_r = torch.zeros(2, 12, dtype=torch.bool)
    for method in ("chamfer", "ot"):
        got = patch_pair_scores(cand, ref, method, cand_mask=empty_c, ref_mask=empty_r)
        assert torch.isfinite(got).all()
        assert torch.allclose(got, patch_pair_scores(cand, ref, method), atol=1e-5)   # falls back to comparing everything
