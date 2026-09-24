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
