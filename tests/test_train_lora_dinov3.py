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
    assert 0.0 <= res["best_val_auroc"] <= 1.0


def test_run_training_needs_two_objects(tmp_path):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 2 objects"):
        run_training(_FakeExt(), [_video("A_0", "A", (220, 30, 30), rng)], [], _args(tmp_path))
