"""Tests for aero_eyes/models/geco2_finetune_data.py -- the GeCo2 finetune
dataset loader (see docs/GECO2_FINETUNE_PLAN.md).

Scope, stated explicitly: everything here runs CPU-only, offline, with no
GeCo2 checkpoint and no CUDA ops extension. It covers video-ID bookkeeping
(train/val split, test-leakage guard), present/absent frame-pool
construction, the two-level sampler's statistics, reference-image domain
randomization, and (where the local environment allows) the shared GT
coordinate-conversion function.

It does NOT and CANNOT cover anything in aero_eyes/models/geco2_train_wrapper.py
-- models.counter.CNT, build_model, MultiScaleDeformableAttention, the SAM2
backbone, real RoI-Align against backbone features, or an actual
.backward() call. That requires a GPU with the compiled CUDA ops extension
and the real GeCo2 checkpoint; full validation of the training wrapper and
loop happens only via notebooks/train_geco2_aeroeyes_vastai.ipynb's
--dry-run cell on a rented GPU box. Do not read broader coverage into this
file than what's here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy not importable (DLL blocked?)", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.models.geco2_finetune_data import (  # noqa: E402
    DEFAULT_HOLDOUT_CATEGORIES,
    EXPECTED_TRAIN_VIDEOS,
    HELD_OUT_TEST_VIDEOS,
    Geco2FinetuneDataset,
    RefImageCache,
    assert_no_test_leakage,
    build_present_absent_pools,
    jitter_box,
    sample_ref_downscale_factor,
    split_train_val,
    validate_training_video_ids,
    video_category,
)
from aero_eyes.types import Box  # noqa: E402

FIXTURE_ID = "synth001"
# scripts/make_synthetic_fixture.py::_make_synthetic_video defaults: 30
# total frames, object present in frames [obj_enter, obj_exit) = [5, 26),
# i.e. frames 5-25 inclusive are present, 0-4 and 26-29 are absent.
FIXTURE_PRESENT_FRAMES = set(range(5, 26))
FIXTURE_ABSENT_FRAMES = set(range(30)) - FIXTURE_PRESENT_FRAMES


@pytest.fixture(scope="session")
def synth_fixture(tmp_path_factory):
    """Build the synthetic fixture once per test session."""
    out_dir = tmp_path_factory.mktemp("geco2_finetune_fixtures")
    from scripts.make_synthetic_fixture import make_fixture
    make_fixture(out_dir, FIXTURE_ID)
    return out_dir


@pytest.fixture
def cfg(synth_fixture, tmp_path):
    """Minimal AeroEyesConfig pointing at the synth fixture, with MobileSAM
    segmentation disabled -- these tests are about the dataset/sampling
    logic, not segmentation, and disabling it avoids any chance of a
    mobile_sam import/network dependency making the test non-hermetic.
    """
    from aero_eyes.config import AeroEyesConfig, DataConfig, GTConfig, ProjectConfig
    from aero_eyes.config import SegmentationConfig, Stage123Geco2Config

    return AeroEyesConfig(
        project=ProjectConfig(work_dir=str(tmp_path / "runs"), use_cache=False, seed=42),
        data=DataConfig(
            data_root=str(synth_fixture),
            refs_subdir="refs",
            video_glob="*.mp4",
            num_references=3,
            gt=GTConfig(global_file=str(synth_fixture / FIXTURE_ID / "gt.json")),
        ),
        stage123_geco2=Stage123Geco2Config(
            segmentation=SegmentationConfig(enabled=False),
        ),
    )


# ---------------------------------------------------------------------------
# video_category / split_train_val / assert_no_test_leakage
# ---------------------------------------------------------------------------

def test_video_category():
    assert video_category("Backpack_0") == "Backpack"
    assert video_category("Backpack_1") == "Backpack"
    assert video_category("Person1_0") == "Person1"
    assert video_category("MobilePhone_1") == "MobilePhone"


def test_assert_no_test_leakage_passes_on_training_ids():
    assert_no_test_leakage(sorted(EXPECTED_TRAIN_VIDEOS))  # must not raise


def test_assert_no_test_leakage_raises_on_test_id():
    with pytest.raises(ValueError, match="held-out TEST"):
        assert_no_test_leakage(["BlackBox_0", "Backpack_0"])


def test_validate_training_video_ids_raises_on_unexpected_set():
    with pytest.raises(ValueError, match="does not match the expected 14"):
        validate_training_video_ids(["Foo_0", "Foo_1"])


def test_validate_training_video_ids_passes_on_expected_set():
    validate_training_video_ids(sorted(EXPECTED_TRAIN_VIDEOS))  # must not raise


def test_split_train_val_default_holdout():
    train_ids, val_ids = split_train_val(sorted(EXPECTED_TRAIN_VIDEOS))
    assert set(val_ids) == {"Lifering_0", "Lifering_1", "Person1_0", "Person1_1"}
    assert set(train_ids) == EXPECTED_TRAIN_VIDEOS - set(val_ids)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == EXPECTED_TRAIN_VIDEOS
    # Every held-out category is entirely absent from train, and vice versa.
    assert {video_category(v) for v in val_ids} == set(DEFAULT_HOLDOUT_CATEGORIES)
    assert not ({video_category(v) for v in train_ids} & set(DEFAULT_HOLDOUT_CATEGORIES))


def test_split_train_val_custom_holdout():
    train_ids, val_ids = split_train_val(sorted(EXPECTED_TRAIN_VIDEOS), holdout_categories=("Jacket",))
    assert set(val_ids) == {"Jacket_0", "Jacket_1"}
    assert "Jacket_0" not in train_ids and "Jacket_1" not in train_ids


def test_split_train_val_raises_on_wrong_video_set():
    with pytest.raises(ValueError, match="does not match the expected 14"):
        split_train_val(["Backpack_0", "Backpack_1"])  # missing 12 videos


def test_split_train_val_raises_on_test_leakage():
    corrupted = sorted(EXPECTED_TRAIN_VIDEOS - {"Backpack_0"}) + ["BlackBox_0"]
    with pytest.raises(ValueError, match="held-out TEST"):
        split_train_val(corrupted)


# ---------------------------------------------------------------------------
# build_present_absent_pools
# ---------------------------------------------------------------------------

def test_build_present_absent_pools_against_synth_fixture(synth_fixture):
    from aero_eyes.utils.io import load_gt

    gt_path = synth_fixture / FIXTURE_ID / "gt.json"
    gt = load_gt(gt_path, FIXTURE_ID)
    video_path = synth_fixture / FIXTURE_ID / "video.mp4"

    present, absent, total = build_present_absent_pools(video_path, gt)

    assert total == 30
    assert set(present) == FIXTURE_PRESENT_FRAMES
    assert set(absent) == FIXTURE_ABSENT_FRAMES
    assert set(present).isdisjoint(absent)
    assert set(present) | set(absent) == set(range(total))


# ---------------------------------------------------------------------------
# sample_ref_downscale_factor
# ---------------------------------------------------------------------------

def test_sample_ref_downscale_factor_range():
    rng = np.random.default_rng(0)
    samples = [sample_ref_downscale_factor(rng, lo=0.03, hi=1.0) for _ in range(500)]
    assert all(0.03 <= s <= 1.0 for s in samples)
    # Log-uniform: mean of log(samples) should land near the midpoint of
    # [log(lo), log(hi)], not near the midpoint of [lo, hi] (which a
    # linear-uniform sampler would instead produce, biased toward 1.0).
    log_samples = np.log(samples)
    expected_log_mid = (np.log(0.03) + np.log(1.0)) / 2
    assert log_samples.mean() == pytest.approx(expected_log_mid, abs=0.3)


def test_sample_ref_downscale_factor_rejects_bad_range():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_ref_downscale_factor(rng, lo=0.0, hi=1.0)
    with pytest.raises(ValueError):
        sample_ref_downscale_factor(rng, lo=0.5, hi=0.1)


# ---------------------------------------------------------------------------
# convert_gt_box_to_canvas -- requires GECO2/utils/data.py importable
# (pycocotools), guarded/skipped if unavailable in this environment.
# ---------------------------------------------------------------------------

def test_convert_gt_box_to_canvas_scale():
    pytest.importorskip("torch")
    from aero_eyes.models.geco2_detector import _ensure_geco2_on_path
    from aero_eyes.models.geco2_finetune_data import convert_gt_box_to_canvas

    geco2_repo = Path(__file__).resolve().parent.parent / "GECO2"
    _ensure_geco2_on_path(str(geco2_repo))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gt_box = Box(x1=100.0, y1=50.0, x2=200.0, y2=150.0)
    try:
        _, box_canvas, scale = convert_gt_box_to_canvas(frame, gt_box, image_size=1024.0)
    except ImportError as e:
        pytest.skip(
            f"GECO2/utils/data.py not importable in this environment ({e}) -- likely "
            "missing pycocotools, a GECO2-internal dependency unrelated to this "
            "function's own coordinate-conversion logic."
        )
        return

    # 640x480 video, image_size=1024 -> scaling_factor = 1024 / max(480,640) = 1.6 exactly.
    assert scale == pytest.approx(1024.0 / 640.0)
    expected = (100.0 * scale, 50.0 * scale, 200.0 * scale, 150.0 * scale)
    assert box_canvas == pytest.approx(expected)


def test_convert_gt_box_to_canvas_none_box():
    pytest.importorskip("torch")
    from aero_eyes.models.geco2_detector import _ensure_geco2_on_path
    from aero_eyes.models.geco2_finetune_data import convert_gt_box_to_canvas

    geco2_repo = Path(__file__).resolve().parent.parent / "GECO2"
    _ensure_geco2_on_path(str(geco2_repo))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        padded, box_canvas, scale = convert_gt_box_to_canvas(frame, None, image_size=1024.0)
    except ImportError as e:
        pytest.skip(f"GECO2/utils/data.py not importable in this environment ({e})")
        return

    assert box_canvas is None
    assert padded.shape[-2:] == (1024, 1024)


# ---------------------------------------------------------------------------
# RefImageCache + Geco2FinetuneDataset -- statistical sanity checks
# ---------------------------------------------------------------------------

def test_ref_image_cache_loads_native_refs(cfg):
    cache = RefImageCache(cfg, [FIXTURE_ID])
    imgs, boxes = cache.get(FIXTURE_ID)
    assert len(imgs) == 3
    assert boxes == [None, None, None]  # segmentation disabled in `cfg` fixture
    for img in imgs:
        assert img.shape[:2] == (224, 224)  # native ref image resolution from make_synthetic_fixture.py


def test_dataset_present_absent_sampler_statistics(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=400, p_present=0.5, seed=123,
    )
    n_present = n_absent = 0
    for i in range(len(ds)):
        sample = ds[i]
        assert sample.video_id == FIXTURE_ID
        if sample.is_present:
            n_present += 1
            assert sample.frame_idx in FIXTURE_PRESENT_FRAMES
            assert sample.gt_box is not None
        else:
            n_absent += 1
            assert sample.frame_idx in FIXTURE_ABSENT_FRAMES
            assert sample.gt_box is None
        assert len(sample.ref_images) == 3
        assert len(sample.ref_boxes) == 3

    frac_present = n_present / (n_present + n_absent)
    # Loose band around p_present=0.5 -- this is a statistical sanity check,
    # not an exact-count assertion.
    assert 0.35 <= frac_present <= 0.65
    assert ds.present_count == n_present
    assert ds.absent_count == n_absent


def test_dataset_ref_images_are_downscaled_per_step(cfg):
    """Each ref image should be resized (usually shrunk) independently per
    step -- confirms _apply_ref_downscale is actually being exercised with a
    freshly-sampled factor, not a fixed one, per docs/GECO2_FINETUNE_PLAN.md
    point 4."""
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=20,
        ref_downscale_range=(0.03, 0.5),  # force visible shrinkage every step
        seed=7,
    )
    native_h, native_w = 224, 224
    shapes_seen = set()
    for i in range(len(ds)):
        sample = ds[i]
        for img in sample.ref_images:
            h, w = img.shape[:2]
            assert h <= native_h and w <= native_w
            shapes_seen.add((h, w))
    # With hi=0.5, the native 224x224 ref should never survive un-shrunk,
    # and different steps should produce different shrunk sizes (fresh
    # per-step sampling, not a single fixed factor).
    assert all(h < native_h for h, w in shapes_seen)
    assert len(shapes_seen) > 1


def test_dataset_requires_nonempty_video_ids(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    with pytest.raises(ValueError):
        Geco2FinetuneDataset(cfg, [], ref_cache, steps_per_epoch=10)


# ---------------------------------------------------------------------------
# jitter_box
# ---------------------------------------------------------------------------

def test_jitter_box_noop_at_zero():
    rng = np.random.default_rng(0)
    box = (10.0, 20.0, 30.0, 50.0)
    assert jitter_box(rng, box, 0.0) == box


def test_jitter_box_perturbs_within_expected_bounds():
    rng = np.random.default_rng(0)
    box = (100.0, 100.0, 200.0, 160.0)  # w=100, h=60
    for _ in range(200):
        x1, y1, x2, y2 = jitter_box(rng, box, jitter=0.2)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Scale within [0.8, 1.2] of original.
        assert 100.0 * 0.8 <= w <= 100.0 * 1.2 + 1e-6
        assert 60.0 * 0.8 <= h <= 60.0 * 1.2 + 1e-6
        # Center shifted by at most 0.2 * original w/h.
        assert abs(cx - 150.0) <= 100.0 * 0.2 + 1e-6
        assert abs(cy - 130.0) <= 60.0 * 0.2 + 1e-6


# ---------------------------------------------------------------------------
# Geco2FinetuneDataset.max_dynamic_exemplars
# ---------------------------------------------------------------------------

def test_dataset_dynamic_exemplars_off_by_default(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=20, seed=1)
    for i in range(len(ds)):
        sample = ds[i]
        assert len(sample.ref_images) == 3
        assert len(sample.ref_boxes) == 3
        assert sample.num_dynamic_exemplars == 0
    assert ds.dynamic_exemplar_count == 0


def test_dataset_dynamic_exemplars_adds_extra_when_enabled(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=60, max_dynamic_exemplars=2, seed=3,
    )
    counts_seen = set()
    running_total = 0
    for i in range(len(ds)):
        sample = ds[i]
        assert 0 <= sample.num_dynamic_exemplars <= 2
        assert len(sample.ref_images) == 3 + sample.num_dynamic_exemplars
        assert len(sample.ref_boxes) == 3 + sample.num_dynamic_exemplars
        # The extra exemplars' boxes must be real (present-frame GT), not None.
        for extra_box in sample.ref_boxes[3:]:
            assert extra_box is not None
        counts_seen.add(sample.num_dynamic_exemplars)
        running_total += sample.num_dynamic_exemplars
    # Over 60 steps with Uniform{0,1,2}, expect to see some variation, not a
    # single fixed count every time.
    assert len(counts_seen) > 1
    assert ds.dynamic_exemplar_count == running_total


def test_dataset_dynamic_exemplars_never_reuse_the_query_frame(cfg):
    """An extra exemplar must come from a DIFFERENT present frame than the
    current step's own query frame -- otherwise the model could trivially
    match itself instead of learning genuine appearance invariance."""
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=60, max_dynamic_exemplars=2, seed=5,
    )
    checked_any = False
    for i in range(len(ds)):
        sample = ds[i]
        if not sample.is_present or sample.num_dynamic_exemplars == 0:
            continue
        checked_any = True
        for extra_img in sample.ref_images[3:]:
            assert not np.array_equal(extra_img, sample.frame_bgr), (
                "extra exemplar must not be the current query frame itself"
            )
    assert checked_any, "test setup produced no present+dynamic-exemplar step to check"
