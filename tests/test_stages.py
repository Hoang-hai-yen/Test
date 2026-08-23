"""Per-stage independence tests using the synthetic fixture.

Verifies: each stage runs standalone, consuming the previous stage's
on-disk artifact and producing its own.

All tests mock weight downloads and heavy model inference so they run
offline without a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if numpy or cv2 can't be imported (e.g. blocked
# by Windows Application Control policy on the test machine).
np = pytest.importorskip("numpy", reason="numpy not importable (DLL blocked?)", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synth001"
FIXTURE_ID = "synth001"


@pytest.fixture(scope="session", autouse=True)
def synth_fixture(tmp_path_factory):
    """Build the synthetic fixture once per test session."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    from scripts.make_synthetic_fixture import make_fixture
    make_fixture(out_dir, FIXTURE_ID)
    return out_dir


@pytest.fixture
def cfg(synth_fixture, tmp_path):
    """Return a minimal AeroEyesConfig pointing at the synth fixture."""
    from aero_eyes.config import AeroEyesConfig, DataConfig, GTConfig, RuntimeConfig

    raw = {
        "project": {
            "work_dir": str(tmp_path / "runs"),
            "use_cache": False,
            "seed": 42,
        },
        "data": {
            "data_root": str(synth_fixture),
            "refs_subdir": "refs",
            "video_glob": "*.mp4",
            "num_references": 3,
            "gt": {
                "global_file": str(synth_fixture / FIXTURE_ID / "gt.json"),
                "box_format": "xyxy",
                "normalized": False,
                "frame_index_base": 0,
                "absent_encoding": "omit",
                "one_object_per_video": True,
            },
            "submission": {
                "path_name": "submission.json",
                "box_format": "xyxy",
                "normalized": False,
                "frame_index_base": 0,
                "absent_encoding": "omit",
            },
        },
        "runtime": {
            "device": "cpu",
            "num_workers": 1,
            "batch_size": 2,
            "log_level": "DEBUG",
            "save_visualizations": False,
        },
        "stage1": {
            "segmentation": {
                "enabled": True,
                "model": "mobilesam",
                "weights": None,
                "fallback_if_missing": "passthrough",
            },
            "feature_extractor": {
                "model": "dinov2",
                "dinov2_variant": "vits14",
                "weights": None,
                "image_size": 224,
            },
            "prototype": {
                "fusion": "mean",
                "l2_normalize": True,
                "cache_name": "prototype.npz",
            },
        },
        "stage2": {
            "keyframe_interval": 5,
            "sahi": {"use_sahi": False, "tile": [320, 320], "overlap": 0.25},
            "proposal_model": "yolov11n",
            "yolov11n": {"weights": "yolo11n.pt", "conf": 0.05, "iou": 0.5,
                          "max_det": 50, "classes": None},
            "fastsam_s": {"weights": "FastSAM-s.pt", "conf": 0.2, "iou": 0.7, "imgsz": 320},
            "candidate": {"min_box_area": 16, "max_candidates_per_keyframe": 50,
                          "feature_crop_pad": 0.1},
        },
        "stage3": {
            "similarity": "cosine",
            "match_threshold": 0.0,  # accept everything in tests
            "nms_iou": 0.5,
            "topk_per_keyframe": 3,
            "calibrate": {"enabled": False, "target_metric": "st_iou",
                          "search_range": [0.4, 0.75], "steps": 4},
        },
        "stage4": {
            "tracker": "none",
            "builtin": {"algorithm": "csrt"},
            "litetrack": {"onnx_path": None, "input_size": 128},
            "tracker_conf_threshold": 0.4,
            "max_track_age": 10,
        },
        "stage5": {
            "temporal_smoothing": {"enabled": True, "method": "ema", "ema_alpha": 0.6},
            "min_tube_length": 1,
            "fill_short_gaps": 2,
        },
        "accuracy": {
            "mode": "baseline",
            "cheap_boosters": {
                "multi_scale_scan": False,
                "scales": [1.0],
                "tuned_nms": False,
                "multi_reference_embedding": False,
            },
            "max_accuracy": {
                "synthetic_viewpoint_aug": {
                    "enabled": False,
                    "method": "homography",
                    "num_synth_views": 2,
                    "pitch_range_deg": [40, 85],
                    "fold_into_prototype": True,
                },
                "domain_prompter": {"enabled": False, "num_prompts": 2, "strength": 0.3},
            },
        },
        "eval": {"metric": "st_iou", "spatial_iou_type": "standard", "report_per_video": True},
    }

    from aero_eyes.config import AeroEyesConfig
    return AeroEyesConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_dinov2(feat_dim: int = 384):
    """Return a mock DINOv2FeatureExtractor that produces random L2-normalized features."""
    mock = MagicMock()
    mock._feature_dim.return_value = feat_dim

    def mock_extract(images, batch_size=16):
        n = len(images)
        feats = np.random.default_rng(0).standard_normal((n, feat_dim)).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
        return feats / norms

    def mock_extract_crops(frame_bgr, boxes, pad_ratio=0.1, batch_size=16):
        n = len(boxes)
        if n == 0:
            return np.zeros((0, feat_dim), dtype=np.float32)
        feats = np.random.default_rng(1).standard_normal((n, feat_dim)).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
        return feats / norms

    mock.extract.side_effect = mock_extract
    mock.extract_crops.side_effect = mock_extract_crops
    return mock


def _mock_proposals():
    """Return a mock ProposalModel producing a few random boxes."""
    mock = MagicMock()

    def mock_propose(image_bgr):
        from aero_eyes.types import Box
        h, w = image_bgr.shape[:2]
        rng = np.random.default_rng(0)
        boxes = []
        for _ in range(5):
            x1 = float(rng.integers(0, w // 2))
            y1 = float(rng.integers(0, h // 2))
            x2 = float(x1 + rng.integers(10, 60))
            y2 = float(y1 + rng.integers(10, 60))
            x2 = min(x2, w - 1)
            y2 = min(y2, h - 1)
            boxes.append(Box(x1, y1, x2, y2, score=float(rng.random())))
        return boxes

    mock.propose.side_effect = mock_propose
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stage1_produces_prototype(cfg, synth_fixture):
    """Stage 1 writes prototype.npz with correct shape."""
    feat_dim = 384
    mock_extractor = _mock_dinov2(feat_dim)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.segmentation.MobileSAMSegmenter") as mock_seg_cls:
        mock_seg = MagicMock()
        mock_seg.segment.return_value = np.ones((224, 224), dtype=bool)
        mock_seg_cls.return_value = mock_seg

        from aero_eyes.stages.stage1 import run_stage1
        proto_path = run_stage1(cfg, FIXTURE_ID)

    assert proto_path.exists(), f"prototype.npz not found at {proto_path}"
    from aero_eyes.utils.io import read_prototype
    prototype, meta, per_ref = read_prototype(proto_path)
    assert prototype.ndim == 1
    assert prototype.shape[0] == feat_dim
    assert abs(np.linalg.norm(prototype) - 1.0) < 1e-5  # L2-normalized
    assert meta["sample_id"] == FIXTURE_ID


def test_stage2_produces_candidates(cfg, synth_fixture):
    """Stage 2 writes candidates.json with expected structure."""
    mock_extractor = _mock_dinov2()
    mock_prop = _mock_proposals()

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop):
        from aero_eyes.stages.stage2 import run_stage2
        cand_path = run_stage2(cfg, FIXTURE_ID)

    assert cand_path.exists(), f"candidates.json not found at {cand_path}"
    with open(cand_path) as f:
        data = json.load(f)
    assert "frames" in data
    assert "schema_version" in data
    # At least some keyframes should have candidates
    assert len(data["frames"]) > 0


def test_stage2_dense_scan_fallback(cfg, synth_fixture):
    """When the proposal model finds nothing, stage2.dense_scan should add
    DINOv2-patch-similarity fallback candidates tagged 'candidate_dense'."""
    feat_dim = 384
    mock_extractor = _mock_dinov2(feat_dim)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.segmentation.MobileSAMSegmenter") as mock_seg_cls:
        mock_seg = MagicMock()
        mock_seg.segment.return_value = np.ones((224, 224), dtype=bool)
        mock_seg_cls.return_value = mock_seg
        from aero_eyes.stages.stage1 import run_stage1
        run_stage1(cfg, FIXTURE_ID)

    from aero_eyes.utils.io import read_prototype
    work_dir = Path(cfg.project.work_dir) / FIXTURE_ID
    prototype, _, _ = read_prototype(work_dir / cfg.stage1.prototype.cache_name)

    cfg_dict = cfg.model_dump()
    cfg_dict["stage2"]["dense_scan"] = {
        "enabled": True, "trigger_min_proposals": 1, "sim_threshold": 0.5,
        "min_blob_patches": 1, "max_dense_candidates_per_keyframe": 5,
    }
    from aero_eyes.config import AeroEyesConfig
    cfg2 = AeroEyesConfig.model_validate(cfg_dict)

    # Proposal model finds nothing on any keyframe -- forces the fallback.
    mock_prop_empty = MagicMock()
    mock_prop_empty.propose.side_effect = lambda image_bgr: []

    def mock_extract_dense_grid(image_bgr, target_size=630):
        grid = np.zeros((4, 4, feat_dim), dtype=np.float32)
        grid[1:3, 1:3] = prototype  # guaranteed cosine sim == 1.0 vs the prototype
        return grid, 10.0, 10.0

    mock_extractor.extract_dense_grid = MagicMock(side_effect=mock_extract_dense_grid)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop_empty):
        from aero_eyes.stages.stage2 import run_stage2
        cand_path = run_stage2(cfg2, FIXTURE_ID)

    from aero_eyes.stages.stage2 import read_candidates_with_features
    candidates, feat_matrix = read_candidates_with_features(cand_path)
    assert feat_matrix is not None and feat_matrix.shape[0] > 0
    all_sources = [d.source for dets in candidates.values() for d in dets]
    assert "candidate_dense" in all_sources


def test_stage3_produces_detections(cfg, synth_fixture):
    """Stage 3 reads prototype + candidates and writes detections.json."""
    # First run stages 1 and 2 with mocks
    feat_dim = 384
    mock_extractor = _mock_dinov2(feat_dim)
    mock_prop = _mock_proposals()

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.segmentation.MobileSAMSegmenter") as mock_seg_cls:
        mock_seg = MagicMock()
        mock_seg.segment.return_value = np.ones((224, 224), dtype=bool)
        mock_seg_cls.return_value = mock_seg
        from aero_eyes.stages.stage1 import run_stage1
        run_stage1(cfg, FIXTURE_ID)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop):
        from aero_eyes.stages.stage2 import run_stage2
        run_stage2(cfg, FIXTURE_ID)

    from aero_eyes.stages.stage3 import run_stage3
    det_path = run_stage3(cfg, FIXTURE_ID)

    assert det_path.exists(), f"detections.json not found at {det_path}"
    with open(det_path) as f:
        data = json.load(f)
    assert "schema_version" in data
    assert "frames" in data


def test_stage4_produces_tracks_with_none_tracker(cfg, synth_fixture):
    """Stage 4 with tracker=none writes tracks.json."""
    feat_dim = 384
    mock_extractor = _mock_dinov2(feat_dim)
    mock_prop = _mock_proposals()

    # Run stages 1-3 first
    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.segmentation.MobileSAMSegmenter") as mock_seg_cls:
        mock_seg = MagicMock()
        mock_seg.segment.return_value = np.ones((224, 224), dtype=bool)
        mock_seg_cls.return_value = mock_seg
        from aero_eyes.stages.stage1 import run_stage1
        run_stage1(cfg, FIXTURE_ID)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop):
        from aero_eyes.stages.stage2 import run_stage2
        run_stage2(cfg, FIXTURE_ID)

    from aero_eyes.stages.stage3 import run_stage3
    run_stage3(cfg, FIXTURE_ID)

    with patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop), \
         patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor):
        from aero_eyes.stages.stage4 import run_stage4
        tracks_path = run_stage4(cfg, FIXTURE_ID)

    assert tracks_path.exists(), f"tracks.json not found at {tracks_path}"
    with open(tracks_path) as f:
        data = json.load(f)
    assert "schema_version" in data
    assert "frames" in data


def test_stage4_litetrack_missing_path_raises(cfg):
    """Stage 4 with litetrack + no onnx_path raises a clear error."""
    from pydantic import ValidationError
    import copy

    cfg_dict = cfg.model_dump()
    cfg_dict["stage4"]["tracker"] = "litetrack"
    cfg_dict["stage4"]["litetrack"]["onnx_path"] = None

    from aero_eyes.config import AeroEyesConfig
    with pytest.raises(Exception) as exc_info:
        AeroEyesConfig.model_validate(cfg_dict)

    assert "litetrack" in str(exc_info.value).lower() or "onnx" in str(exc_info.value).lower()


def test_stage5_produces_submission(cfg, synth_fixture):
    """Stage 5 reads tracks.json and writes a valid submission.json."""
    feat_dim = 384
    mock_extractor = _mock_dinov2(feat_dim)
    mock_prop = _mock_proposals()

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.segmentation.MobileSAMSegmenter") as mock_seg_cls:
        mock_seg = MagicMock()
        mock_seg.segment.return_value = np.ones((224, 224), dtype=bool)
        mock_seg_cls.return_value = mock_seg
        from aero_eyes.stages.stage1 import run_stage1
        run_stage1(cfg, FIXTURE_ID)

    with patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor), \
         patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop):
        from aero_eyes.stages.stage2 import run_stage2
        run_stage2(cfg, FIXTURE_ID)

    from aero_eyes.stages.stage3 import run_stage3
    run_stage3(cfg, FIXTURE_ID)

    with patch("aero_eyes.models.proposals.build_proposal_model", return_value=mock_prop), \
         patch("aero_eyes.models.features.build_feature_extractor", return_value=mock_extractor):
        from aero_eyes.stages.stage4 import run_stage4
        run_stage4(cfg, FIXTURE_ID)

    from aero_eyes.stages.stage5 import run_stage5
    sub_path = run_stage5(cfg, FIXTURE_ID)

    assert sub_path.exists(), f"submission.json not found at {sub_path}"
    with open(sub_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "video_id" in data[0]
    assert "annotations" in data[0]
    bboxes = data[0]["annotations"][0]["bboxes"]
    if bboxes:
        assert all(k in bboxes[0] for k in ("frame", "x1", "y1", "x2", "y2"))


def test_io_round_trip_gt(synth_fixture):
    """GT loading round-trip: load gt.json -> compare frame count."""
    gt_path = synth_fixture / FIXTURE_ID / "gt.json"
    from aero_eyes.utils.io import load_gt
    gt = load_gt(gt_path, FIXTURE_ID)
    assert len(gt) > 0
    # Object is present in frames 5-25 (21 frames)
    assert len(gt) == 21
    for fi, box in gt.items():
        assert 5 <= fi < 26
        assert box.x2 > box.x1
        assert box.y2 > box.y1


def test_geometry_iou():
    """Sanity check box_iou and nms."""
    from aero_eyes.types import Box
    from aero_eyes.utils.geometry import box_iou, nms

    a = Box(0, 0, 10, 10)
    b = Box(0, 0, 10, 10)
    c = Box(100, 100, 200, 200)

    assert abs(box_iou(a, b) - 1.0) < 1e-9
    assert abs(box_iou(a, c) - 0.0) < 1e-9

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(1, 1, 11, 11, score=0.8),
             Box(100, 100, 200, 200, score=0.7)]
    keep = nms(boxes, 0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_dense_patches_to_boxes():
    """A blob of patches matching the prototype should become one Box with
    score == the blob's cosine similarity, in the expected pixel coords."""
    from aero_eyes.utils.geometry import dense_patches_to_boxes

    grid = np.zeros((10, 10, 4), dtype=np.float32)
    proto = np.array([1, 0, 0, 0], dtype=np.float32)
    grid[3:6, 3:6] = proto  # 3x3 blob of patches exactly matching the prototype

    boxes = dense_patches_to_boxes(
        grid, proto, sim_threshold=0.5, min_blob_patches=2,
        scale_x=1.0, scale_y=1.0, patch_size=14,
    )
    assert len(boxes) == 1
    b = boxes[0]
    assert abs(b.score - 1.0) < 1e-5
    assert abs(b.x1 - 3 * 14) < 1e-3
    assert abs(b.y1 - 3 * 14) < 1e-3
    assert abs(b.x2 - 6 * 14) < 1e-3
    assert abs(b.y2 - 6 * 14) < 1e-3

    # Below threshold everywhere -> no boxes.
    assert dense_patches_to_boxes(
        grid, proto, sim_threshold=1.5, min_blob_patches=1,
        scale_x=1.0, scale_y=1.0,
    ) == []

    # Blob too small (< min_blob_patches) -> filtered out.
    assert dense_patches_to_boxes(
        grid, proto, sim_threshold=0.5, min_blob_patches=100,
        scale_x=1.0, scale_y=1.0,
    ) == []


def test_fuse_feature_pyramid():
    """Coarse-to-fine top-down fusion: the same constant vector at every
    scale should fuse to that same (still unit) vector; output shape
    matches the finest grid; every fused token stays unit-norm."""
    from aero_eyes.utils.geometry import fuse_feature_pyramid

    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    coarse = np.tile(v, (2, 2, 1)).astype(np.float32)
    mid = np.tile(v, (4, 4, 1)).astype(np.float32)
    fine = np.tile(v, (8, 8, 1)).astype(np.float32)

    fused = fuse_feature_pyramid([coarse, mid, fine])
    assert fused.shape == fine.shape
    assert np.allclose(np.linalg.norm(fused, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(fused, fine, atol=1e-5)

    # Different vectors at coarse vs fine -> fused direction should be a mix
    # of both (still unit-norm), not exactly either pure vector.
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    coarse2 = np.tile(v, (2, 2, 1)).astype(np.float32)
    fine2 = np.tile(v2, (4, 4, 1)).astype(np.float32)
    fused2 = fuse_feature_pyramid([coarse2, fine2])
    assert fused2.shape == fine2.shape
    assert np.allclose(np.linalg.norm(fused2, axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(fused2[0, 0], v, atol=1e-2)
    assert not np.allclose(fused2[0, 0], v2, atol=1e-2)


def test_box_to_yolo_label():
    """Absolute-pixel xyxy Box -> normalized YOLO (cx, cy, w, h)."""
    from aero_eyes.types import Box
    from scripts.prepare_yolo_finetune_data import box_to_yolo_label

    box = Box(100, 50, 200, 150)
    cx, cy, w, h = box_to_yolo_label(box, img_w=400, img_h=300)
    assert abs(cx - 0.375) < 1e-6
    assert abs(cy - (100 / 300)) < 1e-6
    assert abs(w - 0.25) < 1e-6
    assert abs(h - (100 / 300)) < 1e-6

    # A box covering the whole frame -> cx=cy=0.5, w=h=1.0.
    full = Box(0, 0, 640, 480)
    cx, cy, w, h = box_to_yolo_label(full, img_w=640, img_h=480)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6
    assert abs(w - 1.0) < 1e-6
    assert abs(h - 1.0) < 1e-6
