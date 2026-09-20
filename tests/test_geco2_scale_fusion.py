"""Tests for Track B (docs/GECO2_scale_domain_gap_plan.md):
aero_eyes/models/geco2_scale_fusion.py (ScaleFusionGate, pure torch, no
GECO2/CUDA deps -- fully testable) and
aero_eyes/models/geco2_finetune_data.py's num_ref_scale_variants/
ref_group_ids augmentation (CPU-only, same scope as
tests/test_geco2_finetune_data.py).

Does NOT and CANNOT cover aero_eyes/models/geco2_train_wrapper.py's
encode_exemplars_grad_multiscale/query_backbone_pass or
scripts/train_geco2_aeroeyes.py's run_epoch wiring -- those need a GPU with
the compiled MultiScaleDeformableAttention extension and the real GeCo2
checkpoint (see notebooks/train_geco2_aeroeyes_vastai.ipynb's --dry-run
cell; run `--num-ref-scale-variants 3` there before trusting this on a
rented GPU box).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)
np = pytest.importorskip("numpy", reason="numpy not importable (DLL blocked?)", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.models.geco2_scale_fusion import ScaleFusionGate, build_scale_fusion_gates  # noqa: E402
from aero_eyes.models.geco2_finetune_data import Geco2FinetuneDataset, RefImageCache  # noqa: E402

FIXTURE_ID = "synth001"


# ---------------------------------------------------------------------------
# ScaleFusionGate -- pure torch, CPU-only
# ---------------------------------------------------------------------------

def test_scale_fusion_gate_output_shape():
    gate = ScaleFusionGate(emb_dim=8, hidden_dim=16)
    variant_tokens = torch.randn(1, 3, 8)  # [B, K, D]
    query_context = torch.randn(1, 8)      # [B, D]
    out = gate(variant_tokens, query_context)
    assert out.shape == (1, 8)


def test_scale_fusion_gate_k1_is_weight_one():
    """A single-variant group must always get softmax weight 1.0 -- the
    'cheap no-op' property num_ref_scale_variants=1 callers rely on."""
    gate = ScaleFusionGate(emb_dim=4, hidden_dim=8)
    variant_tokens = torch.randn(1, 1, 4)
    query_context = torch.randn(1, 4)
    out = gate(variant_tokens, query_context)
    assert torch.allclose(out, variant_tokens[:, 0, :])


def test_scale_fusion_gate_rejects_mismatched_emb_dim():
    gate = ScaleFusionGate(emb_dim=8, hidden_dim=16)
    with pytest.raises(ValueError):
        gate(torch.randn(1, 3, 4), torch.randn(1, 8))


def test_scale_fusion_gate_gradients_flow():
    gate = ScaleFusionGate(emb_dim=6, hidden_dim=8)
    variant_tokens = torch.randn(1, 3, 6, requires_grad=True)
    query_context = torch.randn(1, 6)
    out = gate(variant_tokens, query_context)
    out.sum().backward()
    assert variant_tokens.grad is not None
    assert any(p.grad is not None for p in gate.parameters())


def test_scale_fusion_gate_weights_sum_to_one():
    """Indirect check that softmax normalization is actually applied:
    scaling variant_tokens by a huge constant must not change the OUTPUT's
    norm beyond what a convex combination can produce."""
    gate = ScaleFusionGate(emb_dim=4, hidden_dim=8)
    variant_tokens = torch.eye(4)[:3].unsqueeze(0)  # [1, 3, 4] one-hot rows
    query_context = torch.zeros(1, 4)
    out = gate(variant_tokens, query_context)
    # A convex combination of one-hot rows sums to <= 1.0 per component and
    # the whole vector sums to exactly 1.0 (weights sum to 1).
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_build_scale_fusion_gates_has_all_three_levels():
    gates = build_scale_fusion_gates(emb_dim=8, hidden_dim=16)
    assert set(gates.keys()) == {"main", "l1", "l2"}
    assert all(isinstance(g, ScaleFusionGate) for g in gates.values())
    # Independent weights per level (mirrors C_base's own per-level
    # PrototypeAttentionBlock instances) -- not the same module 3 times.
    assert gates["main"] is not gates["l1"]


# ---------------------------------------------------------------------------
# Geco2FinetuneDataset.num_ref_scale_variants / FinetuneSample.ref_group_ids
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synth_fixture(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("geco2_scale_fusion_fixtures")
    from scripts.make_synthetic_fixture import make_fixture
    make_fixture(out_dir, FIXTURE_ID)
    return out_dir


@pytest.fixture
def cfg(synth_fixture, tmp_path):
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


def test_num_ref_scale_variants_default_is_one_group_per_ref(cfg):
    """Backward-compat: default (1) must produce exactly the old shape --
    one entry per fixed ref, ref_group_ids = [0, 1, 2]."""
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=10, seed=1)
    sample = ds[0]
    assert len(sample.ref_images) == 3
    assert sample.ref_group_ids == [0, 1, 2]


def test_num_ref_scale_variants_three_groups_ref_images_correctly(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=10, num_ref_scale_variants=3, seed=2,
    )
    sample = ds[0]
    assert len(sample.ref_images) == 9  # 3 refs x 3 variants
    assert len(sample.ref_boxes) == 9
    assert sample.ref_group_ids == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_num_ref_scale_variants_variants_are_independently_sampled(cfg):
    """Each of the 3 variants for the SAME ref must be independently
    downscaled (not the same factor 3x) -- confirms the augmentation is
    actually re-sampled per variant, not just duplicated."""
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=5,
        num_ref_scale_variants=3, ref_downscale_range=(0.03, 0.9), seed=3,
    )
    sample = ds[0]
    # First ref's 3 variants (indices 0,1,2) should not all be pixel-identical.
    variant_shapes = {sample.ref_images[i].shape[:2] for i in range(3)}
    assert len(variant_shapes) > 1


def test_num_ref_scale_variants_combines_with_dynamic_exemplars(cfg):
    """Dynamic exemplars each get their OWN unique group id, starting after
    the fixed-ref groups -- no scale-fusion grouping among them."""
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    ds = Geco2FinetuneDataset(
        cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=60,
        num_ref_scale_variants=2, max_dynamic_exemplars=2, seed=4,
    )
    for i in range(len(ds)):
        sample = ds[i]
        n_dyn = sample.num_dynamic_exemplars
        assert len(sample.ref_images) == 3 * 2 + n_dyn
        assert len(sample.ref_group_ids) == len(sample.ref_images)
        fixed_group_ids = sample.ref_group_ids[: 3 * 2]
        assert fixed_group_ids == [0, 0, 1, 1, 2, 2]
        dyn_group_ids = sample.ref_group_ids[3 * 2:]
        # Every dynamic exemplar's id is unique and starts at 3 (== number
        # of fixed refs, not fixed*variants -- group ids count DISTINCT ref
        # photos, not entries).
        assert dyn_group_ids == list(range(3, 3 + n_dyn))
        # 3 distinct fixed-ref groups + n_dyn distinct dynamic groups.
        assert len(set(sample.ref_group_ids)) == 3 + n_dyn


def test_num_ref_scale_variants_rejects_zero_or_negative(cfg):
    ref_cache = RefImageCache(cfg, [FIXTURE_ID])
    with pytest.raises(ValueError):
        Geco2FinetuneDataset(cfg, [FIXTURE_ID], ref_cache, steps_per_epoch=10, num_ref_scale_variants=0)
