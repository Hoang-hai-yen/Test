import numpy as np
import pytest

from scripts.fit_pca_whitening import (
    Whitener, WhiteningParams, evaluate_grid, fit_whitener, load_embeddings, make_folds, save_embeddings,
)


def _anisotropic(n=2000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.linspace(5.0, 0.1, d)
    return rng.normal(size=(n, d)) * scales + 3.0


def test_full_whitening_gives_unit_covariance_before_normalisation():
    x = _anisotropic()
    w = fit_whitener(x, WhiteningParams(n_components=16, power=1.0, eps=0.0))
    z = ((x - w.mean) @ w.components) * w.scale
    assert np.allclose(np.cov(z.T), np.eye(16), atol=1e-6)


def test_transform_is_l2_normalised_and_shaped():
    x = _anisotropic()
    w = fit_whitener(x, WhiteningParams(n_components=8, drop_top=2))
    z = w.transform(x[:5])
    assert z.shape == (5, 8)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_n_components_clamped_to_rank():
    x = _anisotropic(n=6, d=16)
    w = fit_whitener(x, WhiteningParams(n_components=16))
    assert w.components.shape[1] == 5  # rank = N - 1


def test_save_load_roundtrip(tmp_path):
    x = _anisotropic()
    w = fit_whitener(x, WhiteningParams(n_components=8, power=0.5))
    w.save(tmp_path / "w.npz")
    w2 = Whitener.load(tmp_path / "w.npz")
    assert np.allclose(w.transform(x[:4]), w2.transform(x[:4]), atol=1e-6)
    assert w2.params == w.params


def _synthetic_emb(seed=0):
    """Two objects x two videos. A big shared 'background' direction makes raw
    cosine of target and clutter both high; the object signal lives in small
    directions."""
    rng = np.random.default_rng(seed)
    d = 32
    common = np.zeros(d)
    common[0] = 10.0
    emb = {}
    for o in range(3):
        sig = np.zeros(d)
        sig[1 + o] = 1.0
        for v in range(2):
            def draw(n, s):
                return common + s * sig + rng.normal(scale=0.3, size=(n, d))
            emb[f"O{o}_{v}"] = {
                "obj": f"O{o}", "refs": draw(3, 1.0), "pos": draw(60, 1.0), "neg": draw(120, 0.0),
            }
    return emb


def test_evaluate_grid_whitening_beats_raw_on_dominant_common_direction():
    emb = _synthetic_emb()
    folds = make_folds(emb, "loo")
    assert len(folds) == 3 and all(set(t).isdisjoint(v) for t, v in folds)
    res = evaluate_grid(emb, [WhiteningParams(n_components=16, power=1.0, eps=0.01)], "pos+neg", folds)
    assert res["grid"][0]["auroc"] > res["baseline"]["auroc"]


def test_embedding_cache_roundtrip(tmp_path):
    emb = _synthetic_emb()
    save_embeddings(emb, tmp_path / "e.npz")
    back = load_embeddings(tmp_path / "e.npz")
    assert set(back) == set(emb)
    assert back["O0_1"]["obj"] == "O0"
    assert np.allclose(back["O0_1"]["pos"], emb["O0_1"]["pos"])


def test_fit_requires_two_vectors():
    with pytest.raises(ValueError):
        fit_whitener(np.zeros((1, 4)), WhiteningParams())


def test_stage3_apply_whitening(tmp_path):
    from aero_eyes.config import WhiteningConfig
    from aero_eyes.stages.stage3 import apply_whitening

    x = _anisotropic(n=500, d=16)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    w = fit_whitener(x, WhiteningParams(n_components=8))
    w.save(tmp_path / "w.npz")
    cfg = WhiteningConfig(enabled=True, weights_path=str(tmp_path / "w.npz"))

    feats, proto, refs = apply_whitening(cfg, x[:10], x[0], [x[1], x[2]])
    assert feats.shape == (10, 8) and proto.shape == (8,) and len(refs) == 2 and refs[0].shape == (8,)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)

    with pytest.raises(ValueError, match="different extractor"):
        apply_whitening(cfg, np.zeros((3, 20), np.float32), np.zeros(20), [])
    with pytest.raises(ValueError, match="weights_path"):
        apply_whitening(WhiteningConfig(enabled=True), x[:2], x[0], [])
    with pytest.raises(FileNotFoundError):
        apply_whitening(WhiteningConfig(enabled=True, weights_path=str(tmp_path / "nope.npz")), x[:2], x[0], [])
