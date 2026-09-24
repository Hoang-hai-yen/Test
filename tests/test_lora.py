"""Unit tests for aero_eyes.models.lora (minimal LoRA for a frozen ViT)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from aero_eyes.models.lora import LoRALinear, apply_lora, load_lora, lora_parameters, save_lora


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = _Attn(d)

    def forward(self, x):
        a = self.attention
        return x + a.o_proj(a.q_proj(x) + a.k_proj(x) + a.v_proj(x))


class _Enc(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layer = nn.ModuleList([_Layer(d) for _ in range(n)])


class _Fake(nn.Module):
    def __init__(self, d=8, n=4):
        super().__init__()
        self.encoder = _Enc(d, n)
        self.head = nn.Linear(d, d)

    def forward(self, x):
        for l in self.encoder.layer:
            x = l(x)
        return self.head(x)


def _model(seed=0):
    torch.manual_seed(seed)
    return _Fake()


def test_wrapped_model_is_identical_at_init():
    m = _model()
    x = torch.randn(5, 8)
    before = m(x)
    apply_lora(m, rank=4)
    assert torch.allclose(m(x), before, atol=1e-6)


def test_only_lora_params_are_trainable():
    m = _model()
    wrapped = apply_lora(m, targets=("q_proj", "v_proj"), rank=4)
    assert len(wrapped) == 2 * 4
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B")) for n in trainable)
    assert len(lora_parameters(m)) == 2 * len(wrapped)


def test_last_n_blocks_restricts_wrapping():
    m = _model()
    wrapped = apply_lora(m, targets=("q_proj",), rank=2, last_n_blocks=2)
    assert wrapped == ["encoder.layer.2.attention.q_proj", "encoder.layer.3.attention.q_proj"]


def test_gradient_reaches_lora_and_a_step_changes_the_output():
    m = _model()
    apply_lora(m, rank=4)
    x = torch.randn(6, 8)
    before = m(x).detach()
    opt = torch.optim.SGD(lora_parameters(m), lr=0.1)
    for _ in range(2):  # first step only moves B (A's grad is 0 while B == 0)
        opt.zero_grad()
        m(x).pow(2).sum().backward()
        opt.step()
    assert not torch.allclose(m(x).detach(), before)
    assert m.encoder.layer[0].attention.q_proj.base.weight.grad is None


def test_save_load_roundtrip_reproduces_outputs(tmp_path):
    m = _model()
    apply_lora(m, rank=4, last_n_blocks=2)
    with torch.no_grad():
        for p in lora_parameters(m):
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 8)
    expected = m(x)
    save_lora(m, tmp_path / "l.pt")

    fresh = _model()  # same base weights (same seed), not wrapped
    meta = load_lora(fresh, tmp_path / "l.pt")
    assert meta["rank"] == 4 and meta["last_n_blocks"] == 2
    assert torch.allclose(fresh(x), expected, atol=1e-6)


def test_no_matching_layers_raises_with_helpful_message():
    with pytest.raises(ValueError, match="no nn.Linear named"):
        apply_lora(_model(), targets=("does_not_exist",))


def test_applying_twice_raises_instead_of_double_wrapping():
    m = _model()
    apply_lora(m, rank=2)
    with pytest.raises(ValueError):
        apply_lora(m, rank=2)


def test_loralinear_keeps_base_frozen():
    base = nn.Linear(4, 4)
    l = LoRALinear(base, rank=2, alpha=4)
    assert not base.weight.requires_grad and l.lora_A.requires_grad and l.lora_B.requires_grad


def test_train_config_roundtrips_and_mismatches_are_reported(tmp_path):
    from aero_eyes.models.lora import config_mismatches

    m = _model()
    apply_lora(m, rank=2)
    save_lora(m, tmp_path / "l.pt", {"preprocess_mode": "stretch", "image_size": 224})
    meta = load_lora(_model(), tmp_path / "l.pt")
    assert meta["train_config"] == {"preprocess_mode": "stretch", "image_size": 224}

    running = {"preprocess_mode": "pad_to_square", "image_size": 224, "unrelated": 1}
    diffs = config_mismatches(meta["train_config"], running)
    assert len(diffs) == 1 and "preprocess_mode" in diffs[0]
    assert config_mismatches({}, running) == []          # old checkpoint without a config: no warnings


def test_checkpoint_without_train_config_still_loads(tmp_path):
    m = _model()
    apply_lora(m, rank=2)
    save_lora(m, tmp_path / "l.pt")
    assert load_lora(_model(), tmp_path / "l.pt")["train_config"] == {}
