"""Minimal LoRA (low-rank residual) adapters for a frozen HuggingFace ViT --
no `peft` dependency. Used by scripts/train_lora_dinov3.py to fine-tune a few
attention projections of DINOv3, and by DINOv3FeatureExtractor
(stage1.feature_extractor.dinov3_lora_weights_path) to load the result.

Targets nn.Linear modules by their LAST name segment (default q_proj/v_proj,
the names transformers' DINOv3ViTAttention uses) inside transformer blocks
matched by a `layer.<idx>.` path segment. B is zero-initialized, so a freshly
wrapped model computes exactly what the frozen model did.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch import nn

_BLOCK_RE = re.compile(r"(?:^|\.)layer\.(\d+)\.")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + (delta * self.scaling).to(out.dtype)


def _block_index(name: str) -> int | None:
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] | list[str] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    last_n_blocks: int | None = None,
) -> list[str]:
    """Freeze every parameter of `model`, then wrap the matching Linear
    layers with LoRA. Only the LoRA A/B matrices are trainable afterwards.
    last_n_blocks=None wraps every transformer block. Returns the wrapped
    module names. Raises if nothing matched (wrong `targets`, or the model
    was already wrapped)."""
    targets = tuple(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    linears = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets and _block_index(n) is not None
    ]
    if not linears:
        sample = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][:8]
        raise ValueError(
            f"apply_lora: no nn.Linear named {targets} inside a 'layer.<idx>.' block was found "
            f"(already wrapped?). Example Linear names in this model: {sample}"
        )
    max_block = max(_block_index(n) for n, _ in linears)
    lo = 0 if last_n_blocks is None else max_block - last_n_blocks + 1
    wrapped: list[str] = []
    for name, mod in linears:
        if _block_index(name) < lo:
            continue
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, LoRALinear(mod, rank, alpha))
        wrapped.append(name)
    model._lora_meta = {
        "targets": list(targets), "rank": rank, "alpha": alpha, "last_n_blocks": last_n_blocks,
    }
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))]


def save_lora(model: nn.Module, path: str | Path, train_config: dict | None = None) -> None:
    """train_config: plain-primitive settings the adapters were trained under
    (preprocess modes, image size, ...), stored so a later run can warn when
    it is not running with the same preprocessing (see config_mismatches)."""
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"meta": model._lora_meta, "state": state, "train_config": dict(train_config or {})}, str(path),
    )


def config_mismatches(saved: dict, current: dict) -> list[str]:
    """Human-readable differences for every key present in BOTH dicts."""
    return [
        f"{k}: trained with {saved[k]!r}, running with {current[k]!r}"
        for k in saved if k in current and saved[k] != current[k]
    ]


def load_lora(model: nn.Module, path: str | Path) -> dict:
    """Wrap `model` per the checkpoint's own metadata (if it isn't wrapped
    yet), then load the saved A/B matrices. Returns the metadata plus the
    checkpoint's "train_config" (empty for checkpoints saved without one)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    if not hasattr(model, "_lora_meta"):
        apply_lora(
            model, targets=meta["targets"], rank=meta["rank"], alpha=meta["alpha"],
            last_n_blocks=meta["last_n_blocks"],
        )
    params = {n: p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}
    if set(params) != set(ckpt["state"]):
        raise ValueError(
            f"LoRA checkpoint {path} does not match this model's LoRA layers "
            f"(checkpoint has {len(ckpt['state'])} tensors, model has {len(params)})."
        )
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(ckpt["state"][n].to(p.device, p.dtype))
    return {**meta, "train_config": ckpt.get("train_config", {})}
