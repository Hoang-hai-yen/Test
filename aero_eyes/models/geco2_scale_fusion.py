"""Track B: learned multi-scale fusion for GeCo2 exemplar tokens (see
docs/GECO2_scale_domain_gap_plan.md).

Closes the train/inference mismatch stage123_geco2.ref_downscale_levels'
inference-only flat concatenation leaves open (multiple scale-variants of
the SAME reference image is a configuration GeCo2 was never trained on --
see that config field's own docstring): trains a small, query-conditioned
gate that learns to weight/select among several scale-variant appearance
tokens of the same underlying reference photo, using the QUERY image's own
backbone features as the conditioning signal -- the query frame is the
only real evidence for "how blurry/small this object actually looks here",
no external metadata (GSD, altitude, ...) needed.

Kept OUTSIDE GECO2/ (this repo's established convention -- see
GeCo2Detector.sam2_refine_boxes's own docstring: extend from outside
vendored code, never edit a line inside GECO2/). Plugs in strictly BEFORE
GECO2/models/query_generator.py::C_base.forward -- that module's signature
and cross-attention behavior are completely untouched; this module only
changes how many tokens PER REFERENCE end up in the flat K/V sequence
C_base already treats generically (1, same as today, instead of K).
"""
from __future__ import annotations

import torch
from torch import nn


class ScaleFusionGate(nn.Module):
    """Learned, query-conditioned weighting over K scale-variant appearance
    tokens of the SAME reference image, for ONE pyramid level (main/l1/l2)
    -- GeCo2's exemplar pipeline needs one instance per level, mirroring
    GECO2/models/query_generator.py::C_base's own per-level instantiation
    of PrototypeAttentionBlock (three independent instances, never shared
    weights across main/l1/l2).

    NOT YET VALIDATED -- see docs/GECO2_scale_domain_gap_plan.md Track B's
    own risk notes (small dataset: only 7 distinct physical objects) before
    trusting a finetuned checkpoint using this in production.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, variant_tokens: torch.Tensor, query_context: torch.Tensor) -> torch.Tensor:
        """variant_tokens: [B, K, D] -- K scale-variants of ONE reference
        image, at this gate's own pyramid level (K=1 is a valid, cheap no-op
        input -- softmax over a single logit is always weight 1.0, so
        num_ref_scale_variants=1 callers can route through this same code
        path without a special case).
        query_context: [B, D] -- this level's own query-image feature,
        global-average-pooled (same operation as GeCo2Detector.
        frame_domain_embedding already performs for domain_calibration, at
        the SAME level).
        Returns: [B, D] -- the fused single token for this reference image.
        """
        b, k, d = variant_tokens.shape
        if d != self.emb_dim:
            raise ValueError(f"variant_tokens last dim ({d}) != emb_dim ({self.emb_dim})")
        q = query_context.unsqueeze(1).expand(-1, k, -1)
        logits = self.gate(torch.cat([variant_tokens, q], dim=-1)).squeeze(-1)  # [B, K]
        weights = torch.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * variant_tokens).sum(dim=1)


def build_scale_fusion_gates(emb_dim: int, hidden_dim: int = 64) -> nn.ModuleDict:
    """One ScaleFusionGate per GeCo2 pyramid level -- the exact set
    aero_eyes.models.geco2_train_wrapper.encode_exemplars_grad_multiscale
    expects, and what gets attached to the model as
    `model.scale_fusion_gates` (see that module's docstring for why a
    ModuleDict keyed by level, not a single shared gate: main/l1/l2 have
    independent weights, same as every other per-level submodule in
    GECO2/models/query_generator.py::C_base).
    """
    return nn.ModuleDict({
        level: ScaleFusionGate(emb_dim, hidden_dim) for level in ("main", "l1", "l2")
    })
