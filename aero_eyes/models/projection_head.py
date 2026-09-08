"""Small trainable head applied on top of a frozen backbone embedding
(dinov2/dinov3/clip/siglip/ensemble) to close the domain gap between
close-up reference photos and tiny aerial crops -- see
scripts/train_projection_head.py for training and
aero_eyes.models.features.ProjectedFeatureExtractor for how this wraps
around an existing extractor at inference time.

Deliberately tiny (single Linear by default, optional one hidden layer):
the training set is a handful of object categories, so a large head would
overfit to those specific categories instead of generalizing to the unseen
objects this project must match at eval time -- see the config docstring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """L2-normalized (optionally 2-layer) projection: in_dim -> out_dim."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hidden_dim": self.hidden_dim,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ProjectionHead":
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        head = cls(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"], hidden_dim=ckpt.get("hidden_dim"))
        head.load_state_dict(ckpt["state_dict"])
        head.eval().to(device)
        return head
