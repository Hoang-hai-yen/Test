"""PCA whitening of L2-normalised embeddings (pure numpy).

Fitted offline by scripts/fit_pca_whitening.py, applied online by Stage 3
(stage3.whitening): a fixed linear map, so it needs no other frame's data and
is causal/streaming-safe.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhiteningParams:
    n_components: int = 128   # principal directions kept (after drop_top)
    drop_top: int = 0         # skip the first N (largest-variance, usually "common") directions
    power: float = 1.0        # 1 = full whitening, 0.5 = square-root, 0 = plain PCA projection
    eps: float = 0.1          # eigenvalue floor as a fraction of the mean eigenvalue


class Whitener:
    def __init__(self, mean: np.ndarray, components: np.ndarray, scale: np.ndarray, params: WhiteningParams):
        self.mean, self.components, self.scale, self.params = mean, components, scale, params

    @property
    def in_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.components.shape[1])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """[N, D] (or [D]) -> L2-normalised [N, k] (or [k])."""
        z = ((np.asarray(x, np.float64) - self.mean) @ self.components) * self.scale
        return (z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)).astype(np.float32)

    def save(self, path: Path, extra: dict | None = None) -> None:
        np.savez(
            path, mean=self.mean, components=self.components, scale=self.scale,
            params=json.dumps(asdict(self.params)), extra=json.dumps(extra or {}),
        )

    @classmethod
    def load(cls, path: Path) -> "Whitener":
        z = np.load(path, allow_pickle=False)
        return cls(z["mean"], z["components"], z["scale"], WhiteningParams(**json.loads(str(z["params"]))))


def fit_whitener(x: np.ndarray, params: WhiteningParams) -> Whitener:
    """PCA on x [N, D]. Directions are ranked by eigenvalue; directions
    [drop_top, drop_top + n_components) are kept and each is divided by
    (eigenvalue + eps * mean_eigenvalue) ** (power / 2). The eigenvalue floor
    stops near-empty directions (noise) from being blown up to unit variance.
    n_components is clamped to the covariance rank (N - 1): beyond it the
    directions carry no data at all."""
    x = np.asarray(x, np.float64)
    n, d = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 vectors to fit whitening, got {n}.")
    mean = x.mean(axis=0)
    xc = x - mean
    evals, evecs = np.linalg.eigh(xc.T @ xc / (n - 1))
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]

    lo = params.drop_top
    k = min(params.n_components, min(d, n - 1) - lo)
    if k < 1:
        raise ValueError(f"drop_top={lo} leaves no directions (rank {min(d, n - 1)}).")
    if k < params.n_components:
        log.warning("n_components %d clamped to %d (covariance rank %d, drop_top %d).",
                    params.n_components, k, min(d, n - 1), lo)
    sel = slice(lo, lo + k)
    floor = params.eps * float(evals.mean())
    scale = (evals[sel] + floor + 1e-12) ** (-params.power / 2.0)
    return Whitener(mean, evecs[:, sel], scale, params)
