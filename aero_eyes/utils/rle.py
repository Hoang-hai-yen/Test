"""COCO run-length encoding of binary masks, without pycocotools.

Same format as pycocotools / Roboflow ("rle_mask"): {"size": [h, w],
"counts": <compressed string>}, runs alternating 0/1 starting with 0, in
column-major order. Used to keep precomputed reference masks inside a single
JSON file (scripts/precompute_ref_masks.py -> --ref-masks).
"""
from __future__ import annotations

import numpy as np


def encode_coco_rle(mask: np.ndarray) -> dict:
    """HxW bool mask -> {"size": [h, w], "counts": compressed string}
    (pycocotools' rleToString)."""
    flat = np.asarray(mask, dtype=bool).T.ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs  # runs start with a (possibly empty) run of 0s
    out = []
    for i, x in enumerate(runs):
        if i > 2:
            x -= runs[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": "".join(out)}


def decode_coco_rle(size: list[int], counts: str | list[int]) -> np.ndarray:
    """COCO RLE -> HxW bool mask. `counts` is the compressed string
    (pycocotools' rleFrString) or an uncompressed list of run lengths."""
    h, w = int(size[0]), int(size[1])
    if isinstance(counts, str):
        runs: list[int] = []
        p = 0
        while p < len(counts):
            x, k, more = 0, 0, True
            while more:
                c = ord(counts[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(runs) > 2:
                x += runs[-2]
            runs.append(x)
    else:
        runs = [int(v) for v in counts]
    if sum(runs) != h * w:
        raise ValueError(f"RLE runs sum to {sum(runs)}, expected {h}*{w}={h * w}")
    values = np.zeros(len(runs), dtype=bool)
    values[1::2] = True
    return np.repeat(values, runs).reshape(w, h).T
