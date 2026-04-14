"""
Aggregate client-side Grad-CAM maps into a global consensus map (project novelty).
Heatmaps must be aligned (same HxW), e.g. all resized to 224x224 before averaging.
"""

from __future__ import annotations

import numpy as np


def mean_aggregate(maps: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not maps:
        raise ValueError("maps must be non-empty")
    stack = np.stack([m.astype(np.float32) for m in maps], axis=0)
    if weights is None:
        return np.clip(stack.mean(axis=0), 0.0, 1.0)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    return np.clip(np.tensordot(w, stack, axes=([0], [0])), 0.0, 1.0)


def iou_binary(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    """IoU on binarized saliency maps."""
    aa = (a >= threshold).astype(np.uint8)
    bb = (b >= threshold).astype(np.uint8)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union > 0 else 1.0
