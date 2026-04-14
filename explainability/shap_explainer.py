"""SHAP helpers (optional; can be slow on full images)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def explain_shap_deep_small_background(
    model: nn.Module,
    input_1x3x224: torch.Tensor,
    background_batch: torch.Tensor,
    device: str | None = None,
) -> np.ndarray:
    """
    DeepExplainer with a small background set; returns abs SHAP sum per pixel (approx).
    For production comparisons, prefer Grad-CAM on images; SHAP here is a thin wrapper.
    """
    import shap

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    bg = background_batch.to(dev)
    x = input_1x3x224.to(dev)

    def f(z: np.ndarray) -> np.ndarray:
        t = torch.tensor(z, dtype=torch.float32, device=dev)
        with torch.no_grad():
            return model(t).cpu().numpy()

    explainer = shap.DeepExplainer(model, bg)
    shap_values = explainer.shap_values(x)
    # shap_values: list per class or single array
    if isinstance(shap_values, list):
        pred = int(model(x).argmax(dim=1).item())
        sv = shap_values[pred]
    else:
        sv = shap_values
    sv = np.array(sv)
    if sv.ndim == 4:
        sv = sv[0]
    imp = np.abs(sv).sum(axis=0)
    imp = imp - imp.min()
    if imp.max() > 0:
        imp = imp / imp.max()
    return imp.astype(np.float32)
