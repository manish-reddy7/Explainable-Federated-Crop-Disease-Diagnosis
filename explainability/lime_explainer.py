"""LIME image explainer (optional; slower, model-agnostic)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from lime import lime_image


def explain_lime(
    model: nn.Module,
    image_hwc_uint8: np.ndarray,
    num_samples: int = 400,
    num_features: int = 10,
    device: str | None = None,
) -> np.ndarray:
    """
    Returns a mask (H, W) with importance weights roughly in [0, 1] for visualization.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    explainer = lime_image.LimeImageExplainer()

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        # images: N x H x W x 3 uint8
        batch = []
        for im in images:
            t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            )
            batch.append(t)
        x = torch.cat(batch, dim=0).to(dev)
        x = (x - mean) / std
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        return prob

    explanation = explainer.explain_instance(
        image_hwc_uint8.astype(np.double),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )
    _, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    m = mask.astype(np.float32)
    if m.max() > 0:
        m = m / m.max()
    return m
