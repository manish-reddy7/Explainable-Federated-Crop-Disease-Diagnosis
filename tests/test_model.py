"""Run from `XFedCrop/`: pytest tests/test_model.py"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.base import build_model  # noqa: E402


def test_resnet_forward():
    m = build_model(num_classes=38, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 38)


def test_grad_cam_runs():
    from explainability.grad_cam import compute_grad_cam  # noqa: E402
    from models.base import grad_cam_target_layer  # noqa: E402

    m = build_model(38, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    layer = grad_cam_target_layer(m)
    assert layer is not None
    hm = compute_grad_cam(m, x, target_class=0)
    assert hm.ndim == 2
    assert hm.min() >= 0 and hm.max() <= 1.0 + 1e-5
