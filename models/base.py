"""ResNet-18 backbone for PlantVillage-style multi-class leaf disease classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with ImageNet pretraining and a linear head for `num_classes` labels.
    Grad-CAM target layer: `model.layer4[-1]` (last BasicBlock conv output).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def grad_cam_target_layer(model: nn.Module) -> nn.Module:
    """Last conv block of ResNet-18 for Captum LayerGradCam."""
    return model.layer4[-1]
