"""Grad-CAM for ResNet-18: Captum when installed, else pure PyTorch hooks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.base import IMAGENET_MEAN, IMAGENET_STD, grad_cam_target_layer


def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    """cam: [1, 1, h, w] or [1, h, w] -> HxW float32 in [0,1]."""
    x = cam.detach().float().cpu().numpy()
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x.astype(np.float32)


def _grad_cam_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Classic Grad-CAM via forward/backward hooks (no Captum)."""
    model.eval()
    device = next(model.parameters()).device
    layer = grad_cam_target_layer(model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def f_hook(_m, _inp, out):
        activations.append(out)

    def b_hook(_m, _gi, go):
        gradients.append(go[0])

    h_f = layer.register_forward_hook(f_hook)
    h_b = layer.register_full_backward_hook(b_hook)
    x = input_tensor.to(device).detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()
    h_f.remove()
    h_b.remove()
    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")
    A = activations[0]
    dA = gradients[0]
    weights = dA.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    return _normalize_cam(cam)


def compute_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """
    Args:
        model: trained ResNet-18.
        input_tensor: (1, 3, H, W) on same device as model (or CPU; will be moved).
        target_class: class index; default = argmax of model output.
    Returns:
        2D numpy heatmap, values in [0,1].
    """
    device = next(model.parameters()).device
    x = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        pred = int(model(x).argmax(dim=1).item())
    if target_class is None:
        target_class = pred

    try:
        from captum.attr import LayerGradCam

        layer = grad_cam_target_layer(model)
        cam_mod = LayerGradCam(model, layer)
        x_in = x.detach().requires_grad_(True)
        attr = cam_mod.attribute(x_in, target=target_class)
        return _normalize_cam(attr)
    except ImportError:
        return _grad_cam_pytorch(model, input_tensor, target_class)


def tensor_to_pil_denorm(t: torch.Tensor) -> Image.Image:
    """Reverse ImageNet norm for a single (3,H,W) tensor in [0,1] for blending."""
    t = t.detach().cpu().clamp(0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnorm = t * std + mean
    unnorm = unnorm.clamp(0, 1).numpy()
    arr = (unnorm.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def overlay_heatmap_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Resize heatmap to image size, apply jet colormap, alpha-blend over RGB image."""
    try:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
    except Exception:  # pragma: no cover
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")

    rgb = pil_image.convert("RGB")
    w, h = rgb.size
    hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    hm_np = np.asarray(hm).astype(np.float32) / 255.0
    colored = cmap(hm_np)[..., :3]
    base = np.asarray(rgb).astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * colored
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def pil_to_model_tensor(pil_image: Image.Image, image_size: int = 224) -> torch.Tensor:
    t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(pil_image.convert("RGB"))
    return t.unsqueeze(0)
