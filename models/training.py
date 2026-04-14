from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_state: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss: torch.Tensor = criterion(logits, y)
        if proximal_mu > 0 and global_state is not None:
            for name, p in model.named_parameters():
                if name not in global_state:
                    continue
                g = global_state[name].to(device)
                loss = loss + (proximal_mu / 2.0) * torch.sum((p - g) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    acc = correct / max(n, 1)
    return total_loss / max(n, 1), acc


def get_global_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def model_params_to_numpy(model: nn.Module) -> tuple[list[str], list]:
    keys = list(model.state_dict().keys())
    arrays = [model.state_dict()[k].detach().cpu().numpy() for k in keys]
    return keys, arrays


def numpy_to_state_dict(keys: list[str], arrays: list, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, device=device) for k, v in zip(keys, arrays)}
