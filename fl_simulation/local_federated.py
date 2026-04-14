"""
In-process federated simulation (FedAvg / FedProx) — no Ray or Flower required.
Matches the project plan; use Flower separately if you install `flwr[simulation]` and Ray.

Run from `XFedCrop/`:
  python fl_simulation/local_federated.py --data_dir path/to/plantvillage --clients 5 --rounds 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.partition import partition_dirichlet, partition_iid, split_train_val_test, subset_from_indices  # noqa: E402
from models.base import build_model  # noqa: E402
from models.training import evaluate, train_one_epoch  # noqa: E402
from models.transforms import eval_transforms, train_transforms  # noqa: E402


def _fedavg_state_dicts(state_dicts: list[dict], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    keys = state_dicts[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        acc = torch.zeros_like(state_dicts[0][k], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += sd[k].float() * (w / total)
        out[k] = acc.to(state_dicts[0][k].dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.0, help="Dirichlet alpha; 0 = IID shards")
    ap.add_argument("--proximal_mu", type=float, default=0.0, help="FedProx μ (0 => FedAvg)")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.clients < 1:
        raise SystemExit("--clients must be >= 1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = ImageFolder(args.data_dir, transform=train_transforms(args.image_size))
    eval_root = ImageFolder(args.data_dir, transform=eval_transforms(args.image_size))
    class_names = train_root.classes
    num_classes = len(class_names)

    train_idx, val_idx, _ = split_train_val_test(train_root, seed=args.seed)
    if args.alpha and args.alpha > 0:
        shards = partition_dirichlet(train_root, train_idx, args.clients, alpha=args.alpha, seed=args.seed)
    else:
        shards = partition_iid(train_idx, args.clients, seed=args.seed)

    client_loaders: list[DataLoader] = []
    counts: list[int] = []
    for s in shards:
        sub = subset_from_indices(train_root, s)
        counts.append(len(sub))
        client_loaders.append(
            DataLoader(
                sub,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )

    val_subset = subset_from_indices(eval_root, val_idx)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = build_model(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "fl_resnet18_best.pt")

    for rnd in range(1, args.rounds + 1):
        global_sd = copy.deepcopy(global_model.state_dict())
        collected: list[dict] = []
        for cid, loader in enumerate(client_loaders):
            local = build_model(num_classes, pretrained=False).to(device)
            local.load_state_dict(global_sd)
            opt = torch.optim.Adam(local.parameters(), lr=args.lr)
            global_state = {k: v.detach().clone().to(device) for k, v in global_sd.items()}
            for _ in range(args.local_epochs):
                train_one_epoch(
                    local,
                    loader,
                    opt,
                    criterion,
                    device,
                    proximal_mu=args.proximal_mu,
                    global_state=global_state if args.proximal_mu > 0 else None,
                )
            collected.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            del local

        agg = _fedavg_state_dicts(collected, counts)
        global_model.load_state_dict(agg)
        va_loss, va_acc = evaluate(global_model, val_loader, criterion, device)
        tag = "FedProx" if args.proximal_mu > 0 else "FedAvg"
        print(f"[{tag}] round {rnd:03d}/{args.rounds}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

        if va_acc >= best_acc:
            best_acc = va_acc
            ckpt = {
                "round": rnd,
                "model_state_dict": global_model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "val_acc": va_acc,
                "proximal_mu": args.proximal_mu,
            }
            torch.save(ckpt, save_path)
            with open(os.path.join(args.out_dir, "class_names.json"), "w", encoding="utf-8") as f:
                json.dump(class_names, f, indent=2)

    print(f"Done. Best val acc={best_acc:.4f}. Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
