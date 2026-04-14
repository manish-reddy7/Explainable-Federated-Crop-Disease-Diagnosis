"""
Centralized baseline training on PlantVillage layout:

  Option A — single root (random stratified train/val/test):
    DATA_DIR/
      Apple___Apple_scab/
      ...

  Option B — Kaggle-style split folders:
    PARENT/train/<classes>/  and  PARENT/val/<classes>/
    Use: --parent_dir PARENT
    Or:  --data_dir PARENT/train --val_dir PARENT/val
"""

from __future__ import annotations

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .base import build_model
from .transforms import eval_transforms, train_transforms
from .training import evaluate, train_one_epoch

from data.partition import split_train_val_test


def main() -> None:
    p = argparse.ArgumentParser(description="XFedCrop centralized ResNet-18 baseline")
    p.add_argument("--data_dir", type=str, default="", help="ImageFolder root (class subfolders), e.g. .../PlantVillage/train")
    p.add_argument(
        "--parent_dir",
        type=str,
        default="",
        help="If set, uses PARENT/train and PARENT/val as ImageFolder roots (overrides --data_dir/--val_dir)",
    )
    p.add_argument(
        "--val_dir",
        type=str,
        default="",
        help="If set with --data_dir, use all of data_dir for training and val_dir for validation (no random split)",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true")
    args = p.parse_args()

    if args.parent_dir:
        train_path = os.path.join(args.parent_dir, "train")
        val_path = os.path.join(args.parent_dir, "val")
    elif args.val_dir and args.data_dir:
        train_path = args.data_dir
        val_path = args.val_dir
    elif args.data_dir:
        train_path = args.data_dir
        val_path = ""
    else:
        p.error("Provide --data_dir, or --parent_dir, or --data_dir with --val_dir")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if val_path:
        train_ds = ImageFolder(train_path, transform=train_transforms(args.image_size))
        val_ds = ImageFolder(val_path, transform=eval_transforms(args.image_size))
        class_names = train_ds.classes
        num_classes = len(class_names)
        if val_ds.classes != class_names:
            raise ValueError(
                f"Train and val class folders differ: train {num_classes} vs val {len(val_ds.classes)}"
            )
    else:
        full = ImageFolder(train_path, transform=train_transforms(args.image_size))
        class_names = full.classes
        num_classes = len(class_names)
        train_idx, val_idx, _test_idx = split_train_val_test(full, seed=args.seed)
        train_ds = Subset(full, train_idx)
        full_eval = ImageFolder(train_path, transform=eval_transforms(args.image_size))
        val_ds = Subset(full_eval, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    path = os.path.join(args.out_dir, "central_resnet18_best.pt")
    for epoch in range(1, args.epochs + 1):
        # swap train transform on subset: re-wrap loaders by recreating train dataset each epoch is heavy;
        # simpler: use train transforms on full subset via Lambda — for clarity we use single train_ds with augment
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "val_acc": va_acc,
            }
            torch.save(ckpt, path)
            with open(os.path.join(args.out_dir, "class_names.json"), "w", encoding="utf-8") as f:
                json.dump(class_names, f, indent=2)
    print(f"Done. Best val acc={best_acc:.4f}. Checkpoint: {path}")


if __name__ == "__main__":
    main()
