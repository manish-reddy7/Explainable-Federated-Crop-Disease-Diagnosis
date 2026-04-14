"""
IID and non-IID (Dirichlet) partitioning for federated simulation.
Expects a torchvision ImageFolder-style dataset: subset.indices maps to global indices.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def split_train_val_test(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split by ImageFolder targets (requires dataset.targets)."""
    if not hasattr(dataset, "targets"):
        raise ValueError("Dataset must have .targets (e.g. torchvision.datasets.ImageFolder).")
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for c in np.unique(targets):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(math.floor(n * train_ratio))
        n_val = int(math.floor(n * val_ratio))
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val :].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def partition_iid(indices: Sequence[int], num_clients: int, seed: int = 42) -> list[list[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    chunks: list[list[int]] = [[] for _ in range(num_clients)]
    for i, idx in enumerate(shuffled):
        chunks[i % num_clients].append(idx)
    return chunks


def partition_dirichlet(
    dataset: torch.utils.data.Dataset,
    indices: Sequence[int],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """
    Label skew: each client's label distribution ~ Dirichlet(alpha).
    Lower alpha => stronger heterogeneity.
    """
    if not hasattr(dataset, "targets"):
        raise ValueError("Dataset must have .targets.")
    targets = np.array(dataset.targets)
    idx_by_class: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        idx_by_class[int(targets[i])].append(int(i))
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    num_classes = int(targets.max()) + 1
    for c in range(num_classes):
        idxs = idx_by_class.get(c, [])
        if not idxs:
            continue
        rng.shuffle(idxs)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        props = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, props)
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices


def subset_from_indices(dataset: torch.utils.data.Dataset, indices: Iterable[int]) -> Subset:
    return Subset(dataset, list(indices))


class SubsetWithTargets(Dataset):
    """Wraps a Subset so batching code can read .targets for stratified operations."""

    def __init__(self, base_dataset: torch.utils.data.Dataset, indices: Sequence[int]):
        self.base = base_dataset
        self.indices = list(indices)
        if not hasattr(base_dataset, "targets"):
            raise ValueError("Base dataset needs .targets.")
        t = np.array(base_dataset.targets)
        self.targets = t[self.indices].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]
