"""Dataset for chess move prediction, reusing existing processed .npz chunks.

Uses chunk-streaming (IterableDataset) to avoid loading all data at once
while keeping sequential access patterns for good throughput.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .move_config import MoveDataConfig, MoveTrainConfig, get_move_config


class MoveIterableDataset(IterableDataset):
    """Streams through .npz chunks, shuffling chunk order each epoch.

    Within each chunk, samples are shuffled. This gives good randomization
    without loading all data into memory or random-access disk thrashing.
    """

    def __init__(self, chunk_paths: list[Path], augment: bool = False):
        self.chunk_paths = list(chunk_paths)
        self.augment = augment
        # Precompute total length (reads only labels from each chunk)
        self._total = 0
        for p in self.chunk_paths:
            data = np.load(p)
            self._total += int((data["labels"] == 1).sum())

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        paths = list(self.chunk_paths)
        random.shuffle(paths)
        for p in paths:
            data = np.load(p)
            mask = data["labels"] == 1
            boards = data["boards"][mask]
            auxs = data["auxs"][mask]
            from_sqs = data["from_sqs"][mask]
            to_sqs = data["to_sqs"][mask]

            indices = np.arange(len(from_sqs))
            np.random.shuffle(indices)

            for i in indices:
                board = boards[i].copy()
                aux = auxs[i].copy()
                fsq = int(from_sqs[i])
                tsq = int(to_sqs[i])

                if self.augment and random.random() < 0.5:
                    board, aux, fsq, tsq = _flip(board, aux, fsq, tsq)

                target = fsq * 64 + tsq
                yield (
                    torch.from_numpy(board),
                    torch.from_numpy(aux),
                    torch.tensor(target, dtype=torch.long),
                )


class MoveMapDataset(Dataset):
    """Map-style dataset for val/test (small enough to hold in memory)."""

    def __init__(self, chunk_paths: list[Path]):
        boards, auxs, from_sqs, to_sqs = [], [], [], []
        for p in chunk_paths:
            data = np.load(p)
            mask = data["labels"] == 1
            boards.append(data["boards"][mask])
            auxs.append(data["auxs"][mask])
            from_sqs.append(data["from_sqs"][mask])
            to_sqs.append(data["to_sqs"][mask])

        self.boards = np.concatenate(boards)
        self.auxs = np.concatenate(auxs)
        self.targets = np.concatenate(from_sqs) * 64 + np.concatenate(to_sqs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.boards[idx].copy()),
            torch.from_numpy(self.auxs[idx].copy()),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


def _flip(board, aux, from_sq, to_sq):
    """Flip colors: swap white/black pieces, flip board vertically."""
    new_board = np.empty_like(board)
    new_board[:6] = board[6:]
    new_board[6:] = board[:6]
    new_board = new_board[:, ::-1, :].copy()

    new_aux = aux.copy()
    new_aux[0] = 1.0 - aux[0]
    new_aux[1], new_aux[3] = aux[3], aux[1]
    new_aux[2], new_aux[4] = aux[4], aux[2]

    return new_board, new_aux, from_sq ^ 56, to_sq ^ 56


def _split_chunk_paths(data_cfg: MoveDataConfig):
    chunk_paths = sorted(data_cfg.processed_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk files in {data_cfg.processed_dir}. Run data pipeline first."
        )
    n = len(chunk_paths)
    n_train = max(1, int(n * data_cfg.train_ratio))
    n_val = max(1, int(n * data_cfg.val_ratio))

    train_paths = chunk_paths[:n_train]
    val_paths = chunk_paths[n_train:n_train + n_val]
    test_paths = chunk_paths[n_train + n_val:]

    if not val_paths:
        val_paths = train_paths[-1:]
    if not test_paths:
        test_paths = val_paths[-1:]
    return train_paths, val_paths, test_paths


def build_move_datasets(data_cfg: MoveDataConfig | None = None):
    """Build train (iterable) / val / test datasets."""
    if data_cfg is None:
        data_cfg = get_move_config().data

    train_paths, val_paths, test_paths = _split_chunk_paths(data_cfg)

    train_ds = MoveIterableDataset(train_paths, augment=data_cfg.augment)
    val_ds = MoveMapDataset(val_paths)
    test_ds = MoveMapDataset(test_paths)

    print(f"Move dataset — train: ~{len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_move_loaders(
    data_cfg: MoveDataConfig | None = None,
    train_cfg: MoveTrainConfig | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for move prediction."""
    if data_cfg is None:
        data_cfg = get_move_config().data
    if train_cfg is None:
        train_cfg = get_move_config().train

    train_ds, val_ds, test_ds = build_move_datasets(data_cfg)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
