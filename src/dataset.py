"""PyTorch Dataset for loading preprocessed chess position chunks."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from .config import DataConfig, TrainConfig, get_config


class ChessChunkDataset(Dataset):
    """Dataset backed by a single .npz chunk file."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.boards = data["boards"]      # (N, 12, 8, 8)
        self.auxs = data["auxs"]          # (N, 13)
        self.from_sqs = data["from_sqs"]  # (N,)
        self.to_sqs = data["to_sqs"]      # (N,)
        self.labels = data["labels"]      # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.boards[idx]),
            torch.from_numpy(self.auxs[idx]),
            torch.tensor(self.from_sqs[idx], dtype=torch.long),
            torch.tensor(self.to_sqs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def build_datasets(
    data_cfg: DataConfig | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load all .npz chunks and split into train/val/test by chunk order.

    Chunks are assigned to splits in order (not shuffled) to approximate
    game-level splitting, since chunks are written sequentially from the PGN.
    """
    if data_cfg is None:
        data_cfg = get_config().data

    chunk_paths = sorted(data_cfg.processed_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk files found in {data_cfg.processed_dir}. "
            "Run the data pipeline first: python -m src.data_pipeline <pgn_path>"
        )

    n = len(chunk_paths)
    n_train = max(1, int(n * data_cfg.train_ratio))
    n_val = max(1, int(n * data_cfg.val_ratio))

    train_paths = chunk_paths[:n_train]
    val_paths = chunk_paths[n_train:n_train + n_val]
    test_paths = chunk_paths[n_train + n_val:]

    # Fall back: if too few chunks, at least put something in each split
    if not val_paths:
        val_paths = train_paths[-1:]
    if not test_paths:
        test_paths = val_paths[-1:]

    train_ds = ConcatDataset([ChessChunkDataset(p) for p in train_paths])
    val_ds = ConcatDataset([ChessChunkDataset(p) for p in val_paths])
    test_ds = ConcatDataset([ChessChunkDataset(p) for p in test_paths])

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_loaders(
    data_cfg: DataConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders."""
    if data_cfg is None:
        data_cfg = get_config().data
    if train_cfg is None:
        train_cfg = get_config().train

    train_ds, val_ds, test_ds = build_datasets(data_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
