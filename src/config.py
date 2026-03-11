"""Centralized configuration for the chessNCE project."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    max_games: int = 0  # 0 = no limit (process entire PGN)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    chunk_size: int = 50_000  # positions per .npz chunk
    random_seed: int = 42
    num_pipeline_workers: int = 8  # parallel PGN processing workers


@dataclass
class ModelConfig:
    # Board encoder (CNN)
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernel: int = 3

    # Move encoder (Embeddings)
    num_squares: int = 64
    square_embed_dim: int = 32  # each of from/to gets this dim

    # Auxiliary features
    aux_dim: int = 13  # side_to_move(1) + castling(4) + en_passant(8)

    # Classifier head
    head_hidden: list[int] = field(default_factory=lambda: [256, 64])
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3  # early stopping patience
    checkpoint_dir: Path = Path("checkpoints")
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    """Return the default configuration."""
    return Config()
