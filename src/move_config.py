"""Configuration for the chess move prediction (Board Transformer) model."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoveDataConfig:
    processed_dir: Path = Path("data/processed")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    augment: bool = True  # random color-flip augmentation


@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    num_pieces: int = 13  # empty(0) + 12 piece types
    num_squares: int = 64
    aux_dim: int = 13
    num_moves: int = 64 * 64  # from_sq * 64 + to_sq


@dataclass
class MoveTrainConfig:
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    warmup_epochs: int = 2
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class MoveConfig:
    data: MoveDataConfig = field(default_factory=MoveDataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    train: MoveTrainConfig = field(default_factory=MoveTrainConfig)


def get_move_config() -> MoveConfig:
    """Return the default move-prediction configuration."""
    return MoveConfig()
