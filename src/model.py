"""ChessMoveClassifier — CNN board encoder + move embeddings → binary classifier."""

import torch
import torch.nn as nn

from .config import ModelConfig, get_config


class ChessMoveClassifier(nn.Module):
    """Classifies whether a (board, move) pair is a historical or random move.

    Architecture:
        Board state (12, 8, 8) → Conv2D stack → flatten
        Auxiliary features (13,) → passthrough
        Move (from_sq, to_sq) → Embedding → concat
        All concatenated → FC head → logit
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = get_config().model

        # --- Board encoder (CNN) ---
        conv_layers = []
        in_channels = 12  # 12 piece planes
        for out_channels in cfg.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=cfg.conv_kernel, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        self.board_encoder = nn.Sequential(*conv_layers)

        # After convolutions, spatial dims are preserved (8x8) due to padding=1
        board_flat_dim = cfg.conv_channels[-1] * 8 * 8

        # --- Move encoder (Embeddings) ---
        self.from_embed = nn.Embedding(cfg.num_squares, cfg.square_embed_dim)
        self.to_embed = nn.Embedding(cfg.num_squares, cfg.square_embed_dim)
        move_dim = cfg.square_embed_dim * 2  # from + to concatenated

        # --- Classifier head ---
        head_input_dim = board_flat_dim + move_dim + cfg.aux_dim
        head_layers = []
        in_dim = head_input_dim
        for hidden_dim in cfg.head_hidden:
            head_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def forward(
        self,
        board: torch.Tensor,     # (B, 12, 8, 8)
        aux: torch.Tensor,       # (B, 13)
        from_sq: torch.Tensor,   # (B,)
        to_sq: torch.Tensor,     # (B,)
    ) -> torch.Tensor:            # (B, 1)  raw logit
        # Board features
        board_feat = self.board_encoder(board)          # (B, C, 8, 8)
        board_feat = board_feat.flatten(start_dim=1)    # (B, C*64)

        # Move features
        from_feat = self.from_embed(from_sq)  # (B, embed_dim)
        to_feat = self.to_embed(to_sq)        # (B, embed_dim)
        move_feat = torch.cat([from_feat, to_feat], dim=1)  # (B, 2*embed_dim)

        # Concatenate everything
        combined = torch.cat([board_feat, move_feat, aux], dim=1)

        return self.classifier(combined)
