"""Board Transformer — predicts chess moves from board positions.

Architecture:
    64 square tokens (piece embedding + position embedding + global projection)
    + CLS token → Transformer Encoder → CLS output → 4096-way move head
"""

import math

import torch
import torch.nn as nn

from .move_config import TransformerConfig, get_move_config


class BoardTransformer(nn.Module):
    """Transformer-based chess move predictor.

    Input:  board (B, 12, 8, 8) + aux (B, 13)
    Output: logits (B, 4096) over from_sq * 64 + to_sq move space
    """

    def __init__(self, cfg: TransformerConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = get_move_config().model

        self.d_model = cfg.d_model

        # Piece embedding: 0=empty, 1-12=piece types
        self.piece_embed = nn.Embedding(cfg.num_pieces, cfg.d_model)
        # Positional embedding for each of the 64 squares
        self.pos_embed = nn.Embedding(cfg.num_squares, cfg.d_model)
        # Project global auxiliary features into token space
        self.global_proj = nn.Linear(cfg.aux_dim, cfg.d_model)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.input_drop = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.move_head = nn.Linear(cfg.d_model, cfg.num_moves)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Small init for output head to start with uniform predictions
        nn.init.normal_(self.move_head.weight, std=0.02)
        nn.init.zeros_(self.move_head.bias)

    def _board_to_piece_indices(self, board: torch.Tensor) -> torch.Tensor:
        """Convert (B, 12, 8, 8) one-hot planes → (B, 64) piece index."""
        B = board.shape[0]
        flat = board.view(B, 12, 64).permute(0, 2, 1)  # (B, 64, 12)
        occupied = flat.any(dim=-1)                      # (B, 64)
        piece_type = flat.argmax(dim=-1) + 1             # (B, 64) 1-12
        return piece_type * occupied.long()              # 0 for empty

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board: (B, 12, 8, 8) piece planes
            aux:   (B, 13) auxiliary features
        Returns:
            (B, 4096) logits over move space
        """
        B = board.shape[0]
        device = board.device

        # Square tokens: piece embedding + positional embedding + global context
        piece_idx = self._board_to_piece_indices(board)                # (B, 64)
        piece_emb = self.piece_embed(piece_idx)                        # (B, 64, d)
        sq_idx = torch.arange(64, device=device)
        pos_emb = self.pos_embed(sq_idx).unsqueeze(0)                  # (1, 64, d)
        global_emb = self.global_proj(aux).unsqueeze(1)                # (B, 1, d)

        tokens = piece_emb + pos_emb + global_emb                     # (B, 64, d)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                        # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                       # (B, 65, d)

        tokens = self.input_drop(self.input_norm(tokens))

        # Transformer encoder
        tokens = self.transformer(tokens)                              # (B, 65, d)

        # Use CLS output for move prediction
        cls_out = self.out_norm(tokens[:, 0])                          # (B, d)
        logits = self.move_head(cls_out)                               # (B, 4096)
        return logits
