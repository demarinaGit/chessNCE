"""Training loop for the Board Transformer move predictor."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .move_config import MoveConfig, get_move_config
from .move_dataset import build_move_loaders
from .move_model import BoardTransformer


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy."""
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for board, aux, target in loader:
        board = board.to(device)
        aux = aux.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(board, aux)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0

    for board, aux, target in loader:
        board = board.to(device)
        aux = aux.to(device)
        target = target.to(device)

        logits = model(board, aux)
        loss = criterion(logits, target)

        total_loss += loss.item() * target.size(0)
        correct_1 += (logits.argmax(1) == target).sum().item()
        correct_5 += topk_accuracy(logits, target, 5) * target.size(0)
        correct_10 += topk_accuracy(logits, target, 10) * target.size(0)
        total += target.size(0)

    return (
        total_loss / total,
        correct_1 / total,
        correct_5 / total,
        correct_10 / total,
    )


def train_move_model(cfg: MoveConfig | None = None) -> Path:
    """Full training run. Returns path to best checkpoint."""
    if cfg is None:
        cfg = get_move_config()

    torch.manual_seed(cfg.train.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = build_move_loaders(cfg.data, cfg.train)

    # Model
    model = BoardTransformer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs - cfg.train.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[cfg.train.warmup_epochs]
    )

    criterion = nn.CrossEntropyLoss()

    # Checkpoints
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_move_model.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_top1, val_top5, val_top10 = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} "
            f"val_top5={val_top5:.4f} val_top10={val_top10:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_top10": val_top10,
                    "config": cfg,
                },
                best_path,
            )
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(
                    f"  → Early stopping after {cfg.train.patience} epochs "
                    "without improvement"
                )
                break

    print(f"Training complete. Best checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    train_move_model()
