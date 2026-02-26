"""Training loop for the ChessMoveClassifier."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from .config import get_config, Config
from .dataset import build_loaders
from .model import ChessMoveClassifier


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for board, aux, from_sq, to_sq, label in loader:
        board = board.to(device)
        aux = aux.to(device)
        from_sq = from_sq.to(device)
        to_sq = to_sq.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(board, aux, from_sq, to_sq).squeeze(1)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        preds = (logits > 0).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on a dataset. Returns (avg_loss, accuracy, auc_roc)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for board, aux, from_sq, to_sq, label in loader:
        board = board.to(device)
        aux = aux.to(device)
        from_sq = from_sq.to(device)
        to_sq = to_sq.to(device)
        label = label.to(device)

        logits = model(board, aux, from_sq, to_sq).squeeze(1)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        preds = (logits > 0).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

        probs = torch.sigmoid(logits).cpu()
        all_probs.extend(probs.tolist())
        all_labels.extend(label.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


def train(cfg: Config | None = None) -> Path:
    """Full training run. Returns path to best checkpoint.

    Args:
        cfg: Configuration. Uses defaults if None.

    Returns:
        Path to the best model checkpoint.
    """
    if cfg is None:
        cfg = get_config()

    torch.manual_seed(cfg.train.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = build_loaders(cfg.data, cfg.train)

    # Model
    model = ChessMoveClassifier(cfg.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer, scheduler, loss
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Checkpoint directory
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }, best_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"  → Early stopping after {cfg.train.patience} epochs without improvement")
                break

    print(f"Training complete. Best checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    train()
