"""Evaluation: load best checkpoint, compute metrics, generate plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .config import get_config, Config
from .dataset import build_loaders
from .model import ChessMoveClassifier


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on a DataLoader and return (labels, probabilities)."""
    model.eval()
    all_labels = []
    all_probs = []

    for board, aux, from_sq, to_sq, label in loader:
        board = board.to(device)
        aux = aux.to(device)
        from_sq = from_sq.to(device)
        to_sq = to_sq.to(device)

        logits = model(board, aux, from_sq, to_sq).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(label.numpy().tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: Path) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Chess Move Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, save_path: Path) -> None:
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["Random", "Historical"]
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate(
    checkpoint_path: str | Path | None = None,
    cfg: Config | None = None,
) -> dict:
    """Run full evaluation on the test set.

    Args:
        checkpoint_path: Path to model checkpoint. Defaults to checkpoints/best_model.pt.
        cfg: Configuration. Uses defaults if None.

    Returns:
        Dict with accuracy, auc, classification_report, etc.
    """
    if cfg is None:
        cfg = get_config()
    if checkpoint_path is None:
        checkpoint_path = cfg.train.checkpoint_dir / "best_model.pt"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ChessMoveClassifier(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # Data — only need test loader
    _, _, test_loader = build_loaders(cfg.data, cfg.train)

    # Predictions
    labels, probs = collect_predictions(model, test_loader, device)
    preds = (probs > 0.5).astype(float)

    # Metrics
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    report = classification_report(labels, preds, target_names=["Random", "Historical"])

    print(f"\n{'='*50}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC-ROC:  {auc:.4f}")
    print(f"{'='*50}")
    print(report)

    # Plots
    plot_dir = Path("results")
    plot_dir.mkdir(exist_ok=True)
    plot_roc_curve(labels, probs, plot_dir / "roc_curve.png")
    plot_confusion_matrix(labels, preds, plot_dir / "confusion_matrix.png")

    return {"accuracy": acc, "auc": auc, "report": report}


if __name__ == "__main__":
    evaluate()
