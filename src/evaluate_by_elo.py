"""Evaluation sliced by ELO rating: AUC and accuracy per rating decile."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from .config import Config, get_config
from .model import ChessMoveClassifier


def load_test_chunks_with_elo(cfg: Config) -> dict[str, np.ndarray]:
    """Load test-split chunks and return arrays including ELO."""
    chunk_paths = sorted(cfg.data.processed_dir.glob("chunk_*.npz"))
    n = len(chunk_paths)
    n_train = max(1, int(n * cfg.data.train_ratio))
    n_val = max(1, int(n * cfg.data.val_ratio))
    test_paths = chunk_paths[n_train + n_val:]
    if not test_paths:
        test_paths = chunk_paths[-1:]

    arrays = {k: [] for k in ("boards", "auxs", "from_sqs", "to_sqs", "labels", "elos")}
    for p in test_paths:
        data = np.load(p)
        for k in arrays:
            arrays[k].append(data[k])

    return {k: np.concatenate(v) for k, v in arrays.items()}


@torch.no_grad()
def predict(model, boards, auxs, from_sqs, to_sqs, device, batch_size=1024):
    """Run model and return probabilities."""
    model.eval()
    probs = []
    n = len(boards)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        b = torch.from_numpy(boards[i:j]).to(device)
        a = torch.from_numpy(auxs[i:j]).to(device)
        f = torch.tensor(from_sqs[i:j], dtype=torch.long, device=device)
        t = torch.tensor(to_sqs[i:j], dtype=torch.long, device=device)
        logits = model(b, a, f, t).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def evaluate_by_elo(
    checkpoint_path: str | Path | None = None,
    cfg: Config | None = None,
    n_buckets: int = 10,
) -> None:
    if cfg is None:
        cfg = get_config()
    if checkpoint_path is None:
        checkpoint_path = cfg.train.checkpoint_dir / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ChessMoveClassifier(cfg.model).to(device)
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load test data with ELO
    data = load_test_chunks_with_elo(cfg)
    labels = data["labels"]
    elos = data["elos"]
    print(f"Test samples: {len(labels)}, ELO range: {elos.min()}–{elos.max()}")

    # Filter out samples with missing ELO (0)
    valid = elos > 0
    if not valid.all():
        n_dropped = (~valid).sum()
        print(f"Dropping {n_dropped} samples with missing ELO")
        for k in data:
            data[k] = data[k][valid]
        labels = data["labels"]
        elos = data["elos"]

    # Get predictions
    probs = predict(model, data["boards"], data["auxs"],
                    data["from_sqs"], data["to_sqs"], device)
    preds = (probs > 0.5).astype(float)

    # Compute decile boundaries so each bucket has ~equal samples
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(elos, percentiles)
    # Deduplicate boundaries for ties
    boundaries = np.unique(boundaries)

    results = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i < len(boundaries) - 2:
            mask = (elos >= lo) & (elos < hi)
        else:
            mask = (elos >= lo) & (elos <= hi)

        n_samples = mask.sum()
        if n_samples < 10:
            continue

        bucket_labels = labels[mask]
        bucket_probs = probs[mask]
        bucket_preds = preds[mask]

        acc = accuracy_score(bucket_labels, bucket_preds)
        # AUC needs both classes present
        if len(np.unique(bucket_labels)) == 2:
            auc = roc_auc_score(bucket_labels, bucket_probs)
        else:
            auc = float("nan")

        results.append({
            "lo": int(lo), "hi": int(hi),
            "n": int(n_samples), "acc": acc, "auc": auc,
        })

    # Print table
    print(f"\n{'ELO Range':>15} {'Samples':>8} {'Accuracy':>10} {'AUC':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['lo']:>6}–{r['hi']:<6} {r['n']:>8} {r['acc']:>10.4f} {r['auc']:>8.4f}")

    # Overall for reference
    overall_acc = accuracy_score(labels, preds)
    overall_auc = roc_auc_score(labels, probs)
    print("-" * 45)
    print(f"{'Overall':>15} {len(labels):>8} {overall_acc:>10.4f} {overall_auc:>8.4f}")

    # Plot
    plot_dir = Path("results")
    plot_dir.mkdir(exist_ok=True)
    _plot_elo_metrics(results, overall_auc, overall_acc, plot_dir)


def _plot_elo_metrics(results, overall_auc, overall_acc, plot_dir):
    mid_elos = [(r["lo"] + r["hi"]) / 2 for r in results]
    aucs = [r["auc"] for r in results]
    accs = [r["acc"] for r in results]
    labels = [f"{r['lo']}–{r['hi']}\n(n={r['n']:,})" for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC by ELO
    bars1 = ax1.bar(range(len(results)), aucs, color="#4C72B0", edgecolor="white")
    ax1.axhline(y=overall_auc, color="red", linestyle="--", alpha=0.7,
                label=f"Overall AUC = {overall_auc:.4f}")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("AUC by ELO Rating Bucket")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.45, max(aucs + [overall_auc]) + 0.05)
    for bar, val in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Accuracy by ELO
    bars2 = ax2.bar(range(len(results)), accs, color="#55A868", edgecolor="white")
    ax2.axhline(y=overall_acc, color="red", linestyle="--", alpha=0.7,
                label=f"Overall Acc = {overall_acc:.4f}")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by ELO Rating Bucket")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.45, max(accs + [overall_acc]) + 0.05)
    for bar, val in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Chess Move Classifier — Performance by Player Rating", fontsize=13)
    plt.tight_layout()
    save_path = plot_dir / "elo_analysis.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nELO analysis plot saved to {save_path}")


if __name__ == "__main__":
    evaluate_by_elo()
