"""Evaluation for the Board Transformer move predictor."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .move_config import MoveConfig, get_move_config
from .move_dataset import build_move_datasets, _split_chunk_paths
from .move_model import BoardTransformer


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


@torch.no_grad()
def evaluate_move_model(
    checkpoint_path: str | Path | None = None,
    cfg: MoveConfig | None = None,
) -> dict:
    """Evaluate the move prediction model on the test set.

    Returns dict with top-k accuracies and per-ELO breakdown.
    """
    if cfg is None:
        cfg = get_move_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if checkpoint_path is None:
        checkpoint_path = cfg.train.checkpoint_dir / "best_move_model.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BoardTransformer(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load test data (with ELO info for per-bucket analysis)
    _, _, test_ds = build_move_datasets(cfg.data)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0
    )

    # Also load raw test chunks to get ELO ratings
    chunk_paths = sorted(cfg.data.processed_dir.glob("chunk_*.npz"))
    n = len(chunk_paths)
    n_train = max(1, int(n * cfg.data.train_ratio))
    n_val = max(1, int(n * cfg.data.val_ratio))
    test_paths = chunk_paths[n_train + n_val:]
    if not test_paths:
        test_paths = chunk_paths[n_train:n_train + n_val][-1:]

    # Collect ELOs from test chunks
    elos_list = []
    for p in test_paths:
        data = np.load(p)
        mask = data["labels"] == 1
        elos_list.append(data["elos"][mask])
    test_elos = np.concatenate(elos_list)

    # Run predictions
    all_preds = []
    all_targets = []
    all_logits = []

    for board, aux, target in test_loader:
        board = board.to(device)
        aux = aux.to(device)
        logits = model(board, aux)

        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(target)

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Overall metrics
    top1 = topk_accuracy(all_logits, all_targets, 1)
    top5 = topk_accuracy(all_logits, all_targets, 5)
    top10 = topk_accuracy(all_logits, all_targets, 10)

    print(f"\n{'='*50}")
    print(f"Test Results ({len(all_targets)} samples)")
    print(f"{'='*50}")
    print(f"  Top-1 accuracy:  {top1:.4f} ({top1*100:.1f}%)")
    print(f"  Top-5 accuracy:  {top5:.4f} ({top5*100:.1f}%)")
    print(f"  Top-10 accuracy: {top10:.4f} ({top10*100:.1f}%)")

    results = {"top1": top1, "top5": top5, "top10": top10, "n_samples": len(all_targets)}

    # Per-ELO bucket analysis
    if len(test_elos) == len(all_targets):
        valid = test_elos > 0
        if valid.sum() > 0:
            valid_elos = test_elos[valid]
            valid_logits = all_logits[valid]
            valid_targets = all_targets[valid]

            n_buckets = 5
            bucket_edges = np.percentile(valid_elos, np.linspace(0, 100, n_buckets + 1))
            bucket_edges[-1] += 1

            elo_results = []
            print(f"\nPer-ELO Bucket Analysis:")
            print(f"{'ELO Range':>18} | {'Top-1':>6} | {'Top-5':>6} | {'Top-10':>6} | {'N':>6}")
            print("-" * 60)

            for i in range(n_buckets):
                lo, hi = bucket_edges[i], bucket_edges[i + 1]
                mask = (valid_elos >= lo) & (valid_elos < hi)
                if mask.sum() == 0:
                    continue
                b_logits = valid_logits[mask]
                b_targets = valid_targets[mask]
                b_top1 = topk_accuracy(b_logits, b_targets, 1)
                b_top5 = topk_accuracy(b_logits, b_targets, 5)
                b_top10 = topk_accuracy(b_logits, b_targets, 10)
                label = f"{int(lo)}–{int(hi)}"
                print(f"  {label:>16} | {b_top1:.4f} | {b_top5:.4f} | {b_top10:.4f} | {mask.sum():>6}")
                elo_results.append({
                    "elo_lo": int(lo), "elo_hi": int(hi),
                    "top1": b_top1, "top5": b_top5, "top10": b_top10,
                    "n": int(mask.sum()),
                })

            results["elo_buckets"] = elo_results
            _plot_elo_results(elo_results, results, cfg.train.results_dir)

    return results


def _plot_elo_results(elo_results, overall, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['elo_lo']}–{r['elo_hi']}" for r in elo_results]
    top1s = [r["top1"] for r in elo_results]
    top5s = [r["top5"] for r in elo_results]
    top10s = [r["top10"] for r in elo_results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, top1s, width, label="Top-1", color="#2196F3")
    ax.bar(x, top5s, width, label="Top-5", color="#4CAF50")
    ax.bar(x + width, top10s, width, label="Top-10", color="#FF9800")

    ax.set_xlabel("ELO Rating Bucket")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Move Prediction Accuracy by ELO "
        f"(Overall: Top-1={overall['top1']:.1%}, Top-5={overall['top5']:.1%})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = plot_dir / "move_prediction_by_elo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    evaluate_move_model()
