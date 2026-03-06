"""Play the Board Transformer against Stockfish at various ELO levels to estimate rating."""

import sys
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import torch

from .self_play import choose_move, load_model


def play_vs_stockfish(
    model,
    device: torch.device,
    stockfish_path: str = "stockfish",
    stockfish_elo: int = 1000,
    model_color: chess.Color = chess.WHITE,
    temperature: float = 0.0,
    max_moves: int = 200,
    move_time: float = 0.05,
) -> tuple[str, str, int]:
    """Play one game: model vs Stockfish.

    Returns (result_str, termination, n_moves).
    result_str is from the model's perspective: 'win', 'loss', or 'draw'.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

    board = chess.Board()
    n_moves = 0

    while not board.is_game_over() and n_moves < max_moves:
        if board.turn == model_color:
            move = choose_move(model, board, device, temperature=temperature)
        else:
            result = engine.play(board, chess.engine.Limit(time=move_time))
            move = result.move
        board.push(move)
        n_moves += 1

    engine.quit()

    outcome = board.outcome()
    n_full = (n_moves + 1) // 2

    if outcome is None:
        return "draw", "MOVE_LIMIT", n_full

    termination = outcome.termination.name
    if outcome.winner is None:
        return "draw", termination, n_full
    elif outcome.winner == model_color:
        return "win", termination, n_full
    else:
        return "loss", termination, n_full


def estimate_elo(
    n_games_per_level: int = 20,
    elo_levels: list[int] | None = None,
    temperature: float = 0.1,
):
    """Play the model at multiple Stockfish ELO levels and estimate its rating."""
    if elo_levels is None:
        elo_levels = [1320, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

    print("Loading model...")
    model, device = load_model()
    print(f"Device: {device}\n")

    level_results = {}

    for elo in elo_levels:
        wins, draws, losses = 0, 0, 0
        print(f"--- Playing vs Stockfish ELO {elo} ({n_games_per_level} games) ---")

        for i in range(n_games_per_level):
            # Alternate colors
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            result, term, n_moves = play_vs_stockfish(
                model, device,
                stockfish_elo=elo,
                model_color=color,
                temperature=temperature,
            )
            if result == "win":
                wins += 1
            elif result == "draw":
                draws += 1
            else:
                losses += 1

        score = (wins + 0.5 * draws) / n_games_per_level
        level_results[elo] = {
            "wins": wins, "draws": draws, "losses": losses,
            "score": score, "n": n_games_per_level,
        }
        print(f"  W/D/L: {wins}/{draws}/{losses}  Score: {score:.0%}")
        print()

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY — Model vs Stockfish (temp={temperature})")
    print(f"{'='*60}")
    print(f"{'SF ELO':>8} | {'W':>3} {'D':>3} {'L':>3} | {'Score':>6} | {'Bar'}")
    print("-" * 60)

    estimated_elo = None
    prev_elo = None
    prev_score = None

    for elo in elo_levels:
        r = level_results[elo]
        bar_w = "█" * r["wins"]
        bar_d = "▓" * r["draws"]
        bar_l = "░" * r["losses"]
        print(f"  {elo:>6} | {r['wins']:>3} {r['draws']:>3} {r['losses']:>3} | {r['score']:>5.0%} | {bar_w}{bar_d}{bar_l}")

        # Find where score crosses 50%
        if prev_score is not None and estimated_elo is None:
            if prev_score >= 0.5 and r["score"] < 0.5:
                # Linear interpolation
                frac = (prev_score - 0.5) / (prev_score - r["score"])
                estimated_elo = prev_elo + frac * (elo - prev_elo)
        prev_elo = elo
        prev_score = r["score"]

    # Edge cases
    if estimated_elo is None:
        if all(level_results[e]["score"] >= 0.5 for e in elo_levels):
            estimated_elo = elo_levels[-1]
            print(f"\n  Model wins ≥50% at all levels — ELO is ≥ {estimated_elo}")
        elif all(level_results[e]["score"] < 0.5 for e in elo_levels):
            estimated_elo = elo_levels[0]
            print(f"\n  Model loses >50% at all levels — ELO is ≤ {estimated_elo}")

    if estimated_elo is not None:
        print(f"\n  ★ Estimated model ELO: ~{int(estimated_elo)}")

    return level_results, estimated_elo


if __name__ == "__main__":
    n = 20
    temp = 0.1
    levels = None
    for i, arg in enumerate(sys.argv):
        if arg == "-n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        if arg == "-t" and i + 1 < len(sys.argv):
            temp = float(sys.argv[i + 1])
        if arg == "--levels" and i + 1 < len(sys.argv):
            levels = [int(x) for x in sys.argv[i + 1].split(",")]
    estimate_elo(n_games_per_level=n, elo_levels=levels, temperature=temp)
