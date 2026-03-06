"""Self-play: the Board Transformer plays both sides of a chess game."""

import json
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from .data_pipeline import board_to_aux, board_to_tensor
from .move_config import get_move_config
from .move_model import BoardTransformer


def load_model(checkpoint_path: str | Path | None = None) -> tuple[BoardTransformer, torch.device]:
    cfg = get_move_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = cfg.train.checkpoint_dir / "best_move_model.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BoardTransformer(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def get_legal_move_mask(board: chess.Board, device: torch.device) -> torch.Tensor:
    """Build a boolean mask over the 4096 move space for current legal moves."""
    mask = torch.zeros(4096, dtype=torch.bool, device=device)
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        mask[idx] = True
    return mask


def _maybe_promote(board: chess.Board, move: chess.Move) -> chess.Move:
    """Add queen promotion if a pawn reaches the back rank."""
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        dest_rank = chess.square_rank(move.to_square)
        if (piece.color == chess.WHITE and dest_rank == 7) or \
           (piece.color == chess.BLACK and dest_rank == 0):
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


@torch.no_grad()
def choose_move(
    model: BoardTransformer,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.0,
) -> chess.Move:
    """Pick a move. temperature=0 → argmax, >0 → softmax sampling."""
    board_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    aux_t = torch.from_numpy(board_to_aux(board)).unsqueeze(0).to(device)

    logits = model(board_t, aux_t).squeeze(0)  # (4096,)

    legal_mask = get_legal_move_mask(board, device)
    logits[~legal_mask] = float("-inf")

    if temperature <= 0:
        idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()

    from_sq, to_sq = divmod(idx, 64)
    move = chess.Move(from_sq, to_sq)
    return _maybe_promote(board, move)


def play_game(
    model: BoardTransformer,
    device: torch.device,
    max_moves: int = 200,
    temperature: float = 0.0,
) -> chess.pgn.Game:
    """Play a full game with the model controlling both sides."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Board Transformer Self-Play"
    game.headers["White"] = "BoardTransformer"
    game.headers["Black"] = "BoardTransformer"

    node = game
    move_num = 0

    while not board.is_game_over() and move_num < max_moves:
        move = choose_move(model, board, device, temperature=temperature)
        board.push(move)
        node = node.add_variation(move)
        move_num += 1

    if board.is_game_over():
        result = board.result()
    else:
        result = "1/2-1/2"
    game.headers["Result"] = result

    return game


def _classify_opening(moves: list[chess.Move]) -> str:
    """Classify the opening based on the first few moves (ECO-lite)."""
    san_moves = []
    board = chess.Board()
    for m in moves[:6]:
        san_moves.append(board.san(m))
        board.push(m)

    first = san_moves[0] if san_moves else ""
    second = san_moves[1] if len(san_moves) > 1 else ""
    third = san_moves[2] if len(san_moves) > 2 else ""
    fourth = san_moves[3] if len(san_moves) > 3 else ""

    if first == "e4":
        if second == "e5":
            if third == "Nf3":
                if fourth == "Nc6":
                    fifth = san_moves[4] if len(san_moves) > 4 else ""
                    if fifth == "Bb5":
                        return "Ruy Lopez"
                    if fifth == "Bc4":
                        return "Italian Game"
                    if fifth == "d4":
                        return "Scotch Game"
                    return "King's Knight Opening"
                if fourth == "Nf6":
                    return "Petrov's Defense"
                return "King's Knight Opening"
            if third == "f4":
                return "King's Gambit"
            if third == "Bc4":
                return "Bishop's Opening"
            return "Open Game (1.e4 e5)"
        if second == "c5":
            return "Sicilian Defense"
        if second == "e6":
            return "French Defense"
        if second == "c6":
            return "Caro-Kann Defense"
        if second == "d5":
            return "Scandinavian Defense"
        if second == "Nf6":
            return "Alekhine's Defense"
        if second == "d6":
            return "Pirc Defense"
        if second == "g6":
            return "Modern Defense"
        return f"1.e4 {second}"
    if first == "d4":
        if second == "d5":
            if third == "c4":
                return "Queen's Gambit"
            return "Queen's Pawn Game"
        if second == "Nf6":
            if third == "c4":
                if fourth == "g6":
                    return "King's Indian Defense"
                if fourth == "e6":
                    return "Nimzo/Queen's Indian"
                return "Indian Defense"
            return "Indian Defense"
        if second == "f5":
            return "Dutch Defense"
        return f"1.d4 {second}"
    if first == "Nf3":
        return "Reti Opening"
    if first == "c4":
        return "English Opening"
    return f"1.{first}"


def run_batch(n_games: int = 100, temperature: float = 0.5):
    """Play n_games and print aggregate statistics."""
    print(f"Loading model...")
    model, device = load_model()
    print(f"Playing {n_games} games (temperature={temperature}) on {device}...\n")

    results = Counter()
    terminations = Counter()
    openings = Counter()
    game_lengths = []

    for i in range(n_games):
        game = play_game(model, device, temperature=temperature)
        moves = list(game.mainline_moves())
        n_plies = len(moves)
        n_full_moves = (n_plies + 1) // 2

        results[game.headers["Result"]] += 1
        game_lengths.append(n_full_moves)

        board = game.board()
        for m in moves:
            board.push(m)
        outcome = board.outcome()
        if outcome:
            terminations[outcome.termination.name] += 1
        else:
            terminations["MOVE_LIMIT"] += 1

        opening = _classify_opening(moves)
        openings[opening] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_games} games complete...")

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS — {n_games} self-play games (temperature={temperature})")
    print(f"{'='*60}")

    print(f"\n--- Outcomes ---")
    for res, count in results.most_common():
        pct = count / n_games * 100
        print(f"  {res:>10}: {count:>3} ({pct:.0f}%)")

    print(f"\n--- Termination Types ---")
    for term, count in terminations.most_common():
        pct = count / n_games * 100
        print(f"  {term:>15}: {count:>3} ({pct:.0f}%)")

    print(f"\n--- Game Length (full moves) ---")
    lengths = sorted(game_lengths)
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {lengths[len(lengths)//2]}")

    # Histogram buckets
    buckets = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 150), (151, 200)]
    print(f"\n  Length distribution:")
    for lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l <= hi)
        if count > 0:
            bar = "█" * count
            print(f"    {lo:>3}–{hi:<3}: {count:>3} {bar}")

    print(f"\n--- Openings ---")
    for opening, count in openings.most_common():
        pct = count / n_games * 100
        print(f"  {opening:<30} {count:>3} ({pct:.0f}%)")


def main():
    if "--batch" in sys.argv:
        n = 100
        temp = 0.5
        for i, arg in enumerate(sys.argv):
            if arg == "-n" and i + 1 < len(sys.argv):
                n = int(sys.argv[i + 1])
            if arg == "-t" and i + 1 < len(sys.argv):
                temp = float(sys.argv[i + 1])
        run_batch(n_games=n, temperature=temp)
        return

    print("Loading model...")
    model, device = load_model()
    print(f"Playing self-play game on {device}...\n")

    game = play_game(model, device)

    print(game)
    print()

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    outcome = board.outcome()
    if outcome:
        print(f"Result: {game.headers['Result']} — {outcome.termination.name}")
    else:
        print(f"Result: {game.headers['Result']} (move limit)")


if __name__ == "__main__":
    main()
