"""Data pipeline: PGN → (board_tensor, aux_features, move, label) pairs.

Reads Lichess PGN files, iterates through games/positions, and produces
balanced positive (actual move) / negative (random legal move) examples.
Saves results as .npz chunks for efficient loading.
"""

import random
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from .config import DataConfig, get_config

# Piece-to-plane mapping: 12 planes for the 8x8x12 board tensor
# Order: wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK
_PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a chess.Board to an 8x8x12 binary numpy array."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = _PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[plane, rank, file] = 1.0
    return tensor


def board_to_aux(board: chess.Board) -> np.ndarray:
    """Extract auxiliary features: side to move (1), castling (4), en passant file (8)."""
    aux = np.zeros(13, dtype=np.float32)
    # Side to move: 1 = white, 0 = black
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0
    # Castling rights
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # En passant file (one-hot, 8 bits)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        aux[5 + ep_file] = 1.0
    return aux


def _get_elo(game: chess.pgn.Game) -> tuple[int, int]:
    """Extract WhiteElo and BlackElo from game headers, defaulting to 0."""
    try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
    except ValueError:
        white_elo = 0
    try:
        black_elo = int(game.headers.get("BlackElo", "0"))
    except ValueError:
        black_elo = 0
    return white_elo, black_elo


def extract_samples_from_game(game: chess.pgn.Game) -> list[dict]:
    """Extract (board, actual_move, random_move) pairs from a single game.

    Returns a list of dicts, each with:
        board_tensor, aux, from_sq, to_sq, label, elo
    Two entries per position: label=1 for actual move, label=0 for random.
    elo is the rating of the player whose turn it is.
    """
    white_elo, black_elo = _get_elo(game)
    samples = []
    board = game.board()
    for move in game.mainline_moves():
        legal_moves = list(board.legal_moves)
        if len(legal_moves) < 2:
            # Only one legal move — skip (no meaningful negative)
            board.push(move)
            continue

        tensor = board_to_tensor(board)
        aux = board_to_aux(board)
        elo = white_elo if board.turn == chess.WHITE else black_elo

        # Positive sample: actual move
        samples.append({
            "board": tensor,
            "aux": aux,
            "from_sq": move.from_square,
            "to_sq": move.to_square,
            "label": 1,
            "elo": elo,
        })

        # Negative sample: random legal move (excluding actual move)
        neg_moves = [m for m in legal_moves if m != move]
        neg_move = random.choice(neg_moves)
        samples.append({
            "board": tensor,
            "aux": aux,
            "from_sq": neg_move.from_square,
            "to_sq": neg_move.to_square,
            "label": 0,
            "elo": elo,
        })

        board.push(move)

    return samples


def save_chunk(samples: list[dict], chunk_idx: int, output_dir: Path) -> Path:
    """Save a list of sample dicts as a compressed .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    boards = np.stack([s["board"] for s in samples])
    auxs = np.stack([s["aux"] for s in samples])
    from_sqs = np.array([s["from_sq"] for s in samples], dtype=np.int64)
    to_sqs = np.array([s["to_sq"] for s in samples], dtype=np.int64)
    labels = np.array([s["label"] for s in samples], dtype=np.float32)
    elos = np.array([s["elo"] for s in samples], dtype=np.int32)

    path = output_dir / f"chunk_{chunk_idx:04d}.npz"
    np.savez_compressed(path, boards=boards, auxs=auxs,
                        from_sqs=from_sqs, to_sqs=to_sqs, labels=labels,
                        elos=elos)
    return path


def process_pgn(pgn_path: str | Path, cfg: DataConfig | None = None) -> None:
    """Process a PGN file into .npz chunks.

    Args:
        pgn_path: Path to the PGN file.
        cfg: Data configuration. Uses defaults if None.
    """
    if cfg is None:
        cfg = get_config().data

    random.seed(cfg.random_seed)
    pgn_path = Path(pgn_path)

    samples: list[dict] = []
    chunk_idx = 0
    game_count = 0

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        pbar = tqdm(total=cfg.max_games, desc="Processing games", unit="game")
        while game_count < cfg.max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_samples = extract_samples_from_game(game)
            samples.extend(game_samples)
            game_count += 1
            pbar.update(1)

            # Flush chunk if big enough
            if len(samples) >= cfg.chunk_size:
                save_chunk(samples[:cfg.chunk_size], chunk_idx, cfg.processed_dir)
                samples = samples[cfg.chunk_size:]
                chunk_idx += 1

        pbar.close()

    # Save remaining samples
    if samples:
        save_chunk(samples, chunk_idx, cfg.processed_dir)
        chunk_idx += 1

    print(f"Processed {game_count} games → {chunk_idx} chunks in {cfg.processed_dir}")


if __name__ == "__main__":
    import sys

    pgn_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/lichess_db_standard_rated_2024-01.pgn"
    process_pgn(pgn_file)
