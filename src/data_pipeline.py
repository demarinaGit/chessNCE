"""Data pipeline: PGN → (board_tensor, aux_features, move, label) pairs.

Reads Lichess PGN files, iterates through games/positions, and produces
balanced positive (actual move) / negative (random legal move) examples.
Saves results as .npz chunks for efficient loading.

Supports parallel processing via multiprocessing for large PGN files.
"""

import multiprocessing as mp
import os
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
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
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
    """
    white_elo, black_elo = _get_elo(game)
    samples = []
    board = game.board()
    for move in game.mainline_moves():
        legal_moves = list(board.legal_moves)
        if len(legal_moves) < 2:
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

    path = output_dir / f"chunk_{chunk_idx:06d}.npz"
    np.savez_compressed(path, boards=boards, auxs=auxs,
                        from_sqs=from_sqs, to_sqs=to_sqs, labels=labels,
                        elos=elos)
    return path


# ---------------------------------------------------------------------------
# PGN byte-offset scanning: find game boundaries for parallel processing
# ---------------------------------------------------------------------------

def scan_game_offsets(pgn_path: Path, max_games: int = 0) -> list[int]:
    """Scan a PGN file and return byte offsets where each game starts.

    Games in Lichess PGN files start with '[Event ' at the beginning of a line,
    preceded by a blank line (or file start).
    """
    offsets = [0]
    with open(pgn_path, "rb") as f:
        for line in f:
            if line.startswith(b"[Event ") and offsets[-1] != f.tell() - len(line):
                offsets.append(f.tell() - len(line))
                if max_games and len(offsets) > max_games:
                    break
    return offsets


def _partition_offsets(offsets: list[int], num_workers: int) -> list[tuple[int, int, int]]:
    """Divide game offsets into roughly equal partitions for workers.

    Returns list of (start_byte, end_byte, num_games) tuples.
    end_byte == -1 means read to EOF.
    """
    n = len(offsets)
    chunk_size = max(1, n // num_workers)
    partitions = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        if start_idx >= n:
            break
        end_idx = min((i + 1) * chunk_size, n)
        start_byte = offsets[start_idx]
        end_byte = offsets[end_idx] if end_idx < n else -1
        num_games = end_idx - start_idx
        partitions.append((start_byte, end_byte, num_games))
    return partitions


# ---------------------------------------------------------------------------
# Worker function for parallel processing
# ---------------------------------------------------------------------------

def _worker_process_range(args: tuple) -> tuple[int, int]:
    """Process a byte range of a PGN file. Runs in a subprocess.

    Args:
        args: (pgn_path, start_byte, end_byte, chunk_base_idx, chunk_size,
               output_dir, seed)

    Returns:
        (games_processed, chunks_written)
    """
    pgn_path, start_byte, end_byte, chunk_base_idx, chunk_size, output_dir, seed = args
    random.seed(seed)
    output_dir = Path(output_dir)

    samples: list[dict] = []
    chunk_idx = chunk_base_idx
    game_count = 0

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        f.seek(start_byte)
        while True:
            if end_byte > 0 and f.tell() >= end_byte:
                break
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_samples = extract_samples_from_game(game)
            samples.extend(game_samples)
            game_count += 1

            while len(samples) >= chunk_size:
                save_chunk(samples[:chunk_size], chunk_idx, output_dir)
                samples = samples[chunk_size:]
                chunk_idx += 1

    # Save remaining samples
    if samples:
        save_chunk(samples, chunk_idx, output_dir)
        chunk_idx += 1

    return game_count, chunk_idx - chunk_base_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pgn(pgn_path: str | Path, cfg: DataConfig | None = None) -> None:
    """Process a PGN file into .npz chunks using parallel workers.

    Args:
        pgn_path: Path to the PGN file.
        cfg: Data configuration. Uses defaults if None.
    """
    if cfg is None:
        cfg = get_config().data

    pgn_path = Path(pgn_path)
    num_workers = cfg.num_pipeline_workers

    print(f"Scanning game boundaries in {pgn_path.name}...")
    offsets = scan_game_offsets(pgn_path, max_games=cfg.max_games)
    total_games = len(offsets)
    print(f"Found {total_games:,} games")

    if total_games == 0:
        print("No games found.")
        return

    # Single-worker fast path
    if num_workers <= 1:
        _process_pgn_single(pgn_path, cfg)
        return

    partitions = _partition_offsets(offsets, num_workers)
    print(f"Distributing across {len(partitions)} workers")

    # Pre-allocate chunk index ranges so workers don't collide
    # Estimate ~120 samples/game, 2 samples/position ≈ 60 positions/game
    est_samples_per_game = 120
    worker_args = []
    chunk_base = 0
    for i, (start_byte, end_byte, num_games) in enumerate(partitions):
        est_chunks = max(1, (num_games * est_samples_per_game) // cfg.chunk_size + 1)
        worker_args.append((
            str(pgn_path), start_byte, end_byte, chunk_base,
            cfg.chunk_size, str(cfg.processed_dir),
            cfg.random_seed + i,
        ))
        chunk_base += est_chunks * 2  # generous spacing to avoid collisions

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=len(partitions)) as pool:
        results = pool.map(_worker_process_range, worker_args)

    total_processed = sum(r[0] for r in results)
    total_chunks = sum(r[1] for r in results)
    print(f"Processed {total_processed:,} games → {total_chunks:,} chunks in {cfg.processed_dir}")


def _process_pgn_single(pgn_path: Path, cfg: DataConfig) -> None:
    """Single-worker fallback for small files or debugging."""
    random.seed(cfg.random_seed)

    samples: list[dict] = []
    chunk_idx = 0
    game_count = 0
    max_games = cfg.max_games if cfg.max_games > 0 else float("inf")

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc="Processing games", unit="game")
        while game_count < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_samples = extract_samples_from_game(game)
            samples.extend(game_samples)
            game_count += 1
            pbar.update(1)

            while len(samples) >= cfg.chunk_size:
                save_chunk(samples[:cfg.chunk_size], chunk_idx, cfg.processed_dir)
                samples = samples[cfg.chunk_size:]
                chunk_idx += 1

        pbar.close()

    if samples:
        save_chunk(samples, chunk_idx, cfg.processed_dir)
        chunk_idx += 1

    print(f"Processed {game_count:,} games → {chunk_idx} chunks in {cfg.processed_dir}")


if __name__ == "__main__":
    import sys

    pgn_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/lichess_db_standard_rated_2025-01.pgn"
    process_pgn(pgn_file)
