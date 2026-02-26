# chessNCE

A binary classifier that distinguishes **actual chess moves from historical games** from **random legal moves**, using a Noise Contrastive Estimation (NCE) framing.

## Overview

Given a board position, can we tell whether a move was played by a human or chosen at random from legal moves? This project trains a CNN-based neural network to answer that question using ~100K games from the [Lichess open database](https://database.lichess.org/).

**Input representation:**
- Board state as an 8×8×12 binary tensor (one plane per piece-type/color)
- Auxiliary features: side to move, castling rights, en passant (13 bits)
- Move encoded as (from\_square, to\_square) indices, each embedded

**Architecture:** CNN board encoder → move embeddings → FC classifier → P(historical)

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/demarinaGit/chessNCE.git
cd chessNCE
pip install -r requirements.txt
```

## Usage

### 1. Download data

Fetch a Lichess monthly PGN dump (default: January 2024):

```bash
./scripts/download_data.sh          # or: ./scripts/download_data.sh 2024-03
```

### 2. Preprocess

Parse PGN games into training-ready `.npz` chunks:

```bash
python -m src.data_pipeline data/raw/lichess_db_standard_rated_2024-01.pgn
```

### 3. Train

```bash
python -m src.train
```

Trains with early stopping. Best checkpoint saved to `checkpoints/best_model.pt`.

### 4. Evaluate

```bash
python -m src.evaluate
```

Prints accuracy, AUC-ROC, classification report, and saves plots to `results/`.

## Project Structure

```
chessNCE/
├── src/
│   ├── config.py          # Hyperparameters and paths
│   ├── data_pipeline.py   # PGN → (board, move, label) extraction
│   ├── dataset.py         # PyTorch Dataset + DataLoader
│   ├── model.py           # ChessMoveClassifier (CNN + embeddings)
│   ├── train.py           # Training loop
│   └── evaluate.py        # Metrics, ROC curve, confusion matrix
├── scripts/
│   └── download_data.sh   # Lichess PGN downloader
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Configuration

All hyperparameters are in `src/config.py`. Key defaults:

| Parameter | Value |
|-----------|-------|
| Max games | 100,000 |
| Batch size | 1,024 |
| Learning rate | 1e-3 |
| Epochs | 20 (early stopping, patience=5) |
| CNN channels | 32 → 64 → 128 |
| Dropout | 0.3 |
