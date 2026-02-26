#!/usr/bin/env bash
# Download a Lichess monthly PGN database for training data.
# Usage: ./scripts/download_data.sh [YYYY-MM]
# Default: 2024-01 (a reasonably sized month)

set -euo pipefail

MONTH="${1:-2024-01}"
BASE_URL="https://database.lichess.org/standard"
FILENAME="lichess_db_standard_rated_${MONTH}.pgn.zst"
OUTPUT_DIR="data/raw"

mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/${FILENAME%.zst}" ]; then
    echo "Already have decompressed PGN: $OUTPUT_DIR/${FILENAME%.zst}"
    exit 0
fi

if [ -f "$OUTPUT_DIR/$FILENAME" ]; then
    echo "Compressed file exists, skipping download."
else
    echo "Downloading $FILENAME from Lichess..."
    curl -L --progress-bar -o "$OUTPUT_DIR/$FILENAME" "$BASE_URL/$FILENAME"
fi

echo "Decompressing $FILENAME..."
if command -v zstd &> /dev/null; then
    zstd -d "$OUTPUT_DIR/$FILENAME" -o "$OUTPUT_DIR/${FILENAME%.zst}" --rm
elif command -v python3 &> /dev/null; then
    python3 -c "
import zstandard, sys
with open('$OUTPUT_DIR/$FILENAME', 'rb') as fin, open('$OUTPUT_DIR/${FILENAME%.zst}', 'wb') as fout:
    dctx = zstandard.ZstdDecompressor()
    dctx.copy_stream(fin, fout)
"
    rm "$OUTPUT_DIR/$FILENAME"
else
    echo "Error: need zstd or python3+zstandard to decompress."
    exit 1
fi

echo "Done! PGN saved to $OUTPUT_DIR/${FILENAME%.zst}"
