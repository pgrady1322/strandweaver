#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  Package training data for Google Colab upload
# ──────────────────────────────────────────────────────────────────
#  Creates a compressed tarball of graph CSVs for upload to Colab.
#  Only CSVs are included (no temporary files or logs).
#
#  Usage:
#    ./scripts/package_training_data.sh
#
#  Output:
#    graph_csvs.tar.gz  (~300-400 MB compressed from 1.4 GB)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

DATA_DIR="training_data_10x"
OUTPUT="graph_csvs.tar.gz"

if [ ! -d "$DATA_DIR" ]; then
    echo "✗ Training data directory '${DATA_DIR}' not found."
    exit 1
fi

CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')
DATA_SIZE=$(du -sh "$DATA_DIR" | cut -f1)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Package Training Data for Colab Upload                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Source    : ${DATA_DIR}/  (${DATA_SIZE}, ${CSV_COUNT} CSVs) "
echo "║  Output    : ${OUTPUT}                                       "
echo "╚══════════════════════════════════════════════════════════════╝"
echo

echo "  Compressing..."
# Use find + tar to reliably include only CSVs (portable across macOS/Linux)
find "$DATA_DIR" -name '*.csv' -print0 | tar czf "$OUTPUT" --null -T -

TARBALL_SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  ✓ Created ${OUTPUT} (${TARBALL_SIZE})"
echo
echo "  Upload to Colab:"
echo "    1. Open notebooks/XGBoost_Retraining_Colab.ipynb in Colab"
echo "    2. Run the Setup cells"
echo "    3. Upload ${OUTPUT} when prompted"
echo "    4. Run all remaining cells"
echo
echo "  Compression ratio: ${DATA_SIZE} → ${TARBALL_SIZE}"
