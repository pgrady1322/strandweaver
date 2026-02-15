#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  StrandWeaver · Full Model Retrain (Hybrid Resampling Pipeline)
# ──────────────────────────────────────────────────────────────────
#  Retrains all 5 model families against the 200-genome training
#  dataset using the updated hybrid resampling strategy for class-
#  imbalanced tasks.
#
#  Usage:
#    ./scripts/retrain_all_models.sh
#
#  Prerequisite:
#    - training_data_10x/ must exist (1.4 GB, 200 genomes)
#    - sw-testing conda env or equivalent with xgboost, sklearn
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

DATA_DIR="training_data_10x"
OUTPUT_DIR="trained_models_10x"
BACKUP_DIR="trained_models_10x_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="retrain_$(date +%Y%m%d_%H%M%S).log"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     StrandWeaver · Full Model Retrain (Hybrid Pipeline)    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Data dir  : ${DATA_DIR}                                    "
echo "║  Output dir: ${OUTPUT_DIR}                                  "
echo "║  Log file  : ${LOG_FILE}                                    "
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── Verify training data exists ──────────────────────────────────
if [ ! -d "$DATA_DIR" ]; then
    echo "✗ Training data directory '${DATA_DIR}' not found."
    echo "  Run generate_10x_training.sh first."
    exit 1
fi

CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')
echo "  Found ${CSV_COUNT} CSV files in ${DATA_DIR}/"

if [ "$CSV_COUNT" -lt 100 ]; then
    echo "✗ Expected ≥100 CSVs (got ${CSV_COUNT}). Is the dataset complete?"
    exit 1
fi

# ── Backup existing models if present ────────────────────────────
if [ -d "$OUTPUT_DIR" ]; then
    echo "  Backing up existing models → ${BACKUP_DIR}/"
    mv "$OUTPUT_DIR" "$BACKUP_DIR"
fi

# ── Run training ─────────────────────────────────────────────────
echo
echo "  Starting training... (logging to ${LOG_FILE})"
echo

python3 -m strandweaver.user_training.train_models \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --models edge_ai path_gnn diploid_ai ul_routing sv_ai \
    --n-folds 5 \
    --val-split 0.15 \
    --seed 42 \
    -v \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=$?

echo
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "✗ Training failed (exit code ${TRAIN_EXIT}). Check ${LOG_FILE}."
    # Restore backup
    if [ -d "$BACKUP_DIR" ]; then
        echo "  Restoring backup models..."
        rm -rf "$OUTPUT_DIR" 2>/dev/null
        mv "$BACKUP_DIR" "$OUTPUT_DIR"
    fi
    exit 1
fi

# ── Verify outputs ───────────────────────────────────────────────
echo
echo "  Verifying model outputs..."
EXPECTED_FILES=(
    "edgewarden/edgewarden_hifi.pkl"
    "edgewarden/scaler_hifi.pkl"
    "edgewarden/training_metadata.json"
    "pathgnn/pathgnn_scorer.pkl"
    "pathgnn/training_metadata.json"
    "diploid/diploid_model.pkl"
    "diploid/training_metadata.json"
    "ul_routing/ul_routing_model.pkl"
    "ul_routing/training_metadata.json"
    "sv_detector/sv_detector_model.pkl"
    "sv_detector/training_metadata.json"
    "training_report.json"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "${OUTPUT_DIR}/${f}" ]; then
        echo "  ✗ Missing: ${OUTPUT_DIR}/${f}"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "  ✗ ${MISSING} expected files missing!"
    exit 1
fi

echo "  ✓ All expected model files present"
echo

# ── Print comparison vs backup ───────────────────────────────────
if [ -d "$BACKUP_DIR" ]; then
    echo "  Model sizes (new vs backup):"
    for subdir in edgewarden pathgnn diploid ul_routing sv_detector; do
        NEW_SIZE=$(du -sh "${OUTPUT_DIR}/${subdir}" 2>/dev/null | cut -f1)
        OLD_SIZE=$(du -sh "${BACKUP_DIR}/${subdir}" 2>/dev/null | cut -f1)
        echo "    ${subdir}: ${OLD_SIZE:-N/A} → ${NEW_SIZE:-N/A}"
    done
    echo
    echo "  Backup preserved at: ${BACKUP_DIR}/"
    echo "  Remove with: rm -rf ${BACKUP_DIR}"
fi

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Retrain complete! Models saved to ${OUTPUT_DIR}/         "
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "  Next steps:"
echo "    1. Review ${LOG_FILE} for detailed metrics"
echo "    2. Compare metrics with training_report.json"
echo "    3. Port models to main repo: cp -r ${OUTPUT_DIR}/ ../strandweaver/${OUTPUT_DIR}/"
echo "    4. Commit all changes"
