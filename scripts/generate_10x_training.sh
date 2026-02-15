#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# StrandWeaver — 10× Expanded Training Data Generation
# ═══════════════════════════════════════════════════════════════════════
# Generates ~200 diverse diploid genomes (1 Mb each) across 8 batches
# with varied repeat densities, GC content, SV density, and read types.
#
# Uses --graph-only mode: reads are simulated in-memory for realistic
# coverage and overlap features, but NO FASTQ/FASTA files are written.
# This is ~10× faster and uses ~95% less disk than full mode.
#
# Expected runtime : ~2–4 hours on a MacBook (M-series)
# Expected disk    : ~3 GB (graph CSVs only)
#
# Usage:
#   chmod +x scripts/generate_10x_training.sh
#   nohup ./scripts/generate_10x_training.sh > training_10x.log 2>&1 &
#
# To monitor:
#   tail -f training_10x.log
#
# The script is resumable — it skips batches whose output dirs already
# contain the expected number of genome_* subdirectories.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

OUT_ROOT="training_data_10x"
LOG_FILE="training_10x.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    StrandWeaver · 10× Training Data Generation             ║"
echo "║    Started: $TIMESTAMP                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUT_ROOT"

# ── Helper ─────────────────────────────────────────────────────────
generate_batch() {
    local BATCH_NAME="$1"
    local NUM_GENOMES="$2"
    local GENOME_SIZE="$3"
    local REPEAT_DENSITY="$4"
    local GC_CONTENT="$5"
    local SNP_RATE="$6"
    local SV_DENSITY="$7"
    local SEED="$8"
    shift 8
    # Remaining args are --read-types / --coverage pairs
    local EXTRA_ARGS=("$@")

    local BATCH_DIR="${OUT_ROOT}/${BATCH_NAME}"

    # ── Resume check ──────────────────────────────────────────────
    if [ -d "$BATCH_DIR" ]; then
        local EXISTING=$(ls -d "$BATCH_DIR"/genome_* 2>/dev/null | wc -l | tr -d ' ')
        if [ "$EXISTING" -ge "$NUM_GENOMES" ]; then
            echo "  ⏭  Batch '$BATCH_NAME' already complete ($EXISTING genomes), skipping"
            return 0
        fi
        echo "  ↻  Batch '$BATCH_NAME' incomplete ($EXISTING/$NUM_GENOMES), regenerating"
        rm -rf "$BATCH_DIR"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch: $BATCH_NAME"
    echo "    Genomes       : $NUM_GENOMES × ${GENOME_SIZE} bp"
    echo "    Repeat density: $REPEAT_DENSITY"
    echo "    GC content    : $GC_CONTENT"
    echo "    SNP rate      : $SNP_RATE"
    echo "    SV density    : $SV_DENSITY"
    echo "    Seed          : $SEED"
    echo "    Read args     : ${EXTRA_ARGS[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local T_START=$(date +%s)

    python3 -m strandweaver.cli train generate-data \
        --genome-size "$GENOME_SIZE" \
        -n "$NUM_GENOMES" \
        --repeat-density "$REPEAT_DENSITY" \
        --gc-content "$GC_CONTENT" \
        --snp-rate "$SNP_RATE" \
        --sv-density "$SV_DENSITY" \
        --graph-training \
        --graph-only \
        --seed "$SEED" \
        "${EXTRA_ARGS[@]}" \
        -o "$BATCH_DIR"

    local T_END=$(date +%s)
    local ELAPSED=$(( T_END - T_START ))
    local MINS=$(( ELAPSED / 60 ))
    local SECS=$(( ELAPSED % 60 ))
    echo "  ✓ Batch '$BATCH_NAME' done in ${MINS}m ${SECS}s"
}

T_TOTAL_START=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════
#  BATCH DEFINITIONS — 8 batches, 200 genomes total
#
#  Strategy: vary repeat density (0.20–0.65), GC (0.32–0.55),
#  SV density (5e-6 – 5e-5), and read-type combos to maximise
#  the diversity the models see.
# ═══════════════════════════════════════════════════════════════════════

# ── Batch 1: Baseline multi-tech (moderate complexity) ─────────────
#    25 genomes, HiFi 30× + ONT 20× + Hi-C 15×
generate_batch "batch01_baseline" 25 1000000 \
    0.30 0.42 0.001 0.00001 100 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 30 --coverage 20 --coverage 15

# ── Batch 2: Repeat-rich (high repeat density) ────────────────────
#    25 genomes, HiFi 30× + ONT 25× + Hi-C 15×
generate_batch "batch02_repeat_rich" 25 1000000 \
    0.55 0.40 0.001 0.00002 200 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 30 --coverage 25 --coverage 15

# ── Batch 3: Very repeat-rich + high SV density ───────────────────
#    25 genomes, HiFi 25× + ONT 30× + Hi-C 20×
generate_batch "batch03_extreme_repeat" 25 1000000 \
    0.65 0.38 0.0015 0.00005 300 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 25 --coverage 30 --coverage 20

# ── Batch 4: GC-rich genomes (bird/reptile-like) ──────────────────
#    25 genomes, HiFi 30× + ONT 20× + Hi-C 10×
generate_batch "batch04_gc_rich" 25 1000000 \
    0.35 0.55 0.0008 0.00001 400 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 30 --coverage 20 --coverage 10

# ── Batch 5: AT-rich genomes (Plasmodium/AT-biased) ───────────────
#    25 genomes, HiFi 30× + ONT 25× + Hi-C 15×
generate_batch "batch05_at_rich" 25 1000000 \
    0.40 0.32 0.0012 0.00002 500 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 30 --coverage 25 --coverage 15

# ── Batch 6: Low-repeat + high SV (clean genome, structural events) ─
#    25 genomes, HiFi 30× + ONT 20× + Hi-C 15×
generate_batch "batch06_low_repeat_high_sv" 25 1000000 \
    0.20 0.42 0.001 0.00004 600 \
    --read-types hifi --read-types ont --read-types hic \
    --coverage 30 --coverage 20 --coverage 15

# ── Batch 7: Ultra-long routing data (HiFi + UL) ──────────────────
#    25 genomes, HiFi 25× + UL 15×
generate_batch "batch07_ultralong" 25 1000000 \
    0.40 0.42 0.001 0.00002 700 \
    --read-types hifi --read-types ultra_long \
    --coverage 25 --coverage 15

# ── Batch 8: Mixed-everything (max diversity) ─────────────────────
#    25 genomes, HiFi 20× + ONT 15× + Hi-C 10× + UL 10×
generate_batch "batch08_all_tech" 25 1000000 \
    0.45 0.44 0.0012 0.00003 800 \
    --read-types hifi --read-types ont --read-types hic --read-types ultra_long \
    --coverage 20 --coverage 15 --coverage 10 --coverage 10

# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════
T_TOTAL_END=$(date +%s)
T_TOTAL=$(( T_TOTAL_END - T_TOTAL_START ))
T_HRS=$(( T_TOTAL / 3600 ))
T_MIN=$(( (T_TOTAL % 3600) / 60 ))

TOTAL_GENOMES=$(find "$OUT_ROOT" -maxdepth 2 -type d -name "genome_*" | wc -l | tr -d ' ')
TOTAL_CSVS=$(find "$OUT_ROOT" -name "*.csv" | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$OUT_ROOT" | cut -f1)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             Data Generation Complete                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Total genomes : $TOTAL_GENOMES"
echo "║  Total CSVs    : $TOTAL_CSVS"
echo "║  Disk usage    : $TOTAL_SIZE"
echo "║  Wall time     : ${T_HRS}h ${T_MIN}m"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Retrain all models on expanded data ────────────────────────────
echo "Starting model retraining on expanded dataset..."
echo ""

python3 -m strandweaver.cli train run \
    --data-dir "$OUT_ROOT" \
    -o trained_models_10x \
    --n-folds 5 \
    --seed 42

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ All done!  Models saved to trained_models_10x/"
echo "═══════════════════════════════════════════════════════════════"
