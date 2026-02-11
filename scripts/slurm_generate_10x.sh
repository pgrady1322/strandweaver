#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# StrandWeaver — 10× Training Data Generation (SLURM Array Job)
# ═══════════════════════════════════════════════════════════════════════
# Runs all 8 batches IN PARALLEL as a SLURM array on Mantis.
# Each task generates 25 × 1 Mb genomes → finishes in ~3–4 hours.
#
# Usage on Mantis:
#   cd /core/labs/Oneill/pgrady/strandweaver
#   sbatch scripts/slurm_generate_10x.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/batch_*_${SLURM_ARRAY_JOB_ID}.stdout
#
# After all 8 finish, submit the retrain job:
#   sbatch scripts/slurm_retrain.sh   (or run manually)
# ═══════════════════════════════════════════════════════════════════════
#SBATCH --job-name=sw-datagen
#SBATCH --array=1-8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=08:00:00
#SBATCH -o logs/batch_%a_%A.stdout
#SBATCH -e logs/batch_%a_%A.stderr

set -euo pipefail

# ── Setup ──────────────────────────────────────────────────────────
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs

# Load conda/mamba — adjust this if your Mantis setup differs
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
    module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
fi

# Activate the strandweaver environment
# Change this name to match your Mantis environment
conda activate sw-testing 2>/dev/null || conda activate strandweaver 2>/dev/null || {
    echo "ERROR: Could not activate conda env. Create one with:"
    echo "  conda create -n sw-testing python=3.11 && conda activate sw-testing"
    echo "  cd $PROJ_DIR && pip install -e '.[all]'"
    exit 1
}

OUT_ROOT="training_data_10x"
mkdir -p "$OUT_ROOT"

# ── Batch definitions (indexed by SLURM_ARRAY_TASK_ID) ────────────
# Format:  NAME  REPEAT_DENSITY  GC  SNP_RATE  SV_DENSITY  SEED  READ_ARGS...
case $SLURM_ARRAY_TASK_ID in
    1)
        BATCH="batch01_baseline"
        REPEAT=0.30; GC=0.42; SNP=0.001; SV=0.00001; SEED=100
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 30 --coverage 20 --coverage 15"
        ;;
    2)
        BATCH="batch02_repeat_rich"
        REPEAT=0.55; GC=0.40; SNP=0.001; SV=0.00002; SEED=200
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 30 --coverage 25 --coverage 15"
        ;;
    3)
        BATCH="batch03_extreme_repeat"
        REPEAT=0.65; GC=0.38; SNP=0.0015; SV=0.00005; SEED=300
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 25 --coverage 30 --coverage 20"
        ;;
    4)
        BATCH="batch04_gc_rich"
        REPEAT=0.35; GC=0.55; SNP=0.0008; SV=0.00001; SEED=400
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 30 --coverage 20 --coverage 10"
        ;;
    5)
        BATCH="batch05_at_rich"
        REPEAT=0.40; GC=0.32; SNP=0.0012; SV=0.00002; SEED=500
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 30 --coverage 25 --coverage 15"
        ;;
    6)
        BATCH="batch06_low_repeat_high_sv"
        REPEAT=0.20; GC=0.42; SNP=0.001; SV=0.00004; SEED=600
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --coverage 30 --coverage 20 --coverage 15"
        ;;
    7)
        BATCH="batch07_ultralong"
        REPEAT=0.40; GC=0.42; SNP=0.001; SV=0.00002; SEED=700
        READ_ARGS="--read-types hifi --read-types ultra_long --coverage 25 --coverage 15"
        ;;
    8)
        BATCH="batch08_all_tech"
        REPEAT=0.45; GC=0.44; SNP=0.0012; SV=0.00003; SEED=800
        READ_ARGS="--read-types hifi --read-types ont --read-types hic --read-types ultra_long --coverage 20 --coverage 15 --coverage 10 --coverage 10"
        ;;
    *)
        echo "ERROR: Invalid array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

BATCH_DIR="${OUT_ROOT}/${BATCH}"

# ── Resume check ───────────────────────────────────────────────────
if [ -d "$BATCH_DIR" ]; then
    EXISTING=$(ls -d "$BATCH_DIR"/genome_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$EXISTING" -ge 25 ]; then
        echo "⏭  Batch '$BATCH' already complete ($EXISTING genomes), skipping"
        exit 0
    fi
fi

# ── Run ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SLURM Job $SLURM_JOB_ID  ·  Task $SLURM_ARRAY_TASK_ID"
echo "  Batch: $BATCH"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START=$(date +%s)

python3 -m strandweaver.cli train generate-data \
    --genome-size 1000000 \
    -n 25 \
    --repeat-density "$REPEAT" \
    --gc-content "$GC" \
    --snp-rate "$SNP" \
    --sv-density "$SV" \
    --seed "$SEED" \
    --graph-training \
    $READ_ARGS \
    -o "$BATCH_DIR"

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Batch '$BATCH' complete in ${ELAPSED} minutes"
echo "  Genomes: $(ls -d "$BATCH_DIR"/genome_* 2>/dev/null | wc -l | tr -d ' ')"
echo "  Size:    $(du -sh "$BATCH_DIR" | cut -f1)"
echo "  End:     $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
