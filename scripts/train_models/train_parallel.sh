#!/bin/bash

# Parallel Training Launcher for ErrorSmith V2 with MPS GPU Acceleration
# 
# Usage:
#   ./train_parallel.sh                    # Run all (tech-heads + distill-mlp + distill-cnn)
#   ./train_parallel.sh --mode tech-heads  # Tech heads only
#   ./train_parallel.sh --mode distill-mlp # MLP distillation only
#   ./train_parallel.sh --mode all         # All improvements (default)
#   ./train_parallel.sh --tail tech_heads  # Show logs for tech_heads job
#
# Options:
#   --mode <mode>           Training mode: all, tech-heads, distill-mlp, distill-cnn, distill-both
#   --epochs-heads <N>      Epochs for tech heads (default: 20)
#   --epochs-distill <N>    Epochs for distillation (default: 50)
#   --batch-size <N>        Batch size (default: 32)
#   --max-parallel <N>      Max parallel jobs (default: 3)
#   --device <device>       Device: mps, cuda, cpu, auto (default: auto)
#   --tail <job>            Tail logs for specific job

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     ErrorSmith V2 Parallel Training with MPS GPU Acceleration         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
}

# Print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --mode <mode>           Training mode (default: all)
                          - all:           All improvements
                          - tech-heads:    Technology-specific heads only
                          - distill-mlp:   MLP student only
                          - distill-cnn:   CNN student only
                          - distill-both:  Both MLP and CNN students

  --epochs-heads <N>      Epochs for tech heads (default: 20)
  --epochs-distill <N>    Epochs for distillation (default: 50)
  --batch-size <N>        Batch size (default: 32)
  --max-parallel <N>      Max parallel jobs (default: 3)
  --device <device>       Device to use (default: auto)
                          - auto:  Auto-detect (MPS > CUDA > CPU)
                          - mps:   Apple Silicon (Metal Performance Shaders)
                          - cuda:  NVIDIA GPU
                          - cpu:   CPU only

  --tail <job>            Print last 20 lines of job log
  --help, -h              Show this help message

Examples:
  # Run all improvements in parallel on MPS GPU
  $0

  # Run only tech heads on MPS
  $0 --mode tech-heads --device mps

  # Run distillation with 50 epochs on auto-detected GPU
  $0 --mode distill-mlp --epochs-distill 50

  # View logs for tech_heads job
  $0 --tail tech_heads

EOF
}

# Parse arguments
MODE="all"
EPOCHS_HEADS="20"
EPOCHS_DISTILL="50"
BATCH_SIZE="32"
MAX_PARALLEL="3"
DEVICE="auto"
TAIL_JOB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --epochs-heads)
            EPOCHS_HEADS="$2"
            shift 2
            ;;
        --epochs-distill)
            EPOCHS_DISTILL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tail)
            TAIL_JOB="$2"
            shift 2
            ;;
        --help|-h)
            print_header
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Print header
print_header

# Check environment
echo -e "${YELLOW}Checking environment...${NC}"
if ! command -v venv_arm64/bin/python3 &> /dev/null; then
    echo -e "${RED}✗ Python environment not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python environment OK${NC}"

# Check repository structure
echo -e "${YELLOW}Checking repository structure...${NC}"
BASELINE_MODEL="$REPO_ROOT/models/error_predictor_v2.pt"
TRAINING_DATA="$REPO_ROOT/training_data/read_correction_v2/base_error"
SCRIPTS_DIR="$REPO_ROOT/scripts/train_models"

if [[ ! -f "$BASELINE_MODEL" ]]; then
    echo -e "${RED}✗ Baseline model not found: $BASELINE_MODEL${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Baseline model found${NC}"

if [[ ! -d "$TRAINING_DATA" ]]; then
    echo -e "${RED}✗ Training data not found: $TRAINING_DATA${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training data found${NC}"

if [[ ! -f "$SCRIPTS_DIR/run_parallel_training.py" ]]; then
    echo -e "${RED}✗ Parallel training script not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training scripts found${NC}"

# Change to repo root
cd "$REPO_ROOT"

# If tail mode, handle it
if [[ -n "$TAIL_JOB" ]]; then
    echo -e "${YELLOW}Fetching logs for: $TAIL_JOB${NC}"
    venv_arm64/bin/python3 "$SCRIPTS_DIR/run_parallel_training.py" \
        --tail "$TAIL_JOB"
    exit $?
fi

# Print configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Mode:              ${GREEN}$MODE${NC}"
echo -e "  Device:            ${GREEN}$DEVICE${NC}"
echo -e "  Tech heads epochs: ${GREEN}$EPOCHS_HEADS${NC}"
echo -e "  Distill epochs:    ${GREEN}$EPOCHS_DISTILL${NC}"
echo -e "  Batch size:        ${GREEN}$BATCH_SIZE${NC}"
echo -e "  Max parallel:      ${GREEN}$MAX_PARALLEL${NC}"
echo ""

# Run parallel training
echo -e "${YELLOW}Starting parallel training...${NC}"
venv_arm64/bin/python3 "$SCRIPTS_DIR/run_parallel_training.py" \
    --mode "$MODE" \
    --baseline "$BASELINE_MODEL" \
    --data "$TRAINING_DATA" \
    --output models \
    --device "$DEVICE" \
    --max-parallel "$MAX_PARALLEL" \
    --epochs-heads "$EPOCHS_HEADS" \
    --epochs-distill "$EPOCHS_DISTILL" \
    --batch-size "$BATCH_SIZE"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ✅ TRAINING COMPLETED SUCCESSFULLY                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                       ❌ TRAINING FAILED                              ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Check logs for details:${NC}"
    echo -e "  ${BLUE}models/logs/*.log${NC}"
fi

exit $EXIT_CODE
