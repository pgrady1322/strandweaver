#!/bin/bash
# Master script to train both read correction AI models

set -e

echo "=================================================="
echo "Training Read Correction AI Models"
echo "=================================================="
echo ""
echo "Training data: ALL 6 technologies"
echo "  - ONT R9"
echo "  - ONT R10"  
echo "  - PacBio HiFi"
echo "  - PacBio CLR"
echo "  - Illumina"
echo "  - Ancient DNA"
echo ""
echo "Models to train:"
echo "  1. Adaptive K-mer Selector (XGBoost)"
echo "  2. Base Error Predictor (1D CNN with PyTorch MPS)"
echo ""
echo "=================================================="
echo ""

# Activate environment
source venv_arm64/bin/activate

# Create logs directory
mkdir -p logs/model_training

# Train K-mer Selector
echo "🚀 [1/2] Training Adaptive K-mer Selector..."
echo "  Model: XGBoost"
echo "  Data: 523,000 examples (6 technologies)"
echo "  GPU: CPU-based (XGBoost MPS not supported)"
echo ""

python scripts/train_models/train_kmer_selector.py \
    --data training_data/read_correction/adaptive_kmer \
    --output models/kmer_selector.pkl \
    --model-type xgboost \
    2>&1 | tee logs/model_training/kmer_selector_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo ""

# Train Base Error Predictor
echo "🚀 [2/2] Training Base Error Predictor..."
echo "  Model: 1D CNN"
echo "  Data: 240,000 examples (6 technologies)"
echo "  GPU: MPS-accelerated (Apple Silicon)"
echo ""

python scripts/train_models/train_base_error_predictor.py \
    --data training_data/read_correction/base_error \
    --output models/base_error_predictor.pt \
    --model-type cnn \
    --epochs 20 \
    --batch-size 256 \
    2>&1 | tee logs/model_training/base_error_predictor_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "✅ All models trained successfully!"
echo "=================================================="
echo ""
echo "Trained models:"
echo "  - models/kmer_selector.pkl (XGBoost)"
echo "  - models/base_error_predictor.pt (PyTorch CNN)"
echo ""
echo "Logs saved to: logs/model_training/"
echo ""
