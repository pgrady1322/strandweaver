#!/usr/bin/env python3
"""
Training Setup Guide and Configuration

This file provides:
1. Step-by-step setup instructions
2. Configuration management for training scripts
3. Quick-start commands
4. Data preparation steps
5. Troubleshooting tips
"""

import os
from pathlib import Path
import json

# ==============================================================================
# TRAINING SETUP GUIDE
# ==============================================================================

SETUP_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║         ERRORSMITH V2 MEDIUM-EFFORT IMPROVEMENTS - TRAINING SETUP          ║
╚════════════════════════════════════════════════════════════════════════════╝

THREE IMPROVEMENTS READY TO TRAIN:

1. ERROR ENSEMBLE (✓ Deploy Immediately)
   - No training needed
   - Integrates into inference pipeline
   - See: error_ensemble.py

2. PER-TECHNOLOGY FINE-TUNING (→ 1 hour GPU)
   - Fine-tunes 5 technology-specific heads
   - Expected: +2-3% accuracy
   - Script: train_tech_specific_heads.py

3. KNOWLEDGE DISTILLATION (→ 1-2 hours GPU)
   - Trains compact student model
   - Options: MLP (5K params) or CNN (500 params)
   - Expected: 6-10× speedup, 95-99% accuracy retention
   - Script: train_knowledge_distillation.py

════════════════════════════════════════════════════════════════════════════════
PREREQUISITE: BASELINE MODEL
════════════════════════════════════════════════════════════════════════════════

You need a trained ErrorSmith v2 baseline model. This should be at:
  models/error_predictor_v2.pt

If you don't have it:
  1. Run: python3 scripts/train_models/train_base_error_predictor_v2.py
  2. Point to its output location

════════════════════════════════════════════════════════════════════════════════
PREREQUISITE: TRAINING DATA
════════════════════════════════════════════════════════════════════════════════

Training data should be pickle files in:
  training_data/read_correction_v2/base_error/

Files should follow naming pattern:
  base_error_*_batch_*.pkl

Each pickle file contains a list of tuples:
  [(BaseContext, label), (BaseContext, label), ...]

If you don't have training data:
  1. Generate using your data preparation pipeline
  2. Point training scripts to correct directory with --data flag

════════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP SETUP
════════════════════════════════════════════════════════════════════════════════

STEP 1: Verify Your Environment
────────────────────────────────
  ✓ Python 3.8+ installed
  ✓ PyTorch installed (CPU or GPU)
  ✓ Required packages: numpy, scikit-learn, transformers
  
  Check:
    cd ~/Documents/GitHub_Repositories/smartassembler
    venv_arm64/bin/python3 -c "import torch; print(torch.__version__)"
    venv_arm64/bin/python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

STEP 2: Check Data and Models
────────────────────────────
  ✓ Baseline model exists:
      ls -lh models/error_predictor_v2.pt
  
  ✓ Training data exists:
      ls -lh training_data/read_correction_v2/base_error/ | head -20
      # Should show multiple base_error_*_batch_*.pkl files
  
  ✓ Output directories exist:
      mkdir -p models/tech_specific_heads
      mkdir -p models/student_models

STEP 3: Create Training Configuration (OPTIONAL)
─────────────────────────────────────────────
  Copy/modify scripts/train_models/training_config.py with your settings

STEP 4: Run Training Scripts

  ┌─ OPTION A: Per-Technology Fine-Tuning (Recommended First)
  │
  │  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
  │      --baseline models/error_predictor_v2.pt \\
  │      --data training_data/read_correction_v2/base_error \\
  │      --output models/tech_specific_heads \\
  │      --epochs 20 \\
  │      --batch-size 32 \\
  │      --learning-rate 1e-4 \\
  │      --device cuda
  │
  │  Expected time: 5-10 minutes (with GPU)
  │  Expected output: 5 head checkpoints + summary.json
  │
  
  ┌─ OPTION B: Knowledge Distillation
  │
  │  # For MLP student (recommended - good balance)
  │  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
  │      --teacher models/error_predictor_v2.pt \\
  │      --student-type mlp \\
  │      --data training_data/read_correction_v2/base_error \\
  │      --output models/student_models \\
  │      --epochs 50 \\
  │      --batch-size 32 \\
  │      --learning-rate 1e-3 \\
  │      --temperature 4.0 \\
  │      --device cuda
  │
  │  # OR for CNN student (smaller, faster)
  │  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
  │      --teacher models/error_predictor_v2.pt \\
  │      --student-type cnn \\
  │      ... (same other flags)
  │
  │  Expected time: 30-60 minutes (with GPU)
  │  Expected output: student_mlp.pt or student_cnn.pt + summary.json
  │
  
  ┌─ OPTION C: Both in Parallel
  │  Run both training scripts in separate terminals for maximum efficiency

════════════════════════════════════════════════════════════════════════════════
PARAMETER EXPLANATION
════════════════════════════════════════════════════════════════════════════════

Per-Technology Fine-Tuning:
  --baseline          Path to trained v2 model
  --data              Path to training data directory
  --output            Where to save head checkpoints
  --epochs            Epochs per technology (default: 20)
  --batch-size        Batch size (default: 32)
  --learning-rate     Learning rate for heads (default: 1e-4)
                      ↳ Lower = more stable, slower convergence
                      ↳ Higher = faster, but may be unstable
  --val-split         Fraction for validation (default: 0.2)
  --device            'cuda' or 'cpu' (default: auto-detect)

Knowledge Distillation:
  --teacher           Path to trained teacher (v2) model
  --student-type      'mlp' (5K params) or 'cnn' (500 params)
                      ↳ mlp: Better accuracy, slightly larger
                      ↳ cnn: Smaller, faster, good for edge deployment
  --epochs            Training epochs (default: 50)
  --learning-rate     Student learning rate (default: 1e-3)
  --temperature       Probability softening (default: 4.0)
                      ↳ Higher = smoother probabilities, softer targets
                      ↳ Lower = sharper probabilities
  --alpha             Hard target weight (default: 0.3)
                      ↳ 0.3 = 30% hard, 70% soft targets
                      ↳ Adjust based on training dynamics

════════════════════════════════════════════════════════════════════════════════
MONITORING TRAINING
════════════════════════════════════════════════════════════════════════════════

While training, you'll see:
  [2024-XX-XX HH:MM:SS] [INFO] Epoch [1/20] - Train Loss: 0.5234, Val Loss: 0.4123, Accuracy: 0.7234
  [2024-XX-XX HH:MM:SS] [INFO] Epoch [2/20] - Train Loss: 0.4512, Val Loss: 0.3876, Accuracy: 0.7521
  ...

Good signs:
  ✓ Validation loss decreasing
  ✓ Accuracy increasing
  ✓ Loss curves smooth

Warnings:
  ⚠ Loss not decreasing → Lower learning rate
  ⚠ Loss oscillating → Too high learning rate
  ⚠ GPU memory error → Reduce batch size

════════════════════════════════════════════════════════════════════════════════
AFTER TRAINING
════════════════════════════════════════════════════════════════════════════════

Per-Technology Fine-Tuning Output:
  models/tech_specific_heads/
  ├── head_ont_r9.pt          (checkpoint for ONT R9)
  ├── head_ont_r10.pt         (checkpoint for ONT R10)
  ├── head_hifi.pt            (checkpoint for PacBio HiFi)
  ├── head_illumina.pt        (checkpoint for Illumina)
  ├── head_adna.pt            (checkpoint for ancient DNA)
  └── training_summary.json   (metrics and hyperparameters)

Knowledge Distillation Output:
  models/student_models/
  ├── student_mlp.pt          (5K parameter student) OR
  ├── student_cnn.pt          (500 parameter student)
  └── training_summary.json   (metrics, speedup, accuracy)

════════════════════════════════════════════════════════════════════════════════
NEXT STEPS: DEPLOYMENT
════════════════════════════════════════════════════════════════════════════════

1. Per-Tech Heads:
   from scripts.train_models.tech_specific_heads import MultiHeadErrorPredictor
   
   model = MultiHeadErrorPredictor(
       baseline_encoder_path='models/error_predictor_v2.pt',
       heads_dir='models/tech_specific_heads',
   )
   
   prediction = model.predict(
       seq_features=features.sequence_features,
       tech_features=features.tech_features,
       tech_mask=features.tech_feature_mask,
       technology='ont_r10',
   )

2. Knowledge Distillation:
   from scripts.train_models.knowledge_distillation import StudentNetworkMLP
   import torch
   
   student = StudentNetworkMLP(input_dim=45, hidden_dim=64, output_dim=1)
   checkpoint = torch.load('models/student_models/student_mlp.pt')
   student.load_state_dict(checkpoint['state_dict'])
   
   prediction = torch.sigmoid(student(features))

════════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

Q: "CUDA out of memory"
A: Reduce batch size: --batch-size 16 or --batch-size 8

Q: "No module named 'train_models'"
A: Make sure you're in the smartassembler directory:
   cd ~/Documents/GitHub_Repositories/smartassembler
   Then run training script

Q: "Data directory not found"
A: Check path:
   ls training_data/read_correction_v2/base_error/
   If different, use --data flag with correct path

Q: "Model not found"
A: Check baseline location:
   ls models/error_predictor_v2.pt
   If different name/location, use --baseline or --teacher flag

Q: "Accuracy not improving"
A: Try:
   - Lower learning rate (--learning-rate 1e-5)
   - Train longer (--epochs 30 instead of 20)
   - Check if data distribution is reasonable

Q: "Process killed (out of memory)"
A: Probably system RAM exceeded:
   - Reduce batch size
   - Use fewer batches for testing first

════════════════════════════════════════════════════════════════════════════════
"""

print(SETUP_GUIDE)

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    "baseline_model": "models/error_predictor_v2.pt",
    "training_data_dir": "training_data/read_correction_v2/base_error",
    
    "tech_specific_heads": {
        "output_dir": "models/tech_specific_heads",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "val_split": 0.2,
        "technologies": ["ont_r9", "ont_r10", "hifi", "illumina", "adna"],
    },
    
    "knowledge_distillation": {
        "output_dir": "models/student_models",
        "student_types": ["mlp", "cnn"],
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "temperature": 4.0,
        "alpha": 0.3,
        "val_split": 0.2,
    },
    
    "common": {
        "device": "cuda",  # or "cpu"
        "num_workers": 4,
        "seed": 42,
    },
}

# ==============================================================================
# QUICK START COMMANDS
# ==============================================================================

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          QUICK START COMMANDS                             ║
╚════════════════════════════════════════════════════════════════════════════╝

SETUP:
  cd ~/Documents/GitHub_Repositories/smartassembler

OPTION 1: Fine-tune technology-specific heads (5 heads, ~30 min GPU)
  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
      --baseline models/error_predictor_v2.pt \\
      --data training_data/read_correction_v2/base_error \\
      --output models/tech_specific_heads \\
      --device cuda

OPTION 2: Distill knowledge into compact MLP (5K params, ~1 hour GPU)
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type mlp \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --device cuda

OPTION 3: Distill knowledge into ultra-compact CNN (500 params, ~1 hour GPU)
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type cnn \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --device cuda

RUN ALL (parallel execution in separate terminals):
  Terminal 1: python3 scripts/train_models/train_tech_specific_heads.py --device cuda
  Terminal 2: python3 scripts/train_models/train_knowledge_distillation.py --student-type mlp --device cuda
  Terminal 3: python3 scripts/train_models/train_knowledge_distillation.py --student-type cnn --device cuda

════════════════════════════════════════════════════════════════════════════════
"""

print(QUICK_START)

# ==============================================================================
# SAVE CONFIGURATION
# ==============================================================================

def save_config(output_path: Path = None):
    """Save default configuration to JSON."""
    if output_path is None:
        output_path = Path(__file__).parent / "training_config.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    
    print(f"✓ Configuration saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    
    # Optionally save config
    if len(sys.argv) > 1 and sys.argv[1] == '--save-config':
        save_config()
    
    print("\n✅ Setup guide printed. Ready to begin training!")
