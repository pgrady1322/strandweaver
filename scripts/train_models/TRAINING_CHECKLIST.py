#!/usr/bin/env python3
"""
TRAINING SETUP SUMMARY - QUICK START CHECKLIST

Use this as a quick reference before starting training.
"""

CHECKLIST = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    TRAINING SETUP QUICK START CHECKLIST                   ║
╚════════════════════════════════════════════════════════════════════════════╝

BEFORE YOU START - Pre-Flight Checklist (5 minutes)
═════════════════════════════════════════════════════════════════════════════

□ Step 1: Navigate to repository
  cd ~/Documents/GitHub_Repositories/smartassembler
  pwd  # Should show: .../smartassembler

□ Step 2: Validate environment
  venv_arm64/bin/python3 scripts/train_models/validate_training_setup.py
  Look for: "✅ READY TO TRAIN!"

□ Step 3: Check baseline model exists
  ls -lh models/error_predictor_v2.pt
  Should show file size > 100MB

□ Step 4: Check training data exists
  ls training_data/read_correction_v2/base_error/ | head
  Should show: base_error_*_batch_*.pkl files

□ Step 5: Create output directories
  mkdir -p models/tech_specific_heads
  mkdir -p models/student_models


YOUR TRAINING OPTIONS
═════════════════════════════════════════════════════════════════════════════

Option A: PER-TECHNOLOGY FINE-TUNING (Fastest - 30 min)
────────────────────────────────────────────────────────────
Expected gain: +2-3% accuracy
Training time: 5-10 minutes (GPU)
Output: 5 technology-specific heads

COMMAND:
  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
      --baseline models/error_predictor_v2.pt \\
      --data training_data/read_correction_v2/base_error \\
      --output models/tech_specific_heads \\
      --epochs 20 \\
      --batch-size 32 \\
      --learning-rate 1e-4 \\
      --device cuda

Run this option if: You want quick accuracy gains


Option B: KNOWLEDGE DISTILLATION - MLP (Best balance - 1 hour)
───────────────────────────────────────────────────────────────
Expected gain: 7× speedup, 95-99% accuracy retention
Training time: 30-60 minutes (GPU)
Output: 5K parameter student model (18 KB)

COMMAND:
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type mlp \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --epochs 50 \\
      --batch-size 32 \\
      --learning-rate 1e-3 \\
      --temperature 4.0 \\
      --device cuda

Run this option if: You want speed + good accuracy


Option C: KNOWLEDGE DISTILLATION - CNN (Smallest - 1 hour)
────────────────────────────────────────────────────────────
Expected gain: 10× speedup, 90-95% accuracy retention
Training time: 30-60 minutes (GPU)
Output: 500 parameter student model (5 KB)

COMMAND:
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type cnn \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --epochs 50 \\
      --batch-size 32 \\
      --learning-rate 1e-3 \\
      --temperature 4.0 \\
      --device cuda

Run this option if: You need ultra-compact model for edge devices


Option D: MAXIMUM IMPACT (All improvements - 2 hours GPU)
───────────────────────────────────────────────────────────

TERMINAL 1 (Tech heads):
  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
      --baseline models/error_predictor_v2.pt \\
      --data training_data/read_correction_v2/base_error \\
      --output models/tech_specific_heads \\
      --device cuda

TERMINAL 2 (Distillation MLP):
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type mlp \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --device cuda

TERMINAL 3 (Distillation CNN):
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type cnn \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --device cuda

Run this option if: You have multi-GPU system or want everything


DURING TRAINING - What to Expect
═════════════════════════════════════════════════════════════════════════════

You'll see output like:
  [2024-01-15 14:32:10] [INFO] Epoch [1/20] - Train Loss: 0.5234, Val Loss: 0.4123, Accuracy: 0.7234
  [2024-01-15 14:32:15] [INFO] Epoch [2/20] - Train Loss: 0.4512, Val Loss: 0.3876, Accuracy: 0.7412
  [2024-01-15 14:32:20] [INFO] ✓ Saved checkpoint: models/tech_specific_heads/head_ont_r9.pt

GOOD SIGNS:
  ✓ Validation loss decreasing
  ✓ Accuracy improving
  ✓ Checkpoint files being saved
  ✓ No error messages

BAD SIGNS (need to fix):
  ✗ CUDA out of memory → Reduce batch size to 16 or 8
  ✗ Loss not decreasing → Lower learning rate (1e-5)
  ✗ Loss oscillating → Too high learning rate
  ✗ Early stopping → Train longer (epochs 30+)


AFTER TRAINING - What You Get
═════════════════════════════════════════════════════════════════════════════

PER-TECHNOLOGY HEADS:
  models/tech_specific_heads/
  ├── head_ont_r9.pt        (technology-specific head for ONT R9)
  ├── head_ont_r10.pt       (technology-specific head for ONT R10)
  ├── head_hifi.pt          (technology-specific head for PacBio HiFi)
  ├── head_illumina.pt      (technology-specific head for Illumina)
  ├── head_adna.pt          (technology-specific head for ancient DNA)
  └── training_summary.json (metrics)

KNOWLEDGE DISTILLATION (MLP):
  models/student_models/
  ├── student_mlp.pt        (5K parameter student, ~18 KB)
  └── training_summary.json (speedup metrics)

KNOWLEDGE DISTILLATION (CNN):
  models/student_models/
  ├── student_cnn.pt        (500 parameter student, ~5 KB)
  └── training_summary.json (speedup metrics)


TROUBLESHOOTING - Common Issues
═════════════════════════════════════════════════════════════════════════════

ISSUE: "CUDA out of memory"
SOLUTION:
  Option 1: Reduce batch size
    --batch-size 16  (or 8)
  Option 2: Use CPU (slower)
    --device cpu
  Option 3: Close other applications using GPU
    Check with: nvidia-smi

ISSUE: "No module named 'train_models'"
SOLUTION:
  Make sure you're in the right directory:
    cd ~/Documents/GitHub_Repositories/smartassembler
  Then verify path to models/data exist

ISSUE: "Data directory not found"
SOLUTION:
  Check data location:
    ls training_data/read_correction_v2/base_error/
  If different, add --data /correct/path to command

ISSUE: "Accuracy not improving"
SOLUTION:
  1. Lower learning rate: --learning-rate 1e-5
  2. Train longer: --epochs 30
  3. Check data quality
  4. Look at validation loss (not training)

ISSUE: "Process killed" or "Segmentation fault"
SOLUTION:
  Probably system memory exceeded:
  - Reduce batch size to 8 or 4
  - Run on GPU instead of CPU
  - Close other applications


NEXT STEPS AFTER TRAINING
═════════════════════════════════════════════════════════════════════════════

1. Verify Training Completed Successfully:
   Check summary files:
     cat models/tech_specific_heads/training_summary.json
     cat models/student_models/training_summary.json

2. Deploy Models (see COMPLETE_REFERENCE.py for code examples):
   
   For per-tech heads:
     from tech_specific_heads import MultiHeadErrorPredictor
     model = MultiHeadErrorPredictor(...)
   
   For knowledge distillation:
     from knowledge_distillation import StudentNetworkMLP
     student = StudentNetworkMLP(...)

3. Test on Validation Set:
   - Compare accuracy vs baseline
   - Check inference speed improvements
   - Validate on multiple sequencing technologies

4. Prepare for Production:
   - A/B test improvements
   - Monitor for performance drift
   - Set up logging and metrics


QUICK REFERENCE - File Locations
═════════════════════════════════════════════════════════════════════════════

Training Scripts:
  - train_tech_specific_heads.py       Main per-tech training
  - train_knowledge_distillation.py    Main distillation training
  - validate_training_setup.py         Pre-flight checklist
  - TRAINING_SETUP.py                  Detailed setup guide
  - COMPLETE_REFERENCE.py              Full reference documentation

Models:
  - models/error_predictor_v2.pt       Baseline (teacher)
  - models/tech_specific_heads/        Output: per-tech heads
  - models/student_models/             Output: student models

Training Data:
  - training_data/read_correction_v2/base_error/   Training data location

Support Modules:
  - error_ensemble.py                  Ensemble module (ready to deploy)
  - tech_specific_heads.py             Per-tech head classes
  - knowledge_distillation.py          Distillation classes
  - tech_aware_feature_builder.py      Feature generation
  - data_augmentation.py               MixUp, cutout, etc.


DECISION TREE - Which Option to Choose?
═════════════════════════════════════════════════════════════════════════════

Do you want to start training RIGHT NOW?
│
├─ YES → Do you have GPU available?
│  │
│  ├─ YES → Option A (per-tech heads, 30 min)
│  │        └─ Fastest path to +2-3% accuracy
│  │
│  └─ NO → Option D (all 3, ~2 hours with GPU)
│           └─ Maximize improvements while GPU runs
│
└─ NO → Read the docs first
         - TRAINING_SETUP.py
         - COMPLETE_REFERENCE.py


═════════════════════════════════════════════════════════════════════════════════
                            READY TO TRAIN?
                          Copy and paste this command:

            venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
                --baseline models/error_predictor_v2.pt \\
                --data training_data/read_correction_v2/base_error \\
                --output models/tech_specific_heads \\
                --device cuda

═════════════════════════════════════════════════════════════════════════════════
"""

print(CHECKLIST)
