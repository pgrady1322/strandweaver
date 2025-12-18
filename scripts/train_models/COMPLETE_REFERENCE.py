#!/usr/bin/env python3
"""
ERRORSMITH V2 MEDIUM-EFFORT IMPROVEMENTS
Complete Training and Deployment Reference

This document provides everything you need to understand and deploy:
1. Per-Technology Fine-Tuning Heads
2. Knowledge Distillation Students
3. Error Ensemble Heuristics
"""

REFERENCE = """
╔════════════════════════════════════════════════════════════════════════════╗
║    ERRORSMITH V2 MEDIUM-EFFORT IMPROVEMENTS - COMPLETE REFERENCE           ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
PART 1: OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

THREE IMPROVEMENTS TO DEPLOY (in order of effort):

1. ERROR ENSEMBLE with Heuristics (✓ READY - No training)
   ├─ Purpose: Combine neural predictions with rule-based heuristics
   ├─ Expected gain: +2-4% accuracy
   ├─ Training needed: None
   ├─ Deployment: Immediate
   ├─ Code: scripts/train_models/error_ensemble.py
   ├─ Module: ErrorEnsemble class
   └─ Usage:
       from error_ensemble import ErrorEnsemble
       ensemble = ErrorEnsemble(model=trained_v2_model, neural_weight=0.6)
       prediction = ensemble.predict(features)

2. PER-TECHNOLOGY FINE-TUNING (→ 30 min GPU)
   ├─ Purpose: Adapt model to specific sequencing technologies
   ├─ Expected gain: +2-3% accuracy per technology
   ├─ Training needed: Yes (1 hour total GPU)
   ├─ Deployment: ~1 week production testing
   ├─ Code: scripts/train_models/train_tech_specific_heads.py
   ├─ Module: MultiHeadErrorPredictor class
   ├─ Output: 5 technology-specific heads
   └─ Usage:
       from tech_specific_heads import MultiHeadErrorPredictor
       model = MultiHeadErrorPredictor(encoder, tech_heads)
       pred = model.predict(..., technology='ont_r10')

3. KNOWLEDGE DISTILLATION (→ 1-2 hours GPU)
   ├─ Purpose: Compress model for faster inference
   ├─ Expected gain: 6-10× speedup, 95-99% accuracy retention
   ├─ Training needed: Yes
   ├─ Deployment: Immediate (smaller model)
   ├─ Code: scripts/train_models/train_knowledge_distillation.py
   ├─ Module: StudentNetworkMLP or StudentNetworkCNN classes
   ├─ Options: MLP (5K params) or CNN (500 params)
   └─ Usage:
       from knowledge_distillation import StudentNetworkMLP
       student = StudentNetworkMLP(45, 64, 1)  # 45D input, 64 hidden, 1 output
       pred = torch.sigmoid(student(features))

═══════════════════════════════════════════════════════════════════════════════
PART 2: QUICK START (5 minutes)
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Validate your setup
  cd ~/Documents/GitHub_Repositories/smartassembler
  venv_arm64/bin/python3 scripts/train_models/validate_training_setup.py

  This checks:
    ✓ Python environment and packages
    ✓ Baseline model exists (models/error_predictor_v2.pt)
    ✓ Training data exists (training_data/read_correction_v2/base_error/)
    ✓ Output directories can be created

STEP 2: Read the setup guide
  python3 scripts/train_models/TRAINING_SETUP.py

  Or view details in:
    - README: TRAINING_SETUP.py (in this directory)
    - Quick commands: TRAINING_SETUP.py (QUICK_START section)

STEP 3: Pick your training path

  A) FASTEST PATH (per-tech heads only):
     venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py

  B) BEST SPEEDUP (knowledge distillation only):
     venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
         --student-type mlp

  C) MAXIMUM ACCURACY (both):
     Terminal 1: python3 scripts/train_models/train_tech_specific_heads.py
     Terminal 2: python3 scripts/train_models/train_knowledge_distillation.py

═══════════════════════════════════════════════════════════════════════════════
PART 3: DETAILED TRAINING INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

TRAINING OPTION A: Per-Technology Fine-Tuning (RECOMMENDED FIRST)
──────────────────────────────────────────────────────────────────

What it does:
  - Takes trained v2 Transformer encoder
  - Creates 5 technology-specific classification heads (one per technology)
  - Fine-tunes each head on technology-specific data
  - Expected +2-3% accuracy improvement

Expected runtime: 5-10 minutes (GPU)
Expected output: 5 checkpoint files (head_ont_r9.pt, head_ont_r10.pt, ...)

Command:
  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
      --baseline models/error_predictor_v2.pt \\
      --data training_data/read_correction_v2/base_error \\
      --output models/tech_specific_heads \\
      --epochs 20 \\
      --batch-size 32 \\
      --learning-rate 1e-4 \\
      --device cuda

Parameter details:
  --baseline          Path to trained ErrorSmith v2 model
  --data              Path to directory with base_error_*_batch_*.pkl files
  --output            Where to save technology-specific heads
  --epochs            Training epochs per technology (20-30 recommended)
  --batch-size        Batch size (32 for GPU, 8 for limited memory)
  --learning-rate     Head learning rate (1e-4 typically good)
  --device            'cuda' for GPU, 'cpu' for CPU

What happens:
  [Epoch 1/20] ont_r9      - Acc: 0.7234, Train: 0.5123, Val: 0.4456
  [Epoch 2/20] ont_r9      - Acc: 0.7412, Train: 0.4234, Val: 0.3876
  ...
  [Epoch 20/20] ont_r10    - Acc: 0.7824, Train: 0.3123, Val: 0.2456

Output files:
  models/tech_specific_heads/
  ├── head_ont_r9.pt       (2.1 MB)
  ├── head_ont_r10.pt      (2.1 MB)
  ├── head_hifi.pt         (2.1 MB)
  ├── head_illumina.pt     (2.1 MB)
  ├── head_adna.pt         (2.1 MB)
  └── training_summary.json


TRAINING OPTION B: Knowledge Distillation (BEST FOR SPEED)
───────────────────────────────────────────────────────────

What it does:
  - Takes trained v2 Transformer (teacher)
  - Trains compact student: MLP (5K params) or CNN (500 params)
  - Uses soft targets from teacher + hard targets from data
  - Expected 6-10× faster inference, 95-99% accuracy retention

Expected runtime: 30-60 minutes (GPU)
Expected output: student_mlp.pt or student_cnn.pt

Command (MLP - recommended):
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

Command (CNN - ultra-compact):
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

Parameter details:
  --teacher           Path to trained teacher (ErrorSmith v2)
  --student-type      'mlp' (5K params, 95% accuracy) or 'cnn' (500 params, 90%)
  --epochs            Training epochs (50-100 recommended)
  --temperature       Probability softening (3.0-5.0, higher = softer targets)
  --alpha             Hard target weight (0.2-0.5, rest is soft targets)

What happens:
  [Epoch 1/50] Loss: 0.2143, Accuracy: 0.7234
  [Epoch 2/50] Loss: 0.1876, Accuracy: 0.7412
  ...
  [Epoch 50/50] Loss: 0.0543, Accuracy: 0.7534
  
  Teacher inference: 2.28 ms/batch
  Student inference: 0.30 ms/batch
  Speedup: 7.6x

Output files:
  models/student_models/
  ├── student_mlp.pt (18 KB, 5K params)
  ├── student_cnn.pt (5 KB, 500 params)
  └── training_summary.json


═══════════════════════════════════════════════════════════════════════════════
PART 4: DEPLOYMENT
═══════════════════════════════════════════════════════════════════════════════

DEPLOYING PER-TECHNOLOGY HEADS
───────────────────────────────

After training, you have 5 technology-specific heads. Use like this:

  import torch
  from scripts.train_models.tech_specific_heads import MultiHeadErrorPredictor
  
  # Create predictor with all heads
  predictor = MultiHeadErrorPredictor(
      baseline_path='models/error_predictor_v2.pt',
      heads_dir='models/tech_specific_heads',
  )
  
  # Predict for specific technology
  for sample in data:
      prediction = predictor.predict(
          seq_features=sample['seq_features'],     # (21,)
          tech_features=sample['tech_features'],   # (20,)
          tech_mask=sample['tech_mask'],           # (20,) boolean
          technology=sample['technology'],         # 'ont_r10', 'hifi', etc.
      )
      print(f"Error probability: {prediction:.4f}")
  
  # Batch prediction (mixed technologies)
  predictions = predictor.predict_batch(
      seq_features=batch['seq_features'],         # (batch, 21)
      tech_features=batch['tech_features'],       # (batch, 20)
      tech_mask=batch['tech_mask'],               # (batch, 20)
      technologies=batch['technologies'],         # List of tech names
  )


DEPLOYING KNOWLEDGE DISTILLATION
─────────────────────────────────

After training, use compact student for fast inference:

  import torch
  from scripts.train_models.knowledge_distillation import StudentNetworkMLP
  
  # Load student model
  student = StudentNetworkMLP(input_dim=45, hidden_dim=64, output_dim=1)
  checkpoint = torch.load('models/student_models/student_mlp.pt')
  student.load_state_dict(checkpoint['state_dict'])
  student.eval()
  
  # Single prediction
  with torch.no_grad():
      logits = student(features)  # features: (45,) - combined seq + tech
      probability = torch.sigmoid(logits).item()
  
  # Batch prediction
  with torch.no_grad():
      batch_logits = student(batch_features)  # (batch, 1)
      batch_probs = torch.sigmoid(batch_logits).squeeze()
  
  # Speed comparison
  import time
  start = time.time()
  for _ in range(1000):
      _ = torch.sigmoid(student(features))
  elapsed = time.time() - start
  print(f"1000 predictions in {elapsed*1000:.1f} ms = {elapsed/1000*1e6:.1f} µs each")


═══════════════════════════════════════════════════════════════════════════════
PART 5: MONITORING AND TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

WHAT TO EXPECT DURING TRAINING
──────────────────────────────

Good signs:
  ✓ Validation loss decreasing each epoch
  ✓ Accuracy improving
  ✓ Smooth curves (not oscillating)
  ✓ Model size ~2-18 MB (tech heads/distillation)

Example log output:
  [2024-01-15 14:32:10] [INFO] Epoch [1/20] - Train Loss: 0.5234, Val Loss: 0.4123, Accuracy: 0.7234
  [2024-01-15 14:32:15] [INFO] Epoch [2/20] - Train Loss: 0.4512, Val Loss: 0.3876, Accuracy: 0.7412
  [2024-01-15 14:32:20] [INFO] Epoch [3/20] - Train Loss: 0.4012, Val Loss: 0.3612, Accuracy: 0.7523

Warning signs:
  ⚠ "CUDA out of memory" → Reduce batch size (--batch-size 16)
  ⚠ Loss not decreasing → Lower learning rate (--learning-rate 1e-5)
  ⚠ Loss oscillating → Too high learning rate
  ⚠ "No batch files found" → Wrong data directory
  ⚠ Early stopping → Too early, try --epochs 30 or --patience 10


TROUBLESHOOTING COMMON ISSUES
──────────────────────────────

Q: CUDA out of memory
A: Try:
   1. Reduce batch size: --batch-size 16 (or 8)
   2. Use CPU: --device cpu (will be slower)
   3. Close other programs using GPU

Q: "ModuleNotFoundError: No module named 'train_models'"
A: Make sure you're in the right directory:
   cd ~/Documents/GitHub_Repositories/smartassembler
   Then verify the path is correct in your command

Q: "Data directory not found"
A: Check the path exists:
   ls training_data/read_correction_v2/base_error/
   If it's in a different location, use --data /correct/path

Q: "Model not found"
A: Check baseline exists:
   ls models/error_predictor_v2.pt
   If it's not trained, run:
   python3 scripts/train_models/train_base_error_predictor_v2.py

Q: Accuracy not improving
A: Try:
   - Lower learning rate: --learning-rate 1e-5
   - Train longer: --epochs 30
   - Check if data is reasonable (not all one class)
   - Look at validation accuracy (not training)

Q: Training very slow
A: Check:
   - Are you using GPU? (--device cuda)
   - Is GPU actually being used? (nvidia-smi in terminal)
   - Try smaller model options if available

Q: Process killed (out of memory)
A: System RAM probably exceeded:
   - Reduce batch size: --batch-size 8
   - Run on GPU instead of CPU
   - Test with smaller dataset first


═══════════════════════════════════════════════════════════════════════════════
PART 6: EXPECTED RESULTS & TIMELINE
═══════════════════════════════════════════════════════════════════════════════

PER-TECHNOLOGY HEADS
────────────────────

Timeline:
  Setup & validation:     ~5 min (one-time)
  Fine-tuning heads:      5-15 min (GPU)
  Total:                  ~20 min

Expected improvements:
  - ont_r9:     72-75% → 74-77%   (+2-3%)
  - ont_r10:    73-76% → 75-78%   (+2-3%)
  - hifi:       76-79% → 78-81%   (+2-3%)
  - illumina:   70-73% → 72-75%   (+2-3%)
  - adna:       65-68% → 67-70%   (+2-3%)

Output files:
  - 5 head checkpoints (2 MB each)
  - 1 training summary (JSON)


KNOWLEDGE DISTILLATION - MLP
────────────────────────────

Timeline:
  Setup & validation:     ~5 min (one-time)
  Teacher loading:        ~1 min
  Training student:       30-60 min (GPU)
  Total:                  ~45 min

Expected performance:
  - Accuracy:             72-78% → 71-76%     (95-99% retention)
  - Inference speed:      ~2.3 ms → ~0.3 ms   (7-8× faster)
  - Model size:           156 MB → 18 KB       (8600× smaller!)
  - Params:               156K → 5K             (31× fewer)

Output file:
  - student_mlp.pt (18 KB)


KNOWLEDGE DISTILLATION - CNN
────────────────────────────

Timeline:
  Setup & validation:     ~5 min (one-time)
  Training student:       30-60 min (GPU)
  Total:                  ~40 min

Expected performance:
  - Accuracy:             72-78% → 70-75%     (90-95% retention)
  - Inference speed:      ~2.3 ms → ~0.2 ms   (10-12× faster)
  - Model size:           156 MB → 5 KB        (30000× smaller!)
  - Params:               156K → 500            (312× fewer)

Output file:
  - student_cnn.pt (5 KB)


═══════════════════════════════════════════════════════════════════════════════
PART 7: FILE REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Core Training Scripts:
  scripts/train_models/train_tech_specific_heads.py     (790 lines)
    - Main script for per-technology fine-tuning
    - Usage: python3 train_tech_specific_heads.py --device cuda
    - Output: 5 technology-specific head checkpoints

  scripts/train_models/train_knowledge_distillation.py   (820 lines)
    - Main script for knowledge distillation
    - Usage: python3 train_knowledge_distillation.py --student-type mlp
    - Output: student_mlp.pt or student_cnn.pt

Setup and Validation:
  scripts/train_models/TRAINING_SETUP.py                 (400 lines)
    - Setup guide with step-by-step instructions
    - Contains default configurations
    - Usage: python3 TRAINING_SETUP.py --save-config

  scripts/train_models/validate_training_setup.py        (350 lines)
    - Validates environment, models, and data
    - Checks all prerequisites before training
    - Usage: python3 validate_training_setup.py

Support Modules (Already implemented):
  scripts/train_models/tech_specific_heads.py
    - MultiHeadErrorPredictor: Inference wrapper
    - TechSpecificCalibration: Per-tech parameters
    - TechnologySpecificHead: Individual head architecture

  scripts/train_models/knowledge_distillation.py
    - StudentNetworkMLP: 5K parameter student
    - StudentNetworkCNN: 500 parameter student
    - KnowledgeDistillationLoss: Training loss function
    - DistillationTrainer: Full training pipeline

  scripts/train_models/error_ensemble.py
    - ErrorEnsemble: Combines NN + heuristics
    - 5 heuristic classes (ready to deploy)

═══════════════════════════════════════════════════════════════════════════════
PART 8: NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

IMMEDIATE (This week):
  1. Run: python3 scripts/train_models/validate_training_setup.py
  2. Choose training path (A, B, or C)
  3. Start training (5-60 min depending on choice)

SHORT-TERM (Next week):
  1. Validate results on held-out test set
  2. Compare accuracy of improvements
  3. Deploy best option to production

MID-TERM (Next 2-3 weeks):
  1. A/B test per-tech heads vs baseline
  2. A/B test student models vs baseline
  3. Monitor inference latency improvements
  4. Collect user feedback

LONG-TERM:
  1. Consider advanced improvements (multi-task learning, uncertainty)
  2. Auto-scale deployment based on technology detected
  3. Retrain periodically on new data
  4. Monitor performance drift

═══════════════════════════════════════════════════════════════════════════════
"""

print(REFERENCE)

if __name__ == '__main__':
    print("\n✅ Complete reference guide printed above.")
    print("\nFor setup instructions, run:")
    print("  python3 TRAINING_SETUP.py")
    print("\nFor quick validation, run:")
    print("  python3 validate_training_setup.py")
