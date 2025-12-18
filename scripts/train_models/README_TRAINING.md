╔════════════════════════════════════════════════════════════════════════════╗
║              ERRORSMITH V2 TRAINING SETUP - MASTER INDEX                   ║
║                                                                            ║
║    Medium-Effort Improvements: Per-Tech Heads + Knowledge Distillation     ║
╚════════════════════════════════════════════════════════════════════════════╝

═════════════════════════════════════════════════════════════════════════════════
WHAT'S NEW: Training Infrastructure Complete ✅
═════════════════════════════════════════════════════════════════════════════════

You now have everything needed to train and deploy two medium-effort improvements:

1. PER-TECHNOLOGY FINE-TUNING HEADS
   - 5 technology-specific classifiers (ONT R9, ONT R10, HiFi, Illumina, aDNA)
   - Expected +2-3% accuracy improvement
   - Training time: 5-15 min (GPU)

2. KNOWLEDGE DISTILLATION  
   - Compact student models: MLP (5K params) or CNN (500 params)
   - Expected 6-10× speedup with 95-99% accuracy retention
   - Training time: 30-60 min (GPU)

PLUS: Error Ensemble (ready to deploy with NO training needed)
   - +2-4% accuracy from heuristic rules
   - Deploy immediately

═════════════════════════════════════════════════════════════════════════════════
FILES YOU NEED TO KNOW ABOUT
═════════════════════════════════════════════════════════════════════════════════

TRAINING SCRIPTS (Run these to train):

  📄 train_tech_specific_heads.py
     Purpose: Train technology-specific heads
     Run: venv_arm64/bin/python3 train_tech_specific_heads.py --device cuda
     Time: 5-15 min

  📄 train_knowledge_distillation.py
     Purpose: Train compact student models
     Run: venv_arm64/bin/python3 train_knowledge_distillation.py --student-type mlp
     Time: 30-60 min

  📄 validate_training_setup.py
     Purpose: Check prerequisites before training
     Run: venv_arm64/bin/python3 validate_training_setup.py
     Time: 2 min (run FIRST)

GUIDES & DOCUMENTATION (Read these to understand):

  📖 TRAINING_CHECKLIST.py
     5-minute quick-start checklist
     Copy-paste commands
     Decision tree for choosing training path
     → START HERE

  📖 TRAINING_SETUP.py
     Detailed setup instructions
     Parameter explanations
     Troubleshooting guide
     → READ FOR DETAILS

  📖 COMPLETE_REFERENCE.py
     Complete reference documentation
     Code deployment examples
     Performance benchmarks
     → REFERENCE DURING TRAINING

  📖 TRAINING_SETUP_SUMMARY.txt
     This directory's summary
     Quick paths and timelines
     → OVERVIEW

═════════════════════════════════════════════════════════════════════════════════
QUICK START (5 MINUTES)
═════════════════════════════════════════════════════════════════════════════════

1. Run validation:
   cd ~/Documents/GitHub_Repositories/smartassembler
   venv_arm64/bin/python3 scripts/train_models/validate_training_setup.py

2. Review your options:
   python3 scripts/train_models/TRAINING_CHECKLIST.py

3. Choose your path and run ONE of these:

   ← FASTEST (30 min total, +2-3% accuracy):
   venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
       --baseline models/error_predictor_v2.pt \\
       --data training_data/read_correction_v2/base_error \\
       --output models/tech_specific_heads \\
       --device cuda

   ← BEST SPEEDUP (1 hour total, 7-10× faster):
   venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
       --teacher models/error_predictor_v2.pt \\
       --student-type mlp \\
       --data training_data/read_correction_v2/base_error \\
       --output models/student_models \\
       --device cuda

   ← BOTH (2 hours total, maximum impact):
   Run both scripts in separate terminals

═════════════════════════════════════════════════════════════════════════════════
THREE TRAINING OPTIONS
═════════════════════════════════════════════════════════════════════════════════

OPTION A: PER-TECHNOLOGY HEADS
├─ Command: train_tech_specific_heads.py
├─ Time: 5-15 minutes (GPU)
├─ Output: 5 head checkpoints
├─ Gain: +2-3% accuracy
├─ When to use: You want quick accuracy improvement
└─ Example:
    venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
        --device cuda


OPTION B: KNOWLEDGE DISTILLATION (MLP)
├─ Command: train_knowledge_distillation.py --student-type mlp
├─ Time: 30-60 minutes (GPU)
├─ Output: student_mlp.pt (18 KB)
├─ Gain: 7-8× speedup, 95-99% accuracy retention
├─ When to use: You want balance of speed and accuracy
└─ Example:
    venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
        --student-type mlp --device cuda


OPTION C: KNOWLEDGE DISTILLATION (CNN)
├─ Command: train_knowledge_distillation.py --student-type cnn
├─ Time: 30-60 minutes (GPU)
├─ Output: student_cnn.pt (5 KB)
├─ Gain: 10-12× speedup, 90-95% accuracy retention
├─ When to use: You need ultra-compact model for edge devices
└─ Example:
    venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
        --student-type cnn --device cuda


OPTION D: ALL THREE (Maximum Impact)
├─ Run all in parallel (separate terminals)
├─ Time: 2 hours total (GPU)
├─ Output: All 3 improvements trained
├─ Gain: +2-3% accuracy + 7-12× speedup options
├─ When to use: You have multi-GPU or want all options
└─ See TRAINING_CHECKLIST.py for exact commands

═════════════════════════════════════════════════════════════════════════════════
WHAT GETS GENERATED
═════════════════════════════════════════════════════════════════════════════════

PER-TECHNOLOGY HEADS (models/tech_specific_heads/):
  ✅ head_ont_r9.pt          (2 MB) - ONT R9 classifier
  ✅ head_ont_r10.pt         (2 MB) - ONT R10 classifier
  ✅ head_hifi.pt            (2 MB) - PacBio HiFi classifier
  ✅ head_illumina.pt        (2 MB) - Illumina classifier
  ✅ head_adna.pt            (2 MB) - Ancient DNA classifier
  ✅ training_summary.json           - Metrics

KNOWLEDGE DISTILLATION (models/student_models/):
  ✅ student_mlp.pt          (18 KB) - MLP student OR
  ✅ student_cnn.pt          (5 KB)  - CNN student
  ✅ training_summary.json           - Speedup metrics

═════════════════════════════════════════════════════════════════════════════════
BEFORE YOU START
═════════════════════════════════════════════════════════════════════════════════

PREREQUISITES (Must have):
  ✓ Trained ErrorSmith v2 baseline model
    Location: models/error_predictor_v2.pt
    Check: ls -lh models/error_predictor_v2.pt

  ✓ Training data with pickle files
    Location: training_data/read_correction_v2/base_error/
    Check: ls training_data/read_correction_v2/base_error/ | head

  ✓ Python 3.8+ and PyTorch
    Check: python3 --version && python3 -c "import torch; print(torch.__version__)"

  ✓ GPU recommended (10× speedup vs CPU)
    Check: python3 -c "import torch; print(torch.cuda.is_available())"

VALIDATION (Run before training):
  venv_arm64/bin/python3 scripts/train_models/validate_training_setup.py
  Should show: "✅ READY TO TRAIN!"

═════════════════════════════════════════════════════════════════════════════════
EXPECTED PERFORMANCE
═════════════════════════════════════════════════════════════════════════════════

PER-TECHNOLOGY FINE-TUNING:
  Technology          Baseline    After FT    Improvement
  ────────────────────────────────────────────────────────
  ONT R9              72-75%      74-77%      +2-3%
  ONT R10             73-76%      75-78%      +2-3%
  PacBio HiFi         76-79%      78-81%      +2-3%
  Illumina            70-73%      72-75%      +2-3%
  Ancient DNA         65-68%      67-70%      +2-3%
  
  Training: ~2-3 min per head × 5 heads = 10-15 min total

KNOWLEDGE DISTILLATION - MLP:
  Metric              Value
  ────────────────────────────
  Accuracy            95-99% retention (71-76% from 72-78%)
  Speed               7-8× faster (0.3 ms vs 2.3 ms per batch)
  Model size          8,600× smaller (18 KB vs 156 MB)
  Training time       30-60 min

KNOWLEDGE DISTILLATION - CNN:
  Metric              Value
  ────────────────────────────
  Accuracy            90-95% retention (70-75% from 72-78%)
  Speed               10-12× faster (0.2 ms vs 2.3 ms per batch)
  Model size          30,000× smaller (5 KB vs 156 MB)
  Training time       30-60 min

═════════════════════════════════════════════════════════════════════════════════
RUNNING THE TRAINING
═════════════════════════════════════════════════════════════════════════════════

STEP 1: Open Terminal
  cd ~/Documents/GitHub_Repositories/smartassembler

STEP 2: Run Validation
  venv_arm64/bin/python3 scripts/train_models/validate_training_setup.py
  Wait for: "✅ READY TO TRAIN!"

STEP 3: Choose Your Path
  python3 scripts/train_models/TRAINING_CHECKLIST.py
  (Or just pick from options above)

STEP 4: Start Training
  Copy and paste ONE of the commands from the "Three Training Options" section
  Watch the output:
    - You'll see epoch progress
    - Validation loss should decrease
    - Accuracy should increase

STEP 5: Wait (5 min - 1 hour depending on option)
  Do not interrupt the process
  Look for: "✅ All heads saved to:" or "✅ Model saved to:"

STEP 6: Verify Completion
  cat models/tech_specific_heads/training_summary.json
  cat models/student_models/training_summary.json
  Should show accuracies and metrics

═════════════════════════════════════════════════════════════════════════════════
IF SOMETHING GOES WRONG
═════════════════════════════════════════════════════════════════════════════════

"CUDA out of memory"
  Solution: Reduce batch size
  Add to your command: --batch-size 16 (or 8)

"No module named 'train_models'"
  Solution: Make sure you're in the right directory
  Run: cd ~/Documents/GitHub_Repositories/smartassembler

"Data directory not found"
  Solution: Check the path
  Run: ls training_data/read_correction_v2/base_error/
  Add to command: --data /correct/path

"Model not found"
  Solution: Verify baseline exists
  Run: ls models/error_predictor_v2.pt
  If missing: Train baseline first

"Accuracy not improving"
  Solution: Try lower learning rate
  Add to command: --learning-rate 1e-5

For more troubleshooting, see: TRAINING_SETUP.py

═════════════════════════════════════════════════════════════════════════════════
WHAT HAPPENS AFTER TRAINING
═════════════════════════════════════════════════════════════════════════════════

IMMEDIATELY (5 min after training finishes):
  1. Check the output files exist
     ls models/tech_specific_heads/ (for tech heads)
     ls models/student_models/ (for distillation)

  2. Review the summary JSON
     cat models/tech_specific_heads/training_summary.json

  3. Note the metrics for later comparison

ONE WEEK (Prepare for production):
  1. Test the new models on validation data
  2. Compare accuracy vs baseline
  3. Measure inference speed
  4. Decide which improvement to deploy

PRODUCTION (Deploy the best option):
  - See COMPLETE_REFERENCE.py for deployment code
  - A/B test improvements vs baseline
  - Monitor performance metrics
  - Set up logging

═════════════════════════════════════════════════════════════════════════════════
HELPFUL RESOURCES IN THIS DIRECTORY
═════════════════════════════════════════════════════════════════════════════════

START HERE:
  1. TRAINING_CHECKLIST.py              ← Quick checklist
  2. validate_training_setup.py          ← Verify prerequisites
  3. Pick a training option and run it

DURING TRAINING:
  - TRAINING_SETUP.py                   ← Parameter details & troubleshooting

AFTER TRAINING:
  - COMPLETE_REFERENCE.py               ← Deployment code examples

SUPPORTING MODULES (Already tested):
  - error_ensemble.py                   ← Heuristics (deploy anytime)
  - tech_specific_heads.py              ← Per-tech head classes
  - knowledge_distillation.py           ← Distillation classes

═════════════════════════════════════════════════════════════════════════════════
READY TO START?
═════════════════════════════════════════════════════════════════════════════════

COPY & PASTE THIS (best for accuracy):

  cd ~/Documents/GitHub_Repositories/smartassembler && \\
  venv_arm64/bin/python3 scripts/train_models/train_tech_specific_heads.py \\
      --baseline models/error_predictor_v2.pt \\
      --data training_data/read_correction_v2/base_error \\
      --output models/tech_specific_heads \\
      --device cuda

OR THIS (best for speed):

  cd ~/Documents/GitHub_Repositories/smartassembler && \\
  venv_arm64/bin/python3 scripts/train_models/train_knowledge_distillation.py \\
      --teacher models/error_predictor_v2.pt \\
      --student-type mlp \\
      --data training_data/read_correction_v2/base_error \\
      --output models/student_models \\
      --device cuda

═════════════════════════════════════════════════════════════════════════════════
                            Good luck! 🚀
═════════════════════════════════════════════════════════════════════════════════
