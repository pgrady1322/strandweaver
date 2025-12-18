# ErrorSmith v2: Technology-Aware Transformer Architecture

## Overview

ErrorSmith v2 is a significant upgrade to the base error prediction model, addressing the fundamental insight that **sequencing error patterns are technology-dependent**.

### Key Improvement: Technology-Aware Feature Engineering

**The Problem:** ErrorSmith v1 used generic features (sequence context, quality, coverage) that don't account for technology-specific error signatures:
- **ONT**: Homopolymer deletions (3D bias, 2× more deletions than substitutions)
- **PacBio HiFi**: Rare but position-biased errors (ends of reads)
- **Illumina**: Cycle-dependent quality scores, quality score inflation
- **aDNA**: C→T deamination at read terminals
- **PacBio CLR**: High error rates, polymerase-dependent patterns

### The Solution: Adaptive Feature Masking

Features are now **dynamically selected based on technology**:

```
Technology: ONT R9
├── Core Sequence Features (25D) ✓ Always present
├── Homopolymer Features (4D) ✓ Only for ONT/CLR
├── Quality Context (5D) ✗ Skip (quality unreliable)
├── Coverage Features (3D) ✓ Include
├── Position Bias (3D) ✗ Skip (not position-dependent)
├── Deamination (3D) ✗ Skip (not ancient DNA)
└── Tech Embedding (4D) ✓ Include
Total: 25 + 4 + 3 + 4 = 36D effective

Technology: Illumina
├── Core Sequence Features (25D) ✓ Always present
├── Homopolymer Features (4D) ✗ Skip (not homopolymer-dependent)
├── Quality Context (5D) ✓ Include (highly reliable)
├── Coverage Features (3D) ✓ Include
├── Position Bias (3D) ✗ Skip (not position-dependent)
├── Deamination (3D) ✗ Skip (not ancient DNA)
└── Tech Embedding (4D) ✓ Include
Total: 25 + 5 + 3 + 4 = 37D effective

Technology: Ancient DNA
├── Core Sequence Features (25D) ✓ Always present
├── Homopolymer Features (4D) ✗ Skip
├── Quality Context (5D) ✗ Skip (unreliable)
├── Coverage Features (3D) ✓ Include
├── Position Bias (3D) ✓ Include (errors at ends)
├── Deamination (3D) ✓ Include (C→T/G→A patterns)
└── Tech Embedding (4D) ✓ Include
Total: 25 + 3 + 3 + 3 + 4 = 38D effective
```

## Architecture Components

### 1. TechAwareFeatureBuilder (`tech_aware_feature_builder.py`)

Generates feature vectors with technology-aware masking.

**Core Features (25D, always present):**
- One-hot encoded sequence context (21D): 10bp left + center + 10bp right
- GC content in local window (1D)
- Sequence complexity/entropy (1D)
- Dinucleotide scores (2D)

**Technology-Specific Features (selective, 0-20D active per technology):**

| Feature Set | ONT R9 | ONT R10 | HiFi | CLR | Illumina | aDNA |
|-------------|--------|---------|------|-----|----------|------|
| Homopolymer (4D) | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Quality Context (5D) | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Coverage (3D) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Position Bias (3D) | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Deamination (3D) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Tech Embedding (4D) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Total Active** | **36D** | **39D** | **37D** | **41D** | **37D** | **38D** |

### 2. TechAwareTransformer (`train_base_error_predictor_v2.py`)

Replaces CNN with Transformer architecture.

**Architecture:**
```
Input
├── Sequence Branch
│   ├── One-hot encoding (21D)
│   ├── Linear embedding (→64D)
│   ├── Positional encoding
│   └── Transformer encoder (3 layers, 4 heads)
│
├── Tech Features Branch
│   ├── Tech-aware features (variable D, masked)
│   ├── Linear embedding (→32D)
│   └── Zero-pad to 64D
│
├── Concatenate branches (2 × 64D)
│
├── Transformer encoder (3 layers, 4 heads)
│   └── Multi-head attention over [seq, tech]
│
├── Adaptive pooling
│   └── Learn which positions matter (attention weights)
│
└── Classification head
    ├── Concatenate [pooled_seq, tech_features]
    ├── Dense layer (→64D)
    ├── ReLU + Dropout
    └── Sigmoid output
```

**Key Advantages:**
- **Attention**: Learns long-range dependencies in sequence context
- **Dual branches**: Separates sequence from quality/coverage signals
- **Dynamic pooling**: Learns to focus on relevant positions
- **Tech-aware**: Conditional feature masking per technology

### 3. FocalLoss for Class Imbalance

Instead of binary cross-entropy:

```python
Focal Loss = -α(1 - p_t)^γ * BCE

where:
  p_t = predicted probability of true class
  γ = 2.0 (focusing parameter)
  α = 0.25 (class balance weight)
```

**Effect**: Down-weights easy examples (correct bases predicted as correct), focuses on hard negatives/positives (misclassified bases).

## Expected Performance

### v1 (CNN) Baseline
- Overall accuracy: **65.05%**
- Per-technology: Uniform performance
- Loss function: Standard BCE
- Training time: ~2.5 hours

### v2 (Transformer) Expected
- Overall accuracy: **72-78%** (+7-13% improvement)
  - Sequence attention: +3-5%
  - Tech-aware features: +2-4%
  - Focal loss: +1-2%
  - Transformer vs CNN: +2-3%
  
- Per-technology differentiation:
  - ONT R9: 70-75% (homopolymer-sensitive)
  - ONT R10: 75-80% (improved quality, less homopolymers)
  - HiFi: 80-85% (position-based features)
  - CLR: 65-70% (high baseline error rate)
  - Illumina: 85-90% (quality + cycle-dependent)
  - aDNA: 70-75% (deamination patterns)

- Training time: ~4-5 hours (larger model, more examples)

## Usage

### Quick Start: Train v2 Model

```bash
# Using default settings
./train_errorsmith_v2.sh

# Custom epochs and batch size
./train_errorsmith_v2.sh venv_arm64 30 512

# Manual training
venv_arm64/bin/python3 scripts/train_models/train_base_error_predictor_v2.py \
    --data training_data/read_correction_v2/base_error \
    --output models/base_error_predictor_v2.pt \
    --epochs 25 \
    --batch-size 32 \
    --learning-rate 0.0005
```

### Compare v1 vs v2

```bash
# Evaluate both models on test set
python3 scripts/compare_errorsmith_models.py \
    --data training_data/read_correction_v2/base_error \
    --model-v1 models/base_error_predictor.pt \
    --model-v2 models/base_error_predictor_v2.pt \
    --limit 10000
```

### Use v2 Model in Production

```python
from scripts.train_models.train_base_error_predictor_v2 import TechAwareTransformer
from scripts.train_models.tech_aware_feature_builder import TechAwareFeatureBuilder
import torch

# Load model
device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
checkpoint = torch.load('models/base_error_predictor_v2.pt', map_location=device)
model = TechAwareTransformer().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Build features
feature_builder = TechAwareFeatureBuilder()
features = feature_builder.build_features(
    base='A',
    left_context='ACGTACGTAC',
    right_context='TGCATGCATG',
    quality_score=20,
    kmer_coverage=15,
    position=100,
    read_length=1000,
    technology='ont_r9'
)

# Predict
with torch.no_grad():
    pred = model(
        torch.FloatTensor(features.sequence_features).unsqueeze(0).to(device),
        torch.FloatTensor(features.tech_features).unsqueeze(0).to(device),
        torch.BoolTensor(features.tech_feature_mask).unsqueeze(0).to(device),
    )

print(f"Error probability: {pred.item():.4f}")
```

## Implementation Strategy

### Phase 1: Feature Engineering (✓ Complete)
- [x] Tech profile definitions
- [x] Feature masking logic
- [x] Feature builder class with all 55-60D features
- [x] Test on sample data

### Phase 2: Transformer Model (✓ Complete)
- [x] Dual-branch architecture (sequence + tech)
- [x] Transformer encoder with attention
- [x] Focal loss implementation
- [x] Training loop with early stopping

### Phase 3: Training & Evaluation (Ready to execute)
- [ ] Train on full 17.1M example dataset (~4-5 hours)
- [ ] Compare with v1 baseline (expected +7-13% improvement)
- [ ] Per-technology analysis
- [ ] Integration into assembly pipeline

### Phase 4: Optimization (Future)
- [ ] Hyperparameter tuning (embedding_dim, num_heads, num_layers)
- [ ] Ensemble with heuristic rules (homopolymer voting, coverage consensus)
- [ ] Knowledge distillation for faster inference
- [ ] ONNX export for cross-platform deployment

## Files Created

```
scripts/train_models/
├── tech_aware_feature_builder.py      # 380 lines: Feature engineering
├── train_base_error_predictor_v2.py   # 420 lines: Transformer training
└── compare_errorsmith_models.py       # 300 lines: v1 vs v2 comparison

root/
└── train_errorsmith_v2.sh             # 80 lines: Training launcher
```

## Modifications to ml_interfaces.py (None Needed)

The `BaseContext` dataclass already contains all necessary fields:
- `base`: The base in question
- `quality_score`: Quality score (optional)
- `left_context` / `right_context`: Sequence context
- `kmer_coverage`: K-mer coverage (optional)
- `technology`: Technology identifier
- `position`: Position in read

No changes to ml_interfaces.py required!

## Performance Expectations Timeline

**Baseline (v1):** 65.05% accuracy

**After tech-aware features:** 70-72% (+5-7%)

**After Transformer + Focal Loss:** 75-78% (+10-13% total)

**With ensemble heuristics:** 78-82% (+13-17% total, but inference-time only)

## Troubleshooting

### Memory Issues During Training
```bash
# Reduce batch size
--batch-size 128

# Reduce model size
# In TechAwareTransformer: embedding_dim=32, num_layers=2
```

### Poor Performance on Specific Technology
Check tech_aware_feature_builder.py TECH_PROFILES - may need feature adjustments for that technology.

### Convergence Issues
- Increase learning rate: `--learning-rate 0.001`
- Reduce dropout: Change dropout=0.2 to dropout=0.1
- Longer training: `--epochs 40`

## References

- Focal Loss: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Transformer Architecture: Vaswani et al. (2017) "Attention Is All You Need"
- Technology-aware ML: Principles from competitive ML (different models per data distribution)
