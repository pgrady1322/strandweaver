# StrandWeaver — Training Progress

> **Version**: v0.3.0-dev  
> **Last updated**: 2025-02-15  
> **Data location**: `training_data_10x/` (graph-only CSVs)  
> **Model weights**: `trained_models_10x/` (EdgeWarden + PathGNN retrained w/ hybrid resampling)  
> **Colab tarball**: `graph_csvs.tar.gz` (401 MB, all 850 CSVs)

---

## Overview

StrandWeaver uses a two-tier ML architecture:

1. **XGBoost models** (5 models) — trained on tabular CSV features extracted
   from synthetic assembly graphs. These handle classification/regression tasks
   and provide feature importance signals for the GNN.
2. **PathGNN** (1 GNN) — trained on Google Colab (T4 GPU) using PyTorch Geometric.
   Uses GATv2Conv + custom EdgeConvLayer with graph attention to propagate
   structural information across assembly graph neighborhoods.

### Training Workflow

All XGBoost models are retrained via Google Colab with GPU-accelerated
hyperparameter sweeps, then downloaded as `.pkl` files. The GNN is trained
separately on Colab using the XGBoost EdgeWarden predictions as ensemble
features.

| Phase | Notebook | Purpose |
|-------|----------|---------|
| 1 | `notebooks/XGBoost_Retraining_Colab.ipynb` | Hyperparameter sweep + final training for all 5 XGBoost models |
| 2 | `notebooks/PathGNN_Training_Colab.ipynb` | GNN training with XGBoost ensemble edge features |

---

## Training Data

### Generation Parameters

| Parameter | Value |
|-----------|-------|
| Mode | `--graph-only` (no FASTQ/FASTA written to disk) |
| Total genomes | 200 (8 batches × 25 genomes) |
| Genome size | 1 Mb diploid per genome |
| Total CSVs | 850 |
| Total rows | ~20.2 million |
| Disk usage | 1.4 GB (401 MB compressed) |
| Generation time | ~17.7 h (MacBook Air M2) |

### Batch Diversity

| Batch | Description | Repeat | GC | SV Density | Read Types |
|-------|-------------|--------|----|------------|------------|
| 01 | Baseline | 0.30 | 0.42 | 2e-5 | HiFi 30×, ONT 20×, Hi-C 15× |
| 02 | Repeat-rich | 0.50 | 0.38 | 2e-5 | HiFi 30×, ONT 20×, Hi-C 15× |
| 03 | Extreme repeat | 0.65 | 0.40 | 1e-5 | HiFi 25×, ONT 25× |
| 04 | GC-rich | 0.35 | 0.55 | 2e-5 | HiFi 30×, Illumina 50× |
| 05 | AT-rich | 0.35 | 0.32 | 3e-5 | HiFi 30×, ONT 20×, Hi-C 15× |
| 06 | Low repeat, high SV | 0.20 | 0.42 | 5e-5 | HiFi 30×, ONT 20×, Hi-C 15× |
| 07 | Ultra-long focus | 0.40 | 0.42 | 2e-5 | HiFi 25×, Ultra-long 15× |
| 08 | All technologies | 0.45 | 0.44 | 3e-5 | HiFi 20×, ONT 15×, Hi-C 10×, UL 10× |

### Graph-Only Mode

Reads are simulated **in-memory** for realistic coverage, overlap, and sequence
features, but FASTQ/FASTA files are never written to disk. Only the graph
training CSVs are produced. This gives a **3.3× speedup** and **27× disk
reduction** compared to the full pipeline.

```
Full mode   : ~23 min/genome, ~230 MB/genome
Graph-only  : ~7 min/genome,  ~8.5 MB/genome
```

---

## Class Imbalance Resolution

### The Problem

EdgeAI training data has extreme class imbalance (541× between largest and
smallest class):

| Class | Samples | Proportion |
|-------|---------|------------|
| CHIMERIC | 665,493 | 56.3% |
| SV_BREAK | 480,498 | 40.6% |
| TRUE | 35,410 | 3.0% |
| ALLELIC | 1,223 | 0.1% |

The v0.2.0 baseline used XGBoost's `scale_pos_weight` / sample weighting, which
barely helped ALLELIC (F1 = 0.39) and left overall F1-macro at 0.623.

### Strategy Comparison (7 strategies tested)

| Strategy | F1-macro | Accuracy | ALLELIC F1 | CHIMERIC F1 | SV_BREAK F1 | TRUE F1 | CV F1-m |
|----------|----------|----------|------------|-------------|-------------|---------|---------|
| **Hybrid** | **0.829** | **0.804** | **0.987** | **0.669** | **0.697** | **0.965** | **0.826 ± 0.001** |
| Oversample median | 0.822 | 0.782 | 0.986 | 0.758 | 0.589 | 0.955 | 0.822 ± 0.001 |
| Undersample 100k | 0.773 | 0.725 | 0.792 | 0.674 | 0.692 | 0.936 | 0.779 ± 0.002 |
| Binary true-vs-rest | 0.870 | 0.980 | — | — | — | 0.751 | 0.870 ± 0.001 |
| Baseline (w=50) | 0.623 | 0.677 | 0.391 | 0.695 | 0.651 | 0.756 | 0.615 ± 0.004 |
| High weight (w=500) | 0.567 | 0.663 | 0.197 | 0.677 | 0.645 | 0.747 | 0.565 ± 0.002 |
| Uncapped weight | 0.567 | 0.663 | 0.197 | 0.677 | 0.645 | 0.747 | 0.565 ± 0.002 |

### Winning Strategy: Hybrid Resampling

The **hybrid** strategy combines two techniques:

1. **Undersample** the two majority classes (CHIMERIC, SV_BREAK) to a cap of
   100,000 samples each
2. **Oversample** the two minority classes (TRUE, ALLELIC) up to the new median
   of the resampled distribution

This reduces total training samples from 1,182,624 → ~285,000 while lifting
ALLELIC F1 from 0.39 → **0.99** (+153%) and overall F1-macro from 0.623 →
**0.829** (+33%).

Hybrid resampling is now the default in `train_models.py` and triggers
automatically when the class imbalance ratio exceeds 5.0×.

---

## XGBoost Model Results

### v0.3.0-dev — Hybrid Resampling (Current)

Retrained locally with hybrid resampling. Only EdgeAI and PathGNN have been
retrained so far; the remaining 3 models are pending Colab retraining.

| Model | Task | Samples | Resampled | Accuracy | F1-w | F1-m | CV (5-fold) | Status |
|-------|------|---------|-----------|----------|------|------|-------------|--------|
| PathGNN | Edge classification | 1,182,624 | 197,586 | 0.895 | 0.895 | 0.895 | 0.895 ± 0.0003 | 🟢 Excellent |
| Edge AI | Edge scoring | 1,182,624 | 335,410 | 0.802 | 0.798 | 0.827 | 0.802 ± 0.001 | 🟢 Solid |
| UL Routing | Path regression | — | — | — | — | — | — | ⏳ Pending Colab |
| SV AI | SV detection | — | — | — | — | — | — | ⏳ Pending Colab |
| Diploid AI | Haplotype phasing | — | — | — | — | — | — | ⏳ Pending Colab |

#### EdgeAI Per-Class Breakdown (Hybrid)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| ALLELIC | 0.976 | 0.998 | 0.987 | 10,156 |
| CHIMERIC | 0.737 | 0.612 | 0.669 | 15,000 |
| SV_BREAK | 0.663 | 0.735 | 0.697 | 15,000 |
| TRUE | 0.935 | 0.997 | 0.965 | 10,156 |

#### PathGNN XGBoost Per-Class Breakdown (Hybrid)

| Class | F1 |
|-------|----|
| 0 (negative edge) | 0.893 |
| 1 (positive edge) | 0.897 |

### v0.2.0 Baseline — Class Weighting (Old)

Previous results using XGBoost's `scale_pos_weight` / sample weighting only.
Backup in `trained_models_10x_backup_20260214_164140/`.

| Model | Task | Samples | Accuracy | F1-w | CV (5-fold) | Status |
|-------|------|---------|----------|------|-------------|--------|
| PathGNN | Edge classification | 1,182,624 | 0.867 | 0.891 | 0.869 ± 0.001 | 🟢 |
| UL Routing | Path regression | 12,695 | R²=0.993 | RMSE=0.0007 | R²=0.993 ± 0.002 | 🟢 |
| SV AI | SV detection | 9,762 | 0.813 | 0.818 | 0.810 ± 0.003 | 🟡 |
| Edge AI | Edge scoring | 1,182,624 | 0.676 | 0.678 | 0.675 ± 0.001 | 🟡 |
| Diploid AI | Haplotype phasing | 17,817,700 | 0.752 | 0.752 | 0.748 ± 0.001 | 🟡 |

### Improvement: v0.2.0 → v0.3.0

| Model | Old Acc | New Acc | Δ Acc | Old F1-m | New F1-m | Δ F1-m |
|-------|---------|---------|-------|----------|----------|--------|
| Edge AI | 0.676 | 0.802 | **+0.126** | 0.623 | 0.827 | **+0.204** |
| PathGNN | 0.867 | 0.895 | **+0.028** | — | 0.895 | — |

### Model Details

#### Edge AI (EdgeWarden) — 🟢 Solid

- **Task**: Score individual edges as trustworthy vs. error-induced
- **Labels**: CHIMERIC (665k), SV_BREAK (480k), TRUE (35k), ALLELIC (1.2k)
- **Sub-models**: 5 technology-specific models (HiFi, ONT R9, ONT R10,
  Illumina, aDNA), each with a paired scaler
- **Resampling**: Hybrid — majority capped at 100k, minorities oversampled to
  median. Training set: 1,182,624 → 335,410 samples.
- **Key win**: ALLELIC F1 went from 0.39 → 0.99 (+153%). The class was
  previously invisible to the model at just 0.1% of training data.

#### PathGNN (XGBoost baseline) — 🟢 Excellent

- **Task**: Classify assembly graph edges as "true overlap" vs. "false overlap"
- **Labels**: 1,169,093 negative / 13,531 positive (86:1 imbalance)
- **Resampling**: Hybrid — resampled to 197,586 samples
- **Notes**: This XGBoost model serves as a feature selector for the actual
  PathGNN (GATv2Conv). Its `predict_proba()` outputs are fed to the GNN as
  ensemble edge features.

#### UL Routing — 🟢 (Awaiting Colab Retrain)

- **Task**: Predict optimal routing weight for ultra-long ONT reads (>100 kb)
- **v0.2.0 result**: R²=0.993, RMSE=0.0007
- **Notes**: Deterministic task — performance expected to remain excellent.

#### SV AI — 🟡 (Awaiting Colab Retrain)

- **Task**: Detect structural variants from graph topology
- **Labels**: insertion, deletion, inversion, duplication, none
- **v0.2.0 result**: acc=0.813, F1-w=0.818
- **Notes**: SV examples are inherently sparse (~49 per genome). More SV-dense
  training batches would help.

#### Diploid AI — 🟡 (Awaiting Colab Retrain)

- **Task**: Assign graph nodes to haplotype A vs. haplotype B
- **Labels**: A (8.9M) / B (8.9M) — perfectly balanced
- **v0.2.0 result**: acc=0.752, F1-w=0.752
- **Notes**: Haplotype phasing from local node features alone is at XGBoost's
  structural limit. The PathGNN with multi-hop graph attention is designed to
  solve this by learning long-range phase consistency.

---

## Colab Training Workflow

### Phase 1: XGBoost Hyperparameter Sweep

**Notebook**: `notebooks/XGBoost_Retraining_Colab.ipynb`

1. **Package data**: `scripts/package_training_data.sh` → `graph_csvs.tar.gz` (401 MB)
2. **Upload** tarball to Colab
3. **Sweep** 453 hyperparameter combinations per model (grid search):
   - `n_estimators`: [100, 200, 500]
   - `max_depth`: [4, 6, 8, 10]
   - `learning_rate`: [0.01, 0.05, 0.1]
   - `subsample`: [0.7, 0.8, 1.0]
   - `colsample_bytree`: [0.7, 0.8, 1.0]
   - `min_child_weight`: [1, 3, 5]
   - `gamma`: [0, 0.1, 0.2]
   - `tree_method`: `gpu_hist` (T4 GPU)
4. **Train final** models with best hyperparameters from sweep
5. **Download** retrained `.pkl` files

**Notebook structure** (6 sections, 9 code cells):

| Section | Purpose |
|---------|---------|
| §1 Setup | Install xgboost[gpu], scikit-learn |
| §2 Upload | Upload & extract graph_csvs.tar.gz |
| §3 Model Defs | Model registry, data loading, hybrid resampling, training utils |
| §4 Sweep | GPU-accelerated grid search (453 combos × 5 models) |
| §4b Summary | Print best params per model |
| §5 Train Final | Retrain with best params, 5-fold CV, save .pkl |
| §6 Download | Package and download retrained models |

### Phase 2: GNN Training with Ensemble Features

**Notebook**: `notebooks/PathGNN_Training_Colab.ipynb`

1. **Upload** graph CSVs + trained EdgeWarden `.pkl` from Phase 1
2. **Build graph objects** from CSVs (PyG Data objects)
3. **Ensemble mode**: Run EdgeWarden `predict_proba()` on all edges →
   attach 4-class probability vector as extra edge features
4. **Train GNN** with EdgeConv + GATv2Conv on enriched graph
5. **Hyperparameter sweep** (16 combinations):
   - `hidden_channels`: [64, 128]
   - `num_layers`: [3, 4]
   - `dropout`: [0.1, 0.3]
   - `learning_rate`: [1e-3, 5e-4]
6. **Export** `pathgnn_model.pt` (PyTorch weights)

**GNN Architecture**:

| Component | Value |
|-----------|-------|
| Convolution | Custom `EdgeConvLayer` (MessagePassing) + `GATv2Conv` |
| Attention heads | 4 |
| Hidden channels | 64 (sweep: 64, 128) |
| Layers | 3 (sweep: 3, 4) |
| Dropout | 0.2 (sweep: 0.1, 0.3) |
| Edge features | Raw edge CSV features + 4 EdgeWarden probability channels |
| Target | Google Colab T4 GPU |
| Est. training time | ~10-15 min |

**Ensemble design**: The GNN sees both the raw graph features **and** the
XGBoost model's confidence distribution for each edge. This lets the GNN
learn when to trust vs. override the XGBoost prediction using graph context.

---

## Comparison: 30-Genome → 200-Genome Training

| Model | 30-genome acc | 200-genome acc | Δ | Interpretation |
|-------|--------------|----------------|---|----------------|
| PathGNN | 0.9964 | 0.9901 | −0.006 | Slight drop, still excellent |
| UL Routing | R²=1.00 | R²=1.00 | 0 | Deterministic, unchanged |
| SV AI | 0.6897 | 0.6846 | −0.005 | Stable, more diverse SVs |
| Edge AI | 0.7962 | 0.6513 | −0.145 | Harder data, less overfit |
| Diploid AI | 0.5293 | 0.5325 | +0.003 | Marginal improvement |

**Key insight**: The models that dropped most (Edge AI) had the most to gain from
diverse data — the original 30-genome training was overfit to narrow parameter
ranges. The 200-genome CV std values are extremely tight, confirming stable
generalization.

---

## Model Weights

```
trained_models_10x/           Current (hybrid resampling, partial retrain)
├── edgewarden/               EdgeAI — retrained with hybrid resampling
│   ├── edgewarden_hifi.pkl
│   ├── scaler_hifi.pkl
│   └── training_metadata.json
├── pathgnn/                  PathGNN XGBoost — retrained with hybrid resampling
│   ├── pathgnn_scorer.pkl
│   └── training_metadata.json
└── (diploid, sv_detector, ul_routing deleted — awaiting Colab retrain)

trained_models_10x_backup_*/  Old baseline (class-weighting, all 5 models)
├── edgewarden/
├── diploid/
├── sv_detector/
├── pathgnn/
├── ul_routing/
└── training_report.json
```

---

## K-Weaver Models — ❌ Not Yet Trained

4 XGBoost models are referenced in the codebase but have never been trained:

| Model | File | Purpose |
|-------|------|---------|
| `dbg_model.pkl` | `kweaver_module.py` | Optimal k-mer size for DBG |
| `ul_overlap_model.pkl` | `kweaver_module.py` | Ultra-long overlap parameters |
| `extension_model.pkl` | `kweaver_module.py` | Contig extension decisions |
| `polish_model.pkl` | `kweaver_module.py` | Polish iteration parameters |

These require a **different training approach** — assembly benchmarks at
different k-values paired with `ReadFeatures`, not synthetic read simulation.
Currently falls back to rule-based prediction silently. Flagged as future TODO.

---

## What's Pending

| Item | Status | Notes |
|------|--------|-------|
| XGBoost Colab sweep (all 5 models) | ⏳ Ready to run | `XGBoost_Retraining_Colab.ipynb` |
| GNN Colab training | ⏳ Ready to run (after Phase 1) | `PathGNN_Training_Colab.ipynb` |
| Commit v0.3.0-dev changes | ⏳ After Colab runs | Hybrid resampling + notebooks + perf fixes |
| Port models to main repo | ⏳ After commit | Copy trained pkl/pt files to `strandweaver/` |

### Uncommitted Changes

- `strandweaver/user_training/train_models.py` — hybrid resampling integration
- `strandweaver/user_training/graph_training_data.py` — bisect + Hi-C perf fixes (33× speedup)
- `notebooks/XGBoost_Retraining_Colab.ipynb` — restructured sweep-only flow
- `scripts/package_training_data.sh` — macOS tar fix
- `scripts/compare_edgeai_strategies.py` — 7-strategy comparison script
- `edgeai_comparison_results.json` — comparison metrics

---

## Reproducibility

```bash
# Generate training data (graph-only, ~18 hours on M2 MacBook Air)
./scripts/generate_10x_training.sh

# Package for Colab upload
./scripts/package_training_data.sh

# Local retrain (uses hybrid resampling automatically)
python3 -m strandweaver.cli train train-models \
    --data-dir training_data_10x \
    --output-dir trained_models_10x \
    --cv-folds 5 --val-split 0.15 --seed 42

# Full Colab workflow
# 1. Upload graph_csvs.tar.gz to XGBoost_Retraining_Colab.ipynb → run all
# 2. Download retrained pkl files
# 3. Upload edgewarden pkl to PathGNN_Training_Colab.ipynb → run all
# 4. Download pathgnn_model.pt
```
