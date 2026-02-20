# StrandWeaver — Training Progress (Internal)

> **Version**: v0.3.0-dev  
> **Last updated**: 2025-02-20  
> **Data location**: `training_data_10x/` (graph-only CSVs)  
> **Model weights**: `trained_models/` (5 XGBoost + PathGNN GNN retrained v2 + 4 K-Weaver regressors)  
> **Colab tarball**: `graph_csvs.tar.gz` (401 MB, all 850 CSVs)  
> **K-Weaver models**: `trained_models/kweaver/` (4 XGBoost `.pkl`, Colab-trained)  
> **Git LFS**: Configured for `*.pkl`, `*.pt`, `trained_models/*.tar.gz`

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

The v1 baseline used XGBoost's `scale_pos_weight` / sample weighting, which
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

## XGBoost Model Results — v2 (Current, Deployed)

All 5 XGBoost models retrained via Colab with GPU-accelerated hyperparameter
sweeps + hybrid resampling. Deployed to `trained_models/` in the main repo.
Metrics from `trained_models/training_report.json`.

| Model | Task | Raw Samples | Resampled | Accuracy | F1-w | F1-m | CV (5-fold) | Status |
|-------|------|-------------|-----------|----------|------|------|-------------|--------|
| Edge AI | Edge scoring | 1,182,624 | 335,410 | 0.881 | 0.880 | 0.896 | 0.878 ± 0.002 | 🟢 Excellent |
| PathGNN | Edge classification | 1,182,624 | 197,586 | 0.897 | 0.897 | 0.897 | 0.897 ± 0.001 | 🟢 Excellent |
| Diploid AI | Haplotype phasing | 1,151,100 | 1,151,100 | 0.862 | 0.862 | 0.862 | 0.858 ± 0.001 | 🟢 Solid |
| SV AI | SV detection | 15,857 | 15,857 | 0.823 | 0.828 | 0.557 | 0.817 ± 0.005 | 🟡 Adequate |
| UL Routing | Path regression | 12,695 | 12,695 | R²=0.997 | RMSE=0.0004 | — | R²=0.997 ± 0.0003 | 🟢 Excellent |

### EdgeAI v2 Per-Class Breakdown

| Class | F1 |
|-------|-----|
| ALLELIC | 1.000 |
| CHIMERIC | 0.816 |
| SV_BREAK | 0.803 |
| TRUE | 0.968 |

### Diploid AI v2 Per-Class Breakdown

| Class | F1 |
|-------|-----|
| Haplotype A | 0.861 |
| Haplotype B | 0.862 |

### SV AI v2 Per-Class Breakdown

| Class | F1 |
|-------|-----|
| insertion | 0.319 |
| deletion | 0.716 |
| inversion | 0.338 |
| duplication | 0.410 |
| none | 1.000 |

Note: SV minority classes remain weak — inherently sparse (~49 SVs per genome).
More SV-dense training batches or focal loss could help.

---

## PathGNN (GNN) — Retrained v2

The GNN is trained on Colab using PyTorch Geometric, with the v2 XGBoost
EdgeWarden `predict_proba()` outputs as ensemble edge features.

| Component | Value |
|-----------|-------|
| Convolution | Custom `EdgeConvLayer` (MessagePassing) + `GATv2Conv` |
| Attention heads | 4 |
| Hidden channels | 64 (swept: 64, 128) |
| Layers | 3 (swept: 3, 4) |
| Dropout | 0.2 (swept: 0.1, 0.3) |
| Edge features | Raw edge CSV features + 4 EdgeWarden probability channels |
| Target | Google Colab T4 GPU |
| Weight file | `trained_models/pathgnn/pathgnn_model.pt` |

**Ensemble design**: The GNN sees both the raw graph features **and** the
XGBoost model's confidence distribution for each edge. This lets the GNN
learn when to trust vs. override the XGBoost prediction using graph context.

---

## Improvement Summary: v1 → v2

| Model | v1 Acc | v2 Acc | Δ Acc | v1 F1-m | v2 F1-m | Δ F1-m |
|-------|--------|--------|-------|---------|---------|--------|
| Edge AI | 0.676 | 0.881 | **+0.205** | 0.623 | 0.896 | **+0.273** |
| PathGNN | 0.867 | 0.897 | **+0.030** | — | 0.897 | — |
| Diploid AI | 0.752 | 0.862 | **+0.110** | — | 0.862 | — |
| SV AI | 0.813 | 0.823 | **+0.010** | — | 0.557 | — |
| UL Routing | R²=0.993 | R²=0.997 | **+0.004** | — | — | — |

---

## Comparison: 30-Genome → 200-Genome Training

| Model | 30-genome acc | 200-genome acc (v1) | Δ | Interpretation |
|-------|--------------|---------------------|---|----------------|
| PathGNN | 0.9964 | 0.9901 | −0.006 | Slight drop, still excellent |
| UL Routing | R²=1.00 | R²=1.00 | 0 | Deterministic, unchanged |
| SV AI | 0.6897 | 0.6846 | −0.005 | Stable, more diverse SVs |
| Edge AI | 0.7962 | 0.6513 | −0.145 | Harder data, less overfit |
| Diploid AI | 0.5293 | 0.5325 | +0.003 | Marginal improvement |

**Key insight**: The models that dropped most (Edge AI) had the most to gain from
diverse data — the original 30-genome training was overfit to narrow parameter
ranges. The 200-genome CV std values are extremely tight, confirming stable
generalization. The v2 hybrid resampling sweep recovered all lost accuracy and
then some.

---

## Model Weights (Deployed)

```
trained_models/                    v2 — All models retrained, deployed to main repo
├── edgewarden/                    EdgeWarden — 5 tech-specific models + scalers
│   ├── edgewarden_hifi.pkl
│   ├── edgewarden_ont_r9.pkl
│   ├── edgewarden_ont_r10.pkl
│   ├── edgewarden_illumina.pkl
│   ├── edgewarden_adna.pkl
│   ├── scaler_hifi.pkl
│   ├── scaler_ont_r9.pkl
│   ├── scaler_ont_r10.pkl
│   ├── scaler_illumina.pkl
│   ├── scaler_adna.pkl
│   └── training_metadata.json
├── pathgnn/                       PathGNN — XGBoost scorer + GNN weights
│   ├── pathgnn_scorer.pkl
│   ├── pathgnn_model.pt
│   └── training_metadata.json
├── diploid/                       Diploid AI — haplotype phasing
│   ├── diploid_model.pkl
│   └── training_metadata.json
├── sv_detector/                   SV AI — structural variant detection
│   ├── sv_detector_model.pkl
│   └── training_metadata.json
├── ul_routing/                    UL Routing — ultra-long read routing
│   ├── ul_routing_model.pkl
│   └── training_metadata.json
├── kweaver/                       K-Weaver — optimal k-mer size prediction
│   ├── dbg_model.pkl
│   ├── ul_overlap_model.pkl
│   ├── extension_model.pkl
│   ├── polish_model.pkl
│   └── training_results.json
├── training_report.json           Full metrics for all models
└── TRAINING.md                    (Public-facing training doc)
```

All `.pkl` and `.pt` files tracked via **Git LFS** (configured in `.gitattributes`).

---

## K-Weaver Models — ✅ Trained (v1)

4 XGBoost regressors predicting optimal k-mer sizes from 19 `ReadFeatures`.
Trained on Google Colab (T4 GPU) using Optuna hyperparameter optimization
(20 trials) with CUDA/hist tree method. 300 synthetic assembly benchmarks
pairing different k-values with `ReadFeatures` across HiFi, ONT, and Illumina
technologies.

**Notebook**: `strandweaver/training/notebooks/KWeaver_Training_Colab.ipynb`

| Model | Target | R² | MAE | CV R² (5-fold) | Samples | Status |
|-------|--------|-----|-----|----------------|---------|--------|
| `dbg_model.pkl` | Best DBG k | 0.863 | 0.63 | 0.863 ± 0.064 | 300 | 🟢 Solid |
| `ul_overlap_model.pkl` | Best UL overlap k | 0.982 | 9.36 | 0.982 ± 0.020 | 300 | 🟢 Excellent |
| `extension_model.pkl` | Best extension k | 0.849 | 1.20 | 0.849 ± 0.074 | 300 | 🟢 Solid |
| `polish_model.pkl` | Best polish k | 0.881 | 1.51 | 0.881 ± 0.067 | 300 | 🟢 Solid |

### Key Design Decisions

- **Gaussian read-length prior** (55% weight): Technology-specific ideal
  k/read_length ratios prevent biologically inverted predictions (e.g., HiFi
  getting k=13 while Illumina gets k=102).
- **UL applicability marking**: `ul_applicable` flag set `False` when read N50
  < 50 Kb, with `ul_confidence` capped at 0.3. Pipeline warns at assembly time.
- **Fast k-scoring proxy**: `_fast_k_score()` uses k-mer spectrum analysis
  (~0.3s/k) instead of full DBG construction (~45s/k), reducing the k-sweep
  from ~90 hours to ~37 minutes.
- **Model search path**: `_load_models()` checks both
  `strandweaver/preprocessing/training/trained_models/` (package-internal) and
  `trained_models/kweaver/` (repo-level Colab export). Falls back to rule-based
  prediction if models not found.

Deployed to `trained_models/kweaver/` — commit `9ee961f`.

---

## Colab Training Workflow

### Phase 0: K-Weaver k-Mer Prediction Models

**Notebook**: `strandweaver/training/notebooks/KWeaver_Training_Colab.ipynb`

1. **Self-contained**: Generates synthetic assembly benchmarks internally (300 samples)
2. **Optuna sweep**: 20 trials per model with CUDA/hist tree method
3. **Trains 4 regressors**: `dbg_model`, `ul_overlap_model`, `extension_model`, `polish_model`
4. **Download** tarball → extract to `trained_models/kweaver/`

### Phase 0b: ErrorSmith Error Classification Models (Pending)

**Notebook**: `strandweaver/training/notebooks/ErrorSmith_Training_Colab.ipynb`

1. **Requires** CHM13 reference BAMs or SRA download
2. Uses minimap2 (apt-get install) for alignment-based error profiling
3. **Download** trained model to `trained_models/errorsmith/`

### Phase 1: XGBoost Hyperparameter Sweep

**Notebook**: `notebooks/XGBoost_Retraining_Colab.ipynb`

1. **Package data**: `scripts/package_training_data.sh` → `graph_csvs.tar.gz` (401 MB)
2. **Upload** tarball to Colab
3. **Sweep** hyperparameter combinations per model (Bayesian + grid):
   - `n_estimators`: [100–1000]
   - `max_depth`: [3–14]
   - `learning_rate`: [0.01–0.12]
   - `subsample`: [0.7–1.0]
   - `colsample_bytree`: [0.7–1.0]
   - `min_child_weight`: [1–10]
   - `gamma`: [0–0.35]
   - `reg_alpha`, `reg_lambda`: tuned per model
   - `tree_method`: `gpu_hist` (T4 GPU)
4. **Train final** models with best hyperparameters from sweep
5. **Download** retrained `.pkl` files

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

---

## Pipeline Integration

All 7 model slots are wired in `pipeline.py::_load_ai_models()`:

| Slot | Model | Loader | Fallback |
|------|-------|--------|----------|
| `adaptive_kmer` | K-Weaver | pickle | Rule-based k selection |
| `base_error_classifier` | ErrorSmith | pickle | Classical correction |
| `edge_ai` | EdgeWarden (dir) | per-tech pkl | Rule-based filtering |
| `path_gnn` | PathGNN | PyTorch `.pt` | Heuristic path scoring |
| `diploid_ai` | DiploidAI | pickle | Heuristic phasing |
| `ul_routing_ai` | UL Routing | pickle | Heuristic UL routing |
| `sv_ai` | SV Detector | pickle | Heuristic SV detection |

DiploidAI is additionally wired into `HaplotypeDetangler` at all 3 pipeline
call sites (main, HiFi, ONT) for 26-feature XGBoost-based node classification.

---

## What's Done

| Item | Status | Commit |
|------|--------|--------|
| 200-genome training data generation | ✅ Complete | `training_data_10x/` |
| Hybrid resampling strategy | ✅ Complete | `train_models.py` |
| Colab XGBoost sweep (all 5 models) | ✅ Complete | `XGBoost_Retraining_Colab.ipynb` |
| Colab GNN training | ✅ Complete | `PathGNN_Training_Colab.ipynb` |
| v2 models deployed to main repo | ✅ Complete | `cc1a5e2` |
| Git LFS configured | ✅ Complete | `947617e` + `63f3cff` |
| DiploidAI wired into HaplotypeDetangler | ✅ Complete | `cc1a5e2` |
| All model slots wired in pipeline | ✅ Complete | `cc1a5e2` |
| Genomics audit (G1–G24) — all resolved | ✅ Complete | `21bd8c8` |
| K-Weaver ML models (4 XGBoost) | ✅ Complete | `9ee961f` |
| K-Weaver Colab training notebook | ✅ Complete | `a611edb` |
| K-Weaver UL applicability marking | ✅ Complete | `f29150a` |
| K-Weaver assembly-time UL warning | ✅ Complete | `dcf933d` |
| 268 tests passing | ✅ Complete | — |

## What's Pending

| Item | Status | Notes |
|------|--------|-------|
| ErrorSmith ML models | ❌ Not started | Colab notebook ready (`ErrorSmith_Training_Colab.ipynb`), needs CHM13 BAM data |
| SV AI minority class improvement | 🟡 Future | Focal loss or SV-dense training batches |
| Real genome validation | 🟡 Future | Test v2 models on CHM13 / HG002 assemblies |
| v0.2 release | 🟡 In progress | Trained models ready; remaining v0.2 roadmap items in README |

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
    --output-dir trained_models \
    --cv-folds 5 --val-split 0.15 --seed 42

# Full Colab workflow
# 1. Upload graph_csvs.tar.gz to XGBoost_Retraining_Colab.ipynb → run all
# 2. Download retrained pkl files
# 3. Upload edgewarden pkl to PathGNN_Training_Colab.ipynb → run all
# 4. Download pathgnn_model.pt
```
