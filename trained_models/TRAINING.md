# 🧠 AI Model Training

**Custom Model Training & Retraining Guide for StrandWeaver**

[![Models](https://img.shields.io/badge/models-7%20AI%20modules-green.svg)](#model-architecture)
[![Training Data](https://img.shields.io/badge/training%20data-200%20genomes-blue.svg)](#training-data-generation)
[![XGBoost](https://img.shields.io/badge/XGBoost-9%20models-orange.svg)](#xgboost-models)
[![GNN](https://img.shields.io/badge/GNN-GATv2Conv-purple.svg)](#pathgnn-graph-neural-network)

StrandWeaver ships with pre-trained models for all 7 AI modules. This guide covers how the models were trained, how to generate custom training data for your organism or sequencing technology, and how to retrain from scratch.

> **Note**: Pre-trained models are included with the v0.2+ release. You only need this guide if you want to train custom models for organism-specific optimization.

---

## 📋 Table of Contents

- [Model Architecture](#-model-architecture)
- [Shipped Models](#-shipped-models)
- [Training Data Generation](#-training-data-generation)
- [Custom Training](#-custom-training)
- [Colab GPU Training](#-colab-gpu-training)
- [Model Performance](#-model-performance)
- [Advanced: Class Imbalance](#-advanced-class-imbalance)

---

## 🏗️ Model Architecture

StrandWeaver uses a **two-tier ML architecture**:

| Tier | Framework | Models | Purpose |
|------|-----------|--------|---------|
| **Tabular** | XGBoost | 9 models | Classification & regression on extracted graph/read features |
| **Graph** | PyTorch Geometric | 1 GNN | Graph-aware edge classification using GATv2Conv attention |

The XGBoost models handle structured feature analysis (edge scoring, haplotype phasing, SV detection, ultra-long routing), while the GNN propagates information across assembly graph neighborhoods to capture structural patterns that tabular features miss.

### Ensemble Design

The PathGNN receives both raw graph features **and** EdgeWarden's XGBoost `predict_proba()` outputs as edge features. This lets the GNN learn when to trust vs. override the tabular model's predictions using graph context.

```
Assembly Graph CSVs
        │
        ├──► XGBoost Models (5×) ──► Edge scores, SV calls, phasing, routing
        │         │
        │         ▼ predict_proba()
        │         │
        └────────►├──► PathGNN (GATv2Conv) ──► Final edge classification
                  │
           Raw features + XGBoost probabilities
```

---

## 📦 Shipped Models

All pre-trained models are stored in `trained_models/` and managed via Git LFS:

| Module | Model | File(s) | Task |
|--------|-------|---------|------|
| 🛡️ **EdgeWarden** | XGBoost (×5) | `edgewarden/edgewarden_{tech}.pkl` | Edge quality scoring (per-technology) |
| � **K-Weaver** | XGBoost (×4) | `kweaver/{dbg,ul_overlap,extension,polish}_model.pkl` | Optimal k-mer size prediction |
| �🧬 **PathGNN** | GATv2Conv GNN | `pathgnn/pathgnn_model.pt` | Graph-aware edge classification |
| 🔀 **DiploidAI** | XGBoost | `diploid/diploid_model.pkl` | Haplotype phasing (26 features) |
| 🔍 **SVScribe** | XGBoost | `sv_detector/sv_detector_model.pkl` | Structural variant detection |
| 🧵 **ThreadCompass** | XGBoost | `ul_routing/ul_routing_model.pkl` | Ultra-long read routing |

EdgeWarden includes 5 technology-specific models with paired feature scalers:
- `edgewarden_hifi.pkl` / `scaler_hifi.pkl`
- `edgewarden_ont_r9.pkl` / `scaler_ont_r9.pkl`
- `edgewarden_ont_r10.pkl` / `scaler_ont_r10.pkl`
- `edgewarden_illumina.pkl` / `scaler_illumina.pkl`
- `edgewarden_adna.pkl` / `scaler_adna.pkl`

> **Fallback behavior**: If any model file is missing, StrandWeaver automatically falls back to optimized heuristic scoring. The pipeline never fails due to a missing model.

K-Weaver includes 4 regression models predicting optimal k-mer sizes from 19 `ReadFeatures`:
- `dbg_model.pkl` — De Bruijn graph construction k
- `ul_overlap_model.pkl` — Ultra-long read overlap k
- `extension_model.pkl` — Contig extension k
- `polish_model.pkl` — Polishing iteration k

K-Weaver models also enforce **UL applicability**: when read N50 < 50 Kb, UL predictions are marked non-applicable with capped confidence, and the pipeline warns at assembly time.

---

## 🧪 Training Data Generation

Training data is generated from **synthetic diploid genomes** that produce realistic assembly graphs with known ground truth.

### Quick Start

```bash
# Generate training data (graph-only mode, ~7 min/genome)
strandweaver train generate-training-data \
    --num-genomes 25 \
    --genome-size 1000000 \
    --graph-only \
    --output-dir training_data/batch_01

# Generate with custom parameters
strandweaver train generate-training-data \
    --num-genomes 25 \
    --genome-size 1000000 \
    --graph-only \
    --repeat-fraction 0.50 \
    --gc-content 0.38 \
    --sv-density 2e-5 \
    --hifi-coverage 30 \
    --ont-coverage 20 \
    --hic-coverage 15 \
    --output-dir training_data/batch_02
```

### Graph-Only Mode

The `--graph-only` flag simulates reads **in-memory** for realistic coverage, overlap, and sequence features without writing FASTQ/FASTA to disk. This provides:

| Metric | Full Mode | Graph-Only |
|--------|-----------|------------|
| Time per genome | ~23 min | ~7 min |
| Disk per genome | ~230 MB | ~8.5 MB |
| **Speedup** | — | **3.3×** |
| **Disk savings** | — | **27×** |

### Recommended Batch Diversity

For robust model training, generate batches across a range of genome parameters:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Repeat fraction | 0.20–0.65 | Captures simple → highly repetitive genomes |
| GC content | 0.32–0.55 | AT-rich to GC-rich variation |
| SV density | 1e-5–5e-5 | Low to high structural variation |
| Read types | Mix of all | Multi-technology generalization |

The pre-trained models were trained on **200 genomes** across 8 diverse batches (850 CSVs, ~20.2M rows, 1.4 GB).

---

## 🚀 Custom Training

### Train All Models

```bash
# Train using local data (hybrid resampling applied automatically)
strandweaver train train-models \
    --data-dir training_data/ \
    --output-dir trained_models/ \
    --cv-folds 5 \
    --val-split 0.15 \
    --seed 42
```

### Train Individual Models

```bash
# Retrain only EdgeWarden
strandweaver train train-models \
    --data-dir training_data/ \
    --output-dir trained_models/ \
    --model edge_ai

# Retrain only DiploidAI
strandweaver train train-models \
    --data-dir training_data/ \
    --output-dir trained_models/ \
    --model diploid_ai
```

### Validate Models

```bash
# Run 5-fold cross-validation on trained models
strandweaver train evaluate-models \
    --data-dir training_data/ \
    --model-dir trained_models/ \
    --cv-folds 5
```

---

## ☁️ Colab GPU Training

For large-scale hyperparameter sweeps, we recommend Google Colab with a T4 GPU.

### Phase 0a: K-Weaver Training

1. **Open** `strandweaver/training/notebooks/KWeaver_Training_Colab.ipynb`

2. **Self-contained**: Generates 300 synthetic assembly benchmarks internally — no external data upload needed

3. **Run all cells** — Optuna sweeps 20 trials per model with CUDA/hist

4. **Download** tarball → extract to `trained_models/kweaver/`

### Phase 0b: ErrorSmith Training

1. **Open** `strandweaver/training/notebooks/ErrorSmith_Training_Colab.ipynb`

2. **Upload** CHM13 BAMs or use automatic SRA download

3. **Run all cells** — minimap2 alignment + error profiling + model training

4. **Download** trained model to `trained_models/errorsmith/`

### Phase 1: XGBoost Sweep

1. **Package training data**:
   ```bash
   # Creates graph_csvs.tar.gz (~401 MB for 200 genomes)
   ./scripts/package_training_data.sh
   ```

2. **Upload** `graph_csvs.tar.gz` to Google Colab

3. **Open** `notebooks/XGBoost_Retraining_Colab.ipynb` and run all cells

4. **Download** retrained `.pkl` files to `trained_models/`

### Phase 2: GNN Training

1. **Upload** graph CSVs + trained `edgewarden/` models from Phase 1

2. **Open** `notebooks/PathGNN_Training_Colab.ipynb` and run all cells

3. **Download** `pathgnn_model.pt` to `trained_models/pathgnn/`

### Hyperparameter Search Space

<details>
<summary>XGBoost sweep parameters</summary>

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 100–1000 |
| `max_depth` | 3–14 |
| `learning_rate` | 0.01–0.12 |
| `subsample` | 0.7–1.0 |
| `colsample_bytree` | 0.7–1.0 |
| `min_child_weight` | 1–10 |
| `gamma` | 0–0.35 |
| `tree_method` | `gpu_hist` |

</details>

<details>
<summary>GNN sweep parameters</summary>

| Parameter | Values |
|-----------|--------|
| Hidden channels | 64, 128 |
| Layers | 3, 4 |
| Dropout | 0.1, 0.3 |
| Learning rate | 1e-3, 5e-4 |
| Attention heads | 4 |
| Convolution | EdgeConvLayer + GATv2Conv |

</details>

---

## 📊 Model Performance

### Current Model Metrics (v2)

| Model | Accuracy | F1 (weighted) | F1 (macro) | CV (5-fold) |
|-------|----------|----------------|------------|-------------|
| 🛡️ EdgeWarden | 0.881 | 0.880 | 0.896 | 0.878 ± 0.002 |
| 🧬 PathGNN | 0.897 | 0.897 | 0.897 | 0.897 ± 0.001 |
| 🔀 DiploidAI | 0.862 | 0.862 | 0.862 | 0.858 ± 0.001 |
| 🔍 SVScribe | 0.823 | 0.828 | 0.557 | 0.817 ± 0.005 |
| 🧵 ThreadCompass | R²=0.997 | RMSE=0.0004 | — | R²=0.997 ± 0.0003 |

### K-Weaver Metrics (v1)

Trained on 300 synthetic assembly benchmarks using Optuna (20 trials) + 5-fold CV, CUDA/hist tree method.

| Model | Target | R² | MAE | CV (5-fold) |
|-------|--------|-----|-----|-------------|
| 🧠 DBG | Best DBG k | 0.863 | 0.63 | 0.863 ± 0.064 |
| 🧠 UL Overlap | Best UL overlap k | 0.982 | 9.36 | 0.982 ± 0.020 |
| 🧠 Extension | Best extension k | 0.849 | 1.20 | 0.849 ± 0.074 |
| 🧠 Polish | Best polish k | 0.881 | 1.51 | 0.881 ± 0.067 |

### EdgeWarden Per-Class Performance

| Edge Class | F1 Score | Description |
|------------|----------|-------------|
| TRUE | 0.968 | Genuine overlaps |
| ALLELIC | 1.000 | Haplotype-specific edges |
| CHIMERIC | 0.816 | Chimeric / artifact edges |
| SV_BREAK | 0.803 | Structural variant boundaries |

### SVScribe Per-Class Performance

| SV Type | F1 Score |
|---------|----------|
| Deletion | 0.716 |
| Duplication | 0.410 |
| Inversion | 0.338 |
| Insertion | 0.319 |
| None | 1.000 |

> **Note**: SV minority classes (insertion, inversion) have lower F1 due to inherently sparse training examples (~49 SVs per 1 Mb genome). Custom training with higher `--sv-density` can improve these for SV-rich organisms.

---

## 🔬 Advanced: Class Imbalance

Assembly graph edge labels are naturally imbalanced (CHIMERIC edges outnumber ALLELIC edges by 541×). StrandWeaver handles this automatically with **hybrid resampling**:

1. **Undersample** majority classes to a 100,000-sample cap
2. **Oversample** minority classes to the new median

This strategy activates automatically when the class imbalance ratio exceeds 5.0×.

| Metric | Without Resampling | With Hybrid Resampling |
|--------|-------------------|----------------------|
| F1-macro | 0.623 | **0.829** (+33%) |
| ALLELIC F1 | 0.391 | **1.000** (+156%) |
| Training samples | 1,182,624 | ~285,000 |

For custom training, hybrid resampling is the default behavior. To disable it:

```bash
strandweaver train train-models \
    --data-dir training_data/ \
    --output-dir trained_models/ \
    --no-rebalance
```

---

## 📚 Additional Resources

- [Main README](../README.md) — Full pipeline documentation
- [Training Report](training_report.json) — Detailed metrics for all models
- [K-Weaver Training Results](kweaver/training_results.json) — K-Weaver model metrics
- [K-Weaver Training Notebook](../strandweaver/training/notebooks/KWeaver_Training_Colab.ipynb)
- [ErrorSmith Training Notebook](../strandweaver/training/notebooks/ErrorSmith_Training_Colab.ipynb)
- [XGBoost Colab Notebook](../Non_Main_Commit_Files/notebooks/XGBoost_Retraining_Colab.ipynb)
- [PathGNN Colab Notebook](../Non_Main_Commit_Files/notebooks/PathGNN_Training_Colab.ipynb)

---

## 📄 License

StrandWeaver is dual-licensed under [Academic](../docs/LICENSE_ACADEMIC.md) and [Commercial](../docs/LICENSE_COMMERCIAL.md) licenses. See the main repository for details.
