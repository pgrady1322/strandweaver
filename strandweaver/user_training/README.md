# User-Configurable Training Data Generation

This directory provides a flexible, parameter-driven system for generating custom training data for StrandWeaver ML models. Unlike the scenario-based training system, this gives you full control over all genome and sequencing parameters.

## Quick Start

### Command-Line Interface

```bash
# Basic usage — 10 diploid genomes (1 Mb each) with HiFi reads
strandweaver train generate-data --genome-size 1000000 -n 10 -o training_data/test

# Multi-technology dataset with custom coverage
strandweaver train generate-data --genome-size 5000000 -n 50 \
  --read-types hifi --read-types ont --read-types ultra_long --read-types hic \
  --coverage 30 --coverage 20 --coverage 10 --coverage 15 \
  -o training_data/multi_tech

# Repeat-rich genomes
strandweaver train generate-data --genome-size 2000000 -n 20 \
  --repeat-density 0.60 --gc-content 0.35 \
  -o training_data/repeat_rich

# Generate data with graph training labels for model training
strandweaver train generate-data --genome-size 5000000 -n 50 \
  --read-types hifi --read-types ont --read-types ultra_long --read-types hic \
  --coverage 30 --coverage 20 --coverage 10 --coverage 20 \
  --graph-training -o training_data/with_graphs

# Train all 5 models from the generated CSVs
strandweaver train run --data-dir training_data/with_graphs -o trained_models/
```

### Python API

```python
from strandweaver.user_training import *

# Configure genome
genome_config = UserGenomeConfig(
    genome_size=1_000_000,      # 1 Mb genomes
    num_genomes=10,             # Generate 10 genomes
    gc_content=0.42,            # 42% GC
    repeat_density=0.30,        # 30% repeats
    ploidy=Ploidy.DIPLOID,      # Diploid genomes
    snp_rate=0.001,             # 0.1% SNPs
    sv_density=0.00001          # SV every 100kb
)

# Configure read types
read_configs = [
    UserReadConfig(ReadType.HIFI, coverage=30),
    UserReadConfig(ReadType.ULTRA_LONG, coverage=10),
    UserReadConfig(ReadType.HIC, coverage=20)
]

# Create full configuration
config = UserTrainingConfig(
    genome_config=genome_config,
    read_configs=read_configs,
    output_dir='training_data/custom'
)

# Generate data
summary = generate_custom_training_data(config)
print(f"Generated {summary['num_genomes_generated']} genomes in {summary['generation_time_human']}")
```

## Configuration Parameters

### Genome Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genome_size` | int | 1,000,000 | Genome size in bp (100 bp - 1 Gb) |
| `num_genomes` | int | 10 | Number of independent genomes to generate |
| `gc_content` | float | 0.42 | Base GC content (0.0 - 1.0) |
| `repeat_density` | float | 0.30 | Fraction of genome that is repetitive |
| `ploidy` | Ploidy | DIPLOID | Ploidy level (HAPLOID, DIPLOID, TRIPLOID, TETRAPLOID) |
| `snp_rate` | float | 0.001 | SNP rate per bp between haplotypes |
| `indel_rate` | float | 0.0001 | Small indel rate per bp |
| `sv_density` | float | 0.00001 | Structural variant density per bp |
| `sv_types` | List[str] | ['deletion', 'insertion', 'inversion', 'duplication'] | SV types to include |
| `centromere_count` | int | 1 | Number of centromeric regions |
| `gene_dense_fraction` | float | 0.30 | Fraction of genome that is gene-dense (higher GC) |
| `random_seed` | int | None | Random seed for reproducibility |

### Read Parameters

Each read type has its own configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `read_type` | ReadType | - | Technology (ILLUMINA, HIFI, ONT, ULTRA_LONG, HIC, ANCIENT_DNA) |
| `coverage` | float | 30.0 | Sequencing coverage depth |
| `read_length_mean` | int | Tech-specific | Mean read length in bp |
| `read_length_std` | int | Tech-specific | Read length standard deviation |
| `error_rate` | float | Tech-specific | Base error rate |
| `insert_size_mean` | int | 500 | Insert size for paired-end (Illumina, Hi-C) |
| `insert_size_std` | int | 100 | Insert size standard deviation |

**Technology-Specific Defaults:**

- **Illumina**: 150 bp reads, 0.1% error, 500 bp inserts
- **HiFi**: 15 kb reads, 0.1% error
- **ONT**: 20 kb reads, 5% error
- **Ultra-long**: 100 kb reads, 5% error
- **Hi-C**: 150 bp paired-end, 0.1% error
- **Ancient DNA**: 50 bp reads, 2% error, C→T damage

## Output Structure

```
training_data/
├── training_config.json          # Configuration used
├── generation_summary.json       # Statistics and metadata
└── genome_0000/
    ├── haplotype_A.fasta         # Haplotype A sequence
    ├── haplotype_B.fasta         # Haplotype B sequence
    ├── sv_truth.json             # Ground-truth SVs
    ├── metadata.json             # Genome metadata
    ├── hifi.fastq               # HiFi reads (if requested)
    ├── ont.fastq                # ONT reads (if requested)
    ├── ultralong.fastq          # Ultra-long reads (if requested)
    ├── hic_R1.fastq             # Hi-C R1 (if requested)
    ├── hic_R2.fastq             # Hi-C R2 (if requested)
    ├── illumina_R1.fastq        # Illumina R1 (if requested)
    ├── illumina_R2.fastq        # Illumina R2 (if requested)
    └── graph_training/          # (only with --graph-training)
        ├── edge_ai_training_g0000.csv      # EdgeAI features + labels
        ├── path_gnn_training_g0000.csv     # PathGNN features + labels
        ├── diploid_ai_training_g0000.csv   # DiploidAI features + labels
        ├── ul_route_training_g0000.csv     # UL routing features + scores
        ├── sv_detect_training_g0000.csv    # SV detection features + labels
        ├── overlap_graph_g0000.gfa         # Overlap graph (GFA v1)
        └── graph_summary.json              # Graph statistics
```

## Use Cases

### 1. Quick Test Dataset
```bash
strandweaver train generate-data --genome-size 100000 -n 5 \
  --read-types hifi --coverage 10 -o test_data
```

### 2. Production HiFi + UL + Hi-C Dataset
```bash
strandweaver train generate-data --genome-size 5000000 -n 100 \
  --read-types hifi --read-types ultra_long --read-types hic \
  --coverage 30 --coverage 10 --coverage 20 \
  -o production_data
```

### 3. Repeat-Heavy Genomes
```bash
strandweaver train generate-data --genome-size 2000000 -n 50 \
  --repeat-density 0.70 --read-types hifi --read-types ont \
  -o repeat_heavy
```

### 4. High Heterozygosity (SV-Dense)
```bash
strandweaver train generate-data --genome-size 3000000 -n 30 \
  --snp-rate 0.002 --sv-density 0.0001 \
  -o high_hetero
```

### 5. Ancient DNA Dataset
```bash
strandweaver train generate-data --genome-size 1000000 -n 20 \
  --read-types ancient_dna --read-types illumina \
  --coverage 5 --coverage 10 \
  -o ancient_dna
```

### 6. Full Graph Training Dataset (for model training)
```bash
# Generate data with graph labels
strandweaver train generate-data --genome-size 5000000 -n 100 \
  --read-types hifi --read-types ont --read-types ultra_long --read-types hic \
  --coverage 40 --coverage 20 --coverage 10 --coverage 30 \
  --graph-training --min-overlap-bp 500 --min-overlap-identity 0.90 \
  -o graph_training_full

# Train all 5 models from the generated CSVs
strandweaver train run --data-dir graph_training_full/ -o trained_models/
```

## Performance Estimates

Training data generation time varies with genome size and complexity:

| Genome Size | # Genomes | Read Types | Graph? | Estimated Time |
|-------------|-----------|------------|--------|----------------|
| 100 kb | 10 | 1 (HiFi) | No | ~1 minute |
| 1 Mb | 10 | 1 (HiFi) | No | ~3 minutes |
| 1 Mb | 100 | 1 (HiFi) | No | ~20 minutes |
| 5 Mb | 50 | 3 (HiFi, UL, Hi-C) | No | ~60 minutes |
| 5 Mb | 50 | 3 (HiFi, UL, Hi-C) | Yes | ~75 minutes |
| 10 Mb | 20 | 4 (all) | Yes | ~120 minutes |

Model training time (after data generation):

| # Training Rows | Models | CV Folds | Estimated Time |
|-----------------|--------|----------|----------------|
| 1,000 | All 5 | 5 | ~30 seconds |
| 10,000 | All 5 | 5 | ~2 minutes |
| 100,000 | All 5 | 5 | ~15 minutes |
| 1,000,000 | All 5 | 5 | ~60 minutes |

*Times are approximate and depend on CPU, repeat density, and SV complexity*

## Advanced Usage

### Save and Load Configurations

```python
import json
from strandweaver.user_training import UserTrainingConfig

# Save configuration
config.to_dict()
with open('my_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)

# Load configuration
with open('my_config.json', 'r') as f:
    config_dict = json.load(f)
config = UserTrainingConfig.from_dict(config_dict)
```

### Reproducible Datasets

```python
# Use random seed for reproducibility
genome_config = UserGenomeConfig(
    genome_size=1_000_000,
    num_genomes=10,
    random_seed=42  # Same seed = same genomes
)
```

### Custom Read Parameters

```python
# Override technology defaults
hifi_config = UserReadConfig(
    read_type=ReadType.HIFI,
    coverage=50,              # Higher coverage
    read_length_mean=20000,   # Longer reads
    error_rate=0.0005         # Lower error rate
)
```

## Troubleshooting

### Memory Issues
- Reduce `genome_size` or `num_genomes`
- Generate genomes sequentially instead of parallel
- Use smaller read coverage values

### Slow Generation
- Reduce `repeat_density` (repeats are computationally expensive)
- Lower `sv_density` (SV insertion is slower)
- Use fewer read types
- Decrease `num_workers` if CPU-bound

### Disk Space
- Each 1 Mb diploid genome ≈ 2 MB on disk (uncompressed FASTA)
- Read files vary by coverage: 30x HiFi on 1 Mb ≈ 450 MB
- Graph training CSVs add ~1–5 MB per genome depending on coverage
- Use `--no-compress` flag cautiously for large datasets

### Graph Training Issues
- **0 overlaps detected**: Increase coverage or decrease `--min-overlap-bp`. Reads must overlap by ≥500 bp by default.
- **Too few training rows**: Increase `--num-genomes` or use more read types. Each genome produces one set of CSVs.
- **Missing model dependencies**: `train_models` requires `xgboost` and `scikit-learn`. Install with `pip install strandweaver[ai]`.
- **Training fails with "insufficient data"**: Minimum 10 samples per model by default. Generate more genomes or lower the threshold with `--min-samples` (Python API).

## Comparison to Scenario-Based Training

| Feature | Scenario-Based | User-Configurable |
|---------|----------------|-------------------|
| **Ease of use** | ✅ Very easy | ⚠️ Requires parameter knowledge |
| **Flexibility** | ❌ Fixed scenarios | ✅ Full control |
| **Reproducibility** | ✅ Named scenarios | ✅ Saved configurations |
| **Graph training** | ❌ Not available | ✅ Labelled graphs + CSV export |
| **Model training** | ❌ Manual | ✅ Integrated training runner |
| **Best for** | Quick starts, standard use cases | Custom organisms, specific research |

## Training Models with Generated Data

Once you have generated training data (with `--graph-training`), use the training runner to train all five graph-related ML models.

### Prerequisites

```bash
# Install training dependencies (XGBoost + scikit-learn are required)
pip install strandweaver[ai]

# Or install manually
pip install xgboost scikit-learn numpy
```

> **Note:** PyTorch is optional — it is only used to create a small scaffold checkpoint for PathGNN pipeline integration. All five core models are trained with XGBoost.

### Quick Start

```bash
# Train all 5 model types from graph training CSVs
strandweaver train run --data-dir training_data/production -o trained_models/
```

That single command:
1. Recursively discovers CSV files under `training_data/` (across all `genome_*/graph_training/` directories)
2. Trains each model with train/val split + early stopping
3. Runs k-fold cross-validation for robust metrics
4. Saves model weights in the pipeline-compatible directory layout
5. Writes a full `training_report.json` with metrics and configuration

### What Gets Trained

| Model | CSV Pattern | Task | Features | Labels |
|-------|-------------|------|----------|--------|
| **EdgeAI** | `edge_ai_training_g*.csv` | Multiclass (5 classes) | 17-D: overlap length/identity, read lengths, coverage, GC, repeat fraction, k-mer diversity, branching, Hi-C, MAPQ | TRUE · ALLELIC · REPEAT · SV_BREAK · CHIMERIC |
| **PathGNN** | `path_gnn_training_g*.csv` | Binary | 16-D: overlap metrics, coverage consistency, GC similarity, repeat match, branching, path support, Hi-C, topology, UL support, SV evidence | 0 (not on correct path) · 1 (on correct path) |
| **DiploidAI** | `diploid_ai_training_g*.csv` | Multiclass (5 classes) | 10-D: coverage, GC, repeat fraction, k-mer diversity, branching, Hi-C density, allele frequency, heterozygosity, phase consistency, mappability | A · B · BOTH · REPEAT · UNKNOWN |
| **UL Routing** | `ul_route_training_g*.csv` | Regression | 12-D: path length, branches, coverage mean/std, sequence identity, MAPQ, gaps, k-mer/orientation consistency, UL span, route complexity | Float score 0.0 – 1.0 |
| **SV Detection** | `sv_detect_training_g*.csv` | Multiclass (6 classes) | 14-D: coverage mean/std/median, GC, repeat fraction, k-mer diversity, branching, Hi-C disruption, UL support, MAPQ, region length, breakpoint precision, allele balance, phase switch rate | deletion · insertion · inversion · duplication · translocation · none |

### Saved Model Layout

The training runner writes models in the exact directory structure that `strandweaver pipeline --model-dir` expects:

```
trained_models/
├── edgewarden/
│   ├── edgewarden_hifi.pkl            # XGBoost classifier
│   ├── scaler_hifi.pkl                # StandardScaler
│   ├── edgewarden_ont_r9.pkl
│   ├── scaler_ont_r9.pkl
│   ├── edgewarden_ont_r10.pkl
│   ├── scaler_ont_r10.pkl
│   ├── edgewarden_illumina.pkl
│   ├── scaler_illumina.pkl
│   ├── edgewarden_adna.pkl
│   ├── scaler_adna.pkl
│   └── training_metadata.json
├── pathgnn/
│   ├── pathgnn_scorer.pkl             # XGBoost binary classifier
│   ├── pathgnn_model.pt               # Scaffold checkpoint (pipeline compat)
│   └── training_metadata.json
├── diploid/
│   ├── diploid_model.pkl              # XGBoost multiclass + scaler + encoder
│   └── training_metadata.json
├── ul_routing/
│   ├── ul_routing_model.pkl           # XGBoost regressor + scaler
│   └── training_metadata.json
├── sv_detector/
│   ├── sv_detector_model.pkl          # XGBoost multiclass + scaler + encoder
│   └── training_metadata.json
└── training_report.json               # Full training metrics & configuration
```

### CLI Reference

```
strandweaver train run [OPTIONS]

Input / Output:
  --data-dir DIR          Directory containing graph training CSVs (required)
  -o, --output-dir DIR    Directory to save model weights (default: trained_models/)

Model Selection:
  --models MODEL          Which models to train (repeat for multiple, default: all 5)
                          Choices: edge_ai, path_gnn, diploid_ai, ul_routing, sv_ai

Hyperparameters:
  --max-depth N           XGBoost max tree depth (overrides per-model default)
  --learning-rate FLOAT   XGBoost learning rate
  --n-estimators N        Number of boosting rounds
  --n-folds N             Cross-validation folds (default: 5)
  --val-split FLOAT       Hold-out validation fraction (default: 0.15)
  --seed INT              Random seed (default: 42)

EdgeWarden Options:
  --edgewarden-techs TECH [...]
                          Technology slots to save (default: hifi ont_r9 ont_r10 illumina adna)

Output Options:
  -v, --verbose           DEBUG-level logging
```

### Training Examples

```bash
# Train only EdgeAI and DiploidAI with custom depth
strandweaver train run --data-dir training_data/ -o models/ \
  --models edge_ai --models diploid_ai --max-depth 8

# Quick 3-fold cross-validation
strandweaver train run --data-dir training_data/ -o models/ --n-folds 3
```

### Python API

```python
from strandweaver.user_training import train_all_models, ModelTrainingConfig

config = ModelTrainingConfig(
    data_dir='training_data/production',
    output_dir='trained_models/',
    n_folds=5,
    validation_split=0.15,
    random_seed=42,
)

report = train_all_models(config)

# Inspect results
for name, info in report['models'].items():
    m = info['metrics']
    if 'val_accuracy' in m:
        print(f"{name}: acc={m['val_accuracy']:.4f}  CV={m['cv_accuracy_mean']:.4f}")
    else:
        print(f"{name}: RMSE={m['val_rmse']:.4f}  R²={m['val_r2']:.4f}")
```

### Loading Saved Models

```python
from strandweaver.user_training import load_trained_model

# Load a model bundle
bundle = load_trained_model('trained_models/diploid/diploid_model.pkl')
model  = bundle['model']        # XGBClassifier
scaler = bundle['scaler']       # StandardScaler
le     = bundle['label_encoder'] # LabelEncoder

# Predict on new data
import numpy as np
X_new = scaler.transform([feature_vector])
pred  = model.predict(X_new)
label = le.inverse_transform(pred)
```

---

## Graph Training Data Architecture

This section describes what happens under the hood when you pass `--graph-training` to the data generation CLI.

### Pipeline Overview

For each simulated genome, the graph training module:

```
Simulated Reads  ──►  Overlap Detection  ──►  Noise Injection
                         (sweep-line)           (false edges)
        │                                           │
        ▼                                           ▼
   Build Overlap   ◄────────────────────   Directed Graph
     Graph                                  (nodes + edges)
        │
        ├──►  Extract Edge Features (17-D EdgeAI + 16-D PathGNN)
        ├──►  Extract Node Features (10-D signal for DiploidAI)
        ├──►  Label Edges (ground truth from haplotype origin)
        ├──►  Label Nodes (ground truth from haplotype assignment)
        ├──►  Compute UL Route Features (12-D per ultra-long read)
        ├──►  Compute SV Region Features (14-D per region)
        │
        └──►  Export CSVs + GFA
```

### Overlap Detection

Reads are grouped by chromosome. For every pair of reads on the same chromosome, the module checks for coordinate overlap (minimum `--min-overlap-bp`, default 500 bp) with an identity threshold (`--min-overlap-identity`, default 0.90). Overhangs must be < 30% of read length to filter out chimeric-looking alignments. This is a coordinate-based approach (using the known truth positions) rather than alignment-based, making it fast and exact for synthetic data.

### Noise Edge Injection

To give the classifiers negative training examples, a configurable fraction (default 10%) of false edges are injected between random read pairs that do **not** truly overlap. These edges are labelled `CHIMERIC` and have random overlap statistics, teaching the model to distinguish real signal from noise.

### Ground-Truth Labelling

**Edge labels** are assigned based on the haplotype origin of the two reads:
- `TRUE` — both reads from the same haplotype and truly overlapping
- `ALLELIC` — reads from different haplotypes (A vs B)
- `REPEAT` — reads from repeat-flagged regions on the same haplotype
- `SV_BREAK` — reads that span an SV breakpoint
- `CHIMERIC` — injected noise edge (no real overlap)

**Node labels** are assigned from the read's haplotype tag:
- `HAP_A` / `HAP_B` — from haplotype A or B
- `BOTH` — ambiguous / shared region
- `REPEAT` — from a repeat region
- `UNKNOWN` — fallback

### Feature Schemas

Features are computed per-edge or per-node and exported as CSV rows. The column order matches the model classes exactly:

**EdgeAI (17-D):**
`overlap_length`, `overlap_identity`, `read1_length`, `read2_length`, `coverage_r1`, `coverage_r2`, `gc_content_r1`, `gc_content_r2`, `repeat_fraction_r1`, `repeat_fraction_r2`, `kmer_diversity_r1`, `kmer_diversity_r2`, `branching_factor_r1`, `branching_factor_r2`, `hic_support`, `mapping_quality_r1`, `mapping_quality_r2`

**PathGNN (16-D):**
`overlap_length`, `overlap_identity`, `coverage_consistency`, `gc_similarity`, `repeat_match`, `branching_score`, `path_support`, `hic_contact`, `mapping_quality`, `kmer_match`, `sequence_complexity`, `orientation_score`, `distance_score`, `topology_score`, `ul_support`, `sv_evidence`

**DiploidAI Node Signals (10-D):**
`coverage`, `gc_content`, `repeat_fraction`, `kmer_diversity`, `branching_factor`, `hic_contact_density`, `allele_frequency`, `heterozygosity`, `phase_consistency`, `mappability`

**UL Routing (12-D):**
`path_length`, `num_branches`, `coverage_mean`, `coverage_std`, `sequence_identity`, `mapping_quality`, `num_gaps`, `gap_size_mean`, `kmer_consistency`, `orientation_consistency`, `ul_span`, `route_complexity`

**SV Detection (14-D):**
`coverage_mean`, `coverage_std`, `coverage_median`, `gc_content`, `repeat_fraction`, `kmer_diversity`, `branching_complexity`, `hic_disruption_score`, `ul_support`, `mapping_quality`, `region_length`, `breakpoint_precision`, `allele_balance`, `phase_switch_rate`

---

## Complete End-to-End Workflow

```bash
# ── 1. Generate synthetic diploid genomes with reads ──────────────
strandweaver train generate-data --genome-size 5000000 -n 100 \
  --read-types hifi --read-types ont --read-types ultra_long --read-types hic \
  --coverage 40 --coverage 20 --coverage 10 --coverage 30 \
  --repeat-density 0.45 --snp-rate 0.001 --sv-density 0.00005 \
  --graph-training -o training_data/production

# ── 2. Inspect what was generated ─────────────────────────────────
cat training_data/production/generation_summary.json
ls training_data/production/genome_0000/graph_training/

# ── 3. Train all 5 models ────────────────────────────────────────
strandweaver train run --data-dir training_data/production -o trained_models/ --n-folds 5

# ── 4. Review training report ────────────────────────────────────
cat trained_models/training_report.json | python -m json.tool

# ── 5. Assemble with your trained models ──────────────────────────
strandweaver pipeline \
  --hifi-long-reads sample_hifi.fastq.gz \
  --ont-ul sample_ul.fastq.gz \
  --hic-r1 sample_R1.fastq.gz --hic-r2 sample_R2.fastq.gz \
  --model-dir trained_models/ \
  -o assembly_output/ -t 32
```

## Output Structure (with graph training)

```
training_data/
├── training_config.json              # Configuration used
├── generation_summary.json           # Statistics and metadata
└── genome_0000/
    ├── haplotype_A.fasta             # Haplotype A sequence
    ├── haplotype_B.fasta             # Haplotype B sequence
    ├── sv_truth.json                 # Ground-truth SVs
    ├── metadata.json                 # Genome metadata
    ├── hifi.fastq                    # HiFi reads
    ├── ont.fastq                     # ONT reads
    ├── ultralong.fastq               # Ultra-long reads
    ├── hic_R1.fastq                  # Hi-C R1
    ├── hic_R2.fastq                  # Hi-C R2
    └── graph_training/
        ├── edge_ai_training_g0000.csv      # 17 features + edge label
        ├── path_gnn_training_g0000.csv     # 16 features + binary label
        ├── diploid_ai_training_g0000.csv   # 10 features + haplotype label
        ├── ul_route_training_g0000.csv     # 12 features + route score
        ├── sv_detect_training_g0000.csv    # 14 features + SV type
        ├── overlap_graph_g0000.gfa         # Graph in GFA v1 format
        └── graph_summary.json              # Graph statistics
```

## Training Data Requirements

**Minimum dataset size for training:**

| Use Case | Genomes | Genome Size | Read Types | Coverage |
|----------|---------|-------------|------------|----------|
| **Quick test** | 5–10 | 100 kb – 1 Mb | 1 (HiFi) | 10× |
| **Development** | 20–50 | 1 Mb – 5 Mb | 2–3 | 20–30× |
| **Production** | 100–500 | 5 Mb – 10 Mb | 3–4 | 30–40× |
| **High-performance** | 500+ | 10 Mb+ | 4+ | 40×+ |

**Recommended coverage by model:**
- **EdgeAI / PathGNN**: ≥20× long-read coverage (more reads → more edges → more training rows)
- **DiploidAI**: ≥30× per haplotype for robust node labels
- **UL Routing**: ≥5× ultra-long coverage (fewer reads, but longer)
- **SV Detection**: ≥10× combined long-read coverage; include `--sv-density 0.0001` for balanced positive/negative examples

**Rules of thumb:**
- Each 1 Mb genome at 30× HiFi produces ~2,000–4,000 graph edges (training rows)
- 100 genomes × 3,000 edges = 300,000 rows — sufficient for all five models
- Cross-validation folds should have ≥50 samples each; use `--n-folds 3` if data is small

---

**StrandWeaver User Training Infrastructure** | February 2026