# StrandWeaver

**AI & ML-Powered Multi-Technology Genome Assembler with GPU Acceleration**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-v0.2_Beta-green.svg)](docs/MASTER_DEVELOPMENT_ROADMAP.md)
[![Models](https://img.shields.io/badge/trained%20models-6%20XGBoost%20%2B%201%20GNN-orange.svg)](trained_models/TRAINING.md)
[![License](https://img.shields.io/badge/license-Dual%20License%20(Academic/Commercial)-blue.svg)](LICENSE_ACADEMIC.md)

**StrandWeaver** is a next-generation genome assembly pipeline combining machine-learning optimized technology-aware error correction, graph-based assembly with haplotype-aware graph neural network path resolution, and expanded features including comprehensive structural variant detection for ancient DNA, Illumina, ONT, ultra-long ONT, and PacBio sequencing data. Its goal is to relieve manual curation bottlenecks in traditional high-contiguity genome assembly by applying AI/ML to genome graph paths and other complex regions. It uses these technologies to improve accuracy and contiguity, but also to provide functional annotations, such as structural variants, during the assembly process.

StrandWeaver is optimized for NVIDIA GPU acceleration, but can also take advantage of MacOS Silicon MPS hardware for GPU acceleration.

StrandWeaver is inspired by the brilliant work of the authors of MaSuRCA *(1)*, Verkko *(2)*, and Hifiasm *(3)*, but includes novel elements such as:
- **Neural network-based haplotype assembly**: Graph topology simplification with strict protection of biological variation (SNPs, indels, CNVs) across haploid and diploid genomes (polyploid in a future release)
- **Multi-technology integration**: Seamless combination of ONT, PacBio HiFi, ultra-long reads, and Hi-C data in a unified assembly graph, and expanded methods for short read & ancient DNA (PacBio SBB and Illumina)
- **Dynamic K-mer size selection**: StrandWeaver optimizes different k-mer sizes for all parts of the assembly process based on heuristics and trained models
- **Comprehensive Path Scoring**: A score for each graph path is provided based on a comprehensive set of heuristics, including coverage, error profiles, and sequence repetitive complexity, as well as trained models
- **Ancient DNA optimization**: Machine learning models trained to profile and repair deamination damage (C→T/G→A patterns) with configurable confidence thresholds
- **Easy User Model Training**: CLI commands provided to generate a set of training data matching your genome / clade of choice

The pipeline can be custom trained using provided scripts for any data type (new sequencing technology) or organism-specific scenario (genomes with extreme repeat content, high heterozygosity, or complex structural variation). **Pre-trained models ship with v0.2+ and are used automatically.** See the [AI Model Training Guide](trained_models/TRAINING.md) for custom training.

> **🚀 v0.2.0 Beta (February 2026):** Trained ML models now ship for 6 of 8 AI modules (EdgeWarden, ErrorSmith, PathGNN, DiploidAI, ThreadCompass, SVScribe). All models retrained on 200 synthetic diploid genomes via Colab GPU sweeps. Pre-trained weights load automatically — no user setup required. See [v0.2 Release Notes](#v02-release-notes) below.
>
> **v0.3+:** K-Weaver trained models. Standalone `assemble` command. Polyploid assembly support. BUSCO integration.

### 🎯 Model Performance at a Glance

| Module | Type | Accuracy / R² | F1-macro | CV (5-fold) |
|--------|------|---------------|----------|-------------|
| 🛡️ EdgeWarden | XGBoost (×5) | 0.881 | 0.896 | 0.878 ± 0.002 |
| 🔧 ErrorSmith | XGBoost | 0.866 | 0.865 | 0.866 ± 0.001 |
| 🧬 PathGNN | GATv2Conv GNN | 0.897 | 0.897 | 0.897 ± 0.001 |
| 🔀 DiploidAI | XGBoost | 0.862 | 0.862 | 0.858 ± 0.001 |
| 🧵 ThreadCompass | XGBoost | R²=0.997 | — | R²=0.997 ± 0.0003 |
| 🔍 SVScribe | XGBoost | 0.823 | 0.557 | 0.817 ± 0.005 |
| 🧠 K-Weaver (DBG) | XGBoost | R²=0.863 | — | 0.863 ± 0.064 |
| 🧠 K-Weaver (UL Overlap) | XGBoost | R²=0.982 | — | 0.982 ± 0.020 |
| 🧠 K-Weaver (Extension) | XGBoost | R²=0.849 | — | 0.849 ± 0.074 |
| 🧠 K-Weaver (Polish) | XGBoost | R²=0.881 | — | 0.881 ± 0.067 |

> See the [AI Model Training Guide](trained_models/TRAINING.md) for full per-class breakdowns and training details.

---

## ✨ Key Features

### Core Assembly
- 🧬 **Multi-Technology Support**: Ancient DNA, Illumina, ONT R9/R10, PacBio HiFi/CLR, Ultra-long reads, with upcoming support for specific biases in ONT / PacBio chemistries & flow cells
- 🔀 **Hybrid Assembly**: Intelligently combine data from multiple sequencing platforms
- 📊 **Multi-Stage Assembly Pipeline**:
  - **Hybrid De Bruijn Graph (DBG) or Overlap-Layout-Consensus (OLC) based on input**: K-mer-based graph construction for long reads or short reads
  - **Edge Filtering**: AI-powered edge quality assessment with 80-feature extraction
  - **Path Resolution**: Haplotype-aware path selection (duplicate paths are preserved) with variation protection
  - **String Graph Overlay**: Ultra-long read integration for long-range connections
  - **Graph Neural Network Routing**: UL read-guided path optimization
  - **Hi-C Scaffolding**: Chromosome-scale scaffolding with proximity ligation
  - **Haplotype Phasing**: Spectral clustering with Hi-C contact matrices
- 🧬 **Diploid Assembly**: 
  - Protects SNP-level variation (configurable >99.5% identity threshold)
  - Never collapses across haplotype boundaries
  - Maintains diploid structure throughout assembly
  - Iterative refinement with phasing context (configurable 2-3 cycles) that backfeeds information to ML models for following iterations
  - Graph-guided and Hi-C-guided haplotype separation
- 🏛️ **Ancient DNA Mode**: Enhanced mapDamage2-inspired correction with C→T/G→A deamination modeling

### Advanced Features
- 🎯 **80-Feature Edge Scoring**: Edges are scores with comprehensive feature extraction:
  - 26 static features (graph topology, coverage, node properties)
  - 34 temporal features (quality/coverage trajectories, error patterns)
  - 20 expanded features (sequence complexity, boundaries, systematic errors)
  - If coverage / data does not adequately fulfull temporal / expanded features, falls back to static features
- 🔄 **Iterative Refinement**: 2-3 assembly iteration cycles with phasing-aware filtering, in which preliminary rounds feed phasing info into the next round to preserve strong paths
- 🧠 **AI-Powered Path Selection**: Graph neural networks optimize contig paths through complex regions
- 📊 **Structural Variant Detection**: ML module SVScribe identifies variants *during* assembly:
  - Deletions, insertions, inversions, duplications, translocations
  - Graph topology signatures
  - Ultra-long read spanning evidence
  - Hi-C long-range validation
- 📄 **Comprehensive Output**: 
  - Assembly graphs (GFA format)
  - BandageNG compatible coverage and Hi-C visualization files
  - Detailed statistics (N50, L50, coverage, QV)
  - SV calls (VCF/JSON)
  - Phasing information
  - IGV / UCSC overage tracks for all read types
  - Chromosome identification using node features, repeat detection, and synteny to allow easier downstream processing
- 🧪 **Training Infrastructure**: Pre-trained models ship with v0.2+; custom training supported for organism-specific optimization
- 🔌 **Modular Architecture**: All AI features can be selectively disabled for classical heuristics

### AI/ML Features (v0.2 — Trained Models)
- **8-Module AI Subsystem** — 6 ship with trained XGBoost/GNN weights, 2 use optimized heuristics:
  1. ⚙️ **K-Weaver**: K-mer optimization with rule-based selection *(trained models in v0.3)*
  2. 🧠 **ErrorSmith**: Per-base error classification with trained XGBoost — 5 error classes across 6 chemistries (acc: 0.87, F1-macro: 0.87)
  3. 🧠 **EdgeWarden**: 80-feature edge filtering with trained XGBoost — 5 technology-specific models (acc: 0.88, F1-macro: 0.90)
  4. 🧠 **PathGNN**: Graph neural network edge classification with GATv2Conv attention (acc: 0.90, F1-macro: 0.90)
  5. 🧠 **DiploidAI**: XGBoost haplotype phasing with 26-feature node classification (acc: 0.86, F1-macro: 0.86)
  6. 🧠 **ThreadCompass**: Ultra-long read routing with XGBoost regression (R²: 0.997)
  7. 🧠 **SVScribe**: Structural variant detection with XGBoost classifier (acc: 0.82, F1-weighted: 0.83)

- 🧠 = trained model ships | ⚙️ = optimized heuristic (trained models planned for v0.3)

---

## 📋 Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [AI/ML Training](#aiml-training)
- [Documentation](#documentation)

---

## 🔧 Installation

### Requirements
- Python 3.9+
- 8 GB RAM minimum (32+ GB recommended for large genomes)
- Disk space: 50-100 GB for intermediate files (genome-dependent)

### Dependencies

StrandWeaver uses a modular dependency system with three installation tiers:

**Core Dependencies (Always Installed)**:
- **Bioinformatics**: `biopython>=1.79`, `pysam>=0.19.0`
- **Numerical**: `numpy>=1.21.0`, `scipy>=1.9.0`, `pandas>=1.3.0`
- **Graph Processing**: `networkx>=2.6.0`
- **Hi-C/Phasing**: `scipy>=1.9.0`, `scikit-learn>=1.3.0`
- **CLI/IO**: `click>=8.0.0`, `pyyaml>=6.0`, `tqdm>=4.62.0`, `h5py>=3.5.0`
- **Performance**: `numba>=0.54.0`, `joblib>=1.1.0`

**AI/ML Dependencies (Optional - `[ai]` flag)**:
- **PyTorch**: `torch>=2.0.0` (CPU or GPU support)
- **Graph Neural Networks**: `pytorch-geometric>=2.3.0`
- **Gradient Boosting**: `xgboost>=2.0.0`

These are required for:
- Loading pre-trained models (shipped with v0.2+)
- Custom model training
- GPU-accelerated assembly (if CUDA available)

**Development Dependencies (Optional - `[dev]` flag)**:
- **Testing**: `pytest>=7.0.0`, `pytest-cov>=4.0.0`
- **Visualization**: `matplotlib>=3.4.0`, `seaborn>=0.11.0`
- **Documentation**: Development tools for contributors

**GPU Support**:
- CUDA 11.8+ for NVIDIA GPUs (automatic via PyTorch)
- MPS backend for Apple Silicon (automatic in macOS 12.3+)
- CPU fallback if no GPU detected

**Platform Compatibility**:
- ✅ Linux (x86_64, ARM64)
- ✅ macOS (Intel, Apple Silicon with MPS acceleration)
- ✅ Windows (via WSL2 recommended)

### Install from GitHub (Recommended)

StrandWeaver has several dependencies, especially if you plan on installing the AI/ML training dependencies, so it is **highly** recommended to install in a virtual environment (conda, python venv).

```bash
# Basic installation
pip install git+https://github.com/pgrady1322/strandweaver.git

# Recommended: Complete installation with all dependencies
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"

# With AI/training dependencies (PyTorch, XGBoost)
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[ai]"

# With developmental testing features
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[dev]"
```

---

## 🚀 Quick Start

StrandWeaver offers two execution modes:

| Mode | Usage | Best For |
|------|-------|----------|
| **Direct** | `strandweaver <command> [options]` | Small/medium datasets, local workstation, testing |
| **Nextflow** | `strandweaver <command> [options] --nextflow` | Large datasets, HPC clusters, parallel processing |

### Python CLI Mode

**Basic Long Read Assembly Example (PacBio)**
```bash
strandweaver pipeline \
  --hifi-long-reads hifi_reads.fastq \
  -o assembly_output/ \
  -t 8
```

**Hybrid Assembly with Multiple Ultra-Long Read Types**

*Note:* The `--ont-ul` flag is used for path-finding reads. Any platform of long reads can be provided, but shorter long reads will degrade the assembly. The `--ont-ul` name is retained for clarity / comparison with other assemblers.
```bash
strandweaver pipeline \
  --hifi-long-reads hifi_reads.fastq \
  --ont-long-reads ont_reads.fastq.gz \
  --ont-ul ultralong_reads.fastq \
  -o assembly_output/ \
  -t 16
```

**Mixed Technology Assembly with Hi-C**

*Note:* ANY platform of proximity ligation tech can be provided. StrandWeaver will optimize for Hi-C and Omni-C just as well as Pore-C and CiFi.
```bash
strandweaver pipeline \
  --hifi-long-reads hifi_reads.fastq \
  --ont-ul ultralong_reads.fastq \
  --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq \
  -o assembly_output/ \
  -t 16
```

### Nextflow Mode (HPC/Cloud)

**Local Execution**
```bash
nextflow run strandweaver/nextflow/main.nf \
  --hifi hifi_reads.fastq \
  --ont ont_reads.fastq \
  --hic_r1 hic_R1.fastq \
  --hic_r2 hic_R2.fastq \
  --outdir results/ \
  -profile local
```

**SLURM Cluster with Singularity**
```bash
nextflow run strandweaver/nextflow/main.nf \
  --hifi hifi_reads.fastq \
  --ont ont_reads.fastq \
  --ont_ul ultralong_reads.fastq \
  --hic_r1 hic_R1.fastq \
  --hic_r2 hic_R2.fastq \
  --outdir results/ \
  -profile slurm,singularity \
  -resume
```

**Huge Genome Mode** (parallel k-mer extraction)
```bash
nextflow run strandweaver/nextflow/main.nf \
  --hifi hifi_reads.fastq \
  --huge \
  --outdir results/ \
  -profile slurm
```

See [nextflow/README.md](nextflow/README.md) for complete Nextflow documentation.

#### Nextflow Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `local` | Direct execution on local machine | Testing, small datasets |
| `docker` | Docker containerization | Reproducibility, dependency isolation |
| `singularity` | Singularity containers | HPC clusters (no root required) |
| `slurm` | SLURM cluster scheduler | HPC parallel processing |
| `test` | Use synthetic *E. coli* data | Quick validation |

Combine profiles with commas: `-profile slurm,singularity`

### Individual Processing Commands

StrandWeaver provides standalone commands for each processing stage for instances in which you may want just corrected reads or read error profiles. StrandWeaver also supports mapping of reads to GFA graphs, and calling SVs on GFA graphs. Each command supports both direct and Nextflow execution.

#### Error Correction
```bash
# Direct mode
strandweaver correct --hifi reads.fq.gz -o corrected/ -t 8

# Nextflow mode (automatic parallelization)
strandweaver correct --hifi reads.fq.gz -o corrected/ \
  --nextflow --nf-profile slurm --correction-batch-size 100000
```

##### ErrorSmith Chemistry Designation

ErrorSmith uses chemistry-aware models trained on specific sequencing platforms.
You can specify a chemistry to match your data; **if your exact kit / flow cell
combination is not listed, pick the closest available model.**

| Flag | Choices | Default |
|------|---------|---------|
| `--hifi-chemistry` | `pacbio_hifi_sequel2` | `pacbio_hifi_sequel2` |
| `--ont-chemistry` | `ont_lsk110_r941`, `ont_lsk114_r1041` | `ont_lsk114_r1041` |
| `--ont-ul-chemistry` | `ont_ulk001_r941`, `ont_ulk114_r1041` | `ont_ulk001_r941` |
| `--illumina-chemistry` | `illumina_hiseq2500` | `illumina_hiseq2500` |

```bash
# Example: ONT R10.4.1 ligation + ultra-long with separate chemistries
strandweaver pipeline \
  --ont-long-reads ligation.bam --ont-chemistry ont_lsk114_r1041 \
  --ont-ul ultralong.bam --ont-ul-chemistry ont_ulk114_r1041 \
  -o assembly_output/ -t 16
```

The `profile` and `batch-correct` commands accept a single `--chemistry` flag
with any of the six values above.

#### K-mer Extraction
```bash
# Direct mode
strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl -t 8

# Nextflow mode (huge genomes >10GB)
strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl \
  --nextflow --nf-profile slurm --kmer-batch-size 2000000
```

#### Edge Scoring
```bash
# Direct mode
strandweaver nf-score-edges -e edges.json -a aligns.bam -o scored.json -t 8

# Nextflow mode (large graphs)
strandweaver nf-score-edges -e edges.json -a aligns.bam -o scored.json \
  --nextflow --nf-profile slurm --edge-batch-size 10000
```

#### Ultra-Long Read Mapping
```bash
# Direct mode
strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf --use-gpu

# Nextflow mode
strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf \
  --nextflow --nf-profile slurm --use-gpu --ul-batch-size 100
```

#### Hi-C Alignment
```bash
# Direct mode
strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \
  -g graph.gfa -o aligns.bam -t 8

# Nextflow mode (large Hi-C datasets)
strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \
  -g graph.gfa -o aligns.bam \
  --nextflow --nf-profile slurm --hic-batch-size 500000
```

#### Structural Variant Detection
```bash
# Direct mode
strandweaver nf-detect-svs -g graph.gfa -o variants.vcf -t 8

# Nextflow mode (large graphs)
strandweaver nf-detect-svs -g graph.gfa -o variants.vcf \
  --nextflow --nf-profile slurm --sv-batch-size 1000
```

### Performance Guidelines

**When to use Direct mode:** Dataset < 10GB, local workstation with 8+ cores, testing and debugging.

**When to use Nextflow mode:** Dataset > 10GB, HPC cluster available, need resume capability, want automatic parallelization.

| Command | Direct (1 node) | Nextflow (20 nodes) | Speedup |
|---------|----------------|---------------------|--------|
| `correct` | 20 hours | 2 hours | 10× |
| `extract-kmers` | 8 hours | 1.5 hours | 5× |
| `nf-score-edges` | 8 hours | 1.5 hours | 5× |
| `map-ul` | 6 hours | 1 hour | 6× |
| `align-hic` | 10 hours | 1.5 hours | 7× |
| `nf-detect-svs` | 4 hours | 1 hour | 4× |

---

## 🎯 Use Cases

### Machine-Learning-Tuned Genome Assembly with SV Calls
Combine ONT, HiFi, ultra-long reads, and Hi-C for chromosome-scale phased assemblies:
```bash
strandweaver pipeline \
  --ont-long-reads ont.fastq \
  --hifi-long-reads hifi.fastq \
  --ont-ul ultralong.fastq \
  --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq \
  --use-ai \
  -o genome_assembly/ \
  -t 32
```

### Ancient DNA Assembly
Optimize for deamination damage with specialized error correction:
```bash
strandweaver pipeline \
  -r1 ancient_reads.fastq --technology1 ancient \
  -o ancient_assembly/ \
  -t 16
```
Note that the assembly can also be run WITHOUT damage awareness features for comparison.
### SV-Rich Genome Analysis
Detect structural variants during assembly for cancer or population genomics:
```bash
strandweaver pipeline \
  --hifi-long-reads tumor.fastq \
  --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq \
  --min-sv-size 30 \
  -o tumor_assembly/ \
  -t 24
```

### Highly Heterozygous Diploid Assembly
Maintain haplotype separation for F1 hybrids or outcrossing species:
```bash
strandweaver pipeline \
  --hifi-long-reads hifi.fastq \
  --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq \
  --ploidy diploid \
  --edge-filter-mode strict \
  -o diploid_assembly/ \
  -t 32
```

---

## 🔬 Pipeline Ordering

### Preprocessing
1. **Classify**: Auto-detect sequencing technologies from FASTQ headers (supports ONT chemistry detection with LongBow)
2. **KWeaver**: ML-based k-mer optimization with rule-based fallback for dynamic k-mer selection
3. **Profile**: Error pattern profiling (substitutions, indels, homopolymers) with visualization
4. **Correct**: Technology-aware read correction (ONTCorrector, PacBioCorrector)

### Core Assembly
5. **Graph Building**: Graph construction (type of graph based on read type) from reads with streaming architecture
6. **EdgeWarden**: AI-powered graph edge filtering with 80-feature scoring
7. **PathWeaver**: GNN-based haplotype-aware path resolution with variation protection
8. **StringGraph**: Ultra-long read overlay for long-range connections
9. **ThreadCompass**: UL read routing optimization with trained models
10. **Hi-C Integration**: Proximity ligation contact matrix construction and edge addition
11. **HaplotypeDetangler**: Hi-C-augmented phasing via spectral clustering
12. **Iteration**: 3+ refinement cycles with phasing-aware filtering
13. **SVScribe**: Graph-based structural variant detection (DEL, INS, INV, DUP, TRA)
14. **Iterate or Finalize**: Contig and scaffold extraction with comprehensive statistics, or pass graph with feature scoring to pipeline for iteration 2+.

### Post-Assembly Analysis
15. **Misassembly Report**: Putative misassembly detection using multi-signal evidence (EdgeWarden confidence, coverage discontinuities, UL read conflicts, Hi-C violations). Outputs TSV and BED reports for genome-browser visualization.
    - `--misassembly-report` / `--no-misassembly-report` — Enabled by default
    - `--misassembly-min-confidence HIGH|MEDIUM|LOW` — Minimum confidence to flag (default: MEDIUM)
    - `--misassembly-format tsv,bed,json` — Comma-separated output formats (default: tsv,bed)
16. **Chromosome Classification**: Multi-tier scaffold classification to identify chromosomes vs. assembly artifacts.
    - **Tier 1** (always): Length, coverage, GC, connectivity, telomere detection
    - **Tier 2** (always): Gene content analysis (ORF / BLAST / Augustus / BUSCO)
    - **Tier 3** (`--id-chromosomes-advanced`): Hi-C self-contact patterns, synteny
    - Telomere flags: `--telomere-sequence` (default: TTAGGG), `--telomere-min-units` (default: 10), `--telomere-search-depth` (default: 5000 bp)

### Output Generation
17. **GFA Export**: Assembly graphs in GFA format with sequences
18. **BandageNG**: Visualization files with coverage tracks and final 0 - 1 range StrandWeaver scores (long/UL/Hi-C).
19. **Statistics**: N50, L50, coverage metrics, variation protection counts
20. **SV Calls**: Structural variants in VCF and JSON formats
21. **Phasing Info**: Haplotype assignments and confidence scores

### Post-Assembly CLI Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--misassembly-report` / `--no-misassembly-report` | Enabled | Generate misassembly report (TSV + BED) |
| `--misassembly-min-confidence` | `MEDIUM` | Minimum confidence for flags: `HIGH`, `MEDIUM`, `LOW` |
| `--misassembly-format` | `tsv,bed` | Comma-separated output formats: `tsv`, `bed`, `json` |
| `--id-chromosomes` | Off | Enable scaffold → chromosome classification (Tiers 1-2) |
| `--id-chromosomes-advanced` | Off | Add Hi-C pattern analysis & synteny (Tier 3) |
| `--gene-detection-method` | `orf` | Gene detection: `orf` (no deps), `blast`, `augustus`, `busco` |
| `--blast-db` | `nr` | BLAST database for gene detection |
| `--telomere-sequence` | `TTAGGG` | Telomere repeat motif. Alternatives: `TTTAGGG` (plants), `TTAGG` (insects) |
| `--telomere-min-units` | `10` | Minimum tandem repeats to call a telomere |
| `--telomere-search-depth` | `5000` | Base-pairs to search at each scaffold end |

---

## 🤖 AI/ML Features & Training

### Current AI/ML Capabilities

**K-Weaver** (K-mer Optimization):
- ML prediction for 4 assembly stages (DBG, UL overlap, extension, polish)
- Rule-based fallback if models unavailable
- Technology-specific k-mer selection

**ErrorSmith** (Technology-Aware Error Correction):
- Technology-specific error profiling (ONT, PacBio, Illumina, Ancient DNA)
- Homopolymer error detection and correction
- Ancient DNA deamination damage repair (C→T/G→A patterns)
- Confidence-based correction with configurable thresholds

**EdgeWarden** (Graph Edge Filtering):
- 80-feature extraction: static (26) + temporal (34) + expanded (20)
- Integrates alignment data (quality scores, coverage arrays)
- Technology-specific defaults (ONT: mean Q12, HiFi: mean Q30)
- Graceful degradation to 26 features if alignment data unavailable

**PathWeaver** (Haplotype-Aware Path Resolution):
- Graph topology simplification (NOT sequence extraction)
- Strict variation protection (>99.5% identity threshold)
- Never collapses across haplotype boundaries
- Maintains diploid structure (both alleles preserved)
- First iteration: maximum protection, subsequent iterations: standard

**ThreadCompass** (Ultra-Long Read Routing):
- GNN-based path optimization for ultra-long reads
- Long-range connection validation
- Multi-start pathfinding with confidence scoring
- Iterative refinement with topology feedback

**HaplotypeDetangler** (Hi-C-Augmented Phasing):
- Spectral clustering with Hi-C contact matrices
- Chromosome-scale haplotype separation
- Iterative phasing with assembly refinement
- Backpropagation of phasing context to earlier stages

**SVScribe** (Structural Variant Detection):
- Assembly-time SV calling (DEL, INS, INV, DUP, TRA)
- Graph topology signature analysis
- Ultra-long read spanning evidence
- Hi-C long-range validation

### User-Configurable Training Data Generation

StrandWeaver includes a complete pipeline for generating custom training data, building labelled assembly graphs, and training the five graph-related ML models used by the assembler. See the [user_training module documentation](strandweaver/user_training/README.md) for full details.

#### Step 1 — Generate Synthetic Genomes & Reads

```bash
# Full mode (writes FASTQ + graph CSVs)
strandweaver train generate-data \
  --genome-size 5000000 \
  -n 100 \
  --read-types hifi --read-types ont --read-types ultra_long --read-types hic \
  --coverage 30 --coverage 20 --coverage 10 --coverage 15 \
  --graph-training \
  -o training_data/custom

# ⚡ Fast graph-only mode (~3.3× faster, ~27× less disk)
strandweaver train generate-data \
  --genome-size 1000000 \
  -n 200 \
  --graph-training --graph-only \
  -o training_data/fast_graphs
```

The `--graph-training` flag enables graph synthesis: for every simulated genome, StrandWeaver builds an overlap graph from the reads, labels every edge and node with ground-truth haplotype/topology information, and exports feature CSVs for all five model types (EdgeAI, PathGNN, DiploidAI, UL Routing, SV Detection).

The `--graph-only` flag simulates reads in-memory without writing FASTQ/FASTA to disk, producing only the graph training CSVs (~7 min/genome vs ~23 min/genome).

#### Step 2 — Train Models

```bash
# Train all 5 model types with cross-validation
strandweaver train run \
  --data-dir training_data/custom \
  -o trained_models/

# Train specific models with custom hyperparameters
strandweaver train run \
  --data-dir training_data/custom \
  -o trained_models/ \
  --models edge_ai --models diploid_ai \
  --max-depth 8 --n-folds 3
```

This discovers the CSV files under every `genome_*/graph_training/` subdirectory, trains XGBoost classifiers/regressors for each model type with cross-validation, and saves weights in the exact directory layout the pipeline expects. Hybrid resampling activates automatically when class imbalance exceeds 5.0×.

#### Step 3 — Assemble with Trained Models

```bash
# Pre-trained models load automatically from trained_models/
strandweaver pipeline \
  --hifi-long-reads reads.fastq.gz \
  -o assembly/

# Or specify a custom model directory
strandweaver pipeline \
  --hifi-long-reads reads.fastq.gz \
  --model-dir my_custom_models/ \
  -o assembly/
```

**Configurable Parameters:**
- Genome characteristics: size, GC content, repeat density, ploidy
- Variation: SNP rate, indel rate, SV density and types
- Sequencing: read types (Illumina, HiFi, ONT, Ultra-long, Hi-C, Ancient DNA), coverage, error rates
- Graph: overlap length/identity thresholds, noise edge fraction, GFA export

See [strandweaver/user_training/README.md](strandweaver/user_training/README.md) for the complete parameter reference, graph training architecture, and advanced recipes. For Colab GPU training workflows, see [trained_models/TRAINING.md](trained_models/TRAINING.md).

**Output Files**:
```
output/
├── contigs.fasta                  # Primary assembly contigs
├── final_assembly.fasta           # Polished, length-filtered contigs
├── scaffolds.fasta                # Hi-C scaffolded sequences
├── assembly_graph.gfa             # Assembly graph (GFA format)
├── assembly_stats.json            # N50, L50, coverage statistics
├── misassembly_report.tsv         # Putative misassemblies (tab-delimited)
├── misassembly_report.bed         # Misassemblies (genome browser BED)
├── chromosome_classification.json # Scaffold → chromosome classification
├── sv_calls.vcf                   # Structural variant calls
├── phasing_info.json              # Haplotype assignments
├── coverage_long.csv              # Long read coverage (BandageNG)
├── coverage_ul.csv                # Ultra-long coverage (BandageNG)
├── coverage_hic.csv               # Hi-C support (BandageNG)
├── kmer_predictions.json          # K-mer optimization results
├── error_profile_<tech>_<n>.json  # Per-technology error profiles
└── pipeline.log                   # Complete execution log
```

---

## � Troubleshooting

### Installation Issues

**Problem: `ModuleNotFoundError` or import errors**
```bash
# Solution: Reinstall with all dependencies
pip install --force-reinstall "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"
```

**Problem: Python version incompatibility**
```bash
# Check Python version (requires 3.9+)
python3 --version

# Create conda environment with correct Python version
conda create -n strandweaver python=3.11
conda activate strandweaver
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"
```

**Problem: PyTorch/GPU issues**
```bash
# For CUDA support, install PyTorch separately first
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"

# Verify GPU detection
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

### Assembly Quality Issues

**Problem: Low N50 or fragmented assembly**
- **Check error rate & coverage**: Aim for 30×+ HiFi or 50×+ ONT
  ```bash
  strandweaver profile -i reads.fastq -o profile.json
  ```
- **Add ultra-long reads, or a subset of your LONGEST READS**: Dramatically improves contiguity
  ```bash
  strandweaver pipeline --hifi-long-reads hifi.fastq --ont-ul ultralong.fastq -o improved/
  ```
- **Enable Hi-C scaffolding**: For chromosome-scale assemblies
  ```bash
  strandweaver pipeline --hifi-long-reads hifi.fastq --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq -o scaffolded/
  ```

**Problem: Collapsed heterozygous regions**
```bash
# Use diploid mode with strict edge filtering to preserve variation
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  --ploidy diploid \
  --edge-filter-mode strict \
  -o diploid_assembly/
```

**Problem: Assembly produces too many contigs (over-fragmented)**
- **Reduce edge filtering stringency**:
  ```bash
  strandweaver pipeline --hifi-long-reads reads.fastq --edge-filter-mode lenient -o assembly/
  ```
- **Increase k-mer size**: For high-coverage, low-error data
  ```bash
  strandweaver pipeline --hifi-long-reads reads.fastq --kmer-size-assembly 51 -o assembly/
  ```

**Problem: Assembly is chimeric or has misassemblies**
- **Enable stricter filtering**:
  ```bash
  strandweaver pipeline --hifi-long-reads reads.fastq --edge-filter-mode strict -o assembly/
  ```
- **Add Hi-C validation**: Long-range contact validation prevents chimeras
  ```bash
  strandweaver pipeline --hifi-long-reads hifi.fastq --hic-r1 hic_R1.fastq --hic-r2 hic_R2.fastq -o validated/
  ```

### Performance & Resource Issues

**Problem: Out of memory (OOM) errors**
```bash
# Limit memory usage
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --memory-limit 16

# Or reduce graph coverage via sampling
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --sample-size-graph 500000
```

**Problem: Assembly is too slow**
```bash
# Increase threads (use all available cores)
strandweaver pipeline --hifi-long-reads reads.fastq -t $(nproc) -o assembly/

# Disable AI features for faster heuristic-only assembly
strandweaver pipeline --hifi-long-reads reads.fastq --classical -o fast_assembly/

# Skip profiling step if reads are already well-characterized
strandweaver pipeline --hifi-long-reads reads.fastq --skip-profiling -o quick_assembly/
```

**Problem: Disk space issues**
```bash
# Export only FASTA (skip GFA graphs to save space)
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --output-format fasta

# Use a separate output directory on a larger drive
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o /mnt/large_drive/assembly/
```

### Input Data Issues

**Problem: "Unsupported file format" error**
```bash
# StrandWeaver accepts: FASTQ, FASTA, gzipped variants
# Check file format
file reads.fastq

# Convert BAM to FASTQ if needed
samtools bam2fq reads.bam > reads.fastq

# Decompress if needed
gunzip -c reads.fastq.gz > reads.fastq
```

**Problem: Technology auto-detection fails**
```bash
# Manually specify read technology
strandweaver pipeline \
  -r1 reads.fastq --technology1 ont \
  -o assembly/

# Supported: illumina, ancient, ont, ont_ultralong, pacbio
```

**Problem: Ancient DNA damage not detected**
```bash
# Explicitly specify ancient DNA technology
strandweaver pipeline \
  -r1 ancient_reads.fastq --technology1 ancient \
  -o ancient_assembly/

# Check damage profile first
strandweaver profile -i ancient_reads.fastq --technology ancient -o damage_profile.json
```

### AI/ML Issues

**Problem: AI features not working**
```bash
# Check if AI dependencies installed
python3 -c "import torch, xgboost; print('AI dependencies OK')"

# Install AI dependencies if missing
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[ai]"
```

**Problem: "No trained models found" warning**
```bash
# Pre-trained models ship with v0.2+ and should load automatically.
# If you see this warning, the trained_models/ directory may be missing.

# Reinstall to restore pre-trained models:
pip install --force-reinstall "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"

# Or train custom models:
strandweaver train generate-data --genome-size 5000000 --graph-training --graph-only -o training_data/
strandweaver train run --data-dir training_data/ -o trained_models/
# See trained_models/TRAINING.md for the complete training guide
```

**Problem: GPU not being used**
```bash
# Force GPU usage with explicit backend
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --gpu-backend cuda

# On Apple Silicon
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --gpu-backend mps

# Check GPU memory usage during assembly
watch -n 1 nvidia-smi
```

### Output Issues

**Problem: No structural variants detected**
```bash
# Ensure SV detection is happening (enabled by default in the pipeline)
# SVs require ultra-long or Hi-C data for validation
strandweaver pipeline \
  --hifi-long-reads hifi.fastq \
  --ont-ul ultralong.fastq \
  --min-sv-size 30 \
  -o assembly/
```

**Problem: Missing output files**
```bash
# Check pipeline.log for errors
tail -100 output/pipeline.log

# Export all output formats (FASTA + GFA) with intermediate graphs
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --output-format both \
  --export-intermediate-graphs
```

**Problem: GFA file won't load in BandageNG**
```bash
# Validate GFA format
grep "^S" assembly_graph.gfa | head -5

# Regenerate with both FASTA + GFA export
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --output-format both
```

### Common Error Messages

**`ValueError: Coverage too low for reliable assembly`**
- Solution: Increase sequencing coverage (aim for 30×+ minimum)

**`RuntimeError: Graph construction failed - no valid k-mer overlaps`**
- Solution: Try different k-mer size with `--kmer-size-assembly 31` or `--kmer-size-assembly 51`

**`MemoryError: Unable to allocate array`**
- Solution: Use `--memory-limit 16` to cap memory or reduce coverage with `--sample-size-graph 500000`

**`ImportError: cannot import name 'ThreadCompass'`**
- Solution: Reinstall package with `pip install --force-reinstall strandweaver`

**`FileNotFoundError: [Errno 2] No such file or directory`**
- Solution: Use absolute paths for input/output files or check current working directory

### Getting Help

**Check version and installation**:
```bash
strandweaver --version
strandweaver --help
```

**Enable verbose logging**:
```bash
strandweaver pipeline \
  --hifi-long-reads reads.fastq \
  -o assembly/ \
  --log-level DEBUG
```

**Report issues**: [GitHub Issues](https://github.com/pgrady1322/strandweaver/issues)

**Contact**: patrickgsgrady@gmail.com

---

## 🗺️ Roadmap & Future Enhancements

Planned features and known integration gaps for upcoming releases.

### v0.2 Release Notes

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | **Trained ML Models** | ✅ Complete | XGBoost + GNN weights for 5 of 7 AI modules (EdgeWarden, PathGNN, DiploidAI, ThreadCompass, SVScribe). Trained on 200 synthetic diploid genomes via Colab GPU sweeps with hybrid resampling. |
| 2 | **DiploidAI Integration** | ✅ Complete | 26-feature XGBoost haplotype phasing wired into HaplotypeDetangler at all 3 pipeline call sites |
| 3 | **Bubble-Aware Local Phasing** | ✅ Complete | Genomics audit item G2 — local phasing with allelic bubble detection |
| 4 | **Genomics Audit (24 items)** | ✅ Complete | G1–G24 all resolved — critical, high, and moderate priority fixes |
| 5 | **Git LFS** | ✅ Complete | Model weights (`.pkl`, `.pt`) tracked via Git LFS |
| 6 | **Graph-Only Training Mode** | ✅ Complete | `--graph-only` flag: 3.3× faster data generation, 27× less disk |
| 7 | **Training Documentation** | ✅ Complete | [AI Model Training Guide](trained_models/TRAINING.md) with performance benchmarks |

### v0.3 — Planned

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | **K-Weaver & ErrorSmith Trained Models** | Planned | Requires assembly benchmark approach (not synthetic graph training) |
| 2 | **PacBio Flowcell / Chemistry Metadata** | Almost Complete | `--pacbio-platform`, `--pacbio-chemistry` flags for the `profile` command |
| 3 | **Native ONT Metadata Detection** | Almost Complete | `--ont-detect` flag is wired but calls a placeholder. Replace with POD5 / FAST5 / Dorado header parsing |
| 4 | **Standalone `assemble` / `nf-build-contigs`** | Stub only | Wire to the same assembly engine used by the `pipeline` command |
| 5 | **Technology-Specific Read Subsampling** | ✅ Complete | `--subsample-hifi`, `--subsample-ont`, `--subsample-ont-ul`, `--subsample-illumina`, `--subsample-ancient` on `pipeline` command. `nf-merge` removed (redundant). |
| 6 | **QV Score Optimization & Gap Filling** | ✅ Complete | `qv`, `polish`, `gap-fill` CLI commands + `QVEstimator`, `IterativePolisher`, `GapFiller` modules, fully wired into `_step_finish()` |
| 7 | **Validate — Reference Comparison** | Planned | `--reference` flag accepted but comparison logic not implemented |
| 8 | **BUSCO Integration** | Planned | `--busco-lineage` present on `validate` but not wired to BUSCO |
| 9 | **Decontamination Screening** | Planned | `--decontaminate` present on `pipeline` but no screening step implemented |
| 10 | **Polyploid Assembly** | Planned | `--ploidy` currently limited to `haploid` / `diploid`; polyploid mode is a future target |

---

## 📚 Documentation

**Available Documentation:**
- [AI Model Training Guide](trained_models/TRAINING.md) - Pre-trained model details, performance benchmarks, custom training & retraining
- [User Training Module](strandweaver/user_training/README.md) - Synthetic genome generation & graph training data pipeline

---

## 📄 License

See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md. Please note that this program is restricted for commercial use (or an industry-funded academic project) without a license granted by the developer.

**Licensing FAQ**

***What license does this software use?***

This software is released under a dual-license model:
	1.	A Noncommercial Academic License (default)
	2.	A Commercial License for industry use or commercially sponsored academic work

See the files LICENSE and COMMERCIAL_LICENSE.md for full terms.

***Who can use the software for free?***

Anyone at a nonprofit academic or research institution may use the software for noncommercial academic research.

This includes:
	•	undergraduate and graduate students
	•	postdocs
	•	faculty
	•	research staff
	•	nonprofit research institutes

As long as the project is not funded by industry and is not intended for commercial outcomes.

***What counts as commercial use?***

Commercial use includes, but is not limited to:
	•	use by any for-profit company, startup, or commercial lab
	•	use in paid consulting, contract work, or fee-for-service analysis
	•	integration into commercial pipelines, workflows, or products
	•	research that is industry-funded, even if conducted at a university
	•	research intended for commercial application, technology transfer, or IP generation

If your work will generate revenue, support a commercial product, or is funded by industry, you need a commercial license.

***Can academic users with industry funding use the software for free?***

No.
If a project receives any industry funding or sponsorship, even partially, it is considered commercial use.
A commercial license is required.

***Can I use the software for benchmarking or internal R&D at a company?***

Not under the default license.

Companies must obtain a commercial license for:
	•	benchmarking
	•	internal research
	•	prototyping
	•	feasibility studies
	•	integration testing

***Can I modify the software?***

	•	Yes, for noncommercial academic use.
	•	Yes, for commercial use if covered by a commercial license.

Redistribution or sublicensing is restricted.

***Can I distribute a modified version?***

Academic users may distribute modified versions only within academic, noncommercial contexts.

Any distribution tied to commercial use requires separate licensed permission.

***How do I get a commercial license?***

Contact the developer:

patrickgsgrady(at)gmail.com

Licenses are tailored to each use case.

***Why use a dual-license model?***

This model allows:
	•	free access for researchers
	•	sustainable support and development
	•	protection against unlicensed commercial exploitation
	•	flexibility for industry users who need broader rights

***Is the software open source?***

It is source-available, but not open-source under OSI definitions because:
	•	it restricts commercial use
	•	it restricts redistribution under some conditions

This model balances openness with responsible and sustainable use.

---

## 📧 Contact

**Patrick Grady** | dr.pgrady(at)gmail.com

---

## 📈 Citation and References

```bibtex
@software{strandweaver2026,
  author = {Grady, Patrick; Green, Rich},
  title = {StrandWeaver: AI-Powered Multi-Technology Genome Assembler},
  year = {2026},
  url = {https://github.com/pgrady1322/strandweaver}
}
```

1) Zimin AV, Puiu D, Luo MC, Zhu T, Koren S, Yorke JA, Dvorak J, Salzberg S. Hybrid assembly of the large and highly repetitive genome of Aegilops tauschii, a progenitor of bread wheat, with the mega-reads algorithm. Genome Research. 2017 Jan 1:066100.
2) Rautiainen, M., Nurk, S., Walenz, B.P. et al. Telomere-to-telomere assembly of diploid chromosomes with Verkko. Nat Biotechnol 41, 1474–1482 (2023). https://doi.org/10.1038/s41587-023-01662-6
3) Cheng, H., Concepcion, G.T., Feng, X. et al. Haplotype-resolved de novo assembly using phased assembly graphs with hifiasm. Nat Methods 18, 170–175 (2021). https://doi.org/10.1038/s41592-020-01056-5
---

**StrandWeaver** 🧬⚡🤖
