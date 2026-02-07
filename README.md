# StrandWeaver

**AI-Powered Multi-Technology Genome Assembler with GPU Acceleration**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-v0.1_Beta-yellow.svg)](docs/MASTER_DEVELOPMENT_ROADMAP.md)
[![License](https://img.shields.io/badge/license-Dual%20License%20(Academic/Commercial)-blue.svg)](LICENSE_ACADEMIC.md)

**StrandWeaver** is a next-generation genome assembly pipeline combining machine-learning optimized technology-aware error correction, graph-based assembly with haplotype-aware path resolution, and comprehensive structural variant detection for ancient DNA, Illumina, ONT, ultra-long ONT, and PacBio sequencing data. Its goal is to relieve manual curation bottlenecks in traditional high-contiguity genome assembly by applying AI/ML to genome graph paths and other complex regions. It uses these technologies to improve accuracy and contiguity, but also to provide functional annotations, such as structural variants, during the assembly process.

StrandWeaver is inspired by MaSuRCA, Verkko, and Hifiasm, but includes novel elements such as:
- **Neural network-based haplotype assembly**: Graph topology simplification with strict protection of biological variation (SNPs, indels, CNVs) across diploid genomes
- **Multi-technology integration**: Seamless combination of ONT, PacBio HiFi, ultra-long reads, and Hi-C data in a unified assembly graph
- **Dynamic K-mer size selection**: StrandWeaver chooses different k-mer sizes for different parts of the assembly process based on rule-trained models
- **Comprehensive Path Scoring**: A path score for each graph path is provided based on a comprehensive set of rules, including coverage, error profiles, and sequence repetitive complexity
- **Ancient DNA optimization**: Machine learning models trained to profile and repair deamination damage (C→T/G→A patterns) with configurable confidence thresholds

The pipeline can be custom trained using provided scripts for any data type (new sequencing technology) or organism-specific scenario (genomes with extreme repeat content, high heterozygosity, or complex structural variation). **Generic models will be provided with the v0.2 release.**

> **🚀 v0.1.0 Release (February 2026):** Beta release with complete end-to-end assembly pipeline. All AI modules functional using optimized heuristics. Core pipeline does not currently include trained models. GPU acceleration fully operational. See [v0.1 Release Notes](#v01-release-notes) below.
>
> **v0.1.1:** Expanded user documentation and additional scripts for model training. Expected March 2026.
>
> **v0.2+:** Trained ML models released (after extensive testing on heuristic versions and model training). Architecture for rebuilding genome after custom graph modifications released.

---

## ✨ Key Features

### Core Assembly
- 🧬 **Multi-Technology Support**: Ancient DNA, Illumina, ONT R9/R10, PacBio HiFi/CLR, Ultra-long reads
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
- 📊 **Structural Variant Detection**: Assembly-time SV calling with graph topology analysis

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
- 🧪 **Training Infrastructure**: Generic models provided for v0.2+, and custom training can be done for organism-specific optimization
- 🔌 **Modular Architecture**: All AI features can be disabled for classical heuristics

### AI/ML Features (v0.1 - Heuristic-Based)
- **Complete AI Subsystem Suite** (using optimized heuristics - trained models in v0.2+):
  1. ✅ **K-Weaver**: K-mer optimization with rule-based selection
  2. ✅ **ErrorSmith**: Technology-specific error profiling
  3. ✅ **EdgeWarden**: 80-feature edge filtering with heuristic scoring
  4. ✅ **PathWeaver**: Haplotype-aware path resolution with variation protection
  5. ✅ **ThreadCompass**: Ultra-long read routing optimization
  6. ✅ **HaplotypeDetangler**: Hi-C-augmented phasing with spectral clustering
  7. ✅ **SVScribe**: Assembly-time structural variant detection
- **Training Infrastructure**: Complete synthetic data generation pipeline for custom model training
- **Classical Fallbacks**: All AI modules functional with heuristic defaults
- **Planned**: Trained ML models (XGBoost, PyTorch) for v0.2 release

---

## 📋 Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [AI/ML Training](#aiml-training)
- [Documentation](#documentation)

**📚 Full Documentation:**
- [Features Guide](docs/FEATURES.md) - Complete feature documentation

**Coming soon (v0.1.1)**:
- Complete tutorials and usage examples
- Benchmarks and performance comparisons

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
- Custom model training
- GPU-accelerated assembly (if CUDA available)
- Advanced AI features in v0.2+ (current v0.1 uses heuristics)

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

# With AI/training dependencies (PyTorch, XGBoost)
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[ai]"

# Complete installation with all dependencies
pip install "git+https://github.com/pgrady1322/strandweaver.git#egg=strandweaver[all]"
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
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --output contigs.fasta \
  --threads 8
```

**Hybrid Assembly with Multiple Ultra-Long Read Types**

*Note:* The --ont-ul flag is used for path-finding reads. Any platform of long reads can be provided, but shorter long reads will degrade the assembly. The ont-ul name is retained for clarity / comparison with other assemblers.
```bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --ont ont_reads.fastq.gz \
  --ont-ul ultralong_reads.fastq \ 
  --output assembly.fasta \
  --threads 16
```

**Mixed Technology Assembly with Hi-C**

*Note:* ANY platform of proximity ligation tech can be provided. StrandWeaver will optimize for Hi-C and Omni-C just as well as Pore-C and CiFi.
```bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --ont-ul ultralong_reads.fastq \
  --hic hic_R1.fastq hic_R2.fastq \
  --output assembly.fasta \
  --threads 16
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

StrandWeaver provides standalone commands for each processing stage. Each command supports both direct and Nextflow execution.

#### Error Correction
```bash
# Direct mode
strandweaver correct --hifi reads.fq.gz -o corrected/ -t 8

# Nextflow mode (automatic parallelization)
strandweaver correct --hifi reads.fq.gz -o corrected/ \
  --nextflow --nf-profile slurm --correction-batch-size 100000
```

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
strandweaver score-edges -e edges.json -a aligns.bam -o scored.json -t 8

# Nextflow mode (large graphs)
strandweaver score-edges -e edges.json -a aligns.bam -o scored.json \
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
strandweaver detect-svs -g graph.gfa -o variants.vcf -t 8

# Nextflow mode (large graphs)
strandweaver detect-svs -g graph.gfa -o variants.vcf \
  --nextflow --nf-profile slurm --sv-batch-size 1000
```

### Performance Guidelines

**When to use Direct mode:** Dataset < 10GB, local workstation with 8+ cores, testing and debugging.

**When to use Nextflow mode:** Dataset > 10GB, HPC cluster available, need resume capability, want automatic parallelization.

| Command | Direct (1 node) | Nextflow (20 nodes) | Speedup |
|---------|----------------|---------------------|--------|
| `correct` | 20 hours | 2 hours | 10× |
| `extract-kmers` | 8 hours | 1.5 hours | 5× |
| `score-edges` | 8 hours | 1.5 hours | 5× |
| `map-ul` | 6 hours | 1 hour | 6× |
| `align-hic` | 10 hours | 1.5 hours | 7× |
| `detect-svs` | 4 hours | 1 hour | 4× |

---

## 🎯 Use Cases

### Machine-Learning-Tuned Genome Assembly with SV Calls
Combine ONT, HiFi, ultra-long reads, and Hi-C for chromosome-scale phased assemblies:
```bash
strandweaver assemble \
--ont ont.fastq \
--hifi hifi.fastq \
--ont-ul ultralong.fastq \
--hic hic_R1.fastq hic_R2.fastq \
--output genome.fasta \
--ai-enabled \
--detect-svs \
--threads 32
```

### Ancient DNA Assembly
Optimize for deamination damage with specialized error correction:
```bash
strandweaver assemble \
--ancient-dna ancient_reads.fastq \
--output ancient_genome.fasta \
--damage-aware \
--threads 16
```
Note that the assembly can also be run WITHOUT damage awareness features for comparison.
### SV-Rich Genome Analysis
Detect structural variants during assembly for cancer or population genomics:
```bash
strandweaver assemble \
--hifi tumor.fastq \
--hic hic_R1.fastq hic_R2.fastq \
--output tumor_assembly.fasta \
--detect-svs \
--sv-mode sensitive \
--threads 24
```

### Highly Heterozygous Diploid Assembly
Maintain haplotype separation for F1 hybrids or outcrossing species:
```bash
strandweaver assemble \
--hifi hifi.fastq \
--hic hic_R1.fastq hic_R2.fastq \
--output diploid.fasta \
--preserve-heterozygosity \
--min-identity 0.995 \
--threads 32
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
16. **Chromosome Classification**: Multi-tier scaffold classification (gene content, telomere detection, Hi-C self-contact patterns) to identify chromosomes vs. assembly artifacts.

### Output Generation
17. **GFA Export**: Assembly graphs in GFA format with sequences
18. **BandageNG**: Visualization files with coverage tracks and final 0 - 1 range StrandWeaver scores (long/UL/Hi-C).
19. **Statistics**: N50, L50, coverage metrics, variation protection counts
20. **SV Calls**: Structural variants in VCF and JSON formats
21. **Phasing Info**: Haplotype assignments and confidence scores

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

StrandWeaver includes a flexible system for generating custom training data. See the [user_training module documentation](strandweaver/user_training/README.md) for details.

**Quick Example:**
```bash
# Generate custom training data with specific parameters
python -m strandweaver.user_training.generate_training_data \
  --genome-size 5000000 \
  --num-genomes 100 \
  --read-types hifi ont ultra_long hic \
  --coverage 30 20 10 15 \
  --output training_data/custom
```

**Configurable Parameters:**
- Genome characteristics: size, GC content, repeat density, ploidy
- Variation: SNP rate, indel rate, SV density and types
- Sequencing: read types (Illumina, HiFi, ONT, Ultra-long, Hi-C, Ancient DNA), coverage, error rates

See [strandweaver/user_training/README.md](strandweaver/user_training/README.md) for complete documentation on generating and using custom training data.

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
├── error_profile.json             # Error profiling results
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
- **Check coverage**: Aim for 30×+ HiFi or 50×+ ONT
  ```bash
  strandweaver profile --input reads.fastq --output profile.json
  ```
- **Add ultra-long reads**: Dramatically improves contiguity
  ```bash
  strandweaver assemble --hifi hifi.fastq --ont-ul ultralong.fastq --output improved.fasta
  ```
- **Enable Hi-C scaffolding**: For chromosome-scale assemblies
  ```bash
  strandweaver assemble --hifi hifi.fastq --hic hic_R1.fastq hic_R2.fastq --output scaffolded.fasta
  ```

**Problem: Collapsed heterozygous regions**
```bash
# Increase identity threshold to preserve variation
strandweaver assemble \
  --input reads.fastq \
  --preserve-heterozygosity \
  --min-identity 0.995 \
  --output diploid.fasta
```

**Problem: Assembly produces too many contigs (over-fragmented)**
- **Reduce edge filtering stringency**:
  ```bash
  strandweaver assemble --input reads.fastq --edge-filter-mode permissive --output assembly.fasta
  ```
- **Increase k-mer size**: For high-coverage, low-error data
  ```bash
  strandweaver assemble --input reads.fastq --kmer-size 51 --output assembly.fasta
  ```

**Problem: Assembly is chimeric or has misassemblies**
- **Enable stricter filtering**:
  ```bash
  strandweaver assemble --input reads.fastq --edge-filter-mode strict --output assembly.fasta
  ```
- **Add Hi-C validation**: Long-range contact validation prevents chimeras
  ```bash
  strandweaver assemble --hifi hifi.fastq --hic hic_R1.fastq hic_R2.fastq --output validated.fasta
  ```

### Performance & Resource Issues

**Problem: Out of memory (OOM) errors**
```bash
# Reduce memory usage with streaming mode
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --streaming \
  --max-memory 16G

# Or process in batches
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --batch-size 100000
```

**Problem: Assembly is too slow**
```bash
# Increase threads (use all available cores)
strandweaver assemble --input reads.fastq --threads $(nproc) --output assembly.fasta

# Disable AI features for faster heuristic-only assembly
strandweaver assemble --input reads.fastq --no-ai --output fast_assembly.fasta

# Use rapid mode (skips iterative refinement)
strandweaver assemble --input reads.fastq --rapid --output quick_assembly.fasta
```

**Problem: Disk space issues**
```bash
# Enable cleanup of intermediate files
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --cleanup-intermediate

# Specify temporary directory on larger drive
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --temp-dir /mnt/large_drive/tmp
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
strandweaver assemble \
  --input reads.fastq \
  --technology ont \
  --output assembly.fasta

# Supported: illumina, ont, hifi, ultralong, ancient
```

**Problem: Ancient DNA damage not detected**
```bash
# Explicitly enable ancient DNA mode
strandweaver assemble \
  --ancient-dna ancient_reads.fastq \
  --damage-aware \
  --output ancient_assembly.fasta

# Check damage profile first
strandweaver profile --ancient-dna ancient_reads.fastq --output damage_profile.json
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
# v0.1 uses optimized heuristics (no models needed)
# This is expected behavior - assembly will complete successfully

# For v0.2+: Download pre-trained models
strandweaver download-models --destination ~/.strandweaver/models

# Or train custom models (advanced)
python3 -m strandweaver.user_training.generate_training_data --output training_data/
# See strandweaver/user_training/README.md for details
```

**Problem: GPU not being used**
```bash
# Force GPU usage
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --device cuda

# Check GPU memory usage during assembly
watch -n 1 nvidia-smi
```

### Output Issues

**Problem: No structural variants detected**
```bash
# Ensure SV detection is enabled
strandweaver assemble \
  --input reads.fastq \
  --detect-svs \
  --sv-mode sensitive \
  --output assembly.fasta

# SVs require ultra-long or Hi-C data for validation
strandweaver assemble \
  --hifi hifi.fastq \
  --ont-ul ultralong.fastq \
  --detect-svs \
  --output assembly.fasta
```

**Problem: Missing output files**
```bash
# Check pipeline.log for errors
tail -100 output/pipeline.log

# Enable all output formats
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --export-gfa \
  --export-bandage \
  --export-stats \
  --detect-svs
```

**Problem: GFA file won't load in BandageNG**
```bash
# Validate GFA format
grep "^S" assembly_graph.gfa | head -5

# Regenerate with sequence export
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --export-gfa \
  --gfa-include-sequences
```

### Common Error Messages

**`ValueError: Coverage too low for reliable assembly`**
- Solution: Increase sequencing coverage (aim for 30×+ minimum) or use `--min-coverage 10` to override

**`RuntimeError: Graph construction failed - no valid k-mer overlaps`**
- Solution: Try different k-mer size with `--kmer-size 31` or `--kmer-size 51`

**`MemoryError: Unable to allocate array`**
- Solution: Enable streaming mode with `--streaming` or reduce batch size with `--batch-size 50000`

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
strandweaver assemble \
  --input reads.fastq \
  --output assembly.fasta \
  --verbose \
  --log-level DEBUG
```

**Generate diagnostic report**:
```bash
strandweaver diagnose --output diagnostic_report.txt
```

**Report issues**: [GitHub Issues](https://github.com/pgrady1322/strandweaver/issues)

**Contact**: patrickgsgrady@gmail.com

---

## �📚 Documentation

**Available Documentation:**
- [Features Guide](docs/FEATURES.md) - Complete feature documentation
- [User Training Guide](strandweaver/user_training/README.md) - Generate custom training data

**Coming in v0.1.1:**
- Complete user tutorials and examples
- Training guide for custom models
- Performance benchmarks and comparisons to other assemblers

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

## 📈 Citation

```bibtex
@software{strandweaver2026,
  author = {Grady, Patrick; Green, Rich},
  title = {StrandWeaver: AI-Powered Multi-Technology Genome Assembler},
  year = {2026},
  url = {https://github.com/pgrady1322/strandweaver}
}
```

---

**StrandWeaver** 🧬⚡🤖
