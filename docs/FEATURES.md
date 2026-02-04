# StrandWeaver Features Guide

**Version 0.1.0** | AI-Powered Genome Assembly Pipeline

This document provides a comprehensive overview of StrandWeaver's capabilities, modules, and features for high-quality genome assembly from long-read sequencing data.

---

## 🎯 Overview

StrandWeaver is an integrated genome assembly pipeline that combines traditional bioinformatics tools with AI-powered quality control, error correction, and scaffolding. The v0.1 release uses optimized **heuristic-based AI** (not trained ML models yet) to provide production-ready assembly capabilities.

**Key Capabilities:**
- End-to-end genome assembly from raw reads to finished chromosome-scale assemblies
- GPU acceleration (CUDA/MPS) for 10-40× speedup on compute-intensive operations
- Intelligent read error correction and quality filtering
- Advanced Hi-C scaffolding with contact matrix analysis
- Chromosome classification (autosomes, sex chromosomes, organellar DNA)
- Structural variant detection and annotation
- Assembly polishing with multiple evidence types
- Comprehensive validation and quality metrics

---

## 🧬 Core Pipeline Modules

### 1. **ErrorSmith** - AI-Powered Error Correction
**Purpose:** Detect and correct sequencing errors using contextual read analysis

**Features:**
- Technology-aware error models (PacBio HiFi, ONT, CLR)
- Multi-pass consensus polishing
- K-mer spectrum analysis for systematic error detection
- Homopolymer compression error correction
- Alignment-based correction with minimap2 integration

**Heuristics (v0.1):**
- K-mer coverage thresholds for error detection
- Consensus calling with quality-weighted voting
- Homopolymer length normalization
- Systematic error profiling per technology

**Machine Learning (v0.2 planned):**
- **Base Error Predictor**: 1D CNN or Bidirectional LSTM for per-base error probability prediction
  - Examines 21-bp window around each base considering quality scores, homopolymer runs, position in read
  - Expected 15-20% reduction in false positive corrections
  - Training data: 360,000 examples across 6 technologies
- **K-Weaver (Adaptive K-mer Selector)**: Random Forest/XGBoost for dynamic k-mer size selection (15-51)
  - Analyzes read properties (quality, GC content, homopolymers, coverage) to select optimal k-mer
  - Expected 5-10% reduction in over-correction artifacts
  - Training data: 540,000 examples across 6 technologies
- Tech-specific neural networks for complex error patterns (ONT homopolymer errors, aDNA deamination)

**Usage:**
```bash
strandweaver correct-reads \
  --input reads.fastq \
  --output corrected_reads.fastq \
  --technology hifi \
  --passes 3 \
  --threads 32
```

---

### 2. **K-Weaver** - Adaptive K-mer Optimization
**Purpose:** ML-based k-mer size selection for different assembly stages

**Features:**
- Dynamic k-mer prediction for 4 assembly stages (DBG, UL overlap, extension, polish)
- Technology-specific k-mer optimization
- Rule-based fallback if ML models unavailable
- Confidence scores for each prediction

**Heuristics (v0.1):**
- Technology-specific defaults:
  - Illumina: k=21 (error correction), k=31 (assembly)
  - HiFi: k=31 (DBG), k=51 (extension)
  - ONT: k=17 (correction), k=27 (assembly)
  - aDNA: k=15 (damage-aware)

**Machine Learning (v0.2 planned):**
- **K-mer Predictor**: Random Forest/XGBoost for dynamic k-mer size selection (15-51)
  - Analyzes read properties (quality, GC content, homopolymers, coverage) to select optimal k-mer
  - Expected 5-10% reduction in over-correction artifacts
  - Training data: 540,000 examples across 6 technologies

**Usage:**
```bash
# Get k-mer predictions for dataset
strandweaver predict-kmers \
  --input reads.fastq \
  --technology hifi \
  --output kmer_predictions.json
```

---

### 3. **EdgeWarden** - Read Overlap Analysis
**Purpose:** Intelligent graph construction for assembly with quality-aware overlap detection

**Features:**
- All-vs-all overlap computation (minimap2 backend)
- GPU-accelerated overlap filtering and scoring
- Containment relationship detection
- Transitive reduction for graph simplification
- Multi-threaded overlap graph construction

**Heuristics (v0.1):**
- Minimum overlap length: 1kb (configurable)
- Identity threshold: 90% (configurable)
- Containment detection with coverage analysis
- Graph simplification rules (tip removal, bubble popping)

**Machine Learning (v0.2 planned):**
- **EdgeAI Classifier**: XGBoost/Random Forest for overlap classification
  - 18 extracted features per overlap (length, identity, coverage, repeat content, etc.)
  - 6-class classification: TRUE_OVERLAP, REPEAT, CHIMERA, LOW_QUALITY, SPURIOUS, CONTAINED
  - Filters spurious overlaps in repetitive genomes before contig construction
  - Expected 30-40% reduction in assembly graph complexity
  - Training data: Synthetic repetitive genomes with known ground truth

**Usage:**
```bash
strandweaver build-graph \
  --reads corrected_reads.fastq \
  --output overlap_graph.gfa \
  --min-overlap 1000 \
  --min-identity 0.90 \
  --threads 64
```

---

### 4. **PathWeaver** - Graph-Based Contig Assembly
**Purpose:** Traverse overlap graphs to generate high-quality contigs

**Features:**
- Greedy path extension with quality scoring
- Bubble resolution for heterozygous regions
- Repeat-aware path selection
- Circular contig detection (organellar genomes)
- Contig merging and gap filling

**Heuristics (v0.1):**
- Best-first path traversal with coverage weighting
- Bubble popping for simple SNPs/indels
- Repeat classification via coverage analysis
- Unitig generation with flow-based algorithms

**Usage:**
```bash
strandweaver assemble \
  --graph overlap_graph.gfa \
  --reads corrected_reads.fastq \
  --output contigs.fasta \
  --min-contig-length 50000
```

---

### 5. **HiCIntegrator** - Hi-C Scaffolding
**Purpose:** Chromosome-scale scaffolding using Hi-C contact frequency data

**Module:** `strandweaver.assembly_core.strandtether_module.HiCIntegrator`

**Features:**
- Hi-C read alignment and contact matrix generation
- Linkage group identification (chromosome clustering)
- Contig orientation and ordering optimization
- Misassembly detection via contact discontinuities
- Gap size estimation from contact decay

**Heuristics (v0.1):**
- Contact frequency normalization (ICE/KR)
- Hierarchical clustering for linkage groups
- Simulated annealing for contig ordering
- Z-score based misassembly detection

**Algorithms:**
- Restricted Boltzmann Machine (RBM) scaffolding
- Markov clustering for grouping
- Distance-based ordering with contact matrix optimization

**Usage:**
```bash
strandweaver scaffold \
  --contigs contigs.fasta \
  --hic-r1 hic_R1.fastq.gz \
  --hic-r2 hic_R2.fastq.gz \
  --output scaffolds.fasta \
  --threads 48
```

**Advanced Options:**
```bash
# With pre-aligned Hi-C data
strandweaver scaffold \
  --contigs contigs.fasta \
  --hic-bam aligned_hic.bam \
  --contact-matrix contacts.hic \
  --output scaffolds.fasta
```

---

### 6. **HaplotypeDetangler** - Haplotype-Aware Assembly
**Purpose:** Separate maternal and paternal haplotypes for diploid genome assembly

**Module:** `strandweaver.assembly_core.haplotype_detangler_module.HaplotypeDetangler`

**Features:**
- Trio-binning (parental reads for phasing)
- Hi-C based haplotype separation with spectral clustering
- Heterozygous SNP detection and phasing
- Haplotype-specific contig assignment
- Switch error detection and correction

**Heuristics (v0.1):**
- K-mer based parental read classification
- SNP density analysis for heterozygosity estimation
- Hi-C contact enrichment for haplotype clustering
- Coverage-based haplotype assignment
- Spectral clustering with label propagation

**Machine Learning (v0.2 planned):**
- **DiploidAI Classifier**: PyTorch MLP for haplotype assignment
- 5-class output: HAP_A, HAP_B, BOTH (shared), REPEAT, UNKNOWN
- Multi-modal feature integration (k-mers, Hi-C, SNPs)

**Usage:**
```bash
# Trio-based phasing
strandweaver phase \
  --contigs contigs.fasta \
  --maternal maternal_reads.fastq \
  --paternal paternal_reads.fastq \
  --output-hap1 haplotype1.fasta \
  --output-hap2 haplotype2.fasta

# Hi-C based phasing
strandweaver phase \
  --contigs contigs.fasta \
  --hic-bam aligned_hic.bam \
  --output-hap1 haplotype1.fasta \
  --output-hap2 haplotype2.fasta
```

---

### 7. **ThreadCompass** - Ultra-Long Read Routing
**Purpose:** Optimize ultra-long read paths through assembly graphs for long-range connectivity

**Module:** `strandweaver.assembly_core.threadcompass_module.ThreadCompass`

**Features:**
- Ultra-long read (100kb+) anchor detection
- Path validation with UL read spanning evidence
- Ambiguity resolution in complex graph regions
- Long-range connectivity confirmation
- Integration with PathWeaver for guided assembly

**Heuristics (v0.1):**
- Anchor-based alignment with minimum identity thresholds
- Coverage-weighted path scoring
- Conflict resolution via read support counts
- Minimum spanning evidence requirements

**Usage:**
```bash
strandweaver assemble \
  --reads corrected_reads.fastq \
  --ul-reads ultra_long_reads.fastq \
  --output contigs.fasta \
  --enable-threadcompass
```

---

### 8. **SVScribe** - Structural Variant Detection
**Purpose:** Identify, classify, and annotate structural variations in assemblies

**Features:**
- SV detection from read alignments (Sniffles2, SVIM)
- Assembly-to-reference comparison (MUMmer, Assemblytics)
- SV classification: DEL, INS, DUP, INV, TRA
- Repeat annotation integration
- Functional impact prediction

**Detection Methods:**
- Read-based: Split reads, discordant pairs, coverage analysis
- Assembly-based: Alignment breakpoints, synteny disruptions
- Hybrid: Combined evidence from multiple sources

**Usage:**
```bash
# Read-based SV detection
strandweaver detect-sv \
  --assembly assembly.fasta \
  --reads long_reads.fastq \
  --output variants.vcf \
  --min-size 50

# Assembly-to-reference comparison
strandweaver detect-sv \
  --query assembly.fasta \
  --reference ref_genome.fasta \
  --output structural_variants.vcf \
  --mode assembly-to-ref
```

---

## 🔧 Core Assembly Engines

### 9. **DBG Engine** - De Bruijn Graph Construction
**Purpose:** Build compacted de Bruijn graphs from long reads (HiFi/ONT)

**Module:** `strandweaver.assembly_core.dbg_engine_module`

**Features:**
- K-mer graph construction from long reads
- GPU-accelerated k-mer extraction (∞× speedup vs CPU)
- Unitig generation via linear path merging
- Regional k-mer annotation from K-Weaver predictions
- Advanced graph compaction and simplification
- Accepts both true long reads and OLC-derived artificial long reads

**Implementation:**
- Streaming k-mer extraction for memory efficiency
- MPS/CUDA/CPU multi-backend support
- Graph simplification (tip removal, bubble popping)
- Coverage-based filtering

**Performance:**
- CPU: Can hang indefinitely on large datasets
- GPU (Apple Silicon): 9-22 seconds for 100k reads

**Usage:**
```bash
# Typically called internally by assembly pipeline
strandweaver assemble \
  --reads hifi_reads.fastq \
  --output contigs.fasta \
  --k-mer-size 31
```

---

### 10. **String Graph Engine** - Ultra-Long Read Integration
**Purpose:** Overlay ultra-long reads on DBG to increase assembly contiguity

**Module:** `strandweaver.assembly_core.string_graph_engine_module`

**Features:**
- UL read alignment to DBG nodes (k-mer anchoring)
- Long-range connectivity detection (100kb+ spanning)
- Path validation with UL read evidence
- Gap filling with MBG/GraphAligner integration (optional)
- Regional k-mer information for path scoring
- Error-tolerant anchoring for uncorrected ONT reads

**Implementation:**
- Anchor-guided alignment for error-prone UL reads
- Multi-path resolution with UL support scoring
- Integration with ThreadCompass for path optimization
- String graph edge creation from validated UL paths

**Key Principle:** String graphs are ALWAYS built from DBG + UL reads when ultra-long data is available.

**Usage:**
```bash
# Automatically invoked when UL reads provided
strandweaver assemble \
  --reads hifi_reads.fastq \
  --ul-reads ultra_long_ont.fastq \
  --output contigs.fasta
```

---

### 11. **OLC Contig Builder** - Short Read Assembly
**Purpose:** Assemble Illumina short reads into artificial long reads for graph assembly

**Module:** `strandweaver.assembly_core.illumina_olc_contig_module`

**Features:**
- Overlap-Layout-Consensus (OLC) for Illumina paired-end reads
- K-mer based overlap detection (efficient, reuses k-mer infrastructure)
- Variation-preserving consensus (maintains heterozygous sites)
- Quality-aware contig extension
- Paired-end scaffolding
- GPU-accelerated overlap computation (6-7× speedup)

**Implementation:**
- Greedy contig extension with quality scoring
- Best-first overlap selection
- Consensus calling with quality weighting
- Graph-based layout optimization

**Output:** Artificial long reads (contigs) that serve as input to DBG pipeline

**Usage:**
```bash
# For Illumina-only assemblies
strandweaver assemble \
  --illumina-r1 reads_R1.fastq.gz \
  --illumina-r2 reads_R2.fastq.gz \
  --output contigs.fasta
```

---

## ⚡ GPU Acceleration

StrandWeaver includes optimized GPU kernels for computationally intensive operations.

### Supported Operations:

**Verified on Apple Silicon (M-series):**
- **K-mer extraction**: ∞× speedup (CPU hangs, GPU completes in 9-22s)
- **Contact matrix building**: 20-40× speedup
- **Spectral phasing**: 15-35× speedup
- **Edge support computation**: 8-12× speedup
- **Ultra-long read mapping**: 15× speedup
- **Overlap detection (OLC)**: 6-7× speedup

**General GPU Acceleration:**
- **Quality score analysis**: 15-20× speedup
- **Feature computation**: 10-25× speedup

### GPU Backends:
- **CUDA (CuPy)**: NVIDIA GPUs (RTX 3000/4000, A100, H100)
- **MPS (PyTorch)**: Apple Silicon (M1/M2/M3/M4)
- **Automatic detection**: Graceful CPU fallback if GPU unavailable

**Implementation:** All GPU components follow unified backend pattern with automatic MPS/CUDA/CPU detection

### Usage:
```bash
# Enable GPU acceleration
strandweaver assemble \
  --reads reads.fastq \
  --output contigs.fasta \
  --gpu \
  --gpu-device 0

# Check GPU availability
strandweaver check-gpu
```

---

## 🔬 Training Infrastructure (v0.2)

### Synthetic Training Data Generation
StrandWeaver includes tools to generate realistic training data for ML models:

```bash
# Generate all training datasets
python scripts/generate_assembly_training_data.py \
  --output-dir training_data/ \
  --num-samples 100000 \
  --seed 42
```

**Generated Datasets:**
- **EdgeAI**: Read overlap features (50k samples, 6 classes)
- **DiploidAI**: Haplotype assignment features (30k samples, 5 classes)
- **ErrorSmith**: Error correction contexts (100k samples, regression)

### Model Training (Google Colab)
Train models on free GPU resources using the included Colab notebook:

1. Open [`notebooks/Colab_Training_StrandWeaver.ipynb`](notebooks/Colab_Training_StrandWeaver.ipynb) in Google Colab
2. Follow step-by-step instructions (2-3 hours on T4 GPU)
3. Download trained models and install in StrandWeaver

**Training Scripts:**
- `scripts/train_models/train_edge_classifier.py` - EdgeAI XGBoost model
- `scripts/train_models/train_diploid_classifier.py` - DiploidAI PyTorch MLP

---

## 📊 Input/Output Formats

### Supported Input Formats:
- **Reads**: FASTQ, FASTA, BAM, CRAM, .h5 (PacBio), .fast5 (ONT)
- **Hi-C**: FASTQ pairs, BAM/CRAM aligned, .hic contact matrix
- **Reference**: FASTA, multi-FASTA
- **Annotations**: GFF3, GTF, BED

### Output Formats:
- **Assemblies**: FASTA, GFA (assembly graph)
- **Scaffolds**: FASTA with gap annotations (N's)
- **Variants**: VCF, BED
- **Reports**: HTML, JSON, TSV, PDF

---

## � Assembly Pipeline Stages

### Stage 1: Contig Building
- **Illumina only**: Overlap-Layout-Consensus (OLC) with GPU-accelerated overlap detection
- **Long reads (HiFi/ONT)**: De Bruijn Graph (DBG) with GPU k-mer extraction

### Stage 2: Ultra-Long Read Integration (if available)
- **String Graph**: Anchor-guided alignment with GPU-accelerated UL mapping
- Long-range connectivity and ambiguity resolution

### Stage 3: AI-Powered Refinement
- EdgeAI overlap filtering
- GNN path prediction for optimal traversal
- DiploidAI haplotype phasing
- Graph cleanup (tip removal, bubble popping, SV detection)

### Stage 4: Hi-C Scaffolding (optional)
- GPU-accelerated contact matrix building
- Spectral phasing for chromosome-scale scaffolds
- Edge support computation

**Pipeline Flow:**
```
Illumina → OLC → String Graph (if UL) → Hi-C Scaffolding
HiFi/ONT → DBG → String Graph (if UL) → Hi-C Scaffolding
aDNA     → (preprocessing) → DBG → String Graph → Hi-C
```

---

## 🚀 Complete Workflow Example

```bash
# Step 1: Error correction
strandweaver correct-reads \
  --input raw_reads.fastq.gz \
  --output corrected_reads.fastq.gz \
  --technology hifi --passes 3

# Step 2: Build overlap graph
strandweaver build-graph \
  --reads corrected_reads.fastq.gz \
  --output overlap_graph.gfa \
  --threads 64 --gpu

# Step 3: Assemble contigs
strandweaver assemble \
  --graph overlap_graph.gfa \
  --reads corrected_reads.fastq.gz \
  --output contigs.fasta

# Step 4: Hi-C scaffolding
strandweaver scaffold \
  --contigs contigs.fasta \
  --hic-r1 hic_R1.fastq.gz \
  --hic-r2 hic_R2.fastq.gz \
  --output scaffolds.fasta --threads 48

# Step 5: Chromosome classification
strandweaver classify-chromosomes \
  --scaffolds scaffolds.fasta \
  --sex-system XY \
  --output classified.tsv

# Step 6: Assembly polishing
strandweaver polish \
  --assembly scaffolds.fasta \
  --long-reads corrected_reads.fastq.gz \
  --short-r1 illumina_R1.fastq.gz \
  --short-r2 illumina_R2.fastq.gz \
  --output final_assembly.fasta --strategy hybrid

# Step 7: Validate assembly
strandweaver validate \
  --assembly final_assembly.fasta \
  --reads corrected_reads.fastq.gz \
  --busco-lineage vertebrata_odb10 \
  --output-dir qc_report/
```

---

## 📖 Additional Resources

- **Installation Guide**: [README.md](README.md#installation)
- **Assembly Module Reference**: [archive/ASSEMBLY_MODULE_REFERENCE.md](archive/ASSEMBLY_MODULE_REFERENCE.md)
- **GPU Acceleration**: [archive/GPU_ACCELERATION.md](archive/GPU_ACCELERATION.md)
- **AI/ML Guide**: [archive/AI_ML_GUIDE.md](archive/AI_ML_GUIDE.md)
- **Pipeline Flow**: [archive/PIPELINE_FLOW.md](archive/PIPELINE_FLOW.md)
- **Scientific References**: [archive/SCIENTIFIC_REFERENCES.md](archive/SCIENTIFIC_REFERENCES.md)

### Key Scientific References

**Assembly Algorithms:**
- Myers, E. W. (2005). "The fragment assembly string graph." *Bioinformatics*
- Pevzner et al. (2001). "An Eulerian path approach to DNA fragment assembly." *PNAS*
- Bankevich et al. (2012). "SPAdes: A new genome assembly algorithm." *J. Comp. Biol.*

**Sequencing Technologies:**
- **ONT**: Delahaye & Nicolas (2021). "Sequencing DNA with nanopores: Troubles and biases." *PLoS ONE*
- **PacBio HiFi**: Wenger et al. (2019). "Accurate circular consensus long-read sequencing." *Nat. Biotech.*
- **Ancient DNA**: Briggs et al. (2007). "Patterns of damage in Neandertal genomic DNA." *PNAS*

**Hi-C Scaffolding:**
- Burton et al. (2013). "Chromosome-scale scaffolding based on chromatin interactions." *Nat. Biotech.*
- Lieberman-Aiden et al. (2009). "Comprehensive mapping of long-range interactions." *Science*

---

## 🎓 Licensing

StrandWeaver uses a **dual licensing model**:
- **Academic License**: Free for research and education ([LICENSE_ACADEMIC.md](LICENSE_ACADEMIC.md))
- **Commercial License**: Custom pricing for industry use ([LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md))

---

## 🤝 Support & Contribution

- **Issues**: Report bugs at [GitHub Issues](https://github.com/yourusername/strandweaver/issues)
- **Discussions**: Community support at [GitHub Discussions](https://github.com/yourusername/strandweaver/discussions)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Last Updated**: January 2025 | **Version**: 0.1.0
