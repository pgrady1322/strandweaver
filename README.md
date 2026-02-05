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

### Install from GitHub (Recommended)

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

### Basic Long Read Assembly Example (PacBio)
```bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --output contigs.fasta \
  --threads 8
```

### Hybrid Assembly with Multiple Ultra-Long Read Types
*Note:* The --ont-ul flag is used for path-finding reads. Any platform of long reads can be provided, but shorter long reads will degrade the assembly. The ont-ul name is retained for clarity / comparison with other assemblers.
```bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --ont ont_reads.fastq.gz \
  --ont-ul ultralong_reads.fastq \ 
  --output assembly.fasta \
  --threads 16
```

### Mixed Technology Assembly with Hi-C
*Note:* ANY platform of proximity ligation tech can be provided. StrandWeaver will optimize for Hi-C and Omni-C just as well as Pore-C and CiFi.
```bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --ont-ul ultralong_reads.fastq \
  --hic hic_R1.fastq hic_R2.fastq \
  --output assembly.fasta \
  --threads 16
```

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

### Output Generation
15. **GFA Export**: Assembly graphs in GFA format with sequences
16. **BandageNG**: Visualization files with coverage tracks and final 0 - 1 range StrandWeaver scores (long/UL/Hi-C).
17. **Statistics**: N50, L50, coverage metrics, variation protection counts
18. **SV Calls**: Structural variants in VCF and JSON formats
19. **Phasing Info**: Haplotype assignments and confidence scores

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
├── scaffolds.fasta                # Hi-C scaffolded sequences
├── assembly_graph.gfa             # Assembly graph (GFA format)
├── assembly_stats.json            # N50, L50, coverage statistics
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

## 📚 Documentation

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
