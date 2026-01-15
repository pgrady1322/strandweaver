# StrandWeaver

**AI-Powered Multi-Technology Genome Assembler with GPU Acceleration**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](docs/MASTER_DEVELOPMENT_ROADMAP.md)
[![License](https://img.shields.io/badge/license-Dual%20Academic%2FCommercial-blue.svg)](LICENSE)

StrandWeaver is a next-generation genome assembly pipeline combining machine-learning optimized technology-aware error correction, graph-based assembly with haplotype-aware path resolution, and comprehensive structural variant detection for ancient DNA, Illumina, ONT, ultra-long ONT, and PacBio sequencing data. Its goal is to relieve manual curation bottlenecks in traditional high-contiguity genome assembly by applying AI/ML to genome graph paths and other complex regions. It uses these technologies to improve accuracy and contiguity, but also to provide functional annotations, such as structural variants, during the assembly process.

StrandWeaver is inspired by MaSuRCA, Verkko, and Hifiasm, but includes novel elements such as:
- **Haplotype-aware assembly**: Graph topology simplification with strict protection of biological variation (SNPs, indels, CNVs) across diploid genomes
- **Ancient DNA optimization**: Machine learning models trained to profile and repair deamination damage (C→T/G→A patterns) with configurable confidence thresholds
- **Multi-technology integration**: Seamless combination of ONT, PacBio HiFi, ultra-long reads, and Hi-C data in a unified assembly graph

The pipeline can be custom trained using provided scripts for any data type (new sequencing technology) or organism-specific scenario (genomes with extreme repeat content, high heterozygosity, or complex structural variation).

> **🚀 Latest (December 2025):** Complete end-to-end assembly pipeline with AI-powered modules for edge filtering, path resolution, Hi-C scaffolding, and structural variant detection. Full haplotype-aware diploid assembly with comprehensive output generation including GFA graphs, BandageNG visualization, and SV calls.

---

## ✨ Key Features

### Core Assembly
- 🧬 **Multi-Technology Support**: Ancient DNA, Illumina, ONT R9/R10, PacBio HiFi/CLR, Ultra-long reads
- 🔀 **Hybrid Assembly**: Intelligently combine data from multiple sequencing platforms
- 📊 **Multi-Stage Assembly Pipeline**: 
  - **De Bruijn Graph (DBG)**: K-mer-based graph construction for long reads
  - **EdgeWarden Filtering**: AI-powered edge quality assessment with 80-feature extraction
  - **PathWeaver Resolution**: Haplotype-aware path selection with variation protection
  - **String Graph Overlay**: Ultra-long read integration for long-range connections
  - **ThreadCompass Routing**: UL read-guided path optimization
  - **Hi-C Scaffolding**: Chromosome-scale scaffolding with proximity ligation
  - **Haplotype Phasing**: Spectral clustering with Hi-C contact matrices
- 🧬 **Diploid Assembly**: 
  - Protects SNP-level variation (>99.5% identity threshold)
  - Never collapses across haplotype boundaries
  - Maintains diploid structure throughout assembly
  - Iterative refinement with phasing context (2-3 cycles)
  - Hi-C-guided haplotype separation
- 🏛️ **Ancient DNA Mode**: Enhanced mapDamage2-inspired correction with C→T/G→A deamination modeling
- 📊 **Structural Variant Detection**: Assembly-time SV calling with graph topology analysis

### Advanced Features
- 🎯 **80-Feature Edge Scoring**: EdgeWarden uses comprehensive feature extraction:
  - 26 static features (graph topology, coverage, node properties)
  - 34 temporal features (quality/coverage trajectories, error patterns)
  - 20 expanded features (sequence complexity, boundaries, systematic errors)
  - Integrates alignment data when available, gracefully degrades to 26 features
- 🔄 **Iterative Refinement**: 2-3 assembly iteration cycles with phasing-aware filtering
- 🧠 **AI-Powered Path Selection**: Graph neural networks optimize contig paths through complex regions
- 🧬 **Hi-C Integration**: 
  - Proximity ligation contact matrix construction
  - Long-range connectivity validation
  - Chromosome-scale scaffolding with gap size estimation
  - Haplotype phasing via spectral clustering
- 📊 **Structural Variant Detection**: SVScribe identifies variants during assembly:
  - Deletions, insertions, inversions, duplications, translocations
  - Graph topology signatures
  - Ultra-long read spanning evidence
  - Hi-C long-range validation
- 📄 **Comprehensive Output**: 
  - Assembly graphs (GFA format)
  - BandageNG-compatible visualization files
  - Detailed statistics (N50, L50, coverage)
  - SV calls (VCF/JSON)
  - Phasing information
  - Coverage tracks for all read types
- 🧪 **Training Infrastructure**: Generate custom training data for organism-specific optimization
- 🔌 **Modular Architecture**: All AI features can be disabled for classical heuristics

### AI-Powered Features
- **Complete AI Subsystem Suite**:
  1. ✅ **K-Weaver**: K-mer optimization with ML prediction and rule-based fallback
  2. ✅ **ErrorSmith**: Technology-specific error profiling
  3. ✅ **EdgeWarden**: 80-feature edge filtering with alignment integration
  4. ✅ **PathWeaver**: GNN-based haplotype-aware path resolution with variation protection
  5. ✅ **ThreadCompass**: Ultra-long read routing optimization through assembly graphs
  6. ✅ **HaplotypeDetangler**: Hi-C-augmented phasing with spectral clustering
  7. ✅ **SVScribe**: Assembly-time structural variant detection
- **Training Infrastructure**: Complete data generation pipeline for custom model training
- **Pre-trained Models**: Available for common scenarios (human, model organisms)
- **Classical Fallbacks**: All AI modules have deterministic heuristic alternatives

---

## 📋 Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [GPU Acceleration](#gpu-acceleration)
- [AI/ML Training](#aiml-training)
- [Documentation](#documentation)

**📚 Full Documentation:**
- [User Guide](docs/USER_GUIDE.md) - Complete tutorials and usage examples
- [AI/ML Features Guide](docs/AI_ML_GUIDE.md) - Comprehensive AI documentation
- [Training Guide](TRAINING_GUIDE.md) - Model training instructions
- [Benchmarks](BENCHMARKS.md) - Performance comparisons
- [Module Reference](ASSEMBLY_MODULE_REFERENCE.md) - Python module documentation
- [Pipeline Flow](docs/PIPELINE_FLOW.md) - Visual pipeline diagram
- [Scientific References](docs/SCIENTIFIC_REFERENCES.md) - Citations

---

## 🔧 Installation

### Requirements
- Python 3.9+
- 8 GB RAM minimum (32+ GB recommended for large genomes)
- Disk space: 50-100 GB for intermediate files (genome-dependent)

### Basic Installation
\`\`\`bash
git clone https://github.com/pgrady1322/strandweaver.git
cd strandweaver
pip install -e .
\`\`\`

### Installation with AI Dependencies
\`\`\`bash
pip install -e ".[ai]"
\`\`\`

### Installation with Hi-C Support
\`\`\`bash
pip install -e ".[hic]"
\`\`\`

### Complete Installation (Recommended)
\`\`\`bash
pip install -e ".[all]"
\`\`\`

---

## 🚀 Quick Start

### Long Read Assembly (ONT/PacBio)
\`\`\`bash
strandweaver assemble \
  --ont ont_reads.fastq \
  --output contigs.fasta \
  --threads 8
\`\`\`

### Hybrid Assembly with Ultra-Long Reads
\`\`\`bash
strandweaver assemble \
  --hifi hifi_reads.fastq \
  --ont-ul ultralong_reads.fastq \
  --output assembly.fasta \
  --threads 16
\`\`\`

### Mixed Technology Assembly (Full Pipeline)
\`\`\`bash
strandweaver assemble \
  --ont ont_reads.fastq \
  --hifi hifi_reads.fastq \
  --ont-ul ultralong_reads.fastq \
  --output assembly.fasta \
  --config custom_config.yaml \
  --threads 16
\`\`\`

---

## 🎯 Use Cases

### Reference-Quality Genome Assembly
Combine ONT, HiFi, ultra-long reads, and Hi-C for chromosome-scale phased assemblies:
\`\`\`bash
strandweaver assemble \\\n  --ont ont.fastq \\\n  --hifi hifi.fastq \\\n  --ont-ul ultralong.fastq \\\n  --hic hic_R1.fastq hic_R2.fastq \\\n  --output genome.fasta \\\n  --ai-enabled \\\n  --detect-svs \\\n  --threads 32
\`\`\`

### Ancient DNA Assembly
Optimize for deamination damage with specialized error correction:
\`\`\`bash
strandweaver assemble \\\n  --ancient-dna ancient_reads.fastq \\\n  --output ancient_genome.fasta \\\n  --damage-aware \\\n  --threads 16
\`\`\`

### SV-Rich Genome Analysis
Detect structural variants during assembly for cancer or population genomics:
\`\`\`bash
strandweaver assemble \\\n  --hifi tumor.fastq \\\n  --hic hic_R1.fastq hic_R2.fastq \\\n  --output tumor_assembly.fasta \\\n  --detect-svs \\\n  --sv-mode sensitive \\\n  --threads 24
\`\`\`

### Highly Heterozygous Diploid Assembly
Maintain haplotype separation for F1 hybrids or outcrossing species:
\`\`\`bash
strandweaver assemble \\\n  --hifi hifi.fastq \\\n  --hic hic_R1.fastq hic_R2.fastq \\\n  --output diploid.fasta \\\n  --preserve-heterozygosity \\\n  --min-identity 0.995 \\\n  --threads 32
\`\`\`

---

## 🔬 Pipeline Modules

### Preprocessing
1. **CLASSIFY**: Auto-detect sequencing technologies from FASTQ headers (supports ONT chemistry detection with LongBow)
2. **KWEAVER**: ML-based k-mer optimization with rule-based fallback for dynamic k-mer selection
3. **PROFILE**: Error pattern profiling (substitutions, indels, homopolymers) with visualization
4. **CORRECT**: Technology-aware read correction (ONTCorrector, PacBioCorrector)

### Core Assembly
5. **DBG**: De Bruijn graph construction from long reads with streaming architecture
6. **EdgeWarden**: AI-powered graph edge filtering with 80-feature scoring
7. **PathWeaver**: GNN-based haplotype-aware path resolution with variation protection
8. **StringGraph**: Ultra-long read overlay for long-range connections
9. **ThreadCompass**: UL read routing optimization with trained models
10. **Hi-C Integration**: Proximity ligation contact matrix construction and edge addition
11. **HaplotypeDetangler**: Hi-C-augmented phasing via spectral clustering
12. **Iteration**: 2-3 refinement cycles with phasing-aware filtering
13. **SVScribe**: Graph-based structural variant detection (DEL, INS, INV, DUP, TRA)
14. **Finalize**: Contig and scaffold extraction with comprehensive statistics

### Output Generation
15. **GFA Export**: Assembly graphs in GFA format with sequences
16. **BandageNG**: Visualization files with coverage tracks and final 0 - 1 range StrandWeaver scores (long/UL/Hi-C)
17. **Statistics**: N50, L50, coverage metrics, variation protection counts
18. **SV Calls**: Structural variants in VCF and JSON formats
19. **Phasing Info**: Haplotype assignments and confidence scores

---

## 🤖 AI/ML Features & Training

### Current AI Capabilities

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

**K-Weaver** (K-mer Optimization):
- ML prediction for 4 assembly stages (DBG, UL overlap, extension, polish)
- Rule-based fallback if models unavailable
- Technology-specific k-mer selection

### Generate Custom Training Data

\`\`\`bash
python scripts/generate_assembly_training_data.py \
  --scenario fast_balanced \
  --output-dir training_data/assembly_ai \
  --num-workers 8
\`\`\`

**Available Scenarios**:
- \`simple\` - 10 genomes, 100kb (testing)
- \`fast_balanced\` - 20 genomes, 500kb (quick training)
- \`balanced\` - 100 genomes, 1Mb (production)
- \`repeat_heavy\` - 50 genomes, 2Mb, 60% repeats
- \`sv_dense\` - 50 genomes, 1Mb, high SV density
- \`diploid_focus\` - 100 genomes, 1Mb, 2% heterozygosity
- \`ultra_long_focus\` - 30 genomes, 5Mb, UL-optimized

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) and [docs/AI_ML_GUIDE.md](docs/AI_ML_GUIDE.md) for model training details.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [AI/ML Guide](docs/AI_ML_GUIDE.md) | Comprehensive AI features and models |
| [Training Guide](TRAINING_GUIDE.md) | Training data generation and model training |
| [GPU Guide](GPU_ACCELERATION_GUIDE.md) | GPU setup, backends, HPC usage |
| [Module Reference](ASSEMBLY_MODULE_REFERENCE.md) | Python module documentation |
| [Pipeline Flow](docs/PIPELINE_FLOW.md) | Visual pipeline flowchart |
| [Scientific References](docs/SCIENTIFIC_REFERENCES.md) | Citations and algorithms |
| [Development Roadmap](docs/MASTER_DEVELOPMENT_ROADMAP.md) | Project status |
| [AI/ML Status](docs/AI_ML_IMPLEMENTATION_STATUS.md) | AI/ML progress (internal) |
| [Training Log](docs/TRAINING_LOG.md) | Training runs and model versions |

---

## 🛠️ Pipeline Status

**Current State**: Production-Ready End-to-End Assembly Pipeline

**✅ Complete Pipeline Capabilities**:
- ✅ **Preprocessing**: Read classification, k-mer optimization, error profiling, correction
- ✅ **Core Assembly**: DBG construction, edge filtering, path resolution, string graph overlay
- ✅ **Advanced Features**: Hi-C scaffolding, haplotype phasing, SV detection
- ✅ **AI Modules**: 7 trained subsystems (K-Weaver, ErrorSmith, EdgeWarden, PathWeaver, ThreadCompass, HaplotypeDetangler, SVScribe)
- ✅ **Output Generation**: GFA export, BandageNG integration, comprehensive statistics
- ✅ **Iteration System**: 2-3 refinement cycles with phasing context
- ✅ **Variation Protection**: SNP/indel/CNV detection with strict haplotype boundaries

**Performance**:
- **Human genome** (3.1 Gb): ~8-12 hours (64 cores, 128 GB RAM)
- **Model organisms** (100-500 Mb): ~30-90 minutes (16 cores, 32 GB RAM)
- **Microbial genomes** (1-10 Mb): ~5-15 minutes (8 cores, 16 GB RAM)

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

**Benchmarks** (compared to existing assemblers):
- **Contiguity**: Comparable to Hifiasm + manual curation
- **Accuracy**: >99.95% consensus accuracy with HiFi + Hi-C
- **Haplotype Resolution**: Superior phasing with Hi-C integration
- **SV Detection**: 15-20% more SVs detected vs post-assembly calling
- **Manual Curation**: 60-80% reduction in required manual intervention

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance comparisons.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete user manual with tutorials |
| [AI/ML Guide](docs/AI_ML_GUIDE.md) | Comprehensive AI features and models |
| [Training Guide](TRAINING_GUIDE.md) | Training data generation and model training |
| [Benchmarks](BENCHMARKS.md) | Performance comparisons and validation |
| [Module Reference](ASSEMBLY_MODULE_REFERENCE.md) | Python module API documentation |
| [Pipeline Flow](docs/PIPELINE_FLOW.md) | Visual pipeline flowchart |
| [Scientific References](docs/SCIENTIFIC_REFERENCES.md) | Citations and algorithms |
| [Hi-C Integration Guide](docs/HIC_INTEGRATION.md) | Hi-C data preparation and scaffolding |
| [SV Detection Guide](docs/SV_DETECTION.md) | Structural variant calling documentation |

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

**Patrick Grady** | patrickgsgrady(at)gmail.com

---

## 📈 Citation

\`\`\`bibtex
@software{strandweaver2025,
  author = {Grady, Patrick},
  title = {StrandWeaver: AI-Powered Multi-Technology Genome Assembler},
  year = {2025},
  url = {https://github.com/pgrady1322/strandweaver}
}
\`\`\`

---

**StrandWeaver** 🧬⚡🤖
