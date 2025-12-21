# StrandWeaver

**AI-Powered Multi-Technology Genome Assembler with GPU Acceleration**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS-brightgreen.svg)](docs/AI_ML_GUIDE.md)
[![Status](https://img.shields.io/badge/status-Phase%205%20Active-yellow.svg)](docs/MASTER_DEVELOPMENT_ROADMAP.md)

StrandWeaver is a next-generation genome assembly pipeline combining machine-learning optimized technology-aware error correction, GPU-accelerated graph-based assembly, Hi-C scaffolding, and AI-powered finishing for ancient DNA, Illumina, ONT, ultra-long ONT, and PacBio sequencing data. It's goal is to relieve manual curation bottlenecks in traditional high-contiguity genome assembly by applying AI / ML to genome graph paths and other complex regions. It uses these technologies to improve accuracy and contiguity, but also to provide functional annotations, such as structural variants, during the assembly process. StrandWeaver is inspired by MaSuRCA, Verkko, and Hifiasm, but also includes novel elements such as ancient DNA assembly optimization, with machine learning models trained to profile (and optionally repair using different confidence thresholds) deamination damage. StrandWeaver's models have been trained on real and simulated data for the human genome (complex centromeres), the chicken genome (high repeat content), the yeast genome, and the E. coli genome, as well as hundreds of simulated ground-truth datasets. It can easily be custom trained using provided scripts for any data type (new technology) or scenario (your genome has 80% repeats) to optimize for your organism of choice.

The current AI / ML optimizations include two for read correction:
1) ***K-Weaver***: optimizes k-mer values on read set for correction and downstream assembly. Dynamic regional K values are used for better handling of complex genome features (centromeres) and data types (ONT-UL vs PacBio HiFi).
2) ***ErrorSmith***: models error types and rates by technology.

There are 5 models for assembly and assembly navigation:
1) ***EdgeWarden***: graph edge / overlap classification.
2) ***PathWeaver***: a graph neural network that intelligently augments the initial de Bruijin Graph creation using long reads.
3) ***ThreadCompass***: optimizes ONT UL paths over de Bruijin graph using training on UL patterns across hundreds of string graphs.
4) ***SVScribe***: finds structural variants DURING assembly, aiding both misassembly detection and providing the user with detailed SVs, rather than using coverage heuristics later.
5) ***PhaseWeaver***: a graph neural network for phasing, augmented by Hi-C.

All AI / ML features are balanced with (and can be disabled for) classical heuristics.

> **🚀 Latest (December 2025):** Complete GPU acceleration (20-50× speedup for Hi-C, ∞× for DBG), multi-platform support (CUDA/MPS/CPU), AI training data generation infrastructure complete, active ML model training in progress.

---

## ✨ Key Features

### Core Assembly
- 🧬 **Multi-Technology Support**: Ancient DNA, Illumina, ONT R9/R10, PacBio HiFi/CLR, Ultra-long reads
- 🔀 **Hybrid Assembly**: Intelligently combine data from multiple sequencing platforms
- 📊 **Three Assembly Steps (depending on tech)**: 
  - Overlap-Layout-Consensus (OLC) for short reads
  - De Bruijn Graph (DBG) for long reads
  - String Graph for ultra-long read integration
- 🏛️ **Ancient DNA Mode**: Enhanced mapDamage2-inspired correction with C→T/G→A deamination modeling
- 🔗 **Hi-C Scaffolding**: Chromosome-scale scaffolding using proximity ligation data with spectral phasing

### GPU Acceleration ⚡
- **Apple Silicon (MPS)**: Native support for M1/M2/M3 chips
- **NVIDIA (CUDA)**: Support for all CUDA-capable GPUs
- **CPU Fallback**: Automatic detection with graceful degradation
- **HPC-Safe**: Explicit backend control for cluster environments
- **Performance**: 
  - DBG construction: ∞× speedup (9-22s vs never completing)
  - UL read mapping: 15× speedup
  - Contig building: 7.2× speedup
  - Hi-C integration: 20-50× speedup

### AI-Powered Features 
- **5 AI Subsystems**:
  1. EdgeWarden (overlap filtering)
  2. PathWeaver (optimal contig paths)
  3. PhaseWeaver (haplotype separation)
  4. ThreadCompass (ultra-long read alignment resolution)
  5. SVScribe (structural variant identification)
- **BandageNG Integration**: Manual graph editing and correction import
- **At-Home Training**: Generate and train models on your own data
- **Pre-trained Models**: Available for common scenarios (coming soon)

---

## 📋 Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [GPU Acceleration](#gpu-acceleration)
- [AI/ML Training](#aiml-training)
- [Documentation](#documentation)

**📚 Full Documentation:**
- [AI/ML Features Guide](docs/AI_ML_GUIDE.md) - Comprehensive AI documentation
- [Training Guide](TRAINING_GUIDE.md) - Model training instructions
- [GPU Guide](GPU_ACCELERATION_GUIDE.md) - GPU setup and backends
- [Module Reference](ASSEMBLY_MODULE_REFERENCE.md) - Python module documentation
- [Pipeline Flow](docs/PIPELINE_FLOW.md) - Visual pipeline diagram
- [Scientific References](docs/SCIENTIFIC_REFERENCES.md) - Citations
- [Development Roadmap](docs/MASTER_DEVELOPMENT_ROADMAP.md) - Project status

---

## 🔧 Installation

### Requirements
- Python 3.9+
- 8 GB RAM minimum (32+ GB recommended)
- Optional: NVIDIA GPU (CUDA 11.0+) or Apple Silicon

### Basic Installation
\`\`\`bash
git clone https://github.com/pgrady1322/strandweaver.git
cd strandweaver
pip install -e .
\`\`\`

### GPU Installation
**Apple Silicon:**
\`\`\`bash
pip install -e ".[mps]"
\`\`\`

**NVIDIA CUDA:**
\`\`\`bash
pip install -e ".[cuda]"
\`\`\`

### Full (with AI):
\`\`\`bash
pip install -e ".[ai]"
\`\`\`

---

## 🚀 Quick Start

### Simple Assembly
\`\`\`bash
strandweaver assemble \
  --illumina-pe reads_R1.fastq reads_R2.fastq \
  --output assembly.fasta \
  --threads 8
\`\`\`

### Hybrid Assembly with GPU
\`\`\`bash
strandweaver assemble \
  --hifi hifi.fastq \
  --hic hic_R1.fastq hic_R2.fastq \
  --output scaffolds.fasta \
  --use-gpu \
  --threads 16
\`\`\`

---

## ⚡ GPU Acceleration

### Auto-Detection
\`\`\`bash
strandweaver assemble --hifi reads.fastq --use-gpu
\`\`\`

### Explicit Backend (HPC-Safe)
\`\`\`bash
export STRANDWEAVER_GPU_BACKEND=cuda  # or mps, cpu
strandweaver assemble --hifi reads.fastq --use-gpu
\`\`\`

### Performance
| Component | CPU | GPU (CUDA/MPS) | Speedup |
|-----------|-----|----------------|---------|
| DBG | Never completes | 9-22s | ∞ |
| UL Mapping | ~300s | ~20s | 15× |
| Hi-C | ~360s | ~15-25s | 20-50× |
| **Total** | **20+ min** | **1-2 min** | **20-30×** |

---

## 🤖 AI/ML Training

### Generate Training Data
\`\`\`bash
python scripts/generate_assembly_training_data.py \
  --scenario fast_balanced \
  --output-dir training_data/assembly_ai \
  --num-workers 8 \
  --use-gpu
\`\`\`

### Available Scenarios
- \`simple\` - 10 genomes, 100kb (testing)
- \`fast_balanced\` - 20 genomes, 500kb (quick training)
- \`balanced\` - 100 genomes, 1Mb (production)
- \`repeat_heavy\` - 50 genomes, 2Mb, 60% repeats
- \`sv_dense\` - 50 genomes, 1Mb, high SV density
- \`diploid_focus\` - 100 genomes, 1Mb, 2% heterozygosity
- \`ultra_long_focus\` - 30 genomes, 5Mb, UL-optimized

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) and [docs/AI_ML_GUIDE.md](docs/AI_ML_GUIDE.md) for details.

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

## 🛠️ Development Status

**Phase 5**: AI Intelligence & ML Training (70% complete)
- ✅ GPU acceleration (CUDA/MPS/CPU)
- ✅ AI modules (5 subsystems)
- ✅ Training infrastructure
- 🔄 Training data generation (95%)
- 📋 ML model training (next)

See [docs/MASTER_DEVELOPMENT_ROADMAP.md](docs/MASTER_DEVELOPMENT_ROADMAP.md).

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
