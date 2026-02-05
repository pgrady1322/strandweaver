# User-Configurable Training Data Generation

This directory provides a flexible, parameter-driven system for generating custom training data for StrandWeaver ML models. Unlike the scenario-based training system, this gives you full control over all genome and sequencing parameters.

## Quick Start

### Command-Line Interface

```bash
# Basic usage - 10 diploid genomes (1Mb each) with HiFi reads
python -m strandweaver.user_training.generate_training_data \
  --genome-size 1000000 \
  --num-genomes 10 \
  --output training_data/test

# Multi-technology dataset with custom coverage
python -m strandweaver.user_training.generate_training_data \
  --genome-size 5000000 \
  --num-genomes 50 \
  --read-types hifi ont ultra_long hic \
  --coverage 30 20 10 15 \
  --output training_data/multi_tech

# Repeat-rich genomes
python -m strandweaver.user_training.generate_training_data \
  --genome-size 2000000 \
  --num-genomes 20 \
  --repeat-density 0.60 \
  --gc-content 0.35 \
  --output training_data/repeat_rich
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
    └── illumina_R2.fastq        # Illumina R2 (if requested)
```

## Use Cases

### 1. Quick Test Dataset
```bash
python -m strandweaver.user_training.generate_training_data \
  --genome-size 100000 \
  --num-genomes 5 \
  --read-types hifi \
  --coverage 10 \
  --output test_data
```

### 2. Production HiFi + UL + Hi-C Dataset
```bash
python -m strandweaver.user_training.generate_training_data \
  --genome-size 5000000 \
  --num-genomes 100 \
  --read-types hifi ultra_long hic \
  --coverage 30 10 20 \
  --output production_data
```

### 3. Repeat-Heavy Genomes
```bash
python -m strandweaver.user_training.generate_training_data \
  --genome-size 2000000 \
  --num-genomes 50 \
  --repeat-density 0.70 \
  --read-types hifi ont \
  --output repeat_heavy
```

### 4. High Heterozygosity (SV-Dense)
```bash
python -m strandweaver.user_training.generate_training_data \
  --genome-size 3000000 \
  --num-genomes 30 \
  --snp-rate 0.002 \
  --sv-density 0.0001 \
  --sv-types deletion insertion inversion duplication translocation \
  --output high_hetero
```

### 5. Ancient DNA Dataset
```bash
python -m strandweaver.user_training.generate_training_data \
  --genome-size 1000000 \
  --num-genomes 20 \
  --read-types ancient_dna illumina \
  --coverage 5 10 \
  --output ancient_dna
```

## Performance Estimates

Training data generation time varies with genome size and complexity:

| Genome Size | # Genomes | Read Types | Estimated Time |
|-------------|-----------|------------|----------------|
| 100 kb | 10 | 1 (HiFi) | ~1 minute |
| 1 Mb | 10 | 1 (HiFi) | ~3 minutes |
| 1 Mb | 100 | 1 (HiFi) | ~20 minutes |
| 5 Mb | 50 | 3 (HiFi, UL, Hi-C) | ~60 minutes |
| 10 Mb | 20 | 4 (all) | ~90 minutes |

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
- Use `--no-compress` flag cautiously for large datasets

## Comparison to Scenario-Based Training

| Feature | Scenario-Based | User-Configurable |
|---------|----------------|-------------------|
| **Ease of use** | ✅ Very easy | ⚠️ Requires parameter knowledge |
| **Flexibility** | ❌ Fixed scenarios | ✅ Full control |
| **Reproducibility** | ✅ Named scenarios | ✅ Saved configurations |
| **Best for** | Quick starts, standard use cases | Custom organisms, specific research |

## Training Models with Generated Data

Once you've generated training data, use it to train StrandWeaver's AI models.

### Installation Requirements

Training requires additional dependencies beyond the base StrandWeaver installation:

```bash
# Install StrandWeaver with training dependencies
pip install strandweaver[ai]

# Or install all optional dependencies (includes training, dev tools, Hi-C)
pip install strandweaver[all]

# Or install training dependencies separately
pip install torch>=2.0.0 pytorch-geometric>=2.3.0 xgboost>=2.0.0
```

**Note**: The base `pip install strandweaver` only installs inference/assembly dependencies. Training functionality requires the `[ai]` extra to install PyTorch, PyTorch Geometric, and XGBoost.

### Using the ML Training System

```python
from strandweaver.training.ml_training_system import MLTrainingSystem

# Initialize training system
trainer = MLTrainingSystem(
    training_data_dir='training_data/my_dataset',  # Your generated data
    output_dir='models/my_models',
    num_epochs=50,
    batch_size=32
)

# Train all models
trainer.train_all_models()

# Or train specific models
trainer.train_kmer_selector()
trainer.train_repeat_classifier()
trainer.train_haplotype_separator()
```

### Command-Line Training

```bash
# Train all models on your custom dataset
python -m strandweaver.training.main_training_workflow \
  --data-dir training_data/my_dataset \
  --output-dir models/my_models \
  --epochs 50

# Train specific model types
python -m strandweaver.training.main_training_workflow \
  --data-dir training_data/my_dataset \
  --output-dir models/my_models \
  --models kmer_selector repeat_classifier haplotype_separator \
  --epochs 50
```

### Training Configuration

Create a training configuration file (`training_config.yaml`):

```yaml
# Data paths
training_data: training_data/my_dataset
validation_split: 0.2
output_dir: models/my_models

# Training parameters
epochs: 50
batch_size: 32
learning_rate: 0.001
early_stopping_patience: 10

# Models to train
models:
  - kmer_selector       # K-mer analysis for optimal k
  - repeat_classifier   # Repeat detection
  - error_corrector     # Read error correction (ErrorSmith)
  - thread_predictor    # Threading paths (ThreadCompass)
  - haplotype_separator # Haplotype phasing (HaplotypeDetangler)
  - sv_detector         # Structural variants (SVScribe)
  - coverage_analyzer   # Coverage depth analysis
  - gc_bias_corrector   # GC bias correction

# Hardware
gpu: true
num_workers: 8
```

Then train with:

```bash
python -m strandweaver.training.main_training_workflow --config training_config.yaml
```

### Evaluating Trained Models

```python
from strandweaver.training.ml_training_system import MLTrainingSystem

# Load trained models
trainer = MLTrainingSystem.from_checkpoint('models/my_models')

# Evaluate on validation data
metrics = trainer.evaluate_all_models('training_data/my_dataset/validation')

print(f"K-mer Selector Accuracy: {metrics['kmer_selector']['accuracy']:.4f}")
print(f"Repeat Classifier F1: {metrics['repeat_classifier']['f1_score']:.4f}")
print(f"Haplotype Separator Precision: {metrics['haplotype_separator']['precision']:.4f}")
```

### Using Custom Models in Assembly

Once trained, use your custom models for assembly:

```python
from strandweaver import StrandWeaverAssembler

# Create assembler with custom models
assembler = StrandWeaverAssembler(
    read_files=['sample_hifi.fastq', 'sample_ont.fastq'],
    output_dir='assembly_output',
    model_dir='models/my_models',  # Your trained models
    genome_size='5M',
    ploidy=2
)

# Run assembly with your custom-trained AI
assembly_results = assembler.run()
```

### Training Data Requirements

**Minimum dataset size for training:**
- **Development/Testing**: 10-20 genomes, 100 kb - 1 Mb each
- **Production**: 100-500 genomes, 1 Mb - 10 Mb each
- **High-Performance**: 1000+ genomes, 5 Mb - 50 Mb each

**Recommended read coverage:**
- K-mer analysis: ≥20x coverage
- Error correction: ≥30x coverage
- Haplotype phasing: ≥40x coverage per haplotype
- SV detection: ≥10x long-read coverage

### Training Workflow Example

Complete workflow from data generation to trained models:

```bash
# 1. Generate training data
python -m strandweaver.user_training.generate_training_data \
  --genome-size 5000000 \
  --num-genomes 100 \
  --read-types hifi ont hic \
  --coverage 40 20 30 \
  --output training_data/production

# 2. Inspect generated data
ls training_data/production/
cat training_data/production/generation_summary.json

# 3. Train models
python -m strandweaver.training.main_training_workflow \
  --data-dir training_data/production \
  --output-dir models/production_models \
  --epochs 50 \
  --gpu

# 4. Evaluate models
python -m strandweaver.training.evaluate_models \
  --model-dir models/production_models \
  --test-data training_data/production/validation

# 5. Use models for assembly
strandweaver assemble \
  --reads sample.fastq \
  --model-dir models/production_models \
  --output assembly_output
```

### Training Tips

**For repeat-heavy genomes:**
```python
# Generate data with high repeat density
genome_config = UserGenomeConfig(
    genome_size=5_000_000,
    num_genomes=100,
    repeat_density=0.60  # 60% repeats
)

# Train with increased focus on repeat regions
trainer.train_repeat_classifier(
    epochs=100,  # More epochs
    augment_repeats=True,  # Data augmentation
    focus_weight=2.0  # Increased loss weight on repeats
)
```

**For high-heterozygosity genomes:**
```python
# Generate SV-rich data
genome_config = UserGenomeConfig(
    genome_size=5_000_000,
    num_genomes=100,
    snp_rate=0.002,
    sv_density=0.0001  # Dense SVs
)

# Train haplotype separator with SV focus
trainer.train_haplotype_separator(
    epochs=75,
    sv_aware=True,  # SV-aware training
    complex_regions=True  # Include difficult phasing regions
)
```

## Next Steps

After generating training data:

1. **Inspect the data**: Check `generation_summary.json` for statistics
2. **Validate reads**: Use `samtools` to check FASTQ quality
3. **Train models**: Use `strandweaver.training.ml_training_system`
4. **Evaluate**: Test trained models on real data

See the main [Training Guide](../training/README.md) for model training instructions.

---

**StrandWeaver User Training Infrastructure** | February 2026
