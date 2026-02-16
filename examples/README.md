# StrandWeaver Example Data

This directory contains tools and example datasets for testing StrandWeaver.

## Quick Start

Generate synthetic example data (takes ~1-2 minutes):

```bash
python generate_example_data.py
```

This creates a 5 Mb bacterial genome with realistic sequencing reads:
- **HiFi reads** (30× coverage)
- **ONT reads** (50× coverage)  
- **Illumina paired-end** (100× coverage)
- **Hi-C proximity ligation** (20× coverage)

## Output Structure

```
examples/
├── generate_example_data.py    Script to generate test data
├── ecoli_synthetic/             Generated example dataset
│   ├── genome/
│   │   └── reference.fasta      5 Mb reference genome
│   └── reads/
│       ├── hifi_reads.fastq.gz
│       ├── ont_reads.fastq.gz
│       ├── illumina_R1.fastq.gz
│       ├── illumina_R2.fastq.gz
│       ├── hic_R1.fastq.gz
│       └── hic_R2.fastq.gz
└── README.md                    This file
```

## Testing Assembly

Once generated, test StrandWeaver with the example data:

```bash
cd ecoli_synthetic

# Basic assembly with long reads
strandweaver core-assemble \
    --hifi-long-reads reads/hifi_reads.fastq.gz \
    --ont-long-reads reads/ont_reads.fastq.gz \
    --output asm_out/

# Full assembly with all data types
strandweaver core-assemble \
    --hifi-long-reads reads/hifi_reads.fastq.gz \
    --ont-long-reads reads/ont_reads.fastq.gz \
    --illumina-r1 reads/illumina_R1.fastq.gz --illumina-r2 reads/illumina_R2.fastq.gz \
    --hic-r1 reads/hic_R1.fastq.gz --hic-r2 reads/hic_R2.fastq.gz \
    --output asm_out/
```

## Data Characteristics

### Reference Genome
- **Length:** 5,000,000 bp (5 Mb)
- **GC Content:** 50%
- **Type:** Random sequence (E. coli-like)

### Simulated Reads
| Technology | Coverage | Read Length | Accuracy | Error Profile |
|------------|----------|-------------|----------|---------------|
| PacBio HiFi | 30× | 10-25 kb | 99.9% | Random substitutions |
| ONT | 50× | 10-100 kb | 95% | Substitutions + indels |
| Illumina PE | 100× | 150 bp × 2 | 99.9% | Random substitutions |
| Hi-C | 20× | 150 bp × 2 | 99.9% | Long-range pairs |

## Customization

Edit `generate_example_data.py` to adjust:
- Genome length (default: 5 Mb)
- GC content (default: 50%)
- Coverage depths
- Read length distributions
- Error rates

Example:
```python
# In generate_example_data.py
GENOME_LENGTH = 10_000_000  # 10 Mb instead of 5 Mb
hifi_reads = simulate_hifi_reads(genome, coverage=50.0)  # 50× instead of 30×
```

## Real Data

For real datasets, see:
- [NCBI SRA](https://www.ncbi.nlm.nih.gov/sra)
- [European Nucleotide Archive](https://www.ebi.ac.uk/ena)
- [PacBio Datasets](https://www.pacb.com/connect/datasets/)
- [ONT Community Datasets](https://community.nanoporetech.com/)

## Regeneration

Example data is not version-controlled. Regenerate anytime:

```bash
# Clean and regenerate
rm -rf ecoli_synthetic
python generate_example_data.py
```

---

**Note:** This is synthetic data for testing purposes only. For production assemblies, use real sequencing data.
