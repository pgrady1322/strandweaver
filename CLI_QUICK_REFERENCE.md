# StrandWeaver CLI Quick Reference

## Command Modes

All StrandWeaver processing commands support two execution modes:

| Mode | Usage | Best For |
|------|-------|----------|
| **Direct** | `strandweaver <command> [options]` | Small/medium datasets, local workstation, testing |
| **Nextflow** | `strandweaver <command> [options] --nextflow` | Large datasets, HPC clusters, parallel processing |

## Available Commands

### Error Correction

```bash
# Direct mode
strandweaver correct --hifi reads.fq.gz -o corrected/ -t 8

# Nextflow mode (automatic parallelization)
strandweaver correct --hifi reads.fq.gz -o corrected/ \
  --nextflow --nf-profile slurm --correction-batch-size 100000
```

**Options:**
- `--hifi PATH`: HiFi reads to correct
- `--ont PATH`: ONT reads to correct
- `-o, --output PATH`: Output directory (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--nextflow`: Enable Nextflow mode
- `--correction-batch-size INT`: Reads per batch (Nextflow)
- `--max-correction-jobs INT`: Max parallel jobs (Nextflow)

---

### K-mer Extraction

```bash
# Direct mode (normal genomes)
strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl -t 8

# Nextflow mode (huge genomes >10GB)
strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl \
  --nextflow --nf-profile slurm --kmer-batch-size 2000000
```

**Options:**
- `--hifi PATH`: HiFi reads
- `--ont PATH`: ONT reads
- `-k, --kmer-size INT`: K-mer size (required)
- `-o, --output PATH`: Output PKL file (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--nextflow`: Enable Nextflow mode
- `--kmer-batch-size INT`: Reads per batch (Nextflow)
- `--max-kmer-jobs INT`: Max parallel jobs (Nextflow)

---

### Edge Scoring

```bash
# Direct mode
strandweaver score-edges -e edges.json -a aligns.bam -o scored.json -t 8

# Nextflow mode (large graphs)
strandweaver score-edges -e edges.json -a aligns.bam -o scored.json \
  --nextflow --nf-profile slurm --edge-batch-size 10000
```

**Options:**
- `-e, --edges PATH`: Graph edges JSON (required)
- `-a, --alignments PATH`: Read alignments BAM/PAF (required)
- `-o, --output PATH`: Output scored edges JSON (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--nextflow`: Enable Nextflow mode
- `--edge-batch-size INT`: Edges per batch (Nextflow)
- `--max-edge-jobs INT`: Max parallel jobs (Nextflow)

---

### Ultra-Long Read Mapping

```bash
# Direct mode
strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf --use-gpu

# Nextflow mode (many UL reads)
strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf \
  --nextflow --nf-profile slurm --use-gpu --ul-batch-size 100
```

**Options:**
- `-u, --ul-reads PATH`: Ultra-long reads (required)
- `-g, --graph PATH`: Assembly graph GFA (required)
- `-o, --output PATH`: Output PAF file (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--use-gpu`: Enable GPU acceleration
- `--nextflow`: Enable Nextflow mode
- `--ul-batch-size INT`: UL reads per batch (Nextflow)
- `--max-ul-jobs INT`: Max parallel jobs (Nextflow)

---

### Hi-C Alignment

```bash
# Direct mode
strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \
  -g graph.gfa -o aligns.bam -t 8

# Nextflow mode (large Hi-C datasets)
strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \
  -g graph.gfa -o aligns.bam \
  --nextflow --nf-profile slurm --hic-batch-size 500000
```

**Options:**
- `--hic-r1 PATH`: Hi-C R1 reads (required)
- `--hic-r2 PATH`: Hi-C R2 reads (required)
- `-g, --graph PATH`: Assembly graph GFA (required)
- `-o, --output PATH`: Output BAM file (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--nextflow`: Enable Nextflow mode
- `--hic-batch-size INT`: Read pairs per batch (Nextflow)
- `--max-hic-jobs INT`: Max parallel jobs (Nextflow)

---

### Structural Variant Detection

```bash
# Direct mode
strandweaver detect-svs -g graph.gfa -o variants.vcf -t 8

# Nextflow mode (large graphs)
strandweaver detect-svs -g graph.gfa -o variants.vcf \
  --nextflow --nf-profile slurm --sv-batch-size 1000
```

**Options:**
- `-g, --graph PATH`: Assembly graph GFA (required)
- `-o, --output PATH`: Output VCF file (required)
- `-t, --threads INT`: CPU threads for direct mode
- `--nextflow`: Enable Nextflow mode
- `--sv-batch-size INT`: Graph nodes per batch (Nextflow)
- `--max-sv-jobs INT`: Max parallel jobs (Nextflow)

---

## Nextflow Options (All Commands)

When using `--nextflow` flag, these options are available:

### Execution Control
- `--nf-profile TEXT`: Profile to use (default: local)
  - `local`: Single machine
  - `docker`: Docker containers
  - `singularity`: Singularity containers (HPC-friendly)
  - `slurm`: SLURM cluster scheduler
  - Can combine: `slurm,singularity`
  
- `--nf-resume`: Resume from last checkpoint if workflow was interrupted

### Profiles Explained

| Profile | Description | Use Case |
|---------|-------------|----------|
| `local` | Direct execution on local machine | Testing, small datasets |
| `docker` | Docker containerization | Reproducibility, dependency isolation |
| `singularity` | Singularity containers | HPC clusters (no root required) |
| `slurm` | SLURM cluster scheduler | HPC parallel processing |
| `test` | Use synthetic E. coli data | Quick validation |

**Combine profiles:** `-profile slurm,singularity` for SLURM + containers

## Common Workflows

### Local Workstation (Small Dataset)

```bash
# Direct mode - simple and fast
strandweaver correct --hifi reads.fq.gz -o corrected/ -t 8
strandweaver extract-kmers --hifi corrected/corrected_hifi.fastq.gz -k 31 -o kmers.pkl -t 8
```

### HPC Cluster (Large Dataset)

```bash
# Nextflow mode - automatic parallelization
strandweaver correct \
  --hifi large_reads.fq.gz \
  -o corrected/ \
  --nextflow \
  --nf-profile slurm,singularity \
  --nf-resume \
  --correction-batch-size 50000 \
  --max-correction-jobs 40

strandweaver extract-kmers \
  --hifi corrected/corrected_hifi.fastq.gz \
  -k 31 \
  -o kmers.pkl \
  --nextflow \
  --nf-profile slurm,singularity \
  --kmer-batch-size 2000000 \
  --max-kmer-jobs 20
```

### Testing Before Large Run

```bash
# Test on small subset first
head -40000 large_reads.fq > test_subset.fq

# Run direct mode
strandweaver correct --hifi test_subset.fq -o test_out/ -t 4

# If successful, scale up with Nextflow
strandweaver correct --hifi large_reads.fq.gz -o corrected/ \
  --nextflow --nf-profile slurm
```

## Performance Guidelines

### When to Use Direct Mode
- Dataset < 10GB
- Local workstation with 8+ cores
- Testing and debugging
- Quick analysis

### When to Use Nextflow Mode
- Dataset > 10GB
- HPC cluster available
- Need resume capability
- Want automatic parallelization
- Limited local resources

### Typical Speedups (Nextflow vs Direct)

| Command | Direct (1 node) | Nextflow (20 nodes) | Speedup |
|---------|----------------|---------------------|---------|
| `correct` | 20 hours | 2 hours | 10× |
| `extract-kmers` | 8 hours | 1.5 hours | 5× |
| `score-edges` | 8 hours | 1.5 hours | 5× |
| `map-ul` | 6 hours | 1 hour | 6× |
| `align-hic` | 10 hours | 1.5 hours | 7× |
| `detect-svs` | 4 hours | 1 hour | 4× |

## Troubleshooting

### Command Not Found
```bash
# Install StrandWeaver first
pip install -e /path/to/strandweaver
```

### Nextflow Not Found
```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

### Check Nextflow Execution Status
```bash
# View recent runs
nextflow log

# View specific run details
nextflow log <run-name> -f status,name,exit

# Generate execution report
nextflow run main.nf ... -with-report report.html
```

### Resume Failed Workflow
```bash
# Always use --nf-resume to restart from where it failed
strandweaver correct --hifi reads.fq.gz -o corrected/ \
  --nextflow --nf-resume
```

## Getting Help

```bash
# Main help
strandweaver --help

# Command-specific help
strandweaver correct --help
strandweaver extract-kmers --help
strandweaver map-ul --help

# Nextflow help
nextflow run main.nf --help
```

## See Also

- [CLI Refactor Summary](../CLI_REFACTOR_SUMMARY.md) - Architecture details
- [Nextflow README](../nextflow/README.md) - Full Nextflow documentation
- [StrandWeaver README](../README.md) - Project overview
