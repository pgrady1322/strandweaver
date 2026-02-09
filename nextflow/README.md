# StrandWeaver Nextflow Orchestration

## Overview

This directory contains the Nextflow implementation of the StrandWeaver assembly pipeline. Nextflow provides HPC cluster support, automatic parallelization, and container-based execution.

## Quick Start

### Recommended: Use StrandWeaver CLI with --nextflow Flag

The easiest way to run Nextflow workflows is through the StrandWeaver CLI:

```bash
# Error correction with automatic parallelization
strandweaver correct \
  --hifi your_hifi.fastq.gz \
  --ont your_ont.fastq.gz \
  -o corrected_reads/ \
  --nextflow \
  --nf-profile local

# Run on SLURM cluster with Singularity
strandweaver correct \
  --hifi your_hifi.fastq.gz \
  -o corrected_reads/ \
  --nextflow \
  --nf-profile slurm,singularity \
  --nf-resume \
  --correction-batch-size 50000 \
  --max-correction-jobs 40

# Other parallelizable commands
strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl --nextflow
strandweaver nf-score-edges -e edges.json -a aligns.bam -o scored.json --nextflow
strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf --nextflow
strandweaver align-hic --hic-r1 R1.fq --hic-r2 R2.fq -g graph.gfa -o aligns.bam --nextflow
strandweaver nf-detect-svs -g graph.gfa -o variants.vcf --nextflow
```

### Alternative: Direct Nextflow Execution

You can also run Nextflow workflows directly:

```bash
# Test with synthetic E. coli data
cd nextflow
nextflow run main.nf -profile test,local

# Run with your data
nextflow run main.nf \
  --hifi your_hifi.fastq.gz \
  --ont your_ont.fastq.gz \
  --ont_ul your_ultralong.fastq.gz \
  --hic_r1 your_hic_R1.fastq.gz \
  --hic_r2 your_hic_R2.fastq.gz \
  --outdir results/ \
  -profile local

# SLURM cluster with Singularity
nextflow run main.nf \
  --hifi your_hifi.fastq.gz \
  --ont your_ont.fastq.gz \
  --hic_r1 your_hic_R1.fastq.gz \
  --hic_r2 your_hic_R2.fastq.gz \
  --outdir results/ \
  -profile slurm,singularity \
  -resume
```

## Parallelization Strategy

The pipeline implements stage-level parallelization for single-genome assembly:

### Parallel Stages (Batch Processing)

1. **Error Correction**: Reads split into 100K batches (20 parallel jobs)
2. **Edge Scoring**: Edges split into 10K batches (8 parallel jobs)
3. **UL Mapping**: UL reads split into 100-read batches (10 parallel jobs)
4. **Hi-C Alignment**: Hi-C pairs split into 500K batches (15 parallel jobs)
5. **SV Detection**: Graph partitioned into 1K-node regions (10 parallel jobs)
6. **K-mer Extraction** (--huge mode): Reads split into 1M batches (12 parallel jobs)

### Sequential Stages (Full Dataset)

1. **Read Classification**: Technology detection across all reads
2. **Error Profiling**: K-mer spectrum analysis (needs complete dataset)
3. **K-Weaver**: Global statistics for k-mer prediction
4. **Graph Building**: GPU-optimized (unless --huge flag)
5. **PathWeaver**: GNN requires complete graph topology
6. **Hi-C Phasing**: Spectral clustering on full contact matrix

## Parameters

### StrandWeaver CLI Integration

When using `strandweaver <command> --nextflow`, the following options control Nextflow execution:

#### General Options
- `--nf-profile`: Execution profile (local, docker, singularity, slurm)
- `--nf-resume`: Resume from last checkpoint

#### Batch Size Control
- `--correction-batch-size`: Reads per correction batch (default: 100000)
- `--edge-batch-size`: Edges per scoring batch (default: 10000)
- `--ul-batch-size`: UL reads per mapping batch (default: 100)
- `--hic-batch-size`: Hi-C pairs per alignment batch (default: 500000)
- `--sv-batch-size`: Graph nodes per SV detection batch (default: 1000)
- `--kmer-batch-size`: Reads per k-mer extraction batch (default: 1000000)

#### Parallelization Control
- `--max-correction-jobs`: Max parallel correction jobs (default: 20)
- `--max-edge-jobs`: Max edge scoring jobs (default: 8)
- `--max-ul-jobs`: Max UL mapping jobs (default: 10)
- `--max-hic-jobs`: Max Hi-C alignment jobs (default: 15)
- `--max-sv-jobs`: Max SV detection jobs (default: 10)
- `--max-kmer-jobs`: Max k-mer extraction jobs (default: 12)

### Direct Nextflow Parameters

When running `nextflow run main.nf` directly:

### Input Files
- `--hifi`: PacBio HiFi reads (FASTQ)
- `--ont`: Oxford Nanopore reads (FASTQ)
- `--ont_ul`: Ultra-long ONT reads (FASTQ)
- `--illumina_r1`, `--illumina_r2`: Illumina paired-end reads
- `--hic_r1`, `--hic_r2`: Hi-C proximity ligation reads

### Assembly Options
- `--enable_ai`: Enable AI features (default: true)
- `--detect_svs`: Detect structural variants (default: true)
- `--huge`: Enable parallel k-mer extraction for huge genomes (default: false)
- `--preserve_heterozygosity`: Preserve diploid variation (default: true)
- `--min_identity`: Minimum identity threshold (default: 0.995)

### Parallelization Control
- `--correction_batch_size`: Reads per correction job (default: 100000)
- `--edge_batch_size`: Edges per scoring job (default: 10000)
- `--ul_batch_size`: UL reads per mapping job (default: 100)
- `--hic_batch_size`: Read pairs per Hi-C job (default: 500000)
- `--sv_batch_size`: Nodes per SV detection job (default: 1000)
- `--kmer_batch_size`: Reads per k-mer job in --huge mode (default: 1000000)

### Max Parallel Jobs
- `--max_correction_jobs`: Max simultaneous correction jobs (default: 20)
- `--max_edge_jobs`: Max edge scoring jobs (default: 8)
- `--max_ul_jobs`: Max UL mapping jobs (default: 10)
- `--max_hic_jobs`: Max Hi-C alignment jobs (default: 15)
- `--max_sv_jobs`: Max SV detection jobs (default: 10)
- `--max_kmer_jobs`: Max k-mer extraction jobs for --huge (default: 12)

## Profiles

### Local
```bash
-profile local
```
Single-machine execution.

### SLURM
```bash
-profile slurm
```
SLURM cluster execution with GPU support.

### Docker
```bash
-profile docker
```
Run with Docker containers.

### Singularity
```bash
-profile singularity
```
Run with Singularity containers (HPC-friendly).

### Test
```bash
-profile test
```
Use synthetic E. coli data for testing.

## Resume Capability

Nextflow automatically tracks completed tasks. Use `-resume` to restart from where it left off:

```bash
nextflow run main.nf \
  --hifi reads.fastq.gz \
  --outdir results/ \
  -profile slurm,singularity \
  -resume
```

## Directory Structure

```
nextflow/
├── main.nf                    # Main entry point
├── nextflow.config           # Configuration
├── workflows/
│   └── strandweaver.nf       # Assembly workflow
├── modules/local/            # Process definitions
│   ├── profile_errors.nf     # Sequential
│   ├── correct_batch.nf      # Parallel
│   ├── score_edges_batch.nf  # Parallel
│   ├── map_ul_batch.nf       # Parallel
│   ├── align_hic_batch.nf    # Parallel
│   ├── detect_svs_batch.nf   # Parallel
│   ├── extract_kmers_batch.nf # Parallel (--huge)
│   └── ...
├── conf/
│   ├── base.config           # Base resources
│   ├── slurm.config          # SLURM settings
│   ├── docker.config         # Docker settings
│   ├── singularity.config    # Singularity settings
│   └── test.config           # Test data
└── bin/                      # Helper scripts
    ├── partition_graph.py
    └── extract_edges.py
```

## Expected Performance

For a 30× HiFi + 50× ONT + 10× UL + 20× Hi-C human genome:

| Stage | Sequential | Parallel | Speedup |
|-------|-----------|----------|---------|
| Error Profiling | 2h | 2h | 1× |
| Error Correction | 20h | **2h** | **10×** |
| Graph Building | 4h | 4h | 1× |
| Edge Scoring | 8h | **1.5h** | **5×** |
| UL Mapping | 6h | **1h** | **6×** |
| Hi-C Alignment | 10h | **1.5h** | **7×** |
| PathWeaver | 3h | 3h | 1× |
| SV Detection | 4h | **1h** | **4×** |

**Total: 57h → 16h (3.6× speedup)**

## Troubleshooting

### Check Pipeline Status
```bash
nextflow log
```

### View Execution Report
```bash
nextflow run main.nf ... -with-report report.html
```

### View Resource Usage
```bash
nextflow run main.nf ... -with-timeline timeline.html
```

### Enable Debug Logging
```bash
nextflow run main.nf ... -with-trace
```

## Requirements

- Nextflow ≥ 22.10.0
- StrandWeaver Python package installed
- Optional: Docker or Singularity for containers
- Optional: SLURM cluster for HPC execution
