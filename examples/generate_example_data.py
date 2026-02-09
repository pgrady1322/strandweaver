#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Generate simplified example data — synthetic 5 Mb genome with multi-technology reads
(HiFi, ONT, Illumina, Hi-C) for testing.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from pathlib import Path
import gzip
import logging
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
#                    SIMPLE SEQUENCE GENERATION
# ============================================================================

def generate_random_sequence(length: int, gc_content: float = 0.50, seed: int = None) -> str:
    """Generate random DNA sequence with specified GC content."""
    if seed is not None:
        random.seed(seed)
    
    gc_prob = gc_content / 2
    at_prob = (1 - gc_content) / 2
    
    bases = ['A', 'C', 'G', 'T']
    weights = [at_prob, gc_prob, gc_prob, at_prob]
    
    return ''.join(random.choices(bases, weights=weights, k=length))


def generate_quality_string(length: int, mean_quality: int = 30) -> str:
    """Generate Phred quality string."""
    qualities = []
    for _ in range(length):
        q = max(2, min(40, int(random.gauss(mean_quality, 3))))
        qualities.append(chr(q + 33))
    return ''.join(qualities)


def introduce_errors(sequence: str, error_rate: float) -> str:
    """Introduce sequencing errors."""
    if error_rate == 0:
        return sequence
    
    seq_list = list(sequence)
    for i in range(len(seq_list)):
        if random.random() < error_rate:
            bases = ['A', 'C', 'G', 'T']
            if seq_list[i] in bases:
                bases.remove(seq_list[i])
            seq_list[i] = random.choice(bases)
    
    return ''.join(seq_list)


def reverse_complement(seq: str) -> str:
    """Return reverse complement of sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join([complement.get(b, 'N') for b in reversed(seq)])


def extract_read(genome: str, pos: int, length: int, strand: str = '+') -> str:
    """Extract a read from genome at position."""
    if pos + length > len(genome):
        pos = len(genome) - length
    if pos < 0:
        pos = 0
    
    read = genome[pos:pos+length]
    
    if strand == '-':
        read = reverse_complement(read)
    
    return read


# ============================================================================
#                    FASTQ WRITING FUNCTIONS
# ============================================================================

def write_fasta(filename: Path, sequence: str, header: str = "reference"):
    """Write sequence to FASTA file."""
    logger.info(f"  Writing {filename.name}...")
    with open(filename, 'w') as f:
        f.write(f">{header} length={len(sequence)}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")


def write_fastq_gz(filename: Path, reads: list):
    """Write reads to gzipped FASTQ file."""
    logger.info(f"  Writing {filename.name} ({len(reads):,} reads)...")
    with gzip.open(filename, 'wt') as f:
        for read_id, seq, qual in reads:
            f.write(f"@{read_id}\n{seq}\n+\n{qual}\n")


def write_paired_fastq_gz(filename_r1: Path, filename_r2: Path, read_pairs: list):
    """Write paired reads to gzipped FASTQ files."""
    logger.info(f"  Writing {filename_r1.name} and {filename_r2.name} ({len(read_pairs):,} pairs)...")
    with gzip.open(filename_r1, 'wt') as f1, gzip.open(filename_r2, 'wt') as f2:
        for read_id, seq1, qual1, seq2, qual2 in read_pairs:
            f1.write(f"@{read_id}/1\n{seq1}\n+\n{qual1}\n")
            f2.write(f"@{read_id}/2\n{seq2}\n+\n{qual2}\n")


# ============================================================================
#                    READ SIMULATION
# ============================================================================

def simulate_hifi_reads(genome: str, coverage: float, read_length_mean: int = 15000):
    """Simulate PacBio HiFi reads."""
    reads = []
    genome_length = len(genome)
    num_reads = int((genome_length * coverage) / read_length_mean)
    
    for i in range(num_reads):
        # Random position and length
        read_length = int(random.gauss(read_length_mean, read_length_mean * 0.3))
        read_length = max(5000, min(25000, read_length))
        pos = random.randint(0, genome_length - read_length)
        strand = random.choice(['+', '-'])
        
        # Extract and add errors (HiFi is very accurate)
        read_seq = extract_read(genome, pos, read_length, strand)
        read_seq = introduce_errors(read_seq, error_rate=0.001)  # 0.1% error
        
        # Generate quality
        qual = generate_quality_string(len(read_seq), mean_quality=30)
        
        reads.append((f"hifi_{i:06d}", read_seq, qual))
    
    return reads


def simulate_ont_reads(genome: str, coverage: float, read_length_mean: int = 30000):
    """Simulate Oxford Nanopore reads."""
    reads = []
    genome_length = len(genome)
    num_reads = int((genome_length * coverage) / read_length_mean)
    
    for i in range(num_reads):
        # Random position and length (ONT has wider length distribution)
        read_length = int(random.gauss(read_length_mean, read_length_mean * 0.5))
        read_length = max(1000, min(100000, read_length))
        pos = random.randint(0, genome_length - min(read_length, genome_length))
        strand = random.choice(['+', '-'])
        
        # Extract and add errors (ONT has more errors)
        read_seq = extract_read(genome, pos, read_length, strand)
        read_seq = introduce_errors(read_seq, error_rate=0.05)  # 5% error
        
        # Generate quality
        qual = generate_quality_string(len(read_seq), mean_quality=13)
        
        reads.append((f"ont_{i:06d}", read_seq, qual))
    
    return reads


def simulate_illumina_reads(genome: str, coverage: float, read_length: int = 150, insert_size: int = 400):
    """Simulate Illumina paired-end reads."""
    read_pairs = []
    genome_length = len(genome)
    num_pairs = int((genome_length * coverage) / (read_length * 2))
    
    for i in range(num_pairs):
        # Random position
        fragment_size = int(random.gauss(insert_size, 50))
        fragment_size = max(read_length * 2, min(1000, fragment_size))
        pos = random.randint(0, genome_length - fragment_size)
        strand = random.choice(['+', '-'])
        
        # Extract fragment
        fragment = extract_read(genome, pos, fragment_size, strand)
        
        # R1 from start of fragment
        r1 = fragment[:read_length]
        r1 = introduce_errors(r1, error_rate=0.001)  # 0.1% error
        qual1 = generate_quality_string(len(r1), mean_quality=35)
        
        # R2 from end of fragment (reverse complement)
        r2 = reverse_complement(fragment[-read_length:])
        r2 = introduce_errors(r2, error_rate=0.001)
        qual2 = generate_quality_string(len(r2), mean_quality=35)
        
        read_pairs.append((f"illumina_{i:06d}", r1, qual1, r2, qual2))
    
    return read_pairs


def simulate_hic_reads(genome: str, coverage: float, read_length: int = 150):
    """Simulate Hi-C proximity ligation reads."""
    read_pairs = []
    genome_length = len(genome)
    num_pairs = int((genome_length * coverage) / (read_length * 2))
    
    for i in range(num_pairs):
        # Two random positions (long-range contacts)
        pos1 = random.randint(0, genome_length - read_length)
        pos2 = random.randint(0, genome_length - read_length)
        strand1 = random.choice(['+', '-'])
        strand2 = random.choice(['+', '-'])
        
        # Extract reads from two different positions
        r1 = extract_read(genome, pos1, read_length, strand1)
        r1 = introduce_errors(r1, error_rate=0.001)
        qual1 = generate_quality_string(len(r1), mean_quality=35)
        
        r2 = extract_read(genome, pos2, read_length, strand2)
        r2 = introduce_errors(r2, error_rate=0.001)
        qual2 = generate_quality_string(len(r2), mean_quality=35)
        
        read_pairs.append((f"hic_{i:06d}", r1, qual1, r2, qual2))
    
    return read_pairs


# ============================================================================
#                    README GENERATION
# ============================================================================

def write_readme(output_dir: Path, genome_length: int, coverage_info: dict):
    """Write README file."""
    readme = f"""# StrandWeaver Example Data

## Overview

Synthetic bacterial genome for testing StrandWeaver assembly pipeline.

- **Genome Size:** {genome_length:,} bp ({genome_length/1000:.1f} kb)
- **GC Content:** 50%
- **Generated:** {datetime.now().strftime("%Y-%m-%d")}

## Files

```
ecoli_synthetic/
├── genome/
│   └── reference.fasta          Reference genome
└── reads/
    ├── hifi_reads.fastq.gz      HiFi reads ({coverage_info['hifi']}× coverage)
    ├── ont_reads.fastq.gz       ONT reads ({coverage_info['ont']}× coverage)
    ├── illumina_R1.fastq.gz     Illumina R1 ({coverage_info['illumina']}× coverage)
    ├── illumina_R2.fastq.gz     Illumina R2
    ├── hic_R1.fastq.gz          Hi-C R1 ({coverage_info['hic']}× coverage)
    └── hic_R2.fastq.gz          Hi-C R2
```

## Quick Start

```bash
# Basic assembly with HiFi + ONT
strandweaver assemble \\
    --hifi reads/hifi_reads.fastq.gz \\
    --ont reads/ont_reads.fastq.gz \\
    --output assembly.fasta

# Assembly with all data types
strandweaver assemble \\
    --hifi reads/hifi_reads.fastq.gz \\
    --ont reads/ont_reads.fastq.gz \\
    --illumina reads/illumina_R1.fastq.gz reads/illumina_R2.fastq.gz \\
    --hic reads/hic_R1.fastq.gz reads/hic_R2.fastq.gz \\
    --output assembly.fasta
```

## Read Statistics

| Technology | Coverage | Read Length | Accuracy | Reads |
|------------|----------|-------------|----------|-------|
| PacBio HiFi | {coverage_info['hifi']}× | 10-25 kb | 99.9% | {coverage_info['hifi_count']:,} |
| ONT | {coverage_info['ont']}× | 10-100 kb | 95% | {coverage_info['ont_count']:,} |
| Illumina PE | {coverage_info['illumina']}× | 150 bp × 2 | 99.9% | {coverage_info['illumina_count']:,} pairs |
| Hi-C | {coverage_info['hic']}× | 150 bp × 2 | 99.9% | {coverage_info['hic_count']:,} pairs |

---

**Note:** This is synthetic data for testing purposes only.
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)


# ============================================================================
#                    MAIN FUNCTION
# ============================================================================

def main():
    """Generate complete example dataset."""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("  StrandWeaver Example Data Generator")
    print("=" * 70)
    print()
    
    # Setup output directories
    base_dir = Path(__file__).parent
    output_dir = base_dir / "ecoli_synthetic"
    genome_dir = output_dir / "genome"
    reads_dir = output_dir / "reads"
    
    genome_dir.mkdir(parents=True, exist_ok=True)
    reads_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration (5 Mb bacterial genome)
    GENOME_LENGTH = 5_000_000  # 5 Mb
    GC_CONTENT = 0.50
    
    # Step 1: Generate genome
    logger.info("Step 1: Generating reference genome...")
    random.seed(42)
    genome = generate_random_sequence(GENOME_LENGTH, GC_CONTENT, seed=42)
    write_fasta(genome_dir / "reference.fasta", genome, "reference")
    print()
    
    # Step 2: Simulate HiFi reads
    logger.info("Step 2: Simulating HiFi reads (30× coverage)...")
    hifi_reads = simulate_hifi_reads(genome, coverage=30.0)
    write_fastq_gz(reads_dir / "hifi_reads.fastq.gz", hifi_reads)
    print()
    
    # Step 3: Simulate ONT reads
    logger.info("Step 3: Simulating ONT reads (50× coverage)...")
    ont_reads = simulate_ont_reads(genome, coverage=50.0)
    write_fastq_gz(reads_dir / "ont_reads.fastq.gz", ont_reads)
    print()
    
    # Step 4: Simulate Illumina reads
    logger.info("Step 4: Simulating Illumina reads (100× coverage)...")
    illumina_reads = simulate_illumina_reads(genome, coverage=100.0)
    write_paired_fastq_gz(
        reads_dir / "illumina_R1.fastq.gz",
        reads_dir / "illumina_R2.fastq.gz",
        illumina_reads
    )
    print()
    
    # Step 5: Simulate Hi-C reads
    logger.info("Step 5: Simulating Hi-C reads (20× coverage)...")
    hic_reads = simulate_hic_reads(genome, coverage=20.0)
    write_paired_fastq_gz(
        reads_dir / "hic_R1.fastq.gz",
        reads_dir / "hic_R2.fastq.gz",
        hic_reads
    )
    print()
    
    # Step 6: Write README
    logger.info("Step 6: Writing README...")
    coverage_info = {
        'hifi': 30,
        'ont': 50,
        'illumina': 100,
        'hic': 20,
        'hifi_count': len(hifi_reads),
        'ont_count': len(ont_reads),
        'illumina_count': len(illumina_reads),
        'hic_count': len(hic_reads)
    }
    write_readme(output_dir, GENOME_LENGTH, coverage_info)
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"  COMPLETE! ({elapsed:.1f}s)")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print(f"  ├── genome/reference.fasta ({GENOME_LENGTH:,} bp)")
    print(f"  └── reads/")
    print(f"      ├── hifi_reads.fastq.gz ({len(hifi_reads):,} reads)")
    print(f"      ├── ont_reads.fastq.gz ({len(ont_reads):,} reads)")
    print(f"      ├── illumina_R1/R2.fastq.gz ({len(illumina_reads):,} pairs)")
    print(f"      └── hic_R1/R2.fastq.gz ({len(hic_reads):,} pairs)")
    print()
    print("Test with:")
    print("  cd examples/ecoli_synthetic")
    print("  strandweaver assemble --hifi reads/hifi_reads.fastq.gz \\")
    print("                        --ont reads/ont_reads.fastq.gz \\")
    print("                        --output assembly.fasta")
    print("=" * 70)


if __name__ == "__main__":
    main()

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
