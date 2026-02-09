#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Generate 1 Mb test dataset — synthetic genome with HiFi reads for testing
graph assembly and compaction.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from pathlib import Path
import gzip
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
#                    SEQUENCE GENERATION
# ============================================================================

def generate_random_sequence(length: int, gc_content: float = 0.50, seed: int = 42) -> str:
    """Generate random DNA sequence with specified GC content."""
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
    """Introduce sequencing errors (substitutions only for simplicity)."""
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
        # Wrap around to avoid boundary issues
        pos = pos % (len(genome) - length)
    if pos < 0:
        pos = 0
    
    read = genome[pos:pos+length]
    
    if strand == '-':
        read = reverse_complement(read)
    
    return read


# ============================================================================
#                    FASTQ WRITING
# ============================================================================

def write_fasta(filename: Path, sequence: str, header: str = "test_genome_1mb"):
    """Write sequence to FASTA file."""
    logger.info(f"  Writing reference: {filename.name} ({len(sequence):,} bp)")
    with open(filename, 'w') as f:
        f.write(f">{header} length={len(sequence)}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")


def write_fastq_gz(filename: Path, reads: list):
    """Write reads to gzipped FASTQ file."""
    total_bases = sum(len(seq) for _, seq, _ in reads)
    logger.info(f"  Writing reads: {filename.name} ({len(reads):,} reads, {total_bases:,} bp)")
    with gzip.open(filename, 'wt') as f:
        for read_id, seq, qual in reads:
            f.write(f"@{read_id}\n{seq}\n+\n{qual}\n")


# ============================================================================
#                    READ SIMULATION
# ============================================================================

def simulate_hifi_reads(genome: str, coverage: float, read_length_mean: int = 15000, seed: int = 123):
    """
    Simulate PacBio HiFi reads with realistic characteristics.
    
    Args:
        genome: Reference genome sequence
        coverage: Target coverage depth (e.g., 30 for 30×)
        read_length_mean: Mean read length in bp
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    reads = []
    genome_length = len(genome)
    num_reads = int((genome_length * coverage) / read_length_mean)
    
    logger.info(f"Simulating {num_reads:,} HiFi reads for {coverage}× coverage...")
    
    for i in range(num_reads):
        # Random read length with realistic distribution
        read_length = int(random.gauss(read_length_mean, read_length_mean * 0.3))
        read_length = max(5000, min(25000, read_length))
        
        # Random position and strand
        pos = random.randint(0, genome_length - 1)
        strand = random.choice(['+', '-'])
        
        # Extract read
        read_seq = extract_read(genome, pos, read_length, strand)
        
        # Add realistic HiFi errors (very low error rate, mostly substitutions)
        read_seq = introduce_errors(read_seq, error_rate=0.001)  # 0.1% error rate
        
        # Generate high-quality scores
        qual = generate_quality_string(len(read_seq), mean_quality=30)
        
        reads.append((f"hifi_read_{i:07d}", read_seq, qual))
    
    return reads


# ============================================================================
#                    MAIN EXECUTION
# ============================================================================

def main():
    """Generate 1 Mb test dataset."""
    logger.info("=" * 70)
    logger.info("StrandWeaver 1 Mb Test Dataset Generator")
    logger.info("=" * 70)
    
    # Configuration
    genome_size = 1_000_000  # 1 Mb
    gc_content = 0.50
    hifi_coverage = 50  # 50× coverage for good assembly
    hifi_read_length = 15_000
    
    # Create output directory
    output_dir = Path("test_data_1mb")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"\nOutput directory: {output_dir}")
    
    # Generate reference genome
    logger.info(f"\n1. Generating {genome_size:,} bp reference genome (GC={gc_content:.1%})...")
    genome = generate_random_sequence(genome_size, gc_content=gc_content, seed=42)
    
    # Write reference
    ref_file = output_dir / "reference.fasta"
    write_fasta(ref_file, genome)
    
    # Simulate HiFi reads
    logger.info(f"\n2. Simulating HiFi reads ({hifi_coverage}× coverage)...")
    hifi_reads = simulate_hifi_reads(genome, coverage=hifi_coverage, 
                                     read_length_mean=hifi_read_length, seed=123)
    
    # Write HiFi reads
    hifi_file = output_dir / "hifi_reads.fastq.gz"
    write_fastq_gz(hifi_file, hifi_reads)
    
    # Calculate actual coverage
    total_bases = sum(len(seq) for _, seq, _ in hifi_reads)
    actual_coverage = total_bases / genome_size
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Dataset Generation Complete!")
    logger.info("=" * 70)
    logger.info(f"Reference genome:  {ref_file}")
    logger.info(f"  Size:            {genome_size:,} bp")
    logger.info(f"  GC content:      {gc_content:.1%}")
    logger.info(f"\nHiFi reads:        {hifi_file}")
    logger.info(f"  Read count:      {len(hifi_reads):,}")
    logger.info(f"  Total bases:     {total_bases:,}")
    logger.info(f"  Target coverage: {hifi_coverage}×")
    logger.info(f"  Actual coverage: {actual_coverage:.1f}×")
    logger.info(f"  Mean length:     {hifi_read_length:,} bp")
    logger.info(f"  Error rate:      0.1%")
    logger.info("\nTo test with StrandWeaver:")
    logger.info(f"  python3 -m strandweaver.cli pipeline \\")
    logger.info(f"    -r1 {hifi_file} \\")
    logger.info(f"    --technology1 pacbio \\")
    logger.info(f"    -o test_output/assembly_1mb \\")
    logger.info(f"    -t 4 \\")
    logger.info(f"    --gpu-backend cpu \\")
    logger.info(f"    --skip-profiling")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
