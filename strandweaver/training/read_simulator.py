"""
Read Simulator for Training Data Generation

Simulates reads from synthetic genomes:
- Illumina paired-end reads
- PacBio HiFi long reads
- Oxford Nanopore continuous long reads (CLR)
- Oxford Nanopore ultralong reads (UL)
- Hi-C proximity ligation read pairs
- Ancient DNA fragments (with damage patterns)

All simulators return reads with true genomic coordinates for ground-truth labeling.

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
#                           CONFIGURATION
# ============================================================================

@dataclass
class IlluminaConfig:
    """Configuration for Illumina read simulation."""
    coverage: float = 30.0  # Target coverage
    read_length: int = 150  # Read length (bp)
    insert_size_mean: int = 350  # Insert size mean
    insert_size_std: int = 50  # Insert size std dev
    error_rate: float = 0.001  # Substitution error rate
    random_seed: Optional[int] = None


@dataclass
class HiFiConfig:
    """Configuration for PacBio HiFi read simulation."""
    coverage: float = 30.0
    read_length_mean: int = 15_000
    read_length_std: int = 5_000
    error_rate: float = 0.001  # Very low for HiFi
    random_seed: Optional[int] = None


@dataclass
class ONTConfig:
    """Configuration for ONT continuous long read simulation."""
    coverage: float = 30.0
    read_length_mean: int = 20_000
    read_length_std: int = 10_000
    error_rate: float = 0.05  # Higher indel rate
    indel_fraction: float = 0.7  # Fraction of errors that are indels
    random_seed: Optional[int] = None


@dataclass
class ULConfig:
    """Configuration for ONT ultralong read simulation."""
    coverage: float = 5.0  # Lower coverage for UL
    read_length_mean: int = 100_000  # 100kb reads
    read_length_std: int = 50_000
    error_rate: float = 0.08  # Slightly higher for UL
    indel_fraction: float = 0.75
    random_seed: Optional[int] = None


@dataclass
class HiCConfig:
    """Configuration for Hi-C read simulation."""
    num_pairs: int = 1_000_000  # Number of read pairs
    read_length: int = 150
    cis_fraction: float = 0.90  # Fraction of intra-chromosomal contacts
    distance_decay_rate: float = 1.0  # Exponential decay with distance
    random_seed: Optional[int] = None


@dataclass
class AncientDNAConfig:
    """Configuration for ancient DNA fragment simulation."""
    coverage: float = 10.0
    fragment_length_mean: int = 50
    fragment_length_std: int = 20
    damage_rate: float = 0.20  # C->T deamination rate at ends
    damage_length: int = 5  # Damage in first/last N bases
    error_rate: float = 0.005
    random_seed: Optional[int] = None


# ============================================================================
#                           READ DATA STRUCTURES
# ============================================================================

@dataclass
class SimulatedRead:
    """
    A simulated sequencing read with ground-truth coordinates.
    
    Attributes:
        read_id: Unique read identifier
        sequence: Read sequence
        quality: Quality scores (Phred+33 ASCII)
        haplotype: Source haplotype ('A', 'B', or None for mixed)
        chrom: Source chromosome
        start_pos: True start position in haplotype
        end_pos: True end position in haplotype
        strand: Strand orientation ('+' or '-')
        errors: List of introduced errors
    """
    read_id: str
    sequence: str
    quality: str
    haplotype: Optional[str]
    chrom: str
    start_pos: int
    end_pos: int
    strand: str = '+'
    errors: List[Dict] = field(default_factory=list)
    
    def to_fastq(self) -> str:
        """Convert to FASTQ format."""
        return f"@{self.read_id}\n{self.sequence}\n+\n{self.quality}\n"


@dataclass
class SimulatedReadPair:
    """Paired-end read (Illumina or Hi-C)."""
    read1: SimulatedRead
    read2: SimulatedRead
    insert_size: int = 0
    
    def to_fastq(self) -> Tuple[str, str]:
        """Convert to FASTQ format (R1, R2)."""
        return (self.read1.to_fastq(), self.read2.to_fastq())


# ============================================================================
#                     ERROR INTRODUCTION FUNCTIONS
# ============================================================================

def introduce_substitution_errors(sequence: str, error_rate: float) -> Tuple[str, List[Dict]]:
    """Introduce random substitution errors."""
    seq_list = list(sequence)
    errors = []
    
    for i in range(len(seq_list)):
        if random.random() < error_rate:
            original = seq_list[i]
            bases = ['A', 'C', 'G', 'T']
            bases.remove(original)
            seq_list[i] = random.choice(bases)
            errors.append({'type': 'substitution', 'pos': i, 'original': original, 'new': seq_list[i]})
    
    return ''.join(seq_list), errors


def introduce_indel_errors(sequence: str, error_rate: float, indel_fraction: float) -> Tuple[str, List[Dict]]:
    """Introduce insertion and deletion errors (ONT-like)."""
    result = []
    errors = []
    i = 0
    
    while i < len(sequence):
        if random.random() < error_rate:
            if random.random() < indel_fraction:
                # Indel error
                if random.random() < 0.5:
                    # Insertion
                    insert_base = random.choice(['A', 'C', 'G', 'T'])
                    result.append(insert_base)
                    errors.append({'type': 'insertion', 'pos': i, 'base': insert_base})
                    result.append(sequence[i])
                else:
                    # Deletion - skip this base
                    errors.append({'type': 'deletion', 'pos': i, 'base': sequence[i]})
            else:
                # Substitution
                original = sequence[i]
                bases = ['A', 'C', 'G', 'T']
                bases.remove(original)
                new_base = random.choice(bases)
                result.append(new_base)
                errors.append({'type': 'substitution', 'pos': i, 'original': original, 'new': new_base})
        else:
            result.append(sequence[i])
        i += 1
    
    return ''.join(result), errors


def generate_quality_string(length: int, mean_qual: int = 30) -> str:
    """Generate random quality scores."""
    qualities = []
    for _ in range(length):
        qual = max(2, int(random.gauss(mean_qual, 5)))
        qual = min(qual, 40)
        qualities.append(chr(qual + 33))  # Phred+33
    return ''.join(qualities)


# ============================================================================
#                     ILLUMINA READ SIMULATION
# ============================================================================

def simulate_illumina_reads(genome: str, config: IlluminaConfig, haplotype: str = 'A') -> List[SimulatedReadPair]:
    """
    Simulate Illumina paired-end reads.
    
    Args:
        genome: Source genome sequence
        config: Illumina configuration
        haplotype: Haplotype identifier
    
    Returns:
        List of simulated read pairs
    """
    if config.random_seed is not None:
        random.seed(config.random_seed)
    
    genome_length = len(genome)
    num_reads = int((genome_length * config.coverage) / (2 * config.read_length))
    
    logger.info(f"Simulating {num_reads} Illumina read pairs ({config.coverage}x coverage)...")
    
    read_pairs = []
    
    for i in range(num_reads):
        # Random start position
        insert_size = max(config.read_length * 2, int(random.gauss(config.insert_size_mean, config.insert_size_std)))
        start_pos = random.randint(0, genome_length - insert_size)
        end_pos = start_pos + insert_size
        
        # Extract fragment
        fragment = genome[start_pos:end_pos]
        
        # Read 1 (forward)
        r1_seq = fragment[:config.read_length]
        r1_seq, r1_errors = introduce_substitution_errors(r1_seq, config.error_rate)
        r1_qual = generate_quality_string(len(r1_seq))
        
        read1 = SimulatedRead(
            read_id=f"illumina_{haplotype}_{i}_R1",
            sequence=r1_seq,
            quality=r1_qual,
            haplotype=haplotype,
            chrom='chr1',
            start_pos=start_pos,
            end_pos=start_pos + len(r1_seq),
            strand='+',
            errors=r1_errors
        )
        
        # Read 2 (reverse)
        r2_seq = fragment[-config.read_length:]
        # Reverse complement
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        r2_seq = ''.join(complement.get(b, b) for b in r2_seq[::-1])
        r2_seq, r2_errors = introduce_substitution_errors(r2_seq, config.error_rate)
        r2_qual = generate_quality_string(len(r2_seq))
        
        read2 = SimulatedRead(
            read_id=f"illumina_{haplotype}_{i}_R2",
            sequence=r2_seq,
            quality=r2_qual,
            haplotype=haplotype,
            chrom='chr1',
            start_pos=end_pos - config.read_length,
            end_pos=end_pos,
            strand='-',
            errors=r2_errors
        )
        
        read_pairs.append(SimulatedReadPair(read1, read2, insert_size))
    
    logger.info(f"Generated {len(read_pairs)} Illumina read pairs")
    return read_pairs


# ============================================================================
#                     LONG READ SIMULATION (HIFI/ONT)
# ============================================================================

def simulate_long_reads(
    genome: str,
    config,  # HiFiConfig or ONTConfig or ULConfig
    haplotype: str = 'A',
    read_type: str = 'hifi'
) -> List[SimulatedRead]:
    """
    Simulate long reads (HiFi, ONT, or UL).
    
    Args:
        genome: Source genome sequence
        config: Read configuration
        haplotype: Haplotype identifier
        read_type: 'hifi', 'ont', or 'ul'
    
    Returns:
        List of simulated long reads
    """
    if config.random_seed is not None:
        random.seed(config.random_seed)
    
    genome_length = len(genome)
    num_reads = int((genome_length * config.coverage) / config.read_length_mean)
    
    logger.info(f"Simulating {num_reads} {read_type.upper()} reads ({config.coverage}x coverage)...")
    
    reads = []
    
    for i in range(num_reads):
        # Random read length
        read_length = max(1000, int(random.gauss(config.read_length_mean, config.read_length_std)))
        read_length = min(read_length, genome_length)
        
        # Random start position
        start_pos = random.randint(0, genome_length - read_length)
        end_pos = start_pos + read_length
        
        # Extract sequence
        sequence = genome[start_pos:end_pos]
        
        # Introduce errors
        if read_type == 'hifi':
            # Low error, mostly substitutions
            sequence, errors = introduce_substitution_errors(sequence, config.error_rate)
        else:  # ONT or UL
            # Higher error, mostly indels
            sequence, errors = introduce_indel_errors(sequence, config.error_rate, config.indel_fraction)
        
        # Generate quality
        mean_qual = 20 if read_type == 'hifi' else 10
        quality = generate_quality_string(len(sequence), mean_qual)
        
        read = SimulatedRead(
            read_id=f"{read_type}_{haplotype}_{i}",
            sequence=sequence,
            quality=quality,
            haplotype=haplotype,
            chrom='chr1',
            start_pos=start_pos,
            end_pos=end_pos,
            strand='+',
            errors=errors
        )
        
        reads.append(read)
    
    logger.info(f"Generated {len(reads)} {read_type.upper()} reads")
    return reads


# ============================================================================
#                     HI-C READ SIMULATION
# ============================================================================

def simulate_hic_reads(genome_A: str, genome_B: str, config: HiCConfig) -> List[SimulatedReadPair]:
    """
    Simulate Hi-C proximity ligation read pairs.
    
    Args:
        genome_A: Haplotype A sequence
        genome_B: Haplotype B sequence
        config: Hi-C configuration
    
    Returns:
        List of Hi-C read pairs with contact information
    """
    if config.random_seed is not None:
        random.seed(config.random_seed)
    
    logger.info(f"Simulating {config.num_pairs} Hi-C read pairs...")
    
    read_pairs = []
    
    for i in range(config.num_pairs):
        # Decide if cis or trans contact
        is_cis = random.random() < config.cis_fraction
        
        # First read from random haplotype
        hap1 = random.choice(['A', 'B'])
        genome1 = genome_A if hap1 == 'A' else genome_B
        pos1 = random.randint(0, len(genome1) - config.read_length)
        
        if is_cis:
            # Second read from same haplotype, distance-dependent
            # Exponential decay with distance
            max_distance = len(genome1) // 2
            distance = int(random.expovariate(1.0 / (max_distance * config.distance_decay_rate)))
            distance = min(distance, max_distance)
            
            # Random direction
            if random.random() < 0.5:
                pos2 = min(pos1 + distance, len(genome1) - config.read_length)
            else:
                pos2 = max(pos1 - distance, 0)
            
            hap2 = hap1
            genome2 = genome1
        else:
            # Trans contact - different haplotype or distant region
            hap2 = 'B' if hap1 == 'A' else 'A'
            genome2 = genome_B if hap2 == 'B' else genome_A
            pos2 = random.randint(0, len(genome2) - config.read_length)
        
        # Extract reads
        seq1 = genome1[pos1:pos1 + config.read_length]
        seq2 = genome2[pos2:pos2 + config.read_length]
        
        # Introduce errors
        seq1, err1 = introduce_substitution_errors(seq1, 0.001)
        seq2, err2 = introduce_substitution_errors(seq2, 0.001)
        
        # Create reads
        read1 = SimulatedRead(
            read_id=f"hic_{i}_R1",
            sequence=seq1,
            quality=generate_quality_string(len(seq1)),
            haplotype=hap1,
            chrom='chr1',
            start_pos=pos1,
            end_pos=pos1 + len(seq1),
            strand='+',
            errors=err1
        )
        
        read2 = SimulatedRead(
            read_id=f"hic_{i}_R2",
            sequence=seq2,
            quality=generate_quality_string(len(seq2)),
            haplotype=hap2,
            chrom='chr1',
            start_pos=pos2,
            end_pos=pos2 + len(seq2),
            strand='+',
            errors=err2
        )
        
        read_pairs.append(SimulatedReadPair(read1, read2, abs(pos2 - pos1)))
    
    logger.info(f"Generated {len(read_pairs)} Hi-C read pairs")
    return read_pairs


# ============================================================================
#                     WRITE READS TO FILES
# ============================================================================

def write_fastq(reads: List[SimulatedRead], output_path: str) -> None:
    """Write single-end reads to FASTQ file."""
    with open(output_path, 'w') as f:
        for read in reads:
            f.write(read.to_fastq())
    logger.info(f"Wrote {len(reads)} reads to {output_path}")


def write_paired_fastq(pairs: List[SimulatedReadPair], output_r1: str, output_r2: str) -> None:
    """Write paired-end reads to FASTQ files."""
    with open(output_r1, 'w') as f1, open(output_r2, 'w') as f2:
        for pair in pairs:
            r1_fq, r2_fq = pair.to_fastq()
            f1.write(r1_fq)
            f2.write(r2_fq)
    logger.info(f"Wrote {len(pairs)} read pairs to {output_r1}, {output_r2}")
