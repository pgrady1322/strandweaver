"""
Synthetic Data Generator for ML Training

This module consolidates genome simulation, read simulation, and ground-truth labeling
into a single comprehensive system for generating ML training data.

Components:
1. Genome Simulation - Generate diploid genomes with SVs, repeats, and realistic features
2. Read Simulation - Simulate reads from multiple sequencing technologies
3. Ground-Truth Labeling - Create labeled training examples from synthetic data

Technologies Supported:
- Illumina paired-end reads
- PacBio HiFi long reads
- Oxford Nanopore CLR and ultralong reads
- Hi-C proximity ligation
- Ancient DNA fragments (with damage patterns)

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation (Consolidated)
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum
from pathlib import Path
import re

logger = logging.getLogger(__name__)


# ============================================================================
#                    PART 1: GENOME SIMULATION
# ============================================================================



# ============================================================================
#                           CONFIGURATION
# ============================================================================

@dataclass
class GenomeConfig:
    """Configuration for synthetic genome generation."""
    
    # Basic genome parameters
    length: int = 1_000_000  # Total genome length (bp)
    gc_content: float = 0.42  # Overall GC content (human-like)
    
    # Repeat parameters
    repeat_density: float = 0.45  # Fraction of genome in repeats
    tandem_repeat_fraction: float = 0.10  # Of repeat content
    interspersed_repeat_fraction: float = 0.35  # SINEs, LINEs, etc.
    
    # Centromeric regions
    num_centromeres: int = 1  # Number of centromeric arrays
    centromere_length: int = 50_000  # Length of each centromere
    alpha_sat_unit: int = 171  # Alpha satellite repeat unit length
    
    # Gene density
    gene_dense_fraction: float = 0.30  # Fraction of genome that's gene-dense
    gene_dense_gc: float = 0.50  # GC in gene-dense regions
    gene_poor_gc: float = 0.38  # GC in gene-poor regions
    
    # Diploid variation parameters
    snp_rate: float = 0.001  # 1 SNP per 1000 bp (human heterozygosity)
    indel_rate: float = 0.0001  # 1 indel per 10,000 bp
    indel_length_mean: int = 3  # Mean indel length
    indel_length_std: int = 2  # Std dev of indel length
    
    # Structural variant parameters
    sv_density: float = 0.00001  # 1 SV per 100kb
    sv_deletion_fraction: float = 0.30
    sv_insertion_fraction: float = 0.20
    sv_inversion_fraction: float = 0.20
    sv_duplication_fraction: float = 0.20
    sv_translocation_fraction: float = 0.10
    
    # SV size ranges (bp)
    sv_min_size: int = 50
    sv_max_size: int = 100_000
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


class SVType(Enum):
    """Types of structural variants."""
    DELETION = "deletion"
    INSERTION = "insertion"
    INVERSION = "inversion"
    DUPLICATION = "duplication"
    TRANSLOCATION = "translocation"


@dataclass
class StructuralVariant:
    """
    Ground-truth structural variant annotation.
    
    Attributes:
        sv_type: Type of structural variant
        haplotype: Which haplotype has this SV ('A' or 'B')
        chrom: Chromosome/contig name
        pos: Position in reference coordinates
        end: End position (for deletions, inversions, duplications)
        size: Size of the variant
        sequence: Inserted/duplicated sequence (if applicable)
        description: Human-readable description
    """
    sv_type: SVType
    haplotype: str  # 'A' or 'B'
    chrom: str
    pos: int
    end: int
    size: int
    sequence: Optional[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sv_type': self.sv_type.value,
            'haplotype': self.haplotype,
            'chrom': self.chrom,
            'pos': self.pos,
            'end': self.end,
            'size': self.size,
            'sequence': self.sequence[:100] if self.sequence else None,  # Truncate
            'description': self.description
        }


@dataclass
class DiploidGenome:
    """
    Diploid genome with two haplotypes and SV ground truth.
    
    Attributes:
        hapA: Haplotype A sequence
        hapB: Haplotype B sequence
        sv_truth_table: List of all structural variants
        snp_positions: Positions of SNPs
        indel_positions: Positions of indels
        reference: Original haploid reference (before diploid variation)
        config: Configuration used to generate this genome
    """
    hapA: str
    hapB: str
    sv_truth_table: List[StructuralVariant]
    snp_positions: List[int] = field(default_factory=list)
    indel_positions: List[int] = field(default_factory=list)
    reference: Optional[str] = None
    config: Optional[GenomeConfig] = None
    
    @property
    def length_A(self) -> int:
        """Length of haplotype A."""
        return len(self.hapA)
    
    @property
    def length_B(self) -> int:
        """Length of haplotype B."""
        return len(self.hapB)
    
    @property
    def num_svs(self) -> int:
        """Total number of structural variants."""
        return len(self.sv_truth_table)
    
    def get_svs_by_type(self, sv_type: SVType) -> List[StructuralVariant]:
        """Get all SVs of a specific type."""
        return [sv for sv in self.sv_truth_table if sv.sv_type == sv_type]
    
    def get_svs_by_haplotype(self, haplotype: str) -> List[StructuralVariant]:
        """Get all SVs in a specific haplotype."""
        return [sv for sv in self.sv_truth_table if sv.haplotype == haplotype]


# ============================================================================
#                     SEQUENCE GENERATION FUNCTIONS
# ============================================================================

def generate_random_sequence(length: int, gc_content: float = 0.42, seed: Optional[int] = None) -> str:
    """
    Generate random DNA sequence with specified GC content.
    
    Args:
        length: Length of sequence to generate
        gc_content: Fraction of G+C bases (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        Random DNA sequence string
    """
    if seed is not None:
        random.seed(seed)
    
    # Calculate base probabilities
    gc_prob = gc_content / 2  # Split GC equally between G and C
    at_prob = (1 - gc_content) / 2  # Split AT equally between A and T
    
    bases = ['A', 'C', 'G', 'T']
    weights = [at_prob, gc_prob, gc_prob, at_prob]
    
    # Generate sequence
    sequence = ''.join(random.choices(bases, weights=weights, k=length))
    
    return sequence


def generate_tandem_repeat(unit: str, num_copies: int, mutation_rate: float = 0.01) -> str:
    """
    Generate tandem repeat with optional mutations.
    
    Args:
        unit: Repeat unit sequence
        num_copies: Number of repeat copies
        mutation_rate: Probability of mutation per base
    
    Returns:
        Tandem repeat sequence
    """
    sequence = unit * num_copies
    
    # Introduce random mutations
    if mutation_rate > 0:
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < mutation_rate:
                # Mutate to different base
                bases = ['A', 'C', 'G', 'T']
                bases.remove(seq_list[i])
                seq_list[i] = random.choice(bases)
        sequence = ''.join(seq_list)
    
    return sequence


def generate_alpha_satellite_array(length: int, unit_length: int = 171) -> str:
    """
    Generate alpha satellite-like centromeric repeat array.
    
    Alpha satellites are ~171 bp repeating units found in centromeres.
    
    Args:
        length: Total length of array
        unit_length: Length of repeat unit (default 171 bp)
    
    Returns:
        Alpha satellite array sequence
    """
    # Generate a consensus unit
    consensus = generate_random_sequence(unit_length, gc_content=0.42)
    
    # Calculate number of copies
    num_copies = length // unit_length
    
    # Generate array with slight variation between copies
    array = generate_tandem_repeat(consensus, num_copies, mutation_rate=0.05)
    
    # Pad to exact length
    if len(array) < length:
        array += generate_random_sequence(length - len(array), gc_content=0.42)
    else:
        array = array[:length]
    
    return array


def insert_repeats(sequence: str, config: GenomeConfig) -> str:
    """
    Insert various repeat types into a sequence.
    
    Args:
        sequence: Base sequence
        config: Genome configuration
    
    Returns:
        Sequence with repeats inserted
    """
    seq_list = list(sequence)
    length = len(sequence)
    
    # Calculate total repeat content
    total_repeat_bases = int(length * config.repeat_density)
    tandem_bases = int(total_repeat_bases * config.tandem_repeat_fraction)
    
    # Insert tandem repeats
    inserted = 0
    while inserted < tandem_bases:
        # Random position
        pos = random.randint(0, length - 1)
        
        # Random unit length (2-20 bp)
        unit_len = random.randint(2, 20)
        
        # Random number of copies (5-50)
        num_copies = random.randint(5, 50)
        
        # Generate tandem repeat
        unit = generate_random_sequence(unit_len, config.gc_content)
        repeat = generate_tandem_repeat(unit, num_copies, mutation_rate=0.02)
        
        # Insert into sequence
        end_pos = min(pos + len(repeat), length)
        actual_len = end_pos - pos
        seq_list[pos:end_pos] = list(repeat[:actual_len])
        
        inserted += actual_len
    
    return ''.join(seq_list)


def insert_centromeres(sequence: str, config: GenomeConfig) -> str:
    """
    Insert centromeric arrays into sequence.
    
    Args:
        sequence: Base sequence
        config: Genome configuration
    
    Returns:
        Sequence with centromeres inserted
    """
    seq_list = list(sequence)
    length = len(sequence)
    
    # Insert centromeres at evenly spaced positions
    spacing = length // (config.num_centromeres + 1)
    
    for i in range(config.num_centromeres):
        pos = spacing * (i + 1)
        
        # Generate alpha satellite array
        centro_seq = generate_alpha_satellite_array(
            config.centromere_length,
            config.alpha_sat_unit
        )
        
        # Insert into sequence
        end_pos = min(pos + len(centro_seq), length)
        actual_len = end_pos - pos
        seq_list[pos:end_pos] = list(centro_seq[:actual_len])
    
    return ''.join(seq_list)


def modulate_gc_content(sequence: str, config: GenomeConfig) -> str:
    """
    Create GC content variation (gene-dense vs gene-poor regions).
    
    Args:
        sequence: Base sequence
        config: Genome configuration
    
    Returns:
        Sequence with GC variation
    """
    seq_list = list(sequence)
    length = len(sequence)
    
    # Determine gene-dense regions (first 30% of genome)
    gene_dense_end = int(length * config.gene_dense_fraction)
    
    # Adjust GC in gene-dense region
    for i in range(gene_dense_end):
        if seq_list[i] in ['A', 'T']:
            # Convert some AT to GC
            if random.random() < (config.gene_dense_gc - config.gc_content):
                seq_list[i] = random.choice(['G', 'C'])
        elif seq_list[i] in ['G', 'C']:
            # Convert some GC to AT if GC is too high
            if random.random() < (config.gc_content - config.gene_dense_gc):
                seq_list[i] = random.choice(['A', 'T'])
    
    # Adjust GC in gene-poor region
    for i in range(gene_dense_end, length):
        if seq_list[i] in ['G', 'C']:
            # Convert some GC to AT
            if random.random() < (config.gc_content - config.gene_poor_gc):
                seq_list[i] = random.choice(['A', 'T'])
    
    return ''.join(seq_list)


# ============================================================================
#                     HAPLOID GENOME GENERATION
# ============================================================================

def generate_haploid_genome(config: GenomeConfig) -> str:
    """
    Generate synthetic haploid reference genome.
    
    Creates a realistic genome with:
    - Base GC content
    - Tandem and interspersed repeats
    - Centromeric arrays
    - GC content variation (gene-dense vs gene-poor)
    
    Args:
        config: Genome configuration
    
    Returns:
        Haploid reference sequence
    """
    logger.info(f"Generating haploid genome: {config.length:,} bp")
    
    if config.random_seed is not None:
        random.seed(config.random_seed)
    
    # Generate base sequence
    logger.info(f"  Generating base sequence (GC={config.gc_content:.2%})...")
    sequence = generate_random_sequence(config.length, config.gc_content, config.random_seed)
    
    # Insert repeats
    logger.info(f"  Inserting repeats (density={config.repeat_density:.2%})...")
    sequence = insert_repeats(sequence, config)
    
    # Insert centromeres
    if config.num_centromeres > 0:
        logger.info(f"  Inserting {config.num_centromeres} centromere(s)...")
        sequence = insert_centromeres(sequence, config)
    
    # Modulate GC content
    logger.info(f"  Modulating GC content (gene regions)...")
    sequence = modulate_gc_content(sequence, config)
    
    logger.info(f"Haploid genome complete: {len(sequence):,} bp")
    
    return sequence


# ============================================================================
#                   DIPLOID VARIATION GENERATION
# ============================================================================

def introduce_snps(sequence: str, snp_rate: float) -> Tuple[str, List[int]]:
    """
    Introduce SNPs into a sequence.
    
    Args:
        sequence: Input sequence
        snp_rate: Probability of SNP per base
    
    Returns:
        (modified_sequence, list of SNP positions)
    """
    seq_list = list(sequence)
    snp_positions = []
    
    for i in range(len(seq_list)):
        if random.random() < snp_rate:
            # Mutate to different base
            bases = ['A', 'C', 'G', 'T']
            bases.remove(seq_list[i])
            seq_list[i] = random.choice(bases)
            snp_positions.append(i)
    
    return ''.join(seq_list), snp_positions


def introduce_indels(sequence: str, config: GenomeConfig) -> Tuple[str, List[int]]:
    """
    Introduce small indels into a sequence.
    
    Args:
        sequence: Input sequence
        config: Genome configuration
    
    Returns:
        (modified_sequence, list of indel positions)
    """
    indel_positions = []
    length = len(sequence)
    num_indels = int(length * config.indel_rate)
    
    # Get random positions for indels
    positions = sorted(random.sample(range(length), min(num_indels, length)))
    
    # Build modified sequence
    result = []
    last_pos = 0
    
    for pos in positions:
        # Copy up to this position
        result.append(sequence[last_pos:pos])
        
        # Determine indel size
        indel_size = max(1, int(random.gauss(config.indel_length_mean, config.indel_length_std)))
        
        # 50% insertion, 50% deletion
        if random.random() < 0.5:
            # Insertion
            insert_seq = generate_random_sequence(indel_size, config.gc_content)
            result.append(insert_seq)
            indel_positions.append(pos)
        else:
            # Deletion - skip bases
            last_pos = min(pos + indel_size, length)
            indel_positions.append(pos)
            continue
        
        last_pos = pos
    
    # Append remainder
    result.append(sequence[last_pos:])
    
    return ''.join(result), indel_positions


def introduce_sv_deletion(sequence: str, pos: int, size: int) -> Tuple[str, StructuralVariant]:
    """
    Introduce deletion SV.
    
    Args:
        sequence: Input sequence
        pos: Position of deletion
        size: Size of deletion
    
    Returns:
        (modified_sequence, SV object)
    """
    end = min(pos + size, len(sequence))
    actual_size = end - pos
    
    modified = sequence[:pos] + sequence[end:]
    
    sv = StructuralVariant(
        sv_type=SVType.DELETION,
        haplotype='',  # Set by caller
        chrom='chr1',
        pos=pos,
        end=end,
        size=actual_size,
        description=f"{actual_size}bp deletion at position {pos}"
    )
    
    return modified, sv


def introduce_sv_insertion(sequence: str, pos: int, size: int, gc_content: float) -> Tuple[str, StructuralVariant]:
    """
    Introduce insertion SV.
    
    Args:
        sequence: Input sequence
        pos: Position of insertion
        size: Size of insertion
        gc_content: GC content for inserted sequence
    
    Returns:
        (modified_sequence, SV object)
    """
    insert_seq = generate_random_sequence(size, gc_content)
    
    modified = sequence[:pos] + insert_seq + sequence[pos:]
    
    sv = StructuralVariant(
        sv_type=SVType.INSERTION,
        haplotype='',
        chrom='chr1',
        pos=pos,
        end=pos,
        size=size,
        sequence=insert_seq,
        description=f"{size}bp insertion at position {pos}"
    )
    
    return modified, sv


def introduce_sv_inversion(sequence: str, pos: int, size: int) -> Tuple[str, StructuralVariant]:
    """
    Introduce inversion SV.
    
    Args:
        sequence: Input sequence
        pos: Position of inversion
        size: Size of inversion
    
    Returns:
        (modified_sequence, SV object)
    """
    end = min(pos + size, len(sequence))
    actual_size = end - pos
    
    # Reverse complement the inverted region
    inverted = sequence[pos:end][::-1]
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    inverted = ''.join(complement.get(b, b) for b in inverted)
    
    modified = sequence[:pos] + inverted + sequence[end:]
    
    sv = StructuralVariant(
        sv_type=SVType.INVERSION,
        haplotype='',
        chrom='chr1',
        pos=pos,
        end=end,
        size=actual_size,
        description=f"{actual_size}bp inversion at position {pos}"
    )
    
    return modified, sv


def introduce_sv_duplication(sequence: str, pos: int, size: int) -> Tuple[str, StructuralVariant]:
    """
    Introduce tandem duplication SV.
    
    Args:
        sequence: Input sequence
        pos: Position of duplication
        size: Size of duplicated region
    
    Returns:
        (modified_sequence, SV object)
    """
    end = min(pos + size, len(sequence))
    actual_size = end - pos
    
    # Duplicate the region
    dup_seq = sequence[pos:end]
    modified = sequence[:end] + dup_seq + sequence[end:]
    
    sv = StructuralVariant(
        sv_type=SVType.DUPLICATION,
        haplotype='',
        chrom='chr1',
        pos=pos,
        end=end,
        size=actual_size,
        sequence=dup_seq,
        description=f"{actual_size}bp tandem duplication at position {pos}"
    )
    
    return modified, sv


def introduce_structural_variants(sequence: str, config: GenomeConfig, haplotype: str) -> Tuple[str, List[StructuralVariant]]:
    """
    Introduce structural variants into a haplotype.
    
    Args:
        sequence: Input sequence
        config: Genome configuration
        haplotype: Haplotype name ('A' or 'B')
    
    Returns:
        (modified_sequence, list of SVs)
    """
    svs = []
    length = len(sequence)
    num_svs = int(length * config.sv_density)
    
    logger.info(f"  Introducing {num_svs} structural variants in haplotype {haplotype}...")
    
    # Determine SV types based on fractions
    sv_types = (
        [SVType.DELETION] * int(num_svs * config.sv_deletion_fraction) +
        [SVType.INSERTION] * int(num_svs * config.sv_insertion_fraction) +
        [SVType.INVERSION] * int(num_svs * config.sv_inversion_fraction) +
        [SVType.DUPLICATION] * int(num_svs * config.sv_duplication_fraction) +
        [SVType.TRANSLOCATION] * int(num_svs * config.sv_translocation_fraction)
    )
    
    # Pad to exact number
    while len(sv_types) < num_svs:
        sv_types.append(random.choice([SVType.DELETION, SVType.INSERTION]))
    
    random.shuffle(sv_types)
    
    # Introduce each SV
    for sv_type in sv_types:
        # Random position (avoid ends)
        pos = random.randint(1000, length - config.sv_max_size - 1000)
        
        # Random size
        size = random.randint(config.sv_min_size, config.sv_max_size)
        
        # Apply SV
        if sv_type == SVType.DELETION:
            sequence, sv = introduce_sv_deletion(sequence, pos, size)
        elif sv_type == SVType.INSERTION:
            sequence, sv = introduce_sv_insertion(sequence, pos, size, config.gc_content)
        elif sv_type == SVType.INVERSION:
            sequence, sv = introduce_sv_inversion(sequence, pos, size)
        elif sv_type == SVType.DUPLICATION:
            sequence, sv = introduce_sv_duplication(sequence, pos, size)
        else:  # TRANSLOCATION - simplified as insertion
            sequence, sv = introduce_sv_insertion(sequence, pos, size, config.gc_content)
            sv.sv_type = SVType.TRANSLOCATION
            sv.description = f"{size}bp translocation at position {pos}"
        
        sv.haplotype = haplotype
        svs.append(sv)
        
        # Update length after modification
        length = len(sequence)
    
    logger.info(f"  Haplotype {haplotype} SVs: {len(svs)} variants")
    
    return sequence, svs


# ============================================================================
#                     DIPLOID GENOME GENERATION
# ============================================================================

def generate_diploid_genome(config: GenomeConfig) -> DiploidGenome:
    """
    Generate synthetic diploid genome with ground-truth labels.
    
    Creates two haplotypes with:
    - SNPs at specified rate
    - Small indels
    - Structural variants (deletions, insertions, inversions, etc.)
    
    Args:
        config: Genome configuration
    
    Returns:
        DiploidGenome object with both haplotypes and SV ground truth
    """
    logger.info("="*80)
    logger.info("Generating diploid genome")
    logger.info("="*80)
    
    # Generate haploid reference
    reference = generate_haploid_genome(config)
    
    # Create haplotype A (start from reference)
    logger.info("\nCreating haplotype A...")
    hapA = reference
    
    # Introduce SNPs
    logger.info(f"  Introducing SNPs (rate={config.snp_rate})...")
    hapA, snps_A = introduce_snps(hapA, config.snp_rate)
    logger.info(f"  Introduced {len(snps_A)} SNPs")
    
    # Introduce indels
    logger.info(f"  Introducing indels (rate={config.indel_rate})...")
    hapA, indels_A = introduce_indels(hapA, config)
    logger.info(f"  Introduced {len(indels_A)} indels")
    
    # Introduce SVs
    hapA, svs_A = introduce_structural_variants(hapA, config, 'A')
    
    # Create haplotype B (start from reference)
    logger.info("\nCreating haplotype B...")
    hapB = reference
    
    # Introduce SNPs (different positions)
    logger.info(f"  Introducing SNPs (rate={config.snp_rate})...")
    hapB, snps_B = introduce_snps(hapB, config.snp_rate)
    logger.info(f"  Introduced {len(snps_B)} SNPs")
    
    # Introduce indels
    logger.info(f"  Introducing indels (rate={config.indel_rate})...")
    hapB, indels_B = introduce_indels(hapB, config)
    logger.info(f"  Introduced {len(indels_B)} indels")
    
    # Introduce SVs
    hapB, svs_B = introduce_structural_variants(hapB, config, 'B')
    
    # Combine SV truth tables
    all_svs = svs_A + svs_B
    
    # Create diploid genome
    diploid = DiploidGenome(
        hapA=hapA,
        hapB=hapB,
        sv_truth_table=all_svs,
        snp_positions=snps_A + snps_B,
        indel_positions=indels_A + indels_B,
        reference=reference,
        config=config
    )
    
    logger.info("\n" + "="*80)
    logger.info("Diploid genome generation complete")
    logger.info("="*80)
    logger.info(f"Reference length: {len(reference):,} bp")
    logger.info(f"Haplotype A length: {diploid.length_A:,} bp")
    logger.info(f"Haplotype B length: {diploid.length_B:,} bp")
    logger.info(f"Total SNPs: {len(diploid.snp_positions)}")
    logger.info(f"Total indels: {len(diploid.indel_positions)}")
    logger.info(f"Total SVs: {diploid.num_svs}")
    logger.info(f"  Deletions: {len(diploid.get_svs_by_type(SVType.DELETION))}")
    logger.info(f"  Insertions: {len(diploid.get_svs_by_type(SVType.INSERTION))}")
    logger.info(f"  Inversions: {len(diploid.get_svs_by_type(SVType.INVERSION))}")
    logger.info(f"  Duplications: {len(diploid.get_svs_by_type(SVType.DUPLICATION))}")
    logger.info(f"  Translocations: {len(diploid.get_svs_by_type(SVType.TRANSLOCATION))}")
    logger.info("="*80)
    
    return diploid


# ============================================================================
#                    PART 2: READ SIMULATION
# ============================================================================



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


# ============================================================================
#                    PART 3: GROUND-TRUTH LABELING
# ============================================================================



# ============================================================================
#                           LABEL TYPES
# ============================================================================

class EdgeLabel(Enum):
    """Ground-truth classification for graph edges."""
    TRUE = "true"                    # Legitimate overlap in reference
    REPEAT = "repeat"                # Both reads from same repeat region
    CHIMERIC = "chimeric"            # Reads from different loci/chromosomes
    ALLELIC = "allelic"              # Reads from homologous haplotype positions
    SV_BREAK = "sv_break"            # Edge crosses SV breakpoint
    UNKNOWN = "unknown"              # Cannot determine (shouldn't happen with synthetic)


class NodeHaplotype(Enum):
    """Haplotype assignment for graph nodes."""
    HAP_A = "A"                      # Node from haplotype A only
    HAP_B = "B"                      # Node from haplotype B only
    BOTH = "both"                    # Node shared by both haplotypes
    REPEAT = "repeat"                # Node from repeat region (ambiguous)
    UNKNOWN = "unknown"              # Cannot determine


class SVType(Enum):
    """Structural variant types (matches genome_simulator.py)."""
    DELETION = "deletion"
    INSERTION = "insertion"
    INVERSION = "inversion"
    DUPLICATION = "duplication"
    TRANSLOCATION = "translocation"


# ============================================================================
#                           DATA STRUCTURES
# ============================================================================

@dataclass
class ReadAlignment:
    """
    Alignment of a simulated read to reference genome.
    
    Attributes:
        read_id: Read identifier
        ref_chrom: Reference chromosome
        ref_start: Start position on reference
        ref_end: End position on reference
        haplotype: Source haplotype ('A' or 'B')
        strand: Alignment strand ('+' or '-')
        identity: Alignment identity (0-1)
        is_repeat: Whether read originates from repeat region
    """
    read_id: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    haplotype: str
    strand: str = '+'
    identity: float = 1.0
    is_repeat: bool = False


@dataclass
class EdgeGroundTruth:
    """
    Ground-truth label for a graph edge.
    
    Attributes:
        source_node: Source node ID
        target_node: Target node ID
        label: Edge classification
        explanation: Human-readable explanation
        read1_pos: Position of read 1 on reference
        read2_pos: Position of read 2 on reference
        overlap_distance: True distance between reads on reference
        crosses_sv: Whether edge crosses an SV breakpoint
        sv_type: Type of SV if crosses_sv is True
    """
    source_node: str
    target_node: str
    label: EdgeLabel
    explanation: str
    read1_pos: Optional[Tuple[str, int, int]] = None  # (chrom, start, end)
    read2_pos: Optional[Tuple[str, int, int]] = None
    overlap_distance: Optional[int] = None
    crosses_sv: bool = False
    sv_type: Optional[SVType] = None


@dataclass
class NodeGroundTruth:
    """
    Ground-truth label for a graph node.
    
    Attributes:
        node_id: Node identifier
        haplotype: Haplotype assignment
        ref_positions: List of reference positions this node maps to
        is_repeat: Whether node comes from repeat region
        spanning_reads: List of read IDs that contain this node
        sv_association: SV this node is associated with (if any)
    """
    node_id: str
    haplotype: NodeHaplotype
    ref_positions: List[Tuple[str, int, int]] = field(default_factory=list)
    is_repeat: bool = False
    spanning_reads: List[str] = field(default_factory=list)
    sv_association: Optional[Dict] = None


@dataclass
class PathGroundTruth:
    """
    Ground-truth path through the graph (for GNN training).
    
    Attributes:
        path_id: Path identifier
        node_sequence: Ordered list of node IDs
        haplotype: Which haplotype this path represents
        ref_chrom: Reference chromosome
        ref_start: Start position on reference
        ref_end: End position on reference
        is_correct: Whether this is a correct path
        confidence: Confidence score (0-1)
    """
    path_id: str
    node_sequence: List[str]
    haplotype: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    is_correct: bool = True
    confidence: float = 1.0


@dataclass
class ULRouteGroundTruth:
    """
    Ground-truth routing for ultralong reads.
    
    Attributes:
        read_id: UL read identifier
        correct_path: Correct node sequence for this read
        alternative_paths: Incorrect alternative paths
        ref_positions: Reference positions spanned by this read
        num_nodes: Number of nodes in correct path
        path_length: Total path length in bp
    """
    read_id: str
    correct_path: List[str]
    alternative_paths: List[List[str]] = field(default_factory=list)
    ref_positions: Tuple[str, int, int] = ("chr1", 0, 0)
    num_nodes: int = 0
    path_length: int = 0


@dataclass
class SVGroundTruth:
    """
    Ground-truth structural variant annotation.
    
    Attributes:
        sv_id: SV identifier
        sv_type: Type of structural variant
        haplotype: Which haplotype contains this SV
        ref_chrom: Reference chromosome
        ref_start: Start position
        ref_end: End position
        size: SV size in bp
        graph_signature: Graph features associated with this SV
    """
    sv_id: str
    sv_type: SVType
    haplotype: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    size: int
    graph_signature: Dict = field(default_factory=dict)


# ============================================================================
#                    READ ALIGNMENT TO REFERENCE
# ============================================================================

def align_simulated_reads_to_reference(
    reads: List,  # List[SimulatedRead]
    reference_hapA: str,
    reference_hapB: str,
    min_identity: float = 0.90
) -> Dict[str, ReadAlignment]:
    """
    Align simulated reads back to their source reference genomes.
    
    Since reads are synthetic, we already know their true positions from
    SimulatedRead.start_pos and SimulatedRead.end_pos. This function just
    creates ReadAlignment objects with that information.
    
    Args:
        reads: List of SimulatedRead objects
        reference_hapA: Haplotype A reference sequence
        reference_hapB: Haplotype B reference sequence
        min_identity: Minimum alignment identity (not used for synthetic)
    
    Returns:
        Dictionary mapping read_id -> ReadAlignment
    """
    logger.info(f"Aligning {len(reads)} simulated reads to reference genomes...")
    
    alignments = {}
    
    for read in reads:
        # Simulated reads already have ground truth positions
        alignment = ReadAlignment(
            read_id=read.read_id,
            ref_chrom=read.chrom,
            ref_start=read.start_pos,
            ref_end=read.end_pos,
            haplotype=read.haplotype if hasattr(read, 'haplotype') else 'A',
            strand=read.strand if hasattr(read, 'strand') else '+',
            identity=1.0,  # Perfect alignment for synthetic reads
            is_repeat=False  # Will be determined later
        )
        
        alignments[read.read_id] = alignment
    
    logger.info(f"Created {len(alignments)} read alignments")
    return alignments


# ============================================================================
#                    EDGE LABELING
# ============================================================================

def label_graph_edge(
    source_node_id: str,
    target_node_id: str,
    alignments: Dict[str, ReadAlignment],
    sv_truth_table: List,
    node_to_read_ids: Optional[Dict[str, List[str]]] = None,
    max_true_distance: int = 10000,
    repeat_threshold: int = 3
) -> EdgeGroundTruth:
    """
    Determine ground-truth label for a graph edge.
    
    Classification logic:
    1. TRUE: Both reads from same haplotype, overlapping/adjacent positions
    2. ALLELIC: Reads from different haplotypes at homologous positions
    3. REPEAT: Both reads map to known repeat regions
    4. SV_BREAK: Edge crosses an SV breakpoint
    5. CHIMERIC: Reads from distant loci (shouldn't happen in good assembly)
    
    Args:
        source_read_id: Source read identifier
        target_read_id: Target read identifier
        alignments: Dictionary of read alignments
        sv_truth_table: List of StructuralVariant objects
        max_true_distance: Max distance for "true" overlap (bp)
        repeat_threshold: Number of alignments to consider "repeat"
    
    Returns:
        EdgeGroundTruth object with classification
    """
    # Get alignments - try exact match first, then fuzzy match
    # Try direct read-id lookup, but nodes may be unitigs. Prefer node_to_read_ids mapping.
    aln1 = None
    aln2 = None
    
    # Fuzzy matching: node IDs might have suffixes like "_L", "_R", etc.
    # Resolve via mapping first
    if node_to_read_ids:
        for rid in node_to_read_ids.get(source_node_id, []):
            if rid in alignments:
                aln1 = alignments[rid]
                break
    
    if node_to_read_ids:
        for rid in node_to_read_ids.get(target_node_id, []):
            if rid in alignments:
                aln2 = alignments[rid]
                break

    # If still missing, try resolving via node_to_read_ids mapping
    # Fallback fuzzy match on alignment keys
    if not aln1:
        for read_id in alignments.keys():
            if read_id in source_node_id or source_node_id in read_id:
                aln1 = alignments[read_id]
                break
    if not aln2:
        for read_id in alignments.keys():
            if read_id in target_node_id or target_node_id in read_id:
                aln2 = alignments[read_id]
                break
    
    if not aln1 or not aln2:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.UNKNOWN,
            explanation="Missing alignment data for node-derived IDs"
        )
    
    # Same chromosome?
    if aln1.ref_chrom != aln2.ref_chrom:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.CHIMERIC,
            explanation=f"Reads from different chromosomes: {aln1.ref_chrom} vs {aln2.ref_chrom}",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end)
        )
    
    # Calculate distance between reads
    if aln1.ref_start <= aln2.ref_start:
        distance = aln2.ref_start - aln1.ref_end
    else:
        distance = aln1.ref_start - aln2.ref_end
    
    # Check if edge crosses SV breakpoint
    crosses_sv = False
    sv_type = None
    for sv in sv_truth_table:
        if sv.chrom != aln1.ref_chrom:
            continue
        
        # Check if edge spans SV breakpoint
        sv_start = sv.pos
        sv_end = sv.end
        
        read1_span = (aln1.ref_start, aln1.ref_end)
        read2_span = (aln2.ref_start, aln2.ref_end)
        
        # Does edge cross this SV?
        if (read1_span[0] < sv_start < read2_span[1] or
            read2_span[0] < sv_start < read1_span[1] or
            read1_span[0] < sv_end < read2_span[1] or
            read2_span[0] < sv_end < read1_span[1]):
            crosses_sv = True
            sv_type = SVType(sv.sv_type.value)
            break
    
    if crosses_sv:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.SV_BREAK,
            explanation=f"Edge crosses {sv_type.value} SV breakpoint",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance,
            crosses_sv=True,
            sv_type=sv_type
        )
    
    # Check if both reads are repeats
    if aln1.is_repeat and aln2.is_repeat:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.REPEAT,
            explanation="Both reads from repeat regions",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance
        )
    
    # Different haplotypes?
    if aln1.haplotype != aln2.haplotype:
        # Check if reads are at homologous positions (similar coordinates)
        if abs(aln1.ref_start - aln2.ref_start) < max_true_distance:
            return EdgeGroundTruth(
                source_node=source_node_id,
                target_node=target_node_id,
                label=EdgeLabel.ALLELIC,
                explanation=f"Reads from different haplotypes ({aln1.haplotype} vs {aln2.haplotype}) at homologous positions",
                read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
                read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
                overlap_distance=distance
            )
        else:
            return EdgeGroundTruth(
                source_node=source_node_id,
                target_node=target_node_id,
                label=EdgeLabel.CHIMERIC,
                explanation=f"Reads from different haplotypes at distant positions",
                read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
                read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
                overlap_distance=distance
            )
    
    # Same haplotype, close distance = TRUE overlap
    if abs(distance) <= max_true_distance:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.TRUE,
            explanation=f"Legitimate overlap (distance={distance}bp, same haplotype)",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance
        )
    
    # Same haplotype but distant = chimeric (shouldn't happen often)
    return EdgeGroundTruth(
        source_node=source_node_id,
        target_node=target_node_id,
        label=EdgeLabel.CHIMERIC,
        explanation=f"Same haplotype but distant loci (distance={distance}bp)",
        read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
        read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
        overlap_distance=distance
    )


def label_all_graph_edges(
    edges: List[Tuple[str, str]],
    alignments: Dict[str, ReadAlignment],
    sv_truth_table: List,
    node_to_read_ids: Optional[Dict[str, List[str]]] = None
) -> Dict[Tuple[str, str], EdgeGroundTruth]:
    """
    Label all edges in assembly graph.
    
    Args:
        edges: List of (source_id, target_id) tuples
        alignments: Dictionary of read alignments
        sv_truth_table: List of structural variants
    
    Returns:
        Dictionary mapping (source, target) -> EdgeGroundTruth
    """
    logger.info(f"Labeling {len(edges)} graph edges...")
    logger.info(f"Available alignments: {len(alignments)} reads")
    
    edge_labels = {}
    label_counts = {label: 0 for label in EdgeLabel}
    missing_count = 0
    
    # Debug: Check first few edges
    if edges:
        sample_edges = edges[:min(3, len(edges))]
        sample_alns = list(alignments.keys())[:min(3, len(alignments))]
        logger.info(f"Sample edge IDs: {sample_edges}")
        logger.info(f"Sample alignment IDs: {sample_alns}")
    
    for source, target in edges:
        label = label_graph_edge(source, target, alignments, sv_truth_table, node_to_read_ids)
        edge_labels[(source, target)] = label
        label_counts[label.label] += 1
        
        if label.label == EdgeLabel.UNKNOWN:
            missing_count += 1
    
    logger.info("Edge labeling complete:")
    for label_type, count in label_counts.items():
        if count > 0:
            logger.info(f"  {label_type.value}: {count} edges")
    
    if missing_count > 0:
        logger.warning(f"WARNING: {missing_count}/{len(edges)} edges labeled as UNKNOWN (missing alignment data)")
        logger.warning("This indicates edge node IDs don't match read IDs in alignment dictionary")
    
    return edge_labels


# ============================================================================
#                    NODE HAPLOTYPE LABELING
# ============================================================================

def label_node_haplotype(
    node_id: str,
    node_sequence: str,
    reference_hapA: str,
    reference_hapB: str,
    alignments: Dict[str, ReadAlignment],
    min_identity: float = 0.95
) -> NodeGroundTruth:
    """
    Determine haplotype assignment for a graph node.
    
    Strategy:
    1. Find all reads that span this node
    2. Check haplotype assignments of those reads
    3. If all reads from haplotype A → HAP_A
    4. If all reads from haplotype B → HAP_B
    5. If reads from both haplotypes → BOTH (shared sequence)
    6. If reads map to multiple locations → REPEAT
    
    Args:
        node_id: Node identifier
        node_sequence: Node sequence
        reference_hapA: Haplotype A reference
        reference_hapB: Haplotype B reference
        alignments: Read alignments
        min_identity: Minimum identity for alignment
    
    Returns:
        NodeGroundTruth object
    """
    # Find reads that contain this node
    # In assembly graphs, nodes are often derived from reads, so node_id matches/contains read_id
    spanning_reads = []
    haplotypes_seen = set()
    
    # Check which reads contributed to this node
    for read_id, aln in alignments.items():
        # Node ID typically contains the read ID (e.g., "read_123" or "read_123_L" or "read_123_R")
        if read_id in node_id or node_id in read_id or read_id == node_id:
            spanning_reads.append(read_id)
            haplotypes_seen.add(aln.haplotype)
        # Also check if node sequence matches read sequence (if node_sequence provided)
        elif node_sequence and len(node_sequence) > 50:
            # For longer sequences, check substring match
            # This catches k-mer based nodes that don't have read IDs
            for other_read_id, other_aln in alignments.items():
                if node_sequence in other_aln.read_sequence if hasattr(other_aln, 'read_sequence') else False:
                    spanning_reads.append(other_read_id)
                    haplotypes_seen.add(other_aln.haplotype)
                    break
    
    # Determine haplotype
    if len(haplotypes_seen) == 0:
        haplotype = NodeHaplotype.UNKNOWN
    elif len(haplotypes_seen) == 1:
        hap = list(haplotypes_seen)[0]
        haplotype = NodeHaplotype.HAP_A if hap == 'A' else NodeHaplotype.HAP_B
    else:
        # Reads from both haplotypes
        haplotype = NodeHaplotype.BOTH
    
    return NodeGroundTruth(
        node_id=node_id,
        haplotype=haplotype,
        spanning_reads=spanning_reads,
        is_repeat=False  # Would check against repeat annotations
    )


# ============================================================================
#                    PATH LABELING (GNN)
# ============================================================================

def extract_correct_paths(
    alignments: Dict[str, ReadAlignment],
    reference_hapA: str,
    reference_hapB: str
) -> List[PathGroundTruth]:
    """
    Extract correct paths through the graph for GNN training.
    
    Strategy:
    1. Sort reads by reference position (per haplotype)
    2. Reads in order = correct path
    3. Create path for each contig/chromosome
    
    Args:
        alignments: Read alignments
        reference_hapA: Haplotype A reference
        reference_hapB: Haplotype B reference
    
    Returns:
        List of PathGroundTruth objects
    """
    logger.info("Extracting correct paths for GNN training...")
    
    paths = []
    
    # Group reads by haplotype and chromosome
    by_haplotype = {}
    for read_id, aln in alignments.items():
        key = (aln.haplotype, aln.ref_chrom)
        if key not in by_haplotype:
            by_haplotype[key] = []
        by_haplotype[key].append((read_id, aln))
    
    # Create paths for each haplotype/chromosome
    for (haplotype, chrom), reads in by_haplotype.items():
        # Sort by start position
        reads.sort(key=lambda x: x[1].ref_start)
        
        # Create path
        node_sequence = [read_id for read_id, _ in reads]
        
        if len(node_sequence) > 0:
            first_aln = reads[0][1]
            last_aln = reads[-1][1]
            
            path = PathGroundTruth(
                path_id=f"path_{haplotype}_{chrom}",
                node_sequence=node_sequence,
                haplotype=haplotype,
                ref_chrom=chrom,
                ref_start=first_aln.ref_start,
                ref_end=last_aln.ref_end,
                is_correct=True,
                confidence=1.0
            )
            paths.append(path)
    
    logger.info(f"Extracted {len(paths)} correct paths")
    return paths


# ============================================================================
#                    UL ROUTING LABELING
# ============================================================================

def label_ul_read_routes(
    ul_reads: List,  # List[SimulatedRead] for UL reads
    alignments: Dict[str, ReadAlignment],
    graph_nodes: List[str]
) -> List[ULRouteGroundTruth]:
    """
    Determine correct routing for ultralong reads through graph.
    
    UL reads span multiple graph nodes. This function determines the
    correct node sequence each UL read should traverse.
    
    Args:
        ul_reads: List of ultralong SimulatedRead objects
        alignments: Read alignments
        graph_nodes: List of node IDs in graph
    
    Returns:
        List of ULRouteGroundTruth objects
    """
    logger.info(f"Labeling routes for {len(ul_reads)} ultralong reads...")
    
    routes = []
    
    for ul_read in ul_reads:
        if ul_read.read_id not in alignments:
            continue
        
        aln = alignments[ul_read.read_id]
        
        # Find all nodes that overlap with this UL read's span
        # Strategy: Find all reads whose reference positions overlap with this UL read
        # then find nodes derived from those reads
        correct_path = []
        
        ul_start = aln.ref_start
        ul_end = aln.ref_end
        ul_chrom = aln.ref_chrom
        ul_hap = aln.haplotype
        
        # Find all reads that overlap this UL read's span (same haplotype)
        overlapping_reads = []
        for other_read_id, other_aln in alignments.items():
            if other_aln.haplotype != ul_hap or other_aln.ref_chrom != ul_chrom:
                continue
            
            # Check for overlap
            if not (other_aln.ref_end < ul_start or other_aln.ref_start > ul_end):
                overlapping_reads.append((other_read_id, other_aln.ref_start))
        
        # Sort by reference position to get correct path order
        overlapping_reads.sort(key=lambda x: x[1])
        
        # Find nodes derived from these reads
        for read_id, _ in overlapping_reads:
            # Look for nodes that match this read
            for node_id in graph_nodes:
                if read_id in node_id or node_id in read_id or read_id == node_id:
                    if node_id not in correct_path:  # Avoid duplicates
                        correct_path.append(node_id)
        
        # Fallback: if no overlapping nodes found, use the UL read itself as single node
        if not correct_path:
            correct_path = [ul_read.read_id]
        
        route = ULRouteGroundTruth(
            read_id=ul_read.read_id,
            correct_path=correct_path,
            ref_positions=(aln.ref_chrom, aln.ref_start, aln.ref_end),
            num_nodes=len(correct_path),
            path_length=aln.ref_end - aln.ref_start
        )
        routes.append(route)
    
    logger.info(f"Labeled {len(routes)} UL routes")
    return routes


# ============================================================================
#                    SV LABELING
# ============================================================================

def label_sv_graph_signatures(
    sv_truth_table: List,
    alignments: Dict[str, ReadAlignment],
    graph_edges: List[Tuple[str, str]]
) -> List[SVGroundTruth]:
    """
    Associate graph features with true structural variants.
    
    For each SV, identify the graph signature:
    - Deletions: Coverage drops, missing edges
    - Insertions: Coverage spikes, graph bubbles
    - Inversions: Complex branching, reversed edges
    - Duplications: High coverage regions
    - Translocations: Long-distance edges
    
    Args:
        sv_truth_table: List of StructuralVariant objects
        alignments: Read alignments
        graph_edges: List of graph edges
    
    Returns:
        List of SVGroundTruth objects
    """
    logger.info(f"Labeling graph signatures for {len(sv_truth_table)} SVs...")
    
    sv_labels = []
    
    for i, sv in enumerate(sv_truth_table):
        # Find reads that span this SV
        spanning_reads = []
        for read_id, aln in alignments.items():
            if (aln.ref_chrom == sv.chrom and
                aln.haplotype == sv.haplotype and
                aln.ref_start <= sv.pos <= aln.ref_end):
                spanning_reads.append(read_id)
        
        # Determine graph signature
        signature = {
            'spanning_reads': spanning_reads,
            'num_spanning': len(spanning_reads),
            'sv_size': sv.size
        }
        
        # Type-specific signatures
        if sv.sv_type.value == 'deletion':
            signature['expected_pattern'] = 'coverage_drop'
        elif sv.sv_type.value == 'insertion':
            signature['expected_pattern'] = 'bubble_or_spike'
        elif sv.sv_type.value == 'inversion':
            signature['expected_pattern'] = 'complex_branching'
        elif sv.sv_type.value == 'duplication':
            signature['expected_pattern'] = 'high_coverage'
        elif sv.sv_type.value == 'translocation':
            signature['expected_pattern'] = 'long_distance_edge'
        
        sv_label = SVGroundTruth(
            sv_id=f"sv_{i}",
            sv_type=SVType(sv.sv_type.value),
            haplotype=sv.haplotype,
            ref_chrom=sv.chrom,
            ref_start=sv.pos,
            ref_end=sv.end,
            size=sv.size,
            graph_signature=signature
        )
        sv_labels.append(sv_label)
    
    logger.info(f"Labeled {len(sv_labels)} SV graph signatures")
    return sv_labels


# ============================================================================
#                    MAIN LABELING PIPELINE
# ============================================================================

@dataclass
class GroundTruthLabels:
    """Complete set of ground-truth labels for ML training."""
    edge_labels: Dict[Tuple[str, str], EdgeGroundTruth]
    node_labels: Dict[str, NodeGroundTruth]
    path_labels: List[PathGroundTruth]
    ul_route_labels: List[ULRouteGroundTruth]
    sv_labels: List[SVGroundTruth]


def generate_ground_truth_labels(
    simulated_reads: List,  # List[SimulatedRead]
    ul_reads: List,  # List[SimulatedRead] for UL
    reference_hapA: str,
    reference_hapB: str,
    sv_truth_table: List,  # List[StructuralVariant]
    graph_edges: List[Tuple[str, str]],
    graph_nodes: List[str],
    node_to_read_ids: Optional[Dict[str, List[str]]] = None
) -> GroundTruthLabels:
    """
    Generate complete ground-truth labels for ML training.
    
    This is the main entry point that orchestrates all labeling operations.
    
    Args:
        simulated_reads: All simulated reads
        ul_reads: Ultralong reads subset
        reference_hapA: Haplotype A reference sequence
        reference_hapB: Haplotype B reference sequence
        sv_truth_table: Structural variant truth table
        graph_edges: List of graph edges (source, target)
        graph_nodes: List of graph node IDs
        node_to_read_ids: Optional mapping of node_id -> contributing read_ids
    
    Returns:
        GroundTruthLabels object with all labels
    """
    logger.info("=" * 80)
    logger.info("Generating ground-truth labels for ML training")
    logger.info("=" * 80)
    
    # Step 1: Align reads to reference
    alignments = align_simulated_reads_to_reference(
        simulated_reads, reference_hapA, reference_hapB
    )
    
    # Use provided node_to_read_ids if available; otherwise build from node IDs
    if not node_to_read_ids:
        node_to_read_ids = {}
        for node_id in graph_nodes:
            node_to_read_ids[node_id] = []
            for read_id in alignments.keys():
                if read_id in node_id or node_id in read_id or read_id == node_id:
                    node_to_read_ids[node_id].append(read_id)
    
    # Step 2: Label graph edges (use mapping for robust resolution)
    edge_labels = label_all_graph_edges(graph_edges, alignments, sv_truth_table, node_to_read_ids)    # Step 3: Label node haplotypes
    logger.info(f"Labeling {len(graph_nodes)} graph nodes...")
    
    # Build efficient lookup: node_id -> reads that contain it
    # Most nodes are named after reads (e.g., "read_123" node from "read_123" read)
    node_labels = {}
    
    for node_id in graph_nodes:
        spanning_reads = []
        haplotypes_seen = set()
        
        # Check if this node ID matches any read ID (common in assembly graphs)
        # Node IDs are typically: read_id, read_id_suffix, or kmer-based
        for read_id, aln in alignments.items():
            # Node contains this read, or vice versa
            if read_id in node_id or node_id in read_id or read_id == node_id:
                spanning_reads.append(read_id)
                haplotypes_seen.add(aln.haplotype)
        
        # Determine haplotype from spanning reads
        if len(haplotypes_seen) == 0:
            haplotype = NodeHaplotype.UNKNOWN
        elif len(haplotypes_seen) == 1:
            hap = list(haplotypes_seen)[0]
            haplotype = NodeHaplotype.HAP_A if hap == 'A' else NodeHaplotype.HAP_B
        else:
            # Reads from both haplotypes = shared node
            haplotype = NodeHaplotype.BOTH
        
        # Check if node is from repeat region
        is_repeat = any(alignments[rid].is_repeat for rid in spanning_reads if rid in alignments)
        if is_repeat:
            haplotype = NodeHaplotype.REPEAT
        
        node_labels[node_id] = NodeGroundTruth(
            node_id=node_id,
            haplotype=haplotype,
            spanning_reads=spanning_reads,
            is_repeat=is_repeat
        )
    
    logger.info(f"Labeled {len(node_labels)} nodes:")
    hap_counts = {'A': 0, 'B': 0, 'both': 0, 'repeat': 0, 'unknown': 0}
    for nl in node_labels.values():
        if nl.haplotype == NodeHaplotype.HAP_A:
            hap_counts['A'] += 1
        elif nl.haplotype == NodeHaplotype.HAP_B:
            hap_counts['B'] += 1
        else:
            hap_counts[nl.haplotype.value] += 1
    logger.info(f"  Hap A: {hap_counts.get('A', 0)}, Hap B: {hap_counts.get('B', 0)}, Both: {hap_counts['both']}, Repeat: {hap_counts['repeat']}, Unknown: {hap_counts['unknown']}")
    
    # Step 4: Extract correct paths
    path_labels = extract_correct_paths(alignments, reference_hapA, reference_hapB)
    
    # Step 5: Label UL routes
    ul_route_labels = label_ul_read_routes(ul_reads, alignments, graph_nodes)
    
    # Step 6: Label SV graph signatures
    sv_labels = label_sv_graph_signatures(sv_truth_table, alignments, graph_edges)
    
    logger.info("=" * 80)
    logger.info("Ground-truth labeling complete")
    logger.info("=" * 80)
    logger.info(f"Edge labels: {len(edge_labels)}")
    logger.info(f"Node labels: {len(node_labels)}")
    logger.info(f"Path labels: {len(path_labels)}")
    logger.info(f"UL route labels: {len(ul_route_labels)}")
    logger.info(f"SV labels: {len(sv_labels)}")
    
    return GroundTruthLabels(
        edge_labels=edge_labels,
        node_labels=node_labels,
        path_labels=path_labels,
        ul_route_labels=ul_route_labels,
        sv_labels=sv_labels
    )


# ============================================================================
#                    EXPORT FUNCTIONS
# ============================================================================

def export_labels_to_tsv(labels: GroundTruthLabels, output_dir: Path) -> None:
    """
    Export ground-truth labels to TSV files for inspection.
    
    Args:
        labels: GroundTruthLabels object
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export edge labels
    edge_file = output_dir / "edge_labels.tsv"
    with open(edge_file, 'w') as f:
        f.write("source\ttarget\tlabel\texplanation\tdistance\tcrosses_sv\n")
        for (source, target), label in labels.edge_labels.items():
            f.write(f"{source}\t{target}\t{label.label.value}\t{label.explanation}\t"
                   f"{label.overlap_distance}\t{label.crosses_sv}\n")
    
    # Export node labels
    node_file = output_dir / "node_labels.tsv"
    with open(node_file, 'w') as f:
        f.write("node_id\thaplotype\tis_repeat\tnum_spanning_reads\n")
        for node_id, label in labels.node_labels.items():
            f.write(f"{node_id}\t{label.haplotype.value}\t{label.is_repeat}\t"
                   f"{len(label.spanning_reads)}\n")
    
    # Export SV labels
    sv_file = output_dir / "sv_labels.tsv"
    with open(sv_file, 'w') as f:
        f.write("sv_id\tsv_type\thaplotype\tchrom\tstart\tend\tsize\tnum_spanning_reads\n")
        for label in labels.sv_labels:
            f.write(f"{label.sv_id}\t{label.sv_type.value}\t{label.haplotype}\t"
                   f"{label.ref_chrom}\t{label.ref_start}\t{label.ref_end}\t"
                   f"{label.size}\t{label.graph_signature.get('num_spanning', 0)}\n")
    
    logger.info(f"Exported labels to {output_dir}")
    logger.info(f"  {edge_file}")
    logger.info(f"  {node_file}")
    logger.info(f"  {sv_file}")
