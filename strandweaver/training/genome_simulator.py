"""
Synthetic Genome Simulator for Training Data Generation

This module generates realistic synthetic genomes with configurable features:
- Haploid reference genomes with realistic base composition
- Diploid genomes with SNPs, indels, and structural variants
- Repeat-rich regions (tandem repeats, segmental duplications)
- Centromeric arrays (alpha satellite-like repeats)
- Gene-dense vs gene-poor regions
- Structural variant ground truth

Used to generate training data for:
- GNN path prediction models
- Diploid disentanglement classifiers
- UL routing decision models
- SV detection models
- Overlap classification models

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


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
