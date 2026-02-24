#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Synthetic genome and read data generator for ML training.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations

import logging
import math
import random
import string
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
#                               ENUMERATIONS
# ============================================================================

class SVType(Enum):
    """Structural variant types."""
    DELETION = "deletion"
    INSERTION = "insertion"
    INVERSION = "inversion"
    DUPLICATION = "duplication"
    TRANSLOCATION = "translocation"


# ============================================================================
#                               DATA CLASSES
# ============================================================================

@dataclass
class StructuralVariant:
    """Ground-truth structural variant descriptor."""
    sv_type: SVType
    haplotype: str          # 'A' or 'B'
    chrom: str
    pos: int                # 0-based start
    end: int                # 0-based end
    size: int
    description: str = ""


@dataclass
class GenomeConfig:
    """Configuration for synthetic diploid genome generation."""
    length: int = 1_000_000
    gc_content: float = 0.42
    repeat_density: float = 0.30
    tandem_repeat_fraction: float = 0.6
    num_centromeres: int = 1
    gene_dense_fraction: float = 0.30
    snp_rate: float = 0.001
    indel_rate: float = 0.0001
    sv_density: float = 0.00001
    sv_deletion_fraction: float = 0.25
    sv_insertion_fraction: float = 0.25
    sv_inversion_fraction: float = 0.25
    sv_duplication_fraction: float = 0.25
    sv_translocation_fraction: float = 0.0
    random_seed: Optional[int] = None


@dataclass
class DiploidGenome:
    """Result of diploid genome generation."""
    hapA: str
    hapB: str
    snp_positions: List[int] = field(default_factory=list)
    indel_positions: List[int] = field(default_factory=list)
    sv_truth_table: List[StructuralVariant] = field(default_factory=list)
    repeat_mask_A: Optional[List[bool]] = None
    repeat_mask_B: Optional[List[bool]] = None

    @property
    def length_A(self) -> int:
        return len(self.hapA)

    @property
    def length_B(self) -> int:
        return len(self.hapB)


@dataclass
class SimulatedRead:
    """A single simulated sequencing read."""
    read_id: str
    sequence: str
    quality: str
    haplotype: str          # 'A' or 'B'
    chrom: str
    start_pos: int
    end_pos: int
    strand: str = '+'
    technology: str = 'hifi'


@dataclass
class SimulatedReadPair:
    """A pair of simulated reads (Illumina PE or Hi-C)."""
    read1: SimulatedRead
    read2: SimulatedRead

    @property
    def read_id(self) -> str:
        return self.read1.read_id.rsplit('_R1', 1)[0]

    # Expose attributes for reads_to_read_infos compatibility
    @property
    def sequence(self) -> str:
        return self.read1.sequence

    @property
    def quality(self) -> str:
        return self.read1.quality

    @property
    def haplotype(self) -> str:
        return self.read1.haplotype

    @property
    def chrom(self) -> str:
        return self.read1.chrom

    @property
    def start_pos(self) -> int:
        return self.read1.start_pos

    @property
    def end_pos(self) -> int:
        return self.read2.end_pos

    @property
    def strand(self) -> str:
        return '+'


# ── Read-type configurations ────────────────────────────────────────────

@dataclass
class IlluminaConfig:
    """Illumina paired-end simulation parameters."""
    coverage: float = 30.0
    read_length: int = 150
    insert_size_mean: int = 350
    insert_size_std: int = 50
    error_rate: float = 0.001
    random_seed: Optional[int] = None


@dataclass
class HiFiConfig:
    """PacBio HiFi simulation parameters."""
    coverage: float = 30.0
    read_length_mean: int = 15_000
    read_length_std: int = 5_000
    error_rate: float = 0.001
    random_seed: Optional[int] = None


@dataclass
class ONTConfig:
    """Oxford Nanopore simulation parameters."""
    coverage: float = 30.0
    read_length_mean: int = 20_000
    read_length_std: int = 8_000
    error_rate: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class ULConfig:
    """Ultra-long ONT simulation parameters."""
    coverage: float = 10.0
    read_length_mean: int = 100_000
    read_length_std: int = 30_000
    error_rate: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class HiCConfig:
    """Hi-C paired-end simulation parameters."""
    num_pairs: int = 10_000
    read_length: int = 150
    error_rate: float = 0.001
    random_seed: Optional[int] = None


@dataclass
class AncientDNAConfig:
    """Ancient DNA simulation parameters."""
    coverage: float = 5.0
    fragment_length_mean: int = 50
    fragment_length_std: int = 15
    error_rate: float = 0.02
    damage_rate: float = 0.20
    random_seed: Optional[int] = None


# ============================================================================
#                          GENOME GENERATION
# ============================================================================

def _random_sequence(length: int, gc_content: float, rng: random.Random) -> str:
    """Generate a random DNA sequence with target GC content."""
    gc_prob = gc_content / 2.0       # per-base P(G) = P(C)
    at_prob = (1.0 - gc_content) / 2.0  # per-base P(A) = P(T)
    bases = []
    for _ in range(length):
        r = rng.random()
        if r < at_prob:
            bases.append('A')
        elif r < 2 * at_prob:
            bases.append('T')
        elif r < 2 * at_prob + gc_prob:
            bases.append('G')
        else:
            bases.append('C')
    return ''.join(bases)


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(comp)[::-1]


def _insert_repeats(
    seq: str,
    repeat_density: float,
    tandem_fraction: float,
    rng: random.Random,
) -> Tuple[str, List[bool]]:
    """Scatter interspersed and tandem repeats into a sequence.

    Returns (modified_sequence, repeat_mask) where repeat_mask[i] is True
    if position i is part of a repeat.
    """
    seq_list = list(seq)
    repeat_mask = [False] * len(seq_list)
    target_repeat_bp = int(len(seq) * repeat_density)
    placed_bp = 0

    # Tandem repeats (short motifs repeated)
    tandem_bp = int(target_repeat_bp * tandem_fraction)
    while placed_bp < tandem_bp:
        motif_len = rng.randint(2, 30)
        copies = rng.randint(3, 50)
        total = motif_len * copies
        if total > len(seq) // 4:
            copies = max(3, len(seq) // (4 * motif_len))
            total = motif_len * copies
        pos = rng.randint(0, max(0, len(seq_list) - total - 1))
        motif = _random_sequence(motif_len, 0.42, rng)
        repeat_seq = motif * copies
        for i, base in enumerate(repeat_seq):
            idx = pos + i
            if idx < len(seq_list):
                seq_list[idx] = base
                repeat_mask[idx] = True
        placed_bp += total

    # Interspersed repeats (longer elements scattered)
    while placed_bp < target_repeat_bp:
        elem_len = rng.choice([300, 500, 1000, 2000, 5000, 6000])
        elem_len = min(elem_len, len(seq) // 10)
        if elem_len < 50:
            break
        # Create repeat element
        element = _random_sequence(elem_len, rng.uniform(0.35, 0.55), rng)
        # Place 2–5 copies
        n_copies = rng.randint(2, 5)
        for _ in range(n_copies):
            pos = rng.randint(0, max(0, len(seq_list) - elem_len - 1))
            maybe_rc = element if rng.random() < 0.5 else _reverse_complement(element)
            for i, base in enumerate(maybe_rc):
                idx = pos + i
                if idx < len(seq_list):
                    seq_list[idx] = base
                    repeat_mask[idx] = True
            placed_bp += elem_len

    return ''.join(seq_list), repeat_mask


def _apply_snps(
    seq: str,
    snp_rate: float,
    rng: random.Random,
) -> Tuple[str, List[int]]:
    """Introduce SNPs into a copy of the sequence."""
    seq_list = list(seq)
    snp_positions = []
    alt_bases = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
    for i in range(len(seq_list)):
        if rng.random() < snp_rate:
            orig = seq_list[i].upper()
            if orig in alt_bases:
                seq_list[i] = rng.choice(alt_bases[orig])
                snp_positions.append(i)
    return ''.join(seq_list), snp_positions


def _apply_indels(
    seq: str,
    indel_rate: float,
    rng: random.Random,
) -> Tuple[str, List[int]]:
    """Introduce small indels (1–5 bp) into a sequence."""
    result = []
    indel_positions = []
    i = 0
    while i < len(seq):
        if rng.random() < indel_rate:
            indel_positions.append(i)
            size = rng.randint(1, 5)
            if rng.random() < 0.5:
                # Insertion
                result.append(seq[i])
                result.append(_random_sequence(size, 0.42, rng))
                i += 1
            else:
                # Deletion
                i += min(size, len(seq) - i)
        else:
            result.append(seq[i])
            i += 1
    return ''.join(result), indel_positions


def _apply_svs(
    seq: str,
    config: GenomeConfig,
    rng: random.Random,
    haplotype: str,
    chrom: str,
) -> Tuple[str, List[StructuralVariant]]:
    """Apply structural variants to a haplotype sequence."""
    n_svs = max(0, int(len(seq) * config.sv_density))
    if n_svs == 0:
        return seq, []

    # Build weighted SV type selection
    sv_weights: List[Tuple[SVType, float]] = []
    for sv_type, frac in [
        (SVType.DELETION, config.sv_deletion_fraction),
        (SVType.INSERTION, config.sv_insertion_fraction),
        (SVType.INVERSION, config.sv_inversion_fraction),
        (SVType.DUPLICATION, config.sv_duplication_fraction),
        (SVType.TRANSLOCATION, config.sv_translocation_fraction),
    ]:
        if frac > 0:
            sv_weights.append((sv_type, frac))

    if not sv_weights:
        return seq, []

    types, weights = zip(*sv_weights)
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    svs: List[StructuralVariant] = []
    seq_list = list(seq)

    for _ in range(n_svs):
        sv_type = rng.choices(types, weights=weights, k=1)[0]
        sv_size = rng.randint(500, 10_000)
        sv_size = min(sv_size, len(seq_list) // 20)
        if sv_size < 50:
            continue
        pos = rng.randint(1000, max(1001, len(seq_list) - sv_size - 1000))

        if sv_type == SVType.DELETION:
            end = min(pos + sv_size, len(seq_list))
            actual_size = end - pos
            del seq_list[pos:end]
            svs.append(StructuralVariant(
                sv_type=sv_type, haplotype=haplotype, chrom=chrom,
                pos=pos, end=end, size=actual_size,
                description=f"{actual_size}bp deletion at {pos}"))

        elif sv_type == SVType.INSERTION:
            insert_seq = _random_sequence(sv_size, 0.42, rng)
            seq_list[pos:pos] = list(insert_seq)
            svs.append(StructuralVariant(
                sv_type=sv_type, haplotype=haplotype, chrom=chrom,
                pos=pos, end=pos + sv_size, size=sv_size,
                description=f"{sv_size}bp insertion at {pos}"))

        elif sv_type == SVType.INVERSION:
            end = min(pos + sv_size, len(seq_list))
            segment = seq_list[pos:end]
            rc_seg = list(_reverse_complement(''.join(segment)))
            seq_list[pos:end] = rc_seg
            svs.append(StructuralVariant(
                sv_type=sv_type, haplotype=haplotype, chrom=chrom,
                pos=pos, end=end, size=end - pos,
                description=f"{end - pos}bp inversion at {pos}"))

        elif sv_type == SVType.DUPLICATION:
            end = min(pos + sv_size, len(seq_list))
            segment = seq_list[pos:end]
            seq_list[end:end] = segment
            svs.append(StructuralVariant(
                sv_type=sv_type, haplotype=haplotype, chrom=chrom,
                pos=pos, end=end + len(segment), size=len(segment),
                description=f"{len(segment)}bp duplication at {pos}"))

        elif sv_type == SVType.TRANSLOCATION:
            end = min(pos + sv_size, len(seq_list))
            segment = seq_list[pos:end]
            del seq_list[pos:end]
            new_pos = rng.randint(0, max(0, len(seq_list) - 1))
            seq_list[new_pos:new_pos] = segment
            svs.append(StructuralVariant(
                sv_type=sv_type, haplotype=haplotype, chrom=chrom,
                pos=pos, end=end, size=len(segment),
                description=f"{len(segment)}bp translocation from {pos} to {new_pos}"))

    return ''.join(seq_list), svs


def generate_diploid_genome(config: GenomeConfig) -> DiploidGenome:
    """Generate a synthetic diploid genome with controlled variation.

    1. Creates a reference haplotype with target GC and repeat content
    2. Copies it to create haplotype B
    3. Applies SNPs, indels, and SVs to haplotype B

    Returns a DiploidGenome with both haplotypes and truth annotations.
    """
    rng = random.Random(config.random_seed)

    logger.info("Generating reference haplotype A (%s bp, GC=%.0f%%, repeats=%.0f%%)...",
                f"{config.length:,}", config.gc_content * 100, config.repeat_density * 100)

    # Step 1: Random base sequence
    hapA = _random_sequence(config.length, config.gc_content, rng)

    # Step 2: Insert repeats
    hapA, repeat_mask_A = _insert_repeats(
        hapA, config.repeat_density, config.tandem_repeat_fraction, rng)

    # Step 3: Copy to hapB
    hapB = hapA
    repeat_mask_B = list(repeat_mask_A)

    # Step 4: Apply heterozygous variation to hapB
    hapB, snp_positions = _apply_snps(hapB, config.snp_rate, rng)
    hapB, indel_positions = _apply_indels(hapB, config.indel_rate, rng)

    # Step 5: Structural variants (on hapB only)
    hapB, sv_truth = _apply_svs(hapB, config, rng, haplotype='B', chrom='chr1')

    # Adjust repeat mask for hapB (may have shifted due to indels/SVs)
    if len(repeat_mask_B) > len(hapB):
        repeat_mask_B = repeat_mask_B[:len(hapB)]
    elif len(repeat_mask_B) < len(hapB):
        repeat_mask_B.extend([False] * (len(hapB) - len(repeat_mask_B)))

    logger.info("Diploid genome generated: hapA=%s bp, hapB=%s bp, "
                "%d SNPs, %d indels, %d SVs",
                f"{len(hapA):,}", f"{len(hapB):,}",
                len(snp_positions), len(indel_positions), len(sv_truth))

    return DiploidGenome(
        hapA=hapA,
        hapB=hapB,
        snp_positions=snp_positions,
        indel_positions=indel_positions,
        sv_truth_table=sv_truth,
        repeat_mask_A=repeat_mask_A,
        repeat_mask_B=repeat_mask_B,
    )


# ============================================================================
#                          READ SIMULATION
# ============================================================================

def _quality_string(length: int, mean_qual: int, rng: random.Random) -> str:
    """Generate a FASTQ quality string of given length."""
    quals = []
    for _ in range(length):
        q = max(2, min(41, int(rng.gauss(mean_qual, 3))))
        quals.append(chr(q + 33))
    return ''.join(quals)


def _introduce_errors(seq: str, error_rate: float, rng: random.Random) -> str:
    """Introduce random substitution errors into a sequence."""
    if error_rate <= 0:
        return seq
    seq_list = list(seq)
    alt = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
    for i in range(len(seq_list)):
        if rng.random() < error_rate:
            orig = seq_list[i].upper()
            if orig in alt:
                seq_list[i] = rng.choice(alt[orig])
    return ''.join(seq_list)


def _apply_damage(seq: str, damage_rate: float, rng: random.Random) -> str:
    """Apply ancient DNA damage (C→T deamination near read ends)."""
    seq_list = list(seq)
    for i in range(len(seq_list)):
        # Damage decreases exponentially from both ends
        dist_from_end = min(i, len(seq_list) - 1 - i)
        local_rate = damage_rate * math.exp(-0.3 * dist_from_end)
        if seq_list[i].upper() == 'C' and rng.random() < local_rate:
            seq_list[i] = 'T'
        elif seq_list[i].upper() == 'G' and rng.random() < local_rate:
            seq_list[i] = 'A'
    return ''.join(seq_list)


def _is_repeat_region(
    start: int,
    end: int,
    repeat_mask: Optional[List[bool]],
    threshold: float = 0.5,
) -> bool:
    """Check if a region is predominantly repetitive."""
    if repeat_mask is None:
        return False
    s = max(0, start)
    e = min(len(repeat_mask), end)
    if e <= s:
        return False
    repeat_count = sum(1 for i in range(s, e) if repeat_mask[i])
    return (repeat_count / (e - s)) > threshold


def simulate_illumina_reads(
    haplotype: str,
    config: IlluminaConfig,
    haplotype_label: str = 'A',
    repeat_mask: Optional[List[bool]] = None,
) -> List[SimulatedReadPair]:
    """Simulate Illumina paired-end reads from a haplotype sequence."""
    rng = random.Random(config.random_seed)
    genome_len = len(haplotype)
    num_pairs = int((genome_len * config.coverage) / (2 * config.read_length))
    reads: List[SimulatedReadPair] = []

    for i in range(num_pairs):
        insert_size = max(
            config.read_length * 2 + 10,
            int(rng.gauss(config.insert_size_mean, config.insert_size_std))
        )
        pos = rng.randint(0, max(0, genome_len - insert_size))
        end = min(pos + insert_size, genome_len)

        r1_seq = haplotype[pos:pos + config.read_length]
        r2_start = max(pos, end - config.read_length)
        r2_seq = _reverse_complement(haplotype[r2_start:end])

        if len(r1_seq) < config.read_length or len(r2_seq) < 10:
            continue

        r1_seq = _introduce_errors(r1_seq, config.error_rate, rng)
        r2_seq = _introduce_errors(r2_seq, config.error_rate, rng)

        rid = f"illumina_{haplotype_label}_{i:08d}"
        is_rep = _is_repeat_region(pos, end, repeat_mask)

        r1 = SimulatedRead(
            read_id=f"{rid}_R1", sequence=r1_seq,
            quality=_quality_string(len(r1_seq), 35, rng),
            haplotype=haplotype_label, chrom='chr1',
            start_pos=pos, end_pos=pos + config.read_length,
            strand='+', technology='illumina')
        r2 = SimulatedRead(
            read_id=f"{rid}_R2", sequence=r2_seq,
            quality=_quality_string(len(r2_seq), 35, rng),
            haplotype=haplotype_label, chrom='chr1',
            start_pos=r2_start, end_pos=end,
            strand='-', technology='illumina')
        reads.append(SimulatedReadPair(read1=r1, read2=r2))

    logger.info("  %s PE reads simulated for hap %s (%d×, %d bp)",
                f"{len(reads):,}", haplotype_label,
                config.coverage, config.read_length)
    return reads


def simulate_long_reads(
    haplotype: str,
    config: Any,
    haplotype_label: str = 'A',
    technology: str = 'hifi',
    repeat_mask: Optional[List[bool]] = None,
) -> List[SimulatedRead]:
    """Simulate long reads (HiFi, ONT, UL, or aDNA) from a haplotype.

    Works with HiFiConfig, ONTConfig, ULConfig, or AncientDNAConfig.
    """
    rng = random.Random(config.random_seed)
    genome_len = len(haplotype)

    # Extract length params (handle both naming conventions)
    length_mean = getattr(config, 'read_length_mean',
                          getattr(config, 'fragment_length_mean', 15_000))
    length_std = getattr(config, 'read_length_std',
                         getattr(config, 'fragment_length_std', 5_000))
    error_rate = config.error_rate
    coverage = config.coverage

    num_reads = max(1, int((genome_len * coverage) / length_mean))
    reads: List[SimulatedRead] = []

    is_ancient = isinstance(config, AncientDNAConfig)
    damage_rate = getattr(config, 'damage_rate', 0.0)

    for i in range(num_reads):
        read_len = max(100, int(rng.gauss(length_mean, length_std)))
        read_len = min(read_len, genome_len)
        pos = rng.randint(0, max(0, genome_len - read_len))
        end = pos + read_len

        seq = haplotype[pos:end]
        if len(seq) < 50:
            continue

        # Choose strand
        if rng.random() < 0.5:
            seq = _reverse_complement(seq)
            strand = '-'
        else:
            strand = '+'

        # Apply sequencing errors
        seq = _introduce_errors(seq, error_rate, rng)

        # Apply ancient DNA damage
        if is_ancient and damage_rate > 0:
            seq = _apply_damage(seq, damage_rate, rng)

        # Quality score: HiFi high (~35), ONT lower (~20), aDNA variable
        mean_qual = 35 if technology == 'hifi' else (12 if is_ancient else 20)
        is_rep = _is_repeat_region(pos, end, repeat_mask)

        rid = f"{technology}_{haplotype_label}_{i:08d}"
        reads.append(SimulatedRead(
            read_id=rid, sequence=seq,
            quality=_quality_string(len(seq), mean_qual, rng),
            haplotype=haplotype_label, chrom='chr1',
            start_pos=pos, end_pos=end,
            strand=strand, technology=technology,
        ))

    label = technology.upper()
    logger.info("  %s %s reads simulated for hap %s (%.0f×, mean %s bp)",
                f"{len(reads):,}", label, haplotype_label,
                coverage, f"{length_mean:,}")
    return reads


def simulate_hic_reads(
    hapA: str,
    hapB: str,
    config: HiCConfig,
    repeat_mask_A: Optional[List[bool]] = None,
    repeat_mask_B: Optional[List[bool]] = None,
) -> List[SimulatedReadPair]:
    """Simulate Hi-C paired reads with proximity ligation bias.

    Hi-C read pairs are drawn from positions correlated by a power-law
    contact probability.  ~80% of pairs come from nearby positions;
    ~20% are long-range or inter-haplotype contacts.
    """
    rng = random.Random(config.random_seed)
    genomes = [('A', hapA), ('B', hapB)]
    pairs: List[SimulatedReadPair] = []

    for i in range(config.num_pairs):
        # Pick a source haplotype
        hap_label, hap_seq = rng.choice(genomes)
        genome_len = len(hap_seq)
        read_len = config.read_length

        # Position of the first read
        pos1 = rng.randint(0, max(0, genome_len - read_len))

        # Distance follows a power-law (contact probability ~ 1/d)
        if rng.random() < 0.8:
            # Short-range contact
            distance = int(abs(rng.gauss(0, 50_000)))
        else:
            # Long-range contact
            distance = rng.randint(0, genome_len)

        # 5% chance of inter-haplotype contact
        if rng.random() < 0.05:
            other_label, other_seq = ('B', hapB) if hap_label == 'A' else ('A', hapA)
            pos2 = rng.randint(0, max(0, len(other_seq) - read_len))
            r2_seq = other_seq[pos2:pos2 + read_len]
            r2_hap = other_label
            r2_chrom = 'chr1'
        else:
            direction = 1 if rng.random() < 0.5 else -1
            pos2 = pos1 + direction * distance
            pos2 = max(0, min(genome_len - read_len, pos2))
            r2_seq = hap_seq[pos2:pos2 + read_len]
            r2_hap = hap_label
            r2_chrom = 'chr1'

        r1_seq = hap_seq[pos1:pos1 + read_len]
        if len(r1_seq) < read_len or len(r2_seq) < read_len:
            continue

        r1_seq = _introduce_errors(r1_seq, config.error_rate, rng)
        r2_seq = _introduce_errors(r2_seq, config.error_rate, rng)

        rid = f"hic_{i:08d}"
        r1 = SimulatedRead(
            read_id=f"{rid}_R1", sequence=r1_seq,
            quality=_quality_string(read_len, 35, rng),
            haplotype=hap_label, chrom='chr1',
            start_pos=pos1, end_pos=pos1 + read_len,
            strand='+', technology='hic')
        r2 = SimulatedRead(
            read_id=f"{rid}_R2", sequence=r2_seq,
            quality=_quality_string(len(r2_seq), 35, rng),
            haplotype=r2_hap, chrom=r2_chrom,
            start_pos=pos2, end_pos=pos2 + len(r2_seq),
            strand='-', technology='hic')
        pairs.append(SimulatedReadPair(read1=r1, read2=r2))

    logger.info("  %s Hi-C pairs simulated (%d bp reads)",
                f"{len(pairs):,}", config.read_length)
    return pairs


# ============================================================================
#                            FASTQ I/O
# ============================================================================

def write_fastq(reads: List[SimulatedRead], path: str) -> None:
    """Write single-end reads to a FASTQ file."""
    with open(path, 'w') as fh:
        for read in reads:
            fh.write(f"@{read.read_id}\n")
            fh.write(f"{read.sequence}\n")
            fh.write("+\n")
            fh.write(f"{read.quality}\n")
    logger.info("Wrote %s reads to %s", f"{len(reads):,}", path)


def write_paired_fastq(
    pairs: List[SimulatedReadPair],
    r1_path: str,
    r2_path: str,
) -> None:
    """Write paired-end reads to two FASTQ files."""
    with open(r1_path, 'w') as f1, open(r2_path, 'w') as f2:
        for pair in pairs:
            f1.write(f"@{pair.read1.read_id}\n{pair.read1.sequence}\n"
                     f"+\n{pair.read1.quality}\n")
            f2.write(f"@{pair.read2.read_id}\n{pair.read2.sequence}\n"
                     f"+\n{pair.read2.quality}\n")
    logger.info("Wrote %s read pairs to %s / %s",
                f"{len(pairs):,}", r1_path, r2_path)

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
