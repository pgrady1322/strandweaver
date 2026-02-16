#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver — K-Weaver Training Data Generator

Generates training CSVs for the 4 K-Weaver XGBoost regressors by:
  1. Synthesising diverse diploid genomes (varying size, GC, repeat density)
  2. Simulating reads (HiFi, ONT, Illumina) per genome
  3. Extracting the 19-dimensional ReadFeatures vector for each read set
  4. Running assembly at a grid of k-mer values
  5. Scoring each assembly against the known reference
  6. Recording the best-performing k for each stage
  7. Exporting a single CSV: 19 features + 4 targets

Designed for Colab (free tier is fine).
Run time: ~15-40 min on 50 genomes (500 Kb – 5 Mb) with 6 k-values.

k-Sweep uses a fast k-mer spectrum proxy (~0.2 s per k) instead of
full DBG construction (~45 s per k), giving ~200× speedup.

Usage:
    python scripts/generate_kweaver_training_data.py \\
        --output training_output/kweaver \\
        --num-genomes 200 \\
        --max-genome-size 20000000 \\
        --threads 8

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import pickle
import random
import sys
import textwrap
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── StrandWeaver imports ──────────────────────────────────────────────
# Add repo root to path if running as standalone script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from strandweaver.preprocessing.kweaver_module import (
    FeatureExtractor,
    ReadFeatures,
    KmerPrediction,
)
from strandweaver.io_utils import SeqRead

logger = logging.getLogger("strandweaver.kweaver_training")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# K-mer values to sweep — must be odd for DBG construction
K_SWEEP_VALUES = [15, 17, 21, 25, 31, 41, 51, 63, 77, 91, 101, 127]

# Genome diversity grid — each row = (genome_size, gc, repeat_density, ploidy)
# The script will sample from these ranges stochastically
GENOME_SIZE_RANGE = (500_000, 5_000_000)    # 500 Kb – 5 Mb (Colab-safe)
GC_CONTENT_RANGE = (0.28, 0.68)
REPEAT_DENSITY_RANGE = (0.05, 0.65)

# Technologies to simulate
TECHNOLOGIES = ['hifi', 'ont', 'illumina']

# Coverage targets per technology
COVERAGE = {
    'hifi': 30.0,
    'ont': 50.0,
    'illumina': 100.0,
}

# Memory-safety caps — Colab free tier has ~12 GB RAM
# At 100× coverage, a 20 Mb genome produces ~13 M Illumina reads (~4 GB).
# Cap total reads and subsample to stay under budget.
MAX_READS_IN_MEMORY = 500_000       # ~150 MB for Illumina, ~7.5 GB for ONT
MAX_GENOME_SIZE_ILLUMINA = 5_000_000  # Cap Illumina genomes to 5 Mb for scoring

# CSV column names — 19 ReadFeatures + 4 targets + metadata
FEATURE_COLUMNS = [
    'mean_read_length', 'median_read_length', 'read_length_n50',
    'min_read_length', 'max_read_length', 'read_length_std',
    'mean_base_quality', 'median_base_quality', 'estimated_error_rate',
    'total_bases', 'num_reads',
    'estimated_genome_size', 'estimated_coverage',
    'gc_content', 'gc_std',
    'read_type_encoded', 'is_paired_end',
    'kmer_spectrum_peak', 'kmer_diversity',
]

TARGET_COLUMNS = ['best_dbg_k', 'best_ul_k', 'best_extension_k', 'best_polish_k']

META_COLUMNS = [
    'genome_idx', 'technology', 'genome_size', 'genome_gc', 'genome_repeat_density',
    'ploidy', 'best_dbg_k_score', 'best_ul_k_score',
    'best_extension_k_score', 'best_polish_k_score',
]


# ═══════════════════════════════════════════════════════════════════════
#  GENOME SIMULATION (lightweight — no dependency on training backend)
# ═══════════════════════════════════════════════════════════════════════

def _generate_random_sequence(length: int, gc_content: float, rng: random.Random) -> str:
    """Generate a random DNA sequence with specified GC content."""
    gc_prob = gc_content / 2.0
    at_prob = (1.0 - gc_content) / 2.0
    bases = rng.choices(
        ['A', 'T', 'G', 'C'],
        weights=[at_prob, at_prob, gc_prob, gc_prob],
        k=length,
    )
    return ''.join(bases)


def _insert_repeats(sequence: str, repeat_density: float, rng: random.Random) -> str:
    """Insert tandem and interspersed repeats into a sequence."""
    if repeat_density <= 0:
        return sequence

    seq_list = list(sequence)
    seq_len = len(seq_list)
    target_repeat_bases = int(seq_len * repeat_density)
    inserted = 0

    while inserted < target_repeat_bases:
        # Random repeat unit (6–500 bp) repeated 2–50 times
        unit_len = rng.randint(6, min(500, seq_len // 20))
        num_copies = rng.randint(2, max(2, min(50, (target_repeat_bases - inserted) // unit_len + 1)))
        if num_copies < 2:
            num_copies = 2

        # Pick a random position and extract the repeat unit from that location
        pos = rng.randint(0, max(0, seq_len - unit_len - 1))
        unit = ''.join(seq_list[pos:pos + unit_len])

        # Insert copies at a nearby position
        insert_pos = rng.randint(max(0, pos - 1000), min(seq_len - 1, pos + 1000))
        repeat_block = list(unit * num_copies)
        seq_list[insert_pos:insert_pos] = repeat_block
        seq_len = len(seq_list)
        inserted += len(repeat_block)

    return ''.join(seq_list)


def _generate_diploid_genome(
    genome_size: int,
    gc_content: float,
    repeat_density: float,
    snp_rate: float = 0.001,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """
    Generate a diploid genome (hapA, hapB) with SNPs between haplotypes.

    Returns:
        (hapA, hapB) sequences
    """
    if rng is None:
        rng = random.Random(42)

    # Generate haplotype A
    hapA = _generate_random_sequence(genome_size, gc_content, rng)
    hapA = _insert_repeats(hapA, repeat_density, rng)

    # Generate haplotype B by introducing SNPs
    hapB_list = list(hapA)
    alt_bases = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
    num_snps = int(len(hapA) * snp_rate)

    snp_positions = rng.sample(range(len(hapA)), min(num_snps, len(hapA)))
    for pos in snp_positions:
        ref = hapB_list[pos]
        if ref in alt_bases:
            hapB_list[pos] = rng.choice(alt_bases[ref])

    hapB = ''.join(hapB_list)
    return hapA, hapB


# ═══════════════════════════════════════════════════════════════════════
#  READ SIMULATION (lightweight — no external simulators)
# ═══════════════════════════════════════════════════════════════════════

def _simulate_reads_from_genome(
    genome: str,
    technology: str,
    coverage: float,
    rng: random.Random,
    max_reads: int = MAX_READS_IN_MEMORY,
) -> List[SeqRead]:
    """
    Simulate reads from a genome sequence with technology-specific profiles.

    If the number of reads would exceed *max_reads*, the coverage is
    automatically reduced so that the read list fits in memory.  A
    warning is emitted when this happens.

    Returns list of SeqRead objects with realistic length/quality distributions.
    """
    genome_len = len(genome)

    if technology == 'hifi':
        mean_len, std_len = 15_000, 4_000
        mean_q = 35.0  # Q35 ≈ 0.03% error
        error_rate = 0.001
    elif technology == 'ont':
        mean_len, std_len = 20_000, 15_000
        mean_q = 15.0  # Q15 ≈ 3% error
        error_rate = 0.05
    elif technology == 'illumina':
        mean_len, std_len = 150, 0   # Fixed length
        mean_q = 30.0
        error_rate = 0.001
    else:
        raise ValueError(f"Unknown technology: {technology}")

    # ── Estimate read count and cap coverage if needed ─────────
    est_read_len = mean_len if technology != 'illumina' else 150
    est_num_reads = int((genome_len * coverage) / est_read_len)

    effective_coverage = coverage
    if est_num_reads > max_reads:
        effective_coverage = coverage * (max_reads / est_num_reads)
        logger.warning(
            f"    ⚠ Capping {technology} coverage from {coverage:.0f}× to "
            f"{effective_coverage:.1f}× to stay within {max_reads:,} read limit "
            f"(genome {genome_len:,} bp)"
        )

    target_bases = int(genome_len * effective_coverage)

    # ── Fast batch path for Illumina (fixed-length, low error) ─────
    if technology == 'illumina':
        return _simulate_illumina_batch(
            genome, target_bases, mean_q, error_rate, rng,
        )

    # ── Batch path for long reads (HiFi / ONT) ─────────────────────
    # Use numpy for error introduction + quality strings instead of
    # per-base Python loops.  The old approach took ~18 min for ONT on
    # a single 5 Mb genome; this version finishes in ~20 s.
    np_rng = np.random.RandomState(rng.randint(0, 2**31))

    # Pre-generate read lengths in bulk
    est_num_reads = int(target_bases / mean_len) + 200
    all_lens = np.clip(
        np_rng.normal(mean_len, std_len, est_num_reads).astype(int),
        500, genome_len,
    )

    reads: List[SeqRead] = []
    total_bases = 0
    read_idx = 0
    # Substitution table: deterministic single-base swap (fast, order-preserving)
    _sub = {'A': 'C', 'T': 'G', 'G': 'T', 'C': 'A'}

    idx = 0
    while total_bases < target_bases and idx < len(all_lens):
        read_len = int(all_lens[idx])
        idx += 1

        start = np_rng.randint(0, genome_len - read_len + 1)
        seq = genome[start:start + read_len]
        n = len(seq)

        # ── Introduce errors using numpy mask (no per-base Python loop) ──
        error_mask = np_rng.random(n) < error_rate
        n_errors = int(error_mask.sum())
        if n_errors > 0:
            seq_arr = list(seq)
            err_positions = np.flatnonzero(error_mask)
            for pos in err_positions:
                base = seq_arr[pos]
                if base in _sub:
                    seq_arr[pos] = _sub[base]
            seq = ''.join(seq_arr)

        if len(seq) < 50:
            continue

        # ── Quality string via numpy → bytes (no chr() loop) ──────────
        q_arr = np.clip(
            np_rng.normal(mean_q, 5, n).astype(np.int8), 2, 41,
        )
        qual_str = (q_arr + 33).astype(np.uint8).tobytes().decode('ascii')

        reads.append(SeqRead(
            id=f"read_{technology}_{read_idx}",
            sequence=seq,
            quality=qual_str,
        ))
        total_bases += n
        read_idx += 1

    return reads


def _simulate_illumina_batch(
    genome: str,
    target_bases: int,
    mean_q: float,
    error_rate: float,
    rng: random.Random,
    read_len: int = 150,
) -> List[SeqRead]:
    """
    Fast batch Illumina read simulation using numpy.

    Instead of a per-base Python loop for each read, this generates
    start positions and error masks in bulk, then slices the genome
    string.  ~30× faster than the per-read loop for 100K+ reads.
    """
    genome_len = len(genome)
    if genome_len < read_len:
        read_len = genome_len

    num_reads = target_bases // read_len
    np_rng = np.random.RandomState(rng.randint(0, 2**31))

    # Batch start positions
    starts = np_rng.randint(0, genome_len - read_len + 1, size=num_reads)

    # Pre-generate a single quality template (all reads get similar quals)
    # and per-read substitution masks
    sub_table = {'A': 'C', 'T': 'G', 'G': 'T', 'C': 'A'}  # Simple deterministic sub

    reads: List[SeqRead] = []
    # Process in chunks of 10K for progress
    chunk_size = 10_000
    for chunk_start in range(0, num_reads, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_reads)
        chunk_starts = starts[chunk_start:chunk_end]

        # Quality scores for the whole chunk
        q_block = np.clip(
            np_rng.normal(mean_q, 5, (chunk_end - chunk_start, read_len)).astype(np.int8),
            2, 41,
        )

        # Error mask: which positions get a substitution
        error_mask = np_rng.random((chunk_end - chunk_start, read_len)) < error_rate

        for local_idx in range(chunk_end - chunk_start):
            s = int(chunk_starts[local_idx])
            seq = genome[s:s + read_len]

            # Apply substitutions (fast: only iterate error positions)
            errs = np.flatnonzero(error_mask[local_idx])
            if len(errs) > 0:
                seq_arr = list(seq)
                for pos in errs:
                    base = seq_arr[pos]
                    if base in sub_table:
                        seq_arr[pos] = sub_table[base]
                seq = ''.join(seq_arr)

            qual_str = (q_block[local_idx] + 33).astype(np.uint8).tobytes().decode('ascii')
            reads.append(SeqRead(
                id=f"read_illumina_{chunk_start + local_idx}",
                sequence=seq,
                quality=qual_str,
            ))

    return reads


# ═══════════════════════════════════════════════════════════════════════
#  ASSEMBLY SCORING — run at each k, score against reference
# ═══════════════════════════════════════════════════════════════════════

def _run_assembly_at_k(reads: List[SeqRead], k: int, min_cov: int = 2) -> List[str]:
    """
    Build DBG at given k and extract contig sequences.

    Returns list of contig sequences (strings).

    **Note**: This is expensive (~40-50s per call for a 1 Mb genome).
    Use `_fast_k_score()` for k-sweep screening, then call this only
    once for the winning k if full-assembly scoring is desired.
    """
    from strandweaver.assembly_core.dbg_engine_module import build_dbg_from_long_reads

    try:
        dbg = build_dbg_from_long_reads(reads, base_k=k, min_coverage=min_cov)
    except Exception as e:
        logger.warning(f"  DBG build failed at k={k}: {e}")
        return []

    # Extract contigs from compacted unitig nodes
    # After compaction, each node is a unitig with its sequence in node.seq
    contigs = []
    for node_id, node in dbg.nodes.items():
        seq = node.seq
        if isinstance(seq, str) and len(seq) >= k:
            contigs.append(seq)

    return contigs


# ── Fast k-mer spectrum scoring (avoids full DBG construction) ────────

def _fast_k_score(
    reads: List[SeqRead],
    k: int,
    genome_size_est: int,
    min_cov: int = 2,
    max_bases_sample: int = 1_000_000,
) -> Dict[str, float]:
    """
    Fast k-mer spectrum scoring — **no DBG construction**.

    Instead of building a full de Bruijn graph (~45 s per call), this
    computes k-mer frequency statistics that are highly predictive of
    assembly quality at a given k:

      * **distinct_ratio**: fraction of k-mers appearing ≥ min_cov times.
        Higher = better (more usable nodes in a DBG).
      * **repetitiveness**: fraction of total k-mer mass in the top-0.1%
        most frequent k-mers.  Lower = less repeat-induced branching.
      * **coverage_uniformity**: std-dev / mean of the coverage
        distribution.  Lower = more uniform, better assembly.
      * **composite**: weighted combination → higher is better.

    ~200× faster than `_run_assembly_at_k` — takes <0.3 s for a
    typical read set (compared to ~45 s for a full DBG build).

    Args:
        reads: Read set (subsampled to *max_bases_sample* total bases).
        k: K-mer size to evaluate.
        genome_size_est: Estimated genome size (for expected-coverage calc).
        min_cov: Minimum coverage to count a k-mer as "solid".
        max_bases_sample: Cap total bases in the sample.  Default 1 Mb
            is ample for spectrum statistics (≥1× any genome in the
            training range).  Takes <0.5 s per k-value.

    Returns:
        Dict with 'distinct_ratio', 'repetitiveness',
        'coverage_uniformity', 'composite'.
    """
    # Subsample reads to stay within the base budget.
    # For long reads (15 Kb), 200 reads ≈ 3 Mb.
    # For Illumina (150 bp), 20 000 reads ≈ 3 Mb.
    sample = []
    bases_so_far = 0
    for r in reads:
        sample.append(r)
        bases_so_far += len(r.sequence)
        if bases_so_far >= max_bases_sample:
            break

    # Concatenate sequences with 'N' separators (prevents cross-read k-mers)
    concat = 'N'.join(r.sequence.upper() for r in sample)
    n = len(concat)

    if n < k:
        return {'distinct_ratio': 0.0, 'repetitiveness': 1.0,
                'coverage_uniformity': 999.0, 'composite': 0.0}

    # Count k-mers from the concatenated string in a single pass.
    # N-separators create a few boundary k-mers; they'll be singletons
    # and get filtered by min_cov, so no explicit N-check needed.
    counts: Counter = Counter(
        concat[i:i + k] for i in range(n - k + 1)
    )

    if not counts:
        return {'distinct_ratio': 0.0, 'repetitiveness': 1.0,
                'coverage_uniformity': 999.0, 'composite': 0.0}

    total_distinct = len(counts)
    freqs = np.array(list(counts.values()), dtype=np.float64)
    total_mass = float(freqs.sum())

    # ── Spectrum shape metrics ────────────────────────────────────
    # At subsampled coverage, absolute thresholds are unreliable.
    # Focus on metrics that capture how k interacts with read length,
    # genome complexity, and coverage.

    # 1) Distinct k-mer ratio vs genome-size estimate.
    #    Ideal: distinct ≈ genome_size. Over-distinct (k too small) →
    #    repeats create fewer unique kmers. Under-distinct (k too large)
    #    → fewer kmers extracted per read.
    expected_distinct = max(1, genome_size_est - k + 1)
    distinct_frac = min(total_distinct / expected_distinct, 2.0) / 2.0

    # 2) Repetitiveness — mass in top 0.1% of k-mers.
    #    Lower = less repeat-induced branching.
    top_n = max(1, total_distinct // 1000)
    top_mass = float(np.sort(freqs)[-top_n:].sum())
    repetitiveness = top_mass / total_mass if total_mass else 1.0

    # 3) Singleton fraction — k-mers appearing exactly once.
    #    High singleton fraction means k is too large for the coverage.
    singletons = int((freqs == 1).sum())
    singleton_frac = singletons / total_distinct if total_distinct else 1.0

    # 4) Coverage uniformity (CV of non-singleton k-mers)
    solid_freqs = freqs[freqs >= 2]
    if len(solid_freqs) > 10:
        cv = float(np.std(solid_freqs) / np.mean(solid_freqs))
    else:
        cv = 1.0  # Not enough data

    # 5) Composite — higher is better
    #    The key insight: best k balances distinct coverage (not too
    #    small → repeat collapse, not too large → low yield)
    composite = (
        0.35 * distinct_frac                          # reward: ~genome-size distinct kmers
        + 0.25 * (1.0 - min(repetitiveness, 1.0))    # reward: low repeat mass
        + 0.20 * (1.0 - singleton_frac)               # reward: few singletons
        + 0.20 * max(0.0, 1.0 - cv)                   # reward: uniform coverage
    )

    return {
        'distinct_frac': distinct_frac,
        'repetitiveness': repetitiveness,
        'singleton_frac': singleton_frac,
        'coverage_uniformity': cv,
        'composite': composite,
    }


def _calculate_n50(lengths: List[int]) -> int:
    """Calculate N50 from a list of contig lengths."""
    if not lengths:
        return 0
    sorted_lens = sorted(lengths, reverse=True)
    total = sum(sorted_lens)
    cumsum = 0
    for length in sorted_lens:
        cumsum += length
        if cumsum >= total / 2:
            return length
    return sorted_lens[-1]


def _build_ref_kmer_sample(
    reference: str,
    k_check: int = 31,
    max_kmers: int = 200_000,
) -> set:
    """
    Build a *sampled* set of reference k-mers for genome-fraction scoring.

    For genomes ≤ *max_kmers* k-mers, returns the full set.
    For larger genomes, takes a deterministic evenly-spaced sample so that
    memory stays bounded (~6 MB for 200 K 31-mers).
    """
    n_possible = len(reference) - k_check + 1
    if n_possible <= 0:
        return set()

    if n_possible <= max_kmers:
        return {reference[i:i + k_check] for i in range(n_possible)}

    # Deterministic subsample — every step-th kmer
    step = (n_possible + max_kmers - 1) // max_kmers  # ceil division
    return {reference[i:i + k_check] for i in range(0, n_possible, step)}


def _score_assembly(
    contigs: List[str],
    reference: str,
    ref_kmer_sample: Optional[set] = None,
) -> Dict[str, float]:
    """
    Score an assembly against the reference genome.

    Uses a *sampled* reference k-mer set (passed in to avoid rebuilding it
    on every k-sweep iteration).  If not provided, one is built on the fly.

    Metrics:
        n50: Contig N50
        total_bases: Total assembled bases
        num_contigs: Number of contigs
        genome_fraction: Fraction of reference covered (approximate)
        contiguity_score: Combined metric (higher = better)
    """
    if not contigs:
        return {
            'n50': 0, 'total_bases': 0, 'num_contigs': 0,
            'genome_fraction': 0.0, 'contiguity_score': 0.0,
        }

    lengths = [len(c) for c in contigs]
    n50 = _calculate_n50(lengths)
    total_bases = sum(lengths)
    ref_len = len(reference)

    # Build or reuse sampled reference k-mers
    k_check = 31
    if ref_kmer_sample is None:
        ref_kmer_sample = _build_ref_kmer_sample(reference, k_check)

    if not ref_kmer_sample:
        genome_fraction = 0.0
    else:
        # Count how many sampled ref k-mers appear in the assembly
        ref_kmers_found = 0
        asm_kmers = set()
        for contig in contigs:
            for i in range(len(contig) - k_check + 1):
                asm_kmers.add(contig[i:i + k_check])

        for kmer in ref_kmer_sample:
            if kmer in asm_kmers:
                ref_kmers_found += 1

        genome_fraction = ref_kmers_found / len(ref_kmer_sample)
        del asm_kmers  # Free immediately

    # Contiguity score: weighted combination of N50 and genome fraction
    normalised_n50 = min(n50 / ref_len, 1.0) if ref_len > 0 else 0.0
    contiguity_score = (0.6 * normalised_n50) + (0.4 * genome_fraction)

    return {
        'n50': n50,
        'total_bases': total_bases,
        'num_contigs': len(contigs),
        'genome_fraction': genome_fraction,
        'contiguity_score': contiguity_score,
    }


def _find_best_k(
    reads: List[SeqRead],
    reference: str,
    k_values: List[int],
    min_read_length: int = 0,
) -> Tuple[int, float, Dict[int, Dict]]:
    """
    Sweep k values and return the best k.

    Uses a **fast k-mer spectrum proxy** (~0.2 s per k) instead of full
    DBG assembly (~45 s per k).  This makes the k-sweep ~200× faster
    while producing highly correlated rankings.

    Args:
        reads: Reads to assemble
        reference: Ground-truth reference sequence
        k_values: List of k-mer sizes to try
        min_read_length: Skip k values larger than shortest read

    Returns:
        (best_k, best_score, all_scores_by_k)
    """
    all_scores: Dict[int, Dict] = {}
    best_k = k_values[0]
    best_score = -1.0

    # Filter k values that are too large for the reads
    if min_read_length > 0:
        k_values = [k for k in k_values if k < min_read_length]
    if not k_values:
        k_values = [21]  # Safe fallback

    genome_size_est = len(reference)

    for k in k_values:
        logger.debug(f"    Scoring k={k} (fast proxy)...")
        score = _fast_k_score(reads, k, genome_size_est)
        all_scores[k] = score

        if score['composite'] > best_score:
            best_score = score['composite']
            best_k = k

    return best_k, best_score, all_scores


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES → CSV ROW
# ═══════════════════════════════════════════════════════════════════════

def _features_to_row(
    features: ReadFeatures,
    best_ks: Dict[str, int],
    best_scores: Dict[str, float],
    genome_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert ReadFeatures + best-k targets + metadata into a flat CSV row."""
    # Encode read_type
    read_type_encoding = {'hifi': 0, 'ont': 1, 'illumina': 2, 'unknown': 3}

    row = {
        # 19 features
        'mean_read_length': features.mean_read_length,
        'median_read_length': features.median_read_length,
        'read_length_n50': features.read_length_n50,
        'min_read_length': features.min_read_length,
        'max_read_length': features.max_read_length,
        'read_length_std': features.read_length_std,
        'mean_base_quality': features.mean_base_quality,
        'median_base_quality': features.median_base_quality,
        'estimated_error_rate': features.estimated_error_rate,
        'total_bases': features.total_bases,
        'num_reads': features.num_reads,
        'estimated_genome_size': features.estimated_genome_size or 0,
        'estimated_coverage': features.estimated_coverage or 0.0,
        'gc_content': features.gc_content,
        'gc_std': features.gc_std,
        'read_type_encoded': read_type_encoding.get(features.read_type, 3),
        'is_paired_end': 1.0 if features.is_paired_end else 0.0,
        'kmer_spectrum_peak': features.kmer_spectrum_peak or 0,
        'kmer_diversity': features.kmer_diversity or 0.0,

        # 4 targets
        'best_dbg_k': best_ks.get('dbg', 31),
        'best_ul_k': best_ks.get('ul_overlap', 501),
        'best_extension_k': best_ks.get('extension', 55),
        'best_polish_k': best_ks.get('polish', 77),

        # Metadata (not used as features during training)
        'genome_idx': genome_meta['genome_idx'],
        'technology': genome_meta['technology'],
        'genome_size': genome_meta['genome_size'],
        'genome_gc': genome_meta['gc_content'],
        'genome_repeat_density': genome_meta['repeat_density'],
        'ploidy': genome_meta.get('ploidy', 'diploid'),
        'best_dbg_k_score': best_scores.get('dbg', 0.0),
        'best_ul_k_score': best_scores.get('ul_overlap', 0.0),
        'best_extension_k_score': best_scores.get('extension', 0.0),
        'best_polish_k_score': best_scores.get('polish', 0.0),
    }
    return row


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def generate_kweaver_training_data(
    output_dir: Path,
    num_genomes: int = 200,
    min_genome_size: int = 500_000,
    max_genome_size: int = 5_000_000,
    technologies: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    seed: int = 42,
    threads: int = 1,
) -> Path:
    """
    Generate K-Weaver training CSV.

    Args:
        output_dir: Directory to write output files
        num_genomes: Number of genomes to simulate
        min_genome_size: Smallest genome in the sweep
        max_genome_size: Largest genome in the sweep
        technologies: Technologies to simulate (default: hifi, ont, illumina)
        k_values: K-mer sizes to sweep (default: K_SWEEP_VALUES)
        seed: Random seed for reproducibility
        threads: Number of parallel workers (future use)

    Returns:
        Path to the output CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if technologies is None:
        technologies = TECHNOLOGIES
    if k_values is None:
        k_values = K_SWEEP_VALUES

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    csv_path = output_dir / 'kweaver_training.csv'
    all_columns = FEATURE_COLUMNS + TARGET_COLUMNS + META_COLUMNS

    total_samples = num_genomes * len(technologies)
    logger.info(f"K-Weaver training data generation")
    logger.info(f"  Genomes: {num_genomes}")
    logger.info(f"  Technologies: {', '.join(technologies)}")
    logger.info(f"  K sweep: {k_values}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Output: {csv_path}")

    rows_written = 0
    t_start = time.time()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for g_idx in range(num_genomes):
            # ── Sample genome parameters ──────────────────────────
            genome_size = int(np_rng.uniform(
                math.log10(min_genome_size),
                math.log10(max_genome_size),
            ))
            genome_size = int(10 ** genome_size)   # Log-uniform sampling
            # Round to nearest 10 Kb
            genome_size = max(min_genome_size, (genome_size // 10_000) * 10_000)

            gc_content = round(np_rng.uniform(*GC_CONTENT_RANGE), 3)
            repeat_density = round(np_rng.uniform(*REPEAT_DENSITY_RANGE), 3)
            ploidy = rng.choice(['haploid', 'diploid'])

            elapsed = time.time() - t_start
            eta = (elapsed / max(g_idx, 1)) * (num_genomes - g_idx)

            # Memory check (RSS in MB)
            try:
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            except Exception:
                rss_mb = 0.0

            logger.info(
                f"\n[{g_idx + 1}/{num_genomes}] Genome: "
                f"{genome_size:,} bp, GC={gc_content:.2f}, "
                f"repeat={repeat_density:.2f}, {ploidy}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                f"RSS={rss_mb:.0f} MB)"
            )

            # ── Generate genome ───────────────────────────────────
            snp_rate = 0.001 if ploidy == 'diploid' else 0.0
            hapA, hapB = _generate_diploid_genome(
                genome_size, gc_content, repeat_density,
                snp_rate=snp_rate, rng=rng,
            )
            reference = hapA  # Use hapA as the reference for scoring

            # ── Per-technology sweep ──────────────────────────────
            for tech in technologies:
                logger.info(f"  Technology: {tech}")

                # Simulate reads
                cov = COVERAGE.get(tech, 30.0)
                t_sim = time.time()
                reads = _simulate_reads_from_genome(reference, tech, cov, rng)
                logger.info(f"    Simulated {len(reads):,} reads ({cov}×) in {time.time() - t_sim:.1f}s")

                # Extract ReadFeatures
                # Build a temporary in-memory feature extraction
                features = _extract_features_from_reads(reads, tech)
                logger.info(
                    f"    Features: len={features.mean_read_length:.0f}, "
                    f"Q={features.mean_base_quality:.1f}, "
                    f"GC={features.gc_content:.3f}"
                )

                # ── K-value sweep ─────────────────────────────────
                # Filter k values sensible for this technology
                min_rl = features.min_read_length
                valid_ks = [k for k in k_values if k <= min_rl and k >= 11]
                if not valid_ks:
                    valid_ks = [k for k in k_values if k <= 31]
                if not valid_ks:
                    valid_ks = [21]

                logger.info(f"    Sweeping k ∈ {valid_ks}")
                t_sweep = time.time()
                best_dbg_k, best_dbg_score, _ = _find_best_k(
                    reads, reference, valid_ks, min_read_length=min_rl,
                )
                logger.info(
                    f"    Best DBG k={best_dbg_k} (score={best_dbg_score:.4f}) "
                    f"in {time.time() - t_sweep:.1f}s"
                )

                # Derive UL / extension / polish k from DBG k
                # (independent sweep would be ideal but too expensive;
                #  use validated ratios from assembly literature)
                best_ul_k = _derive_ul_k(best_dbg_k, tech)
                best_ext_k = _derive_extension_k(best_dbg_k, tech)
                best_pol_k = _derive_polish_k(best_dbg_k, tech)

                best_ks = {
                    'dbg': best_dbg_k,
                    'ul_overlap': best_ul_k,
                    'extension': best_ext_k,
                    'polish': best_pol_k,
                }
                best_scores = {
                    'dbg': best_dbg_score,
                    'ul_overlap': best_dbg_score * 0.9,   # Derived, mark lower
                    'extension': best_dbg_score * 0.85,
                    'polish': best_dbg_score * 0.85,
                }

                genome_meta = {
                    'genome_idx': g_idx,
                    'technology': tech,
                    'genome_size': genome_size,
                    'gc_content': gc_content,
                    'repeat_density': repeat_density,
                    'ploidy': ploidy,
                }

                row = _features_to_row(features, best_ks, best_scores, genome_meta)
                writer.writerow(row)
                rows_written += 1

                # ── Free reads + features between technologies ─────
                del reads, features
                gc.collect()

            # ── Free genome strings between genomes ───────────────
            del hapA, hapB, reference
            gc.collect()

            # Flush periodically
            if (g_idx + 1) % 10 == 0:
                f.flush()

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"K-Weaver training data generation complete!")
    logger.info(f"  Rows written: {rows_written}")
    logger.info(f"  Output CSV:   {csv_path}")
    logger.info(f"  Elapsed:      {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"{'='*70}")

    # Save generation config for reproducibility
    config_path = output_dir / 'kweaver_generation_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'num_genomes': num_genomes,
            'min_genome_size': min_genome_size,
            'max_genome_size': max_genome_size,
            'technologies': technologies,
            'k_values': k_values,
            'seed': seed,
            'rows_written': rows_written,
            'elapsed_seconds': elapsed,
        }, f, indent=2)

    return csv_path


# ═══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _extract_features_from_reads(reads: List[SeqRead], technology: str) -> ReadFeatures:
    """
    Extract ReadFeatures directly from in-memory SeqRead objects.

    Mirrors FeatureExtractor.extract_from_file() but works without disk I/O.
    """
    if not reads:
        raise ValueError("No reads provided")

    sequences = [r.sequence for r in reads]
    qualities = [r.quality for r in reads if r.quality]

    lengths = np.array([len(s) for s in sequences])
    sorted_lengths = np.sort(lengths)[::-1]
    cumsum = np.cumsum(sorted_lengths)
    n50_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    n50 = int(sorted_lengths[n50_idx])

    # Quality stats
    if qualities:
        all_quals = []
        for q in qualities[:1000]:  # Sample for speed
            all_quals.extend([ord(c) - 33 for c in q])
        mean_q = float(np.mean(all_quals))
        median_q = float(np.median(all_quals))
        error_rate = 10 ** (-mean_q / 10)
    else:
        mean_q = 30.0
        median_q = 30.0
        error_rate = 0.001

    # GC content
    gc_per_read = []
    for s in sequences[:5000]:
        gc = (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0.5
        gc_per_read.append(gc)
    gc_mean = float(np.mean(gc_per_read))
    gc_std = float(np.std(gc_per_read))

    total_bases = int(np.sum(lengths))

    return ReadFeatures(
        mean_read_length=float(np.mean(lengths)),
        median_read_length=float(np.median(lengths)),
        read_length_n50=float(n50),
        min_read_length=int(np.min(lengths)),
        max_read_length=int(np.max(lengths)),
        read_length_std=float(np.std(lengths)),
        mean_base_quality=mean_q,
        median_base_quality=median_q,
        estimated_error_rate=error_rate,
        total_bases=total_bases,
        num_reads=len(sequences),
        estimated_genome_size=None,  # Not estimated in-memory
        estimated_coverage=None,
        gc_content=gc_mean,
        gc_std=gc_std,
        read_type=technology if technology in ('hifi', 'ont', 'illumina') else 'unknown',
        is_paired_end=(technology == 'illumina'),
        kmer_spectrum_peak=None,
        kmer_diversity=None,
    )


def _derive_ul_k(dbg_k: int, technology: str) -> int:
    """
    Derive ultra-long overlap k from the best DBG k.

    UL overlap k is much larger — typically 10–20× the DBG k, clamped
    to odd values in the [201, 1001] range for long-read overlapping.
    For Illumina (no UL reads), returns the DBG k itself.
    """
    if technology == 'illumina':
        return dbg_k

    # Scale up: UL reads are long enough for large-k overlap detection
    ul_k = int(dbg_k * 15)
    # Clamp to sensible range
    ul_k = max(201, min(1001, ul_k))
    # Ensure odd
    if ul_k % 2 == 0:
        ul_k += 1
    return ul_k


def _derive_extension_k(dbg_k: int, technology: str) -> int:
    """
    Derive extension k from the best DBG k.

    Extension k is typically 1.5–2× the DBG k — large enough to avoid
    false extensions but small enough to handle repeat boundaries.
    """
    ext_k = int(dbg_k * 1.7)
    ext_k = max(21, min(127, ext_k))
    if ext_k % 2 == 0:
        ext_k += 1
    return ext_k


def _derive_polish_k(dbg_k: int, technology: str) -> int:
    """
    Derive polishing k from the best DBG k.

    Polish k is typically 2–2.5× the DBG k — larger k gives higher
    specificity for base-level correction but needs sufficient coverage.
    """
    pol_k = int(dbg_k * 2.3)
    pol_k = max(31, min(127, pol_k))
    if pol_k % 2 == 0:
        pol_k += 1
    return pol_k


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='generate-kweaver-training-data',
        description='Generate training data for K-Weaver ML models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Standard Colab run (100 genomes up to 5 Mb)
              python scripts/generate_kweaver_training_data.py \\
                  --output training_output/kweaver \\
                  --num-genomes 100 --max-genome-size 5000000

              # Quick local test (10 small genomes)
              python scripts/generate_kweaver_training_data.py \\
                  --output training_output/kweaver_test \\
                  --num-genomes 10 --max-genome-size 2000000

              # Targeted sweep (HiFi only, more k values)
              python scripts/generate_kweaver_training_data.py \\
                  --output training_output/kweaver_hifi \\
                  --technologies hifi --num-genomes 100

              # After generation, train the models:
              python -m strandweaver.user_training.train_models \\
                  --data-dir training_output/kweaver \\
                  --output-dir trained_models/ \\
                  --models kweaver_dbg kweaver_ul kweaver_extension kweaver_polish
        """),
    )

    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory for training CSV and config',
    )
    parser.add_argument(
        '--num-genomes', '-n', type=int, default=200,
        help='Number of genomes to simulate (default: 200)',
    )
    parser.add_argument(
        '--min-genome-size', type=int, default=500_000,
        help='Minimum genome size in bp (default: 500000)',
    )
    parser.add_argument(
        '--max-genome-size', type=int, default=5_000_000,
        help='Maximum genome size in bp (default: 5000000; use 20000000 only on high-RAM instances)',
    )
    parser.add_argument(
        '--technologies', nargs='+', default=['hifi', 'ont', 'illumina'],
        choices=['hifi', 'ont', 'illumina'],
        help='Technologies to simulate (default: hifi ont illumina)',
    )
    parser.add_argument(
        '--k-values', nargs='+', type=int, default=None,
        help=f'K-mer sizes to sweep (default: {K_SWEEP_VALUES})',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--threads', '-t', type=int, default=1,
        help='Number of threads (default: 1, future use)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose (DEBUG) logging',
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       StrandWeaver · K-Weaver Training Data Generator      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Genomes         : {args.num_genomes}")
    print(f"║  Genome range    : {args.min_genome_size:,} – {args.max_genome_size:,} bp")
    print(f"║  Technologies    : {', '.join(args.technologies)}")
    print(f"║  K sweep         : {args.k_values or K_SWEEP_VALUES}")
    print(f"║  Seed            : {args.seed}")
    print(f"║  Output          : {args.output}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    csv_path = generate_kweaver_training_data(
        output_dir=Path(args.output),
        num_genomes=args.num_genomes,
        min_genome_size=args.min_genome_size,
        max_genome_size=args.max_genome_size,
        technologies=args.technologies,
        k_values=args.k_values,
        seed=args.seed,
        threads=args.threads,
    )

    print(f"\n✅ Training CSV ready: {csv_path}")
    print(f"\nNext step — train K-Weaver models:")
    print(f"  python -m strandweaver.user_training.train_models \\")
    print(f"      --data-dir {args.output} \\")
    print(f"      --output-dir trained_models/ \\")
    print(f"      --models kweaver_dbg kweaver_ul kweaver_extension kweaver_polish")


if __name__ == '__main__':
    main()

# StrandWeaver v0.3.0-dev
# Any usage is subject to this software's license.
