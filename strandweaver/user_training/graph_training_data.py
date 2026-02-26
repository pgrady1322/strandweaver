#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Graph-format training data for GNN models.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .training_config import GraphTrainingConfig, ReadType

logger = logging.getLogger(__name__)

# CSV schema version — bump whenever column layout changes so notebooks
# can auto-detect old vs new format.
SCHEMA_VERSION = "2.0"
GENERATOR_VERSION = "0.2.0"


# ============================================================================
#                     DATA STRUCTURES
# ============================================================================

@dataclass
class ReadInfo:
    """Lightweight read descriptor carried through the graph pipeline."""
    read_id: str
    sequence: str
    quality: str
    haplotype: str          # 'A' or 'B'
    chrom: str
    start_pos: int
    end_pos: int
    strand: str = '+'
    technology: str = 'hifi'
    is_repeat: bool = False

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos


@dataclass
class Overlap:
    """Pairwise overlap between two reads."""
    read_a_id: str
    read_b_id: str
    overlap_start_a: int
    overlap_end_a: int
    overlap_start_b: int
    overlap_end_b: int
    overlap_length: int
    identity: float
    is_true: bool = True
    is_noise: bool = False


@dataclass
class GraphNode:
    """Node in the synthetic overlap graph (= one read)."""
    node_id: str
    read_info: ReadInfo
    # Core features
    coverage: float = 0.0
    gc_content: float = 0.0
    repeat_fraction: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    # Hi-C contact features (step 6b)
    hic_intra: int = 0
    hic_inter: int = 0
    hic_ratio: float = 0.5
    hic_phase: float = 0.0
    # Graph topology features (step 6c)
    clustering_coeff: float = 0.0
    component_id: int = 0
    component_size: int = 0
    # Sequence complexity features (step 6d)
    shannon_entropy: float = 0.0
    dinucleotide_bias: float = 0.0
    homopolymer_max_run: int = 0
    homopolymer_density: float = 0.0
    low_complexity_fraction: float = 0.0
    # Coverage distribution features (step 6e)
    coverage_skewness: float = 0.0
    coverage_kurtosis: float = 0.0
    coverage_cv: float = 0.0
    coverage_p10: float = 0.0
    coverage_p90: float = 0.0
    # Realistic features replacing old placeholders (step 6f)
    mapping_quality: float = 60.0
    allele_frequency: float = 0.5
    heterozygosity: float = 0.0
    phase_consistency: float = 1.0
    mappability: float = 1.0


@dataclass
class GraphEdge:
    """Edge in the synthetic overlap graph."""
    source: str
    target: str
    overlap: Overlap
    features: Dict[str, float] = field(default_factory=dict)
    label: str = 'UNKNOWN'


@dataclass
class SyntheticGraph:
    """Complete synthetic assembly graph with labels and features."""
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    node_labels: Dict[str, str] = field(default_factory=dict)
    edge_labels: Dict[Tuple[str, str], str] = field(default_factory=dict)
    correct_paths: List[List[str]] = field(default_factory=list)
    ul_routes: List[Dict[str, Any]] = field(default_factory=list)
    sv_labels: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GenomeMetadata:
    """Metadata attached to every training CSV row for stratified analysis."""
    genome_id: int = 0
    genome_size: int = 0
    chromosome_id: str = "chr1"
    read_technology: str = "hifi"
    coverage_depth: float = 30.0
    error_rate: float = 0.001
    ploidy: str = "diploid"
    gc_content_global: float = 0.42
    repeat_density_global: float = 0.30
    heterozygosity_rate: float = 0.001
    random_seed: int = 0
    generator_version: str = GENERATOR_VERSION
    schema_version: str = SCHEMA_VERSION

    def as_row(self) -> List[Any]:
        """Return metadata values in canonical column order."""
        return [
            self.genome_id, self.genome_size, self.chromosome_id,
            self.read_technology, self.coverage_depth, self.error_rate,
            self.ploidy, self.gc_content_global, self.repeat_density_global,
            self.heterozygosity_rate, self.random_seed,
            self.generator_version, self.schema_version,
        ]


METADATA_COLUMNS = [
    'genome_id', 'genome_size', 'chromosome_id',
    'read_technology', 'coverage_depth', 'error_rate',
    'ploidy', 'gc_content_global', 'repeat_density_global',
    'heterozygosity_rate', 'random_seed',
    'generator_version', 'schema_version',
]


# ============================================================================
#                SEQUENCE UTILITY FUNCTIONS
# ============================================================================

def _gc_content(seq: str) -> float:
    """GC fraction of a sequence."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq.upper() if b in ('G', 'C'))
    return gc / len(seq)


def _sequence_identity_estimate(seq_a: str, seq_b: str,
                                max_sample: int = 200) -> float:
    """Fast identity estimate by sampling positions."""
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0
    if min_len <= max_sample:
        matches = sum(a == b for a, b in zip(seq_a[:min_len], seq_b[:min_len]))
        return matches / min_len
    step = max(1, min_len // max_sample)
    matches = total = 0
    for i in range(0, min_len, step):
        if seq_a[i] == seq_b[i]:
            matches += 1
        total += 1
    return matches / total if total else 0.0


def _kmer_diversity(seq: str, k: int = 21) -> float:
    """Fraction of unique k-mers in a sequence (0-1)."""
    if len(seq) < k:
        return 0.0
    total = len(seq) - k + 1
    unique = len({seq[i:i + k] for i in range(total)})
    return unique / total


def _repeat_fraction_estimate(seq: str, k: int = 15, threshold: int = 3) -> float:
    """Fraction of positions covered by k-mers appearing >=threshold times."""
    if len(seq) < k:
        return 0.0
    kmer_counts: Dict[str, int] = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmer_counts[seq[i:i + k]] += 1
    repeat_pos = sum(1 for i in range(len(seq) - k + 1)
                     if kmer_counts[seq[i:i + k]] >= threshold)
    return repeat_pos / max(1, len(seq) - k + 1)


def _shannon_entropy(seq: str) -> float:
    """Shannon entropy of base composition (bits). Max = 2.0 for uniform ACGT."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    length = len(seq_upper)
    counts = {b: 0 for b in 'ACGT'}
    for b in seq_upper:
        if b in counts:
            counts[b] += 1
    entropy = 0.0
    for cnt in counts.values():
        if cnt > 0:
            p = cnt / length
            entropy -= p * math.log2(p)
    return entropy


def _dinucleotide_bias(seq: str) -> float:
    """CpG observed/expected ratio deviation. 0 = no bias."""
    seq_upper = seq.upper()
    if len(seq_upper) < 2:
        return 0.0
    c_count = seq_upper.count('C')
    g_count = seq_upper.count('G')
    cg_count = sum(1 for i in range(len(seq_upper) - 1) if seq_upper[i:i + 2] == 'CG')
    expected = (c_count * g_count) / max(len(seq_upper), 1)
    if expected == 0:
        return 0.0
    return abs(cg_count / expected - 1.0)


def _homopolymer_stats(seq: str) -> Tuple[int, float]:
    """Return (max_homopolymer_run, homopolymer_density_per_kb)."""
    if not seq:
        return 0, 0.0
    seq_upper = seq.upper()
    max_run = 1
    current_run = 1
    num_runs = 0
    for i in range(1, len(seq_upper)):
        if seq_upper[i] == seq_upper[i - 1]:
            current_run += 1
        else:
            if current_run >= 3:
                num_runs += 1
            max_run = max(max_run, current_run)
            current_run = 1
    if current_run >= 3:
        num_runs += 1
    max_run = max(max_run, current_run)
    density = (num_runs / len(seq_upper)) * 1000.0
    return max_run, density


def _low_complexity_fraction(seq: str, k: int = 6) -> float:
    """Fraction of sequence covered by low-complexity k-mers (<=2 distinct bases)."""
    if len(seq) < k:
        return 0.0
    seq_upper = seq.upper()
    total = len(seq_upper) - k + 1
    low = sum(1 for i in range(total) if len(set(seq_upper[i:i + k])) <= 2)
    return low / total


# ============================================================================
#                OVERLAP DETECTION
# ============================================================================

def detect_overlaps(
    reads: List[ReadInfo],
    min_overlap_bp: int = 500,
    min_identity: float = 0.90,
    max_overhang_fraction: float = 0.30,
) -> List[Overlap]:
    """Detect pairwise overlaps using ground-truth coordinates."""
    logger.info(f"Detecting overlaps among {len(reads)} reads "
                f"(min_ovl={min_overlap_bp}, min_id={min_identity:.2f})...")

    by_chrom: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in reads:
        by_chrom[r.chrom].append(r)

    overlaps: List[Overlap] = []

    for chrom, chrom_reads in by_chrom.items():
        chrom_reads.sort(key=lambda r: r.start_pos)
        n = len(chrom_reads)
        for i in range(n):
            ri = chrom_reads[i]
            for j in range(i + 1, n):
                rj = chrom_reads[j]
                if rj.start_pos >= ri.end_pos:
                    break

                ovl_start = max(ri.start_pos, rj.start_pos)
                ovl_end = min(ri.end_pos, rj.end_pos)
                ovl_len = ovl_end - ovl_start
                if ovl_len < min_overlap_bp:
                    continue

                shorter_len = min(ri.length, rj.length)
                overhang_a = ri.length - ovl_len
                overhang_b = rj.length - ovl_len
                if max(overhang_a, overhang_b) / shorter_len > max_overhang_fraction:
                    continue

                offset_a = ovl_start - ri.start_pos
                offset_b = ovl_start - rj.start_pos
                sub_a = ri.sequence[offset_a:offset_a + ovl_len]
                sub_b = rj.sequence[offset_b:offset_b + ovl_len]
                ident = _sequence_identity_estimate(sub_a, sub_b)
                if ident < min_identity:
                    continue

                overlaps.append(Overlap(
                    read_a_id=ri.read_id, read_b_id=rj.read_id,
                    overlap_start_a=offset_a, overlap_end_a=offset_a + ovl_len,
                    overlap_start_b=offset_b, overlap_end_b=offset_b + ovl_len,
                    overlap_length=ovl_len, identity=ident,
                    is_true=(ri.haplotype == rj.haplotype),
                ))

    logger.info(f"Detected {len(overlaps)} overlaps")
    return overlaps


# ============================================================================
#                NOISE EDGE INJECTION
# ============================================================================

def inject_noise_edges(
    reads: List[ReadInfo],
    true_overlaps: List[Overlap],
    fraction: float = 0.10,
    rng: Optional[random.Random] = None,
) -> List[Overlap]:
    """Create false overlap edges for negative-class training examples."""
    if fraction <= 0 or len(reads) < 4:
        return []

    rng = rng or random.Random(42)
    num_noise = max(1, int(len(true_overlaps) * fraction))
    logger.info(f"Injecting {num_noise} noise edges...")

    existing_pairs: Set[Tuple[str, str]] = set()
    for o in true_overlaps:
        existing_pairs.add((o.read_a_id, o.read_b_id))
        existing_pairs.add((o.read_b_id, o.read_a_id))

    noise: List[Overlap] = []
    attempts = 0
    max_attempts = num_noise * 20

    while len(noise) < num_noise and attempts < max_attempts:
        attempts += 1
        a = rng.choice(reads)
        b = rng.choice(reads)
        if a.read_id == b.read_id:
            continue
        pair_key = (a.read_id, b.read_id)
        if pair_key in existing_pairs:
            continue
        existing_pairs.add(pair_key)
        existing_pairs.add((b.read_id, a.read_id))

        max_ovl = min(a.length, b.length, 5000)
        min_ovl = min(200, max_ovl)
        if max_ovl < 20:
            continue
        fake_ovl_len = rng.randint(min_ovl, max_ovl)
        noise.append(Overlap(
            read_a_id=a.read_id, read_b_id=b.read_id,
            overlap_start_a=0, overlap_end_a=fake_ovl_len,
            overlap_start_b=0, overlap_end_b=fake_ovl_len,
            overlap_length=fake_ovl_len, identity=rng.uniform(0.70, 0.95),
            is_true=False, is_noise=True,
        ))

    logger.info(f"Injected {len(noise)} noise edges")
    return noise


# ============================================================================
#               GRAPH CONSTRUCTION
# ============================================================================

def build_overlap_graph(
    reads: List[ReadInfo],
    overlaps: List[Overlap],
) -> SyntheticGraph:
    """Build a directed overlap graph from reads and their overlaps."""
    logger.info("Building overlap graph...")

    nodes: Dict[str, GraphNode] = {}
    read_lookup: Dict[str, ReadInfo] = {}
    for r in reads:
        nodes[r.read_id] = GraphNode(node_id=r.read_id, read_info=r)
        read_lookup[r.read_id] = r

    edges: List[GraphEdge] = []
    for ovl in overlaps:
        ra = read_lookup.get(ovl.read_a_id)
        rb = read_lookup.get(ovl.read_b_id)
        if not ra or not rb:
            continue
        if ra.start_pos <= rb.start_pos:
            src, tgt = ovl.read_a_id, ovl.read_b_id
        else:
            src, tgt = ovl.read_b_id, ovl.read_a_id
        edges.append(GraphEdge(source=src, target=tgt, overlap=ovl))

    for e in edges:
        if e.source in nodes:
            nodes[e.source].out_degree += 1
        if e.target in nodes:
            nodes[e.target].in_degree += 1

    logger.info(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    return SyntheticGraph(nodes=nodes, edges=edges)


# ============================================================================
#              FEATURE EXTRACTION
# ============================================================================

def _build_sorted_read_index(reads: List[ReadInfo]) -> Tuple[List[int], List[int]]:
    """Build sorted start/end arrays for O(log n) coverage queries.

    Returns (sorted_starts, sorted_ends) where each is a sorted list.
    Coverage at position p = (# reads with start <= p) - (# reads with end <= p).
    """
    starts = sorted(r.start_pos for r in reads)
    ends = sorted(r.end_pos for r in reads)
    return starts, ends


def _coverage_at_pos_fast(pos: int, sorted_starts: List[int], sorted_ends: List[int]) -> int:
    """O(log n) coverage query using pre-sorted start/end arrays."""
    # Number of reads that started at or before pos
    started = bisect.bisect_right(sorted_starts, pos)
    # Number of reads that ended at or before pos
    ended = bisect.bisect_right(sorted_ends, pos)
    return started - ended


def _coverage_at_pos(reads: List[ReadInfo], pos: int, chrom: str) -> int:
    """Count how many reads cover a specific position (legacy interface)."""
    return sum(1 for r in reads if r.chrom == chrom and r.start_pos <= pos < r.end_pos)


def _coverage_array_for_node_fast(node: GraphNode,
                                  sorted_starts: List[int],
                                  sorted_ends: List[int],
                                  num_samples: int = 20) -> List[float]:
    """Sample coverage using O(log n) queries instead of O(n) per sample."""
    r = node.read_info
    step = max(1, r.length // num_samples)
    coverages = []
    for pos in range(r.start_pos, r.end_pos, step):
        coverages.append(float(_coverage_at_pos_fast(pos, sorted_starts, sorted_ends)))
    return coverages if coverages else [0.0]


def _coverage_array_for_node(node: GraphNode, chrom_reads: List[ReadInfo],
                             num_samples: int = 20) -> List[float]:
    """Sample coverage at multiple positions across a node's read span (legacy)."""
    r = node.read_info
    step = max(1, r.length // num_samples)
    coverages = []
    for pos in range(r.start_pos, r.end_pos, step):
        coverages.append(float(sum(1 for rd in chrom_reads if rd.start_pos <= pos < rd.end_pos)))
    return coverages if coverages else [0.0]


def extract_node_features(graph: SyntheticGraph, all_reads: List[ReadInfo],
                           genome_length: int, expected_coverage: float = 30.0) -> None:
    """Compute per-node features in-place on the graph."""
    logger.info("Extracting node features...")

    # Build sorted read index per chromosome for O(log n) coverage queries
    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)
    chrom_index: Dict[str, Tuple[List[int], List[int]]] = {}
    for chrom, reads in chrom_reads.items():
        chrom_index[chrom] = _build_sorted_read_index(reads)

    for nid, node in graph.nodes.items():
        r = node.read_info
        seq = r.sequence
        node.gc_content = _gc_content(seq)
        node.repeat_fraction = _repeat_fraction_estimate(seq)
        mid = (r.start_pos + r.end_pos) // 2
        idx = chrom_index.get(r.chrom)
        if idx:
            node.coverage = _coverage_at_pos_fast(mid, idx[0], idx[1])
        else:
            node.coverage = 0

    logger.info(f"Extracted features for {len(graph.nodes)} nodes")


def extract_edge_features(graph: SyntheticGraph, all_reads: List[ReadInfo]) -> None:
    """Compute per-edge features in-place on the graph."""
    logger.info("Extracting edge features...")

    read_lookup = {nid: node.read_info for nid, node in graph.nodes.items()}
    node_lookup = graph.nodes

    for edge in graph.edges:
        r1 = read_lookup.get(edge.source)
        r2 = read_lookup.get(edge.target)
        n1 = node_lookup.get(edge.source)
        n2 = node_lookup.get(edge.target)
        if not r1 or not r2 or not n1 or not n2:
            continue

        ovl = edge.overlap
        gc1, gc2 = n1.gc_content, n2.gc_content
        rep1, rep2 = n1.repeat_fraction, n2.repeat_fraction
        kd1 = _kmer_diversity(r1.sequence)
        kd2 = _kmer_diversity(r2.sequence)
        bf1 = n1.in_degree + n1.out_degree
        bf2 = n2.in_degree + n2.out_degree

        # EdgeAI features
        edge.features['overlap_length'] = float(ovl.overlap_length)
        edge.features['overlap_identity'] = ovl.identity
        edge.features['read1_length'] = float(r1.length)
        edge.features['read2_length'] = float(r2.length)
        edge.features['coverage_r1'] = float(n1.coverage)
        edge.features['coverage_r2'] = float(n2.coverage)
        edge.features['gc_content_r1'] = gc1
        edge.features['gc_content_r2'] = gc2
        edge.features['repeat_fraction_r1'] = rep1
        edge.features['repeat_fraction_r2'] = rep2
        edge.features['kmer_diversity_r1'] = kd1
        edge.features['kmer_diversity_r2'] = kd2
        edge.features['branching_factor_r1'] = float(bf1)
        edge.features['branching_factor_r2'] = float(bf2)
        edge.features['hic_support'] = 0.0  # filled in step 6b
        edge.features['mapping_quality_r1'] = n1.mapping_quality  # filled in 6f
        edge.features['mapping_quality_r2'] = n2.mapping_quality
        # v2.0: graph topology (filled in 6c)
        edge.features['clustering_coeff_r1'] = 0.0
        edge.features['clustering_coeff_r2'] = 0.0
        edge.features['component_size'] = 0.0
        # v2.0: sequence complexity (filled in 6d)
        edge.features['entropy_r1'] = 0.0
        edge.features['entropy_r2'] = 0.0
        edge.features['homopolymer_max_r1'] = 0.0
        edge.features['homopolymer_max_r2'] = 0.0

        # PathGNN features
        cov_consistency = 1.0 - abs(n1.coverage - n2.coverage) / max(n1.coverage, n2.coverage, 1)
        gc_sim = 1.0 - abs(gc1 - gc2)
        rep_match = 1.0 - abs(rep1 - rep2)

        edge.features['coverage_consistency'] = cov_consistency
        edge.features['gc_similarity'] = gc_sim
        edge.features['repeat_match'] = rep_match
        edge.features['branching_score'] = float(bf1 + bf2) / 2.0
        edge.features['path_support'] = 1.0 if ovl.is_true else 0.0
        edge.features['hic_contact'] = 0.0  # filled in step 6b
        edge.features['mapping_quality'] = (n1.mapping_quality + n2.mapping_quality) / 2.0
        edge.features['kmer_match'] = (kd1 + kd2) / 2.0
        edge.features['sequence_complexity'] = _kmer_diversity(
            r1.sequence[-min(500, len(r1.sequence)):] +
            r2.sequence[:min(500, len(r2.sequence))])
        edge.features['orientation_score'] = 1.0 if r1.strand == r2.strand else 0.0
        edge.features['distance_score'] = max(0.0, 1.0 - abs(
            r2.start_pos - r1.end_pos) / max(r1.length, 1))
        edge.features['topology_score'] = cov_consistency * gc_sim
        edge.features['ul_support'] = 0.0
        edge.features['sv_evidence'] = 0.0

    logger.info(f"Extracted features for {len(graph.edges)} edges")


# ============================================================================
#              GRAPH TOPOLOGY FEATURES (step 6c)
# ============================================================================

def compute_graph_topology_features(graph: SyntheticGraph) -> None:
    """Compute per-node graph topology features: clustering coefficient, components."""
    logger.info("Computing graph topology features...")

    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)

    # Local clustering coefficient
    for nid, node in graph.nodes.items():
        nbrs = neighbors.get(nid, set())
        k = len(nbrs)
        if k < 2:
            node.clustering_coeff = 0.0
            continue
        links = 0
        nbr_list = list(nbrs)
        for i in range(len(nbr_list)):
            for j in range(i + 1, len(nbr_list)):
                if nbr_list[j] in neighbors.get(nbr_list[i], set()):
                    links += 1
        max_possible = k * (k - 1) / 2
        node.clustering_coeff = links / max_possible if max_possible > 0 else 0.0

    # Connected components via BFS
    visited: Set[str] = set()
    comp_id = 0
    for start_nid in graph.nodes:
        if start_nid in visited:
            continue
        queue = [start_nid]
        component_members: List[str] = []
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            component_members.append(nid)
            for nbr in neighbors.get(nid, set()):
                if nbr not in visited:
                    queue.append(nbr)
        size = len(component_members)
        for nid in component_members:
            graph.nodes[nid].component_id = comp_id
            graph.nodes[nid].component_size = size
        comp_id += 1

    cc_vals = [n.clustering_coeff for n in graph.nodes.values()]
    mean_cc = statistics.mean(cc_vals) if cc_vals else 0.0
    logger.info(f"Graph topology: {comp_id} components, mean clustering={mean_cc:.3f}")


# ============================================================================
#              SEQUENCE COMPLEXITY FEATURES (step 6d)
# ============================================================================

def compute_sequence_complexity_features(graph: SyntheticGraph) -> None:
    """Compute per-node sequence complexity features in-place."""
    logger.info("Computing sequence complexity features...")

    for nid, node in graph.nodes.items():
        seq = node.read_info.sequence
        node.shannon_entropy = _shannon_entropy(seq)
        node.dinucleotide_bias = _dinucleotide_bias(seq)
        max_run, density = _homopolymer_stats(seq)
        node.homopolymer_max_run = max_run
        node.homopolymer_density = density
        node.low_complexity_fraction = _low_complexity_fraction(seq)

    logger.info(f"Computed sequence complexity for {len(graph.nodes)} nodes")


# ============================================================================
#              COVERAGE DISTRIBUTION FEATURES (step 6e)
# ============================================================================

def compute_coverage_distribution_features(graph: SyntheticGraph,
                                            all_reads: List[ReadInfo]) -> None:
    """Compute per-node coverage distribution stats: skewness, kurtosis, CV, percentiles."""
    logger.info("Computing coverage distribution features...")

    # Build sorted read index per chromosome for O(log n) coverage queries
    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)
    chrom_index: Dict[str, Tuple[List[int], List[int]]] = {}
    for chrom, reads in chrom_reads.items():
        chrom_index[chrom] = _build_sorted_read_index(reads)

    for nid, node in graph.nodes.items():
        idx = chrom_index.get(node.read_info.chrom)
        if idx:
            coverages = _coverage_array_for_node_fast(node, idx[0], idx[1])
        else:
            coverages = [0.0]
        n = len(coverages)
        if n < 3:
            continue

        mean_c = statistics.mean(coverages)
        std_c = statistics.stdev(coverages) if n > 1 else 0.0
        node.coverage_cv = std_c / mean_c if mean_c > 0 else 0.0

        sorted_c = sorted(coverages)
        node.coverage_p10 = sorted_c[max(0, n // 10)]
        node.coverage_p90 = sorted_c[min(n - 1, 9 * n // 10)]

        if std_c > 0:
            node.coverage_skewness = (
                sum((c - mean_c) ** 3 for c in coverages) / n) / (std_c ** 3)
            node.coverage_kurtosis = (
                sum((c - mean_c) ** 4 for c in coverages) / n) / (std_c ** 4) - 3.0
        else:
            node.coverage_skewness = 0.0
            node.coverage_kurtosis = 0.0

    logger.info(f"Computed coverage distribution for {len(graph.nodes)} nodes")


# ============================================================================
#              REALISTIC FEATURE COMPUTATION (step 6f)
# ============================================================================

def compute_realistic_features(graph: SyntheticGraph, all_reads: List[ReadInfo],
                                diploid_genome: Any, expected_coverage: float = 30.0) -> None:
    """Replace hardcoded placeholder features with realistic computed values.

    Replaces the 5 previously hardcoded values:
    - mapping_quality (was 60.0): Now based on repeat content + read length + noise
    - allele_frequency (was 0.5): Now based on haplotype coverage imbalance
    - heterozygosity (was 0.0): Now based on SNP/indel positions in read span
    - phase_consistency (was 1.0): Now based on haplotype consistency of neighbors
    - mappability (was 1.0): Now based on repeat mask or kmer uniqueness
    """
    logger.info("Computing realistic feature values (replacing placeholders)...")

    repeat_mask_A = getattr(diploid_genome, 'repeat_mask_A', None)
    repeat_mask_B = getattr(diploid_genome, 'repeat_mask_B', None)

    # Build sorted read indices per (chrom, haplotype) for O(log n) coverage
    chrom_hap_index: Dict[str, Dict[str, Tuple[List[int], List[int]]]] = defaultdict(dict)
    for r in all_reads:
        key = (r.chrom, r.haplotype)
        chrom_hap_index[r.chrom].setdefault(r.haplotype, ([], []))
    # Collect start/end positions
    _hap_starts: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    _hap_ends: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in all_reads:
        _hap_starts[(r.chrom, r.haplotype)].append(r.start_pos)
        _hap_ends[(r.chrom, r.haplotype)].append(r.end_pos)
    # Sort them
    for key in _hap_starts:
        _hap_starts[key].sort()
        _hap_ends[key].sort()
        chrom_hap_index[key[0]][key[1]] = (_hap_starts[key], _hap_ends[key])

    # Build sorted variant position arrays for O(log n) heterozygosity
    snp_list = sorted(getattr(diploid_genome, 'snp_positions', []))
    indel_list = sorted(getattr(diploid_genome, 'indel_positions', []))
    # Merge into single sorted array
    all_variant_positions = sorted(set(snp_list) | set(indel_list))

    for nid, node in graph.nodes.items():
        r = node.read_info
        seq = r.sequence

        # Mapping quality: repeat regions lower MAPQ, longer reads higher MAPQ
        base_mapq = 60.0
        repeat_penalty = node.repeat_fraction * 40.0  # up to -40 for pure repeat
        length_bonus = min(10.0, r.length / 5000.0)   # up to +10 for long reads
        rng = random.Random(hash(nid) % (2**31))
        noise = rng.gauss(0, 2.0)
        node.mapping_quality = max(0.0, min(60.0,
            base_mapq - repeat_penalty + length_bonus + noise))

        # Allele frequency: ratio of this haplotype's coverage to total
        # Uses O(log n) binary search instead of scanning all reads
        mid = (r.start_pos + r.end_pos) // 2
        same_idx = chrom_hap_index.get(r.chrom, {}).get(r.haplotype)
        other_hap = 'B' if r.haplotype == 'A' else 'A'
        diff_idx = chrom_hap_index.get(r.chrom, {}).get(other_hap)
        same_hap = _coverage_at_pos_fast(mid, same_idx[0], same_idx[1]) if same_idx else 0
        diff_hap = _coverage_at_pos_fast(mid, diff_idx[0], diff_idx[1]) if diff_idx else 0
        total = same_hap + diff_hap
        node.allele_frequency = same_hap / total if total > 0 else 0.5
        node.allele_frequency = max(0.0, min(1.0,
            node.allele_frequency + rng.gauss(0, 0.02)))

        # Heterozygosity: fraction of variant positions in read span
        # Uses O(log n) bisect instead of iterating every base position
        left = bisect.bisect_left(all_variant_positions, r.start_pos)
        right = bisect.bisect_right(all_variant_positions, r.end_pos - 1)
        var_count = right - left
        node.heterozygosity = var_count / max(r.length, 1)

        # Mappability: from repeat mask or kmer diversity fallback
        if r.haplotype == 'A' and repeat_mask_A is not None:
            start = min(r.start_pos, len(repeat_mask_A) - 1)
            end = min(r.end_pos, len(repeat_mask_A))
            if end > start:
                unique = sum(1 for i in range(start, end) if not repeat_mask_A[i])
                node.mappability = unique / (end - start)
            else:
                node.mappability = 1.0
        elif r.haplotype == 'B' and repeat_mask_B is not None:
            start = min(r.start_pos, len(repeat_mask_B) - 1)
            end = min(r.end_pos, len(repeat_mask_B))
            if end > start:
                unique = sum(1 for i in range(start, end) if not repeat_mask_B[i])
                node.mappability = unique / (end - start)
            else:
                node.mappability = 1.0
        else:
            node.mappability = max(0.1, min(1.0, _kmer_diversity(seq) + rng.gauss(0, 0.05)))

    # Phase consistency: requires neighbor info from graph edges
    neighbors_by_node: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges:
        neighbors_by_node[edge.source].append(edge.target)
        neighbors_by_node[edge.target].append(edge.source)

    for nid, node in graph.nodes.items():
        nbrs = neighbors_by_node.get(nid, [])
        if not nbrs:
            node.phase_consistency = 0.5  # no neighbors = uncertain
            continue
        same_hap = sum(1 for nbr in nbrs
                       if nbr in graph.nodes and
                       graph.nodes[nbr].read_info.haplotype == node.read_info.haplotype)
        node.phase_consistency = same_hap / len(nbrs)

    # Update edge features after node features are finalized
    for edge in graph.edges:
        n1 = graph.nodes.get(edge.source)
        n2 = graph.nodes.get(edge.target)
        if n1 and n2:
            edge.features['mapping_quality_r1'] = n1.mapping_quality
            edge.features['mapping_quality_r2'] = n2.mapping_quality
            edge.features['mapping_quality'] = (n1.mapping_quality + n2.mapping_quality) / 2.0
            edge.features['clustering_coeff_r1'] = n1.clustering_coeff
            edge.features['clustering_coeff_r2'] = n2.clustering_coeff
            edge.features['component_size'] = float(max(n1.component_size, n2.component_size))
            edge.features['entropy_r1'] = n1.shannon_entropy
            edge.features['entropy_r2'] = n2.shannon_entropy
            edge.features['homopolymer_max_r1'] = float(n1.homopolymer_max_run)
            edge.features['homopolymer_max_r2'] = float(n2.homopolymer_max_run)

    logger.info("Realistic features computed for all nodes and edges")


# ============================================================================
#              HI-C CONTACT FEATURES (step 6b)
# ============================================================================

def compute_hic_contact_features(graph: SyntheticGraph,
                                  hic_reads: List[ReadInfo]) -> Dict[str, Dict[str, Any]]:
    """Compute per-node Hi-C contact features from simulated Hi-C reads."""
    if not hic_reads:
        logger.info("No Hi-C reads available - contact features will be zero")
        return {}

    logger.info(f"Computing Hi-C contact features from {len(hic_reads)} reads...")

    pairs: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in hic_reads:
        base_id = r.read_id.rsplit('_R', 1)[0] if '_R' in r.read_id else r.read_id
        pairs[base_id].append(r)

    BIN_SIZE = 1000
    node_bins: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        r = node.read_info
        start_bin = r.start_pos // BIN_SIZE
        end_bin = r.end_pos // BIN_SIZE
        for b in range(start_bin, end_bin + 1):
            node_bins[(r.chrom, b)].append(nid)

    def _find_overlapping_nodes(chrom: str, start: int, end: int) -> List[str]:
        start_bin = start // BIN_SIZE
        end_bin = end // BIN_SIZE
        candidates = set()
        for b in range(start_bin, end_bin + 1):
            candidates.update(node_bins.get((chrom, b), []))
        return [nid for nid in candidates
                if graph.nodes[nid].read_info.chrom == chrom
                and graph.nodes[nid].read_info.start_pos < end
                and graph.nodes[nid].read_info.end_pos > start]

    intra_counts: Dict[str, int] = defaultdict(int)
    inter_counts: Dict[str, int] = defaultdict(int)

    for base_id, reads in pairs.items():
        if len(reads) < 2:
            continue
        r1, r2 = reads[0], reads[1]
        nodes_r1 = _find_overlapping_nodes(r1.chrom, r1.start_pos, r1.end_pos)
        nodes_r2 = _find_overlapping_nodes(r2.chrom, r2.start_pos, r2.end_pos)

        for n1 in nodes_r1:
            for n2 in nodes_r2:
                if n1 == n2:
                    continue
                hap1 = graph.nodes[n1].read_info.haplotype
                hap2 = graph.nodes[n2].read_info.haplotype
                if hap1 == hap2:
                    intra_counts[n1] += 1
                    intra_counts[n2] += 1
                else:
                    inter_counts[n1] += 1
                    inter_counts[n2] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for nid in graph.nodes:
        intra = intra_counts.get(nid, 0)
        inter = inter_counts.get(nid, 0)
        total = intra + inter
        ratio = intra / total if total > 0 else 0.5
        phase_signal = (intra - inter) / max(total, 1)
        result[nid] = {'intra': intra, 'inter': inter,
                       'ratio': ratio, 'phase_signal': phase_signal}

    total_contacts = sum(intra_counts.values()) + sum(inter_counts.values())
    nodes_with = sum(1 for v in result.values() if v['intra'] + v['inter'] > 0)
    logger.info(f"Hi-C contacts: {total_contacts} total, "
                f"{nodes_with}/{len(graph.nodes)} nodes have contacts")
    return result



# ============================================================================
#              SV REGION FEATURES
# ============================================================================

def compute_sv_region_features(graph: SyntheticGraph, sv_truth_table: List[Any],
                                all_reads: List[ReadInfo], genome_length: int,
                                metadata: GenomeMetadata) -> List[Dict[str, Any]]:
    """Compute SV detection features with 3-5x hard negatives and coverage distribution."""
    logger.info("Computing SV region features...")

    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)

    sv_rows: List[Dict[str, Any]] = []

    def _region_features(region_reads, region_start, region_end, sv_type, is_positive):
        coverages = []
        window = region_end - region_start
        for pos in range(region_start, region_end, max(1, window // 20)):
            coverages.append(sum(1 for r in region_reads if r.start_pos <= pos < r.end_pos))
        if not coverages:
            coverages = [0]

        hap_a = sum(1 for r in region_reads if r.haplotype == 'A')
        hap_b = sum(1 for r in region_reads if r.haplotype == 'B')
        total_hap = hap_a + hap_b
        allele_balance = min(hap_a, hap_b) / max(total_hap, 1)
        combined_seq = ''.join(r.sequence for r in region_reads[:20])

        cov_mean = statistics.mean(coverages)
        cov_std = statistics.stdev(coverages) if len(coverages) > 1 else 0.0
        cov_median = statistics.median(coverages)
        n = len(coverages)
        sorted_c = sorted(coverages)
        cov_cv = cov_std / cov_mean if cov_mean > 0 else 0.0
        cov_p10 = sorted_c[max(0, n // 10)]
        cov_p90 = sorted_c[min(n - 1, 9 * n // 10)]
        cov_skew = 0.0
        cov_kurt = 0.0
        if cov_std > 0 and n >= 3:
            cov_skew = (sum((c - cov_mean) ** 3 for c in coverages) / n) / (cov_std ** 3)
            cov_kurt = (sum((c - cov_mean) ** 4 for c in coverages) / n) / (cov_std ** 4) - 3.0

        # Realistic mapping quality from graph nodes
        region_mapqs = [graph.nodes[r.read_id].mapping_quality
                        for r in region_reads if r.read_id in graph.nodes]
        mapq = statistics.mean(region_mapqs) if region_mapqs else 40.0

        # Phase switch rate from adjacent reads
        sorted_reads = sorted(region_reads, key=lambda r: r.start_pos)
        switches = sum(1 for i in range(len(sorted_reads) - 1)
                       if sorted_reads[i].haplotype != sorted_reads[i + 1].haplotype)
        phase_switch_rate = switches / max(len(sorted_reads) - 1, 1)

        # Hi-C disruption score
        hic_scores = [graph.nodes[r.read_id].hic_intra + graph.nodes[r.read_id].hic_inter
                      for r in region_reads if r.read_id in graph.nodes]
        hic_disruption = 0.0
        if len(hic_scores) >= 4:
            mid_idx = len(hic_scores) // 2
            left = statistics.mean(hic_scores[:mid_idx])
            right = statistics.mean(hic_scores[mid_idx:])
            hic_disruption = abs(left - right) / max(left + right, 1)

        # ── v2.1 SV-specific features (S4) ──────────────────────────────

        # Depth ratio: left flank coverage / right flank coverage
        # Deletions and duplications cause asymmetric coverage drops/gains
        flank_size = max(window // 4, 200)
        left_flank_reads = [r for r in chrom_reads.get(
            sorted_reads[0].chrom if sorted_reads else 'chr1', [])
            if r.end_pos >= region_start - flank_size and r.start_pos < region_start]
        right_flank_reads = [r for r in chrom_reads.get(
            sorted_reads[0].chrom if sorted_reads else 'chr1', [])
            if r.start_pos <= region_end + flank_size and r.end_pos > region_end]
        left_cov = len(left_flank_reads) if left_flank_reads else 0
        right_cov = len(right_flank_reads) if right_flank_reads else 0
        depth_ratio_flank = left_cov / max(right_cov, 1) if right_cov > 0 else (
            2.0 if left_cov > 0 else 1.0)

        # Split-read count: reads that start or end very close to breakpoint
        # (simulates soft-clipped / split-aligned reads)
        bp_tolerance = max(50, window // 20)
        split_read_count = sum(
            1 for r in region_reads
            if (abs(r.end_pos - region_start) < bp_tolerance or
                abs(r.start_pos - region_end) < bp_tolerance or
                abs(r.start_pos - region_start) < bp_tolerance or
                abs(r.end_pos - region_end) < bp_tolerance)
        )

        # Clip fraction: fraction of reads that partially overlap breakpoint
        # (proxy for soft-clipping signal)
        partial_reads = sum(
            1 for r in region_reads
            if (r.start_pos < region_start and r.end_pos < region_end) or
               (r.start_pos > region_start and r.end_pos > region_end)
        )
        clip_fraction = partial_reads / max(len(region_reads), 1)

        # Bubble size: assembly graph branching topology at region
        region_node_ids = [r.read_id for r in region_reads if r.read_id in graph.nodes]
        branch_nodes = sum(
            1 for nid in region_node_ids
            if (graph.nodes[nid].in_degree + graph.nodes[nid].out_degree) > 2
        )
        bubble_size = branch_nodes / max(len(region_node_ids), 1)

        # Path divergence: how many distinct in/out paths exist at region
        total_in = sum(graph.nodes[nid].in_degree for nid in region_node_ids
                       if nid in graph.nodes)
        total_out = sum(graph.nodes[nid].out_degree for nid in region_node_ids
                        if nid in graph.nodes)
        path_divergence = abs(total_in - total_out) / max(total_in + total_out, 1)

        # UL spanning: does any ultra-long read fully span the SV region?
        ul_spanning = 1.0 if any(
            r.start_pos <= region_start and r.end_pos >= region_end
            for r in region_reads if r.technology in ('ultra_long', 'ul')
        ) else 0.0

        # Coverage drop magnitude: (flank_avg - region_cov) / flank_avg
        flank_avg = (left_cov + right_cov) / 2.0 if (left_cov + right_cov) > 0 else cov_mean
        coverage_drop_magnitude = (flank_avg - cov_mean) / max(flank_avg, 1)

        # Orientation switch rate: strand changes across the region
        if len(sorted_reads) >= 2:
            strand_switches = sum(
                1 for i in range(len(sorted_reads) - 1)
                if sorted_reads[i].strand != sorted_reads[i + 1].strand
            )
            orientation_switch_rate = strand_switches / max(len(sorted_reads) - 1, 1)
        else:
            orientation_switch_rate = 0.0

        row = {
            'coverage_mean': cov_mean, 'coverage_std': cov_std, 'coverage_median': cov_median,
            'gc_content': _gc_content(combined_seq) if combined_seq else 0.42,
            'repeat_fraction': _repeat_fraction_estimate(combined_seq) if combined_seq else 0.0,
            'kmer_diversity': _kmer_diversity(combined_seq) if combined_seq else 0.5,
            'branching_complexity': sum(
                graph.nodes[r.read_id].in_degree + graph.nodes[r.read_id].out_degree
                for r in region_reads if r.read_id in graph.nodes) / max(len(region_reads), 1),
            'hic_disruption_score': hic_disruption,
            'ul_support': sum(1 for r in region_reads if r.technology in ('ultra_long', 'ul')),
            'mapping_quality': mapq,
            'region_length': float(window),
            'breakpoint_precision': 1.0 if is_positive else 0.0,
            'allele_balance': allele_balance,
            'phase_switch_rate': phase_switch_rate,
            # v2.0: coverage distribution features
            'coverage_cv': cov_cv, 'coverage_skewness': cov_skew,
            'coverage_kurtosis': cov_kurt, 'coverage_p10': cov_p10, 'coverage_p90': cov_p90,
            # v2.1: SV-specific features (S4)
            'depth_ratio_flank': depth_ratio_flank,
            'split_read_count': float(split_read_count),
            'clip_fraction': clip_fraction,
            'bubble_size': bubble_size,
            'path_divergence': path_divergence,
            'ul_spanning': ul_spanning,
            'coverage_drop_magnitude': coverage_drop_magnitude,
            'orientation_switch_rate': orientation_switch_rate,
            'sv_type': sv_type,
        }
        for col, val in zip(METADATA_COLUMNS, metadata.as_row()):
            row[col] = val
        return row

    # Positive examples from truth table
    for sv in sv_truth_table:
        region_reads = [r for r in chrom_reads.get(sv.chrom, [])
                        if not (r.end_pos < sv.pos or r.start_pos > sv.end)]
        sv_rows.append(_region_features(region_reads, sv.pos, sv.end, sv.sv_type.value, True))

    # 3-5x hard negatives per positive (improved from 1:1)
    rng = random.Random(12345)
    for sv in sv_truth_table:
        window = max(sv.end - sv.pos, 1000)
        num_negatives = rng.randint(3, 5)
        for neg_idx in range(num_negatives):
            if neg_idx < 2:  # hard negatives: near SV boundaries
                offset = rng.choice([-1, 1]) * rng.randint(window // 2, window * 2)
                rand_start = max(0, sv.pos + offset)
            else:  # easy negatives: random location
                rand_start = rng.randint(0, max(1, genome_length - window))
            rand_end = rand_start + window
            # Skip if overlaps with any true SV
            if any(not (s.end < rand_start or s.pos > rand_end) for s in sv_truth_table):
                continue
            neg_reads = [r for r in chrom_reads.get(sv.chrom, [])
                         if not (r.end_pos < rand_start or r.start_pos > rand_end)]
            sv_rows.append(_region_features(neg_reads, rand_start, rand_end, 'none', False))

    pos_count = sum(1 for r in sv_rows if r['sv_type'] != 'none')
    neg_count = sum(1 for r in sv_rows if r['sv_type'] == 'none')
    logger.info(f"Computed SV features: {len(sv_rows)} rows ({pos_count} positive, {neg_count} negative)")
    return sv_rows


# ============================================================================
#              UL ROUTE FEATURES
# ============================================================================

def compute_ul_route_features(graph: SyntheticGraph, ul_reads: List[ReadInfo],
                               all_reads: List[ReadInfo],
                               metadata: GenomeMetadata) -> List[Dict[str, Any]]:
    """Compute UL routing features with continuous route_score (replaces binary 0/1)."""
    logger.info("Computing UL route features...")

    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)

    rows: List[Dict[str, Any]] = []

    for ul in ul_reads:
        span_reads = [r for r in chrom_reads.get(ul.chrom, [])
                      if r.haplotype == ul.haplotype
                      and not (r.end_pos < ul.start_pos or r.start_pos > ul.end_pos)
                      and r.read_id != ul.read_id]
        span_reads.sort(key=lambda r: r.start_pos)
        path_nodes = [r.read_id for r in span_reads if r.read_id in graph.nodes]

        coverages = [graph.nodes[nid].coverage for nid in path_nodes if nid in graph.nodes] or [0.0]
        branches = [graph.nodes[nid].in_degree + graph.nodes[nid].out_degree
                     for nid in path_nodes if nid in graph.nodes] or [0]

        gaps = []
        for i in range(len(span_reads) - 1):
            gap = span_reads[i + 1].start_pos - span_reads[i].end_pos
            if gap > 0:
                gaps.append(gap)

        path_mapqs = [graph.nodes[nid].mapping_quality for nid in path_nodes if nid in graph.nodes]
        avg_mapq = statistics.mean(path_mapqs) if path_mapqs else 40.0
        kmer_divs = [_kmer_diversity(r.sequence) for r in span_reads[:20]] if span_reads else [0.5]
        kmer_consistency = statistics.mean(kmer_divs)

        if span_reads:
            same_strand = sum(1 for r in span_reads if r.strand == ul.strand)
            orientation_consistency = same_strand / len(span_reads)
        else:
            orientation_consistency = 0.5

        # Continuous route score: 0.5*completeness + 0.3*gap_penalty + 0.2*uniformity
        if len(path_nodes) == 0:
            route_score = 0.0
        else:
            covered_bp = 0
            for r in span_reads:
                ovl_start = max(r.start_pos, ul.start_pos)
                ovl_end = min(r.end_pos, ul.end_pos)
                if ovl_end > ovl_start:
                    covered_bp += ovl_end - ovl_start
            completeness = min(1.0, covered_bp / max(ul.length, 1))
            total_gap = sum(gaps) if gaps else 0
            gap_penalty = max(0.0, 1.0 - total_gap / max(ul.length, 1))
            cov_cv = (statistics.stdev(coverages) / statistics.mean(coverages)
                      if len(coverages) > 1 and statistics.mean(coverages) > 0 else 0.0)
            uniformity = max(0.0, 1.0 - cov_cv)
            route_score = 0.5 * completeness + 0.3 * gap_penalty + 0.2 * uniformity

        row = {
            'path_length': float(ul.length),
            'num_branches': float(sum(1 for b in branches if b > 2)),
            'coverage_mean': statistics.mean(coverages),
            'coverage_std': statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
            'sequence_identity': max(0.9, 1.0 - (getattr(ul, 'error_rate', 0.05) or 0.05)),
            'mapping_quality': avg_mapq,
            'num_gaps': float(len(gaps)),
            'gap_size_mean': statistics.mean(gaps) if gaps else 0.0,
            'kmer_consistency': kmer_consistency,
            'orientation_consistency': orientation_consistency,
            'ul_span': float(ul.end_pos - ul.start_pos),
            'route_complexity': float(len(path_nodes)),
            'route_score': route_score,
            'read_id': ul.read_id,
            'correct_path': path_nodes,
        }
        for col, val in zip(METADATA_COLUMNS, metadata.as_row()):
            row[col] = val
        rows.append(row)

    logger.info(f"Computed UL route features for {len(rows)} reads")
    return rows


# ============================================================================
#              GROUND-TRUTH LABELLING
# ============================================================================

def label_graph(graph: SyntheticGraph, sv_truth_table: List[Any],
                max_true_distance: int = 10_000) -> None:
    """Apply ground-truth labels to edges and nodes.
    
    v2.0 fix: REPEAT label triggers if EITHER read is in a repeat region
    (previously required BOTH reads to be in repeat, which was incorrect).
    """
    logger.info("Labelling graph edges and nodes...")

    for edge in graph.edges:
        r1 = graph.nodes[edge.source].read_info
        r2 = graph.nodes[edge.target].read_info

        if edge.overlap.is_noise:
            edge.label = 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = 'CHIMERIC'
            continue

        if r1.chrom != r2.chrom:
            edge.label = 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = 'CHIMERIC'
            continue

        crosses_sv = False
        for sv in sv_truth_table:
            if hasattr(sv, 'chrom') and sv.chrom != r1.chrom:
                continue
            sv_start = sv.pos if hasattr(sv, 'pos') else 0
            sv_end = sv.end if hasattr(sv, 'end') else 0
            span_min = min(r1.start_pos, r2.start_pos)
            span_max = max(r1.end_pos, r2.end_pos)
            if span_min < sv_start < span_max or span_min < sv_end < span_max:
                crosses_sv = True
                break

        if crosses_sv:
            edge.label = 'SV_BREAK'
            graph.edge_labels[(edge.source, edge.target)] = 'SV_BREAK'
            continue

        # v2.0 fix: EITHER read in repeat (was: both reads in repeat)
        if r1.is_repeat or r2.is_repeat:
            edge.label = 'REPEAT'
            graph.edge_labels[(edge.source, edge.target)] = 'REPEAT'
            continue

        if r1.haplotype != r2.haplotype:
            distance = abs(r1.start_pos - r2.start_pos)
            edge.label = 'ALLELIC' if distance < max_true_distance else 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = edge.label
            continue

        distance = abs(r2.start_pos - r1.end_pos)
        edge.label = 'TRUE' if distance <= max_true_distance else 'CHIMERIC'
        graph.edge_labels[(edge.source, edge.target)] = edge.label

    # Node labels
    for nid, node in graph.nodes.items():
        r = node.read_info
        if r.is_repeat:
            graph.node_labels[nid] = 'REPEAT'
        elif r.haplotype == 'A':
            graph.node_labels[nid] = 'HAP_A'
        elif r.haplotype == 'B':
            graph.node_labels[nid] = 'HAP_B'
        else:
            graph.node_labels[nid] = 'UNKNOWN'

    # Correct paths (ground truth traversals)
    by_hap: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        r = node.read_info
        by_hap[(r.haplotype, r.chrom)].append((r.start_pos, nid))
    for key in by_hap:
        by_hap[key].sort()
        graph.correct_paths.append([nid for _, nid in by_hap[key]])

    edge_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.edge_labels.values():
        edge_counts[lbl] += 1
    node_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.node_labels.values():
        node_counts[lbl] += 1

    logger.info("Edge labels: " + ", ".join(f"{k}={v}" for k, v in sorted(edge_counts.items())))
    logger.info("Node labels: " + ", ".join(f"{k}={v}" for k, v in sorted(node_counts.items())))
    logger.info(f"Correct paths: {len(graph.correct_paths)}")


# ============================================================================
#              EXPORT - FEATURE COLUMN DEFINITIONS
# ============================================================================

EDGE_AI_FEATURES = [
    'overlap_length', 'overlap_identity', 'read1_length', 'read2_length',
    'coverage_r1', 'coverage_r2', 'gc_content_r1', 'gc_content_r2',
    'repeat_fraction_r1', 'repeat_fraction_r2',
    'kmer_diversity_r1', 'kmer_diversity_r2',
    'branching_factor_r1', 'branching_factor_r2',
    'hic_support', 'mapping_quality_r1', 'mapping_quality_r2',
    # v2.0: graph topology
    'clustering_coeff_r1', 'clustering_coeff_r2', 'component_size',
    # v2.0: sequence complexity
    'entropy_r1', 'entropy_r2', 'homopolymer_max_r1', 'homopolymer_max_r2',
]

EDGE_AI_PROVENANCE = [
    'node_id_r1', 'node_id_r2',
    'read1_haplotype', 'read2_haplotype',
    'genomic_distance', 'is_repeat_region',
]

PATH_GNN_FEATURES = [
    'overlap_length', 'overlap_identity', 'coverage_consistency',
    'gc_similarity', 'repeat_match', 'branching_score',
    'path_support', 'hic_contact', 'mapping_quality',
    'kmer_match', 'sequence_complexity', 'orientation_score',
    'distance_score', 'topology_score', 'ul_support', 'sv_evidence',
]

PATH_GNN_PROVENANCE = [
    'node_id_r1', 'node_id_r2',
    'read1_haplotype', 'read2_haplotype',
    'genomic_distance', 'is_repeat_region',
]

NODE_SIGNAL_FEATURES = [
    'coverage', 'gc_content', 'repeat_fraction', 'kmer_diversity',
    'branching_factor', 'hic_contact_density', 'allele_frequency',
    'heterozygosity', 'phase_consistency', 'mappability',
    'hic_intra_contacts', 'hic_inter_contacts',
    'hic_contact_ratio', 'hic_phase_signal',
    # v2.0: graph topology
    'clustering_coeff', 'component_size',
    # v2.0: sequence complexity
    'shannon_entropy', 'dinucleotide_bias',
    'homopolymer_max_run', 'homopolymer_density', 'low_complexity_fraction',
    # v2.0: coverage distribution
    'coverage_skewness', 'coverage_kurtosis', 'coverage_cv',
    'coverage_p10', 'coverage_p90',
]

NODE_PROVENANCE = [
    'node_id', 'read_haplotype', 'read_start_pos', 'read_end_pos',
    'read_length', 'is_in_repeat', 'read_technology',
]

UL_ROUTE_FEATURES = [
    'path_length', 'num_branches', 'coverage_mean', 'coverage_std',
    'sequence_identity', 'mapping_quality', 'num_gaps', 'gap_size_mean',
    'kmer_consistency', 'orientation_consistency', 'ul_span', 'route_complexity',
]

SV_DETECT_FEATURES = [
    'coverage_mean', 'coverage_std', 'coverage_median',
    'gc_content', 'repeat_fraction', 'kmer_diversity',
    'branching_complexity', 'hic_disruption_score',
    'ul_support', 'mapping_quality',
    'region_length', 'breakpoint_precision',
    'allele_balance', 'phase_switch_rate',
    # v2.0: coverage distribution
    'coverage_cv', 'coverage_skewness', 'coverage_kurtosis',
    'coverage_p10', 'coverage_p90',
    # v2.1: SV-specific features (S4 improvement plan)
    'depth_ratio_flank',          # left_flank_cov / right_flank_cov
    'split_read_count',           # reads with partial alignment at breakpoint
    'clip_fraction',              # fraction of reads with soft-clipping
    'bubble_size',                # graph topology: bubble/branch size at region
    'path_divergence',            # graph topology: path count divergence
    'ul_spanning',                # binary: any UL read spans entire SV event
    'coverage_drop_magnitude',    # (flank_cov - region_cov) / flank_cov
    'orientation_switch_rate',    # strand orientation changes across region
]



# ============================================================================
#              EXPORT FUNCTIONS
# ============================================================================

def export_edge_training_csv(graph: SyntheticGraph, output_dir: Path,
                              genome_idx: int = 0,
                              metadata: Optional[GenomeMetadata] = None) -> Path:
    """Export EdgeAI training data with provenance and metadata columns."""
    out = output_dir / f'edge_ai_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = EDGE_AI_FEATURES + EDGE_AI_PROVENANCE + ['label']
        if metadata:
            header = METADATA_COLUMNS + header
        writer.writerow(header)
        for edge in graph.edges:
            r1 = graph.nodes[edge.source].read_info
            r2 = graph.nodes[edge.target].read_info
            row = [edge.features.get(feat, 0.0) for feat in EDGE_AI_FEATURES]
            row.extend([edge.source, edge.target,
                        r1.haplotype, r2.haplotype,
                        abs(r2.start_pos - r1.end_pos),
                        1 if (r1.is_repeat or r2.is_repeat) else 0])
            row.append(edge.label)
            if metadata:
                row = list(metadata.as_row()) + row
            writer.writerow(row)
    logger.info(f"  EdgeAI CSV: {out}  ({len(graph.edges)} rows)")
    return out


def export_path_gnn_training_csv(graph: SyntheticGraph, output_dir: Path,
                                  genome_idx: int = 0,
                                  metadata: Optional[GenomeMetadata] = None) -> Path:
    """Export PathGNN training data with provenance and metadata columns."""
    correct_edge_set: Set[Tuple[str, str]] = set()
    for path in graph.correct_paths:
        for i in range(len(path) - 1):
            correct_edge_set.add((path[i], path[i + 1]))

    out = output_dir / f'path_gnn_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = PATH_GNN_FEATURES + PATH_GNN_PROVENANCE + ['in_correct_path']
        if metadata:
            header = METADATA_COLUMNS + header
        writer.writerow(header)
        for edge in graph.edges:
            r1 = graph.nodes[edge.source].read_info
            r2 = graph.nodes[edge.target].read_info
            row = [edge.features.get(feat, 0.0) for feat in PATH_GNN_FEATURES]
            row.extend([edge.source, edge.target,
                        r1.haplotype, r2.haplotype,
                        abs(r2.start_pos - r1.end_pos),
                        1 if (r1.is_repeat or r2.is_repeat) else 0])
            row.append(1 if (edge.source, edge.target) in correct_edge_set else 0)
            if metadata:
                row = list(metadata.as_row()) + row
            writer.writerow(row)
    logger.info(f"  PathGNN CSV: {out}  ({len(graph.edges)} rows)")
    return out


def export_node_training_csv(graph: SyntheticGraph, output_dir: Path,
                              genome_idx: int = 0,
                              metadata: Optional[GenomeMetadata] = None) -> Path:
    """Export DiploidAI training data with provenance and metadata columns."""
    out = output_dir / f'diploid_ai_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = NODE_SIGNAL_FEATURES + NODE_PROVENANCE + ['haplotype_label']
        if metadata:
            header = METADATA_COLUMNS + header
        writer.writerow(header)
        for nid, node in graph.nodes.items():
            r = node.read_info
            kd = _kmer_diversity(r.sequence)
            bf = node.in_degree + node.out_degree
            hic_density = node.hic_intra + node.hic_inter

            row = [
                node.coverage, node.gc_content, node.repeat_fraction, kd,
                float(bf), float(hic_density),
                node.allele_frequency, node.heterozygosity,
                node.phase_consistency, node.mappability,
                float(node.hic_intra), float(node.hic_inter),
                float(node.hic_ratio), float(node.hic_phase),
                # v2.0: graph topology
                node.clustering_coeff, float(node.component_size),
                # v2.0: sequence complexity
                node.shannon_entropy, node.dinucleotide_bias,
                float(node.homopolymer_max_run), node.homopolymer_density,
                node.low_complexity_fraction,
                # v2.0: coverage distribution
                node.coverage_skewness, node.coverage_kurtosis, node.coverage_cv,
                node.coverage_p10, node.coverage_p90,
            ]
            # Provenance columns
            row.extend([nid, r.haplotype, r.start_pos, r.end_pos,
                        r.length, 1 if r.is_repeat else 0, r.technology])
            row.append(graph.node_labels.get(nid, 'UNKNOWN'))
            if metadata:
                row = list(metadata.as_row()) + row
            writer.writerow(row)
    logger.info(f"  DiploidAI CSV: {out}  ({len(graph.nodes)} rows)")
    return out


def export_ul_route_training_csv(ul_rows: List[Dict[str, Any]], output_dir: Path,
                                  genome_idx: int = 0) -> Path:
    """Export UL routing training data with metadata columns."""
    out = output_dir / f'ul_route_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = METADATA_COLUMNS + UL_ROUTE_FEATURES + ['route_score']
        writer.writerow(header)
        for row in ul_rows:
            vals = [row.get(col, '') for col in METADATA_COLUMNS]
            vals.extend([row.get(feat, 0.0) for feat in UL_ROUTE_FEATURES])
            vals.append(row.get('route_score', 0.0))
            writer.writerow(vals)
    logger.info(f"  UL Route CSV: {out}  ({len(ul_rows)} rows)")
    return out


def export_sv_training_csv(sv_rows: List[Dict[str, Any]], output_dir: Path,
                            genome_idx: int = 0) -> Path:
    """Export SV detection training data with metadata columns."""
    out = output_dir / f'sv_detect_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = METADATA_COLUMNS + SV_DETECT_FEATURES + ['sv_type']
        writer.writerow(header)
        for row in sv_rows:
            vals = [row.get(col, '') for col in METADATA_COLUMNS]
            vals.extend([row.get(feat, 0.0) for feat in SV_DETECT_FEATURES])
            vals.append(row.get('sv_type', 'none'))
            writer.writerow(vals)
    logger.info(f"  SV Detect CSV: {out}  ({len(sv_rows)} rows)")
    return out


def export_gfa(graph: SyntheticGraph, output_dir: Path, genome_idx: int = 0) -> Path:
    """Export graph in GFA v1 format."""
    out = output_dir / f'overlap_graph_g{genome_idx:04d}.gfa'
    with open(out, 'w') as f:
        f.write("H\tVN:Z:1.0\n")
        for nid, node in graph.nodes.items():
            seq = node.read_info.sequence[:50] + '*' if len(node.read_info.sequence) > 50 else node.read_info.sequence
            f.write(f"S\t{nid}\t{seq}\tLN:i:{node.read_info.length}\n")
        for edge in graph.edges:
            cigar = f"{edge.overlap.overlap_length}M"
            f.write(f"L\t{edge.source}\t+\t{edge.target}\t+\t{cigar}\n")
    logger.info(f"  GFA: {out}")
    return out


# ============================================================================
#              MAIN PIPELINE
# ============================================================================

def reads_to_read_infos(simulated_reads: List[Any], technology: str = 'hifi') -> List[ReadInfo]:
    """Convert backend SimulatedRead / SimulatedReadPair objects to ReadInfo."""
    infos: List[ReadInfo] = []
    for read in simulated_reads:
        if hasattr(read, 'read1') and hasattr(read, 'read2'):
            for suffix, sr in [('_R1', read.read1), ('_R2', read.read2)]:
                infos.append(ReadInfo(
                    read_id=sr.read_id + suffix if not sr.read_id.endswith(suffix) else sr.read_id,
                    sequence=sr.sequence, quality=sr.quality,
                    haplotype=sr.haplotype or 'A', chrom=sr.chrom,
                    start_pos=sr.start_pos, end_pos=sr.end_pos,
                    strand=sr.strand, technology=technology))
        else:
            infos.append(ReadInfo(
                read_id=read.read_id, sequence=read.sequence, quality=read.quality,
                haplotype=read.haplotype or 'A', chrom=read.chrom,
                start_pos=read.start_pos, end_pos=read.end_pos,
                strand=read.strand, technology=technology))
    return infos


def generate_graph_training_data(
    all_simulated_reads: Dict[str, List[Any]],
    diploid_genome: Any,
    sv_truth_table: List[Any],
    graph_config: GraphTrainingConfig,
    output_dir: Path,
    genome_idx: int = 0,
    genome_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end graph training data generation for one genome.
    
    Schema v2.0: Includes metadata, provenance, graph topology, sequence
    complexity, coverage distribution, and realistic feature values.
    """
    graph_dir = output_dir / 'graph_training'
    graph_dir.mkdir(parents=True, exist_ok=True)

    gm = genome_metadata or {}
    genome_len = len(diploid_genome.hapA) if hasattr(diploid_genome, 'hapA') else 1_000_000

    # Determine primary technology and coverage
    primary_tech = 'hifi'
    primary_coverage = 30.0
    for tech in ['hifi', 'ont', 'illumina']:
        if tech in all_simulated_reads and all_simulated_reads[tech]:
            primary_tech = tech
            total_bases = sum(getattr(r, 'end_pos', 0) - getattr(r, 'start_pos', 0)
                              for r in all_simulated_reads[tech])
            primary_coverage = total_bases / max(genome_len, 1)
            break

    # Compute genome-level metadata
    snp_positions = getattr(diploid_genome, 'snp_positions', [])
    het_rate = len(snp_positions) / max(genome_len, 1) if snp_positions else 0.001
    hapA_seq = getattr(diploid_genome, 'hapA', '')
    global_gc = _gc_content(hapA_seq[:100000]) if hapA_seq else 0.42
    global_repeat = _repeat_fraction_estimate(hapA_seq[:50000]) if hapA_seq else 0.30
    tech_error_rates = {'hifi': 0.001, 'ont': 0.05, 'illumina': 0.001, 'ultra_long': 0.05}
    error_rate = tech_error_rates.get(primary_tech, 0.01)

    metadata = GenomeMetadata(
        genome_id=gm.get('genome_idx', genome_idx),
        genome_size=genome_len,
        chromosome_id=gm.get('chromosome_id', 'chr1'),
        read_technology=primary_tech,
        coverage_depth=round(primary_coverage, 1),
        error_rate=error_rate,
        ploidy=gm.get('ploidy', 'diploid'),
        gc_content_global=round(global_gc, 4),
        repeat_density_global=round(global_repeat, 4),
        heterozygosity_rate=round(het_rate, 6),
        random_seed=gm.get('random_seed', genome_idx),
        generator_version=GENERATOR_VERSION,
        schema_version=SCHEMA_VERSION,
    )

    # 1. Convert reads to ReadInfo
    # Hi-C reads are short (150 bp) paired-end fragments used ONLY for
    # contact-based features — they cannot form valid overlaps (min 500 bp)
    # and would massively inflate the O(n²) overlap detection.  Keep them
    # in a separate list and exclude them from the overlap graph.
    all_read_infos: List[ReadInfo] = []   # long reads only (graph nodes)
    ul_read_infos: List[ReadInfo] = []
    hic_read_infos: List[ReadInfo] = []
    tech_map = {'illumina': 'illumina', 'hifi': 'hifi', 'ont': 'ont',
                'ultra_long': 'ultra_long', 'hic': 'hic', 'ancient_dna': 'ancient_dna'}
    for tech, reads in all_simulated_reads.items():
        infos = reads_to_read_infos(reads, technology=tech_map.get(tech, tech))
        if tech == 'hic':
            hic_read_infos.extend(infos)
        else:
            all_read_infos.extend(infos)
        if tech in ('ultra_long', 'ul'):
            ul_read_infos.extend(infos)

    logger.info(f"Total reads: {len(all_read_infos)} long-read + "
                f"{len(hic_read_infos)} Hi-C (UL: {len(ul_read_infos)})")

    # 2. Coverage subsampling
    if graph_config.max_coverage_for_graph and len(all_read_infos) > 100:
        total_bases = sum(r.length for r in all_read_infos)
        current_cov = total_bases / genome_len
        if current_cov > graph_config.max_coverage_for_graph:
            keep_frac = graph_config.max_coverage_for_graph / current_cov
            rng = random.Random(42)
            all_read_infos = [r for r in all_read_infos if rng.random() < keep_frac]
            logger.info(f"Subsampled to ~{graph_config.max_coverage_for_graph:.0f}x")

    # 3. Detect overlaps
    overlaps = detect_overlaps(all_read_infos, min_overlap_bp=graph_config.min_overlap_bp,
                               min_identity=graph_config.min_overlap_identity,
                               max_overhang_fraction=graph_config.max_overhang_fraction)

    # 4. Inject noise edges
    if graph_config.add_noise_edges:
        noise = inject_noise_edges(all_read_infos, overlaps, fraction=graph_config.noise_edge_fraction)
        overlaps.extend(noise)

    # 5. Build graph
    graph = build_overlap_graph(all_read_infos, overlaps)

    # 6a. Feature extraction (core node + edge features)
    if graph_config.compute_features:
        extract_node_features(graph, all_read_infos, genome_len)
        extract_edge_features(graph, all_read_infos)

    # 6b. Hi-C contact features
    hic_contacts = compute_hic_contact_features(graph, hic_read_infos)
    for nid in graph.nodes:
        cdata = hic_contacts.get(nid, {})
        graph.nodes[nid].hic_intra = cdata.get('intra', 0)
        graph.nodes[nid].hic_inter = cdata.get('inter', 0)
        graph.nodes[nid].hic_ratio = cdata.get('ratio', 0.5)
        graph.nodes[nid].hic_phase = cdata.get('phase_signal', 0.0)
    for edge in graph.edges:
        src_c = hic_contacts.get(edge.source, {})
        tgt_c = hic_contacts.get(edge.target, {})
        edge.features['hic_support'] = (
            src_c.get('intra', 0) + tgt_c.get('intra', 0)) / max(
            src_c.get('intra', 0) + tgt_c.get('intra', 0) +
            src_c.get('inter', 0) + tgt_c.get('inter', 0), 1)
        edge.features['hic_contact'] = edge.features['hic_support']

    # 6c. Graph topology features (clustering coefficient, components)
    compute_graph_topology_features(graph)

    # 6d. Sequence complexity features (entropy, homopolymers, etc.)
    compute_sequence_complexity_features(graph)

    # 6e. Coverage distribution features (skewness, kurtosis, CV, percentiles)
    compute_coverage_distribution_features(graph, all_read_infos)

    # 6f. Realistic feature values (replaces 5 hardcoded placeholders)
    compute_realistic_features(graph, all_read_infos, diploid_genome)

    # 7. Ground-truth labelling
    label_graph(graph, sv_truth_table)

    # 8. SV region features
    sv_rows: List[Dict[str, Any]] = []
    if graph_config.label_svs and sv_truth_table:
        sv_rows = compute_sv_region_features(graph, sv_truth_table, all_read_infos, genome_len, metadata)

    # 9. UL route features
    ul_rows: List[Dict[str, Any]] = []
    if graph_config.label_ul_routes and ul_read_infos:
        ul_rows = compute_ul_route_features(graph, ul_read_infos, all_read_infos, metadata)

    # 10. Export CSVs and GFA
    output_files: Dict[str, str] = {}
    if graph_config.label_edges:
        p = export_edge_training_csv(graph, graph_dir, genome_idx, metadata)
        output_files['edge_ai_csv'] = str(p)
    if graph_config.label_paths:
        p = export_path_gnn_training_csv(graph, graph_dir, genome_idx, metadata)
        output_files['path_gnn_csv'] = str(p)
    if graph_config.label_nodes:
        p = export_node_training_csv(graph, graph_dir, genome_idx, metadata)
        output_files['diploid_ai_csv'] = str(p)
    if graph_config.label_ul_routes and ul_rows:
        p = export_ul_route_training_csv(ul_rows, graph_dir, genome_idx)
        output_files['ul_route_csv'] = str(p)
    if graph_config.label_svs and sv_rows:
        p = export_sv_training_csv(sv_rows, graph_dir, genome_idx)
        output_files['sv_detect_csv'] = str(p)
    if graph_config.export_gfa:
        p = export_gfa(graph, graph_dir, genome_idx)
        output_files['gfa'] = str(p)

    # Summary JSON
    edge_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.edge_labels.values():
        edge_counts[lbl] += 1
    node_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.node_labels.values():
        node_counts[lbl] += 1

    summary = {
        'schema_version': SCHEMA_VERSION,
        'generator_version': GENERATOR_VERSION,
        'num_reads': len(all_read_infos),
        'num_ul_reads': len(ul_read_infos),
        'num_hic_reads': len(hic_read_infos),
        'num_overlaps': len(overlaps),
        'num_nodes': len(graph.nodes),
        'num_edges': len(graph.edges),
        'num_correct_paths': len(graph.correct_paths),
        'edge_label_distribution': dict(edge_counts),
        'node_label_distribution': dict(node_counts),
        'num_sv_training_rows': len(sv_rows),
        'num_ul_training_rows': len(ul_rows),
        'metadata': {
            'genome_id': metadata.genome_id,
            'genome_size': metadata.genome_size,
            'coverage_depth': metadata.coverage_depth,
            'gc_content_global': metadata.gc_content_global,
            'repeat_density_global': metadata.repeat_density_global,
            'heterozygosity_rate': metadata.heterozygosity_rate,
        },
        'output_files': output_files,
    }

    summary_path = graph_dir / f'graph_summary_g{genome_idx:04d}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Graph data genome {genome_idx}: {len(graph.nodes)} nodes, "
                f"{len(graph.edges)} edges, {len(sv_rows)} SV, {len(ul_rows)} UL "
                f"[schema v{SCHEMA_VERSION}]")

    return summary

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
