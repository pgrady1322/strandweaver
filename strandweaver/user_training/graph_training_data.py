#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Graph Training Data Generator — builds synthetic overlap graphs from simulated
reads and computes feature vectors and ground-truth labels for all ML models.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
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


# ============================================================================
#                     DATA STRUCTURES
# ============================================================================

@dataclass
class ReadInfo:
    """Lightweight read descriptor carried through the graph pipeline.

    Populated from the backend's ``SimulatedRead`` objects.
    """
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
    overlap_start_a: int     # start of overlap region on read A
    overlap_end_a: int
    overlap_start_b: int     # start of overlap region on read B
    overlap_end_b: int
    overlap_length: int
    identity: float          # estimated sequence identity (0–1)
    is_true: bool = True     # from ground-truth — do the reads really overlap?
    is_noise: bool = False   # artificially injected edge


@dataclass
class GraphNode:
    """Node in the synthetic overlap graph (= one read)."""
    node_id: str
    read_info: ReadInfo
    # Populated during feature extraction
    coverage: float = 0.0
    gc_content: float = 0.0
    repeat_fraction: float = 0.0
    in_degree: int = 0
    out_degree: int = 0


@dataclass
class GraphEdge:
    """Edge in the synthetic overlap graph."""
    source: str
    target: str
    overlap: Overlap
    # Populated during feature extraction
    features: Dict[str, float] = field(default_factory=dict)
    label: str = 'UNKNOWN'     # ground-truth label (EdgeLabel enum value)


@dataclass
class SyntheticGraph:
    """Complete synthetic assembly graph with labels and features."""
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    # Ground-truth data (set after labelling)
    node_labels: Dict[str, str] = field(default_factory=dict)
    edge_labels: Dict[Tuple[str, str], str] = field(default_factory=dict)
    correct_paths: List[List[str]] = field(default_factory=list)
    ul_routes: List[Dict[str, Any]] = field(default_factory=list)
    sv_labels: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
#                OVERLAP DETECTION (all-vs-all)
# ============================================================================

def _gc_content(seq: str) -> float:
    """GC fraction of a sequence."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq.upper() if b in ('G', 'C'))
    return gc / len(seq)


def _sequence_identity_estimate(seq_a: str, seq_b: str,
                                max_sample: int = 200) -> float:
    """Fast identity estimate by sampling positions in the overlap region.

    For synthetic reads whose true coordinates are known this is very
    accurate — the sequences really do share the overlap region.
    """
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0
    if min_len <= max_sample:
        matches = sum(a == b for a, b in zip(seq_a[:min_len], seq_b[:min_len]))
        return matches / min_len
    # Sample positions
    step = max(1, min_len // max_sample)
    matches = total = 0
    for i in range(0, min_len, step):
        if seq_a[i] == seq_b[i]:
            matches += 1
        total += 1
    return matches / total if total else 0.0


def _kmer_diversity(seq: str, k: int = 21) -> float:
    """Fraction of unique k-mers in a sequence (0–1)."""
    if len(seq) < k:
        return 0.0
    total = len(seq) - k + 1
    unique = len({seq[i:i + k] for i in range(total)})
    return unique / total


def _repeat_fraction_estimate(seq: str, k: int = 15, threshold: int = 3) -> float:
    """Fraction of positions covered by k-mers appearing ≥*threshold* times."""
    if len(seq) < k:
        return 0.0
    kmer_counts: Dict[str, int] = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmer_counts[seq[i:i + k]] += 1
    repeat_pos = sum(1 for i in range(len(seq) - k + 1)
                     if kmer_counts[seq[i:i + k]] >= threshold)
    return repeat_pos / max(1, len(seq) - k + 1)


def detect_overlaps(
    reads: List[ReadInfo],
    min_overlap_bp: int = 500,
    min_identity: float = 0.90,
    max_overhang_fraction: float = 0.30,
) -> List[Overlap]:
    """Detect pairwise overlaps between reads using ground-truth coordinates.

    Because reads are synthetic, we know their true reference positions
    and can compute exact overlaps without an aligner.  This is much
    faster than minimap2 and gives us perfect ground truth.

    Args:
        reads: List of ReadInfo objects (with known reference coordinates).
        min_overlap_bp: Minimum overlap length to accept.
        min_identity: Minimum sequence identity in the overlap region.
        max_overhang_fraction: Maximum allowed overhang as a fraction of the
            shorter read length.

    Returns:
        List of Overlap objects.
    """
    logger.info(f"Detecting overlaps among {len(reads)} reads "
                f"(min_ovl={min_overlap_bp}, min_id={min_identity:.2f})...")

    # Group reads by (chrom, haplotype) for efficiency — only reads that
    # share the same chromosome can overlap on the reference.  We also allow
    # cross-haplotype overlaps for allelic edges.
    by_chrom: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in reads:
        by_chrom[r.chrom].append(r)

    overlaps: List[Overlap] = []

    for chrom, chrom_reads in by_chrom.items():
        # Sort by start position for sweep-line efficiency
        chrom_reads.sort(key=lambda r: r.start_pos)
        n = len(chrom_reads)
        for i in range(n):
            ri = chrom_reads[i]
            for j in range(i + 1, n):
                rj = chrom_reads[j]
                # Early termination: if rj starts past ri's end, no overlap
                if rj.start_pos >= ri.end_pos:
                    break

                # Compute overlap on reference coordinates
                ovl_start = max(ri.start_pos, rj.start_pos)
                ovl_end = min(ri.end_pos, rj.end_pos)
                ovl_len = ovl_end - ovl_start

                if ovl_len < min_overlap_bp:
                    continue

                # Overhang check
                shorter_len = min(ri.length, rj.length)
                overhang_a = ri.length - ovl_len
                overhang_b = rj.length - ovl_len
                if max(overhang_a, overhang_b) / shorter_len > max_overhang_fraction:
                    continue

                # Sequence identity (sampled)
                # Extract the overlapping subsequences
                offset_a = ovl_start - ri.start_pos
                offset_b = ovl_start - rj.start_pos
                sub_a = ri.sequence[offset_a:offset_a + ovl_len]
                sub_b = rj.sequence[offset_b:offset_b + ovl_len]
                ident = _sequence_identity_estimate(sub_a, sub_b)
                if ident < min_identity:
                    continue

                overlap = Overlap(
                    read_a_id=ri.read_id,
                    read_b_id=rj.read_id,
                    overlap_start_a=offset_a,
                    overlap_end_a=offset_a + ovl_len,
                    overlap_start_b=offset_b,
                    overlap_end_b=offset_b + ovl_len,
                    overlap_length=ovl_len,
                    identity=ident,
                    is_true=(ri.haplotype == rj.haplotype),
                )
                overlaps.append(overlap)

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
    """Create false overlap edges for negative-class training examples.

    Strategies:
      • Random pairs of reads from distant loci → CHIMERIC
      • Cross-haplotype pairs at homologous loci → ALLELIC
      • Reads from repeat regions paired with non-repeat → REPEAT/CHIMERIC

    Returns only the noise edges (caller concatenates with true overlaps).
    """
    if fraction <= 0 or len(reads) < 4:
        return []

    rng = rng or random.Random(42)
    num_noise = max(1, int(len(true_overlaps) * fraction))
    logger.info(f"Injecting {num_noise} noise edges ({fraction:.0%} of {len(true_overlaps)} true overlaps)...")

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

        # Fabricate a plausible-looking overlap
        max_ovl = min(a.length, b.length, 5000)
        min_ovl = min(200, max_ovl)
        if max_ovl < 20:
            continue
        fake_ovl_len = rng.randint(min_ovl, max_ovl)
        noise.append(Overlap(
            read_a_id=a.read_id,
            read_b_id=b.read_id,
            overlap_start_a=0,
            overlap_end_a=fake_ovl_len,
            overlap_start_b=0,
            overlap_end_b=fake_ovl_len,
            overlap_length=fake_ovl_len,
            identity=rng.uniform(0.70, 0.95),
            is_true=False,
            is_noise=True,
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
    """Build a directed overlap graph from reads and their overlaps.

    Nodes = reads.  Edges = overlaps (directed: A → B if A's suffix
    overlaps B's prefix in reference order).

    Returns:
        SyntheticGraph with nodes and edges populated (features/labels empty).
    """
    logger.info("Building overlap graph...")

    # Create nodes
    nodes: Dict[str, GraphNode] = {}
    read_lookup: Dict[str, ReadInfo] = {}
    for r in reads:
        nodes[r.read_id] = GraphNode(node_id=r.read_id, read_info=r)
        read_lookup[r.read_id] = r

    # Create edges (directed: earlier read → later read on reference)
    edges: List[GraphEdge] = []
    for ovl in overlaps:
        ra = read_lookup.get(ovl.read_a_id)
        rb = read_lookup.get(ovl.read_b_id)
        if not ra or not rb:
            continue
        # Direction: read that starts earlier → read that starts later
        if ra.start_pos <= rb.start_pos:
            src, tgt = ovl.read_a_id, ovl.read_b_id
        else:
            src, tgt = ovl.read_b_id, ovl.read_a_id
        edges.append(GraphEdge(source=src, target=tgt, overlap=ovl))

    # Compute in/out degrees
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

def _coverage_at_pos(reads: List[ReadInfo], pos: int, chrom: str) -> int:
    """Count how many reads cover a specific position."""
    return sum(1 for r in reads if r.chrom == chrom
               and r.start_pos <= pos < r.end_pos)


def extract_node_features(
    graph: SyntheticGraph,
    all_reads: List[ReadInfo],
    genome_length: int,
    expected_coverage: float = 30.0,
) -> None:
    """Compute per-node features in-place on the graph.

    For DiploidAI the full 42-D feature vector is:
      32-D GNN embedding (placeholder zeros — filled during actual GNN
          training; here we provide deterministic pseudo-embeddings based
          on sequence composition so the feature columns are present)
      + 10-D signal features:
          coverage, gc_content, repeat_fraction, kmer_diversity,
          branching_factor, hic_contact_density (0 — no Hi-C simulated
          yet), allele_frequency, heterozygosity, phase_consistency,
          mappability
    """
    logger.info("Extracting node features...")

    # Build a position-based coverage array (approximate, sampled)
    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)

    for nid, node in graph.nodes.items():
        r = node.read_info
        seq = r.sequence

        # Basic sequence features
        node.gc_content = _gc_content(seq)
        node.repeat_fraction = _repeat_fraction_estimate(seq)

        # Coverage (sampled at midpoint of this read)
        mid = (r.start_pos + r.end_pos) // 2
        node.coverage = _coverage_at_pos(chrom_reads.get(r.chrom, []), mid, r.chrom)

    logger.info(f"Extracted features for {len(graph.nodes)} nodes")


def extract_edge_features(
    graph: SyntheticGraph,
    all_reads: List[ReadInfo],
) -> None:
    """Compute per-edge features in-place on the graph.

    EdgeAI 17-D feature schema:
        overlap_length, overlap_identity, read1_length, read2_length,
        coverage_r1, coverage_r2, gc_content_r1, gc_content_r2,
        repeat_fraction_r1, repeat_fraction_r2,
        kmer_diversity_r1, kmer_diversity_r2,
        branching_factor_r1, branching_factor_r2,
        hic_support, mapping_quality_r1, mapping_quality_r2

    PathGNN 16-D feature schema (overlapping but different orientation):
        overlap_length, overlap_identity, coverage_consistency,
        gc_similarity, repeat_match, branching_score,
        path_support, hic_contact, mapping_quality,
        kmer_match, sequence_complexity, orientation_score,
        distance_score, topology_score, ul_support, sv_evidence
    """
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

        gc1 = n1.gc_content
        gc2 = n2.gc_content
        rep1 = n1.repeat_fraction
        rep2 = n2.repeat_fraction
        kd1 = _kmer_diversity(r1.sequence)
        kd2 = _kmer_diversity(r2.sequence)
        bf1 = n1.in_degree + n1.out_degree
        bf2 = n2.in_degree + n2.out_degree

        # ── EdgeAI 17-D ──
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
        edge.features['hic_support'] = 0.0              # placeholder — no Hi-C graph yet
        edge.features['mapping_quality_r1'] = 60.0       # synthetic = perfect
        edge.features['mapping_quality_r2'] = 60.0

        # ── PathGNN 16-D (derived from edge-level stats) ──
        cov_consistency = 1.0 - abs(n1.coverage - n2.coverage) / max(n1.coverage, n2.coverage, 1)
        gc_sim = 1.0 - abs(gc1 - gc2)
        rep_match = 1.0 - abs(rep1 - rep2)

        edge.features['coverage_consistency'] = cov_consistency
        edge.features['gc_similarity'] = gc_sim
        edge.features['repeat_match'] = rep_match
        edge.features['branching_score'] = float(bf1 + bf2) / 2.0
        edge.features['path_support'] = 1.0 if ovl.is_true else 0.0
        edge.features['hic_contact'] = 0.0
        edge.features['mapping_quality'] = 60.0
        edge.features['kmer_match'] = (kd1 + kd2) / 2.0
        edge.features['sequence_complexity'] = _kmer_diversity(
            r1.sequence[-min(500, len(r1.sequence)):] +
            r2.sequence[:min(500, len(r2.sequence))]
        )
        edge.features['orientation_score'] = 1.0 if r1.strand == r2.strand else 0.0
        edge.features['distance_score'] = max(0.0, 1.0 - abs(
            r2.start_pos - r1.end_pos) / max(r1.length, 1))
        edge.features['topology_score'] = cov_consistency * gc_sim
        edge.features['ul_support'] = 0.0                # filled in UL labelling
        edge.features['sv_evidence'] = 0.0                # filled in SV labelling

    logger.info(f"Extracted features for {len(graph.edges)} edges")


def compute_sv_region_features(
    graph: SyntheticGraph,
    sv_truth_table: List[Any],
    all_reads: List[ReadInfo],
    genome_length: int,
) -> List[Dict[str, Any]]:
    """Compute 14-D SV detection features for genomic windows.

    Feature schema:
        coverage_mean, coverage_std, coverage_median,
        gc_content, repeat_fraction, kmer_diversity,
        branching_complexity, hic_disruption_score,
        ul_support, mapping_quality,
        region_length, breakpoint_precision,
        allele_balance, phase_switch_rate

    We generate one feature row per SV + a matched set of negative
    (non-SV) regions for balanced training.
    """
    import statistics

    logger.info("Computing SV region features...")

    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)

    sv_rows: List[Dict[str, Any]] = []

    # ── Positive examples (true SVs) ──
    for sv in sv_truth_table:
        sv_start = sv.pos
        sv_end = sv.end
        sv_chrom = sv.chrom
        region_reads = [
            r for r in chrom_reads.get(sv_chrom, [])
            if not (r.end_pos < sv_start or r.start_pos > sv_end)
        ]
        coverages = []
        for pos in range(sv_start, sv_end, max(1, (sv_end - sv_start) // 20)):
            coverages.append(
                sum(1 for r in region_reads
                    if r.start_pos <= pos < r.end_pos))
        if not coverages:
            coverages = [0]

        # Haplotype balance
        hap_a = sum(1 for r in region_reads if r.haplotype == 'A')
        hap_b = sum(1 for r in region_reads if r.haplotype == 'B')
        total_hap = hap_a + hap_b
        allele_balance = min(hap_a, hap_b) / max(total_hap, 1)

        combined_seq = ''.join(r.sequence for r in region_reads[:20])

        row = {
            'coverage_mean': statistics.mean(coverages),
            'coverage_std': statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
            'coverage_median': statistics.median(coverages),
            'gc_content': _gc_content(combined_seq) if combined_seq else 0.42,
            'repeat_fraction': _repeat_fraction_estimate(combined_seq) if combined_seq else 0.0,
            'kmer_diversity': _kmer_diversity(combined_seq) if combined_seq else 0.5,
            'branching_complexity': sum(
                graph.nodes[r.read_id].in_degree + graph.nodes[r.read_id].out_degree
                for r in region_reads if r.read_id in graph.nodes
            ) / max(len(region_reads), 1),
            'hic_disruption_score': 0.0,
            'ul_support': sum(1 for r in region_reads if r.technology in ('ultra_long', 'ul')),
            'mapping_quality': 60.0,
            'region_length': float(sv_end - sv_start),
            'breakpoint_precision': 1.0,  # perfect for synthetic
            'allele_balance': allele_balance,
            'phase_switch_rate': 0.0,
            'sv_type': sv.sv_type.value,    # label
        }
        sv_rows.append(row)

    # ── Negative examples (non-SV windows of similar size) ──
    rng = random.Random(12345)
    for sv in sv_truth_table:
        window = sv.end - sv.pos
        if window < 100:
            window = 1000
        # Pick a random position that doesn't overlap any SV
        for _ in range(5):
            rand_start = rng.randint(0, max(1, genome_length - window))
            rand_end = rand_start + window
            # Check no SV overlap
            overlaps_sv = any(
                not (s.end < rand_start or s.pos > rand_end)
                for s in sv_truth_table
            )
            if not overlaps_sv:
                break
        else:
            continue

        neg_reads = [
            r for r in all_reads
            if r.chrom == sv.chrom
            and not (r.end_pos < rand_start or r.start_pos > rand_end)
        ]
        coverages = []
        for pos in range(rand_start, rand_end, max(1, window // 20)):
            coverages.append(
                sum(1 for r in neg_reads
                    if r.start_pos <= pos < r.end_pos))
        if not coverages:
            coverages = [0]

        hap_a = sum(1 for r in neg_reads if r.haplotype == 'A')
        hap_b = sum(1 for r in neg_reads if r.haplotype == 'B')
        total_hap = hap_a + hap_b
        combined_seq = ''.join(r.sequence for r in neg_reads[:20])

        row = {
            'coverage_mean': statistics.mean(coverages),
            'coverage_std': statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
            'coverage_median': statistics.median(coverages),
            'gc_content': _gc_content(combined_seq) if combined_seq else 0.42,
            'repeat_fraction': _repeat_fraction_estimate(combined_seq) if combined_seq else 0.0,
            'kmer_diversity': _kmer_diversity(combined_seq) if combined_seq else 0.5,
            'branching_complexity': sum(
                graph.nodes[r.read_id].in_degree + graph.nodes[r.read_id].out_degree
                for r in neg_reads if r.read_id in graph.nodes
            ) / max(len(neg_reads), 1),
            'hic_disruption_score': 0.0,
            'ul_support': sum(1 for r in neg_reads if r.technology in ('ultra_long', 'ul')),
            'mapping_quality': 60.0,
            'region_length': float(window),
            'breakpoint_precision': 0.0,
            'allele_balance': min(hap_a, hap_b) / max(total_hap, 1),
            'phase_switch_rate': 0.0,
            'sv_type': 'none',              # label
        }
        sv_rows.append(row)

    logger.info(f"Computed SV features: {len(sv_rows)} rows "
                f"({sum(1 for r in sv_rows if r['sv_type'] != 'none')} positive, "
                f"{sum(1 for r in sv_rows if r['sv_type'] == 'none')} negative)")
    return sv_rows


def compute_ul_route_features(
    graph: SyntheticGraph,
    ul_reads: List[ReadInfo],
    all_reads: List[ReadInfo],
) -> List[Dict[str, Any]]:
    """Compute 12-D UL routing features for each ultralong read route.

    Feature schema:
        path_length, num_branches, coverage_mean, coverage_std,
        sequence_identity, mapping_quality, num_gaps, gap_size_mean,
        kmer_consistency, orientation_consistency,
        ul_span, route_complexity

    Returns:
        One feature-dict per UL read, with 'route_score' as label.
    """
    import statistics

    logger.info("Computing UL route features...")

    chrom_reads: Dict[str, List[ReadInfo]] = defaultdict(list)
    for r in all_reads:
        chrom_reads[r.chrom].append(r)

    rows: List[Dict[str, Any]] = []

    for ul in ul_reads:
        # Find reads that overlap this UL read's span (same haplotype)
        span_reads = [
            r for r in chrom_reads.get(ul.chrom, [])
            if r.haplotype == ul.haplotype
            and not (r.end_pos < ul.start_pos or r.start_pos > ul.end_pos)
            and r.read_id != ul.read_id
        ]
        span_reads.sort(key=lambda r: r.start_pos)
        path_nodes = [r.read_id for r in span_reads if r.read_id in graph.nodes]

        coverages = [graph.nodes[nid].coverage for nid in path_nodes
                      if nid in graph.nodes] or [0.0]
        branches = [graph.nodes[nid].in_degree + graph.nodes[nid].out_degree
                     for nid in path_nodes if nid in graph.nodes] or [0]

        # Gaps between consecutive overlapping reads
        gaps = []
        for i in range(len(span_reads) - 1):
            gap = span_reads[i + 1].start_pos - span_reads[i].end_pos
            if gap > 0:
                gaps.append(gap)

        row = {
            'path_length': float(ul.length),
            'num_branches': float(sum(1 for b in branches if b > 2)),
            'coverage_mean': statistics.mean(coverages),
            'coverage_std': statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
            'sequence_identity': 1.0,        # synthetic = known
            'mapping_quality': 60.0,
            'num_gaps': float(len(gaps)),
            'gap_size_mean': statistics.mean(gaps) if gaps else 0.0,
            'kmer_consistency': statistics.mean(
                [_kmer_diversity(r.sequence) for r in span_reads[:20]]
            ) if span_reads else 0.5,
            'orientation_consistency': 1.0,   # all same strand in synthetic
            'ul_span': float(ul.end_pos - ul.start_pos),
            'route_complexity': float(len(path_nodes)),
            # Label: route quality score (1.0 = perfect for synthetic correct routes)
            'route_score': 1.0 if len(path_nodes) > 0 else 0.0,
            'read_id': ul.read_id,
            'correct_path': path_nodes,
        }
        rows.append(row)

    logger.info(f"Computed UL route features for {len(rows)} reads")
    return rows


# ============================================================================
#              GROUND-TRUTH LABELLING
# ============================================================================

def label_graph(
    graph: SyntheticGraph,
    sv_truth_table: List[Any],
    max_true_distance: int = 10_000,
) -> None:
    """Apply ground-truth labels to all edges and nodes in-place.

    Edge labels: TRUE, ALLELIC, REPEAT, SV_BREAK, CHIMERIC
    Node labels: HAP_A, HAP_B, BOTH, REPEAT, UNKNOWN
    Correct paths: ordered read sequences per haplotype
    """
    logger.info("Labelling graph edges and nodes...")

    # ── Edge labels ──────────────────────────────────────────────────────
    for edge in graph.edges:
        r1 = graph.nodes[edge.source].read_info
        r2 = graph.nodes[edge.target].read_info

        if edge.overlap.is_noise:
            # Noise edges are chimeric by construction
            edge.label = 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = 'CHIMERIC'
            continue

        # Different chromosomes → chimeric
        if r1.chrom != r2.chrom:
            edge.label = 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = 'CHIMERIC'
            continue

        # Check SV crossing
        crosses_sv = False
        for sv in sv_truth_table:
            if hasattr(sv, 'chrom') and sv.chrom != r1.chrom:
                continue
            sv_start = sv.pos if hasattr(sv, 'pos') else 0
            sv_end = sv.end if hasattr(sv, 'end') else 0
            # Does the edge span across the SV breakpoint?
            span_min = min(r1.start_pos, r2.start_pos)
            span_max = max(r1.end_pos, r2.end_pos)
            if span_min < sv_start < span_max or span_min < sv_end < span_max:
                crosses_sv = True
                break

        if crosses_sv:
            edge.label = 'SV_BREAK'
            graph.edge_labels[(edge.source, edge.target)] = 'SV_BREAK'
            continue

        # Both from repeat regions
        if r1.is_repeat and r2.is_repeat:
            edge.label = 'REPEAT'
            graph.edge_labels[(edge.source, edge.target)] = 'REPEAT'
            continue

        # Different haplotypes
        if r1.haplotype != r2.haplotype:
            distance = abs(r1.start_pos - r2.start_pos)
            if distance < max_true_distance:
                edge.label = 'ALLELIC'
            else:
                edge.label = 'CHIMERIC'
            graph.edge_labels[(edge.source, edge.target)] = edge.label
            continue

        # Same haplotype, close → TRUE
        distance = abs(r2.start_pos - r1.end_pos)
        if distance <= max_true_distance:
            edge.label = 'TRUE'
        else:
            edge.label = 'CHIMERIC'
        graph.edge_labels[(edge.source, edge.target)] = edge.label

    # ── Node labels ──────────────────────────────────────────────────────
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

    # ── Correct paths (sorted reads per haplotype) ───────────────────────
    by_hap: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        r = node.read_info
        by_hap[(r.haplotype, r.chrom)].append((r.start_pos, nid))
    for key in by_hap:
        by_hap[key].sort()
        graph.correct_paths.append([nid for _, nid in by_hap[key]])

    # ── Label counts ─────────────────────────────────────────────────────
    edge_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.edge_labels.values():
        edge_counts[lbl] += 1
    node_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.node_labels.values():
        node_counts[lbl] += 1

    logger.info("Edge label distribution: " +
                ", ".join(f"{k}={v}" for k, v in sorted(edge_counts.items())))
    logger.info("Node label distribution: " +
                ", ".join(f"{k}={v}" for k, v in sorted(node_counts.items())))
    logger.info(f"Correct paths extracted: {len(graph.correct_paths)}")


# ============================================================================
#              EXPORT
# ============================================================================

# ── Feature column names (must match ml_training_system.py schemas) ──────

EDGE_AI_FEATURES = [
    'overlap_length', 'overlap_identity', 'read1_length', 'read2_length',
    'coverage_r1', 'coverage_r2', 'gc_content_r1', 'gc_content_r2',
    'repeat_fraction_r1', 'repeat_fraction_r2',
    'kmer_diversity_r1', 'kmer_diversity_r2',
    'branching_factor_r1', 'branching_factor_r2',
    'hic_support', 'mapping_quality_r1', 'mapping_quality_r2',
]

PATH_GNN_FEATURES = [
    'overlap_length', 'overlap_identity', 'coverage_consistency',
    'gc_similarity', 'repeat_match', 'branching_score',
    'path_support', 'hic_contact', 'mapping_quality',
    'kmer_match', 'sequence_complexity', 'orientation_score',
    'distance_score', 'topology_score', 'ul_support', 'sv_evidence',
]

NODE_SIGNAL_FEATURES = [
    'coverage', 'gc_content', 'repeat_fraction', 'kmer_diversity',
    'branching_factor', 'hic_contact_density', 'allele_frequency',
    'heterozygosity', 'phase_consistency', 'mappability',
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
]


def export_edge_training_csv(
    graph: SyntheticGraph,
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export EdgeAI training data (17 features + label) to CSV."""
    out = output_dir / f'edge_ai_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(EDGE_AI_FEATURES + ['label'])
        for edge in graph.edges:
            row = [edge.features.get(feat, 0.0) for feat in EDGE_AI_FEATURES]
            row.append(edge.label)
            writer.writerow(row)
    logger.info(f"  EdgeAI CSV: {out}  ({len(graph.edges)} rows)")
    return out


def export_path_gnn_training_csv(
    graph: SyntheticGraph,
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export PathGNN training data (16 features + binary label) to CSV.

    For each edge, the label is 1 if the edge lies on a correct path,
    0 otherwise.
    """
    # Build set of (src, tgt) pairs on correct paths
    correct_edge_set: Set[Tuple[str, str]] = set()
    for path in graph.correct_paths:
        for i in range(len(path) - 1):
            correct_edge_set.add((path[i], path[i + 1]))

    out = output_dir / f'path_gnn_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(PATH_GNN_FEATURES + ['in_correct_path'])
        for edge in graph.edges:
            row = [edge.features.get(feat, 0.0) for feat in PATH_GNN_FEATURES]
            row.append(1 if (edge.source, edge.target) in correct_edge_set else 0)
            writer.writerow(row)
    logger.info(f"  PathGNN CSV: {out}  ({len(graph.edges)} rows)")
    return out


def export_node_training_csv(
    graph: SyntheticGraph,
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export DiploidAI training data (10 signal features + label) to CSV.

    The 32-D GNN embedding is a placeholder (zeros) — the real GNN
    embeddings are learned during model training.  We export the 10
    concrete signal features here.
    """
    out = output_dir / f'diploid_ai_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(NODE_SIGNAL_FEATURES + ['haplotype_label'])
        for nid, node in graph.nodes.items():
            r = node.read_info
            kd = _kmer_diversity(r.sequence)
            bf = node.in_degree + node.out_degree
            row = [
                node.coverage,
                node.gc_content,
                node.repeat_fraction,
                kd,
                float(bf),
                0.0,  # hic_contact_density (placeholder)
                0.5,  # allele_frequency (placeholder)
                0.0,  # heterozygosity (placeholder)
                1.0,  # phase_consistency (synthetic = perfect)
                1.0,  # mappability (synthetic = perfect)
            ]
            row.append(graph.node_labels.get(nid, 'UNKNOWN'))
            writer.writerow(row)
    logger.info(f"  DiploidAI CSV: {out}  ({len(graph.nodes)} rows)")
    return out


def export_ul_route_training_csv(
    ul_rows: List[Dict[str, Any]],
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export UL routing training data (12 features + score) to CSV."""
    out = output_dir / f'ul_route_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(UL_ROUTE_FEATURES + ['route_score'])
        for row in ul_rows:
            vals = [row.get(feat, 0.0) for feat in UL_ROUTE_FEATURES]
            vals.append(row.get('route_score', 0.0))
            writer.writerow(vals)
    logger.info(f"  UL Route CSV: {out}  ({len(ul_rows)} rows)")
    return out


def export_sv_training_csv(
    sv_rows: List[Dict[str, Any]],
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export SV detection training data (14 features + label) to CSV."""
    out = output_dir / f'sv_detect_training_g{genome_idx:04d}.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(SV_DETECT_FEATURES + ['sv_type'])
        for row in sv_rows:
            vals = [row.get(feat, 0.0) for feat in SV_DETECT_FEATURES]
            vals.append(row.get('sv_type', 'none'))
            writer.writerow(vals)
    logger.info(f"  SV Detect CSV: {out}  ({len(sv_rows)} rows)")
    return out


def export_gfa(
    graph: SyntheticGraph,
    output_dir: Path,
    genome_idx: int = 0,
) -> Path:
    """Export graph in GFA v1 format for visualisation."""
    out = output_dir / f'overlap_graph_g{genome_idx:04d}.gfa'
    with open(out, 'w') as f:
        f.write("H\tVN:Z:1.0\n")
        # Segments
        for nid, node in graph.nodes.items():
            seq = node.read_info.sequence[:50] + '*' if len(node.read_info.sequence) > 50 else node.read_info.sequence
            f.write(f"S\t{nid}\t{seq}\tLN:i:{node.read_info.length}\n")
        # Links
        for edge in graph.edges:
            cigar = f"{edge.overlap.overlap_length}M"
            f.write(f"L\t{edge.source}\t+\t{edge.target}\t+\t{cigar}\n")
    logger.info(f"  GFA: {out}")
    return out


# ============================================================================
#              MAIN PIPELINE
# ============================================================================

def reads_to_read_infos(
    simulated_reads: List[Any],
    technology: str = 'hifi',
) -> List[ReadInfo]:
    """Convert backend SimulatedRead / SimulatedReadPair objects to ReadInfo.

    Handles both single-end reads and paired-end reads (the latter are
    split into two ReadInfo objects with '_R1' / '_R2' suffixes).
    """
    infos: List[ReadInfo] = []
    for read in simulated_reads:
        # Paired-end (SimulatedReadPair)
        if hasattr(read, 'read1') and hasattr(read, 'read2'):
            for suffix, sr in [('_R1', read.read1), ('_R2', read.read2)]:
                infos.append(ReadInfo(
                    read_id=sr.read_id + suffix if not sr.read_id.endswith(suffix) else sr.read_id,
                    sequence=sr.sequence,
                    quality=sr.quality,
                    haplotype=sr.haplotype or 'A',
                    chrom=sr.chrom,
                    start_pos=sr.start_pos,
                    end_pos=sr.end_pos,
                    strand=sr.strand,
                    technology=technology,
                ))
        else:
            infos.append(ReadInfo(
                read_id=read.read_id,
                sequence=read.sequence,
                quality=read.quality,
                haplotype=read.haplotype or 'A',
                chrom=read.chrom,
                start_pos=read.start_pos,
                end_pos=read.end_pos,
                strand=read.strand,
                technology=technology,
            ))
    return infos


def generate_graph_training_data(
    all_simulated_reads: Dict[str, List[Any]],
    diploid_genome: Any,
    sv_truth_table: List[Any],
    graph_config: GraphTrainingConfig,
    output_dir: Path,
    genome_idx: int = 0,
) -> Dict[str, Any]:
    """End-to-end graph training data generation for one genome.

    This is the main entry point called by the config_based_workflow
    after reads have been simulated.

    Args:
        all_simulated_reads: ``{tech_name: [SimulatedRead, ...]}``
        diploid_genome: DiploidGenome object from the backend.
        sv_truth_table: List of StructuralVariant objects.
        graph_config: GraphTrainingConfig with user settings.
        output_dir: Per-genome output directory.
        genome_idx: Genome index (for filenames).

    Returns:
        Summary dict with file paths and statistics.
    """
    graph_dir = output_dir / 'graph_training'
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Convert reads to ReadInfo ─────────────────────────────────────
    all_read_infos: List[ReadInfo] = []
    ul_read_infos: List[ReadInfo] = []
    tech_map = {
        'illumina': 'illumina', 'hifi': 'hifi', 'ont': 'ont',
        'ultra_long': 'ultra_long', 'hic': 'hic', 'ancient_dna': 'ancient_dna',
    }
    for tech, reads in all_simulated_reads.items():
        infos = reads_to_read_infos(reads, technology=tech_map.get(tech, tech))
        all_read_infos.extend(infos)
        if tech in ('ultra_long', 'ul'):
            ul_read_infos.extend(infos)

    logger.info(f"Total reads for graph: {len(all_read_infos)} "
                f"(UL: {len(ul_read_infos)})")

    # ── 2. Optional coverage subsampling ─────────────────────────────────
    if graph_config.max_coverage_for_graph and len(all_read_infos) > 100:
        genome_len = len(diploid_genome.hapA) if hasattr(diploid_genome, 'hapA') else 1_000_000
        total_bases = sum(r.length for r in all_read_infos)
        current_cov = total_bases / genome_len
        if current_cov > graph_config.max_coverage_for_graph:
            keep_frac = graph_config.max_coverage_for_graph / current_cov
            rng = random.Random(42)
            all_read_infos = [r for r in all_read_infos if rng.random() < keep_frac]
            logger.info(f"Subsampled to ~{graph_config.max_coverage_for_graph:.0f}× "
                        f"({len(all_read_infos)} reads)")

    # ── 3. Detect overlaps ───────────────────────────────────────────────
    overlaps = detect_overlaps(
        all_read_infos,
        min_overlap_bp=graph_config.min_overlap_bp,
        min_identity=graph_config.min_overlap_identity,
        max_overhang_fraction=graph_config.max_overhang_fraction,
    )

    # ── 4. Inject noise edges ────────────────────────────────────────────
    if graph_config.add_noise_edges:
        noise = inject_noise_edges(
            all_read_infos, overlaps,
            fraction=graph_config.noise_edge_fraction,
        )
        overlaps.extend(noise)

    # ── 5. Build graph ───────────────────────────────────────────────────
    graph = build_overlap_graph(all_read_infos, overlaps)

    # ── 6. Feature extraction ────────────────────────────────────────────
    genome_len = len(diploid_genome.hapA) if hasattr(diploid_genome, 'hapA') else 1_000_000
    if graph_config.compute_features:
        extract_node_features(graph, all_read_infos, genome_len)
        extract_edge_features(graph, all_read_infos)

    # ── 7. Ground-truth labelling ────────────────────────────────────────
    label_graph(graph, sv_truth_table)

    # ── 8. SV region features ────────────────────────────────────────────
    sv_rows: List[Dict[str, Any]] = []
    if graph_config.label_svs and sv_truth_table:
        sv_rows = compute_sv_region_features(
            graph, sv_truth_table, all_read_infos, genome_len)

    # ── 9. UL route features ─────────────────────────────────────────────
    ul_rows: List[Dict[str, Any]] = []
    if graph_config.label_ul_routes and ul_read_infos:
        ul_rows = compute_ul_route_features(graph, ul_read_infos, all_read_infos)

    # ── 10. Export ───────────────────────────────────────────────────────
    output_files: Dict[str, str] = {}

    if graph_config.label_edges:
        p = export_edge_training_csv(graph, graph_dir, genome_idx)
        output_files['edge_ai_csv'] = str(p)

    if graph_config.label_paths:
        p = export_path_gnn_training_csv(graph, graph_dir, genome_idx)
        output_files['path_gnn_csv'] = str(p)

    if graph_config.label_nodes:
        p = export_node_training_csv(graph, graph_dir, genome_idx)
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

    # ── Summary ──────────────────────────────────────────────────────────
    edge_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.edge_labels.values():
        edge_counts[lbl] += 1
    node_counts: Dict[str, int] = defaultdict(int)
    for lbl in graph.node_labels.values():
        node_counts[lbl] += 1

    summary = {
        'num_reads': len(all_read_infos),
        'num_ul_reads': len(ul_read_infos),
        'num_overlaps': len(overlaps),
        'num_nodes': len(graph.nodes),
        'num_edges': len(graph.edges),
        'num_correct_paths': len(graph.correct_paths),
        'edge_label_distribution': dict(edge_counts),
        'node_label_distribution': dict(node_counts),
        'num_sv_training_rows': len(sv_rows),
        'num_ul_training_rows': len(ul_rows),
        'output_files': output_files,
    }

    # Save summary JSON
    summary_path = graph_dir / f'graph_summary_g{genome_idx:04d}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Graph training data for genome {genome_idx}: "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, "
                f"{len(sv_rows)} SV rows, {len(ul_rows)} UL rows")

    return summary

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
