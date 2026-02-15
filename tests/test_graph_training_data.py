#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0-dev

Graph training data unit tests — sequence utilities, dataclasses, overlap
detection, noise injection, graph construction, and CSV/GFA export.

All tests use only the Python standard library and pytest (no numpy/sklearn).

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial)
"""

import csv
import math
import random
from pathlib import Path

import pytest

from strandweaver.user_training.graph_training_data import (
    EDGE_AI_FEATURES,
    EDGE_AI_PROVENANCE,
    METADATA_COLUMNS,
    NODE_PROVENANCE,
    NODE_SIGNAL_FEATURES,
    PATH_GNN_FEATURES,
    PATH_GNN_PROVENANCE,
    SV_DETECT_FEATURES,
    UL_ROUTE_FEATURES,
    GenomeMetadata,
    GraphEdge,
    GraphNode,
    Overlap,
    ReadInfo,
    SyntheticGraph,
    _dinucleotide_bias,
    _gc_content,
    _homopolymer_stats,
    _kmer_diversity,
    _low_complexity_fraction,
    _repeat_fraction_estimate,
    _sequence_identity_estimate,
    _shannon_entropy,
    build_overlap_graph,
    detect_overlaps,
    export_edge_training_csv,
    export_gfa,
    export_node_training_csv,
    export_path_gnn_training_csv,
    export_sv_training_csv,
    export_ul_route_training_csv,
    inject_noise_edges,
)


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _make_read(read_id, seq, chrom="chr1", start=0, haplotype="A", **kw):
    """Quick ReadInfo factory for tests."""
    return ReadInfo(
        read_id=read_id,
        sequence=seq,
        quality="I" * len(seq),
        haplotype=haplotype,
        chrom=chrom,
        start_pos=start,
        end_pos=start + len(seq),
        **kw,
    )


def _random_seq(length, seed=0):
    """Generate a random ACGT sequence of given length."""
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENCE UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class TestGCContent:
    def test_empty(self):
        assert _gc_content("") == 0.0

    def test_all_gc(self):
        assert _gc_content("GGCC") == 1.0

    def test_all_at(self):
        assert _gc_content("AATT") == 0.0

    def test_mixed(self):
        assert abs(_gc_content("ACGT") - 0.5) < 1e-9

    def test_case_insensitive(self):
        assert _gc_content("acgt") == _gc_content("ACGT")


class TestSequenceIdentity:
    def test_identical(self):
        seq = "ACGTACGTACGT"
        assert _sequence_identity_estimate(seq, seq) == 1.0

    def test_empty(self):
        assert _sequence_identity_estimate("", "ACGT") == 0.0

    def test_completely_different(self):
        # All mismatches when comparing complement-like strings
        a = "AAAA"
        b = "CCCC"
        assert _sequence_identity_estimate(a, b) == 0.0

    def test_half_match(self):
        a = "AACCGG"
        b = "AATTGG"
        # positions 0,1 match (A,A), 2,3 differ, 4,5 match (G,G) → 4/6
        assert abs(_sequence_identity_estimate(a, b) - 4.0 / 6.0) < 1e-9

    def test_different_lengths_uses_shorter(self):
        a = "ACGT"
        b = "ACGTNNNN"
        # Only compares first 4 positions
        assert _sequence_identity_estimate(a, b) == 1.0


class TestKmerDiversity:
    def test_too_short(self):
        assert _kmer_diversity("ACGT", k=5) == 0.0

    def test_all_unique(self):
        # A long enough random sequence should have non-trivial diversity
        seq = _random_seq(200, seed=1)
        d = _kmer_diversity(seq, k=3)
        assert 0.0 < d <= 1.0

    def test_homopolymer_low_diversity(self):
        seq = "A" * 100
        d = _kmer_diversity(seq, k=5)
        # Only one unique k-mer → 1/(100-5+1) ≈ 0.01
        assert d < 0.05


class TestRepeatFraction:
    def test_no_repeats(self):
        seq = _random_seq(200, seed=2)
        frac = _repeat_fraction_estimate(seq, k=15, threshold=3)
        # A random 200bp sequence shouldn't have many 15-mers repeated 3×
        assert frac < 0.1

    def test_tandem_repeat(self):
        unit = "AACCGG"
        seq = unit * 50  # 300bp tandem repeat
        frac = _repeat_fraction_estimate(seq, k=6, threshold=3)
        assert frac > 0.5

    def test_short_sequence(self):
        assert _repeat_fraction_estimate("AC", k=15) == 0.0


class TestShannonEntropy:
    def test_empty(self):
        assert _shannon_entropy("") == 0.0

    def test_single_base(self):
        # All A → only one symbol → entropy = 0
        assert _shannon_entropy("AAAA") == 0.0

    def test_uniform(self):
        # Equal proportions of ACGT → max entropy = 2.0 bits
        seq = "ACGT" * 25
        assert abs(_shannon_entropy(seq) - 2.0) < 1e-9

    def test_two_bases(self):
        seq = "AACCAACCAACC"
        entropy = _shannon_entropy(seq)
        # Two equally frequent bases → entropy = 1.0
        assert abs(entropy - 1.0) < 1e-9


class TestDinucleotideBias:
    def test_short_sequence(self):
        assert _dinucleotide_bias("A") == 0.0

    def test_returns_float(self):
        seq = _random_seq(100, seed=3)
        result = _dinucleotide_bias(seq)
        assert isinstance(result, float)


class TestHomopolymerStats:
    def test_empty(self):
        max_run, density = _homopolymer_stats("")
        assert max_run == 0
        assert density == 0.0

    def test_no_homopolymer(self):
        max_run, density = _homopolymer_stats("ACGT")
        assert max_run == 1

    def test_long_run(self):
        seq = "CCGT" + "A" * 20 + "CGTC"
        max_run, density = _homopolymer_stats(seq)
        assert max_run == 20

    def test_density_per_kb(self):
        # 1000 bases, sprinkle some runs ≥3
        seq = "ACGT" * 200 + "AAA" + "CCC" + "A" * 194
        max_run, density = _homopolymer_stats(seq)
        assert density > 0.0


class TestLowComplexity:
    def test_too_short(self):
        assert _low_complexity_fraction("ACG", k=6) == 0.0

    def test_homopolymer_is_low_complexity(self):
        seq = "A" * 50
        frac = _low_complexity_fraction(seq, k=6)
        assert frac == 1.0  # every 6-mer has only 1 distinct base

    def test_diverse_sequence(self):
        seq = _random_seq(200, seed=4)
        frac = _low_complexity_fraction(seq, k=6)
        assert frac < 0.3  # random seq should be mostly complex


# ═══════════════════════════════════════════════════════════════════════
#  DATACLASS BASICS
# ═══════════════════════════════════════════════════════════════════════

class TestReadInfo:
    def test_length_property(self):
        r = _make_read("r1", "ACGT" * 10, start=100)
        assert r.length == 40

    def test_default_strand(self):
        r = _make_read("r1", "ACGT")
        assert r.strand == "+"

    def test_default_technology(self):
        r = _make_read("r1", "ACGT")
        assert r.technology == "hifi"


class TestGenomeMetadata:
    def test_defaults(self):
        gm = GenomeMetadata()
        assert gm.genome_id == 0
        assert gm.read_technology == "hifi"

    def test_as_row_length(self):
        gm = GenomeMetadata()
        row = gm.as_row()
        assert len(row) == len(METADATA_COLUMNS)

    def test_as_row_content(self):
        gm = GenomeMetadata(genome_id=5, genome_size=2_000_000)
        row = gm.as_row()
        assert row[0] == 5
        assert row[1] == 2_000_000


class TestFeatureColumnConstants:
    """Ensure column lists are non-empty and have no duplicates."""

    @pytest.mark.parametrize("columns", [
        EDGE_AI_FEATURES,
        EDGE_AI_PROVENANCE,
        PATH_GNN_FEATURES,
        PATH_GNN_PROVENANCE,
        NODE_SIGNAL_FEATURES,
        NODE_PROVENANCE,
        UL_ROUTE_FEATURES,
        SV_DETECT_FEATURES,
        METADATA_COLUMNS,
    ])
    def test_non_empty(self, columns):
        assert len(columns) > 0

    @pytest.mark.parametrize("columns", [
        EDGE_AI_FEATURES,
        PATH_GNN_FEATURES,
        NODE_SIGNAL_FEATURES,
        UL_ROUTE_FEATURES,
        SV_DETECT_FEATURES,
    ])
    def test_no_duplicates(self, columns):
        assert len(columns) == len(set(columns)), f"Duplicates in {columns}"


# ═══════════════════════════════════════════════════════════════════════
#  OVERLAP DETECTION
# ═══════════════════════════════════════════════════════════════════════

class TestDetectOverlaps:
    """Tests for coordinate-based overlap detection."""

    def _overlapping_reads(self):
        """Two reads that genuinely share content in their overlap region.

        Reference:  |---------- 1500 bp ----------|
        r1:         [0 ......... 1000]              (1000 bp)
        r2:              [300 .......... 1100]      (800 bp)
        Overlap:         [300 ... 1000]             (700 bp)
        Overhang:    r1: 300/800 = 0.375 → needs max_overhang_fraction ≥ 0.40
                     r2: 100/800 = 0.125
        """
        ref = _random_seq(1500, seed=10)
        r1 = _make_read("r1", ref[0:1000], start=0)     # 0..1000
        r2 = _make_read("r2", ref[300:1100], start=300)  # 300..1100
        return [r1, r2]

    def test_overlap_found(self):
        reads = self._overlapping_reads()
        overlaps = detect_overlaps(
            reads, min_overlap_bp=100, min_identity=0.5,
            max_overhang_fraction=0.45,
        )
        assert len(overlaps) >= 1

    def test_below_min_overlap(self):
        reads = self._overlapping_reads()
        overlaps = detect_overlaps(
            reads, min_overlap_bp=800, min_identity=0.5,
            max_overhang_fraction=0.45,
        )
        assert len(overlaps) == 0

    def test_no_overlap_different_chroms(self):
        seq = _random_seq(500, seed=11)
        r1 = _make_read("r1", seq, chrom="chr1", start=0)
        r2 = _make_read("r2", seq, chrom="chr2", start=0)
        overlaps = detect_overlaps([r1, r2], min_overlap_bp=100, min_identity=0.5)
        assert len(overlaps) == 0

    def test_no_overlap_disjoint(self):
        seq = _random_seq(100, seed=12)
        r1 = _make_read("r1", seq, start=0)
        r2 = _make_read("r2", seq, start=1000)
        overlaps = detect_overlaps([r1, r2], min_overlap_bp=50, min_identity=0.5)
        assert len(overlaps) == 0

    def test_same_haplotype_is_true(self):
        reads = self._overlapping_reads()
        overlaps = detect_overlaps(
            reads, min_overlap_bp=100, min_identity=0.5,
            max_overhang_fraction=0.45,
        )
        assert all(ovl.is_true for ovl in overlaps)

    def test_different_haplotype(self):
        ref = _random_seq(1500, seed=13)
        r1 = _make_read("r1", ref[0:1000], start=0, haplotype="A")
        r2 = _make_read("r2", ref[300:1100], start=300, haplotype="B")
        overlaps = detect_overlaps(
            [r1, r2], min_overlap_bp=100, min_identity=0.5,
            max_overhang_fraction=0.45,
        )
        if overlaps:
            assert not overlaps[0].is_true


# ═══════════════════════════════════════════════════════════════════════
#  NOISE EDGE INJECTION
# ═══════════════════════════════════════════════════════════════════════

class TestInjectNoiseEdges:

    def _reads_and_overlaps(self, n_reads=20, seed=42):
        """Generate reads and true overlaps for noise injection tests."""
        rng = random.Random(seed)
        reads = []
        for i in range(n_reads):
            seq = _random_seq(500, seed=seed + i)
            reads.append(_make_read(f"r{i}", seq, start=i * 200))
        true_overlaps = []
        for i in range(n_reads - 1):
            true_overlaps.append(Overlap(
                read_a_id=f"r{i}", read_b_id=f"r{i+1}",
                overlap_start_a=200, overlap_end_a=500,
                overlap_start_b=0, overlap_end_b=300,
                overlap_length=300, identity=0.95,
                is_true=True,
            ))
        return reads, true_overlaps

    def test_noise_edges_created(self):
        reads, true_ovl = self._reads_and_overlaps()
        noise = inject_noise_edges(reads, true_ovl, fraction=0.5)
        assert len(noise) > 0

    def test_noise_edges_marked(self):
        reads, true_ovl = self._reads_and_overlaps()
        noise = inject_noise_edges(reads, true_ovl, fraction=0.5)
        for n in noise:
            assert n.is_noise is True
            assert n.is_true is False

    def test_zero_fraction(self):
        reads, true_ovl = self._reads_and_overlaps()
        noise = inject_noise_edges(reads, true_ovl, fraction=0.0)
        assert len(noise) == 0

    def test_too_few_reads(self):
        reads = [_make_read("r0", "ACGT")]
        noise = inject_noise_edges(reads, [], fraction=0.5)
        assert len(noise) == 0

    def test_deterministic_with_seed(self):
        reads, true_ovl = self._reads_and_overlaps()
        noise_a = inject_noise_edges(reads, true_ovl, fraction=0.5, rng=random.Random(99))
        noise_b = inject_noise_edges(reads, true_ovl, fraction=0.5, rng=random.Random(99))
        ids_a = [(n.read_a_id, n.read_b_id) for n in noise_a]
        ids_b = [(n.read_a_id, n.read_b_id) for n in noise_b]
        assert ids_a == ids_b


# ═══════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

class TestBuildOverlapGraph:

    def test_basic_graph(self):
        seq = _random_seq(500, seed=20)
        reads = [
            _make_read("r0", seq, start=0),
            _make_read("r1", seq, start=200),
            _make_read("r2", seq, start=400),
        ]
        overlaps = [
            Overlap("r0", "r1", 200, 500, 0, 300, 300, 0.95),
            Overlap("r1", "r2", 200, 500, 0, 300, 300, 0.95),
        ]
        graph = build_overlap_graph(reads, overlaps)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_degree_tracking(self):
        seq = _random_seq(500, seed=21)
        reads = [
            _make_read("r0", seq, start=0),
            _make_read("r1", seq, start=200),
        ]
        overlaps = [Overlap("r0", "r1", 200, 500, 0, 300, 300, 0.95)]
        graph = build_overlap_graph(reads, overlaps)
        # r0 → r1, so r0 has out_degree=1, r1 has in_degree=1
        assert graph.nodes["r0"].out_degree == 1
        assert graph.nodes["r1"].in_degree == 1

    def test_empty_graph(self):
        reads = [_make_read("r0", "ACGT", start=0)]
        graph = build_overlap_graph(reads, [])
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0


# ═══════════════════════════════════════════════════════════════════════
#  CSV / GFA EXPORT
# ═══════════════════════════════════════════════════════════════════════

def _make_simple_graph():
    """Build a minimal graph suitable for CSV export tests."""
    seq = _random_seq(300, seed=30)
    r0 = _make_read("r0", seq, start=0)
    r1 = _make_read("r1", seq, start=100)
    nodes = {
        "r0": GraphNode(node_id="r0", read_info=r0),
        "r1": GraphNode(node_id="r1", read_info=r1),
    }
    ovl = Overlap("r0", "r1", 100, 300, 0, 200, 200, 0.95)
    edges = [GraphEdge(source="r0", target="r1", overlap=ovl, label="TRUE")]
    graph = SyntheticGraph(
        nodes=nodes,
        edges=edges,
        node_labels={"r0": "HAP_A", "r1": "HAP_A"},
    )
    return graph


class TestExportEdgeCSV:
    def test_creates_file(self, tmp_path):
        graph = _make_simple_graph()
        path = export_edge_training_csv(graph, tmp_path, genome_idx=0)
        assert path.exists()

    def test_correct_row_count(self, tmp_path):
        graph = _make_simple_graph()
        path = export_edge_training_csv(graph, tmp_path, genome_idx=0)
        with open(path) as f:
            rows = list(csv.reader(f))
        # header + 1 edge row
        assert len(rows) == 2

    def test_with_metadata(self, tmp_path):
        graph = _make_simple_graph()
        meta = GenomeMetadata(genome_id=1, genome_size=500_000)
        path = export_edge_training_csv(graph, tmp_path, genome_idx=1, metadata=meta)
        with open(path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        # Metadata columns should be prepended
        for mc in METADATA_COLUMNS:
            assert mc in header

    def test_filename_pattern(self, tmp_path):
        graph = _make_simple_graph()
        path = export_edge_training_csv(graph, tmp_path, genome_idx=7)
        assert path.name == "edge_ai_training_g0007.csv"


class TestExportPathGNNCSV:
    def test_creates_file(self, tmp_path):
        graph = _make_simple_graph()
        path = export_path_gnn_training_csv(graph, tmp_path)
        assert path.exists()

    def test_label_column_present(self, tmp_path):
        graph = _make_simple_graph()
        path = export_path_gnn_training_csv(graph, tmp_path)
        with open(path) as f:
            reader = csv.DictReader(f)
            assert "in_correct_path" in reader.fieldnames


class TestExportNodeCSV:
    def test_creates_file(self, tmp_path):
        graph = _make_simple_graph()
        path = export_node_training_csv(graph, tmp_path)
        assert path.exists()

    def test_correct_row_count(self, tmp_path):
        graph = _make_simple_graph()
        path = export_node_training_csv(graph, tmp_path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # header + 2 node rows
        assert len(rows) == 3


class TestExportULRouteCSV:
    def test_creates_file(self, tmp_path):
        rows = [
            {"path_length": 5, "num_branches": 2, "route_score": 0.8},
        ]
        path = export_ul_route_training_csv(rows, tmp_path)
        assert path.exists()

    def test_header_includes_features(self, tmp_path):
        rows = [{"route_score": 0.5}]
        path = export_ul_route_training_csv(rows, tmp_path)
        with open(path) as f:
            header = csv.DictReader(f).fieldnames
        for feat in UL_ROUTE_FEATURES:
            assert feat in header


class TestExportSVCSV:
    def test_creates_file(self, tmp_path):
        rows = [
            {"sv_type": "deletion", "coverage_mean": 10.0},
        ]
        path = export_sv_training_csv(rows, tmp_path)
        assert path.exists()

    def test_label_column(self, tmp_path):
        rows = [{"sv_type": "insertion"}]
        path = export_sv_training_csv(rows, tmp_path)
        with open(path) as f:
            header = csv.DictReader(f).fieldnames
        assert "sv_type" in header


class TestExportGFA:
    def test_creates_file(self, tmp_path):
        graph = _make_simple_graph()
        path = export_gfa(graph, tmp_path)
        assert path.exists()

    def test_gfa_format(self, tmp_path):
        graph = _make_simple_graph()
        path = export_gfa(graph, tmp_path)
        text = path.read_text()
        # GFA header
        assert text.startswith("H\tVN:Z:1.0")
        # Should have S (segment) and L (link) lines
        assert "\nS\t" in text
        assert "\nL\t" in text

    def test_filename_pattern(self, tmp_path):
        graph = _make_simple_graph()
        path = export_gfa(graph, tmp_path, genome_idx=3)
        assert path.name == "overlap_graph_g0003.gfa"
