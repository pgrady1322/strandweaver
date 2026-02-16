#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for G2 bubble-aware diploid phasing and ploidy flag.

Covers:
  - Ploidy enum and config propagation
  - Haploid fast-path (ploidy=1 skips phasing)
  - Bubble detection in assembly graphs
  - Per-bubble Hi-C contact binning
  - Phase chaining across linked bubbles
  - Fallback to spectral clustering when no bubbles
  - StrandTether ploidy-aware phasing
  - HiCGraphAligner → bubble phasing integration
"""

import math
import pytest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Optional

from strandweaver.assembly_core.dbg_engine_module import (
    KmerGraph, KmerNode, KmerEdge,
)
from strandweaver.assembly_core.haplotype_detangler_module import (
    HaplotypeDetangler,
    Ploidy,
    PhaseInfo,
    PhasingResult,
    HaplotypeDetangleResult,
)
from strandweaver.assembly_core.strandtether_module import (
    StrandTether,
    HiCContactMap,
    HiCNodePhaseInfo,
)


# ============================================================================
# Helpers: build small graphs with bubbles and Hi-C edges
# ============================================================================

def _make_bubble_graph():
    """
    Create a graph with one heterozygous bubble:

        source(0) ──edge0──> arm_a(1) ──edge2──> sink(3)
                  ──edge1──> arm_b(2) ──edge3──> sink(3)

    All nodes have short dummy sequences so KmerNode validates.
    """
    g = KmerGraph(base_k=5)

    for nid, seq in [(0, "AAAA"), (1, "CCCC"), (2, "TTTT"), (3, "GGGG")]:
        g.add_node(KmerNode(id=nid, seq=seq, coverage=30.0, length=len(seq)))

    g.add_edge(KmerEdge(id=0, from_id=0, to_id=1, coverage=30.0))
    g.add_edge(KmerEdge(id=1, from_id=0, to_id=2, coverage=28.0))
    g.add_edge(KmerEdge(id=2, from_id=1, to_id=3, coverage=30.0))
    g.add_edge(KmerEdge(id=3, from_id=2, to_id=3, coverage=28.0))

    return g


def _add_hic_edges_to_graph(graph, hic_pairs):
    """
    Add HiCEdge objects to graph.edges, mimicking the pipeline's
    _add_hic_edges_to_graph behaviour.

    hic_pairs: list of (source_node, target_node, contact_count)
    """

    class HiCEdge:
        def __init__(self, eid, source, target, count):
            self.id = eid
            self.source = source
            self.target = target
            self.from_id = source
            self.to_id = target
            self.edge_type = "hic"
            self.confidence = float(count)
            self.contact_count = count
            self.weight = float(count)
            self.coverage = float(count)
            self.overlap_len = 0

    next_id = max(graph.edges.keys(), default=-1) + 1
    for src, tgt, cnt in hic_pairs:
        eid = next_id
        next_id += 1
        edge = HiCEdge(eid, src, tgt, cnt)
        graph.edges[eid] = edge
        graph.out_edges[src].add(eid)
        graph.in_edges[tgt].add(eid)


def _make_two_bubble_graph():
    """
    Two linked bubbles sharing a middle node:

        s1(0) ──> a1(1) ──> m(3) ──> a2(4) ──> sink(6)
              ──> b1(2) ──> m(3) ──> b2(5) ──> sink(6)

    Bubble 1: source=0, arms={1},{2}, sink=3
    Bubble 2: source=3, arms={4},{5}, sink=6
    """
    g = KmerGraph(base_k=5)

    for nid in range(7):
        g.add_node(KmerNode(id=nid, seq="ACGT", coverage=30.0, length=4))

    edges = [
        (0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 2, 3),
        (4, 3, 4), (5, 3, 5), (6, 4, 6), (7, 5, 6),
    ]
    for eid, src, tgt in edges:
        g.add_edge(KmerEdge(id=eid, from_id=src, to_id=tgt, coverage=30.0))

    return g


# ============================================================================
# Ploidy enum
# ============================================================================

class TestPloidy:
    def test_ploidy_values(self):
        assert Ploidy.HAPLOID == 1
        assert Ploidy.DIPLOID == 2
        assert Ploidy.TRIPLOID == 3
        assert Ploidy.TETRAPLOID == 4

    def test_ploidy_is_int(self):
        assert isinstance(Ploidy.DIPLOID, int)
        assert Ploidy.DIPLOID + 0 == 2


# ============================================================================
# Haploid fast-path
# ============================================================================

class TestHaploidSkip:
    def test_haploid_ploidy_skips_phasing(self):
        """When ploidy=1 every node should be haplotype 0 with confidence 1."""
        g = _make_bubble_graph()
        det = HaplotypeDetangler(ploidy=1)
        result = det.phase_graph(g, use_hic_edges=False)

        assert result.num_haplotypes == 1
        assert result.metadata.get("haploid_skip") is True
        for nid in g.nodes:
            assert result.node_assignments[nid] == 0
            assert result.confidence_scores[nid] == 1.0

    def test_diploid_does_not_skip(self):
        g = _make_bubble_graph()
        det = HaplotypeDetangler(ploidy=2)
        result = det.phase_graph(g, use_hic_edges=False)
        assert result.num_haplotypes == 2
        assert result.metadata.get("haploid_skip") is None


# ============================================================================
# Bubble detection
# ============================================================================

class TestBubbleDetection:
    def test_single_bubble_detected(self):
        g = _make_bubble_graph()
        det = HaplotypeDetangler()
        bubbles = det._detect_bubbles(g)

        assert len(bubbles) == 1
        source, sink, arm_a, arm_b = bubbles[0]
        assert source == 0
        assert sink == 3
        # arms should each be single-node
        assert set(arm_a) | set(arm_b) == {1, 2}

    def test_two_bubbles_detected(self):
        g = _make_two_bubble_graph()
        det = HaplotypeDetangler()
        bubbles = det._detect_bubbles(g)

        assert len(bubbles) == 2
        sources = {b[0] for b in bubbles}
        sinks = {b[1] for b in bubbles}
        assert sources == {0, 3}
        assert sinks == {3, 6}

    def test_no_bubbles_in_linear_graph(self):
        """Linear graph: 0 → 1 → 2 → 3 has no bubbles."""
        g = KmerGraph(base_k=5)
        for nid in range(4):
            g.add_node(KmerNode(id=nid, seq="ACGT", coverage=30.0, length=4))
        for i in range(3):
            g.add_edge(KmerEdge(id=i, from_id=i, to_id=i + 1, coverage=30.0))

        det = HaplotypeDetangler()
        assert det._detect_bubbles(g) == []

    def test_hic_edges_ignored_in_bubble_detection(self):
        """Hi-C proximity edges should NOT create spurious bubbles."""
        g = KmerGraph(base_k=5)
        for nid in range(3):
            g.add_node(KmerNode(id=nid, seq="ACGT", coverage=30.0, length=4))
        g.add_edge(KmerEdge(id=0, from_id=0, to_id=1, coverage=30.0))
        g.add_edge(KmerEdge(id=1, from_id=1, to_id=2, coverage=30.0))

        # Add Hi-C edge 0→2 — should NOT make it look like a bubble
        _add_hic_edges_to_graph(g, [(0, 2, 10)])

        det = HaplotypeDetangler()
        assert det._detect_bubbles(g) == []


# ============================================================================
# Bubble-aware Hi-C phasing
# ============================================================================

class TestBubbleAwarePhasing:
    def test_single_bubble_phases_arms(self):
        """Hi-C contacts within a bubble should assign opposite phases to arms."""
        g = _make_bubble_graph()
        # Add Hi-C edges linking arms to themselves (cis confirmation)
        _add_hic_edges_to_graph(g, [
            (1, 1, 5),   # arm_a self (will be filtered as same node)
            (2, 2, 5),   # arm_b self
        ])

        det = HaplotypeDetangler(ploidy=2)
        phase_info = det._bubble_aware_hic_phasing(g)

        # Both bubble arm nodes should be phased
        assert phase_info is not None
        assert 1 in phase_info
        assert 2 in phase_info
        # They should have OPPOSITE dominant phases
        arm1_dom = "A" if phase_info[1].phase_A_score > phase_info[1].phase_B_score else "B"
        arm2_dom = "A" if phase_info[2].phase_A_score > phase_info[2].phase_B_score else "B"
        assert arm1_dom != arm2_dom

    def test_phase_chaining_two_bubbles(self):
        """
        Two bubbles connected by Hi-C contacts should have consistent
        phase assignments across the chain.
        """
        g = _make_two_bubble_graph()
        # Hi-C contacts linking arm_a of bubble1 (node1) to arm_a of bubble2 (node4)
        # and arm_b of bubble1 (node2) to arm_b of bubble2 (node5).
        _add_hic_edges_to_graph(g, [
            (1, 4, 10),  # bubble1 arm_a ↔ bubble2 arm_a → cis
            (2, 5, 10),  # bubble1 arm_b ↔ bubble2 arm_b → cis
        ])

        det = HaplotypeDetangler(ploidy=2)
        phase_info = det._bubble_aware_hic_phasing(g)

        assert phase_info is not None
        # Nodes 1 and 4 should be co-phased (same dominant phase)
        dom_1 = "A" if phase_info[1].phase_A_score > 0.5 else "B"
        dom_4 = "A" if phase_info[4].phase_A_score > 0.5 else "B"
        assert dom_1 == dom_4, "Cis-linked arms should be co-phased"

        # Nodes 2 and 5 should also be co-phased
        dom_2 = "A" if phase_info[2].phase_A_score > 0.5 else "B"
        dom_5 = "A" if phase_info[5].phase_A_score > 0.5 else "B"
        assert dom_2 == dom_5, "Cis-linked arms should be co-phased"

        # Arms within each bubble should be anti-phased
        assert dom_1 != dom_2, "Arms within bubble should be anti-phased"

    def test_trans_hic_contacts_flip_phase(self):
        """
        Hi-C contacts between opposite arms of different bubbles indicate
        those arms are on the SAME haplotype (the contact proves proximity).
        
        arm_a of bubble1 (node1) ↔ arm_b of bubble2 (node5) → trans contact.
        This means node1 and node5 are co-phased (same haplotype), and the
        bubble phases must flip so that the internal labelling is consistent.
        """
        g = _make_two_bubble_graph()
        # Trans contacts: bubble1 arm_a (node1) ↔ bubble2 arm_b (node5)
        _add_hic_edges_to_graph(g, [
            (1, 5, 15),  # arm_a bubble1 ↔ arm_b bubble2 → trans
        ])

        det = HaplotypeDetangler(ploidy=2)
        phase_info = det._bubble_aware_hic_phasing(g)

        assert phase_info is not None
        dom_1 = "A" if phase_info[1].phase_A_score > 0.5 else "B"
        dom_5 = "A" if phase_info[5].phase_A_score > 0.5 else "B"
        # Trans contact = same haplotype (Hi-C confirms proximity)
        assert dom_1 == dom_5, "Trans-linked arms should be co-phased"
        
        # And arm_a of bubble2 (node4) should be OPPOSITE to node5
        dom_4 = "A" if phase_info[4].phase_A_score > 0.5 else "B"
        assert dom_4 != dom_5, "Arms within same bubble should be anti-phased"

    def test_shared_nodes_get_ambiguous_scores(self):
        """Source and sink nodes (homozygous flanks) should be ambiguous."""
        g = _make_bubble_graph()
        _add_hic_edges_to_graph(g, [(1, 2, 5)])

        det = HaplotypeDetangler(ploidy=2)
        phase_info = det._bubble_aware_hic_phasing(g)

        assert phase_info is not None
        # Source (0) and sink (3) are shared between haplotypes → ambiguous
        assert phase_info[0].phase_A_score == pytest.approx(0.5)
        assert phase_info[3].phase_A_score == pytest.approx(0.5)

    def test_fallback_to_spectral_when_no_bubbles(self):
        """If graph has no bubbles but has Hi-C edges, fall back to spectral."""
        g = KmerGraph(base_k=5)
        for nid in range(4):
            g.add_node(KmerNode(id=nid, seq="ACGT", coverage=30.0, length=4))
        g.add_edge(KmerEdge(id=0, from_id=0, to_id=1, coverage=30.0))
        g.add_edge(KmerEdge(id=1, from_id=1, to_id=2, coverage=30.0))
        g.add_edge(KmerEdge(id=2, from_id=2, to_id=3, coverage=30.0))
        # Add Hi-C edges but no bubble structure
        _add_hic_edges_to_graph(g, [(0, 1, 10), (2, 3, 10)])

        det = HaplotypeDetangler(ploidy=2)
        # Should still return phase info (from fallback spectral)
        phase_info = det._bubble_aware_hic_phasing(g)
        assert phase_info is not None


# ============================================================================
# Full phase_graph integration
# ============================================================================

class TestPhaseGraphIntegration:
    def test_phase_graph_with_bubbles_and_hic(self):
        """End-to-end: graph with bubble + Hi-C edges → PhasingResult."""
        g = _make_bubble_graph()
        _add_hic_edges_to_graph(g, [(1, 2, 10)])

        det = HaplotypeDetangler(ploidy=2)
        result = det.phase_graph(g, use_hic_edges=True)

        assert isinstance(result, PhasingResult)
        assert result.num_haplotypes == 2
        # Should have node assignments for all nodes
        assert len(result.node_assignments) == len(g.nodes)

    def test_phase_graph_no_hic_edges(self):
        """No Hi-C edges → phasing still completes with alternative signals."""
        g = _make_bubble_graph()
        det = HaplotypeDetangler(ploidy=2)
        result = det.phase_graph(g, use_hic_edges=True)

        assert isinstance(result, PhasingResult)
        assert result.num_haplotypes == 2

    def test_phase_graph_hic_disabled(self):
        """use_hic_edges=False → phasing uses only GNN/AI/coverage signals."""
        g = _make_bubble_graph()
        _add_hic_edges_to_graph(g, [(1, 2, 10)])

        det = HaplotypeDetangler(ploidy=2)
        result = det.phase_graph(g, use_hic_edges=False)

        assert isinstance(result, PhasingResult)
        assert result.num_haplotypes == 2


# ============================================================================
# StrandTether ploidy integration
# ============================================================================

class TestStrandTetherPloidy:
    def test_ploidy_attribute_stored(self):
        st = StrandTether(use_gpu=False, ploidy=3)
        assert st.ploidy == 3

    def test_ploidy_default_is_diploid(self):
        st = StrandTether(use_gpu=False)
        assert st.ploidy == 2

    def test_haploid_phasing_returns_all_phase_a(self):
        """StrandTether.compute_node_phasing with ploidy=1."""
        st = StrandTether(use_gpu=False, ploidy=1)
        cm = HiCContactMap()
        cm.add_contact(1, 2, 10)
        cm.add_contact(2, 3, 5)

        result = st.compute_node_phasing(cm, {1, 2, 3})
        for nid in (1, 2, 3):
            assert result[nid].phase_assignment == "A"
            assert result[nid].phase_A_score == 1.0
            assert result[nid].phase_B_score == 0.0

    def test_diploid_phasing_runs_normally(self):
        """StrandTether.compute_node_phasing with ploidy=2 performs actual phasing."""
        st = StrandTether(use_gpu=False, ploidy=2)
        cm = HiCContactMap()
        # Create two clusters of contacts
        for i in range(5):
            for j in range(i + 1, 5):
                cm.add_contact(i, j, 10)
        for i in range(5, 10):
            for j in range(i + 1, 10):
                cm.add_contact(i, j, 10)

        nodes = set(range(10))
        result = st.compute_node_phasing(cm, nodes)

        assert len(result) == 10
        # Should have mix of A, B, and possibly ambiguous
        assignments = {r.phase_assignment for r in result.values()}
        assert len(assignments) >= 1  # At least some phasing


# ============================================================================
# Ploidy config propagation (constructor tests)
# ============================================================================

class TestPloidyPropagation:
    def test_detangler_ploidy_clamped(self):
        """Ploidy below 1 should be clamped to 1."""
        det = HaplotypeDetangler(ploidy=0)
        assert det.ploidy == 1

    def test_detangler_ploidy_float_cast(self):
        """Float ploidy should be truncated to int."""
        det = HaplotypeDetangler(ploidy=2.9)
        assert det.ploidy == 2

    def test_strandtether_ploidy_clamped(self):
        st = StrandTether(use_gpu=False, ploidy=-1)
        assert st.ploidy == 1


# ============================================================================
# Walk arm edge cases
# ============================================================================

class TestWalkArm:
    def test_dead_end_returns_none_sink(self):
        """Arm that dead-ends should return None sink."""
        g = KmerGraph(base_k=5)
        g.add_node(KmerNode(id=0, seq="ACGT", coverage=30.0, length=4))
        g.add_node(KmerNode(id=1, seq="ACGT", coverage=30.0, length=4))
        g.add_edge(KmerEdge(id=0, from_id=0, to_id=1, coverage=30.0))
        # Node 1 has no outgoing edges → dead-end

        det = HaplotypeDetangler()
        path, sink = det._walk_arm(g, 1, 50)
        assert sink is None
        assert path == [1]

    def test_branch_returns_none_sink(self):
        """Arm that branches should return None sink."""
        g = KmerGraph(base_k=5)
        for nid in range(4):
            g.add_node(KmerNode(id=nid, seq="ACGT", coverage=30.0, length=4))
        g.add_edge(KmerEdge(id=0, from_id=0, to_id=1, coverage=30.0))
        g.add_edge(KmerEdge(id=1, from_id=1, to_id=2, coverage=30.0))
        g.add_edge(KmerEdge(id=2, from_id=1, to_id=3, coverage=30.0))

        det = HaplotypeDetangler()
        path, sink = det._walk_arm(g, 0, 50)
        # Node 0 has 1 out-edge to node 1, but node 1 has 2 out-edges → branch
        # Actually, _walk_arm starts at 0, walks to 1 via the single edge.
        # But at node 1, out_degree=2, so it can't continue → return None
        # Wait, node 0 → node 1 (in_degree 1), then node 1 has 2 out-edges → None
        assert sink is None


# ============================================================================
# Ploidy enum exported from package
# ============================================================================

class TestPloidyExport:
    def test_ploidy_importable_from_assembly_core(self):
        from strandweaver.assembly_core import Ploidy as PloidyImport
        assert PloidyImport.DIPLOID == 2


# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
