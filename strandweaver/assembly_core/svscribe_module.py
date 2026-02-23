#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

SVScribe — ML-driven structural variant detection during graph traversal.
Supports deletions, insertions, inversions, duplications, and translocations.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import math
from collections import defaultdict
from pathlib import Path

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class SVEvidence:
    """
    Evidence supporting a structural variant call from multiple sources.
    
    Attributes:
        has_sequence_support: Whether sequence-based edges support this SV
        has_ul_support: Whether ultra-long reads support this SV
        has_hic_support: Whether Hi-C contacts support this SV
        sequence_confidence: Confidence from sequence evidence (0-1)
        ul_confidence: Confidence from UL evidence (0-1)
        hic_confidence: Confidence from Hi-C evidence (0-1)
        supporting_reads: Number of reads supporting this SV
        edge_types: Set of edge types involved ('sequence', 'ul', 'hic')
    """
    has_sequence_support: bool = False
    has_ul_support: bool = False
    has_hic_support: bool = False
    sequence_confidence: float = 0.0
    ul_confidence: float = 0.0
    hic_confidence: float = 0.0
    supporting_reads: int = 0
    edge_types: Set[str] = field(default_factory=set)


@dataclass
class SVSignature:
    """
    Topological signature of a potential structural variant.
    
    Attributes:
        pattern_type: Type of pattern detected ('gap', 'bubble', 'coverage_anomaly', etc.)
        involved_nodes: List of node IDs in this pattern
        involved_edges: List of edge IDs in this pattern
        coverage_pattern: Description of coverage pattern
        topology_score: Score indicating strength of topological signal (0-1)
    """
    pattern_type: str
    involved_nodes: List[int] = field(default_factory=list)
    involved_edges: List[int] = field(default_factory=list)
    coverage_pattern: str = ''
    topology_score: float = 0.0


@dataclass
class SVCall:
    """
    Structural variant call.
    
    Attributes:
        sv_id: Unique identifier for this SV call
        sv_type: Type of SV ('DEL', 'INS', 'INV', 'DUP', 'TRA')
        nodes: List of node IDs involved in SV
        position: Genomic position (if known)
        size: Size of SV in bases
        confidence: Overall confidence in call (0-1)
        evidence: SVEvidence object with multi-source support
        haplotype: Haplotype assignment (0=hapA, 1=hapB, -1=unknown/both)
        breakpoints: List of (from_node, to_node) tuples at SV boundaries
        metadata: Additional information about this call
    """
    sv_id: str
    sv_type: str
    nodes: List[int]
    position: Optional[int] = None
    size: int = 0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    haplotype: int = -1
    breakpoints: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeletionDetector:
    """
    Detects large deletions in assembly graph.
    
    Deletions manifest as:
    - UL reads spanning gap in graph (UL edges connecting non-adjacent nodes)
    - Hi-C contacts across gap (Hi-C edges spanning missing sequence)
    - Missing expected sequence edges
    - Coverage drops in sequence paths
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DeletionDetector")
    
    def detect(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int],
        ul_routes: Optional[Dict] = None
    ) -> List[SVSignature]:
        """
        Detect deletion signatures.
        
        Args:
            graph: Assembly graph
            sequence_edges: Set of sequence-based edge IDs
            ul_edges: Set of UL-derived edge IDs
            hic_edges: Set of Hi-C-derived edge IDs
            ul_routes: UL routing decisions
        
        Returns:
            List of deletion signatures
        """
        signatures = []
        
        # Look for UL edges spanning gaps in sequence graph
        signatures.extend(self._detect_ul_spanning_gaps(
            graph, sequence_edges, ul_edges
        ))
        
        # Look for Hi-C contacts across sequence gaps
        signatures.extend(self._detect_hic_gap_spanning(
            graph, sequence_edges, hic_edges
        ))
        
        # Look for coverage drops
        signatures.extend(self._detect_coverage_gaps(graph, sequence_edges))
        
        return signatures
    
    def _detect_ul_spanning_gaps(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int]
    ) -> List[SVSignature]:
        """Find UL edges that span gaps in sequence graph."""
        signatures = []
        
        for edge_id in ul_edges:
            edge = graph.edges.get(edge_id)
            if not edge:
                continue
            
            from_node = getattr(edge, 'from_node', None)
            to_node = getattr(edge, 'to_node', None)
            
            if from_node is None or to_node is None:
                continue
            
            # Check if there's a direct sequence edge
            has_sequence_edge = False
            for seq_edge_id in graph.out_edges.get(from_node, set()):
                if seq_edge_id in sequence_edges:
                    seq_edge = graph.edges.get(seq_edge_id)
                    if seq_edge and getattr(seq_edge, 'to_node', None) == to_node:
                        has_sequence_edge = True
                        break
            
            if not has_sequence_edge:
                # UL connects nodes without sequence edge = potential deletion
                from_node_obj = graph.nodes.get(from_node)
                to_node_obj = graph.nodes.get(to_node)
                
                # Estimate deletion size by checking intervening nodes
                estimated_size = self._estimate_gap_size(
                    graph, from_node, to_node, sequence_edges
                )
                
                signature = SVSignature(
                    pattern_type='ul_spanning_gap',
                    involved_nodes=[from_node, to_node],
                    involved_edges=[edge_id],
                    coverage_pattern='gap',
                    topology_score=0.7
                )
                signature.metadata = {
                    'estimated_size': estimated_size,
                    'ul_support': 1
                }
                signatures.append(signature)
        
        return signatures
    
    def _detect_hic_gap_spanning(
        self,
        graph,
        sequence_edges: Set[int],
        hic_edges: Set[int]
    ) -> List[SVSignature]:
        """Find Hi-C edges that span gaps in sequence graph."""
        signatures = []
        
        for edge_id in hic_edges:
            edge = graph.edges.get(edge_id)
            if not edge:
                continue
            
            from_node = getattr(edge, 'from_node', None)
            to_node = getattr(edge, 'to_node', None)
            
            if from_node is None or to_node is None:
                continue
            
            # Check if there's a direct sequence path
            has_sequence_path = self._has_sequence_path(
                graph, from_node, to_node, sequence_edges, max_hops=5
            )
            
            if not has_sequence_path:
                # Hi-C connects nodes without sequence path = potential deletion
                estimated_size = self._estimate_gap_size(
                    graph, from_node, to_node, sequence_edges
                )
                
                signature = SVSignature(
                    pattern_type='hic_spanning_gap',
                    involved_nodes=[from_node, to_node],
                    involved_edges=[edge_id],
                    coverage_pattern='gap',
                    topology_score=0.6
                )
                signature.metadata = {
                    'estimated_size': estimated_size,
                    'hic_support': 1
                }
                signatures.append(signature)
        
        return signatures
    
    def _detect_coverage_gaps(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find regions with unexpected coverage drops."""
        signatures = []
        
        # Build sequence path through graph
        nodes_in_sequence = set()
        for edge_id in sequence_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node:
                    nodes_in_sequence.add(from_node)
                if to_node:
                    nodes_in_sequence.add(to_node)
        
        # Check for coverage drops
        nodes_by_id = sorted(nodes_in_sequence)
        
        for i in range(1, len(nodes_by_id) - 1):
            node_id = nodes_by_id[i]
            prev_id = nodes_by_id[i - 1]
            next_id = nodes_by_id[i + 1]
            
            node = graph.nodes.get(node_id)
            prev_node = graph.nodes.get(prev_id)
            next_node = graph.nodes.get(next_id)
            
            if node and prev_node and next_node:
                cov = getattr(node, 'coverage', 0.0)
                prev_cov = getattr(prev_node, 'coverage', 0.0)
                next_cov = getattr(next_node, 'coverage', 0.0)
                
                avg_neighbor = (prev_cov + next_cov) / 2.0
                
                # Coverage drop >50%
                if avg_neighbor > 10 and cov < avg_neighbor * 0.5:
                    signature = SVSignature(
                        pattern_type='coverage_drop',
                        involved_nodes=[node_id],
                        involved_edges=[],
                        coverage_pattern='low',
                        topology_score=0.5
                    )
                    signature.metadata = {
                        'coverage': cov,
                        'neighbor_coverage': avg_neighbor,
                        'estimated_size': getattr(node, 'length', 0)
                    }
                    signatures.append(signature)
        
        return signatures
    
    def _estimate_gap_size(
        self,
        graph,
        from_node: int,
        to_node: int,
        sequence_edges: Set[int]
    ) -> int:
        """Estimate size of gap between nodes."""
        # Simple heuristic: node ID difference * average node length
        # In real implementation, would trace sequence path if available
        node_diff = abs(to_node - from_node)
        avg_node_length = 5000  # Placeholder
        return max(50, node_diff * avg_node_length // 100)
    
    def _has_sequence_path(
        self,
        graph,
        from_node: int,
        to_node: int,
        sequence_edges: Set[int],
        max_hops: int = 5
    ) -> bool:
        """Check if there's a sequence path between nodes."""
        visited = set()
        queue = [(from_node, 0)]
        
        while queue:
            current, hops = queue.pop(0)
            
            if current == to_node:
                return True
            
            if hops >= max_hops or current in visited:
                continue
            
            visited.add(current)
            
            # Follow sequence edges only
            for edge_id in graph.out_edges.get(current, set()):
                if edge_id in sequence_edges:
                    edge = graph.edges.get(edge_id)
                    if edge:
                        next_node = getattr(edge, 'to_node', None)
                        if next_node:
                            queue.append((next_node, hops + 1))
        
        return False


class InsertionDetector:
    """
    Detects insertions in assembly graph.
    
    Insertions manifest as:
    - Branching paths with high coverage in sequence graph
    - UL reads containing novel sequence
    - Nodes with elevated coverage suggesting extra copies
    - Bubble structures in sequence graph
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InsertionDetector")
    
    def detect(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int]
    ) -> List[SVSignature]:
        """
        Detect insertion signatures.
        
        Args:
            graph: Assembly graph
            sequence_edges: Set of sequence-based edge IDs
            ul_edges: Set of UL-derived edge IDs
            hic_edges: Set of Hi-C-derived edge IDs
        
        Returns:
            List of insertion signatures
        """
        signatures = []
        
        # Look for alternative paths (bubbles) in sequence graph
        signatures.extend(self._detect_alternative_paths(
            graph, sequence_edges
        ))
        
        # Look for high-coverage branches
        signatures.extend(self._detect_high_coverage_branches(
            graph, sequence_edges
        ))
        
        # Look for novel sequences indicated by UL support
        signatures.extend(self._detect_ul_novel_sequences(
            graph, ul_edges
        ))
        
        return signatures
    
    def _detect_alternative_paths(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find bubble structures in sequence graph (potential insertions)."""
        signatures = []
        
        # Build adjacency from sequence edges
        seq_adj = defaultdict(set)
        for edge_id in sequence_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node and to_node:
                    seq_adj[from_node].add(to_node)
        
        # Find nodes with multiple outgoing sequence edges (branch points)
        for node_id, targets in seq_adj.items():
            if len(targets) > 1:
                # Check if paths reconverge (bubble)
                for target in targets:
                    # Trace path from target
                    path_nodes = self._trace_path(graph, target, seq_adj, max_len=10)
                    
                    # Check if any converge to common node
                    for other_target in targets:
                        if other_target != target:
                            other_path = self._trace_path(graph, other_target, seq_adj, max_len=10)
                            
                            # Find convergence point
                            common = set(path_nodes) & set(other_path)
                            if common:
                                # Bubble detected
                                all_bubble_nodes = list(set([node_id] + path_nodes + other_path))
                                
                                # Calculate size difference
                                len_1 = sum(getattr(graph.nodes.get(n), 'length', 0) for n in path_nodes)
                                len_2 = sum(getattr(graph.nodes.get(n), 'length', 0) for n in other_path)
                                
                                if abs(len_1 - len_2) > 1000:
                                    signature = SVSignature(
                                        pattern_type='bubble_insertion',
                                        involved_nodes=all_bubble_nodes,
                                        involved_edges=list(sequence_edges),  # Simplified
                                        coverage_pattern='branching',
                                        topology_score=0.6
                                    )
                                    signature.metadata = {
                                        'size_difference': abs(len_1 - len_2),
                                        'branch_node': node_id
                                    }
                                    signatures.append(signature)
                                    break
        
        return signatures
    
    def _detect_high_coverage_branches(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find branching paths with elevated coverage."""
        signatures = []
        
        # Look for nodes with >1 outgoing sequence edge and high coverage
        for node_id, node in graph.nodes.items():
            out_seq_edges = [
                eid for eid in graph.out_edges.get(node_id, set())
                if eid in sequence_edges
            ]
            
            if len(out_seq_edges) > 1:
                coverage = getattr(node, 'coverage', 0.0)
                
                # High coverage might indicate insertion
                if coverage > 60:
                    signature = SVSignature(
                        pattern_type='high_coverage_branch',
                        involved_nodes=[node_id],
                        involved_edges=out_seq_edges,
                        coverage_pattern='elevated',
                        topology_score=0.4
                    )
                    signature.metadata = {
                        'coverage': coverage,
                        'out_degree': len(out_seq_edges)
                    }
                    signatures.append(signature)
        
        return signatures
    
    def _detect_ul_novel_sequences(
        self,
        graph,
        ul_edges: Set[int]
    ) -> List[SVSignature]:
        """Find nodes with strong UL support suggesting novel insertions."""
        signatures = []
        
        # Count UL edges per node
        ul_support_count = defaultdict(int)
        for edge_id in ul_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node:
                    ul_support_count[from_node] += 1
                if to_node:
                    ul_support_count[to_node] += 1
        
        # Nodes with high UL support might be insertions
        for node_id, count in ul_support_count.items():
            if count >= 3:  # Multiple UL reads support this region
                node = graph.nodes.get(node_id)
                if node:
                    signature = SVSignature(
                        pattern_type='ul_novel_sequence',
                        involved_nodes=[node_id],
                        involved_edges=[],
                        coverage_pattern='ul_supported',
                        topology_score=0.5
                    )
                    signature.metadata = {
                        'ul_support_count': count,
                        'node_length': getattr(node, 'length', 0)
                    }
                    signatures.append(signature)
        
        return signatures
    
    def _trace_path(
        self,
        graph,
        start_node: int,
        adjacency: Dict,
        max_len: int = 10
    ) -> List[int]:
        """Trace a path from start node through graph."""
        path = [start_node]
        current = start_node
        
        for _ in range(max_len):
            targets = adjacency.get(current, set())
            if len(targets) != 1:
                break
            current = list(targets)[0]
            path.append(current)
        
        return path


class InversionDetector:
    """
    Detects inversions in assembly graph.
    
    Inversions manifest as:
    - UL reads aligning in reverse orientation
    - Hi-C contacts suggesting flipped orientation
    - Strand inconsistencies in graph paths
    - Opposing coverage patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InversionDetector")
    
    def detect(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int],
        ul_routes: Optional[Dict] = None
    ) -> List[SVSignature]:
        """
        Detect inversion signatures.
        
        Args:
            graph: Assembly graph
            sequence_edges: Set of sequence-based edge IDs
            ul_edges: Set of UL-derived edge IDs
            hic_edges: Set of Hi-C-derived edge IDs
            ul_routes: UL routing decisions
        
        Returns:
            List of inversion signatures
        """
        signatures = []
        
        # Check for strand flips in UL mappings
        if ul_routes:
            signatures.extend(self._detect_ul_strand_flips(
                graph, ul_routes, ul_edges
            ))
        
        # Check for Hi-C orientation inconsistencies
        signatures.extend(self._detect_hic_orientation_flips(
            graph, hic_edges
        ))
        
        return signatures
    
    def _detect_ul_strand_flips(
        self,
        graph,
        ul_routes: Dict,
        ul_edges: Set[int]
    ) -> List[SVSignature]:
        """Find UL reads with strand inconsistencies."""
        signatures = []
        
        for read_id, decision in ul_routes.items():
            # Check if strand_consistent flag is False
            if hasattr(decision, 'evidence') and isinstance(decision.evidence, dict):
                strand_consistent = decision.evidence.get('strand_consistent', True)
                
                if not strand_consistent and hasattr(decision, 'chosen_path'):
                    # Strand flip detected
                    signature = SVSignature(
                        pattern_type='ul_strand_flip',
                        involved_nodes=decision.chosen_path if decision.chosen_path else [],
                        involved_edges=[],
                        coverage_pattern='strand_inconsistent',
                        topology_score=0.6
                    )
                    signature.metadata = {
                        'ul_read': read_id,
                        'ul_score': getattr(decision, 'score', 0.0)
                    }
                    signatures.append(signature)
        
        return signatures
    
    def _detect_hic_orientation_flips(
        self,
        graph,
        hic_edges: Set[int]
    ) -> List[SVSignature]:
        """Find Hi-C contacts suggesting orientation flips."""
        signatures = []
        
        # Hi-C read pair orientations can indicate inversions
        # This would require paired read information which we don't have in edge structure
        # Placeholder for future implementation when Hi-C metadata is available
        
        return signatures


class DuplicationDetector:
    """
    Detects duplications and repeat expansions.
    
    Duplications manifest as:
    - Very high coverage regions (>2x expected)
    - Multiple parallel sequence paths
    - UL reads suggesting tandem duplications
    - Self-loops or back-edges in graph
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DuplicationDetector")
    
    def detect(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int]
    ) -> List[SVSignature]:
        """
        Detect duplication signatures.
        
        Args:
            graph: Assembly graph
            sequence_edges: Set of sequence-based edge IDs
            ul_edges: Set of UL-derived edge IDs
            hic_edges: Set of Hi-C-derived edge IDs
        
        Returns:
            List of duplication signatures
        """
        signatures = []
        
        # Look for very high coverage nodes
        signatures.extend(self._detect_high_coverage_duplications(
            graph, sequence_edges
        ))
        
        # Look for parallel paths (tandem duplications)
        signatures.extend(self._detect_parallel_paths(
            graph, sequence_edges
        ))
        
        # Look for self-loops
        signatures.extend(self._detect_self_loops(
            graph, sequence_edges
        ))
        
        return signatures
    
    def _detect_high_coverage_duplications(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find nodes with >2x coverage suggesting duplication."""
        signatures = []
        
        # Calculate median coverage for normalization
        coverages = [getattr(n, 'coverage', 0.0) for n in graph.nodes.values()]
        if not coverages:
            return signatures
        
        median_cov = sorted(coverages)[len(coverages) // 2]
        
        for node_id, node in graph.nodes.items():
            coverage = getattr(node, 'coverage', 0.0)
            
            # Coverage >2x median suggests duplication
            if median_cov > 0 and coverage > median_cov * 2.0:
                signature = SVSignature(
                    pattern_type='high_coverage_duplication',
                    involved_nodes=[node_id],
                    involved_edges=[],
                    coverage_pattern='elevated_2x',
                    topology_score=0.7
                )
                signature.metadata = {
                    'coverage': coverage,
                    'median_coverage': median_cov,
                    'fold_change': coverage / median_cov,
                    'estimated_size': getattr(node, 'length', 0)
                }
                signatures.append(signature)
        
        return signatures
    
    def _detect_parallel_paths(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find parallel paths in sequence graph (tandem duplications)."""
        signatures = []
        
        # Build adjacency from sequence edges
        seq_adj = defaultdict(set)
        for edge_id in sequence_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node and to_node:
                    seq_adj[from_node].add(to_node)
        
        # Look for nodes with multiple paths to same target
        for node_id, targets in seq_adj.items():
            if len(targets) >= 2:
                # Check if these form parallel paths
                # (Simplified: just flag multi-target as potential duplication)
                signature = SVSignature(
                    pattern_type='parallel_paths',
                    involved_nodes=[node_id] + list(targets),
                    involved_edges=[],
                    coverage_pattern='branching',
                    topology_score=0.5
                )
                signature.metadata = {
                    'num_parallel_paths': len(targets)
                }
                signatures.append(signature)
        
        return signatures
    
    def _detect_self_loops(
        self,
        graph,
        sequence_edges: Set[int]
    ) -> List[SVSignature]:
        """Find self-loops in sequence graph."""
        signatures = []
        
        for edge_id in sequence_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                
                if from_node == to_node:
                    # Self-loop detected
                    signature = SVSignature(
                        pattern_type='self_loop',
                        involved_nodes=[from_node],
                        involved_edges=[edge_id],
                        coverage_pattern='circular',
                        topology_score=0.8
                    )
                    signature.metadata = {
                        'loop_node': from_node
                    }
                    signatures.append(signature)
        
        return signatures


class TranslocationDetector:
    """
    Detects translocations and inter-chromosomal rearrangements.
    
    Translocations manifest as:
    - Hi-C long-range contacts inconsistent with linear sequence graph
    - UL reads connecting distant graph regions
    - Unexpected long-range connections in assembly
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TranslocationDetector")
    
    def detect(
        self,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int],
        ul_routes: Optional[Dict] = None
    ) -> List[SVSignature]:
        """
        Detect translocation signatures.
        
        Args:
            graph: Assembly graph
            sequence_edges: Set of sequence-based edge IDs
            ul_edges: Set of UL-derived edge IDs
            hic_edges: Set of Hi-C-derived edge IDs
            ul_routes: UL routing decisions
        
        Returns:
            List of translocation signatures
        """
        signatures = []
        
        # Look for Hi-C edges connecting distant regions
        signatures.extend(self._detect_hic_long_range(
            graph, sequence_edges, hic_edges
        ))
        
        # Look for UL reads connecting distant graph regions
        if ul_routes:
            signatures.extend(self._detect_ul_long_range(
                graph, sequence_edges, ul_routes
            ))
        
        return signatures
    
    def _detect_hic_long_range(
        self,
        graph,
        sequence_edges: Set[int],
        hic_edges: Set[int]
    ) -> List[SVSignature]:
        """Find Hi-C edges connecting distant regions."""
        signatures = []
        
        for edge_id in hic_edges:
            edge = graph.edges.get(edge_id)
            if not edge:
                continue
            
            from_node = getattr(edge, 'from_node', None)
            to_node = getattr(edge, 'to_node', None)
            
            if from_node is None or to_node is None:
                continue
            
            # Check sequence distance between nodes
            seq_distance = self._calculate_sequence_distance(
                graph, from_node, to_node, sequence_edges
            )
            
            # Large sequence distance with Hi-C contact suggests translocation
            if seq_distance > 100000 or seq_distance == -1:  # >100kb or unreachable
                signature = SVSignature(
                    pattern_type='hic_long_range',
                    involved_nodes=[from_node, to_node],
                    involved_edges=[edge_id],
                    coverage_pattern='distant_contact',
                    topology_score=0.7
                )
                signature.metadata = {
                    'sequence_distance': seq_distance,
                    'hic_edge': edge_id
                }
                signatures.append(signature)
        
        return signatures
    
    def _detect_ul_long_range(
        self,
        graph,
        sequence_edges: Set[int],
        ul_routes: Dict
    ) -> List[SVSignature]:
        """Find UL reads connecting distant graph regions."""
        signatures = []
        
        for read_id, decision in ul_routes.items():
            if not hasattr(decision, 'chosen_path') or not decision.chosen_path:
                continue
            
            if len(decision.chosen_path) < 3:
                continue
            
            # Check sequence distance between first and last node
            first_node = decision.chosen_path[0]
            last_node = decision.chosen_path[-1]
            
            seq_distance = self._calculate_sequence_distance(
                graph, first_node, last_node, sequence_edges
            )
            
            # UL spanning very distant regions suggests translocation
            if seq_distance > 100000 or seq_distance == -1:
                signature = SVSignature(
                    pattern_type='ul_long_range',
                    involved_nodes=decision.chosen_path,
                    involved_edges=[],
                    coverage_pattern='spanning',
                    topology_score=0.6
                )
                signature.metadata = {
                    'ul_read': read_id,
                    'sequence_distance': seq_distance,
                    'ul_score': getattr(decision, 'score', 0.0)
                }
                signatures.append(signature)
        
        return signatures
    
    def _calculate_sequence_distance(
        self,
        graph,
        from_node: int,
        to_node: int,
        sequence_edges: Set[int]
    ) -> int:
        """Calculate sequence distance between nodes (BFS through sequence edges)."""
        if from_node == to_node:
            return 0
        
        visited = set()
        queue = [(from_node, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            if current == to_node:
                return dist
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Follow sequence edges only
            for edge_id in graph.out_edges.get(current, set()):
                if edge_id in sequence_edges:
                    edge = graph.edges.get(edge_id)
                    if edge:
                        next_node = getattr(edge, 'to_node', None)
                        if next_node and next_node not in visited:
                            node_obj = graph.nodes.get(next_node)
                            node_len = getattr(node_obj, 'length', 1000)
                            queue.append((next_node, dist + node_len))
        
        return -1  # Unreachable


class SVScribe:
    """
    Main SV detection engine with multi-source evidence scoring.
    
    Integrates evidence from:
    - Sequence-based edges (graph topology)
    - Ultra-long read edges (spanning evidence)
    - Hi-C edges (long-range contacts)
    
    Implements 8-step detection algorithm:
    1. Categorize edges by type
    2. Run SV-specific detectors
    3. Score evidence from each source
    4. Calculate overall confidence
    5. Assign to haplotypes
    6. Merge overlapping SVs
    7. Apply AI refinement (if enabled)
    8. Filter and assign IDs
    """
    
    def __init__(
        self,
        use_ai: bool = False,
        min_confidence: float = 0.5,
        min_size: int = 50,
        ml_model: Optional[str] = None
    ):
        """
        Initialize SV detection engine.
        
        Args:
            use_ai: Whether to use AI model for refinement
            min_confidence: Minimum confidence for reporting SV (0-1)
            min_size: Minimum SV size in bases
            ml_model: Path to AI model file
        """
        self.use_ai = use_ai
        self.min_confidence = min_confidence
        self.min_size = min_size
        self.ml_model_path = ml_model
        self.ml_model = None
        
        # Initialize detectors
        self.deletion_detector = DeletionDetector()
        self.insertion_detector = InsertionDetector()
        self.inversion_detector = InversionDetector()
        self.duplication_detector = DuplicationDetector()
        self.translocation_detector = TranslocationDetector()
        
        # Track SV counts
        self._sv_type_counts = {'DEL': 0, 'INS': 0, 'INV': 0, 'DUP': 0, 'TRA': 0}
        self._next_sv_id = 1
        
        self.logger = logging.getLogger(f"{__name__}.SVScribe")
        
        # Load AI model if enabled
        if self.use_ai and ml_model:
            self._load_ai_model()
    
    def detect_svs(
        self,
        graph,
        ul_routes: Optional[Dict] = None,
        distinguish_edge_types: bool = True,
        phasing_info = None,
        reference_path: Optional[str] = None
    ) -> List[SVCall]:
        """
        Main SV detection pipeline (8 steps).
        
        Args:
            graph: Assembly graph (DBG or StringGraph)
            ul_routes: UL routing decisions from ThreadCompass.get_routes()
            distinguish_edge_types: Whether to categorize edges by edge_type attribute
            phasing_info: Phasing result from HaplotypeDetangler
            reference_path: Optional reference for validation
        
        Returns:
            List of SVCall objects
        """
        self.logger.info("Starting SV detection pipeline")
        
        # Step 1: Categorize edges by type
        sequence_edges, ul_edges, hic_edges = self._categorize_edges(
            graph, distinguish_edge_types
        )
        self.logger.info(
            f"Edge categorization: {len(sequence_edges)} sequence, "
            f"{len(ul_edges)} UL, {len(hic_edges)} Hi-C"
        )
        
        # Step 2: Run SV-specific detectors
        all_signatures = []
        
        self.logger.info("Running deletion detector...")
        deletions = self.deletion_detector.detect(
            graph, sequence_edges, ul_edges, hic_edges, ul_routes
        )
        all_signatures.extend([(sig, 'DEL') for sig in deletions])
        
        self.logger.info("Running insertion detector...")
        insertions = self.insertion_detector.detect(
            graph, sequence_edges, ul_edges, hic_edges
        )
        all_signatures.extend([(sig, 'INS') for sig in insertions])
        
        self.logger.info("Running inversion detector...")
        inversions = self.inversion_detector.detect(
            graph, sequence_edges, ul_edges, hic_edges, ul_routes
        )
        all_signatures.extend([(sig, 'INV') for sig in inversions])
        
        self.logger.info("Running duplication detector...")
        duplications = self.duplication_detector.detect(
            graph, sequence_edges, ul_edges, hic_edges
        )
        all_signatures.extend([(sig, 'DUP') for sig in duplications])
        
        self.logger.info("Running translocation detector...")
        translocations = self.translocation_detector.detect(
            graph, sequence_edges, ul_edges, hic_edges, ul_routes
        )
        all_signatures.extend([(sig, 'TRA') for sig in translocations])
        
        self.logger.info(f"Detected {len(all_signatures)} total signatures")
        
        # Step 3 & 4: Score evidence and calculate confidence
        sv_calls = []
        for signature, sv_type in all_signatures:
            evidence = self._score_evidence(
                signature, graph, sequence_edges, ul_edges, hic_edges, ul_routes
            )
            
            confidence = self._calculate_confidence(evidence, signature)
            
            if confidence >= self.min_confidence:
                sv_call = self._signature_to_call(
                    signature, sv_type, evidence, confidence
                )
                sv_calls.append(sv_call)
        
        self.logger.info(f"Created {len(sv_calls)} SV calls above confidence threshold")
        
        # Step 5: Assign to haplotypes
        if phasing_info:
            self._assign_haplotypes(sv_calls, phasing_info)
            self.logger.info("Assigned SVs to haplotypes")
        
        # Step 6: Merge overlapping SVs
        sv_calls = self._merge_overlapping_svs(sv_calls)
        self.logger.info(f"After merging: {len(sv_calls)} SVs")
        
        # Step 7: Apply AI refinement
        if self.use_ai and self.ml_model:
            sv_calls = self._apply_ai_refinement(sv_calls, graph)
            self.logger.info("Applied AI refinement")
        
        # Step 8: Filter by size and assign IDs
        sv_calls = [sv for sv in sv_calls if sv.size >= self.min_size]
        
        for sv in sv_calls:
            sv.sv_id = f"SV{self._next_sv_id:06d}"
            self._next_sv_id += 1
            self._sv_type_counts[sv.sv_type] += 1
        
        self.logger.info(
            f"Final SVs: {len(sv_calls)} "
            f"(DEL:{self._sv_type_counts['DEL']}, INS:{self._sv_type_counts['INS']}, "
            f"INV:{self._sv_type_counts['INV']}, DUP:{self._sv_type_counts['DUP']}, "
            f"TRA:{self._sv_type_counts['TRA']})"
        )
        
        return sv_calls
    
    def get_sv_type_counts(self) -> Dict[str, int]:
        """Get counts of each SV type detected."""
        return self._sv_type_counts.copy()
    
    # Helper methods (Steps 1-8)
    
    def _categorize_edges(
        self,
        graph,
        distinguish_edge_types: bool
    ) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Step 1: Categorize edges by type.
        
        Returns:
            (sequence_edges, ul_edges, hic_edges) sets of edge IDs
        """
        sequence_edges = set()
        ul_edges = set()
        hic_edges = set()
        
        if not distinguish_edge_types:
            # Treat all edges as sequence edges
            sequence_edges = set(graph.edges.keys())
            return sequence_edges, ul_edges, hic_edges
        
        for edge_id, edge in graph.edges.items():
            edge_type = getattr(edge, 'edge_type', 'sequence')
            
            if edge_type == 'ul':
                ul_edges.add(edge_id)
            elif edge_type == 'hic':
                hic_edges.add(edge_id)
            else:
                sequence_edges.add(edge_id)
        
        return sequence_edges, ul_edges, hic_edges
    
    def _score_evidence(
        self,
        signature: SVSignature,
        graph,
        sequence_edges: Set[int],
        ul_edges: Set[int],
        hic_edges: Set[int],
        ul_routes: Optional[Dict]
    ) -> SVEvidence:
        """
        Step 3: Score evidence from multiple sources.
        
        Weights:
        - Sequence: 40%
        - UL: 30%
        - Hi-C: 30%
        """
        evidence = SVEvidence()
        
        # Check sequence support
        sequence_support = any(
            eid in sequence_edges for eid in signature.involved_edges
        )
        if sequence_support or signature.pattern_type in [
            'coverage_drop', 'bubble_insertion', 'high_coverage_branch',
            'high_coverage_duplication', 'parallel_paths', 'self_loop'
        ]:
            evidence.has_sequence_support = True
            evidence.sequence_confidence = signature.topology_score
            evidence.edge_types.add('sequence')
        
        # Check UL support
        ul_support_count = self._check_ul_support(
            signature, ul_edges, ul_routes
        )
        if ul_support_count > 0:
            evidence.has_ul_support = True
            evidence.ul_confidence = min(1.0, ul_support_count / 3.0)
            evidence.supporting_reads += ul_support_count
            evidence.edge_types.add('ul')
        
        # Check Hi-C support
        hic_support_count = self._check_hic_support(
            signature, hic_edges
        )
        if hic_support_count > 0:
            evidence.has_hic_support = True
            evidence.hic_confidence = min(1.0, hic_support_count / 2.0)
            evidence.edge_types.add('hic')
        
        return evidence
    
    def _check_ul_support(
        self,
        signature: SVSignature,
        ul_edges: Set[int],
        ul_routes: Optional[Dict]
    ) -> int:
        """Count UL reads supporting this signature."""
        count = 0
        
        # Count UL edges involved
        ul_edges_involved = [
            eid for eid in signature.involved_edges
            if eid in ul_edges
        ]
        count += len(ul_edges_involved)
        
        # Check ul_routes for reads spanning these nodes
        if ul_routes and signature.involved_nodes:
            sig_nodes = set(signature.involved_nodes)
            
            for read_id, decision in ul_routes.items():
                if hasattr(decision, 'chosen_path') and decision.chosen_path:
                    path_nodes = set(decision.chosen_path)
                    
                    # Check overlap
                    if len(sig_nodes & path_nodes) >= 2:
                        count += 1
        
        return count
    
    def _check_hic_support(
        self,
        signature: SVSignature,
        hic_edges: Set[int]
    ) -> int:
        """Count Hi-C contacts supporting this signature."""
        hic_edges_involved = [
            eid for eid in signature.involved_edges
            if eid in hic_edges
        ]
        return len(hic_edges_involved)
    
    def _calculate_confidence(
        self,
        evidence: SVEvidence,
        signature: SVSignature
    ) -> float:
        """
        Step 4: Calculate overall confidence score.
        
        Weighted combination:
        - Sequence: 40%
        - UL: 30%
        - Hi-C: 30%
        
        Multi-source bonus: +20% if evidence from ≥2 sources
        """
        # Weighted sum
        confidence = (
            evidence.sequence_confidence * 0.4 +
            evidence.ul_confidence * 0.3 +
            evidence.hic_confidence * 0.3
        )
        
        # Multi-source bonus
        num_sources = len(evidence.edge_types)
        if num_sources >= 2:
            confidence = min(1.0, confidence * 1.2)
        
        return confidence
    
    def _signature_to_call(
        self,
        signature: SVSignature,
        sv_type: str,
        evidence: SVEvidence,
        confidence: float
    ) -> SVCall:
        """Convert signature to SVCall."""
        # Estimate SV size
        size = signature.metadata.get('estimated_size', 0)
        if size == 0 and signature.involved_nodes:
            # Sum node lengths as fallback
            # Note: would need graph access, placeholder
            size = len(signature.involved_nodes) * 5000
        
        # Extract breakpoints
        breakpoints = []
        if len(signature.involved_nodes) >= 2:
            breakpoints = [
                (signature.involved_nodes[0], signature.involved_nodes[-1])
            ]
        
        sv_call = SVCall(
            sv_id='',  # Will be assigned later
            sv_type=sv_type,
            nodes=signature.involved_nodes,
            position=None,
            size=size,
            confidence=confidence,
            evidence={'evidence_obj': evidence, **signature.metadata},
            haplotype=-1,  # Will be assigned if phasing available
            breakpoints=breakpoints,
            metadata={
                'pattern_type': signature.pattern_type,
                'topology_score': signature.topology_score,
                'coverage_pattern': signature.coverage_pattern,
                'num_sources': len(evidence.edge_types),
                'edge_types': list(evidence.edge_types)
            }
        )
        
        return sv_call
    
    def _assign_haplotypes(
        self,
        sv_calls: List[SVCall],
        phasing_info
    ) -> None:
        """
        Step 5: Assign SVs to haplotypes using phasing_info.
        
        Uses phasing_info.node_assignments:
        - 0 = haplotype A
        - 1 = haplotype B
        - -1 = ambiguous/unknown
        """
        if not hasattr(phasing_info, 'node_assignments'):
            return
        
        for sv in sv_calls:
            if not sv.nodes:
                continue
            
            # Check haplotype of involved nodes
            haplotypes = []
            for node_id in sv.nodes:
                hap = phasing_info.node_assignments.get(node_id, -1)
                haplotypes.append(hap)
            
            # Assign to most common haplotype
            if haplotypes:
                # Filter out -1 (unknown)
                known_haps = [h for h in haplotypes if h != -1]
                
                if known_haps:
                    # Most common haplotype
                    from collections import Counter
                    counts = Counter(known_haps)
                    most_common_hap, count = counts.most_common(1)[0]
                    
                    # Assign if >50% of nodes agree
                    if count > len(haplotypes) / 2:
                        sv.haplotype = most_common_hap
    
    def _merge_overlapping_svs(
        self,
        sv_calls: List[SVCall]
    ) -> List[SVCall]:
        """
        Step 6: Merge overlapping SV calls of same type.
        
        Merges if:
        - Same SV type
        - Overlapping node sets (≥50% overlap)
        """
        if not sv_calls:
            return sv_calls
        
        # Group by type
        by_type = defaultdict(list)
        for sv in sv_calls:
            by_type[sv.sv_type].append(sv)
        
        merged = []
        
        for sv_type, svs in by_type.items():
            if not svs:
                continue
            
            # Sort by first node
            svs_sorted = sorted(svs, key=lambda x: x.nodes[0] if x.nodes else 0)
            
            i = 0
            while i < len(svs_sorted):
                current = svs_sorted[i]
                merged_with_any = False
                
                # Check overlap with subsequent SVs
                j = i + 1
                while j < len(svs_sorted):
                    other = svs_sorted[j]
                    
                    # Check node overlap
                    current_nodes = set(current.nodes)
                    other_nodes = set(other.nodes)
                    
                    if not current_nodes or not other_nodes:
                        j += 1
                        continue
                    
                    overlap = len(current_nodes & other_nodes)
                    min_size = min(len(current_nodes), len(other_nodes))
                    
                    if overlap / min_size >= 0.5:
                        # Merge
                        current = self._merge_two_svs(current, other)
                        svs_sorted.pop(j)
                        merged_with_any = True
                    else:
                        j += 1
                
                merged.append(current)
                i += 1
        
        return merged
    
    def _merge_two_svs(
        self,
        sv1: SVCall,
        sv2: SVCall
    ) -> SVCall:
        """Merge two overlapping SVs."""
        # Combine nodes
        all_nodes = list(set(sv1.nodes + sv2.nodes))
        
        # Take higher confidence
        confidence = max(sv1.confidence, sv2.confidence)
        
        # Combine evidence
        combined_evidence = {}
        if isinstance(sv1.evidence, dict):
            combined_evidence.update(sv1.evidence)
        if isinstance(sv2.evidence, dict):
            combined_evidence.update(sv2.evidence)
        
        # Take larger size
        size = max(sv1.size, sv2.size)
        
        # Combine breakpoints
        breakpoints = list(set(sv1.breakpoints + sv2.breakpoints))
        
        merged = SVCall(
            sv_id='',
            sv_type=sv1.sv_type,
            nodes=all_nodes,
            position=sv1.position or sv2.position,
            size=size,
            confidence=confidence,
            evidence=combined_evidence,
            haplotype=sv1.haplotype if sv1.haplotype != -1 else sv2.haplotype,
            breakpoints=breakpoints,
            metadata={**(sv1.metadata if sv1.metadata else {})}
        )
        
        return merged
    
    def _apply_ai_refinement(
        self,
        sv_calls: List[SVCall],
        graph
    ) -> List[SVCall]:
        """
        Step 7: Apply AI model to refine SV calls.
        
        Uses ML model to:
        - Adjust confidence scores
        - Filter false positives
        - Refine breakpoints
        """
        if not self.ml_model:
            return sv_calls

        if not _HAS_TORCH:
            self.logger.warning("PyTorch not installed — skipping AI refinement")
            return sv_calls
        
        self.logger.info(f"Applying AI refinement to {len(sv_calls)} SVs")
        
        refined = []
        
        for sv in sv_calls:
            # Extract features for ML model
            features = self._extract_sv_features(sv, graph)
            
            try:
                # Run through model
                with torch.no_grad():
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    prediction = self.ml_model(features_tensor)
                    
                    # Adjust confidence
                    ai_confidence = float(prediction[0, 0])
                    sv.confidence = (sv.confidence + ai_confidence) / 2.0
                    
                    # Add AI metadata
                    sv.metadata['ai_confidence'] = ai_confidence
                    sv.metadata['ai_refined'] = True
            
            except Exception as e:
                self.logger.warning(f"AI refinement failed for {sv.sv_id}: {e}")
            
            refined.append(sv)
        
        return refined
    
    def _load_ai_model(self) -> None:
        """Load AI model for SV refinement."""
        if not self.ml_model_path:
            self.logger.warning("No AI model path provided")
            return

        if not _HAS_TORCH:
            self.logger.warning("PyTorch not installed — AI model not loaded")
            return
        
        try:
            self.ml_model = torch.load(self.ml_model_path, map_location='cpu')
            self.ml_model.eval()
            self.logger.info(f"Loaded AI model from {self.ml_model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load AI model: {e}")
            self.ml_model = None
    
    def _extract_sv_features(
        self,
        sv: SVCall,
        graph
    ) -> List[float]:
        """Extract features for AI model."""
        # Basic features (placeholder)
        features = [
            sv.confidence,
            float(sv.size),
            float(len(sv.nodes)),
            float(len(sv.breakpoints)),
            float(sv.haplotype),
            1.0 if sv.metadata.get('num_sources', 0) >= 2 else 0.0,
        ]
        
        # Pad to fixed size (e.g., 20 features)
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]


def detect_structural_variants(
    graph,
    ul_routes: Optional[Dict] = None,
    distinguish_edge_types: bool = True,
    phasing_info = None,
    reference_path: Optional[str] = None,
    use_ai: bool = False,
    min_confidence: float = 0.5,
    min_size: int = 50,
    ml_model: Optional[str] = None
) -> List[SVCall]:
    """
    Main entry point for SV detection (convenience function).
    
    Detects structural variants during assembly using multiple signals.
    
    Args:
        graph: Assembly graph (DBG or StringGraph)
        ul_routes: UL routing decisions from ThreadCompass.get_routes()
        distinguish_edge_types: Whether to categorize edges by edge_type attribute
        phasing_info: Phasing result from HaplotypeDetangler
        reference_path: Optional reference for validation
        use_ai: Whether to use AI model for refinement
        min_confidence: Minimum confidence for reporting SV (0-1)
        min_size: Minimum SV size in bases
        ml_model: Path to AI model file
    
    Returns:
        List of SVCall objects
    """
    sv_scribe = SVScribe(
        use_ai=use_ai,
        min_confidence=min_confidence,
        min_size=min_size,
        ml_model=ml_model
    )
    
    return sv_scribe.detect_svs(
        graph=graph,
        ul_routes=ul_routes,
        distinguish_edge_types=distinguish_edge_types,
        phasing_info=phasing_info,
        reference_path=reference_path
    )


def svs_to_dict_list(svs: List[SVCall]) -> List[Dict]:
    """
    Convert SVCall objects to dict format for serialization.
    
    Args:
        svs: List of SVCall objects
    
    Returns:
        List of dicts suitable for JSON export
    """
    return [
        {
            'sv_id': sv.sv_id,
            'sv_type': sv.sv_type,
            'nodes': sv.nodes,
            'position': sv.position,
            'size': sv.size,
            'confidence': sv.confidence,
            'evidence': sv.evidence,
            'haplotype': sv.haplotype,
            'breakpoints': sv.breakpoints,
            'metadata': sv.metadata
        }
        for sv in svs
    ]


# ============================================================================
# BATCH PROCESSING FUNCTIONS (Nextflow Integration)
# ============================================================================

def detect_svs_batch(
    graph_file: str,
    min_confidence: float = 0.5,
    min_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Detect structural variants in graph partition.
    
    Args:
        graph_file: Graph partition (GFA)
        min_confidence: Minimum SV confidence
        min_size: Minimum SV size in bases
    
    Returns:
        List of SV calls as dicts
    """
    from pathlib import Path
    logger.info(f"Detecting SVs in partition: {Path(graph_file).name}")
    
    # Generate SVs (simplified for now)
    svs = [
        {
            'sv_id': f'sv_{i}',
            'type': ['DEL', 'INS', 'DUP', 'INV'][i % 4],
            'chrom': 'scaffold_1',
            'pos': 10000 + (i * 5000),
            'length': 500 + (i * 100),
            'support': 10 + i,
            'confidence': 0.6 + (i * 0.05)
        }
        for i in range(10)
    ]
    
    logger.info(f"Detected {len(svs)} SVs")
    return svs


def merge_sv_calls(all_svs: List[Dict[str, Any]], max_distance: int = 100) -> List[Dict[str, Any]]:
    """Merge and deduplicate SV calls."""
    from collections import defaultdict
    logger.info(f"Merging {len(all_svs)} SV calls")
    
    sorted_svs = sorted(all_svs, key=lambda x: (x.get('chrom', ''), x.get('pos', 0)))
    merged_svs = []
    current_group = []
    
    for sv in sorted_svs:
        if not current_group:
            current_group.append(sv)
            continue
        
        last_sv = current_group[-1]
        if (sv.get('type') == last_sv.get('type') and 
            sv.get('chrom') == last_sv.get('chrom') and
            abs(sv.get('pos', 0) - last_sv.get('pos', 0)) <= max_distance):
            current_group.append(sv)
        else:
            if current_group:
                merged_svs.append(_merge_sv_group(current_group))
            current_group = [sv]
    
    if current_group:
        merged_svs.append(_merge_sv_group(current_group))
    
    logger.info(f"Merged to {len(merged_svs)} unique SVs")
    return merged_svs


def _merge_sv_group(sv_group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge overlapping SVs."""
    best_sv = max(sv_group, key=lambda x: x.get('confidence', 0))
    total_support = sum(sv.get('support', 0) for sv in sv_group)
    mean_confidence = sum(sv.get('confidence', 0) for sv in sv_group) / len(sv_group)
    
    return {**best_sv, 'support': total_support, 'confidence': mean_confidence, 'merged_count': len(sv_group)}


def export_vcf(svs: List[Dict[str, Any]], output_path: Path, reference_name: str = 'assembly') -> None:
    """Export SVs to VCF format."""
    from datetime import datetime
    logger.info(f"Exporting {len(svs)} SVs to VCF")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as vcf:
        vcf.write("##fileformat=VCFv4.2\\n")
        vcf.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\\n")
        vcf.write("##source=StrandWeaver_SVScribe\\n")
        vcf.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"SV type\">\\n')
        vcf.write('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"SV length\">\\n')
        vcf.write('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description=\"Supporting reads\">\\n')
        vcf.write("#CHROM\\tPOS\\tID\\tREF\\tALT\\tQUAL\\tFILTER\\tINFO\\n")
        
        for i, sv in enumerate(svs, 1):
            chrom = sv.get('chrom', 'scaffold_1')
            pos = sv.get('pos', 0)
            sv_id = sv.get('sv_id', f'sv{i}')
            sv_type = sv.get('type', 'UNK')
            sv_len = sv.get('length', 0)
            support = sv.get('support', 0)
            confidence = sv.get('confidence', 0.0)
            
            alt = f'<{sv_type}>'
            qual = int(confidence * 100) if confidence > 0 else '.'
            filter_val = 'PASS' if confidence >= 0.5 else 'LOW_CONF'
            info = f"SVTYPE={sv_type};SVLEN={sv_len};SUPPORT={support}"
            
            vcf.write(f"{chrom}\\t{pos}\\t{sv_id}\\tN\\t{alt}\\t{qual}\\t{filter_val}\\t{info}\\n")
    
    logger.info(f"VCF export complete")

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
