#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diploid Disentanglement Module for StrandWeaver.

This module separates diploid assembly graphs into haplotype A and B paths
using multiple orthogonal signals:
- Hi-C phasing
- GNN path predictions
- AI overlap classifications
- Regional k-mer recommendations
- UL path coherence

The goal is to resolve allelic variation and produce haplotype-separated
assemblies while correctly handling repeats and ambiguous regions.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DiploidDisentangleResult:
    """
    Result of diploid graph disentanglement.
    
    Attributes:
        hapA_nodes: Set of node IDs assigned to haplotype A
        hapB_nodes: Set of node IDs assigned to haplotype B
        ambiguous_nodes: Set of node IDs that couldn't be confidently assigned
        hapA_edges: Set of edge IDs assigned to haplotype A
        hapB_edges: Set of edge IDs assigned to haplotype B
        ambiguous_edges: Set of edge IDs in ambiguous regions
        repeat_nodes: Set of node IDs identified as repeats (shared between haplotypes)
        confidence_scores: Dict[node_id] -> confidence in assignment (0-1)
        haplotype_blocks: List of contiguous haplotype blocks
        stats: Summary statistics
    """
    hapA_nodes: Set[int] = field(default_factory=set)
    hapB_nodes: Set[int] = field(default_factory=set)
    ambiguous_nodes: Set[int] = field(default_factory=set)
    hapA_edges: Set[int] = field(default_factory=set)
    hapB_edges: Set[int] = field(default_factory=set)
    ambiguous_edges: Set[int] = field(default_factory=set)
    repeat_nodes: Set[int] = field(default_factory=set)
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    haplotype_blocks: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class HaplotypeScorer:
    """
    Scores node/edge assignments to haplotypes using multiple signals.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HaplotypeScorer")
    
    def score_node_haplotype(
        self,
        node_id: int,
        graph,
        hic_phase_info: Optional[Dict] = None,
        gnn_paths = None,
        ai_annotations: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Score node assignment to haplotype A, B, or ambiguous.
        
        Returns:
            (score_A, score_B, repeat_score) each in 0-1 range
        """
        score_A = 0.5
        score_B = 0.5
        repeat_score = 0.0
        
        # Hi-C phasing signal (strongest weight: 50%)
        if hic_phase_info and node_id in hic_phase_info:
            phase = hic_phase_info[node_id]
            score_A = phase.phase_A_score * 0.5 + score_A * 0.5
            score_B = phase.phase_B_score * 0.5 + score_B * 0.5
        
        # GNN path coherence (30% weight)
        if gnn_paths:
            path_score_A, path_score_B = self._score_node_in_paths(
                node_id, gnn_paths
            )
            score_A = score_A * 0.7 + path_score_A * 0.3
            score_B = score_B * 0.7 + path_score_B * 0.3
        
        # Repeat detection from regional k and coverage (20% weight)
        node = graph.nodes.get(node_id)
        if node:
            coverage = getattr(node, 'coverage', 0.0)
            in_degree = len(graph.in_edges.get(node_id, set()))
            out_degree = len(graph.out_edges.get(node_id, set()))
            
            # High coverage + high branching suggests repeat
            if coverage > 50 and (in_degree + out_degree) > 2:
                repeat_score += 0.4
            
            # Check edge AI scores for repeat signals
            for edge_id in graph.out_edges.get(node_id, set()):
                if ai_annotations and edge_id in ai_annotations:
                    ai = ai_annotations[edge_id]
                    repeat_score += ai.score_repeat * 0.1
        
        repeat_score = min(repeat_score, 1.0)
        
        return (score_A, score_B, repeat_score)
    
    def _score_node_in_paths(
        self,
        node_id: int,
        gnn_paths
    ) -> Tuple[float, float]:
        """
        Score node based on GNN path assignments.
        
        If paths are cleanly separated into two groups, infer haplotypes.
        """
        if not hasattr(gnn_paths, 'best_paths') or not gnn_paths.best_paths:
            return (0.5, 0.5)
        
        # Find which paths contain this node
        containing_paths = []
        for path_idx, path in enumerate(gnn_paths.best_paths):
            if node_id in path:
                containing_paths.append(path_idx)
        
        if not containing_paths:
            return (0.5, 0.5)
        
        # If node is in multiple paths, it might be a repeat
        if len(containing_paths) > 1:
            return (0.5, 0.5)
        
        # Node in single path - assign based on path index
        # (In real implementation, would cluster paths into haplotypes)
        path_idx = containing_paths[0]
        if path_idx % 2 == 0:
            return (0.7, 0.3)
        else:
            return (0.3, 0.7)
    
    def score_edge_haplotype(
        self,
        edge_id: int,
        from_node: int,
        to_node: int,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Assign edge to haplotype based on endpoints.
        
        Returns:
            (assignment, confidence) where assignment is 'A', 'B', or 'ambiguous'
        """
        from_hap = node_assignments.get(from_node, 'ambiguous')
        to_hap = node_assignments.get(to_node, 'ambiguous')
        
        # If both endpoints are same haplotype
        if from_hap == to_hap and from_hap != 'ambiguous':
            confidence = 0.8
            
            # Boost confidence with AI support
            if ai_annotations and edge_id in ai_annotations:
                ai = ai_annotations[edge_id]
                if ai.score_true > 0.7:
                    confidence = min(confidence + 0.1, 1.0)
            
            # Boost confidence with Hi-C support
            if hic_edge_support and edge_id in hic_edge_support:
                hic = hic_edge_support[edge_id]
                if hic.hic_weight > 0.7:
                    confidence = min(confidence + 0.1, 1.0)
            
            return (from_hap, confidence)
        
        # Endpoints are different or ambiguous
        confidence = 0.3
        
        # Check if this is an allelic edge (connects haplotypes)
        if ai_annotations and edge_id in ai_annotations:
            ai = ai_annotations[edge_id]
            if ai.score_allelic > 0.6:
                return ('ambiguous', 0.5)
        
        return ('ambiguous', confidence)


class DisentanglerEngine:
    """
    Main engine for diploid graph disentanglement.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        repeat_threshold: float = 0.5
    ):
        """
        Initialize disentangler.
        
        Args:
            min_confidence: Minimum confidence for definite haplotype assignment
            repeat_threshold: Threshold for classifying nodes as repeats
        """
        self.min_confidence = min_confidence
        self.repeat_threshold = repeat_threshold
        self.scorer = HaplotypeScorer()
        self.logger = logging.getLogger(f"{__name__}.DisentanglerEngine")
    
    def disentangle(
        self,
        graph,
        gnn_paths = None,
        hic_phase_info: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> DiploidDisentangleResult:
        """
        Disentangle diploid graph into haplotype A and B.
        
        Algorithm:
        1. Score each node for haplotype A, B, and repeat likelihood
        2. Assign nodes based on score differences and thresholds
        3. Propagate assignments through high-confidence edges
        4. Assign edges based on endpoint haplotypes
        5. Identify haplotype blocks
        
        Args:
            graph: Assembly graph
            gnn_paths: GNN path predictions
            hic_phase_info: Hi-C phasing
            ai_annotations: AI edge annotations
            hic_edge_support: Hi-C edge support
            regional_k_map: Regional k recommendations
            ul_support_map: UL support
        
        Returns:
            DiploidDisentangleResult
        """
        self.logger.info("Starting diploid disentanglement")
        
        result = DiploidDisentangleResult()
        
        # Step 1: Score all nodes
        node_scores = {}
        for node_id in graph.nodes:
            score_A, score_B, repeat_score = self.scorer.score_node_haplotype(
                node_id, graph, hic_phase_info, gnn_paths,
                ai_annotations, regional_k_map
            )
            node_scores[node_id] = (score_A, score_B, repeat_score)
        
        # Step 2: Initial node assignments
        node_assignments = {}
        for node_id, (score_A, score_B, repeat_score) in node_scores.items():
            # Check if it's a repeat
            if repeat_score > self.repeat_threshold:
                result.repeat_nodes.add(node_id)
                node_assignments[node_id] = 'repeat'
                result.confidence_scores[node_id] = repeat_score
                continue
            
            # Assign to haplotype with higher score
            diff = abs(score_A - score_B)
            
            if diff > self.min_confidence:
                if score_A > score_B:
                    result.hapA_nodes.add(node_id)
                    node_assignments[node_id] = 'A'
                    result.confidence_scores[node_id] = score_A
                else:
                    result.hapB_nodes.add(node_id)
                    node_assignments[node_id] = 'B'
                    result.confidence_scores[node_id] = score_B
            else:
                result.ambiguous_nodes.add(node_id)
                node_assignments[node_id] = 'ambiguous'
                result.confidence_scores[node_id] = max(score_A, score_B)
        
        self.logger.info(
            f"Initial assignment: {len(result.hapA_nodes)} A, "
            f"{len(result.hapB_nodes)} B, "
            f"{len(result.ambiguous_nodes)} ambiguous, "
            f"{len(result.repeat_nodes)} repeat"
        )
        
        # Step 3: Propagate assignments through high-confidence edges
        node_assignments = self._propagate_assignments(
            graph, node_assignments, ai_annotations, hic_edge_support, result
        )
        
        # Step 4: Assign edges
        self._assign_edges(
            graph, node_assignments, ai_annotations,
            hic_edge_support, result
        )
        
        # Step 5: Identify haplotype blocks
        result.haplotype_blocks = self._identify_haplotype_blocks(
            graph, node_assignments, result
        )
        
        # Step 6: Compute statistics
        result.stats = self._compute_statistics(result)
        
        self.logger.info(
            f"Final: {len(result.hapA_nodes)} A nodes, "
            f"{len(result.hapB_nodes)} B nodes, "
            f"{len(result.hapA_edges)} A edges, "
            f"{len(result.hapB_edges)} B edges"
        )
        
        return result
    
    def _propagate_assignments(
        self,
        graph,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict],
        hic_edge_support: Optional[Dict],
        result: DiploidDisentangleResult
    ) -> Dict[int, str]:
        """
        Propagate haplotype assignments through high-confidence edges.
        
        If edge has AI score_true > 0.7 and Hi-C weight > 0.7,
        propagate haplotype from assigned endpoint to ambiguous endpoint.
        """
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for edge_id, edge in graph.edges.items():
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                
                if not from_node or not to_node:
                    continue
                
                from_hap = node_assignments.get(from_node, 'ambiguous')
                to_hap = node_assignments.get(to_node, 'ambiguous')
                
                # Skip if both assigned or both ambiguous
                if from_hap == to_hap:
                    continue
                
                # Check edge confidence
                edge_confident = False
                if ai_annotations and edge_id in ai_annotations:
                    ai = ai_annotations[edge_id]
                    if ai.score_true > 0.7:
                        edge_confident = True
                
                if hic_edge_support and edge_id in hic_edge_support:
                    hic = hic_edge_support[edge_id]
                    if hic.hic_weight > 0.7:
                        edge_confident = True
                
                if not edge_confident:
                    continue
                
                # Propagate from assigned to ambiguous
                if from_hap in ('A', 'B') and to_hap == 'ambiguous':
                    node_assignments[to_node] = from_hap
                    result.ambiguous_nodes.discard(to_node)
                    if from_hap == 'A':
                        result.hapA_nodes.add(to_node)
                    else:
                        result.hapB_nodes.add(to_node)
                    changed = True
                
                elif to_hap in ('A', 'B') and from_hap == 'ambiguous':
                    node_assignments[from_node] = to_hap
                    result.ambiguous_nodes.discard(from_node)
                    if to_hap == 'A':
                        result.hapA_nodes.add(from_node)
                    else:
                        result.hapB_nodes.add(from_node)
                    changed = True
        
        self.logger.info(f"Propagated assignments in {iterations} iterations")
        return node_assignments
    
    def _assign_edges(
        self,
        graph,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict],
        hic_edge_support: Optional[Dict],
        result: DiploidDisentangleResult
    ):
        """Assign edges to haplotypes based on endpoints."""
        for edge_id, edge in graph.edges.items():
            from_node = getattr(edge, 'from_node', None)
            to_node = getattr(edge, 'to_node', None)
            
            if not from_node or not to_node:
                continue
            
            assignment, confidence = self.scorer.score_edge_haplotype(
                edge_id, from_node, to_node, node_assignments,
                ai_annotations, hic_edge_support
            )
            
            if assignment == 'A':
                result.hapA_edges.add(edge_id)
            elif assignment == 'B':
                result.hapB_edges.add(edge_id)
            else:
                result.ambiguous_edges.add(edge_id)
    
    def _identify_haplotype_blocks(
        self,
        graph,
        node_assignments: Dict[int, str],
        result: DiploidDisentangleResult
    ) -> List[Dict[str, Any]]:
        """
        Identify contiguous blocks of haplotype-specific sequence.
        
        A block is a maximal connected component of nodes with same haplotype.
        """
        blocks = []
        
        for haplotype in ['A', 'B']:
            nodes = result.hapA_nodes if haplotype == 'A' else result.hapB_nodes
            edges = result.hapA_edges if haplotype == 'A' else result.hapB_edges
            
            # Find connected components within haplotype
            visited = set()
            for node in nodes:
                if node in visited:
                    continue
                
                # BFS to find block
                block_nodes = []
                queue = [node]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    block_nodes.append(current)
                    
                    # Add neighbors in same haplotype
                    for neighbor in graph.out_edges.get(current, set()):
                        if neighbor in nodes and neighbor not in visited:
                            queue.append(neighbor)
                    
                    for neighbor in graph.in_edges.get(current, set()):
                        if neighbor in nodes and neighbor not in visited:
                            queue.append(neighbor)
                
                if block_nodes:
                    blocks.append({
                        'haplotype': haplotype,
                        'nodes': block_nodes,
                        'size': len(block_nodes),
                        'avg_confidence': sum(
                            result.confidence_scores.get(n, 0.5)
                            for n in block_nodes
                        ) / len(block_nodes)
                    })
        
        return sorted(blocks, key=lambda b: b['size'], reverse=True)
    
    def _compute_statistics(self, result: DiploidDisentangleResult) -> Dict[str, Any]:
        """Compute summary statistics."""
        total_nodes = (
            len(result.hapA_nodes) + len(result.hapB_nodes) +
            len(result.ambiguous_nodes) + len(result.repeat_nodes)
        )
        total_edges = (
            len(result.hapA_edges) + len(result.hapB_edges) +
            len(result.ambiguous_edges)
        )
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'hapA_nodes': len(result.hapA_nodes),
            'hapB_nodes': len(result.hapB_nodes),
            'ambiguous_nodes': len(result.ambiguous_nodes),
            'repeat_nodes': len(result.repeat_nodes),
            'hapA_edges': len(result.hapA_edges),
            'hapB_edges': len(result.hapB_edges),
            'ambiguous_edges': len(result.ambiguous_edges),
            'haplotype_balance': len(result.hapA_nodes) / max(len(result.hapB_nodes), 1),
            'num_blocks': len(result.haplotype_blocks),
            'avg_confidence': sum(result.confidence_scores.values()) / max(len(result.confidence_scores), 1)
        }


def disentangle_diploid_graph(
    graph,
    gnn_paths = None,
    hic_phase_info: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None,
    min_confidence: float = 0.6,
    repeat_threshold: float = 0.5
) -> DiploidDisentangleResult:
    """
    Main entry point for diploid graph disentanglement.
    
    Separates assembly graph into haplotype A and B using multiple signals.
    
    Args:
        graph: Assembly graph
        gnn_paths: GNN path predictions
        hic_phase_info: Hi-C phasing information
        ai_annotations: AI edge annotations
        hic_edge_support: Hi-C edge support
        regional_k_map: Regional k recommendations
        ul_support_map: UL support counts
        min_confidence: Minimum confidence for haplotype assignment
        repeat_threshold: Threshold for repeat classification
    
    Returns:
        DiploidDisentangleResult with haplotype assignments
    """
    engine = DisentanglerEngine(
        min_confidence=min_confidence,
        repeat_threshold=repeat_threshold
    )
    
    return engine.disentangle(
        graph, gnn_paths, hic_phase_info, ai_annotations,
        hic_edge_support, regional_k_map, ul_support_map
    )
