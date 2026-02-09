#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Graph Cleanup — multi-signal edge pruning, bubble resolution, and
phasing-aware graph partitioning into haplotypes A/B.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleanedGraphResult:
    """
    Result of graph cleanup and phasing.
    
    Attributes:
        cleaned_graph: The cleaned graph (DBG or StringGraph)
        haplotype_assignments: Dict[node_id] -> "A", "B", or "ambiguous"
        removed_edges: Set of edge IDs that were pruned
        pruned_branches: Dict[node_id] -> List of removed neighbor nodes
        bubble_resolutions: List of resolved bubble structures
        stats: Cleanup statistics
    """
    cleaned_graph: Any  # DBGGraph or StringGraph
    haplotype_assignments: Dict[int, str] = field(default_factory=dict)
    removed_edges: Set[int] = field(default_factory=set)
    pruned_branches: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    bubble_resolutions: List[Dict] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bubble:
    """
    Represents a bubble structure in the graph.
    
    A bubble is two parallel paths between the same source and sink nodes,
    typically representing allelic variation or errors.
    
    Attributes:
        source_node: Starting node
        sink_node: Ending node
        path_A: List of nodes in first path
        path_B: List of nodes in second path
        support_A: Support score for path A
        support_B: Support score for path B
    """
    source_node: int
    sink_node: int
    path_A: List[int]
    path_B: List[int]
    support_A: float = 0.0
    support_B: float = 0.0


class GraphCleanupEngine:
    """
    Unified graph cleanup engine using multiple signals.
    
    Integrates:
    - Hi-C phasing and support
    - AI edge quality scores
    - Regional k-mer values
    - UL support
    
    Performs:
    - Edge pruning
    - Bubble resolution
    - Haplotype-aware partitioning
    """
    
    def __init__(
        self,
        min_edge_confidence: float = 0.3,
        min_hic_weight: float = 0.2,
        bubble_similarity_threshold: float = 0.95
    ):
        """
        Initialize cleanup engine.
        
        Args:
            min_edge_confidence: Minimum AI confidence to keep edge
            min_hic_weight: Minimum Hi-C weight to keep edge
            bubble_similarity_threshold: Similarity threshold for bubble detection
        """
        self.min_edge_confidence = min_edge_confidence
        self.min_hic_weight = min_hic_weight
        self.bubble_similarity_threshold = bubble_similarity_threshold
        self.logger = logging.getLogger(f"{__name__}.GraphCleanupEngine")
    
    def clean_graph(
        self,
        graph,  # DBGGraph or StringGraph
        hic_phase_info: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None,
        regional_k_map: Optional[Dict[int, int]] = None,
        ul_support_map: Optional[Dict[int, int]] = None
    ) -> CleanedGraphResult:
        """
        Perform comprehensive graph cleanup.
        
        Args:
            graph: DBGGraph or StringGraph to clean
            hic_phase_info: Dict[node_id] -> HiCNodePhaseInfo
            hic_edge_support: Dict[edge_id] -> HiCEdgeSupport
            ai_annotations: Dict[edge_id] -> EdgeAIAnnotation
            regional_k_map: Dict[node_id] -> recommended_k
            ul_support_map: Dict[edge_id] -> ul_support_count
        
        Returns:
            CleanedGraphResult with cleaned graph and metadata
        """
        self.logger.info("Starting graph cleanup")
        
        result = CleanedGraphResult(
            cleaned_graph=graph,
            stats={
                'initial_nodes': len(graph.nodes),
                'initial_edges': len(graph.edges)
            }
        )
        
        # Step 1: Edge pruning
        self.logger.info("Step 1: Pruning low-confidence edges")
        removed_edges = self._prune_low_confidence_edges(
            graph, hic_edge_support, ai_annotations
        )
        result.removed_edges.update(removed_edges)
        
        # Step 2: Bubble resolution
        self.logger.info("Step 2: Resolving bubbles")
        bubble_info = self._resolve_bubbles(
            graph, hic_phase_info, ai_annotations, ul_support_map
        )
        result.bubble_resolutions = bubble_info['resolutions']
        result.removed_edges.update(bubble_info['removed_edges'])
        
        # Step 3: Haplotype assignment
        self.logger.info("Step 3: Assigning haplotypes")
        result.haplotype_assignments = self._assign_haplotypes(
            graph, hic_phase_info, ai_annotations
        )
        
        # Step 4: Update graph statistics
        result.stats['final_nodes'] = len(graph.nodes)
        result.stats['final_edges'] = len(graph.edges) - len(result.removed_edges)
        result.stats['removed_edges'] = len(result.removed_edges)
        result.stats['bubbles_resolved'] = len(result.bubble_resolutions)
        
        phase_A = sum(1 for p in result.haplotype_assignments.values() if p == "A")
        phase_B = sum(1 for p in result.haplotype_assignments.values() if p == "B")
        ambiguous = sum(1 for p in result.haplotype_assignments.values() if p == "ambiguous")
        
        result.stats['phase_A_nodes'] = phase_A
        result.stats['phase_B_nodes'] = phase_B
        result.stats['ambiguous_nodes'] = ambiguous
        
        self.logger.info(
            f"Cleanup complete: {result.stats['removed_edges']} edges removed, "
            f"{result.stats['bubbles_resolved']} bubbles resolved, "
            f"{phase_A} phase A, {phase_B} phase B, {ambiguous} ambiguous"
        )
        
        return result
    
    def _prune_low_confidence_edges(
        self,
        graph,
        hic_edge_support: Optional[Dict],
        ai_annotations: Optional[Dict]
    ) -> Set[int]:
        """
        Remove edges with low confidence from multiple signals.
        
        Removes edges that have:
        - Low AI score AND low Hi-C support, OR
        - High trans Hi-C contacts (contradictory), OR
        - Very low AI confidence
        """
        removed = set()
        
        for edge_id, edge in list(graph.edges.items()):
            should_remove = False
            
            # Get AI annotation
            ai_score = 0.5  # Default neutral
            if ai_annotations and edge_id in ai_annotations:
                ai = ai_annotations[edge_id]
                ai_score = ai.score_true
                
                # Remove if very low AI confidence
                if ai.confidence < 0.2:
                    should_remove = True
                    self.logger.debug(f"Removing edge {edge_id}: very low AI confidence")
            
            # Get Hi-C support
            hic_weight = 0.5  # Default neutral
            if hic_edge_support and edge_id in hic_edge_support:
                hic = hic_edge_support[edge_id]
                hic_weight = hic.hic_weight
                
                # Remove if strong trans contacts
                if hic.trans_contacts > hic.cis_contacts * 2:
                    should_remove = True
                    self.logger.debug(f"Removing edge {edge_id}: high trans contacts")
            
            # Remove if both signals are low
            if ai_score < self.min_edge_confidence and hic_weight < self.min_hic_weight:
                should_remove = True
                self.logger.debug(
                    f"Removing edge {edge_id}: low AI ({ai_score:.2f}) "
                    f"and low Hi-C ({hic_weight:.2f})"
                )
            
            if should_remove:
                removed.add(edge_id)
        
        self.logger.info(f"Pruned {len(removed)} low-confidence edges")
        return removed
    
    def _resolve_bubbles(
        self,
        graph,
        hic_phase_info: Optional[Dict],
        ai_annotations: Optional[Dict],
        ul_support_map: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Detect and resolve bubble structures.
        
        Bubbles represent:
        - Allelic variation (keep both if phased differently)
        - Sequencing errors (keep higher-confidence path)
        - Assembly artifacts (remove lower-quality path)
        """
        bubbles = self._detect_bubbles(graph)
        
        resolutions = []
        removed_edges = set()
        
        for bubble in bubbles:
            resolution = self._resolve_single_bubble(
                bubble, graph, hic_phase_info, ai_annotations, ul_support_map
            )
            
            resolutions.append(resolution)
            if resolution['action'] == 'remove_path':
                removed_edges.update(resolution['removed_edges'])
        
        self.logger.info(f"Resolved {len(bubbles)} bubbles")
        
        return {
            'resolutions': resolutions,
            'removed_edges': removed_edges
        }
    
    def _detect_bubbles(self, graph) -> List[Bubble]:
        """
        Detect bubble structures in the graph.
        
        A bubble is two diverging paths from a source node that reconverge
        at a sink node.
        """
        bubbles = []
        visited_pairs = set()
        
        for node_id in graph.nodes:
            out_neighbors = graph.out_edges.get(node_id, set())
            
            # Check if this node has 2+ outgoing edges
            if len(out_neighbors) >= 2:
                out_list = list(out_neighbors)
                
                # Check pairs of outgoing neighbors
                for i in range(len(out_list)):
                    for j in range(i + 1, len(out_list)):
                        neighbor1 = list(graph.edges.values())[list(graph.edges.keys()).index(out_list[i])].to_node if hasattr(list(graph.edges.values())[0], 'to_node') else out_list[i]
                        neighbor2 = list(graph.edges.values())[list(graph.edges.keys()).index(out_list[j])].to_node if hasattr(list(graph.edges.values())[0], 'to_node') else out_list[j]
                        
                        # Check if paths reconverge
                        reconverge = self._find_reconvergence(
                            graph, neighbor1, neighbor2, max_depth=5
                        )
                        
                        if reconverge:
                            pair_key = tuple(sorted([neighbor1, neighbor2]))
                            if pair_key not in visited_pairs:
                                visited_pairs.add(pair_key)
                                
                                bubble = Bubble(
                                    source_node=node_id,
                                    sink_node=reconverge,
                                    path_A=[neighbor1],
                                    path_B=[neighbor2]
                                )
                                bubbles.append(bubble)
        
        return bubbles
    
    def _find_reconvergence(
        self,
        graph,
        node1: int,
        node2: int,
        max_depth: int = 5
    ) -> Optional[int]:
        """
        Check if two diverging paths reconverge within max_depth.
        
        Returns the reconvergence node ID, or None.
        """
        # Simple BFS from both nodes
        visited1 = {node1}
        visited2 = {node2}
        frontier1 = [node1]
        frontier2 = [node2]
        
        for _ in range(max_depth):
            # Expand frontier1
            new_frontier1 = []
            for node in frontier1:
                for edge_id in graph.out_edges.get(node, set()):
                    edge = graph.edges.get(edge_id)
                    if edge:
                        next_node = getattr(edge, 'to_node', None)
                        if next_node and next_node not in visited1:
                            visited1.add(next_node)
                            new_frontier1.append(next_node)
            
            # Expand frontier2
            new_frontier2 = []
            for node in frontier2:
                for edge_id in graph.out_edges.get(node, set()):
                    edge = graph.edges.get(edge_id)
                    if edge:
                        next_node = getattr(edge, 'to_node', None)
                        if next_node and next_node not in visited2:
                            visited2.add(next_node)
                            new_frontier2.append(next_node)
            
            # Check for intersection
            intersection = visited1 & visited2
            if intersection:
                return list(intersection)[0]
            
            frontier1 = new_frontier1
            frontier2 = new_frontier2
            
            if not frontier1 and not frontier2:
                break
        
        return None
    
    def _resolve_single_bubble(
        self,
        bubble: Bubble,
        graph,
        hic_phase_info: Optional[Dict],
        ai_annotations: Optional[Dict],
        ul_support_map: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Resolve a single bubble by choosing path or keeping both.
        
        Decision logic:
        1. If paths have different Hi-C phases AND good scores → keep both (allelic)
        2. If one path has much better support → remove other (error)
        3. If similar support → keep both (ambiguous)
        """
        # Calculate support for each path
        support_A = self._calculate_path_support(
            bubble.path_A, graph, hic_phase_info, ai_annotations, ul_support_map
        )
        support_B = self._calculate_path_support(
            bubble.path_B, graph, hic_phase_info, ai_annotations, ul_support_map
        )
        
        bubble.support_A = support_A['total']
        bubble.support_B = support_B['total']
        
        # Check phasing
        phase_A = support_A.get('phase', 'ambiguous')
        phase_B = support_B.get('phase', 'ambiguous')
        
        resolution = {
            'bubble': bubble,
            'support_A': support_A,
            'support_B': support_B,
            'phase_A': phase_A,
            'phase_B': phase_B,
            'action': 'keep_both',
            'removed_edges': set()
        }
        
        # Decision logic
        if phase_A != 'ambiguous' and phase_B != 'ambiguous' and phase_A != phase_B:
            # Different haplotypes - keep both
            resolution['action'] = 'keep_both'
            resolution['reason'] = 'different_haplotypes'
        
        elif bubble.support_A > bubble.support_B * 2:
            # Path A much better - remove B
            resolution['action'] = 'remove_path'
            resolution['removed_path'] = 'B'
            resolution['reason'] = 'low_support'
            # Would need to identify and remove edges in path B
        
        elif bubble.support_B > bubble.support_A * 2:
            # Path B much better - remove A
            resolution['action'] = 'remove_path'
            resolution['removed_path'] = 'A'
            resolution['reason'] = 'low_support'
            # Would need to identify and remove edges in path A
        
        else:
            # Similar support - keep both
            resolution['action'] = 'keep_both'
            resolution['reason'] = 'similar_support'
        
        return resolution
    
    def _calculate_path_support(
        self,
        path: List[int],
        graph,
        hic_phase_info: Optional[Dict],
        ai_annotations: Optional[Dict],
        ul_support_map: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate total support for a path through the graph.
        
        Combines:
        - AI edge scores
        - Hi-C support
        - UL support
        - Phase consistency
        """
        total_support = 0.0
        ai_scores = []
        hic_weights = []
        ul_counts = []
        phases = []
        
        for node_id in path:
            # Get phase
            if hic_phase_info and node_id in hic_phase_info:
                phases.append(hic_phase_info[node_id].phase_assignment)
            
            # Get edge scores (edges leading to this node)
            for edge_id in graph.in_edges.get(node_id, set()):
                if ai_annotations and edge_id in ai_annotations:
                    ai_scores.append(ai_annotations[edge_id].score_true)
                
                if ul_support_map and edge_id in ul_support_map:
                    ul_counts.append(ul_support_map[edge_id])
        
        # Aggregate
        avg_ai = sum(ai_scores) / len(ai_scores) if ai_scores else 0.5
        avg_ul = sum(ul_counts) / len(ul_counts) if ul_counts else 0
        
        # Determine consensus phase
        if phases:
            phase_counts = {'A': phases.count('A'), 'B': phases.count('B')}
            consensus_phase = 'A' if phase_counts['A'] > phase_counts['B'] else 'B' if phase_counts['B'] > 0 else 'ambiguous'
        else:
            consensus_phase = 'ambiguous'
        
        total_support = avg_ai * 0.6 + min(avg_ul / 5.0, 1.0) * 0.4
        
        return {
            'total': total_support,
            'ai_avg': avg_ai,
            'ul_avg': avg_ul,
            'phase': consensus_phase
        }
    
    def _assign_haplotypes(
        self,
        graph,
        hic_phase_info: Optional[Dict],
        ai_annotations: Optional[Dict]
    ) -> Dict[int, str]:
        """
        Assign final haplotype labels to all nodes.
        
        Uses:
        - Hi-C phasing as primary signal
        - AI edge quality for propagation
        - Graph connectivity for consistency
        """
        assignments = {}
        
        # Start with Hi-C assignments
        if hic_phase_info:
            for node_id in graph.nodes:
                if node_id in hic_phase_info:
                    assignments[node_id] = hic_phase_info[node_id].phase_assignment
                else:
                    assignments[node_id] = 'ambiguous'
        else:
            # No Hi-C data - all ambiguous
            for node_id in graph.nodes:
                assignments[node_id] = 'ambiguous'
        
        # Propagate phases through high-confidence edges
        if ai_annotations:
            for iteration in range(5):  # Limited propagation
                changed = False
                
                for edge_id, edge in graph.edges.items():
                    if edge_id in ai_annotations:
                        ai = ai_annotations[edge_id]
                        
                        # Only propagate through high-confidence edges
                        if ai.score_true > 0.7:
                            from_node = getattr(edge, 'from_node', None)
                            to_node = getattr(edge, 'to_node', None)
                            
                            if from_node and to_node:
                                from_phase = assignments.get(from_node, 'ambiguous')
                                to_phase = assignments.get(to_node, 'ambiguous')
                                
                                # Propagate definite phase to ambiguous
                                if from_phase in ['A', 'B'] and to_phase == 'ambiguous':
                                    assignments[to_node] = from_phase
                                    changed = True
                                elif to_phase in ['A', 'B'] and from_phase == 'ambiguous':
                                    assignments[from_node] = to_phase
                                    changed = True
                
                if not changed:
                    break
        
        return assignments


def clean_graph(
    graph,
    hic_phase_info: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    regional_k_map: Optional[Dict[int, int]] = None,
    ul_support_map: Optional[Dict[int, int]] = None,
    min_edge_confidence: float = 0.3,
    min_hic_weight: float = 0.2
) -> CleanedGraphResult:
    """
    Convenience function for graph cleanup.
    
    Args:
        graph: DBGGraph or StringGraph
        hic_phase_info: Node phasing information
        hic_edge_support: Edge Hi-C support
        ai_annotations: AI edge annotations
        regional_k_map: Regional k recommendations
        ul_support_map: UL support counts
        min_edge_confidence: Minimum AI confidence threshold
        min_hic_weight: Minimum Hi-C weight threshold
    
    Returns:
        CleanedGraphResult
    """
    engine = GraphCleanupEngine(
        min_edge_confidence=min_edge_confidence,
        min_hic_weight=min_hic_weight
    )
    
    return engine.clean_graph(
        graph, hic_phase_info, hic_edge_support,
        ai_annotations, regional_k_map, ul_support_map
    )

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
