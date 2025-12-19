#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-Driven Structural Variant (SV) Detection During Assembly.

This module identifies structural variants during graph traversal, not after
assembly completion. By detecting SVs during assembly, we can:
1. Make informed decisions about graph topology
2. Preserve both alleles in heterozygous SVs
3. Flag low-confidence regions for manual review
4. Provide early QC feedback

Supported SV types:
- Large deletions (missing edges/nodes compared to expected path)
- Insertions (branching paths with unusual features)
- Inversions (UL/Hi-C signals contradict graph direction)
- Duplications (repeat overexpansion detected via coverage/k-mer)
- Translocations (Hi-C long-range contacts inconsistent with graph)
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SVCall:
    """
    Structural variant call.
    
    Attributes:
        sv_type: Type of SV (deletion, insertion, inversion, duplication, translocation)
        nodes_involved: List of node IDs involved in SV
        confidence: Confidence in call (0-1)
        evidence: Dict of supporting evidence from different signals
        estimated_size: Estimated size of SV in bases
        haplotype: Haplotype assignment ('A', 'B', 'both', 'unknown')
        breakpoints: List of (from_node, to_node) tuples at SV boundaries
        description: Human-readable description
    """
    sv_type: str
    nodes_involved: List[int]
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    estimated_size: int = 0
    haplotype: str = 'unknown'
    breakpoints: List[Tuple[int, int]] = field(default_factory=list)
    description: str = ''


class DeletionDetector:
    """
    Detects large deletions in assembly graph.
    
    Deletions manifest as:
    - UL reads spanning gap in graph
    - Hi-C contacts across gap
    - Missing expected nodes/edges
    - Coverage drops
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DeletionDetector")
    
    def detect(
        self,
        graph,
        gnn_paths,
        ul_routes: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None
    ) -> List[SVCall]:
        """Detect deletion events."""
        deletions = []
        
        # Look for UL reads spanning gaps
        if ul_routes:
            deletions.extend(self._detect_ul_spanning_gaps(graph, ul_routes))
        
        # Look for Hi-C contacts across gaps
        if hic_edge_support:
            deletions.extend(self._detect_hic_gap_spanning(graph, hic_edge_support))
        
        # Look for coverage drops
        deletions.extend(self._detect_coverage_gaps(graph))
        
        return deletions
    
    def _detect_ul_spanning_gaps(
        self,
        graph,
        ul_routes: Dict
    ) -> List[SVCall]:
        """Find UL reads that span gaps in graph."""
        deletions = []
        
        for read_id, decision in ul_routes.items():
            if not decision.chosen_path or len(decision.chosen_path) < 2:
                continue
            
            # Check for large gaps between consecutive nodes
            for i in range(len(decision.chosen_path) - 1):
                node_a = decision.chosen_path[i]
                node_b = decision.chosen_path[i + 1]
                
                # Check if there's a direct edge
                has_edge = False
                for edge_id in graph.out_edges.get(node_a, set()):
                    edge = graph.edges.get(edge_id)
                    if edge and getattr(edge, 'to_node', None) == node_b:
                        has_edge = True
                        break
                
                if not has_edge:
                    # UL connects nodes without graph edge = potential deletion
                    deletion = SVCall(
                        sv_type='deletion',
                        nodes_involved=[node_a, node_b],
                        confidence=decision.confidence * 0.7,
                        evidence={
                            'ul_read': read_id,
                            'ul_score': decision.score,
                            'type': 'ul_spanning'
                        },
                        breakpoints=[(node_a, node_b)],
                        description=f'UL read {read_id} spans gap between nodes'
                    )
                    deletions.append(deletion)
        
        return deletions
    
    def _detect_hic_gap_spanning(
        self,
        graph,
        hic_edge_support: Dict
    ) -> List[SVCall]:
        """Find Hi-C contacts across graph gaps."""
        deletions = []
        
        # Look for node pairs with high Hi-C contacts but no edge
        hic_node_pairs = defaultdict(float)
        
        for edge_id, support in hic_edge_support.items():
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node and to_node:
                    hic_node_pairs[(from_node, to_node)] = support.hic_weight
        
        # Check all Hi-C contacts (would need full Hi-C data, placeholder here)
        # In real implementation, would iterate through Hi-C pairs
        
        return deletions
    
    def _detect_coverage_gaps(self, graph) -> List[SVCall]:
        """Find regions with unexpected coverage drops."""
        deletions = []
        
        # Simple heuristic: look for nodes with very low coverage
        # surrounded by normal coverage
        nodes_by_id = sorted(graph.nodes.keys())
        
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
                    deletion = SVCall(
                        sv_type='deletion',
                        nodes_involved=[node_id],
                        confidence=0.5,
                        evidence={
                            'coverage': cov,
                            'neighbor_coverage': avg_neighbor,
                            'type': 'coverage_drop'
                        },
                        estimated_size=getattr(node, 'length', 0),
                        description='Coverage drop suggests deletion'
                    )
                    deletions.append(deletion)
        
        return deletions


class InsertionDetector:
    """
    Detects insertions in assembly graph.
    
    Insertions manifest as:
    - Branching paths with high coverage
    - Nodes absent in reference (if available)
    - UL reads supporting extra sequence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InsertionDetector")
    
    def detect(
        self,
        graph,
        gnn_paths,
        diploid_result = None,
        ai_annotations: Optional[Dict] = None
    ) -> List[SVCall]:
        """Detect insertion events."""
        insertions = []
        
        # Look for alternative paths with good support
        insertions.extend(self._detect_alternative_paths(
            graph, gnn_paths, ai_annotations
        ))
        
        # Look for high-coverage branches
        insertions.extend(self._detect_high_coverage_branches(graph))
        
        return insertions
    
    def _detect_alternative_paths(
        self,
        graph,
        gnn_paths,
        ai_annotations: Optional[Dict]
    ) -> List[SVCall]:
        """Find alternative paths that might be insertions."""
        insertions = []
        
        if not hasattr(gnn_paths, 'best_paths'):
            return insertions
        
        # Look for parallel paths (bubbles)
        for i, path_a in enumerate(gnn_paths.best_paths):
            for j, path_b in enumerate(gnn_paths.best_paths[i+1:], i+1):
                # Check if paths share start and end
                if (len(path_a) > 2 and len(path_b) > 2 and
                    path_a[0] == path_b[0] and path_a[-1] == path_b[-1]):
                    
                    # Calculate path lengths
                    len_a = sum(
                        getattr(graph.nodes.get(n), 'length', 0)
                        for n in path_a
                    )
                    len_b = sum(
                        getattr(graph.nodes.get(n), 'length', 0)
                        for n in path_b
                    )
                    
                    # Longer path might be insertion
                    if abs(len_a - len_b) > 1000:  # >1kb difference
                        longer_path = path_a if len_a > len_b else path_b
                        size_diff = abs(len_a - len_b)
                        
                        insertion = SVCall(
                            sv_type='insertion',
                            nodes_involved=longer_path,
                            confidence=0.6,
                            evidence={
                                'path_index': i if len_a > len_b else j,
                                'size_difference': size_diff,
                                'type': 'alternative_path'
                            },
                            estimated_size=size_diff,
                            breakpoints=[(path_a[0], path_a[-1])],
                            description=f'{size_diff}bp insertion in alternative path'
                        )
                        insertions.append(insertion)
        
        return insertions
    
    def _detect_high_coverage_branches(self, graph) -> List[SVCall]:
        """Find branching paths with elevated coverage."""
        insertions = []
        
        # Look for nodes with >1 outgoing edge and high coverage
        for node_id, node in graph.nodes.items():
            out_edges = graph.out_edges.get(node_id, set())
            
            if len(out_edges) > 1:
                coverage = getattr(node, 'coverage', 0.0)
                
                # High coverage might indicate duplication/insertion
                if coverage > 60:
                    insertion = SVCall(
                        sv_type='insertion',
                        nodes_involved=[node_id],
                        confidence=0.4,
                        evidence={
                            'coverage': coverage,
                            'out_degree': len(out_edges),
                            'type': 'high_coverage_branch'
                        },
                        description='High coverage branch suggests insertion'
                    )
                    insertions.append(insertion)
        
        return insertions


class InversionDetector:
    """
    Detects inversions in assembly graph.
    
    Inversions manifest as:
    - UL reads aligning in reverse orientation
    - Hi-C contacts suggesting flipped orientation
    - Strand inconsistencies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InversionDetector")
    
    def detect(
        self,
        graph,
        ul_routes: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None
    ) -> List[SVCall]:
        """Detect inversion events."""
        inversions = []
        
        # Check UL strand consistency
        if ul_routes:
            inversions.extend(self._detect_ul_strand_flips(ul_routes))
        
        # Check Hi-C orientation signals (placeholder)
        # In real implementation, would analyze Hi-C read orientations
        
        return inversions
    
    def _detect_ul_strand_flips(self, ul_routes: Dict) -> List[SVCall]:
        """Find UL reads with strand inconsistencies."""
        inversions = []
        
        for read_id, decision in ul_routes.items():
            # Check if strand_consistent flag is False
            if hasattr(decision, 'evidence') and isinstance(decision.evidence, dict):
                # Would need ULPath objects to check strand consistency
                # Placeholder for now
                pass
        
        return inversions


class DuplicationDetector:
    """
    Detects duplications and repeat expansions.
    
    Duplications manifest as:
    - Very high coverage regions
    - Regional k recommendations suggesting repeat
    - Multiple paths through same sequence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DuplicationDetector")
    
    def detect(
        self,
        graph,
        regional_k_map: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None
    ) -> List[SVCall]:
        """Detect duplication events."""
        duplications = []
        
        # Look for very high coverage nodes
        for node_id, node in graph.nodes.items():
            coverage = getattr(node, 'coverage', 0.0)
            
            # Coverage >2x expected suggests duplication
            if coverage > 80:
                # Check if AI annotations support repeat
                repeat_evidence = 0.0
                for edge_id in graph.out_edges.get(node_id, set()):
                    if ai_annotations and edge_id in ai_annotations:
                        ai = ai_annotations[edge_id]
                        repeat_evidence += ai.score_repeat
                
                avg_repeat = repeat_evidence / max(len(graph.out_edges.get(node_id, set())), 1)
                
                if avg_repeat > 0.5:
                    duplication = SVCall(
                        sv_type='duplication',
                        nodes_involved=[node_id],
                        confidence=0.7,
                        evidence={
                            'coverage': coverage,
                            'repeat_score': avg_repeat,
                            'type': 'high_coverage_repeat'
                        },
                        estimated_size=getattr(node, 'length', 0),
                        description=f'High coverage ({coverage}) suggests duplication'
                    )
                    duplications.append(duplication)
        
        return duplications


class TranslocationDetector:
    """
    Detects translocations and inter-chromosomal rearrangements.
    
    Translocations manifest as:
    - Hi-C long-range contacts inconsistent with linear graph
    - UL reads connecting distant regions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TranslocationDetector")
    
    def detect(
        self,
        graph,
        hic_edge_support: Optional[Dict] = None,
        ul_routes: Optional[Dict] = None
    ) -> List[SVCall]:
        """Detect translocation events."""
        translocations = []
        
        # Look for unexpected Hi-C long-range contacts
        # (Would need full Hi-C contact matrix, placeholder here)
        
        # Look for UL reads connecting distant graph regions
        if ul_routes:
            translocations.extend(self._detect_ul_long_range(graph, ul_routes))
        
        return translocations
    
    def _detect_ul_long_range(self, graph, ul_routes: Dict) -> List[SVCall]:
        """Find UL reads connecting distant graph regions."""
        translocations = []
        
        for read_id, decision in ul_routes.items():
            if not decision.chosen_path or len(decision.chosen_path) < 3:
                continue
            
            # Check graph distance between first and last node
            # (Simplified heuristic: large node ID difference)
            first_node = decision.chosen_path[0]
            last_node = decision.chosen_path[-1]
            
            id_distance = abs(first_node - last_node)
            
            # Very large gap in node IDs might suggest translocation
            if id_distance > 1000:
                translocation = SVCall(
                    sv_type='translocation',
                    nodes_involved=decision.chosen_path,
                    confidence=0.5,
                    evidence={
                        'ul_read': read_id,
                        'node_distance': id_distance,
                        'ul_score': decision.score,
                        'type': 'ul_long_range'
                    },
                    breakpoints=[(first_node, last_node)],
                    description=f'UL read connects distant regions (Δ={id_distance})'
                )
                translocations.append(translocation)
        
        return translocations


class SVDetectionEngine:
    """
    Main engine for structural variant detection.
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize SV detection engine.
        
        Args:
            min_confidence: Minimum confidence for reporting SV
        """
        self.min_confidence = min_confidence
        self.deletion_detector = DeletionDetector()
        self.insertion_detector = InsertionDetector()
        self.inversion_detector = InversionDetector()
        self.duplication_detector = DuplicationDetector()
        self.translocation_detector = TranslocationDetector()
        self.logger = logging.getLogger(f"{__name__}.SVDetectionEngine")
    
    def detect_all_svs(
        self,
        graph,
        gnn_paths = None,
        diploid_result = None,
        ul_routes: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None
    ) -> List[SVCall]:
        """
        Detect all types of structural variants.
        
        Args:
            graph: Assembly graph
            gnn_paths: GNN path predictions
            diploid_result: Diploid disentanglement result
            ul_routes: UL routing decisions
            hic_edge_support: Hi-C edge support
            ai_annotations: AI edge annotations
            regional_k_map: Regional k recommendations
        
        Returns:
            List of SVCall objects
        """
        self.logger.info("Starting SV detection")
        
        all_svs = []
        
        # Detect deletions
        deletions = self.deletion_detector.detect(
            graph, gnn_paths, ul_routes, hic_edge_support
        )
        all_svs.extend(deletions)
        self.logger.info(f"Detected {len(deletions)} potential deletions")
        
        # Detect insertions
        insertions = self.insertion_detector.detect(
            graph, gnn_paths, diploid_result, ai_annotations
        )
        all_svs.extend(insertions)
        self.logger.info(f"Detected {len(insertions)} potential insertions")
        
        # Detect inversions
        inversions = self.inversion_detector.detect(
            graph, ul_routes, hic_edge_support
        )
        all_svs.extend(inversions)
        self.logger.info(f"Detected {len(inversions)} potential inversions")
        
        # Detect duplications
        duplications = self.duplication_detector.detect(
            graph, regional_k_map, ai_annotations
        )
        all_svs.extend(duplications)
        self.logger.info(f"Detected {len(duplications)} potential duplications")
        
        # Detect translocations
        translocations = self.translocation_detector.detect(
            graph, hic_edge_support, ul_routes
        )
        all_svs.extend(translocations)
        self.logger.info(f"Detected {len(translocations)} potential translocations")
        
        # Filter by confidence
        filtered_svs = [
            sv for sv in all_svs
            if sv.confidence >= self.min_confidence
        ]
        
        # Assign haplotypes if diploid result available
        if diploid_result:
            self._assign_haplotypes(filtered_svs, diploid_result)
        
        self.logger.info(
            f"Detected {len(filtered_svs)} SVs above confidence threshold "
            f"(out of {len(all_svs)} total)"
        )
        
        return filtered_svs
    
    def _assign_haplotypes(self, svs: List[SVCall], diploid_result):
        """Assign haplotypes to SVs based on node assignments."""
        for sv in svs:
            if not sv.nodes_involved:
                continue
            
            # Check node haplotypes
            haplotypes = set()
            for node_id in sv.nodes_involved:
                if node_id in diploid_result.hapA_nodes:
                    haplotypes.add('A')
                elif node_id in diploid_result.hapB_nodes:
                    haplotypes.add('B')
            
            if len(haplotypes) == 1:
                sv.haplotype = haplotypes.pop()
            elif len(haplotypes) > 1:
                sv.haplotype = 'both'
            else:
                sv.haplotype = 'unknown'


def detect_structural_variants(
    graph,
    gnn_paths = None,
    diploid_result = None,
    ul_routes: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    min_confidence: float = 0.5
) -> List[SVCall]:
    """
    Main entry point for SV detection.
    
    Detects structural variants during assembly using multiple signals.
    
    Args:
        graph: Assembly graph
        gnn_paths: GNN path predictions
        diploid_result: Diploid disentanglement result
        ul_routes: UL routing decisions
        hic_edge_support: Hi-C edge support
        ai_annotations: AI edge annotations
        regional_k_map: Regional k recommendations
        min_confidence: Minimum confidence for reporting
    
    Returns:
        List of SVCall objects
    """
    engine = SVDetectionEngine(min_confidence=min_confidence)
    
    return engine.detect_all_svs(
        graph, gnn_paths, diploid_result, ul_routes,
        hic_edge_support, ai_annotations, regional_k_map
    )


def svs_to_dict_list(svs: List[SVCall]) -> List[Dict]:
    """
    Convert SVCall objects to dict format for serialization.
    
    Args:
        svs: List of SVCall objects
    
    Returns:
        List of dicts in specified format
    """
    return [
        {
            'sv_type': sv.sv_type,
            'nodes_involved': sv.nodes_involved,
            'confidence': sv.confidence,
            'evidence': sv.evidence,
            'estimated_size': sv.estimated_size,
            'haplotype': sv.haplotype,
            'breakpoints': sv.breakpoints,
            'description': sv.description
        }
        for sv in svs
    ]
