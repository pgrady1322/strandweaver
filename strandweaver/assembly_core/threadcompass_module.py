#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ThreadCompass: Ultra-Long (UL) Read Integration and Routing Engine.

This module integrates UL reads to:
1. Map UL reads to the assembly graph using adaptive k-mer sizing
2. Detect new joins from UL spanning evidence
3. Score existing joins for UL support/conflict
4. Produce 0.0-1.0 normalized UL confidence scores

Designed to work DOWNSTREAM of PathWeaver Pass A and UPSTREAM of PathWeaver Pass B,
providing long-range evidence for iterative refinement.

Author: StrandWeaver Development Team
License: MIT
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
#                         DATA STRUCTURES
# ============================================================================

@dataclass
class ULMapping:
    """UL read mapping to assembly graph."""
    read_id: str
    primary_node: int
    secondary_nodes: List[int] = field(default_factory=list)
    mapping_quality: int = 0
    primary_mapq: int = 0
    secondary_mapq: int = 0
    anchor_count: int = 1
    span_start: int = 0
    span_end: int = 0
    is_multimapping: bool = False


@dataclass
class ULJoinScore:
    """Score for a join evaluated via UL evidence."""
    join_id: str
    from_node: int
    to_node: int
    ul_confidence: float = 0.5
    span_coverage: float = 0.0
    anchor_uniqueness: float = 0.0
    mapping_quality_score: float = 0.0
    supporting_reads: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class ULAnchor:
    """
    Single anchor point where UL read aligns to graph.
    
    Attributes:
        node_id: Graph node ID
        read_pos: Position in read (0-based)
        node_pos: Position in node sequence (0-based)
        strand: True for forward, False for reverse
        score: Alignment score
        length: Alignment length
    """
    node_id: int
    read_pos: int
    node_pos: int
    strand: bool
    score: float
    length: int


@dataclass
class ULPath:
    """
    Candidate path through graph for UL read.
    
    Attributes:
        nodes: Ordered list of node IDs
        anchors: List of ULAnchor objects supporting this path
        total_aligned: Total bases aligned
        gaps: List of (gap_start_pos, gap_end_pos, gap_length) tuples
        strand_consistent: Whether all anchors have consistent strand
    """
    nodes: List[int] = field(default_factory=list)
    anchors: List[ULAnchor] = field(default_factory=list)
    total_aligned: int = 0
    gaps: List[Tuple[int, int, int]] = field(default_factory=list)
    strand_consistent: bool = True


@dataclass
class PathFeatures:
    """
    Feature vector for UL path scoring.
    
    Attributes:
        kmer_agreement: K-mer match score (0-1)
        gnn_edge_confidence: Average GNN edge probability along path
        hic_phase_consistency: Hi-C phase agreement (0-1)
        repeat_score: Average repeat likelihood of nodes (0-1)
        entropy_consistency: Sequence entropy variation
        coverage_consistency: Coverage profile consistency
        ul_support: Number of other UL reads supporting this path
        path_length: Total path length in bases
        num_gaps: Number of unaligned gaps
        gap_penalty: Total penalty for gaps
        strand_consistency: Boolean converted to 0/1
        branching_complexity: Average node branching
    """
    kmer_agreement: float = 0.0
    gnn_edge_confidence: float = 0.0
    hic_phase_consistency: float = 0.0
    repeat_score: float = 0.0
    entropy_consistency: float = 0.0
    coverage_consistency: float = 0.0
    ul_support: int = 0
    path_length: int = 0
    num_gaps: int = 0
    gap_penalty: float = 0.0
    strand_consistency: float = 0.0
    branching_complexity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for ML model input."""
        return {
            'kmer_agreement': self.kmer_agreement,
            'gnn_edge_confidence': self.gnn_edge_confidence,
            'hic_phase_consistency': self.hic_phase_consistency,
            'repeat_score': self.repeat_score,
            'entropy_consistency': self.entropy_consistency,
            'coverage_consistency': self.coverage_consistency,
            'ul_support': float(self.ul_support),
            'path_length': math.log(self.path_length + 1) / math.log(1000000),
            'num_gaps': float(self.num_gaps),
            'gap_penalty': self.gap_penalty,
            'strand_consistency': self.strand_consistency,
            'branching_complexity': self.branching_complexity
        }


@dataclass
class ULRouteDecision:
    """
    Final routing decision for a UL read.
    
    Attributes:
        ul_read_id: Read identifier
        chosen_path: Selected path (list of node IDs)
        score: ML score for chosen path
        alternative_paths: Other candidate paths with scores
        confidence: Confidence in decision (0-1)
        evidence: Supporting evidence for decision
    """
    ul_read_id: str
    chosen_path: List[int]
    score: float
    alternative_paths: List[Tuple[List[int], float]] = field(default_factory=list)
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
#                         THREAD COMPASS ENGINE
# ============================================================================

class ThreadCompass:
    """
    UL read integration and routing.
    
    Maps UL reads to find new joins and score existing ones.
    All scores normalized to [0.0, 1.0] for pipeline integration.
    """
    
    def __init__(
        self,
        graph: Any,
        k_mer_size: int = 31,
        ul_mappings: Optional[List[ULMapping]] = None,
        min_anchor_count: int = 2,
        min_mapq: int = 20,
        mapq_weight: float = 0.25,
    ):
        """
        Initialize ThreadCompass.
        
        Args:
            graph: Assembly graph (nodes, edges)
            k_mer_size: K-mer size from preprocessing (K-Weaver output)
            ul_mappings: List of UL mapping objects
            min_anchor_count: Minimum anchor reads for join support
            min_mapq: Minimum mapping quality threshold
            mapq_weight: Weight for mapping quality in scoring
        """
        self.graph = graph
        self.k_mer_size = k_mer_size
        self.ul_mappings = ul_mappings or []
        self.min_anchor_count = min_anchor_count
        self.min_mapq = min_mapq
        self.mapq_weight = mapq_weight
        self.logger = logging.getLogger(f"{__name__}.ThreadCompass")
        
        # Build index of reads mapping to each node
        self._read_index = {}  # node_id -> List[ULMapping]
        self._build_read_index()
    
    def register_ul_mappings(
        self,
        mappings: List[ULMapping],
    ) -> int:
        """
        Register UL read mappings.
        
        Args:
            mappings: List of UL mapping objects
        
        Returns:
            Number of mappings registered
        """
        self.ul_mappings = mappings
        self._build_read_index()
        self.logger.info(f"Registered {len(mappings)} UL read mappings")
        return len(mappings)
    
    def detect_new_joins(
        self,
        max_joins: int = 100,
        min_ul_confidence: float = 0.6,
    ) -> List[Tuple[int, int, float]]:
        """
        Detect new joins from UL spanning evidence.
        
        Returns highest-confidence joins not yet in the graph.
        
        Args:
            max_joins: Maximum joins to return
            min_ul_confidence: Minimum confidence threshold
        
        Returns:
            List of (from_node, to_node, ul_confidence) tuples
        """
        if not self.ul_mappings:
            self.logger.warning("No UL mappings available")
            return []
        
        # Get existing edges
        existing_edges = set()
        if hasattr(self.graph, 'edges'):
            for edge in self.graph.edges.values():
                if hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
                    existing_edges.add((edge.from_node, edge.to_node))
                    existing_edges.add((edge.to_node, edge.from_node))
        
        # Find all potential joins from UL spanning reads
        candidate_joins = {}  # (from_node, to_node) -> ULJoinScore
        
        for mapping in self.ul_mappings:
            if mapping.is_multimapping:
                continue  # Skip multimapping reads
            
            primary = mapping.primary_node
            for secondary in mapping.secondary_nodes:
                # Score this join
                join_key = (min(primary, secondary), max(primary, secondary))
                
                if join_key not in candidate_joins:
                    score = self.score_join(primary, secondary)
                    candidate_joins[join_key] = score
                else:
                    # Accumulate supporting evidence
                    candidate_joins[join_key].supporting_reads += 1
        
        # Filter and rank
        high_confidence = [
            (score.from_node, score.to_node, score.ul_confidence)
            for score in candidate_joins.values()
            if score.ul_confidence >= min_ul_confidence and 
               (score.from_node, score.to_node) not in existing_edges
        ]
        
        high_confidence.sort(key=lambda x: x[2], reverse=True)
        
        self.logger.info(
            f"Detected {len(high_confidence)} high-confidence new joins from UL; "
            f"returning top {min(max_joins, len(high_confidence))}"
        )
        
        return high_confidence[:max_joins]
    
    def score_join(
        self,
        from_node: int,
        to_node: int,
    ) -> ULJoinScore:
        """
        Score an existing or proposed join using UL evidence.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
        
        Returns:
            ULJoinScore with breakdown
        """
        join_id = f"{from_node}_{to_node}"
        
        # Find reads spanning this join
        spanning_reads = self._find_spanning_reads(from_node, to_node)
        
        if not spanning_reads:
            # No direct spanning evidence; score as neutral
            return ULJoinScore(
                join_id=join_id,
                from_node=from_node,
                to_node=to_node,
                ul_confidence=0.3,
                span_coverage=0.0,
                breakdown={"note": "no_spanning_reads"}
            )
        
        # Compute scores from spanning reads
        span_coverage = self._compute_span_coverage(from_node, to_node, spanning_reads)
        anchor_uniqueness = self._compute_anchor_uniqueness(spanning_reads)
        mapq_score = self._compute_mapq_score(spanning_reads)
        
        # Combine scores
        ul_confidence = (
            0.50 * span_coverage +
            0.25 * anchor_uniqueness +
            self.mapq_weight * mapq_score
        )
        
        ul_confidence = max(0.0, min(1.0, ul_confidence))
        
        score = ULJoinScore(
            join_id=join_id,
            from_node=from_node,
            to_node=to_node,
            ul_confidence=ul_confidence,
            span_coverage=span_coverage,
            anchor_uniqueness=anchor_uniqueness,
            mapping_quality_score=mapq_score,
            supporting_reads=len(spanning_reads),
            breakdown={
                "span_coverage": span_coverage,
                "anchor_uniqueness": anchor_uniqueness,
                "mapping_quality": mapq_score,
            }
        )
        
        return score
    
    def score_path(
        self,
        path_node_ids: List[int],
    ) -> float:
        """
        Score a path using cumulative UL support.
        
        Evaluates UL spanning consistency along the path.
        
        Args:
            path_node_ids: Ordered list of node IDs in path
        
        Returns:
            Path UL score (0.0-1.0)
        """
        if len(path_node_ids) < 2:
            return 0.5  # Neutral score for single-node paths
        
        # Score all adjacent joins in the path
        join_scores = []
        for i in range(len(path_node_ids) - 1):
            from_node = path_node_ids[i]
            to_node = path_node_ids[i + 1]
            
            score = self.score_join(from_node, to_node)
            join_scores.append(score.ul_confidence)
        
        if not join_scores:
            return 0.5
        
        # Average join score (weighted by supporting reads)
        weighted_sum = 0.0
        total_support = 0
        
        for i in range(len(path_node_ids) - 1):
            from_node = path_node_ids[i]
            to_node = path_node_ids[i + 1]
            score = self.score_join(from_node, to_node)
            support = max(1, score.supporting_reads)
            weighted_sum += score.ul_confidence * support
            total_support += support
        
        if total_support == 0:
            return 0.5
        
        avg_score = weighted_sum / total_support
        return min(1.0, max(0.0, avg_score))
    
    # Private helper methods
    
    def _build_read_index(self):
        """Index UL mappings by node for fast lookup."""
        self._read_index.clear()
        for mapping in self.ul_mappings:
            node = mapping.primary_node
            if node not in self._read_index:
                self._read_index[node] = []
            self._read_index[node].append(mapping)
            
            for sec_node in mapping.secondary_nodes:
                if sec_node not in self._read_index:
                    self._read_index[sec_node] = []
                self._read_index[sec_node].append(mapping)
    
    def _find_spanning_reads(
        self,
        from_node: int,
        to_node: int,
    ) -> List[ULMapping]:
        """Find UL reads that span both nodes."""
        spanning = []
        
        reads_at_from = self._read_index.get(from_node, [])
        
        for mapping in reads_at_from:
            if to_node in mapping.secondary_nodes or mapping.primary_node == to_node:
                if not mapping.is_multimapping and mapping.primary_mapq >= self.min_mapq:
                    spanning.append(mapping)
        
        return spanning
    
    def _compute_span_coverage(
        self,
        from_node: int,
        to_node: int,
        spanning_reads: List[ULMapping],
    ) -> float:
        """Compute span coverage score (fraction of expected coverage met)."""
        if not spanning_reads:
            return 0.0
        
        # Expected: at least min_anchor_count reads for good coverage
        coverage_ratio = min(1.0, len(spanning_reads) / max(1, self.min_anchor_count))
        return coverage_ratio
    
    def _compute_anchor_uniqueness(
        self,
        spanning_reads: List[ULMapping],
    ) -> float:
        """
        Compute anchor uniqueness score.
        
        Penalizes multimapping reads.
        """
        if not spanning_reads:
            return 0.0
        
        # All spanning reads should be unique (we already filtered multimappers)
        # Score based on MAPQ distribution
        mapq_scores = [min(1.0, read.primary_mapq / 60.0) for read in spanning_reads]
        
        if not mapq_scores:
            return 0.0
        
        return sum(mapq_scores) / len(mapq_scores)
    
    def _compute_mapq_score(
        self,
        spanning_reads: List[ULMapping],
    ) -> float:
        """Compute average mapping quality score."""
        if not spanning_reads:
            return 0.0
        
        # Normalize MAPQ (0-60) to 0.0-1.0
        scores = [min(1.0, read.primary_mapq / 60.0) for read in spanning_reads]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Return summary statistics on UL mappings."""
        if not self.ul_mappings:
            return {"error": "No UL mappings loaded"}
        
        unique_mappings = [m for m in self.ul_mappings if not m.is_multimapping]
        mapq_scores = [m.primary_mapq for m in self.ul_mappings]
        
        return {
            "total_reads": len(self.ul_mappings),
            "unique_mappings": len(unique_mappings),
            "multimapped_reads": len(self.ul_mappings) - len(unique_mappings),
            "mean_mapq": sum(mapq_scores) / len(mapq_scores) if mapq_scores else 0,
            "min_mapq": min(mapq_scores) if mapq_scores else 0,
            "max_mapq": max(mapq_scores) if mapq_scores else 0,
            "indexed_nodes": len(self._read_index),
        }


class PathFeatureExtractor:
    """
    Extracts features for UL path scoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PathFeatureExtractor")
    
    def extract_features(
        self,
        ul_path: ULPath,
        graph,
        gnn_edge_probs: Optional[Dict] = None,
        hic_phase_info: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> PathFeatures:
        """
        Extract comprehensive feature vector for path.
        
        Args:
            ul_path: Candidate path
            graph: Assembly graph
            gnn_edge_probs: GNN edge probabilities
            hic_phase_info: Hi-C phasing
            regional_k_map: Regional k recommendations
            ul_support_map: UL support counts
        
        Returns:
            PathFeatures object
        """
        features = PathFeatures()
        
        # K-mer agreement (from anchors)
        features.kmer_agreement = self._calculate_kmer_agreement(ul_path, graph)
        
        # GNN edge confidence
        features.gnn_edge_confidence = self._calculate_gnn_confidence(
            ul_path, graph, gnn_edge_probs
        )
        
        # Hi-C phase consistency
        features.hic_phase_consistency = self._calculate_phase_consistency(
            ul_path, hic_phase_info
        )
        
        # Repeat score
        features.repeat_score = self._calculate_repeat_score(ul_path, graph)
        
        # Entropy consistency
        features.entropy_consistency = self._calculate_entropy_consistency(
            ul_path, graph
        )
        
        # Coverage consistency
        features.coverage_consistency = self._calculate_coverage_consistency(
            ul_path, graph
        )
        
        # UL support
        features.ul_support = self._count_ul_support(
            ul_path, graph, ul_support_map
        )
        
        # Path length
        features.path_length = sum(
            getattr(graph.nodes.get(node_id), 'length', 0)
            for node_id in ul_path.nodes
        )
        
        # Gap analysis
        features.num_gaps = len(ul_path.gaps)
        features.gap_penalty = self._calculate_gap_penalty(ul_path)
        
        # Strand consistency
        features.strand_consistency = 1.0 if ul_path.strand_consistent else 0.0
        
        # Branching complexity
        features.branching_complexity = self._calculate_branching(ul_path, graph)
        
        return features
    
    def _calculate_kmer_agreement(self, ul_path: ULPath, graph) -> float:
        """Calculate k-mer agreement score based on alignment scores from anchors."""
        if not ul_path.anchors:
            return 0.0
        
        total_score = sum(anchor.score for anchor in ul_path.anchors)
        total_length = sum(anchor.length for anchor in ul_path.anchors)
        
        if total_length == 0:
            return 0.0
        
        # Normalize score (assume max score ~ length)
        return min(total_score / total_length, 1.0)
    
    def _calculate_gnn_confidence(
        self,
        ul_path: ULPath,
        graph,
        gnn_edge_probs: Optional[Dict]
    ) -> float:
        """Average GNN edge probability along path."""
        if not gnn_edge_probs or len(ul_path.nodes) < 2:
            return 0.5
        
        scores = []
        for i in range(len(ul_path.nodes) - 1):
            from_node = ul_path.nodes[i]
            to_node = ul_path.nodes[i + 1]
            
            # Find edge between these nodes
            for edge_id in graph.out_edges.get(from_node, set()):
                edge = graph.edges.get(edge_id)
                if edge and getattr(edge, 'to_node', None) == to_node:
                    scores.append(gnn_edge_probs.get(edge_id, 0.5))
                    break
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_phase_consistency(
        self,
        ul_path: ULPath,
        hic_phase_info: Optional[Dict]
    ) -> float:
        """
        Calculate Hi-C phase consistency.
        
        If all nodes have same phase, high score.
        If mixed phases, low score.
        """
        if not hic_phase_info:
            return 0.5
        
        phases = []
        for node_id in ul_path.nodes:
            if node_id in hic_phase_info:
                phase_info = hic_phase_info[node_id]
                assignment = phase_info.phase_assignment
                if assignment in ('A', 'B'):
                    phases.append(assignment)
        
        if not phases:
            return 0.5
        
        # All same phase
        if all(p == phases[0] for p in phases):
            return 1.0
        
        # Mixed phases
        a_count = phases.count('A')
        b_count = phases.count('B')
        dominant = max(a_count, b_count)
        
        return dominant / len(phases)
    
    def _calculate_repeat_score(self, ul_path: ULPath, graph) -> float:
        """Average repeat likelihood of nodes in path."""
        scores = []
        
        for node_id in ul_path.nodes:
            node = graph.nodes.get(node_id)
            if node:
                coverage = getattr(node, 'coverage', 0.0)
                in_degree = len(graph.in_edges.get(node_id, set()))
                out_degree = len(graph.out_edges.get(node_id, set()))
                
                repeat_score = 0.0
                if coverage > 50:
                    repeat_score += 0.3
                if (in_degree + out_degree) > 2:
                    repeat_score += 0.4
                
                scores.append(repeat_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_entropy_consistency(self, ul_path: ULPath, graph) -> float:
        """
        Calculate sequence entropy consistency.
        
        Low variance = consistent = good.
        """
        entropies = []
        
        for node_id in ul_path.nodes:
            node = graph.nodes.get(node_id)
            if node:
                seq = getattr(node, 'seq', '')
                entropy = self._sequence_entropy(seq)
                entropies.append(entropy)
        
        if not entropies:
            return 0.5
        
        # Calculate variance
        mean = sum(entropies) / len(entropies)
        variance = sum((e - mean) ** 2 for e in entropies) / len(entropies)
        
        # Low variance = high consistency
        return 1.0 / (1.0 + variance)
    
    def _sequence_entropy(self, seq: str) -> float:
        """Calculate Shannon entropy of sequence."""
        if not seq:
            return 0.0
        
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for base in seq.upper():
            if base in counts:
                counts[base] += 1
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy / 2.0
    
    def _calculate_coverage_consistency(self, ul_path: ULPath, graph) -> float:
        """Calculate coverage consistency along path."""
        coverages = []
        
        for node_id in ul_path.nodes:
            node = graph.nodes.get(node_id)
            if node:
                coverages.append(getattr(node, 'coverage', 0.0))
        
        if not coverages:
            return 0.5
        
        # Calculate coefficient of variation
        mean = sum(coverages) / len(coverages)
        if mean == 0:
            return 0.5
        
        variance = sum((c - mean) ** 2 for c in coverages) / len(coverages)
        cv = math.sqrt(variance) / mean
        
        # Low CV = high consistency
        return 1.0 / (1.0 + cv)
    
    def _count_ul_support(
        self,
        ul_path: ULPath,
        graph,
        ul_support_map: Optional[Dict]
    ) -> int:
        """Count UL reads supporting edges in this path."""
        if not ul_support_map:
            return 0
        
        total = 0
        for i in range(len(ul_path.nodes) - 1):
            from_node = ul_path.nodes[i]
            
            for edge_id in graph.out_edges.get(from_node, set()):
                total += ul_support_map.get(edge_id, 0)
        
        return total
    
    def _calculate_gap_penalty(self, ul_path: ULPath) -> float:
        """
        Calculate penalty for gaps in alignment.
        
        Large gaps or many gaps reduce score.
        """
        if not ul_path.gaps:
            return 0.0
        
        penalty = 0.0
        for _, _, gap_length in ul_path.gaps:
            # Logarithmic penalty
            penalty += math.log(gap_length + 1) / math.log(10000)
        
        return min(penalty, 1.0)
    
    def _calculate_branching(self, ul_path: ULPath, graph) -> float:
        """Average branching factor along path."""
        branching = []
        
        for node_id in ul_path.nodes:
            in_degree = len(graph.in_edges.get(node_id, set()))
            out_degree = len(graph.out_edges.get(node_id, set()))
            branching.append(in_degree + out_degree)
        
        if not branching:
            return 0.0
        
        avg = sum(branching) / len(branching)
        return min(avg / 10.0, 1.0)


class ULRoutingEngine:
    """
    ML-based engine for UL read routing through assembly graph.
    """
    
    def __init__(self, ml_model=None):
        """
        Initialize routing engine.
        
        Args:
            ml_model: Optional ML model with predict(features_dict) method
        """
        self.ml_model = ml_model
        self.feature_extractor = PathFeatureExtractor()
        self.logger = logging.getLogger(f"{__name__}.ULRoutingEngine")
    
    def route_ul_read(
        self,
        ul_read_id: str,
        candidate_paths: List[ULPath],
        graph,
        gnn_edge_probs: Optional[Dict] = None,
        hic_phase_info: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> ULRouteDecision:
        """
        Route single UL read by scoring candidate paths.
        
        Args:
            ul_read_id: Read identifier
            candidate_paths: List of possible paths through graph
            graph: Assembly graph
            gnn_edge_probs: GNN edge probabilities
            hic_phase_info: Hi-C phasing
            regional_k_map: Regional k recommendations
            ul_support_map: UL support counts
        
        Returns:
            ULRouteDecision with chosen path
        """
        if not candidate_paths:
            self.logger.warning(f"No candidate paths for {ul_read_id}")
            return ULRouteDecision(
                ul_read_id=ul_read_id,
                chosen_path=[],
                score=0.0,
                confidence=0.0
            )
        
        # Score all paths
        path_scores = []
        for path in candidate_paths:
            features = self.feature_extractor.extract_features(
                path, graph, gnn_edge_probs, hic_phase_info,
                regional_k_map, ul_support_map
            )
            
            score = self._score_path(features)
            path_scores.append((path, score, features))
        
        # Sort by score
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Best path
        best_path, best_score, best_features = path_scores[0]
        
        # Calculate confidence (gap between best and second-best)
        if len(path_scores) > 1:
            second_score = path_scores[1][1]
            confidence = (best_score - second_score) / max(best_score, 0.01)
            confidence = min(confidence, 1.0)
        else:
            confidence = best_score
        
        # Build decision
        decision = ULRouteDecision(
            ul_read_id=ul_read_id,
            chosen_path=best_path.nodes,
            score=best_score,
            confidence=confidence,
            evidence={
                'features': best_features.to_dict(),
                'num_candidates': len(candidate_paths),
                'total_aligned': best_path.total_aligned,
                'num_anchors': len(best_path.anchors)
            }
        )
        
        # Add alternatives
        for path, score, _ in path_scores[1:4]:  # Top 3 alternatives
            decision.alternative_paths.append((path.nodes, score))
        
        return decision
    
    def _score_path(self, features: PathFeatures) -> float:
        """
        Score path using ML model or heuristic.
        
        Args:
            features: PathFeatures object
        
        Returns:
            Score in 0-1 range
        """
        if self.ml_model:
            return self._score_with_model(features)
        else:
            return self._score_heuristic(features)
    
    def _score_with_model(self, features: PathFeatures) -> float:
        """Use trained ML model."""
        # In real implementation:
        # prediction = self.ml_model.predict(features.to_dict())
        # return prediction
        
        return self._score_heuristic(features)
    
    def _score_heuristic(self, features: PathFeatures) -> float:
        """
        Heuristic scoring function.
        
        Weights:
        - K-mer agreement: 25%
        - GNN edge confidence: 20%
        - Hi-C phase consistency: 15%
        - Coverage consistency: 10%
        - Entropy consistency: 10%
        - Strand consistency: 10%
        - UL support: 5%
        - Penalties: gap, repeat, branching: -5%
        """
        score = 0.0
        
        # Positive signals
        score += features.kmer_agreement * 0.25
        score += features.gnn_edge_confidence * 0.20
        score += features.hic_phase_consistency * 0.15
        score += features.coverage_consistency * 0.10
        score += features.entropy_consistency * 0.10
        score += features.strand_consistency * 0.10
        
        # UL support (log-scaled)
        ul_score = math.log(features.ul_support + 1) / math.log(20)
        score += ul_score * 0.05
        
        # Penalties
        score -= features.gap_penalty * 0.05
        score -= features.repeat_score * 0.03
        score -= features.branching_complexity * 0.02
        
        return max(0.0, min(1.0, score))


def resolve_ul_routes(
    ul_alignments: Dict[str, List[ULPath]],
    graph,
    gnn_edge_probs: Optional[Dict] = None,
    hic_phase_info: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None,
    ml_model=None
) -> Dict[str, ULRouteDecision]:
    """
    Main entry point for UL routing.
    
    Args:
        ul_alignments: Dict[read_id] -> List[candidate_paths]
        graph: Assembly graph
        gnn_edge_probs: GNN edge probabilities
        hic_phase_info: Hi-C phasing
        regional_k_map: Regional k recommendations
        ul_support_map: UL support counts
        ml_model: Optional trained ML model
    
    Returns:
        Dict[read_id] -> ULRouteDecision
    """
    engine = ULRoutingEngine(ml_model=ml_model)
    
    decisions = {}
    for read_id, paths in ul_alignments.items():
        decision = engine.route_ul_read(
            read_id, paths, graph, gnn_edge_probs,
            hic_phase_info, regional_k_map, ul_support_map
        )
        decisions[read_id] = decision
    
    return decisions


__all__ = [
    'ThreadCompass',
    'ULMapping',
    'ULJoinScore',
    'ULAnchor',
    'ULPath',
    'PathFeatures',
    'ULRouteDecision',
    'PathFeatureExtractor',
    'ULRoutingEngine',
    'resolve_ul_routes',
]
