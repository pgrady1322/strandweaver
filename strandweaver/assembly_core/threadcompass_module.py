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


__all__ = [
    'ThreadCompass',
    'ULMapping',
    'ULJoinScore',
]
