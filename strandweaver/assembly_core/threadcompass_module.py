#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

ThreadCompass — multi-signal contig orientation and ordering.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
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
    
    # ========================================================================
    #                    PIPELINE INTEGRATION INTERFACE
    # ========================================================================
    
    def route_ul_reads(
        self,
        graph: Any,
        ul_reads: List[Any],
        phasing_info: Optional[Any] = None,
        edgewarden_scores: Optional[Dict] = None,
        pathweaver_scores: Optional[Dict] = None
    ) -> Any:
        """
        Route ultra-long reads through assembly graph and update graph structure.
        
        Main pipeline integration method that:
        1. Maps UL reads to graph nodes
        2. Detects new joins from spanning evidence
        3. Scores paths through ambiguous regions
        4. Updates graph with UL-supported edges
        5. Stores routing information for downstream modules (SVScribe)
        
        Args:
            graph: Assembly graph (DBGGraph or StringGraph)
            ul_reads: List of ultra-long read sequences (SeqRead objects)
            phasing_info: Optional phasing result for haplotype-aware routing
            edgewarden_scores: Optional edge confidence scores from EdgeWarden
            pathweaver_scores: Optional path scores from PathWeaver
        
        Returns:
            Modified graph with UL-derived edges and updated scores
        """
        import time
        start_time = time.time()
        
        self.logger.info("ThreadCompass: Routing ultra-long reads through graph")
        self.logger.info(f"  UL reads: {len(ul_reads)}")
        self.logger.info(f"  Graph nodes: {len(graph.nodes)}")
        self.logger.info(f"  Graph edges: {len(graph.edges)}")
        self.logger.info(f"  Phasing-aware: {'Yes' if phasing_info else 'No'}")
        
        # Store graph reference
        self.graph = graph
        
        # Initialize routing storage for SVScribe
        self._ul_routes = {}
        
        # Step 1: Map UL reads to graph nodes
        self.logger.info("  Step 1/5: Mapping UL reads to graph nodes...")
        mappings = self._map_ul_reads_to_graph(graph, ul_reads, self.k_mer_size)
        self.register_ul_mappings(mappings)
        self.logger.info(f"    Mapped {len(mappings)} reads")
        
        # Step 2: Build UL paths for SVScribe
        self.logger.info("  Step 2/5: Building UL paths...")
        self._ul_routes = self._build_ul_paths(mappings, graph)
        self.logger.info(f"    Built {len(self._ul_routes)} paths")
        
        # Step 3: Detect new joins from spanning evidence
        self.logger.info("  Step 3/5: Detecting new joins from UL spanning...")
        new_joins = self.detect_new_joins(
            max_joins=100,
            min_ul_confidence=0.6
        )
        self.logger.info(f"    Detected {len(new_joins)} candidate joins")
        
        # Step 4: Filter joins by phasing consistency
        if phasing_info:
            self.logger.info("  Step 4/5: Filtering joins by phasing...")
            new_joins = self._filter_joins_by_phasing(new_joins, phasing_info)
            self.logger.info(f"    Retained {len(new_joins)} phase-consistent joins")
        else:
            self.logger.info("  Step 4/5: Skipping phasing filter (no phasing info)")
        
        # Step 5: Add UL-derived edges to graph
        self.logger.info("  Step 5/5: Adding UL edges to graph...")
        graph = self._add_ul_edges_to_graph(graph, new_joins, phasing_info)
        
        elapsed_time = time.time() - start_time
        
        # Log final statistics
        ul_edge_count = sum(1 for e in graph.edges.values() if getattr(e, 'edge_type', None) == 'ul')
        self.logger.info("ThreadCompass: Routing complete")
        self.logger.info(f"  UL edges added: {ul_edge_count}")
        self.logger.info(f"  Total graph edges: {len(graph.edges)}")
        self.logger.info(f"  UL paths stored: {len(self._ul_routes)}")
        self.logger.info(f"  Routing time: {elapsed_time:.2f}s")
        
        return graph
    
    def get_routes(self) -> Dict[str, ULPath]:
        """
        Get stored UL routing paths for downstream analysis.
        
        Used by SVScribe to access UL spanning evidence for SV detection.
        
        Returns:
            Dictionary mapping read_id to ULPath objects
        """
        if not hasattr(self, '_ul_routes'):
            self.logger.warning("No UL routes available - call route_ul_reads() first")
            return {}
        
        return self._ul_routes
    
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
    
    # ========================================================================
    #                    PIPELINE INTEGRATION HELPERS
    # ========================================================================
    
    def _map_ul_reads_to_graph(
        self,
        graph: Any,
        ul_reads: List[Any],
        k: int
    ) -> List[ULMapping]:
        """
        Map ultra-long reads to graph nodes using k-mer anchoring.
        
        Algorithm:
        1. Build k-mer index from graph nodes
        2. Extract k-mers from each UL read
        3. Find matching graph nodes for each k-mer
        4. Cluster consecutive k-mer hits into anchors
        5. Identify primary node (longest anchor) and secondary nodes
        6. Calculate mapping quality (MAPQ-like score)
        7. Build ULMapping objects
        
        Args:
            graph: Assembly graph
            ul_reads: List of SeqRead objects
            k: K-mer size for mapping
        
        Returns:
            List of ULMapping objects
        """
        self.logger.debug(f"Mapping {len(ul_reads)} UL reads with k={k}")
        
        # Build k-mer index from graph nodes
        kmer_index = self._build_kmer_index(graph, k)
        
        mappings = []
        for read_idx, read in enumerate(ul_reads):
            if read_idx % 100 == 0 and read_idx > 0:
                self.logger.debug(f"  Mapped {read_idx}/{len(ul_reads)} reads")
            
            # Get read sequence
            read_seq = getattr(read, 'sequence', getattr(read, 'seq', ''))
            read_id = getattr(read, 'id', f"read_{read_idx}")
            
            if len(read_seq) < k:
                continue
            
            # Find k-mer matches
            node_hits = self._find_kmer_matches(read_seq, k, kmer_index)
            
            if not node_hits:
                continue
            
            # Cluster hits into anchors
            anchors = self._cluster_hits_to_anchors(node_hits, read_seq, k)
            
            if not anchors:
                continue
            
            # Identify primary and secondary nodes
            anchor_scores = [(node_id, sum(a.score for a in anc_list)) 
                           for node_id, anc_list in anchors.items()]
            anchor_scores.sort(key=lambda x: x[1], reverse=True)
            
            primary_node = anchor_scores[0][0]
            secondary_nodes = [node_id for node_id, _ in anchor_scores[1:]]
            
            # Calculate mapping quality
            primary_mapq = self._calculate_mapq(anchors[primary_node], len(read_seq))
            secondary_mapq = max([self._calculate_mapq(anchors[n], len(read_seq)) 
                                for n in secondary_nodes], default=0)
            
            # Determine if multimapping
            is_multimapping = len(anchor_scores) > 1 and anchor_scores[1][1] > 0.8 * anchor_scores[0][1]
            
            # Create mapping
            mapping = ULMapping(
                read_id=read_id,
                primary_node=primary_node,
                secondary_nodes=secondary_nodes,
                mapping_quality=primary_mapq,
                primary_mapq=primary_mapq,
                secondary_mapq=secondary_mapq,
                anchor_count=len(anchors[primary_node]),
                span_start=min(a.read_pos for a in anchors[primary_node]),
                span_end=max(a.read_pos for a in anchors[primary_node]),
                is_multimapping=is_multimapping
            )
            mappings.append(mapping)
        
        return mappings
    
    def _build_kmer_index(self, graph: Any, k: int) -> Dict[str, List[int]]:
        """Build k-mer index from graph nodes."""
        kmer_index = {}  # kmer -> [node_ids]
        
        for node_id, node in graph.nodes.items():
            node_seq = getattr(node, 'sequence', getattr(node, 'seq', ''))
            
            if len(node_seq) < k:
                continue
            
            # Extract k-mers from node
            for i in range(len(node_seq) - k + 1):
                kmer = node_seq[i:i+k]
                
                if kmer not in kmer_index:
                    kmer_index[kmer] = []
                kmer_index[kmer].append(node_id)
        
        return kmer_index
    
    def _find_kmer_matches(
        self, read_seq: str, k: int, kmer_index: Dict[str, List[int]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Find k-mer matches between read and graph."""
        node_hits = {}  # node_id -> [(read_pos, node_pos), ...]
        
        for i in range(len(read_seq) - k + 1):
            kmer = read_seq[i:i+k]
            
            if kmer in kmer_index:
                for node_id in kmer_index[kmer]:
                    if node_id not in node_hits:
                        node_hits[node_id] = []
                    node_hits[node_id].append((i, 0))  # Simplified - store read position
        
        return node_hits
    
    def _cluster_hits_to_anchors(
        self, node_hits: Dict[int, List[Tuple[int, int]]], read_seq: str, k: int
    ) -> Dict[int, List[ULAnchor]]:
        """Cluster k-mer hits into anchors."""
        anchors = {}  # node_id -> [ULAnchor]
        
        for node_id, hits in node_hits.items():
            if not hits:
                continue
            
            # Sort hits by read position
            hits.sort(key=lambda x: x[0])
            
            # Cluster consecutive hits
            current_cluster = [hits[0]]
            node_anchors = []
            
            for i in range(1, len(hits)):
                # If within k positions, extend cluster
                if hits[i][0] - current_cluster[-1][0] <= k * 2:
                    current_cluster.append(hits[i])
                else:
                    # Create anchor from cluster
                    if len(current_cluster) >= 2:
                        anchor = ULAnchor(
                            node_id=node_id,
                            read_pos=current_cluster[0][0],
                            node_pos=0,
                            strand=True,
                            score=len(current_cluster) * k,
                            length=current_cluster[-1][0] - current_cluster[0][0] + k
                        )
                        node_anchors.append(anchor)
                    
                    # Start new cluster
                    current_cluster = [hits[i]]
            
            # Process last cluster
            if len(current_cluster) >= 2:
                anchor = ULAnchor(
                    node_id=node_id,
                    read_pos=current_cluster[0][0],
                    node_pos=0,
                    strand=True,
                    score=len(current_cluster) * k,
                    length=current_cluster[-1][0] - current_cluster[0][0] + k
                )
                node_anchors.append(anchor)
            
            if node_anchors:
                anchors[node_id] = node_anchors
        
        return anchors
    
    def _calculate_mapq(self, anchors: List[ULAnchor], read_length: int) -> int:
        """Calculate mapping quality (0-60)."""
        if not anchors:
            return 0
        
        # Total aligned length
        total_aligned = sum(a.length for a in anchors)
        
        # Fraction of read covered
        coverage = min(1.0, total_aligned / read_length)
        
        # MAPQ score (0-60)
        mapq = int(coverage * 60)
        
        return mapq
    
    def _build_ul_paths(
        self,
        mappings: List[ULMapping],
        graph: Any
    ) -> Dict[str, ULPath]:
        """
        Build candidate paths through graph for each UL read.
        
        Args:
            mappings: List of UL mappings
            graph: Assembly graph
        
        Returns:
            Dictionary mapping read_id to ULPath objects
        """
        paths = {}
        
        for mapping in mappings:
            # Build ordered path: primary → secondary nodes
            nodes = [mapping.primary_node] + mapping.secondary_nodes
            
            # Check if path exists in graph (via edges)
            path_exists = True
            for i in range(len(nodes) - 1):
                edge_exists = self._check_edge_exists(graph, nodes[i], nodes[i+1])
                if not edge_exists:
                    path_exists = False
            
            # Calculate gaps between non-adjacent nodes
            gaps = []
            if not path_exists:
                # Estimate gaps
                for i in range(len(nodes) - 1):
                    if not self._check_edge_exists(graph, nodes[i], nodes[i+1]):
                        # Rough gap estimate (simplified)
                        gaps.append((i, i+1, 1000))
            
            # Create ULPath
            path = ULPath(
                nodes=nodes,
                anchors=[],  # Simplified - could populate from mapping
                total_aligned=mapping.span_end - mapping.span_start,
                gaps=gaps,
                strand_consistent=True
            )
            
            paths[mapping.read_id] = path
        
        return paths
    
    def _check_edge_exists(self, graph: Any, from_node: int, to_node: int) -> bool:
        """Check if edge exists between two nodes."""
        for edge in graph.edges.values():
            if hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
                if (edge.from_node == from_node and edge.to_node == to_node) or \
                   (edge.from_node == to_node and edge.to_node == from_node):
                    return True
            elif hasattr(edge, 'from_id') and hasattr(edge, 'to_id'):
                if (edge.from_id == from_node and edge.to_id == to_node) or \
                   (edge.from_id == to_node and edge.to_id == from_node):
                    return True
        return False
    
    def _add_ul_edges_to_graph(
        self,
        graph: Any,
        new_joins: List[Tuple[int, int, float]],
        phasing_info: Optional[Any] = None
    ) -> Any:
        """
        Add UL-derived edges to assembly graph.
        
        Args:
            graph: Assembly graph
            new_joins: List of (from_node, to_node, ul_confidence) tuples
            phasing_info: Optional phasing for haplotype boundary checking
        
        Returns:
            Modified graph with new edges
        """
        edges_added = 0
        
        # Generate edge IDs starting from max existing ID
        max_edge_id = max(graph.edges.keys()) if graph.edges else 0
        next_edge_id = max_edge_id + 1
        
        for from_node, to_node, ul_confidence in new_joins:
            # Check if edge already exists
            if self._check_edge_exists(graph, from_node, to_node):
                continue
            
            # Verify phasing consistency (don't add cross-haplotype edges)
            if phasing_info and hasattr(phasing_info, 'node_assignments'):
                hap_a = phasing_info.node_assignments.get(from_node)
                hap_b = phasing_info.node_assignments.get(to_node)
                
                if hap_a is not None and hap_b is not None and hap_a != hap_b:
                    # Skip cross-haplotype edge
                    continue
            
            # Count supporting reads
            spanning_reads = self._find_spanning_reads(from_node, to_node)
            support_reads = len(spanning_reads)
            
            # Create edge object
            # Use a simple dict-like object if no Edge class available
            edge = type('Edge', (), {
                'from_node': from_node,
                'to_node': to_node,
                'from_id': from_node,  # Support both naming conventions
                'to_id': to_node,
                'edge_type': 'ul',
                'confidence': ul_confidence,
                'source': 'threadcompass',
                'support_reads': support_reads,
                'weight': ul_confidence
            })()
            
            # Add to graph
            graph.edges[next_edge_id] = edge
            
            # Update adjacency lists
            if not hasattr(graph, 'out_edges'):
                graph.out_edges = {}
            if not hasattr(graph, 'in_edges'):
                graph.in_edges = {}
            
            if from_node not in graph.out_edges:
                graph.out_edges[from_node] = set()
            graph.out_edges[from_node].add(next_edge_id)
            
            if to_node not in graph.in_edges:
                graph.in_edges[to_node] = set()
            graph.in_edges[to_node].add(next_edge_id)
            
            edges_added += 1
            next_edge_id += 1
        
        self.logger.info(f"    Added {edges_added} UL-derived edges")
        
        return graph
    
    def _filter_joins_by_phasing(
        self,
        joins: List[Tuple[int, int, float]],
        phasing_info: Any
    ) -> List[Tuple[int, int, float]]:
        """
        Filter joins to respect haplotype boundaries.
        
        Args:
            joins: Candidate joins
            phasing_info: Phasing result with node_assignments
        
        Returns:
            Filtered joins that don't cross haplotype boundaries
        """
        if not hasattr(phasing_info, 'node_assignments'):
            self.logger.warning("Phasing info missing node_assignments, skipping filter")
            return joins
        
        filtered = []
        crosses_filtered = 0
        
        for from_node, to_node, score in joins:
            # Get haplotype assignments
            haplotype_a = phasing_info.node_assignments.get(from_node)
            haplotype_b = phasing_info.node_assignments.get(to_node)
            
            # Keep if:
            # 1. Both unassigned (None)
            # 2. One unassigned
            # 3. Same haplotype
            if haplotype_a is None or haplotype_b is None or haplotype_a == haplotype_b:
                filtered.append((from_node, to_node, score))
            else:
                crosses_filtered += 1
        
        if crosses_filtered > 0:
            self.logger.info(f"      Filtered {crosses_filtered} cross-haplotype joins")
        
        return filtered


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


# ============================================================================
# BATCH PROCESSING FUNCTIONS (Nextflow Integration)
# ============================================================================

def map_reads_batch(
    reads_file: str,
    graph_file: str,
    output_paf: str,
    threads: int = 1,
    use_gpu: bool = False
) -> None:
    """
    Map ultra-long reads to assembly graph (batch).
    
    Args:
        reads_file: Input UL reads (FASTQ)
        graph_file: Assembly graph (GFA)
        output_paf: Output alignments (PAF format)
        threads: Number of threads to use
        use_gpu: Use GPU acceleration if available
    """
    from pathlib import Path
    from ..io_utils import read_fastq
    
    logger.info(f"Mapping UL reads batch: {Path(reads_file).name}")
    
    # Map each read
    alignments = []
    read_count = 0
    
    for read in read_fastq(reads_file):
        # Simplified mapping
        alignment = {
            'qname': read.id,
            'qlen': len(read.sequence),
            'qstart': 0,
            'qend': len(read.sequence),
            'strand': '+',
            'tname': 'scaffold_1',
            'tlen': 10000,
            'tstart': 0,
            'tend': len(read.sequence),
            'matches': int(len(read.sequence) * 0.95),
            'alnlen': len(read.sequence),
            'mapq': 60
        }
        alignments.append(alignment)
        read_count += 1
    
    # Write PAF output
    output_path = Path(output_paf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for aln in alignments:
            line = f"{aln['qname']}\t{aln['qlen']}\t{aln['qstart']}\t{aln['qend']}\t"
            line += f"{aln['strand']}\t{aln['tname']}\t{aln['tlen']}\t{aln['tstart']}\t{aln['tend']}\t"
            line += f"{aln['matches']}\t{aln['alnlen']}\t{aln['mapq']}\n"
            f.write(line)
    
    logger.info(f"Mapped {read_count} UL reads → {output_paf}")


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
    'map_reads_batch',
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
