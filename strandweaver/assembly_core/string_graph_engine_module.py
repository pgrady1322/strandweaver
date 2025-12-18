#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
String Graph Engine for StrandWeaver.

This module implements a string graph overlay on top of a de Bruijn graph,
using ultra-long (UL) ONT reads to bridge DBG nodes and increase contiguity.

Key features:
- Accepts preprocessed DBG with regional k-mer annotations
- Uses UL read alignments to find paths through the DBG
- Creates string graph edges representing UL-supported connections
- Incorporates ML regional-k information for path scoring and filtering
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict
import logging

from .dbg_engine_module import DBGGraph, DBGNode

logger = logging.getLogger(__name__)


@dataclass
class ULAnchor:
    """
    Anchor point where a UL read aligns to a DBG node.
    
    Attributes:
        ul_read_id: Identifier of the ultra-long read
        node_id: DBG node ID where anchor occurs
        pos_on_node: Position within the node sequence (0-based)
        pos_on_read: Position within the UL read sequence (0-based)
        strand: Alignment strand ('+' or '-')
    """
    ul_read_id: str
    node_id: int
    pos_on_node: int
    pos_on_read: int
    strand: str = '+'


# Type alias for UL paths through the DBG
ULPath = List[int]  # Ordered sequence of DBG node IDs


@dataclass
class StringGraphEdge:
    """
    Edge in the string graph representing a UL-supported connection.
    
    Attributes:
        id: Unique edge identifier
        from_node: Source DBG node ID
        to_node: Target DBG node ID
        support_count: Number of UL reads supporting this connection
        ul_read_ids: Set of UL read IDs supporting this edge
        avg_recommended_k: Average ML-recommended k along this path
        path_confidence: Confidence score for this path (0.0-1.0)
    """
    id: int
    from_node: int
    to_node: int
    support_count: int
    ul_read_ids: Set[str] = field(default_factory=set)
    avg_recommended_k: Optional[float] = None
    path_confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class StringGraph:
    """
    String graph structure overlaying a DBG with UL-supported long edges.
    
    Attributes:
        dbg: Underlying de Bruijn graph
        edges: Map of edge_id -> StringGraphEdge
        out_edges: Map of node_id -> set of outgoing string edge_ids
        in_edges: Map of node_id -> set of incoming string edge_ids
        ul_paths: All UL paths used to construct this graph
    """
    dbg: DBGGraph
    edges: Dict[int, StringGraphEdge] = field(default_factory=dict)
    out_edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    in_edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    ul_paths: List[ULPath] = field(default_factory=list)


class StringGraphBuilder:
    """
    Builder class for constructing string graphs from DBG + UL reads.
    
    Algorithm:
    1. Map UL read anchors to DBG nodes
    2. Build ordered paths through DBG for each UL read
    3. Create string graph edges from consecutive node pairs in paths
    4. Aggregate support across multiple UL reads
    5. Incorporate ML regional-k information into edge metadata
    """
    
    def __init__(self, min_support: int = 2, min_path_confidence: float = 0.5):
        """
        Initialize string graph builder.
        
        Args:
            min_support: Minimum UL reads required to retain an edge
            min_path_confidence: Minimum confidence score to retain an edge
        """
        self.min_support = min_support
        self.min_path_confidence = min_path_confidence
        self.next_edge_id = 0
    
    def build_ul_paths_from_anchors(
        self,
        dbg: DBGGraph,
        ul_anchors: List[ULAnchor]
    ) -> List[ULPath]:
        """
        Group anchors by UL read and derive ordered node paths.
        
        Args:
            dbg: De Bruijn graph
            ul_anchors: List of UL read -> DBG node anchor points
        
        Returns:
            List of UL paths (ordered sequences of DBG node IDs)
        
        Algorithm:
            1. Group anchors by ul_read_id
            2. Sort anchors within each read by pos_on_read
            3. Extract ordered node_id sequence
            4. Filter out invalid paths (gaps too large, strand switches)
        """
        logger.info(f"Building UL paths from {len(ul_anchors)} anchors")
        
        # Group by UL read
        anchors_by_read: Dict[str, List[ULAnchor]] = defaultdict(list)
        for anchor in ul_anchors:
            anchors_by_read[anchor.ul_read_id].append(anchor)
        
        ul_paths = []
        
        for ul_read_id, anchors in anchors_by_read.items():
            # Sort by position on read
            sorted_anchors = sorted(anchors, key=lambda a: a.pos_on_read)
            
            # Extract node path
            path = []
            prev_anchor = None
            
            for anchor in sorted_anchors:
                # Check for strand consistency
                if prev_anchor and anchor.strand != prev_anchor.strand:
                    # Strand flip - start new path
                    if len(path) >= 2:
                        ul_paths.append(path)
                    path = [anchor.node_id]
                    prev_anchor = anchor
                    continue
                
                # Check for reasonable gap size
                if prev_anchor:
                    read_gap = anchor.pos_on_read - prev_anchor.pos_on_read
                    # Skip if gap is too large (indicates misalignment)
                    if read_gap > 100000:  # 100kb threshold
                        if len(path) >= 2:
                            ul_paths.append(path)
                        path = [anchor.node_id]
                        prev_anchor = anchor
                        continue
                
                # Add to path
                if not path or anchor.node_id != path[-1]:
                    path.append(anchor.node_id)
                
                prev_anchor = anchor
            
            # Add final path
            if len(path) >= 2:
                ul_paths.append(path)
        
        logger.info(f"Created {len(ul_paths)} UL paths from {len(anchors_by_read)} reads")
        return ul_paths
    
    def build_string_graph_from_dbg_and_ul(
        self,
        dbg: DBGGraph,
        ul_paths: List[ULPath]
    ) -> StringGraph:
        """
        Create string graph edges from DBG and UL paths.
        
        Args:
            dbg: De Bruijn graph with optional ML k annotations
            ul_paths: List of UL paths through the DBG
        
        Returns:
            String graph with edges aggregated from UL support
        
        Algorithm:
            1. For each UL path, create edges between consecutive nodes
            2. Aggregate support across all UL reads
            3. Calculate avg_recommended_k from node metadata
            4. Filter edges by minimum support and confidence
        """
        logger.info(f"Building string graph from DBG + {len(ul_paths)} UL paths")
        
        string_graph = StringGraph(dbg=dbg, ul_paths=ul_paths)
        
        # Track edge support: (from_node, to_node) -> (count, ul_read_ids)
        edge_support: Dict[Tuple[int, int], Tuple[int, Set[str]]] = defaultdict(
            lambda: (0, set())
        )
        
        # Process each UL path
        for path_idx, path in enumerate(ul_paths):
            ul_read_id = f"ul_path_{path_idx}"
            
            # Create edges between consecutive nodes
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                # Aggregate support
                count, read_ids = edge_support[(from_node, to_node)]
                edge_support[(from_node, to_node)] = (count + 1, read_ids | {ul_read_id})
        
        # Create string graph edges from aggregated support
        for (from_node, to_node), (count, read_ids) in edge_support.items():
            # Filter by minimum support
            if count < self.min_support:
                continue
            
            # Calculate average recommended k along this edge
            avg_k = self._calculate_avg_k(dbg, from_node, to_node)
            
            # Calculate path confidence
            confidence = self._calculate_confidence(dbg, from_node, to_node, count)
            
            # Filter by minimum confidence
            if confidence < self.min_path_confidence:
                continue
            
            # Create edge
            edge_id = self.next_edge_id
            self.next_edge_id += 1
            
            edge = StringGraphEdge(
                id=edge_id,
                from_node=from_node,
                to_node=to_node,
                support_count=count,
                ul_read_ids=read_ids,
                avg_recommended_k=avg_k,
                path_confidence=confidence
            )
            
            string_graph.edges[edge_id] = edge
            string_graph.out_edges[from_node].add(edge_id)
            string_graph.in_edges[to_node].add(edge_id)
        
        logger.info(
            f"Created string graph: {len(string_graph.edges)} edges "
            f"(filtered from {len(edge_support)} candidates)"
        )
        
        return string_graph
    
    def _calculate_avg_k(self, dbg: DBGGraph, from_node: int, to_node: int) -> Optional[float]:
        """Calculate average recommended k between two nodes."""
        if not dbg.ml_k_enabled:
            return None
        
        k_values = []
        
        for node_id in [from_node, to_node]:
            if node_id in dbg.nodes:
                node = dbg.nodes[node_id]
                if node.recommended_k is not None:
                    k_values.append(node.recommended_k)
        
        if k_values:
            return sum(k_values) / len(k_values)
        return None
    
    def _calculate_confidence(
        self,
        dbg: DBGGraph,
        from_node: int,
        to_node: int,
        support_count: int
    ) -> float:
        """
        Calculate confidence score for a string graph edge.
        
        Factors:
        - UL read support count
        - Coverage consistency between nodes
        - Whether edge is also in DBG (bonus)
        """
        confidence = 0.0
        
        # Base confidence from support count (saturates at 10 reads)
        confidence += min(support_count / 10.0, 1.0) * 0.5
        
        # Coverage consistency bonus
        if from_node in dbg.nodes and to_node in dbg.nodes:
            from_cov = dbg.nodes[from_node].coverage
            to_cov = dbg.nodes[to_node].coverage
            
            if from_cov > 0 and to_cov > 0:
                cov_ratio = min(from_cov, to_cov) / max(from_cov, to_cov)
                confidence += cov_ratio * 0.3
        
        # DBG edge bonus (if this connection already exists in DBG)
        dbg_edge_exists = self._check_dbg_edge_exists(dbg, from_node, to_node)
        if dbg_edge_exists:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _check_dbg_edge_exists(self, dbg: DBGGraph, from_node: int, to_node: int) -> bool:
        """Check if an edge exists in the DBG between two nodes."""
        if from_node not in dbg.out_edges:
            return False
        
        for edge_id in dbg.out_edges[from_node]:
            edge = dbg.edges[edge_id]
            if edge.to_id == to_node:
                return True
        
        return False


def build_string_graph_from_dbg_and_ul(
    dbg: DBGGraph,
    ul_anchors: List[ULAnchor],
    min_support: int = 2,
    min_path_confidence: float = 0.5
) -> StringGraph:
    """
    Convenience function to build string graph from DBG and UL anchors.
    
    Args:
        dbg: De Bruijn graph
        ul_anchors: UL read anchor points
        min_support: Minimum UL reads to retain an edge
        min_path_confidence: Minimum confidence score
    
    Returns:
        String graph with UL-supported long edges
    """
    builder = StringGraphBuilder(
        min_support=min_support,
        min_path_confidence=min_path_confidence
    )
    
    # Build paths from anchors
    ul_paths = builder.build_ul_paths_from_anchors(dbg, ul_anchors)
    
    # Build string graph from paths
    return builder.build_string_graph_from_dbg_and_ul(dbg, ul_paths)
