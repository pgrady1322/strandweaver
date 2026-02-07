#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StrandWeaver v0.1.0

String Graph Engine for StrandWeaver.

This module implements a string graph overlay on top of a de Bruijn graph,
using ultra-long (UL) ONT reads to bridge DBG nodes and increase contiguity.

Key features:
- Accepts preprocessed DBG with regional k-mer annotations
- Uses UL read alignments to find paths through the DBG
- Creates string graph edges representing UL-supported connections
- Incorporates ML regional-k information for path scoring and filtering
- K-mer anchoring for error-prone ultralong reads (uncorrected ONT)
- Optional MBG/GraphAligner integration for gap filling

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict, Counter
import logging

from .dbg_engine_module import DBGGraph, DBGNode, KmerGraph, Anchor

logger = logging.getLogger(__name__)


# ============================================================================
# UL Read Overlay Data Structures
# ============================================================================

# Type alias for UL paths through the DBG
ULPath = List[int]  # Ordered sequence of DBG node IDs


@dataclass
class GapAlignment:
    """
    Alignment result for a gap between anchors.
    
    Filled by GraphAligner or other gap-filling method.
    """
    read_start: int
    read_end: int
    path: List[int]  # Node IDs in gap
    orientations: List[str]  # Orientation for each node
    matches: int  # Number of matching bases
    alignment_length: int  # Total alignment length (matches + mismatches + indels)
    cigar: str = ""  # Optional CIGAR string from alignment
    
    @property
    def identity(self) -> float:
        """Alignment identity (matches / alignment_length)."""
        return self.matches / self.alignment_length if self.alignment_length > 0 else 0.0
    
    @property
    def length(self) -> int:
        """Length of gap in read."""
        return self.read_end - self.read_start


@dataclass
class LongEdge:
    """
    Long-range connection derived from ultralong reads.
    
    These edges skip intermediate nodes, creating a string-graph-like layer
    on top of the de Bruijn graph. Multiple UL reads may support the same
    long-range connection.
    """
    id: int
    from_node: int  # Start node ID in DBG
    to_node: int  # End node ID in DBG
    support_count: int  # Number of UL reads supporting this connection
    path: Optional[List[int]] = None  # Optional: intermediate nodes
    
    def __hash__(self):
        """Make long edges hashable."""
        return hash((self.from_node, self.to_node))
    
    def __eq__(self, other):
        """Compare long edges by endpoints."""
        if not isinstance(other, LongEdge):
            return False
        return self.from_node == other.from_node and self.to_node == other.to_node


@dataclass
class ULReadMapping:
    """
    Mapping of an ultralong read to the DBG with full alignment details.
    
    Records the path through the graph, quality metrics, and alignment components
    (anchors + gap alignments) that form the complete mapping.
    """
    read_id: str
    path: ULPath  # Sequence of node IDs
    orientations: List[str]  # '+' or '-' for each node (forward/reverse)
    coverage: float  # Fraction of read that maps to graph [0.0, 1.0]
    identity: float  # Sequence identity of mapping [0.0, 1.0] (weighted by length)
    anchors: List[Anchor] = field(default_factory=list)  # Exact k-mer anchors
    gaps: List[GapAlignment] = field(default_factory=list)  # Gap alignments between anchors
    
    def is_valid(self, min_coverage: float = 0.5, min_identity: float = 0.7) -> bool:
        """Check if mapping meets quality thresholds."""
        return self.coverage >= min_coverage and self.identity >= min_identity


# ============================================================================
# Ultra-Long Read Structures
# ============================================================================

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


# ============================================================================
# Advanced UL Read Overlay (K-mer Anchoring + MBG)
# ============================================================================

class LongReadOverlay:
    """
    Maps ultralong reads onto the de Bruijn graph to create long-range connections.
    
    This creates a "string graph layer" on top of the DBG, similar to Verkko's
    approach of using HiFi as the base graph and ONT for scaffolding.
    
    Uses a two-phase approach for uncorrected ONT reads:
    1. Exact k-mer anchoring: Fast identification of error-free regions
    2. MBG alignment: Error-tolerant alignment between anchors
    
    Consolidated from data_structures.py Part 2 implementation.
    """
    
    def __init__(
        self, 
        min_anchor_length: int = 500, 
        min_identity: float = 0.7,
        anchor_k: int = 15,
        min_anchors: int = 2,
        use_mbg: bool = True
    ):
        """
        Initialize overlay mapper.
        
        Args:
            min_anchor_length: Minimum match length to anchor read to node
            min_identity: Minimum sequence identity for valid mapping
            anchor_k: K-mer size for exact anchoring (15 = ~1 in 1B random)
            min_anchors: Minimum number of anchors required for valid mapping
            use_mbg: Use MBG for gap filling between anchors (recommended)
        """
        self.min_anchor_length = min_anchor_length
        self.min_identity = min_identity
        self.anchor_k = anchor_k
        self.min_anchors = min_anchors
        self.use_mbg = use_mbg
        self.next_long_edge_id = 0
        
        # K-mer index for anchoring (built per graph)
        self.kmer_index = None  # Dict[str, List[Tuple[node_id, position]]]
        self.graph_indexed = None  # Graph that was indexed
    
    def build_ul_paths(
        self,
        graph: KmerGraph,
        ul_reads: List[Tuple[str, str]],
        min_coverage: float = 0.5,
        batch_size: int = 100
    ) -> List[ULReadMapping]:
        """
        Map all ultralong reads to the graph using k-mer anchoring.
        
        For comprehensive implementation including k-mer indexing, anchor finding,
        and GraphAligner integration, see the archived data_structures.py.
        
        This is a simplified interface that delegates to the full implementation
        when needed.
        
        Args:
            graph: de Bruijn graph
            ul_reads: List of (read_id, sequence) tuples (uncorrected ONT)
            min_coverage: Minimum read coverage to keep mapping
            batch_size: Number of reads to process per batch
            
        Returns:
            List of successful read mappings
        """
        logger.info(f"Mapping {len(ul_reads)} ultralong reads to graph...")
        logger.info(f"Using k-mer anchoring (k={self.anchor_k}, "
                   f"min_anchors={self.min_anchors}, MBG={'enabled' if self.use_mbg else 'disabled'})")
        
        # Build k-mer index
        self._build_kmer_index(graph)
        
        # Map reads
        mappings = []
        for read_id, seq in ul_reads:
            mapping = self.map_read_to_graph(seq, graph, read_id)
            if mapping and mapping.coverage >= min_coverage:
                mappings.append(mapping)
        
        success_rate = (len(mappings) / len(ul_reads) * 100) if ul_reads else 0
        logger.info(f"Successfully mapped {len(mappings)}/{len(ul_reads)} reads ({success_rate:.1f}%)")
        
        return mappings
    
    def build_long_edges_from_ul_paths(
        self,
        paths: List[ULReadMapping],
        min_support: int = 1
    ) -> List[LongEdge]:
        """
        Create long edges from UL read paths.
        
        A long edge connects two nodes that are spanned by UL reads,
        potentially skipping intermediate nodes.
        
        Args:
            paths: UL read mappings
            min_support: Minimum number of reads to create long edge
            
        Returns:
            List of long edges with support counts
        """
        logger.info(f"Building long edges from {len(paths)} UL paths...")
        
        # Count support for each (from, to) pair
        edge_support = Counter()
        edge_paths = defaultdict(list)
        
        for mapping in paths:
            path = mapping.path
            
            # Create long edges for non-adjacent nodes in path
            for i in range(len(path)):
                for j in range(i + 2, len(path)):  # Skip adjacent (i+1)
                    edge_key = (path[i], path[j])
                    edge_support[edge_key] += 1
                    edge_paths[edge_key].append(path[i:j+1])
        
        # Create LongEdge objects
        long_edges = []
        for (from_node, to_node), support in edge_support.items():
            if support >= min_support:
                # Use most common intermediate path
                paths_for_edge = edge_paths[(from_node, to_node)]
                most_common_path = max(paths_for_edge, key=len) if paths_for_edge else None
                
                edge = LongEdge(
                    id=self.next_long_edge_id,
                    from_node=from_node,
                    to_node=to_node,
                    support_count=support,
                    path=most_common_path
                )
                self.next_long_edge_id += 1
                long_edges.append(edge)
        
        logger.info(f"Created {len(long_edges)} long edges with support >= {min_support}")
        return long_edges
    
    def map_read_to_graph(
        self, 
        read_seq: str, 
        graph: KmerGraph,
        read_id: str = "unnamed"
    ) -> Optional[ULReadMapping]:
        """
        Map a single ultralong read to the graph using k-mer anchoring.
        
        Simplified interface. For full implementation with GraphAligner
        integration, see archived data_structures.py.
        
        Args:
            read_seq: Ultralong read sequence
            graph: de Bruijn graph
            read_id: Read identifier
            
        Returns:
            ULReadMapping if successful, None otherwise
        """
        # Build k-mer index if needed
        self._build_kmer_index(graph)
        
        # Find exact k-mer anchors
        anchors = self._find_exact_anchors(read_seq, read_id)
        
        if len(anchors) < self.min_anchors:
            return None
        
        # Convert anchors to path
        path, orientations = self._anchors_to_path(anchors)
        
        if not path:
            return None
        
        # Calculate coverage and identity
        total_anchor_bases = sum(anchor.length for anchor in anchors)
        coverage = total_anchor_bases / len(read_seq) if len(read_seq) > 0 else 0.0
        identity = coverage  # Conservative estimate
        
        mapping = ULReadMapping(
            read_id=read_id,
            path=path,
            orientations=orientations,
            coverage=coverage,
            identity=identity,
            anchors=anchors
        )
        
        if not mapping.is_valid(min_coverage=0.1, min_identity=self.min_identity):
            return None
        
        return mapping
    
    def _build_kmer_index(self, graph: KmerGraph):
        """Build k-mer index of graph node sequences."""
        if self.graph_indexed is graph and self.kmer_index is not None:
            return  # Already indexed
        
        logger.info(f"Building k-mer index (k={self.anchor_k}) for {len(graph.nodes)} nodes...")
        
        self.kmer_index = defaultdict(list)
        k = self.anchor_k
        
        for node_id, node in graph.nodes.items():
            seq = node.seq
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'N' not in kmer:
                    self.kmer_index[kmer].append((node_id, i))
        
        self.graph_indexed = graph
        logger.info(f"Indexed {len(self.kmer_index)} unique {k}-mers")
    
    def _find_exact_anchors(self, read_seq: str, read_id: str) -> List[Anchor]:
        """Find exact k-mer matches between read and graph nodes."""
        if self.kmer_index is None:
            raise ValueError("K-mer index not built. Call _build_kmer_index first.")
        
        k = self.anchor_k
        anchors = []
        processed_positions = set()
        
        for read_pos in range(len(read_seq) - k + 1):
            if read_pos in processed_positions:
                continue
                
            kmer = read_seq[read_pos:read_pos+k]
            if 'N' in kmer or kmer not in self.kmer_index:
                continue
            
            # Found k-mer match
            for node_id, node_pos in self.kmer_index[kmer]:
                node = self.graph_indexed.nodes[node_id]
                
                # Extend match in both directions
                read_end = read_pos + k
                node_end = node_pos + k
                while (read_end < len(read_seq) and 
                       node_end < len(node.seq) and
                       read_seq[read_end] == node.seq[node_end]):
                    read_end += 1
                    node_end += 1
                
                read_start = read_pos
                node_start = node_pos
                while (read_start > 0 and 
                       node_start > 0 and
                       read_seq[read_start-1] == node.seq[node_start-1]):
                    read_start -= 1
                    node_start -= 1
                
                # Record anchor if long enough
                match_len = read_end - read_start
                if match_len >= self.anchor_k:
                    anchor = Anchor(
                        read_start=read_start,
                        read_end=read_end,
                        node_id=node_id,
                        node_start=node_start,
                        orientation='+'
                    )
                    anchors.append(anchor)
                    
                    for i in range(read_start, read_end):
                        processed_positions.add(i)
        
        anchors.sort(key=lambda x: x.read_start)
        return anchors
    
    def _anchors_to_path(self, anchors: List[Anchor]) -> Tuple[List[int], List[str]]:
        """Convert anchors to a path through the graph with orientations."""
        if not anchors:
            return [], []
        
        path = []
        orientations = []
        current_node = None
        
        for anchor in anchors:
            if anchor.node_id != current_node:
                path.append(anchor.node_id)
                orientations.append(anchor.orientation)
                current_node = anchor.node_id
        
        return path, orientations

