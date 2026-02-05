#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StrandWeaver v0.1.0

De Bruijn Graph (DBG) Engine for StrandWeaver.
- Accepts both OLC-derived artificial long reads and true long reads (HiFi, ONT)
- Consumes dynamic k-mer values from the ML regional-k module
- Builds a compacted de Bruijn graph (unitig representation)
- Annotates nodes with ML-recommended k values for regional genome complexity
- GPU-accelerated k-mer extraction and graph building (Apple Silicon MPS, CUDA)
- Linear path merging
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from pathlib import Path
import logging

from strandweaver.utils.hardware_management import GPUKmerExtractor, GPUGraphBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KmerGraphConfig:
    """Configuration for k-mer graph construction."""
    k: int  # K-mer size
    min_kmer_count: int = 2  # Minimum k-mer frequency to include
    canonical: bool = True  # Treat reverse complements as same k-mer
    
    def __post_init__(self):
        """Validate configuration."""
        if self.k < 7:
            raise ValueError(f"k must be >= 7, got {self.k}")
        if self.k % 2 == 0:
            logger.warning(f"Even k-mer size {self.k} may cause palindrome issues. Odd k recommended.")
        if self.min_kmer_count < 1:
            raise ValueError(f"min_kmer_count must be >= 1, got {self.min_kmer_count}")


# ============================================================================
# Core Data Structures (Unified DBG + K-mer Graph)
# ============================================================================

@dataclass
class KmerNode:
    """
    Node in the compacted de Bruijn graph.
    
    After compaction, nodes represent unitigs (maximal non-branching paths).
    Before compaction, each node is a (k-1)-mer.
    """
    id: int
    seq: str  # Sequence of the node (k-1-mer or compacted unitig)
    coverage: float  # Average k-mer coverage
    length: int  # Length of sequence in bases
    recommended_k: Optional[int] = None  # ML-suggested k-mer size for this region
    metadata: Dict = field(default_factory=dict)  # Additional regional information
    
    def __post_init__(self):
        """Validate node data."""
        if self.length != len(self.seq):
            logger.warning(f"Node {self.id}: length {self.length} != seq length {len(self.seq)}")
            self.length = len(self.seq)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class KmerEdge:
    """
    Edge in the de Bruijn graph.
    
    Represents an overlap between two (k-1)-mers, corresponding to a k-mer.
    """
    id: int
    from_id: int  # Source node ID
    to_id: int  # Target node ID
    coverage: float  # K-mer coverage (number of times this k-mer was seen)
    overlap_len: int = 0  # Length of overlap between nodes (k-1)
    
    def __hash__(self):
        """Make edges hashable for set operations."""
        return hash((self.from_id, self.to_id))
    
    def __eq__(self, other):
        """Compare edges by source and target."""
        if not isinstance(other, KmerEdge):
            return False
        return self.from_id == other.from_id and self.to_id == other.to_id


@dataclass
class KmerGraph:
    """
    Compacted de Bruijn graph representation.
    
    Uses adjacency lists for efficient graph traversal.
    """
    nodes: Dict[int, KmerNode] = field(default_factory=dict)
    edges: Dict[int, KmerEdge] = field(default_factory=dict)
    out_edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    in_edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    base_k: int = 31  # Global/base k-mer size used for graph construction
    ml_k_enabled: bool = False  # Whether ML regional-k annotation is enabled
    
    def add_node(self, node: KmerNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.out_edges:
            self.out_edges[node.id] = set()
        if node.id not in self.in_edges:
            self.in_edges[node.id] = set()
    
    def add_edge(self, edge: KmerEdge):
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
        self.out_edges[edge.from_id].add(edge.id)
        self.in_edges[edge.to_id].add(edge.id)
    
    def get_out_neighbors(self, node_id: int) -> List[int]:
        """Get all nodes reachable from this node."""
        return [self.edges[eid].to_id for eid in self.out_edges.get(node_id, [])]
    
    def get_in_neighbors(self, node_id: int) -> List[int]:
        """Get all nodes that reach this node."""
        return [self.edges[eid].from_id for eid in self.in_edges.get(node_id, [])]
    
    def is_linear(self, node_id: int) -> bool:
        """Check if node has exactly one in-edge and one out-edge (linear path)."""
        return len(self.out_edges.get(node_id, [])) == 1 and len(self.in_edges.get(node_id, [])) == 1
    
    def out_degree(self, node_id: int) -> int:
        """Number of outgoing edges."""
        return len(self.out_edges.get(node_id, []))
    
    def in_degree(self, node_id: int) -> int:
        """Number of incoming edges."""
        return len(self.in_edges.get(node_id, []))


# Compatibility aliases
DBGNode = KmerNode
DBGEdge = KmerEdge
DBGGraph = KmerGraph


# ============================================================================
# UL Read Overlay Structures (For Anchor Export)
# ============================================================================

@dataclass
class Anchor:
    """
    Exact k-mer match between read and graph node.
    
    Anchors are error-free regions that guide the alignment process.
    """
    read_start: int  # Start position in read
    read_end: int  # End position in read
    node_id: int  # Graph node ID
    node_start: int  # Start position in node sequence
    orientation: str = '+'  # '+' for forward, '-' for reverse complement
    
    @property
    def length(self) -> int:
        """Length of anchor in bases."""
        return self.read_end - self.read_start


# ============================================================================
# De Bruijn Graph Builder
# ============================================================================
    """
    Builder class for constructing de Bruijn graphs from long reads.
    
    Supports:
    - Variable-length reads (OLC-derived artificial longs, HiFi, ONT)
    - Dynamic k-mer size suggestions from ML module
    - Unitig compaction
    - Regional k-mer annotation
    - GPU-accelerated graph construction (Apple Silicon MPS / NVIDIA CUDA)
    """
    
    def __init__(self, base_k: int = 31, min_coverage: int = 2, use_gpu: bool = True):
        """
        Initialize DBG builder.
        
        Args:
            base_k: Base k-mer size for graph construction
            min_coverage: Minimum k-mer coverage to retain
            use_gpu: Enable GPU acceleration for k-mer extraction and graph building
        """
        self.base_k = base_k
        self.min_coverage = min_coverage
        self.next_node_id = 0
        self.next_edge_id = 0
        self.use_gpu = use_gpu
        
        # Initialize GPU components if requested
        self.gpu_extractor = None
        self.gpu_graph_builder = None
        if use_gpu:
            try:
                self.gpu_extractor = GPUKmerExtractor(k=base_k, use_gpu=True)
                if self.gpu_extractor.use_gpu:
                    logger.info(f"GPU k-mer extraction enabled (k={base_k})")
            except Exception as e:
                logger.warning(f"GPU k-mer extraction unavailable: {e}")
                self.gpu_extractor = None
            
            try:
                self.gpu_graph_builder = GPUGraphBuilder(k=base_k, use_gpu=True)
                if self.gpu_graph_builder.use_gpu:
                    logger.info(f"GPU graph construction enabled (k={base_k})")
            except Exception as e:
                logger.warning(f"GPU graph construction unavailable: {e}")
                self.gpu_graph_builder = None
    
    def build_dbg_from_long_reads(
        self,
        long_reads: List,  # List of SeqRead objects - duck typed for flexibility
        ml_k_model: Optional[Any] = None
    ) -> DBGGraph:
        """
        Build a de Bruijn graph from corrected long reads.
        
        Args:
            long_reads: List of SeqRead objects (corrected)
            ml_k_model: Optional ML model for regional k-mer recommendations
        
        Returns:
            DBGGraph with nodes, edges, and optional ML k annotations
        
        Algorithm:
            1. Extract k-mers from all reads and count occurrences
            2. Filter k-mers by minimum coverage
            3. Build initial graph: (k-1)-mers as nodes, k-mers as edges
            4. Compact linear paths into unitigs
            5. If ML model provided, annotate nodes with recommended k
        """
        logger.info(f"Building DBG from {len(long_reads)} long reads (k={self.base_k})")
        
        # Step 1: Extract and count k-mers
        kmer_counts = self._extract_kmers(long_reads)
        logger.info(f"Extracted {len(kmer_counts)} unique {self.base_k}-mers")
        
        # Step 2: Filter by coverage
        filtered_kmers = {
            kmer: count for kmer, count in kmer_counts.items()
            if count >= self.min_coverage
        }
        logger.info(f"Retained {len(filtered_kmers)}/{len(kmer_counts)} k-mers after coverage filter")
        
        # Step 3: Build raw graph
        graph = self._build_raw_graph(filtered_kmers)
        logger.info(f"Built raw graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Step 4: Compact into unitigs
        graph = self._compact_graph(graph)
        logger.info(f"Compacted to: {len(graph.nodes)} unitigs, {len(graph.edges)} edges")
        
        # Step 5: Annotate with ML k recommendations
        if ml_k_model is not None:
            graph = self._attach_regional_k(graph, ml_k_model, long_reads)
            graph.ml_k_enabled = True
            logger.info("Attached ML regional k recommendations")
        
        return graph
    
    def build_dbg_with_preprocessing(
        self,
        corrected_reads_path: Path,
        preprocessing_result: Any,  # PreprocessingResult from utilities
        ml_k_model: Optional[Any] = None
    ) -> DBGGraph:
        """
        Build DBG using dynamic k-mer from preprocessing pipeline.
        
        This method integrates with PreprocessingCoordinator output to:
        1. Use the preprocessing-selected k-mer size for DBG construction
        2. Load corrected reads from preprocessing output
        3. Apply the selected k-mer while retaining confidence metadata
        
        Args:
            corrected_reads_path: Path to corrected FASTQ from PreprocessingCoordinator
            preprocessing_result: PreprocessingResult with kmer_prediction
            ml_k_model: Optional ML model for additional k recommendations
            
        Returns:
            DBGGraph constructed with preprocessing-selected k-mer
            
        Example:
            >>> from strandweaver.assembly_utils.utilities import PreprocessingCoordinator
            >>> coordinator = PreprocessingCoordinator(technology="ont_r10")
            >>> result = coordinator.run_preprocessing(input_reads, output_dir)
            >>> builder = DeBruijnGraphBuilder()
            >>> dbg = builder.build_dbg_with_preprocessing(result.corrected_reads_path, result)
        """
        corrected_reads_path = Path(corrected_reads_path)
        
        logger.info("=" * 80)
        logger.info("Building DBG with Preprocessing-Selected K-mer")
        logger.info("=" * 80)
        
        # Extract k-mer from preprocessing result
        selected_k = preprocessing_result.kmer_prediction.dbg_k
        k_confidence = preprocessing_result.kmer_prediction.dbg_confidence
        
        logger.info(f"Using preprocessing-selected k-mer: {selected_k}")
        logger.info(f"  Confidence: {k_confidence:.2f}")
        logger.info(f"  Reasoning: {preprocessing_result.kmer_prediction.reasoning or 'N/A'}")
        logger.info(f"  Corrected reads: {corrected_reads_path}")
        
        # Temporarily set base_k to preprocessing selection
        original_k = self.base_k
        self.base_k = selected_k
        
        try:
            # Load corrected reads from FASTQ
            logger.info(f"\nLoading corrected reads from {corrected_reads_path}...")
            corrected_reads = self._load_fastq(corrected_reads_path)
            logger.info(f"Loaded {len(corrected_reads)} corrected reads")
            
            # Build DBG with preprocessing k-mer
            logger.info(f"\nBuilding DBG with k={selected_k}...")
            graph = self.build_dbg_from_long_reads(
                corrected_reads,
                ml_k_model=ml_k_model
            )
            
            # Attach preprocessing metadata
            graph.preprocessing_k = selected_k
            graph.preprocessing_k_confidence = k_confidence
            graph.preprocessing_stats = preprocessing_result.stats.to_dict() if hasattr(preprocessing_result.stats, 'to_dict') else {}
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ DBG construction complete")
            logger.info(f"  Graph nodes: {len(graph.nodes)}")
            logger.info(f"  Graph edges: {len(graph.edges)}")
            logger.info(f"  K-mer used: {selected_k} (from preprocessing)")
            logger.info("=" * 80)
            
            return graph
            
        finally:
            # Restore original k
            self.base_k = original_k
    
    def _load_fastq(self, fastq_path: Path) -> List:
        """
        Load sequences from FASTQ file.
        
        Returns list of objects with at least a 'sequence' attribute.
        """
        from collections import namedtuple
        
        SeqRead = namedtuple('SeqRead', ['read_id', 'sequence', 'quality'])
        reads = []
        
        try:
            with open(fastq_path, 'r') as f:
                while True:
                    header = f.readline().strip()
                    if not header:
                        break
                    
                    if not header.startswith('@'):
                        logger.warning(f"Skipping invalid record (no @): {header[:30]}")
                        continue
                    
                    sequence = f.readline().strip()
                    plus = f.readline().strip()
                    quality = f.readline().strip()
                    
                    read_id = header[1:].split()[0]  # Extract ID after @
                    reads.append(SeqRead(read_id, sequence, quality))
        
        except Exception as e:
            logger.error(f"Error loading FASTQ: {e}")
            raise
        
        return reads
    
    def _extract_kmers(self, reads: List) -> Dict[str, int]:
        """
        Extract k-mers from reads and count occurrences.
        
        Uses GPU acceleration if available for large read sets.
        """
        # Try GPU extraction for large read sets
        if self.gpu_extractor and len(reads) > 1000:
            try:
                sequences = [read.sequence.upper() for read in reads]
                kmer_counts_raw = self.gpu_extractor.extract_kmers(sequences)
                
                # Convert to canonical k-mers
                kmer_counts = Counter()
                for kmer, count in kmer_counts_raw.items():
                    canon = self._canonical_kmer(kmer)
                    kmer_counts[canon] += count
                
                logger.info(f"Extracted {len(kmer_counts)} unique {self.base_k}-mers from {len(reads)} reads (GPU)")
                return dict(kmer_counts)
            except Exception as e:
                logger.warning(f"GPU k-mer extraction failed, falling back to CPU: {e}")
        
        # CPU fallback (original implementation)
        kmer_counts = Counter()
        k = self.base_k
        
        for read in reads:
            seq = read.sequence.upper()
            
            # Extract all k-mers from this read
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                
                # Skip k-mers with ambiguous bases
                if 'N' in kmer:
                    continue
                
                # Use canonical k-mer (lexicographically smaller of fwd/rev)
                canon_kmer = self._canonical_kmer(kmer)
                kmer_counts[canon_kmer] += 1
        
        logger.info(f"Extracted {len(kmer_counts)} unique {k}-mers from {len(reads)} reads (CPU)")
        return dict(kmer_counts)
    
    def _canonical_kmer(self, kmer: str) -> str:
        """Return canonical k-mer (lex-min of kmer and reverse complement)."""
        rc = self._reverse_complement(kmer)
        return min(kmer, rc)
    
    def _reverse_complement(self, seq: str) -> str:
        """Return reverse complement of DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement.get(b, 'N') for b in reversed(seq))
    
    def _build_raw_graph(self, kmer_counts: Dict[str, int]) -> DBGGraph:
        """
        Build raw DBG from filtered k-mers.
        
        Node = (k-1)-mer (prefix/suffix of k-mer)
        Edge = k-mer (connects prefix to suffix)
        
        Uses GPU acceleration for large graphs (5000+ k-mers).
        """
        # Try GPU graph construction for large graphs
        if self.gpu_graph_builder and len(kmer_counts) >= 5000:
            try:
                logger.info(f"Using GPU-accelerated graph construction...")
                node_ids, edge_list, node_coverage = self.gpu_graph_builder.build_graph_from_kmers(
                    kmer_counts, 
                    min_coverage=self.min_coverage
                )
                
                # Convert to DBGGraph format
                graph = DBGGraph(base_k=self.base_k)
                
                # Create nodes
                for node_seq, node_id in node_ids.items():
                    node = DBGNode(
                        id=node_id,
                        seq=node_seq,
                        coverage=node_coverage.get(node_id, 0.0),
                        length=len(node_seq)
                    )
                    graph.nodes[node_id] = node
                
                # Create edges
                for from_id, to_id in edge_list:
                    edge_id = self.next_edge_id
                    self.next_edge_id += 1
                    
                    edge = DBGEdge(
                        id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        coverage=1.0,  # Coverage tracked on nodes
                        overlap_len=self.base_k - 1
                    )
                    graph.edges[edge_id] = edge
                    graph.out_edges[from_id].add(edge_id)
                    graph.in_edges[to_id].add(edge_id)
                
                self.next_node_id = len(graph.nodes)
                logger.info(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges (GPU)")
                return graph
                
            except Exception as e:
                logger.warning(f"GPU graph construction failed, falling back to CPU: {e}")
        
        # CPU fallback (original implementation)
        graph = DBGGraph(base_k=self.base_k)
        
        # Map (k-1)-mer -> node_id
        kmer_to_node: Dict[str, int] = {}
        
        for kmer, count in kmer_counts.items():
            # Get prefix and suffix (k-1)-mers
            prefix = kmer[:-1]  # First k-1 bases
            suffix = kmer[1:]   # Last k-1 bases
            
            # Create nodes if they don't exist
            if prefix not in kmer_to_node:
                node_id = self.next_node_id
                self.next_node_id += 1
                graph.nodes[node_id] = DBGNode(
                    id=node_id,
                    seq=prefix,
                    coverage=0.0,
                    length=len(prefix)
                )
                kmer_to_node[prefix] = node_id
            
            if suffix not in kmer_to_node:
                node_id = self.next_node_id
                self.next_node_id += 1
                graph.nodes[node_id] = DBGNode(
                    id=node_id,
                    seq=suffix,
                    coverage=0.0,
                    length=len(suffix)
                )
                kmer_to_node[suffix] = node_id
            
            # Create edge from prefix to suffix
            from_id = kmer_to_node[prefix]
            to_id = kmer_to_node[suffix]
            
            edge_id = self.next_edge_id
            self.next_edge_id += 1
            
            edge = DBGEdge(
                id=edge_id,
                from_id=from_id,
                to_id=to_id,
                coverage=float(count),
                overlap_len=self.base_k - 1
            )
            
            graph.edges[edge_id] = edge
            graph.out_edges[from_id].add(edge_id)
            graph.in_edges[to_id].add(edge_id)
            
            # Update node coverage
            graph.nodes[from_id].coverage += count
            graph.nodes[to_id].coverage += count
        
        # Normalize node coverage by degree
        for node_id, node in graph.nodes.items():
            total_degree = len(graph.out_edges[node_id]) + len(graph.in_edges[node_id])
            if total_degree > 0:
                node.coverage /= total_degree
        
        return graph
    
    def _compact_graph(self, graph: DBGGraph) -> DBGGraph:
        """
        Compact linear paths into unitigs.
        
        A linear path is a chain of nodes where each node has:
        - Exactly 1 incoming edge
        - Exactly 1 outgoing edge
        
        Merge these into single unitig nodes.
        """
        # Find linear paths
        visited = set()
        unitig_paths: List[List[int]] = []
        
        for node_id in graph.nodes:
            if node_id in visited:
                continue
            
            # Start a new path
            path = [node_id]
            visited.add(node_id)
            
            # Extend forward while linear
            current = node_id
            while True:
                out_edges = graph.out_edges[current]
                if len(out_edges) != 1:
                    break
                
                edge_id = next(iter(out_edges))
                edge = graph.edges[edge_id]
                next_node = edge.to_id
                
                # Check if next node is also linear
                if len(graph.in_edges[next_node]) != 1:
                    break
                
                if next_node in visited:
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            # Extend backward while linear
            current = node_id
            backward_path = []
            while True:
                in_edges = graph.in_edges[current]
                if len(in_edges) != 1:
                    break
                
                edge_id = next(iter(in_edges))
                edge = graph.edges[edge_id]
                prev_node = edge.from_id
                
                # Check if prev node is also linear
                if len(graph.out_edges[prev_node]) != 1:
                    break
                
                if prev_node in visited:
                    break
                
                backward_path.append(prev_node)
                visited.add(prev_node)
                current = prev_node
            
            # Combine backward + forward
            full_path = list(reversed(backward_path)) + path
            
            if len(full_path) > 1:
                unitig_paths.append(full_path)
        
        # Build compacted graph
        compacted = DBGGraph(base_k=graph.base_k, ml_k_enabled=graph.ml_k_enabled)
        
        # Create unitig nodes
        unitig_map: Dict[int, int] = {}  # old_node_id -> new_unitig_id
        
        for path in unitig_paths:
            # Merge nodes in path into single unitig
            unitig_id = self.next_node_id
            self.next_node_id += 1
            
            # Concatenate sequences (overlapping by k-1)
            seq_parts = [graph.nodes[path[0]].seq]
            for node_id in path[1:]:
                # Add last base of each subsequent node
                seq_parts.append(graph.nodes[node_id].seq[-1])
            
            unitig_seq = ''.join(seq_parts)
            
            # Average coverage across path
            avg_coverage = sum(graph.nodes[nid].coverage for nid in path) / len(path)
            
            unitig_node = DBGNode(
                id=unitig_id,
                seq=unitig_seq,
                coverage=avg_coverage,
                length=len(unitig_seq)
            )
            
            compacted.nodes[unitig_id] = unitig_node
            
            # Map all old nodes to this unitig
            for old_id in path:
                unitig_map[old_id] = unitig_id
        
        # Add singleton nodes (not in any path)
        for node_id, node in graph.nodes.items():
            if node_id not in unitig_map:
                compacted.nodes[node.id] = node
                unitig_map[node_id] = node.id
        
        # Rebuild edges between unitigs
        edge_set = set()  # Track unique (from, to) pairs
        
        for edge_id, edge in graph.edges.items():
            from_unitig = unitig_map[edge.from_id]
            to_unitig = unitig_map[edge.to_id]
            
            # Skip self-loops and duplicates
            if from_unitig == to_unitig:
                continue
            
            edge_key = (from_unitig, to_unitig)
            if edge_key in edge_set:
                continue
            
            edge_set.add(edge_key)
            
            new_edge_id = self.next_edge_id
            self.next_edge_id += 1
            
            new_edge = DBGEdge(
                id=new_edge_id,
                from_id=from_unitig,
                to_id=to_unitig,
                coverage=edge.coverage,
                overlap_len=self.base_k - 1
            )
            
            compacted.edges[new_edge_id] = new_edge
            compacted.out_edges[from_unitig].add(new_edge_id)
            compacted.in_edges[to_unitig].add(new_edge_id)
        
        return compacted
    
    def _attach_regional_k(
        self,
        graph: DBGGraph,
        ml_k_model,
        reads: List
    ) -> DBGGraph:
        """
        Annotate graph nodes with ML-recommended k values.
        
        Args:
            graph: Compacted DBG
            ml_k_model: ML model with get_k_for_region() method
            reads: Original reads for feature extraction
        
        Returns:
            Graph with nodes annotated with recommended_k
        """
        # For each node, extract features and query ML model
        for node_id, node in graph.nodes.items():
            # Extract node features
            features = {
                'coverage': node.coverage,
                'length': node.length,
                'gc_content': self._calculate_gc(node.seq),
                'out_degree': len(graph.out_edges[node_id]),
                'in_degree': len(graph.in_edges[node_id])
            }
            
            # Query ML model for recommended k
            try:
                if hasattr(ml_k_model, 'get_k_for_region'):
                    recommended_k = ml_k_model.get_k_for_region(features)
                    node.recommended_k = recommended_k
                else:
                    # Fallback: use base_k
                    node.recommended_k = graph.base_k
            except Exception as e:
                logger.warning(f"Failed to get ML k for node {node_id}: {e}")
                node.recommended_k = graph.base_k
        
        return graph
    
    def _calculate_gc(self, seq: str) -> float:
        """Calculate GC content of sequence."""
        if not seq:
            return 0.0
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq)


def build_dbg_from_long_reads(
    long_reads: List,  # List[SeqRead] - intentionally untyped for duck typing
    base_k: int = 31,
    min_coverage: int = 2,
    ml_k_model: Optional[Any] = None
) -> DBGGraph:
    """
    Convenience function to build DBG from long reads.
    
    Args:
        long_reads: List of SeqRead objects
        base_k: Base k-mer size
        min_coverage: Minimum k-mer coverage threshold
        ml_k_model: Optional ML model for regional k annotation
    
    Returns:
        Compacted DBG with optional ML k annotations
    """
    builder = DeBruijnGraphBuilder(base_k=base_k, min_coverage=min_coverage)
    return builder.build_dbg_from_long_reads(long_reads, ml_k_model=ml_k_model)
