#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

De Bruijn Graph (DBG) Engine — compacted DBG construction from long reads
with ML-recommended regional k-mer values and GPU acceleration.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from pathlib import Path
import logging
import math

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
            raise ValueError(
                f"Even k-mer size {self.k} creates palindrome ambiguity in "
                f"canonical de Bruijn graphs. Use an odd k (e.g. {self.k + 1})."
            )
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


# Aliases
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

class DeBruijnGraphBuilder:
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
        ml_k_model: Optional[Any] = None,
        hic_phase_info: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None
    ) -> DBGGraph:
        """
        Build a de Bruijn graph from corrected long reads.
        
        Args:
            long_reads: List of SeqRead objects (corrected)
            ml_k_model: Optional ML model for regional k-mer recommendations
            hic_phase_info: Optional Dict[node_id] -> phase assignment ('A', 'B', 'ambiguous')
                           from Hi-C phasing. Used for phasing-aware bubble resolution.
            ul_support_map: Optional Dict[edge_id] -> ul_support_count. Used to
                           weight bubble arms by ultra-long read support.
            ai_annotations: Optional Dict[edge_id] -> AI quality score (0-1).
                           Used to weight bubble arms by EdgeWarden confidence.
        
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
        
        # Step 4b: Trim error tips (short low-coverage dead-end unitigs)
        # Uses node coverage from k-mer counting — no read re-mapping needed
        pre_cleanup_nodes = len(graph.nodes)
        graph = self._trim_tips(graph)
        
        # Step 4c: Pop error bubbles (collapse near-identical parallel paths)
        # Heterozygous bubbles (real variation) are PROTECTED
        # When phasing data is available, allelic bubbles are identified by phase
        # assignment rather than just sequence difference count.
        graph = self._pop_error_bubbles(
            graph,
            hic_phase_info=hic_phase_info,
            ul_support_map=ul_support_map,
            ai_annotations=ai_annotations
        )
        
        # Step 4d: Re-compact if cleanup removed anything (creates new linear chains)
        if len(graph.nodes) < pre_cleanup_nodes:
            graph = self._compact_graph(graph)
            logger.info(f"Re-compacted to: {len(graph.nodes)} unitigs, {len(graph.edges)} edges")
        
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
                for from_kmer, to_kmer, edge_cov in edge_list:
                    from_id = node_ids.get(from_kmer)
                    to_id = node_ids.get(to_kmer)
                    if from_id is None or to_id is None:
                        continue  # Skip edges with missing nodes
                    
                    edge_id = self.next_edge_id
                    self.next_edge_id += 1
                    
                    edge = DBGEdge(
                        id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        coverage=float(edge_cov),
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
        
        # Track which (prefix, suffix) edges we've already added to avoid duplicates
        seen_edges: set = set()
        
        for kmer, count in kmer_counts.items():
            # CRITICAL FIX: Add edges for BOTH orientations of each canonical k-mer.
            # Canonical k-mers only preserve one strand's directionality.
            # E.g., if genome has ...GTAC→TACG... but GTACG is canonicalized to
            # CGTAC (its RC), we'd only get edge CGTA→GTAC, missing GTAC→TACG.
            # By adding both orientations, both strands are represented in the graph.
            rc_kmer = self._reverse_complement(kmer)
            orientations = [kmer] if kmer == rc_kmer else [kmer, rc_kmer]
            
            for oriented_kmer in orientations:
                # Get prefix and suffix (k-1)-mers
                prefix = oriented_kmer[:-1]  # First k-1 bases
                suffix = oriented_kmer[1:]   # Last k-1 bases
                
                # Skip if we've already added this exact edge
                edge_key = (prefix, suffix)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                
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
        
        logger.info(f"Built bidirectional graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges "
                     f"from {len(kmer_counts)} canonical k-mers")
        
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
            
            # Concatenate sequences: adjacent nodes overlap by k-2 characters
            # (the suffix of one (k-1)-mer / prefix of the next (k-1)-mer).
            # For raw nodes len=k-1, so seq[(k-2):] == seq[-1:] (1 char). 
            # For unitig nodes (re-compaction), seq[(k-2):] correctly 
            # preserves the full non-overlapping suffix.
            overlap = self.base_k - 2
            seq_parts = [graph.nodes[path[0]].seq]
            for node_id in path[1:]:
                seq_parts.append(graph.nodes[node_id].seq[overlap:])
            
            unitig_seq = ''.join(seq_parts)
            
            # Average coverage across path
            avg_coverage = sum(graph.nodes[nid].coverage for nid in path) / len(path)
            
            unitig_node = DBGNode(
                id=unitig_id,
                seq=unitig_seq,
                coverage=avg_coverage,
                length=len(unitig_seq)
            )
            
            # Use add_node() to properly initialize edge dictionaries
            compacted.add_node(unitig_node)
            
            # Map all old nodes to this unitig
            for old_id in path:
                unitig_map[old_id] = unitig_id
        
        # Add singleton nodes (not in any path)
        for node_id, node in graph.nodes.items():
            if node_id not in unitig_map:
                # Use add_node() to properly initialize edge dictionaries
                compacted.add_node(node)
                unitig_map[node_id] = node.id
        
        # Rebuild edges between unitigs
        edge_set = set()  # Track unique (from, to) pairs
        self_loop_count = 0
        inter_unitig_count = 0
        
        for edge_id, edge in graph.edges.items():
            from_unitig = unitig_map[edge.from_id]
            to_unitig = unitig_map[edge.to_id]
            
            # Skip self-loops (internal path edges)
            if from_unitig == to_unitig:
                self_loop_count += 1
                continue
            
            edge_key = (from_unitig, to_unitig)
            if edge_key in edge_set:
                continue
            
            edge_set.add(edge_key)
            inter_unitig_count += 1
            
            new_edge_id = self.next_edge_id
            self.next_edge_id += 1
            
            new_edge = DBGEdge(
                id=new_edge_id,
                from_id=from_unitig,
                to_id=to_unitig,
                coverage=edge.coverage,
                overlap_len=self.base_k - 1
            )
            
            # Use add_edge() to properly update edge dictionaries
            compacted.add_edge(new_edge)
        
        logger.info(f"  Compaction: {self_loop_count} self-loops filtered, {inter_unitig_count} inter-unitig edges preserved")
        
        return compacted
    
    # ====================================================================
    # Graph Simplification: Tip Trimming & Bubble Popping
    # ====================================================================
    
    def _trim_tips(self, graph: DBGGraph) -> DBGGraph:
        """
        Remove short, low-coverage dead-end unitigs (tips).
        
        Tips are short chains ending at dead-ends, typically caused by
        sequencing errors that create divergent k-mers. High-coverage
        tips (real contig/chromosome ends) are PRESERVED.
        
        Coverage comes from k-mer counting during graph construction —
        no read re-mapping is needed.
        
        Args:
            graph: Compacted DBG graph
            
        Returns:
            Graph with error-derived tips removed
        """
        max_tip_length = self.base_k * 2  # Tips shorter than 2*k bases
        
        # Calculate median coverage for relative threshold
        coverages = [n.coverage for n in graph.nodes.values() if n.coverage > 0]
        if not coverages:
            logger.info("  Tip trimming: no nodes with coverage, skipping")
            return graph
        
        sorted_cov = sorted(coverages)
        median_coverage = sorted_cov[len(sorted_cov) // 2]
        # Tips below 15% of median coverage are error-derived;
        # floor of 1.5 prevents removal when overall coverage is very low
        coverage_threshold = max(median_coverage * 0.15, 1.5)
        
        tips_removed = 0
        nodes_to_remove: Set[int] = set()
        edges_to_remove: Set[int] = set()
        
        for node_id in list(graph.nodes.keys()):
            if node_id in nodes_to_remove:
                continue
            
            in_deg = len(graph.in_edges.get(node_id, set()))
            out_deg = len(graph.out_edges.get(node_id, set()))
            
            # Isolated orphan (no edges at all): evaluate directly as tip
            if in_deg == 0 and out_deg == 0:
                node = graph.nodes[node_id]
                if node.length <= max_tip_length and node.coverage < coverage_threshold:
                    nodes_to_remove.add(node_id)
                    tips_removed += 1
                continue
            
            # Must be a dead-end: one side open, other side connected
            if not ((in_deg == 0 and out_deg > 0) or (out_deg == 0 and in_deg > 0)):
                continue
            
            # Walk from dead-end toward nearest branch point
            tip_nodes, tip_length = self._walk_tip(graph, node_id, nodes_to_remove)
            
            if tip_length > max_tip_length:
                continue  # Too long — likely a real contig end
            
            # Coverage check: protect high-coverage tips (real chromosome ends)
            avg_coverage = sum(graph.nodes[n].coverage for n in tip_nodes) / len(tip_nodes)
            if avg_coverage >= coverage_threshold:
                continue  # High coverage relative to graph median — keep
            
            # Mark tip chain for removal
            for nid in tip_nodes:
                nodes_to_remove.add(nid)
                for eid in graph.out_edges.get(nid, set()):
                    edges_to_remove.add(eid)
                for eid in graph.in_edges.get(nid, set()):
                    edges_to_remove.add(eid)
            tips_removed += 1
        
        # Perform removal
        for eid in edges_to_remove:
            if eid in graph.edges:
                edge = graph.edges[eid]
                del graph.edges[eid]
                graph.out_edges[edge.from_id].discard(eid)
                graph.in_edges[edge.to_id].discard(eid)
        
        for nid in nodes_to_remove:
            if nid in graph.nodes:
                del graph.nodes[nid]
                graph.out_edges.pop(nid, None)
                graph.in_edges.pop(nid, None)
        
        logger.info(f"  Tip trimming: removed {tips_removed} tips "
                     f"({len(nodes_to_remove)} nodes, {len(edges_to_remove)} edges)")
        logger.info(f"    Thresholds: length <= {max_tip_length}bp, "
                     f"coverage < {coverage_threshold:.1f}x (median={median_coverage:.1f}x)")
        
        return graph
    
    def _walk_tip(self, graph: DBGGraph, start_id: int,
                   already_marked: Set[int]) -> Tuple[List[int], int]:
        """
        Walk from a dead-end node along a linear chain toward a branch point.
        
        Returns:
            (node_ids_in_tip, total_sequence_length)
        """
        chain = [start_id]
        total_length = graph.nodes[start_id].length
        
        in_deg = len(graph.in_edges.get(start_id, set()))
        out_deg = len(graph.out_edges.get(start_id, set()))
        
        # Determine walk direction (away from dead-end, toward interior)
        if in_deg == 0 and out_deg > 0:
            direction = 'forward'
        elif out_deg == 0 and in_deg > 0:
            direction = 'backward'
        else:
            return chain, total_length
        
        current = start_id
        visited = {start_id}
        
        while True:
            if direction == 'forward':
                out_edges = graph.out_edges.get(current, set())
                if len(out_edges) != 1:
                    break  # Fork or second dead-end
                edge_id = next(iter(out_edges))
                if edge_id not in graph.edges:
                    break
                next_id = graph.edges[edge_id].to_id
                # Stop if next node has multiple inputs (convergence/branch)
                if len(graph.in_edges.get(next_id, set())) != 1:
                    break
            else:  # backward
                in_edges = graph.in_edges.get(current, set())
                if len(in_edges) != 1:
                    break
                edge_id = next(iter(in_edges))
                if edge_id not in graph.edges:
                    break
                next_id = graph.edges[edge_id].from_id
                if len(graph.out_edges.get(next_id, set())) != 1:
                    break
            
            if next_id in visited or next_id in already_marked:
                break
            if next_id not in graph.nodes:
                break
            
            chain.append(next_id)
            visited.add(next_id)
            # Sequence length: add node length minus k-2 overlap with previous
            total_length += graph.nodes[next_id].length - (self.base_k - 2)
            current = next_id
            
            # Check if we reached another dead-end (isolated fragment)
            cur_in = len(graph.in_edges.get(current, set()))
            cur_out = len(graph.out_edges.get(current, set()))
            if direction == 'forward' and cur_out == 0:
                break  # End of isolated chain
            if direction == 'backward' and cur_in == 0:
                break
        
        return chain, total_length
    
    def _pop_error_bubbles(
        self,
        graph: DBGGraph,
        hic_phase_info: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None,
        error_rate: Optional[float] = None
    ) -> DBGGraph:
        """
        Collapse short parallel paths (bubbles) caused by sequencing errors.
        
        A bubble is two paths between the same source and sink nodes.
        
        Decision logic (multi-signal, phasing-aware):
          1. If Hi-C phasing is available and the two arms have DIFFERENT
             phase assignments (A vs B) → PROTECT (allelic variation)
          2. If arms differ by <= max_error_differences substitutions
             → collapse to the higher-support arm (coverage + UL + AI)
          3. If arms differ by more → PROTECT (potential heterozygosity)
        
        The max_error_differences threshold is now technology-aware (G15):
          max_diffs = max(2, ceil(error_rate × arm_length × 3))
        This allows ONT (10% error) to tolerate more substitutions while
        remaining conservative for HiFi (0.1% error).
        
        Support scoring when phasing/UL/AI data is available:
          support = 0.5 * coverage_norm + 0.3 * ul_support_norm + 0.2 * ai_score
        Falls back to pure coverage when no auxiliary signals are available.
        
        Args:
            graph: DBG graph (ideally after tip trimming)
            hic_phase_info: Optional Dict[node_id] -> phase ('A', 'B', 'ambiguous')
            ul_support_map: Optional Dict[edge_id] -> UL read support count
            ai_annotations: Optional Dict[edge_id] -> AI quality score (0-1)
            error_rate: Optional sequencing error rate (0-1). If None, defaults
                        to 0.001 (HiFi). Pass ~0.05 for ONT R10 or ~0.10 for ONT R9.
            
        Returns:
            Graph with error bubbles collapsed, allelic/heterozygous bubbles intact
        """
        max_bubble_arm_length = self.base_k * 10
        # Technology-aware error threshold (G15 fix):
        # HiFi (0.1% error): ~2 diffs per arm (conservative)
        # ONT R10 (5% error): ~8 diffs in 50bp arm
        # ONT R9 (10% error): ~15 diffs in 50bp arm
        # Floor of 2 prevents collapsing 0-diff arms as "too many"
        tech_error_rate = error_rate if error_rate is not None else 0.001
        
        has_phasing = hic_phase_info is not None and len(hic_phase_info) > 0
        has_ul = ul_support_map is not None and len(ul_support_map) > 0
        has_ai = ai_annotations is not None and len(ai_annotations) > 0
        
        if has_phasing:
            logger.info("  Bubble resolution: phasing-aware mode (Hi-C phase data available)")
        if has_ul:
            logger.info("  Bubble resolution: UL support weighting enabled")
        if has_ai:
            logger.info("  Bubble resolution: AI edge score weighting enabled")
        
        bubbles_found = 0
        bubbles_popped = 0
        bubbles_protected = 0
        bubbles_phased = 0  # Protected specifically by phase signal
        nodes_to_remove: Set[int] = set()
        edges_to_remove: Set[int] = set()
        
        # Check each potential source node (out_degree >= 2)
        for source_id in list(graph.nodes.keys()):
            out_edge_ids = list(graph.out_edges.get(source_id, set()))
            if len(out_edge_ids) < 2:
                continue
            
            # Check all pairs of outgoing edges for bubble structures
            for i in range(len(out_edge_ids)):
                for j in range(i + 1, len(out_edge_ids)):
                    if out_edge_ids[i] not in graph.edges or out_edge_ids[j] not in graph.edges:
                        continue
                    
                    arm_a_start = graph.edges[out_edge_ids[i]].to_id
                    arm_b_start = graph.edges[out_edge_ids[j]].to_id
                    
                    # Skip if either arm already marked for removal
                    if arm_a_start in nodes_to_remove or arm_b_start in nodes_to_remove:
                        continue
                    
                    # Walk each arm to find if they converge at the same sink
                    path_a, sink_a = self._walk_bubble_arm(
                        graph, arm_a_start, max_bubble_arm_length, nodes_to_remove)
                    path_b, sink_b = self._walk_bubble_arm(
                        graph, arm_b_start, max_bubble_arm_length, nodes_to_remove)
                    
                    if sink_a is None or sink_b is None:
                        continue  # Arm doesn't converge cleanly
                    if sink_a != sink_b:
                        continue  # Different sinks — not a bubble
                    
                    bubbles_found += 1
                    
                    # Compare sequences to decide: pop or protect
                    seq_a = self._get_path_sequence(graph, path_a)
                    seq_b = self._get_path_sequence(graph, path_b)
                    n_differences = self._count_differences(seq_a, seq_b)
                    
                    if n_differences is None:
                        # Lengths too different — treat as complex, protect
                        bubbles_protected += 1
                        continue
                    
                    # ── Phase-aware check: if arms belong to different
                    #    haplotypes, ALWAYS protect regardless of sequence
                    #    similarity (even 1-SNP allelic bubbles) ──
                    if has_phasing:
                        phase_a = self._get_arm_phase(
                            path_a, hic_phase_info)
                        phase_b = self._get_arm_phase(
                            path_b, hic_phase_info)
                        
                        if (phase_a in ('A', 'B') and
                                phase_b in ('A', 'B') and
                                phase_a != phase_b):
                            # Confirmed allelic — different haplotypes
                            bubbles_protected += 1
                            bubbles_phased += 1
                            continue
                    
                    # Compute technology-aware max error differences
                    # based on arm length and error rate (G15 fix)
                    arm_len = max(len(seq_a) if seq_a else 0,
                                 len(seq_b) if seq_b else 0)
                    max_error_differences = max(
                        2, math.ceil(tech_error_rate * arm_len * 3)
                    )
                    
                    if n_differences <= max_error_differences:
                        # ── Error bubble → pop (keep higher-support arm) ──
                        support_a = self._calculate_arm_support(
                            path_a, graph, out_edge_ids[i],
                            ul_support_map, ai_annotations)
                        support_b = self._calculate_arm_support(
                            path_b, graph, out_edge_ids[j],
                            ul_support_map, ai_annotations)
                        
                        remove_path = path_b if support_a >= support_b else path_a
                        keep_idx = i if support_a >= support_b else j
                        remove_idx = j if support_a >= support_b else i
                        
                        for nid in remove_path:
                            nodes_to_remove.add(nid)
                            for eid in graph.out_edges.get(nid, set()):
                                edges_to_remove.add(eid)
                            for eid in graph.in_edges.get(nid, set()):
                                edges_to_remove.add(eid)
                        # Also remove the source→removed_arm edge
                        edges_to_remove.add(out_edge_ids[remove_idx])
                        
                        bubbles_popped += 1
                    else:
                        # ── Heterozygous bubble → PROTECT ──
                        bubbles_protected += 1
        
        # Perform removal
        for eid in edges_to_remove:
            if eid in graph.edges:
                edge = graph.edges[eid]
                del graph.edges[eid]
                graph.out_edges[edge.from_id].discard(eid)
                graph.in_edges[edge.to_id].discard(eid)
        
        for nid in nodes_to_remove:
            if nid in graph.nodes:
                del graph.nodes[nid]
                graph.out_edges.pop(nid, None)
                graph.in_edges.pop(nid, None)
        
        phase_detail = f", {bubbles_phased} by Hi-C phase" if bubbles_phased else ""
        logger.info(f"  Bubble popping: found {bubbles_found} bubbles — "
                     f"popped {bubbles_popped} (error), "
                     f"protected {bubbles_protected} (heterozygous{phase_detail})")
        
        return graph
    
    def _get_arm_phase(
        self,
        path_node_ids: List[int],
        hic_phase_info: Dict
    ) -> str:
        """
        Determine the consensus Hi-C phase for a bubble arm.
        
        Returns 'A', 'B', or 'ambiguous' based on majority vote of
        phase-assigned nodes in the arm.
        """
        phase_counts: Dict[str, int] = defaultdict(int)
        for nid in path_node_ids:
            phase = hic_phase_info.get(nid, 'ambiguous')
            if phase in ('A', 'B'):
                phase_counts[phase] += 1
        
        if not phase_counts:
            return 'ambiguous'
        
        best_phase = max(phase_counts, key=phase_counts.get)
        # Require a clear majority (>60% of phased nodes agree)
        total_phased = sum(phase_counts.values())
        if phase_counts[best_phase] / total_phased > 0.6:
            return best_phase
        return 'ambiguous'
    
    def _calculate_arm_support(
        self,
        path_node_ids: List[int],
        graph: DBGGraph,
        entry_edge_id: int,
        ul_support_map: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None
    ) -> float:
        """
        Calculate multi-signal support score for a bubble arm.
        
        Combines:
          - Coverage (always available): 50% weight
          - UL read support (if available): 30% weight
          - AI edge quality (if available): 20% weight
        
        Falls back to pure coverage when auxiliary signals are absent.
        """
        # Coverage component (always available)
        cov = sum(graph.nodes[n].coverage for n in path_node_ids) / max(len(path_node_ids), 1)
        
        # Normalize coverage to 0-1 scale using graph median
        coverages = [n.coverage for n in graph.nodes.values()]
        median_cov = sorted(coverages)[len(coverages) // 2] if coverages else 1.0
        cov_norm = min(cov / max(median_cov, 1.0), 2.0) / 2.0  # Cap at 1.0
        
        # UL support component
        ul_norm = 0.0
        has_ul = ul_support_map is not None and len(ul_support_map) > 0
        if has_ul:
            # Sum UL support for edges in this arm
            ul_total = 0
            for nid in path_node_ids:
                for eid in graph.in_edges.get(nid, set()):
                    ul_total += ul_support_map.get(eid, 0)
                for eid in graph.out_edges.get(nid, set()):
                    ul_total += ul_support_map.get(eid, 0)
            # Also check the entry edge
            ul_total += ul_support_map.get(entry_edge_id, 0)
            ul_norm = min(ul_total / 5.0, 1.0)  # Normalize: 5+ UL reads = max
        
        # AI edge quality component
        ai_norm = 0.5  # Default neutral
        has_ai = ai_annotations is not None and len(ai_annotations) > 0
        if has_ai:
            ai_scores = []
            # Entry edge score
            entry_score = ai_annotations.get(entry_edge_id)
            if entry_score is not None:
                ai_scores.append(entry_score if isinstance(entry_score, (int, float)) else getattr(entry_score, 'score_true', 0.5))
            # Internal edge scores
            for nid in path_node_ids:
                for eid in graph.out_edges.get(nid, set()):
                    score = ai_annotations.get(eid)
                    if score is not None:
                        ai_scores.append(score if isinstance(score, (int, float)) else getattr(score, 'score_true', 0.5))
            if ai_scores:
                ai_norm = sum(ai_scores) / len(ai_scores)
        
        # Weighted combination — adjust weights based on available signals
        if has_ul and has_ai:
            return 0.5 * cov_norm + 0.3 * ul_norm + 0.2 * ai_norm
        elif has_ul:
            return 0.6 * cov_norm + 0.4 * ul_norm
        elif has_ai:
            return 0.7 * cov_norm + 0.3 * ai_norm
        else:
            return cov_norm  # Pure coverage fallback
    
    def _walk_bubble_arm(
        self, graph: DBGGraph, start_id: int,
        max_length: int, excluded: Set[int]
    ) -> Tuple[List[int], Optional[int]]:
        """
        Walk from a bubble arm start along a linear chain until convergence.
        
        Returns:
            (path_node_ids, sink_node_id) or (path, None) if no convergence
        """
        path = [start_id]
        total_length = graph.nodes[start_id].length if start_id in graph.nodes else 0
        current = start_id
        
        while total_length <= max_length:
            out_edges = graph.out_edges.get(current, set())
            if len(out_edges) != 1:
                break  # Fork or dead-end — arm doesn't continue linearly
            
            edge_id = next(iter(out_edges))
            if edge_id not in graph.edges:
                break
            
            next_id = graph.edges[edge_id].to_id
            if next_id in excluded or next_id not in graph.nodes:
                break
            
            # If next node has multiple incoming edges, it's the sink
            if len(graph.in_edges.get(next_id, set())) >= 2:
                return path, next_id
            
            # Next node must be linear (in=1) to continue the arm
            if len(graph.in_edges.get(next_id, set())) != 1:
                break
            
            path.append(next_id)
            total_length += graph.nodes[next_id].length - (self.base_k - 2)
            current = next_id
        
        # Check if current node's sole out-neighbor is a convergence sink
        out_edges = graph.out_edges.get(current, set())
        if len(out_edges) == 1:
            edge_id = next(iter(out_edges))
            if edge_id in graph.edges:
                next_id = graph.edges[edge_id].to_id
                if len(graph.in_edges.get(next_id, set())) >= 2:
                    return path, next_id
        
        return path, None  # No convergence found
    
    def _get_path_sequence(self, graph: DBGGraph, path_node_ids: List[int]) -> str:
        """Concatenate sequences along a path, trimming k-2 overlaps."""
        if not path_node_ids:
            return ""
        
        first_node = graph.nodes.get(path_node_ids[0])
        if not first_node:
            return ""
        
        seq = first_node.seq
        overlap = self.base_k - 2  # Adjacent nodes share k-2 chars
        
        for nid in path_node_ids[1:]:
            node = graph.nodes.get(nid)
            if node and len(node.seq) > overlap:
                seq += node.seq[overlap:]
        
        return seq
    
    def _count_differences(self, seq_a: str, seq_b: str) -> Optional[int]:
        """
        Count differences between two bubble arm sequences.
        
        Returns None if sequences are too different in length (>5%),
        indicating a complex structural difference rather than simple errors.
        """
        if not seq_a or not seq_b:
            return None
        
        len_a, len_b = len(seq_a), len(seq_b)
        
        # Length difference > 5% → not a simple substitution bubble
        if abs(len_a - len_b) / max(len_a, len_b) > 0.05:
            return None
        
        # Same length: fast mismatch count
        if len_a == len_b:
            return sum(1 for a, b in zip(seq_a, seq_b) if a != b)
        
        # Different lengths: edit distance (limit to short seqs for performance)
        if max(len_a, len_b) > 500:
            return None
        
        return self._edit_distance(seq_a, seq_b)
    
    def _edit_distance(self, seq_a: str, seq_b: str) -> int:
        """Compute edit distance with two-row DP (memory-efficient)."""
        m, n = len(seq_a), len(seq_b)
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        
        return prev[n]
    
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
    ml_k_model: Optional[Any] = None,
    hic_phase_info: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None
) -> DBGGraph:
    """
    Convenience function to build DBG from long reads.
    
    Args:
        long_reads: List of SeqRead objects
        base_k: Base k-mer size
        min_coverage: Minimum k-mer coverage threshold
        ml_k_model: Optional ML model for regional k annotation
        hic_phase_info: Optional Hi-C phase assignments for phasing-aware bubble resolution
        ul_support_map: Optional UL read support counts for multi-signal bubble resolution
        ai_annotations: Optional AI edge quality scores for multi-signal bubble resolution
    
    Returns:
        Compacted DBG with optional ML k annotations
    """
    builder = DeBruijnGraphBuilder(base_k=base_k, min_coverage=min_coverage)
    return builder.build_dbg_from_long_reads(
        long_reads, ml_k_model=ml_k_model,
        hic_phase_info=hic_phase_info,
        ul_support_map=ul_support_map,
        ai_annotations=ai_annotations
    )

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
