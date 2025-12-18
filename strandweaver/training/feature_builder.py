"""
Training Feature Builder

Converts ground-truth labels and assembly graphs into ML-ready training examples.
Extracts features for all five AI subsystems:
1. Overlap Classifier (edge features)
2. GNN Path Predictor (graph tensors)
3. Diploid Disentangler (node features + Hi-C)
4. UL Routing Model (path features)
5. SV Detection AI (SV features)

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
#                    OVERLAP CLASSIFIER FEATURES
# ============================================================================

@dataclass
class OverlapFeatures:
    """
    Features for edge/overlap classification.
    
    Matches the 13+ features used in overlap_ai_filter.py:
    - Overlap length
    - Identity percentage
    - Coverage ratio
    - GC content similarity
    - K-mer consistency
    - Position consistency
    - Strand agreement
    - Quality scores
    - Repeat signals
    - etc.
    """
    # Basic overlap metrics
    overlap_length: int = 0
    identity: float = 0.0
    
    # Coverage and depth
    source_coverage: float = 0.0
    target_coverage: float = 0.0
    coverage_ratio: float = 0.0
    
    # Sequence composition
    gc_content_source: float = 0.0
    gc_content_target: float = 0.0
    gc_similarity: float = 0.0
    
    # K-mer consistency
    kmer_consistency: float = 0.0
    shared_kmers: int = 0
    
    # Position and orientation
    position_distance: int = 0
    same_strand: bool = True
    
    # Quality indicators
    avg_quality_source: float = 30.0
    avg_quality_target: float = 30.0
    
    # Repeat indicators
    is_repeat_region: bool = False
    repeat_score: float = 0.0
    
    # Hi-C support (if available)
    hic_support: float = 0.0
    
    # Ground truth label
    label: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overlap_length': self.overlap_length,
            'identity': self.identity,
            'source_coverage': self.source_coverage,
            'target_coverage': self.target_coverage,
            'coverage_ratio': self.coverage_ratio,
            'gc_content_source': self.gc_content_source,
            'gc_content_target': self.gc_content_target,
            'gc_similarity': self.gc_similarity,
            'kmer_consistency': self.kmer_consistency,
            'shared_kmers': self.shared_kmers,
            'position_distance': self.position_distance,
            'same_strand': int(self.same_strand),
            'avg_quality_source': self.avg_quality_source,
            'avg_quality_target': self.avg_quality_target,
            'is_repeat_region': int(self.is_repeat_region),
            'repeat_score': self.repeat_score,
            'hic_support': self.hic_support,
            'label': self.label
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy feature vector (excluding label)."""
        features = [
            self.overlap_length,
            self.identity,
            self.source_coverage,
            self.target_coverage,
            self.coverage_ratio,
            self.gc_content_source,
            self.gc_content_target,
            self.gc_similarity,
            self.kmer_consistency,
            self.shared_kmers,
            self.position_distance,
            float(self.same_strand),
            self.avg_quality_source,
            self.avg_quality_target,
            float(self.is_repeat_region),
            self.repeat_score,
            self.hic_support
        ]
        return np.array(features, dtype=np.float32)


def extract_overlap_features(
    source_read,
    target_read,
    edge_label,
    coverage_dict: Optional[Dict] = None,
    hic_support_dict: Optional[Dict] = None
) -> OverlapFeatures:
    """
    Extract features for a single edge/overlap.
    
    Args:
        source_read: Source read object (SimulatedRead)
        target_read: Target read object (SimulatedRead)
        edge_label: EdgeGroundTruth object with true label
        coverage_dict: Optional coverage information
        hic_support_dict: Optional Hi-C support scores
    
    Returns:
        OverlapFeatures object
    """
    # Calculate overlap metrics
    overlap_len = 0
    identity = 1.0  # For synthetic reads
    
    # Calculate distance
    if hasattr(source_read, 'start_pos') and hasattr(target_read, 'start_pos'):
        position_distance = abs(target_read.start_pos - source_read.end_pos)
    else:
        position_distance = 0
    
    # Check strand
    same_strand = True
    if hasattr(source_read, 'strand') and hasattr(target_read, 'strand'):
        same_strand = (source_read.strand == target_read.strand)
    
    # Get coverage
    source_cov = 1.0
    target_cov = 1.0
    if coverage_dict:
        source_cov = coverage_dict.get(source_read.read_id, 1.0)
        target_cov = coverage_dict.get(target_read.read_id, 1.0)
    
    coverage_ratio = target_cov / source_cov if source_cov > 0 else 1.0
    
    # Calculate GC content
    def calc_gc(seq: str) -> float:
        if not seq:
            return 0.5
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq)
    
    source_gc = calc_gc(source_read.sequence) if hasattr(source_read, 'sequence') else 0.42
    target_gc = calc_gc(target_read.sequence) if hasattr(target_read, 'sequence') else 0.42
    gc_sim = 1.0 - abs(source_gc - target_gc)
    
    # K-mer consistency - count shared k-mers
    k = 21
    if hasattr(source_read, 'sequence') and hasattr(target_read, 'sequence'):
        source_seq = source_read.sequence
        target_seq = target_read.sequence
        if len(source_seq) >= k and len(target_seq) >= k:
            source_kmers = set(source_seq[i:i+k] for i in range(len(source_seq)-k+1))
            target_kmers = set(target_seq[i:i+k] for i in range(len(target_seq)-k+1))
            shared_kmers = len(source_kmers & target_kmers)
            total_kmers = len(source_kmers | target_kmers)
            kmer_consistency = shared_kmers / total_kmers if total_kmers > 0 else 0.0
        else:
            kmer_consistency = 0.5
            shared_kmers = 0
    else:
        kmer_consistency = 0.5
        shared_kmers = 0
    
    # Quality scores - calculate from quality strings
    def calc_avg_qual(qual_str: str) -> float:
        if not qual_str:
            return 30.0
        # Convert Phred+33 ASCII to quality scores
        return sum(ord(c) - 33 for c in qual_str) / len(qual_str) if qual_str else 30.0
    
    avg_qual_source = calc_avg_qual(source_read.quality) if hasattr(source_read, 'quality') else 30.0
    avg_qual_target = calc_avg_qual(target_read.quality) if hasattr(target_read, 'quality') else 30.0
    
    # Repeat indicators
    is_repeat = edge_label.label.value == 'repeat'
    repeat_score = 0.8 if is_repeat else 0.1
    
    # Hi-C support
    hic_sup = 0.0
    if hic_support_dict:
        edge_key = (source_read.read_id, target_read.read_id)
        hic_sup = hic_support_dict.get(edge_key, 0.0)
    
    return OverlapFeatures(
        overlap_length=overlap_len,
        identity=identity,
        source_coverage=source_cov,
        target_coverage=target_cov,
        coverage_ratio=coverage_ratio,
        gc_content_source=source_gc,
        gc_content_target=target_gc,
        gc_similarity=gc_sim,
        kmer_consistency=kmer_consistency,
        shared_kmers=shared_kmers,
        position_distance=position_distance,
        same_strand=same_strand,
        avg_quality_source=avg_qual_source,
        avg_quality_target=avg_qual_target,
        is_repeat_region=is_repeat,
        repeat_score=repeat_score,
        hic_support=hic_sup,
        label=edge_label.label.value
    )


# ============================================================================
#                    GNN PATH PREDICTOR FEATURES
# ============================================================================

@dataclass
class GNNGraphTensors:
    """
    Graph neural network tensors for path prediction.
    
    Format compatible with PyTorch Geometric or DGL.
    """
    # Node features (N x F) where N=num_nodes, F=feature_dim
    node_features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Edge index (2 x E) where E=num_edges
    edge_index: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Edge features (E x F_e) where F_e=edge_feature_dim
    edge_features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Node labels (for supervised learning)
    node_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Path labels (correct node sequences)
    path_sequences: List[List[int]] = field(default_factory=list)
    
    # Metadata
    num_nodes: int = 0
    num_edges: int = 0
    node_dim: int = 0
    edge_dim: int = 0


def build_gnn_graph_tensors(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    node_labels: Dict,
    edge_labels: Dict,
    path_labels: List
) -> GNNGraphTensors:
    """
    Build GNN graph tensors from labeled assembly graph.
    
    Args:
        nodes: List of node IDs
        edges: List of (source, target) edges
        node_labels: Dict mapping node_id -> NodeGroundTruth
        edge_labels: Dict mapping (source, target) -> EdgeGroundTruth
        path_labels: List of PathGroundTruth objects
    
    Returns:
        GNNGraphTensors object
    """
    logger.info(f"Building GNN tensors for {len(nodes)} nodes, {len(edges)} edges")
    
    # Create node ID to index mapping
    node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
    
    # Build node features from node labels and topology
    node_feature_dim = 32
    node_features = np.zeros((len(nodes), node_feature_dim), dtype=np.float32)
    
    # Calculate node degrees
    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}
    for source, target in edges:
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1
    
    for node_id, idx in node_to_idx.items():
        # Features 0-3: One-hot haplotype encoding
        if node_id in node_labels:
            hap = node_labels[node_id].haplotype.value
            if hap == 'A':
                node_features[idx, 0] = 1.0
            elif hap == 'B':
                node_features[idx, 1] = 1.0
            elif hap == 'both':
                node_features[idx, 2] = 1.0
            else:
                node_features[idx, 3] = 1.0  # Unknown/repeat
        
        # Features 4-5: Degree information (normalized)
        node_features[idx, 4] = min(in_degree.get(node_id, 0) / 10.0, 1.0)
        node_features[idx, 5] = min(out_degree.get(node_id, 0) / 10.0, 1.0)
        
        # Feature 6: Is repeat
        if node_id in node_labels:
            node_features[idx, 6] = 1.0 if node_labels[node_id].is_repeat else 0.0
        
        # Features 7-31: Reserved for future use (kept as zeros)
    
    # Build edge index (COO format)
    edge_sources = []
    edge_targets = []
    for source, target in edges:
        if source in node_to_idx and target in node_to_idx:
            edge_sources.append(node_to_idx[source])
            edge_targets.append(node_to_idx[target])
    
    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
    
    # Build edge features from edge labels
    edge_feature_dim = 16
    edge_features = np.zeros((len(edge_sources), edge_feature_dim), dtype=np.float32)
    
    for i, (src_idx, tgt_idx) in enumerate(zip(edge_sources, edge_targets)):
        src_node = nodes[src_idx]
        tgt_node = nodes[tgt_idx]
        edge_key = (src_node, tgt_node)
        
        if edge_key in edge_labels:
            label = edge_labels[edge_key]
            # Features 0-4: One-hot edge type
            label_map = {'true': 0, 'repeat': 1, 'chimeric': 2, 'allelic': 3, 'sv_break': 4}
            if label.label.value in label_map:
                edge_features[i, label_map[label.label.value]] = 1.0
            
            # Feature 5: Overlap distance (normalized)
            if hasattr(label, 'overlap_distance') and label.overlap_distance is not None:
                edge_features[i, 5] = min(label.overlap_distance / 10000.0, 1.0)
            else:
                edge_features[i, 5] = 0.0
            
            # Feature 6: SV indicator
            edge_features[i, 6] = 1.0 if label.crosses_sv else 0.0
            
            # Features 7-15: Reserved for future use
    
    # Build node labels (haplotype assignments)
    node_label_array = np.zeros(len(nodes), dtype=np.int64)
    for node_id, label_obj in node_labels.items():
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            # Convert haplotype to integer label
            if label_obj.haplotype.value == 'A':
                node_label_array[idx] = 0
            elif label_obj.haplotype.value == 'B':
                node_label_array[idx] = 1
            elif label_obj.haplotype.value == 'both':
                node_label_array[idx] = 2
            else:
                node_label_array[idx] = 3  # Unknown/repeat
    
    # Convert path labels to node index sequences
    path_sequences = []
    for path_label in path_labels:
        path_seq = []
        for node_id in path_label.node_sequence:
            if node_id in node_to_idx:
                path_seq.append(node_to_idx[node_id])
        if path_seq:
            path_sequences.append(path_seq)
    
    logger.info(f"Built GNN tensors: {len(nodes)} nodes, {len(edge_sources)} edges, {len(path_sequences)} paths")
    
    return GNNGraphTensors(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_labels=node_label_array,
        path_sequences=path_sequences,
        num_nodes=len(nodes),
        num_edges=len(edge_sources),
        node_dim=node_feature_dim,
        edge_dim=edge_feature_dim
    )


# ============================================================================
#                    DIPLOID DISENTANGLER FEATURES
# ============================================================================

@dataclass
class DiploidNodeFeatures:
    """
    Features for diploid haplotype assignment.
    
    Combines multiple signals:
    - GNN embeddings
    - Hi-C contact patterns
    - Repeat content
    - Coverage patterns
    - Allelic markers
    """
    node_id: str = ""
    
    # GNN-derived features
    gnn_embedding: np.ndarray = field(default_factory=lambda: np.zeros(32))
    
    # Hi-C features
    hic_contacts_A: float = 0.0  # Contacts with known hap-A nodes
    hic_contacts_B: float = 0.0  # Contacts with known hap-B nodes
    hic_ratio: float = 0.5  # A/(A+B)
    
    # Coverage features
    long_read_coverage: float = 0.0
    ul_read_coverage: float = 0.0
    coverage_variance: float = 0.0
    
    # Repeat indicators
    repeat_content: float = 0.0
    tandem_repeat_score: float = 0.0
    
    # Allelic markers
    num_snps: int = 0
    heterozygosity: float = 0.0
    
    # Ground truth
    true_haplotype: str = "unknown"
    label: Any = None  # NodeHaplotype enum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        label_str = self.label.value if self.label else self.true_haplotype
        return {
            'node_id': self.node_id,
            'features': self.to_feature_vector().tolist(),
            'label': label_str
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        features = list(self.gnn_embedding) + [
            self.hic_contacts_A,
            self.hic_contacts_B,
            self.hic_ratio,
            self.long_read_coverage,
            self.ul_read_coverage,
            self.coverage_variance,
            self.repeat_content,
            self.tandem_repeat_score,
            float(self.num_snps),
            self.heterozygosity
        ]
        return np.array(features, dtype=np.float32)


def extract_diploid_features(
    node_id: str,
    node_label,
    gnn_tensors: GNNGraphTensors,
    node_to_idx: Dict[str, int],
    hic_contacts: Optional[Dict] = None
) -> DiploidNodeFeatures:
    """
    Extract features for diploid disentanglement.
    
    Args:
        node_id: Node identifier
        node_label: NodeGroundTruth object
        gnn_tensors: GNN graph tensors
        node_to_idx: Node ID to index mapping
        hic_contacts: Optional Hi-C contact matrix
    
    Returns:
        DiploidNodeFeatures object
    """
    # Get GNN embedding for this node
    idx = node_to_idx.get(node_id, 0)
    gnn_emb = gnn_tensors.node_features[idx] if idx < len(gnn_tensors.node_features) else np.zeros(32)
    
    # Hi-C features - derive from haplotype and spanning reads
    # Nodes from one haplotype should have higher contacts with that haplotype
    hic_A = 1.0
    hic_B = 1.0
    if node_label.haplotype.value == 'A':
        hic_A = 10.0 + len(node_label.spanning_reads) * 0.5
        hic_B = 2.0
    elif node_label.haplotype.value == 'B':
        hic_A = 2.0
        hic_B = 10.0 + len(node_label.spanning_reads) * 0.5
    elif node_label.haplotype.value == 'both':
        # Shared nodes have balanced contacts
        hic_A = 8.0 + len(node_label.spanning_reads) * 0.3
        hic_B = 8.0 + len(node_label.spanning_reads) * 0.3
    else:
        # Unknown/repeat - low signal
        hic_A = 3.0
        hic_B = 3.0
    
    hic_ratio = hic_A / (hic_A + hic_B) if (hic_A + hic_B) > 0 else 0.5
    
    # Coverage - estimate from number of spanning reads
    num_spanning = len(node_label.spanning_reads)
    long_cov = max(1.0, num_spanning * 1.5)  # Assume ~1.5x per spanning read
    ul_cov = max(0.5, num_spanning * 0.2)    # UL reads are rarer
    cov_var = max(1.0, num_spanning * 0.3)   # Higher coverage = more variance
    
    # Repeat content - use actual repeat status
    repeat_content = 0.85 if node_label.is_repeat else 0.15
    # Tandem repeats have multiple copies in local region
    tandem_score = 0.7 if node_label.is_repeat else 0.1
    
    # Allelic markers - nodes with both haplotypes have SNPs
    if node_label.haplotype.value == 'both':
        num_snps = max(3, num_spanning // 5)  # Expect ~1 SNP per 5 reads
        heterozygosity = 0.001  # Human-like
    else:
        num_snps = 0
        heterozygosity = 0.0
    
    return DiploidNodeFeatures(
        node_id=node_id,
        gnn_embedding=gnn_emb,
        hic_contacts_A=hic_A,
        hic_contacts_B=hic_B,
        hic_ratio=hic_ratio,
        long_read_coverage=long_cov,
        ul_read_coverage=ul_cov,
        coverage_variance=cov_var,
        repeat_content=repeat_content,
        tandem_repeat_score=tandem_score,
        num_snps=num_snps,
        heterozygosity=heterozygosity,
        true_haplotype=node_label.haplotype.value,
        label=node_label.haplotype
    )


# ============================================================================
#                    UL ROUTING FEATURES
# ============================================================================

@dataclass
class ULPathFeatures:
    """
    Features for ultralong read routing through graph.
    
    12-dimensional feature vector as defined in ul_routing_ai.py:
    - Path consistency metrics
    - Coverage agreement
    - Sequence identity
    - Graph topology features
    """
    # Path consistency
    path_length: int = 0
    num_nodes: int = 0
    avg_node_degree: float = 0.0
    
    # Coverage agreement
    ul_coverage_support: float = 0.0
    long_read_support: float = 0.0
    coverage_consistency: float = 0.0
    
    # Sequence metrics
    avg_identity: float = 0.0
    total_mismatches: int = 0
    
    # Graph topology
    num_branches: int = 0
    path_complexity: float = 0.0
    alternative_paths: int = 0
    
    # Confidence
    confidence_score: float = 0.0
    
    # Ground truth
    is_correct_path: bool = False
    path_id: str = ""
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'path_id': self.path_id,
            'features': self.to_feature_vector().tolist(),
            'score': self.score
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to 12D feature vector."""
        return np.array([
            float(self.path_length),
            float(self.num_nodes),
            self.avg_node_degree,
            self.ul_coverage_support,
            self.long_read_support,
            self.coverage_consistency,
            self.avg_identity,
            float(self.total_mismatches),
            float(self.num_branches),
            self.path_complexity,
            float(self.alternative_paths),
            self.confidence_score
        ], dtype=np.float32)


def extract_ul_routing_features(
    ul_read,
    candidate_path: List[str],
    correct_path: List[str],
    graph_edges: List[Tuple[str, str]]
) -> ULPathFeatures:
    """
    Extract features for UL read routing.
    
    Args:
        ul_read: Ultralong read object
        candidate_path: Candidate node path
        correct_path: Ground truth correct path
        graph_edges: Graph edges for topology analysis
    
    Returns:
        ULPathFeatures object
    """
    # Path metrics
    path_len = len(ul_read.sequence) if hasattr(ul_read, 'sequence') else 100000
    num_nodes = len(candidate_path)
    
    # Calculate average degree of nodes in path
    node_degrees = {}
    for src, tgt in graph_edges:
        node_degrees[src] = node_degrees.get(src, 0) + 1
        node_degrees[tgt] = node_degrees.get(tgt, 0) + 1
    
    degrees = [node_degrees.get(node, 1) for node in candidate_path]
    avg_degree = sum(degrees) / len(degrees) if degrees else 2.0
    
    # Coverage support - estimate from path length and node count
    # Longer paths with fewer nodes = higher coverage per node
    avg_node_len = path_len / num_nodes if num_nodes > 0 else 1000
    ul_support = min(100.0, avg_node_len / 1000.0 * 15.0)  # Scale by node length
    long_support = ul_support * 2.0  # Long reads are denser
    
    # Coverage consistency - paths with similar node degrees are more consistent
    if degrees:
        avg_deg = sum(degrees) / len(degrees)
        degree_var = sum((d - avg_deg) ** 2 for d in degrees) / len(degrees)
        cov_consistency = max(0.0, 1.0 - (degree_var / (avg_deg + 1) / 10.0))
    else:
        cov_consistency = 0.5
    
    # Sequence metrics - compare candidate to correct path
    matching_nodes = sum(1 for n in candidate_path if n in correct_path)
    avg_ident = matching_nodes / max(len(candidate_path), len(correct_path)) if correct_path else 0.5
    mismatches = len(candidate_path) + len(correct_path) - 2 * matching_nodes
    
    # Topology - count branches at each node in path
    branches = 0
    for node in candidate_path:
        branches += max(0, node_degrees.get(node, 1) - 2)  # Extra edges = branches
    
    complexity = min(1.0, branches / (num_nodes + 1))  # Normalized
    
    # Alternative paths - count other nodes at similar degrees
    alternatives = sum(1 for deg in degrees if deg > 2)
    
    # Confidence
    confidence = 0.92
    
    # Check if correct
    is_correct = (candidate_path == correct_path)
    
    # Generate path ID
    path_id = f"ul_{ul_read.read_id if hasattr(ul_read, 'read_id') else 'unknown'}"
    score = 1.0 if is_correct else 0.0
    
    return ULPathFeatures(
        path_length=path_len,
        num_nodes=num_nodes,
        avg_node_degree=avg_degree,
        ul_coverage_support=ul_support,
        long_read_support=long_support,
        coverage_consistency=cov_consistency,
        avg_identity=avg_ident,
        total_mismatches=mismatches,
        num_branches=branches,
        path_complexity=complexity,
        alternative_paths=alternatives,
        confidence_score=confidence,
        is_correct_path=is_correct,
        path_id=path_id,
        score=score
    )


# ============================================================================
#                    SV DETECTION FEATURES
# ============================================================================

@dataclass
class SVDetectionFeatures:
    """
    Features for structural variant detection.
    
    Combines multiple signals:
    - Coverage patterns (drops, spikes)
    - Graph branching complexity
    - Hi-C contact disruptions
    - UL read support
    - Sequence composition
    """
    # Location
    region_id: str = ""
    chrom: str = "chr1"
    start_pos: int = 0
    end_pos: int = 0
    
    # Coverage features
    avg_coverage: float = 0.0
    coverage_stddev: float = 0.0
    coverage_drop_score: float = 0.0
    coverage_spike_score: float = 0.0
    
    # Graph topology
    num_branches: int = 0
    bubble_score: float = 0.0
    graph_complexity: float = 0.0
    
    # Hi-C features
    hic_contact_disruption: float = 0.0
    cis_trans_ratio: float = 0.0
    
    # UL support
    ul_read_support: int = 0
    ul_spanning_reads: int = 0
    ul_consistency: float = 0.0
    
    # Sequence features
    repeat_content: float = 0.0
    gc_content: float = 0.0
    
    # Ground truth
    true_sv_type: str = "none"
    true_sv_size: int = 0
    sv_type: str = "none"  # For to_dict
    sv_size: int = 0  # For to_dict
    position: str = ""  # For to_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'region_id': self.region_id,
            'position': self.position or f"{self.chrom}:{self.start_pos}-{self.end_pos}",
            'features': self.to_feature_vector().tolist(),
            'sv_type': self.sv_type or self.true_sv_type,
            'sv_size': self.sv_size or self.true_sv_size
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.avg_coverage,
            self.coverage_stddev,
            self.coverage_drop_score,
            self.coverage_spike_score,
            float(self.num_branches),
            self.bubble_score,
            self.graph_complexity,
            self.hic_contact_disruption,
            self.cis_trans_ratio,
            float(self.ul_read_support),
            float(self.ul_spanning_reads),
            self.ul_consistency,
            self.repeat_content,
            self.gc_content
        ], dtype=np.float32)


def extract_sv_features(
    sv_label,
    region_nodes: List[str],
    coverage_dict: Dict,
    hic_contacts: Optional[Dict] = None
) -> SVDetectionFeatures:
    """
    Extract features for SV detection.
    
    Args:
        sv_label: SVGroundTruth object
        region_nodes: Nodes in SV region
        coverage_dict: Coverage information
        hic_contacts: Optional Hi-C contacts
    
    Returns:
        SVDetectionFeatures object
    """
    # Coverage statistics
    coverages = [coverage_dict.get(node, 30.0) for node in region_nodes]
    avg_cov = np.mean(coverages) if coverages else 30.0
    std_cov = np.std(coverages) if coverages else 5.0
    
    # SV-specific patterns
    if sv_label.sv_type.value == 'deletion':
        drop_score = 0.8
        spike_score = 0.1
    elif sv_label.sv_type.value == 'duplication':
        drop_score = 0.1
        spike_score = 0.9
    elif sv_label.sv_type.value == 'insertion':
        drop_score = 0.2
        spike_score = 0.6
    else:
        drop_score = 0.3
        spike_score = 0.3
    
    # Graph features
    num_branches = sv_label.graph_signature.get('num_spanning', 0)
    bubble_score = 0.5 if sv_label.sv_type.value in ['insertion', 'inversion'] else 0.1
    
    # Complexity based on SV type
    complexity_map = {
        'deletion': 0.3,
        'insertion': 0.6,
        'inversion': 0.8,
        'duplication': 0.7,
        'translocation': 0.9
    }
    complexity = complexity_map.get(sv_label.sv_type.value, 0.5)
    
    # Hi-C features - SVs disrupt normal contact patterns
    # Inversions and translocations have strongest Hi-C signatures
    if sv_label.sv_type.value in ['inversion', 'translocation']:
        hic_disruption = 0.75 + (sv_label.size / 100000.0) * 0.2  # Larger = more disruptive
    elif sv_label.sv_type.value in ['deletion', 'duplication']:
        hic_disruption = 0.5 + (sv_label.size / 100000.0) * 0.15
    else:
        hic_disruption = 0.3
    
    hic_disruption = min(1.0, hic_disruption)
    
    # Cis-trans ratio: translocations change this dramatically
    if sv_label.sv_type.value == 'translocation':
        cis_trans = 0.3  # More trans contacts
    else:
        cis_trans = 0.85  # Normal cis dominance
    
    # UL support
    ul_support = sv_label.graph_signature.get('num_spanning', 0)
    ul_spanning = ul_support
    ul_consist = 0.75
    
    # Sequence
    repeat = 0.3
    gc = 0.42
    
    return SVDetectionFeatures(
        region_id=sv_label.sv_id,
        chrom=sv_label.ref_chrom,
        start_pos=sv_label.ref_start,
        end_pos=sv_label.ref_end,
        avg_coverage=avg_cov,
        coverage_stddev=std_cov,
        coverage_drop_score=drop_score,
        coverage_spike_score=spike_score,
        num_branches=num_branches,
        bubble_score=bubble_score,
        graph_complexity=complexity,
        hic_contact_disruption=hic_disruption,
        cis_trans_ratio=cis_trans,
        ul_read_support=ul_support,
        ul_spanning_reads=ul_spanning,
        ul_consistency=ul_consist,
        repeat_content=repeat,
        gc_content=gc,
        true_sv_type=sv_label.sv_type.value,
        true_sv_size=sv_label.size,
        sv_type=sv_label.sv_type.value,
        sv_size=sv_label.size,
        position=f"{sv_label.ref_chrom}:{sv_label.ref_start}-{sv_label.ref_end}"
    )


# ============================================================================
#                    MASTER FEATURE EXTRACTION
# ============================================================================

@dataclass
class TrainingDataset:
    """Complete training dataset for all models."""
    # Overlap classifier
    overlap_features: List[OverlapFeatures] = field(default_factory=list)
    
    # GNN path predictor
    gnn_tensors: Optional[GNNGraphTensors] = None
    
    # Diploid disentangler
    diploid_features: List[DiploidNodeFeatures] = field(default_factory=list)
    
    # UL routing
    ul_routing_features: List[ULPathFeatures] = field(default_factory=list)
    
    # SV detection
    sv_features: List[SVDetectionFeatures] = field(default_factory=list)


def build_training_dataset(
    simulated_reads: List,
    ul_reads: List,
    graph_nodes: List[str],
    graph_edges: List[Tuple[str, str]],
    ground_truth_labels
) -> TrainingDataset:
    """
    Build complete training dataset from labeled assembly graph.
    
    This is the main entry point that extracts features for all ML models.
    
    Args:
        simulated_reads: All simulated reads
        ul_reads: Ultralong reads
        graph_nodes: Graph node IDs
        graph_edges: Graph edges
        ground_truth_labels: GroundTruthLabels object
    
    Returns:
        TrainingDataset with all features
    """
    logger.info("=" * 80)
    logger.info("Building training dataset from labeled graph")
    logger.info("=" * 80)
    
    dataset = TrainingDataset()
    
    # Create read lookup
    read_dict = {r.read_id: r for r in simulated_reads}
    
    # 1. Extract overlap classifier features
    logger.info("Extracting overlap classifier features...")
    for (source_id, target_id), edge_label in ground_truth_labels.edge_labels.items():
        if source_id in read_dict and target_id in read_dict:
            features = extract_overlap_features(
                read_dict[source_id],
                read_dict[target_id],
                edge_label
            )
            dataset.overlap_features.append(features)
    logger.info(f"  Extracted {len(dataset.overlap_features)} overlap examples")
    
    # 2. Build GNN graph tensors
    logger.info("Building GNN graph tensors...")
    dataset.gnn_tensors = build_gnn_graph_tensors(
        graph_nodes,
        graph_edges,
        ground_truth_labels.node_labels,
        ground_truth_labels.edge_labels,
        ground_truth_labels.path_labels
    )
    logger.info(f"  Built GNN tensors: {dataset.gnn_tensors.num_nodes} nodes, {dataset.gnn_tensors.num_edges} edges")
    
    # 3. Extract diploid features
    logger.info("Extracting diploid disentanglement features...")
    node_to_idx = {node: idx for idx, node in enumerate(graph_nodes)}
    for node_id, node_label in ground_truth_labels.node_labels.items():
        features = extract_diploid_features(
            node_id,
            node_label,
            dataset.gnn_tensors,
            node_to_idx
        )
        dataset.diploid_features.append(features)
    logger.info(f"  Extracted {len(dataset.diploid_features)} diploid examples")
    
    # 4. Extract UL routing features
    logger.info("Extracting UL routing features...")
    for ul_route_label in ground_truth_labels.ul_route_labels:
        # For demo, use correct path as candidate
        features = extract_ul_routing_features(
            read_dict.get(ul_route_label.read_id),
            ul_route_label.correct_path,
            ul_route_label.correct_path,
            graph_edges
        )
        dataset.ul_routing_features.append(features)
    logger.info(f"  Extracted {len(dataset.ul_routing_features)} UL routing examples")
    
    # 5. Extract SV detection features
    logger.info("Extracting SV detection features...")
    
    # Build coverage dict from node labels (use spanning read counts)
    coverage_dict = {}
    for node_id, node_label in ground_truth_labels.node_labels.items():
        # Estimate coverage from number of spanning reads
        coverage_dict[node_id] = max(1.0, len(node_label.spanning_reads) * 1.5)
    
    for sv_label in ground_truth_labels.sv_labels:
        # Get nodes in SV region - find nodes whose spanning reads might overlap SV
        # Since we don't have direct access to alignments here, we use a simpler heuristic:
        # Nodes are likely in SV region if they have similar IDs or are from same haplotype
        region_nodes = []
        target_haplotype = sv_label.haplotype
        
        for node_id, node_label in ground_truth_labels.node_labels.items():
            # Include nodes from the same haplotype as the SV
            if node_label.haplotype.value == target_haplotype:
                region_nodes.append(node_id)
                if len(region_nodes) >= 10:  # Limit to 10 nodes per SV
                    break
        
        # Fallback: if no nodes found, use sample of nodes
        if not region_nodes:
            region_nodes = list(graph_nodes)[:min(5, len(graph_nodes))]
        features = extract_sv_features(sv_label, region_nodes, coverage_dict)
        dataset.sv_features.append(features)
    logger.info(f"  Extracted {len(dataset.sv_features)} SV detection examples")
    
    logger.info("=" * 80)
    logger.info("Training dataset complete")
    logger.info("=" * 80)
    logger.info(f"Overlap features: {len(dataset.overlap_features)}")
    logger.info(f"GNN tensors: {dataset.gnn_tensors.num_nodes} nodes, {dataset.gnn_tensors.num_edges} edges")
    logger.info(f"Diploid features: {len(dataset.diploid_features)}")
    logger.info(f"UL routing features: {len(dataset.ul_routing_features)}")
    logger.info(f"SV features: {len(dataset.sv_features)}")
    
    return dataset
