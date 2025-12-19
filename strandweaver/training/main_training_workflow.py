"""
Main Training Workflow for StrandWeaver ML Models

This module orchestrates the complete end-to-end training data generation pipeline:

1. Synthetic Data Generation (via synthetic_data_generator.py)
   - Diploid genome generation with SVs and repeats
   - Multi-technology read simulation (Illumina, HiFi, ONT, UL, Hi-C)
   - Ground-truth labeling

2. Graph Assembly Integration
   - de Bruijn graph construction from simulated reads
   - Graph-based feature extraction

3. Feature Extraction (THIS MODULE)
   - Overlap features for EdgeWarden
   - GNN graph tensors for PathWeaver
   - Haplotype features for Diploid Detangler
   - UL routing features for ThreadCompass
   - SV detection features

4. Training Corpus Generation (THIS MODULE)
   - Scenario-based corpus generation (simple, balanced, repeat-heavy, etc.)
   - Automated feature extraction pipeline
   - Sharded dataset output

Scenarios Available:
- simple: Quick test (10 genomes × 100kb, ~2 min)
- balanced: Production training (100 genomes × 1Mb, ~20 min)
- repeat_heavy: Repeat-rich (50 genomes × 2Mb, 60% repeats, ~30 min)
- sv_dense: SV-focused (50 genomes × 1Mb, 10× SV density, ~15 min)
- diploid_focus: High heterozygosity (100 genomes × 1Mb, 2% het, ~25 min)
- ultra_long_focus: UL-optimized (30 genomes × 5Mb, 50× UL, ~40 min)

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation (Consolidated Phase 2)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import time
from datetime import datetime
from collections import defaultdict
import numpy as np

# Phase 5.3 Module Imports - Synthetic Data Generation (Consolidated Phase 1)
from strandweaver.training.synthetic_data_generator import (
    generate_diploid_genome,
    GenomeConfig,
    StructuralVariant,
    DiploidGenome,
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_paired_fastq,
    write_fastq,
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    SimulatedRead,
    SimulatedReadPair,
    generate_ground_truth_labels,
    GroundTruthLabels,
    export_labels_to_tsv,
    ReadAlignment,
)

# Phase 5.3 Module Imports - Graph Assembly
from strandweaver.assembly_core.dbg_engine_module import (
    DeBruijnGraphBuilder,
    KmerGraphConfig,
    KmerGraph,
    KmerNode,
    KmerEdge,
)

# Phase 5.3 Module Imports - Dataset Writing
from strandweaver.training.dataset_writer import (
    write_sharded_dataset,
    DatasetMetadata,
    create_train_val_test_split,
)

logger = logging.getLogger(__name__)


# ============================================================================
#                    PART 1: FEATURE EXTRACTION
# ============================================================================

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
#                    HAPLOTYPE DETANGLER FEATURES
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
    
    # Haplotype detangler
    haplotype_features: List[DiploidNodeFeatures] = field(default_factory=list)
    
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



# ============================================================================
#                    PART 2: TRAINING CORPUS GENERATION
# ============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a training scenario."""
    name: str
    genome_size: int
    num_genomes: int
    
    # Coverage parameters
    illumina_coverage: float = 30.0
    hifi_coverage: float = 30.0
    ont_coverage: float = 30.0
    ul_coverage: float = 10.0
    hic_coverage: float = 20.0
    
    # Genome complexity
    gc_content: float = 0.42
    repeat_density: float = 0.45
    snp_rate: float = 0.001
    sv_density: float = 0.00005
    sv_max_size: int = 100_000
    
    # Assembly parameters
    kmer_size: int = 31
    min_kmer_count: int = 2
    
    # Dataset parameters
    dataset_format: str = "jsonl"  # jsonl, npz, parquet, or all
    shard_size: int = 5000
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


# ============================================================================
#                    PREDEFINED TRAINING SCENARIOS
# ============================================================================

SCENARIOS = {
    'simple': ScenarioConfig(
        name='simple',
        genome_size=100_000,
        num_genomes=10,
        illumina_coverage=30.0,
        hifi_coverage=30.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=20.0,
        sv_max_size=10_000,
        kmer_size=21,  # Smaller k for small genomes
        dataset_format="jsonl"
    ),
    
    'fast_balanced': ScenarioConfig(
        name='fast_balanced',
        genome_size=500_000,  # Smaller for speed
        num_genomes=20,  # Full dataset
        illumina_coverage=20.0,  # Lower coverage
        hifi_coverage=20.0,
        ont_coverage=20.0,
        ul_coverage=5.0,
        hic_coverage=10.0,
        sv_max_size=50_000,
        kmer_size=27,  # Optimized k
        dataset_format="jsonl"
    ),
    
    'balanced': ScenarioConfig(
        name='balanced',
        genome_size=1_000_000,
        num_genomes=30,  # Reduced from 100 for faster generation
        illumina_coverage=30.0,
        hifi_coverage=30.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=20.0,
        dataset_format="all"  # All formats for production
    ),
    
    'repeat_heavy': ScenarioConfig(
        name='repeat_heavy',
        genome_size=2_000_000,
        num_genomes=50,
        repeat_density=0.60,  # 60% repeats (high)
        illumina_coverage=40.0,
        hifi_coverage=40.0,
        ont_coverage=30.0,
        ul_coverage=15.0,
        hic_coverage=30.0,
        kmer_size=51,  # Larger k for repeats
        dataset_format="all"
    ),
    
    'sv_dense': ScenarioConfig(
        name='sv_dense',
        genome_size=1_000_000,
        num_genomes=50,
        sv_density=0.0005,  # 10× normal SV density
        sv_max_size=500_000,  # Larger SVs
        illumina_coverage=40.0,
        hifi_coverage=40.0,
        ont_coverage=40.0,
        ul_coverage=20.0,  # Higher UL for SV spanning
        hic_coverage=30.0,
        dataset_format="all"
    ),
    
    'diploid_focus': ScenarioConfig(
        name='diploid_focus',
        genome_size=1_000_000,
        num_genomes=100,
        snp_rate=0.02,  # 2% heterozygosity (high)
        illumina_coverage=30.0,
        hifi_coverage=40.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=40.0,  # Higher Hi-C for phasing
        dataset_format="all"
    ),
    
    'ultra_long_focus': ScenarioConfig(
        name='ultra_long_focus',
        genome_size=5_000_000,
        num_genomes=30,
        illumina_coverage=20.0,
        hifi_coverage=30.0,
        ont_coverage=20.0,
        ul_coverage=50.0,  # Very high UL coverage
        hic_coverage=20.0,
        kmer_size=31,
        dataset_format="all"
    ),
}


# ============================================================================
#                    GRAPH ASSEMBLY FROM SIMULATED READS
# ============================================================================

def build_assembly_graph_from_reads(
    illumina_reads: List[SimulatedReadPair],
    hifi_reads: List[SimulatedRead],
    kmer_config: KmerGraphConfig,
    genome_id: str
) -> Tuple[KmerGraph, List[Tuple[str, str]]]:
    """
    Build de Bruijn graph from simulated reads.
    
    Uses the StrandWeaver DeBruijnGraphBuilder to construct a graph
    from accurate reads (Illumina and HiFi).
    
    Args:
        illumina_reads: Simulated Illumina paired-end reads
        hifi_reads: Simulated PacBio HiFi reads
        kmer_config: K-mer graph configuration
        genome_id: Genome identifier for logging
    
    Returns:
        Tuple of (assembled graph, list of (read_id, sequence) tuples)
    """
    logger.info(f"[Genome {genome_id}] Building assembly graph (k={kmer_config.k})...")
    
    # Convert simulated reads to (read_id, sequence) format
    accurate_reads = []
    
    # Add Illumina reads
    for pair in illumina_reads:
        accurate_reads.append((pair.read1.read_id, pair.read1.sequence))
        accurate_reads.append((pair.read2.read_id, pair.read2.sequence))
    
    # Add HiFi reads
    for read in hifi_reads:
        accurate_reads.append((read.read_id, read.sequence))
    
    logger.info(f"[Genome {genome_id}]   Total accurate reads: {len(accurate_reads)}")
    
    # Build de Bruijn graph with GPU acceleration enabled
    builder = DeBruijnGraphBuilder(kmer_config, use_gpu=True)
    
    # Step 1: Build raw k-mer graph with read provenance
    raw_graph, node_to_reads = builder.build_raw_kmer_graph(accurate_reads, kmer_config)
    logger.info(f"[Genome {genome_id}]   Raw graph: {len(raw_graph.nodes)} nodes, {len(raw_graph.edges)} edges")
    
    # Step 2: Compact graph into unitigs, propagating read provenance
    compacted_graph, compacted_node_to_reads = builder.compact_graph(raw_graph, node_to_reads)
    logger.info(f"[Genome {genome_id}]   Compacted: {len(compacted_graph.nodes)} nodes, {len(compacted_graph.edges)} edges")
    
    return compacted_graph, accurate_reads, compacted_node_to_reads


def extract_graph_structure(graph: KmerGraph) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Extract node IDs and edge list from KmerGraph.
    
    Args:
        graph: Assembled k-mer graph
    
    Returns:
        Tuple of (node_ids, edge_list)
    """
    node_ids = [str(nid) for nid in graph.nodes.keys()]
    edge_list = [(str(e.from_id), str(e.to_id)) for e in graph.edges.values()]
    
    return node_ids, edge_list


# ============================================================================
#                    MAIN CORPUS GENERATION FUNCTION
# ============================================================================

def generate_training_corpus(
    scenario: str = 'balanced',
    output_dir: str = 'training_data',
    num_processes: int = 4,
    **override_params
) -> Dict[str, Any]:
    """
    Generate complete training corpus with full ML pipeline.
    
    This is the main entry point that orchestrates:
    1. Genome simulation (diploid with SVs)
    2. Read simulation (all technologies)
    3. Graph assembly (de Bruijn graph)
    4. Ground-truth labeling (all label types)
    5. Feature extraction (all ML models)
    6. Dataset writing (sharded output)
    
    Args:
        scenario: Scenario name ('simple', 'balanced', etc.)
        output_dir: Output directory for training data
        num_processes: Number of parallel processes
        **override_params: Override any ScenarioConfig parameters
    
    Returns:
        Metadata dictionary with generation statistics
    """
    start_time = time.time()
    output_path = Path(output_dir) / scenario
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get scenario configuration
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(SCENARIOS.keys())}")
    
    config = SCENARIOS[scenario]
    
    # Apply parameter overrides
    for key, value in override_params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info("=" * 80)
    logger.info(f"Generating Training Corpus: {scenario}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Genomes: {config.num_genomes} × {config.genome_size:,} bp")
    logger.info(f"  Coverage: Illumina={config.illumina_coverage}×, HiFi={config.hifi_coverage}×, "
                f"ONT={config.ont_coverage}×, UL={config.ul_coverage}×, Hi-C={config.hic_coverage}×")
    logger.info(f"  Complexity: GC={config.gc_content:.1%}, Repeats={config.repeat_density:.1%}, "
                f"SNPs={config.snp_rate:.3%}, SVs={config.sv_density:.5f}")
    logger.info(f"  K-mer: k={config.kmer_size}, min_count={config.min_kmer_count}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 80)
    
    # Initialize aggregate statistics
    total_stats = {
        'num_genomes': config.num_genomes,
        'scenario': scenario,
        'start_time': datetime.now().isoformat(),
        'genomes': [],
        'total_reads': {},
        'total_graph_nodes': 0,
        'total_graph_edges': 0,
        'total_labels': 0,
        'total_features': 0,
    }
    
    # Create K-mer graph configuration
    kmer_config = KmerGraphConfig(
        k=config.kmer_size,
        min_kmer_count=config.min_kmer_count,
        canonical=True
    )
    
    # ========================================================================
    # PHASE 1: Generate genomes, reads, and graphs
    # ========================================================================
    
    genome_data = []  # List of (genome, reads, graph, labels) tuples
    
    for i in range(config.num_genomes):
        genome_id = f"genome_{i:04d}"
        genome_output_dir = output_path / 'per_genome' / genome_id
        
        # Check if this genome has already been processed
        features_file = genome_output_dir / 'features.pkl'
        if features_file.exists():
            logger.info(f"\n{'='*80}")
            logger.info(f"SKIPPING {genome_id} ({i+1}/{config.num_genomes}) - Already processed")
            logger.info(f"{'='*80}")
            logger.info(f"Found existing features at: {features_file}")
            
            # Add to genome_data so it gets included in aggregation
            genome_data.append({
                'genome_id': genome_id,
                'features_file': features_file,
                'skipped': True
            })
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {genome_id} ({i+1}/{config.num_genomes})")
        logger.info(f"{'='*80}")
        
        # Step 1: Generate diploid genome
        logger.info(f"[{genome_id}] Step 1/6: Generating diploid genome...")
        genome_config = GenomeConfig(
            length=config.genome_size,
            gc_content=config.gc_content,
            repeat_density=config.repeat_density,
            snp_rate=config.snp_rate,
            sv_density=config.sv_density,
            sv_max_size=config.sv_max_size,
            random_seed=42 + i
        )
        
        diploid = generate_diploid_genome(genome_config)
        logger.info(f"[{genome_id}]   Generated: {len(diploid.hapA)} bp (A), {len(diploid.hapB)} bp (B)")
        logger.info(f"[{genome_id}]   Variants: {len(diploid.sv_truth_table)} SVs, "
                    f"{len([v for v in diploid.sv_truth_table if v.sv_type == 'deletion'])} deletions, "
                    f"{len([v for v in diploid.sv_truth_table if v.sv_type == 'insertion'])} insertions")
        
        # Step 2: Simulate reads
        logger.info(f"[{genome_id}] Step 2/6: Simulating sequencing reads...")
        
        # Illumina reads
        ill_config = IlluminaConfig(
            coverage=config.illumina_coverage,
            read_length=150,
            insert_size_mean=350,
            insert_size_std=50,
            error_rate=0.001,
            random_seed=42 + i
        )
        ill_reads = simulate_illumina_reads(diploid.hapA, ill_config)
        ill_reads_b = simulate_illumina_reads(diploid.hapB, ill_config)
        all_ill_reads = ill_reads + ill_reads_b
        
        # HiFi reads
        hifi_config = HiFiConfig(
            coverage=config.hifi_coverage,
            read_length_mean=15000,
            read_length_std=5000,
            error_rate=0.001,
            random_seed=42 + i
        )
        hifi_reads = simulate_long_reads(diploid.hapA, hifi_config, 'hifi')
        hifi_reads_b = simulate_long_reads(diploid.hapB, hifi_config, 'hifi')
        all_hifi_reads = hifi_reads + hifi_reads_b
        
        # ONT reads
        ont_config = ONTConfig(
            coverage=config.ont_coverage,
            read_length_mean=10000,
            read_length_std=5000,
            error_rate=0.05,
            indel_fraction=0.7,
            random_seed=42 + i
        )
        ont_reads = simulate_long_reads(diploid.hapA, ont_config, 'ont')
        ont_reads_b = simulate_long_reads(diploid.hapB, ont_config, 'ont')
        all_ont_reads = ont_reads + ont_reads_b
        
        # Ultra-long ONT reads
        ul_config = ULConfig(
            coverage=config.ul_coverage,
            read_length_mean=100000,
            read_length_std=25000,
            error_rate=0.08,
            indel_fraction=0.75,
            random_seed=42 + i
        )
        ul_reads = simulate_long_reads(diploid.hapA, ul_config, 'ultra_long')
        ul_reads_b = simulate_long_reads(diploid.hapB, ul_config, 'ultra_long')
        all_ul_reads = ul_reads + ul_reads_b
        
        # Hi-C reads
        hic_config = HiCConfig(
            num_pairs=int(config.genome_size * config.hic_coverage / 300),  # Estimate pairs from coverage
            read_length=150,
            cis_fraction=0.90,
            distance_decay_rate=1.0,
            random_seed=42 + i
        )
        hic_reads = simulate_hic_reads(diploid.hapA, diploid.hapB, hic_config)
        
        logger.info(f"[{genome_id}]   Illumina: {len(all_ill_reads)} pairs")
        logger.info(f"[{genome_id}]   HiFi: {len(all_hifi_reads)} reads")
        logger.info(f"[{genome_id}]   ONT: {len(all_ont_reads)} reads")
        logger.info(f"[{genome_id}]   UL-ONT: {len(all_ul_reads)} reads")
        logger.info(f"[{genome_id}]   Hi-C: {len(hic_reads)} pairs")
        
        # Step 3: Build assembly graph
        logger.info(f"[{genome_id}] Step 3/6: Building assembly graph...")
        graph, accurate_reads, node_to_reads = build_assembly_graph_from_reads(
            all_ill_reads,
            all_hifi_reads,
            kmer_config,
            genome_id
        )
        
        node_ids, edge_list = extract_graph_structure(graph)
        logger.info(f"[{genome_id}]   Graph: {len(node_ids)} nodes, {len(edge_list)} edges")
        
        # Step 4: Generate ground-truth labels
        logger.info(f"[{genome_id}] Step 4/6: Generating ground-truth labels...")
        
        # Convert SimulatedRead objects to list for labeler
        all_simulated_reads = (
            [pair.read1 for pair in all_ill_reads] + 
            [pair.read2 for pair in all_ill_reads] +
            all_hifi_reads + 
            all_ont_reads
        )
        
        # Convert node_to_reads keys to string for labeler
        node_to_reads_str = {str(k): v for k, v in node_to_reads.items()}
        
        labels = generate_ground_truth_labels(
            simulated_reads=all_simulated_reads,
            ul_reads=all_ul_reads,
            reference_hapA=diploid.hapA,
            reference_hapB=diploid.hapB,
            sv_truth_table=diploid.sv_truth_table,
            graph_edges=edge_list,
            graph_nodes=node_ids,
            node_to_read_ids=node_to_reads_str
        )
        
        logger.info(f"[{genome_id}]   Labels: {len(labels.edge_labels)} edges, "
                    f"{len(labels.node_labels)} nodes, {len(labels.path_labels)} paths, "
                    f"{len(labels.ul_route_labels)} UL routes, {len(labels.sv_labels)} SVs")
        
        # Step 5: Extract ML features
        logger.info(f"[{genome_id}] Step 5/6: Extracting ML features...")
        
        dataset = build_training_dataset(
            simulated_reads=all_simulated_reads,
            ul_reads=all_ul_reads,
            graph_nodes=node_ids,
            graph_edges=edge_list,
            ground_truth_labels=labels
        )
        
        logger.info(f"[{genome_id}]   Features: {len(dataset.overlap_features)} overlap, "
                    f"{len(dataset.diploid_features)} diploid, "
                    f"{len(dataset.ul_routing_features)} UL routing, "
                    f"{len(dataset.sv_features)} SV")
        
        # Update totals BEFORE freeing memory
        total_stats['total_graph_nodes'] += len(node_ids)
        total_stats['total_graph_edges'] += len(edge_list)
        total_stats['total_labels'] += (
            len(labels.edge_labels) + 
            len(labels.node_labels) + 
            len(labels.path_labels) +
            len(labels.ul_route_labels) +
            len(labels.sv_labels)
        )
        total_stats['total_features'] += (
            len(dataset.overlap_features) +
            len(dataset.diploid_features) +
            len(dataset.ul_routing_features) +
            len(dataset.sv_features)
        )
        
        # Save per-genome stats
        total_stats['genomes'].append({
            'id': genome_id,
            'size': config.genome_size,
            'svs': len(diploid.sv_truth_table),
            'reads': {
                'illumina': len(all_ill_reads),
                'hifi': len(all_hifi_reads),
                'ont': len(all_ont_reads),
                'ul': len(all_ul_reads),
                'hic': len(hic_reads)
            },
            'graph': {'nodes': len(node_ids), 'edges': len(edge_list)},
            'labels': {
                'edges': len(labels.edge_labels),
                'nodes': len(labels.node_labels),
                'paths': len(labels.path_labels),
                'ul_routes': len(labels.ul_route_labels),
                'svs': len(labels.sv_labels)
            },
            'features': {
                'overlap': len(dataset.overlap_features),
                'diploid': len(dataset.diploid_features),
                'ul_routing': len(dataset.ul_routing_features),
                'sv': len(dataset.sv_features)
            }
        })
        
        # Step 6 (per-genome): Write genome data immediately to disk
        logger.info(f"[{genome_id}] Step 6/6: Writing genome data to disk...")
        genome_output_dir = output_path / 'per_genome' / genome_id
        genome_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write features as pickle files (lightweight, temporary storage)
        import pickle
        features_file = genome_output_dir / 'features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                'overlap_features': dataset.overlap_features,
                'diploid_features': dataset.diploid_features,
                'ul_routing_features': dataset.ul_routing_features,
                'sv_features': dataset.sv_features,
                'gnn_tensors': dataset.gnn_tensors
            }, f)
        
        # Save minimal metadata (don't keep large objects in memory)
        genome_data.append({
            'genome_id': genome_id,
            'features_file': str(features_file)
        })
        
        logger.info(f"[{genome_id}]   Wrote features to {features_file.name}")
        
        # Free memory by deleting large objects
        del diploid, all_ill_reads, all_hifi_reads, all_ont_reads, all_ul_reads, hic_reads
        del graph, labels, dataset, node_ids, edge_list
        import gc
        gc.collect()
    
    # ========================================================================
    # PHASE 2: Stream features from disk and write sharded datasets incrementally
    # ========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Final aggregation: Streaming {len(genome_data)} genomes to sharded output...")
    logger.info(f"Memory-optimized: Processing one genome at a time")
    logger.info(f"{'='*80}")
    
    # Initialize output directories for each split
    for split_name in ['train', 'val', 'test']:
        for model_type in ['overlap_classifier', 'gnn_path_predictor', 'haplotype_detangler', 'ul_routing', 'sv_detection']:
            split_dir = output_path / split_name / model_type
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine train/val/test split for each genome
    import random
    random.seed(42)
    genome_indices = list(range(len(genome_data)))
    random.shuffle(genome_indices)
    
    num_train = int(len(genome_data) * config.train_split)
    num_val = int(len(genome_data) * config.val_split)
    
    train_indices = set(genome_indices[:num_train])
    val_indices = set(genome_indices[num_train:num_train + num_val])
    test_indices = set(genome_indices[num_train + num_val:])
    
    # Counters for tracking
    feature_counts = {
        'overlap': 0,
        'diploid': 0,
        'ul_routing': 0,
        'sv': 0
    }
    
    shard_counters = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    shard_buffers = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    def write_shard_buffer(split_name, model_type, buffer, shard_idx):
        """Write a buffer to a shard file."""
        if not buffer:
            return
        
        split_dir = output_path / split_name / model_type
        shard_file = split_dir / f'shard_{shard_idx:04d}.jsonl'
        
        with open(shard_file, 'w') as f:
            for example in buffer:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"  Wrote {len(buffer)} examples to {split_name}/{model_type}/shard_{shard_idx:04d}.jsonl")
    
    # Process each genome and write to shards incrementally
    import pickle
    
    for idx, gd in enumerate(genome_data):
        logger.info(f"Processing features from {gd['genome_id']} ({idx+1}/{len(genome_data)})...")
        
        # Determine split
        if idx in train_indices:
            split_name = 'train'
        elif idx in val_indices:
            split_name = 'val'
        else:
            split_name = 'test'
        
        # Load features from disk
        with open(gd['features_file'], 'rb') as f:
            features = pickle.load(f)
        
        # Process overlap features
        for feat in features['overlap_features']:
            shard_buffers[split_name]['overlap_classifier'].append(feat.to_dict())
            feature_counts['overlap'] += 1
            
            if len(shard_buffers[split_name]['overlap_classifier']) >= config.shard_size:
                write_shard_buffer(split_name, 'overlap_classifier', 
                                 shard_buffers[split_name]['overlap_classifier'],
                                 shard_counters[split_name]['overlap_classifier'])
                shard_counters[split_name]['overlap_classifier'] += 1
                shard_buffers[split_name]['overlap_classifier'] = []
        
        # Process diploid features
        for feat in features['diploid_features']:
            shard_buffers[split_name]['haplotype_detangler'].append(feat.to_dict())
            feature_counts['diploid'] += 1
            
            if len(shard_buffers[split_name]['haplotype_detangler']) >= config.shard_size:
                write_shard_buffer(split_name, 'haplotype_detangler',
                                 shard_buffers[split_name]['haplotype_detangler'],
                                 shard_counters[split_name]['haplotype_detangler'])
                shard_counters[split_name]['haplotype_detangler'] += 1
                shard_buffers[split_name]['haplotype_detangler'] = []
        
        # Process UL routing features
        for feat in features['ul_routing_features']:
            shard_buffers[split_name]['ul_routing'].append(feat.to_dict())
            feature_counts['ul_routing'] += 1
            
            if len(shard_buffers[split_name]['ul_routing']) >= config.shard_size:
                write_shard_buffer(split_name, 'ul_routing',
                                 shard_buffers[split_name]['ul_routing'],
                                 shard_counters[split_name]['ul_routing'])
                shard_counters[split_name]['ul_routing'] += 1
                shard_buffers[split_name]['ul_routing'] = []
        
        # Process SV features
        for feat in features['sv_features']:
            shard_buffers[split_name]['sv_detection'].append(feat.to_dict())
            feature_counts['sv'] += 1
            
            if len(shard_buffers[split_name]['sv_detection']) >= config.shard_size:
                write_shard_buffer(split_name, 'sv_detection',
                                 shard_buffers[split_name]['sv_detection'],
                                 shard_counters[split_name]['sv_detection'])
                shard_counters[split_name]['sv_detection'] += 1
                shard_buffers[split_name]['sv_detection'] = []
        
        # Free memory immediately after processing this genome
        del features
        import gc
        gc.collect()
    
    # Write remaining buffered data
    logger.info("Writing remaining buffered data...")
    for split_name in ['train', 'val', 'test']:
        for model_type in ['overlap_classifier', 'haplotype_detangler', 'ul_routing', 'sv_detection']:
            if shard_buffers[split_name][model_type]:
                write_shard_buffer(split_name, model_type,
                                 shard_buffers[split_name][model_type],
                                 shard_counters[split_name][model_type])
    
    logger.info(f"Streamed dataset:")
    logger.info(f"  Overlap features: {feature_counts['overlap']:,}")
    logger.info(f"  Diploid features: {feature_counts['diploid']:,}")
    logger.info(f"  UL routing features: {feature_counts['ul_routing']:,}")
    logger.info(f"  SV features: {feature_counts['sv']:,}")
    
    # ========================================================================
    # PHASE 3: Save metadata and summary
    # ========================================================================
    
    elapsed = time.time() - start_time
    total_stats['end_time'] = datetime.now().isoformat()
    total_stats['elapsed_seconds'] = elapsed
    
    # Add dataset statistics
    total_stats['dataset_stats'] = {
        'overlap_features': feature_counts['overlap'],
        'diploid_features': feature_counts['diploid'],
        'ul_routing_features': feature_counts['ul_routing'],
        'sv_features': feature_counts['sv']
    }
    
    # Add split distribution
    total_stats['splits'] = {
        'train': {'genome_count': len(train_indices)},
        'val': {'genome_count': len(val_indices)},
        'test': {'genome_count': len(test_indices)}
    }
    
    # Save metadata JSON
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING CORPUS GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Genomes: {config.num_genomes}")
    logger.info(f"Total nodes: {total_stats['total_graph_nodes']:,}")
    logger.info(f"Total edges: {total_stats['total_graph_edges']:,}")
    logger.info(f"Total labels: {total_stats['total_labels']:,}")
    logger.info(f"Total features: {total_stats['total_features']:,}")
    logger.info(f"Dataset features:")
    logger.info(f"  Overlap: {feature_counts['overlap']:,}")
    logger.info(f"  Diploid: {feature_counts['diploid']:,}")
    logger.info(f"  UL Routing: {feature_counts['ul_routing']:,}")
    logger.info(f"  SV Detection: {feature_counts['sv']:,}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Metadata: {metadata_file}")
    logger.info("=" * 80)
    
    return total_stats


# ============================================================================
#                    CONVENIENCE FUNCTIONS
# ============================================================================

def list_scenarios() -> List[str]:
    """List available training scenarios."""
    return list(SCENARIOS.keys())


def get_scenario_info(scenario: str) -> Dict[str, Any]:
    """Get configuration info for a scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return asdict(SCENARIOS[scenario])


def estimate_generation_time(scenario: str) -> Dict[str, float]:
    """
    Estimate generation time for a scenario.
    
    Returns:
        Dictionary with 'minutes', 'seconds', 'disk_mb' estimates
    """
    config = SCENARIOS[scenario]
    
    # Rough time estimates (seconds per genome)
    genome_time = config.genome_size / 100_000  # 1 sec per 100kb
    read_time = (
        config.illumina_coverage / 10 +  # ~3 sec for 30×
        config.hifi_coverage / 10 +
        config.ont_coverage / 10 +
        config.ul_coverage / 5
    )
    graph_time = config.genome_size / 50_000  # 2 sec per 100kb
    label_time = 2  # ~2 sec per genome
    feature_time = 3  # ~3 sec per genome
    
    per_genome = genome_time + read_time + graph_time + label_time + feature_time
    total_seconds = per_genome * config.num_genomes
    
    # Rough disk estimates (MB per genome)
    reads_mb = config.genome_size / 1000 * (
        config.illumina_coverage +
        config.hifi_coverage +
        config.ont_coverage +
        config.ul_coverage
    ) / 100
    features_mb = config.genome_size / 10_000  # ~0.1 MB per 1Mb genome
    total_mb = (reads_mb + features_mb) * config.num_genomes
    
    return {
        'seconds': total_seconds,
        'minutes': total_seconds / 60,
        'hours': total_seconds / 3600,
        'disk_mb': total_mb,
        'disk_gb': total_mb / 1024
    }
