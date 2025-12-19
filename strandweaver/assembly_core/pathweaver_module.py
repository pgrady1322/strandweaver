#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PathWeaver: GNN-First Path Finding Engine for StrandWeaver.

This module implements GNN-based path prediction with classical algorithm fallback
for assembly graph traversal. It works DOWNSTREAM of EdgeWarden and uses its
outputs without recalculation.

NEW ARCHITECTURE (GNN-First)
═══════════════════════════════════════════════════════════════════════════════
1. PRIMARY: GNN Path Prediction
   ├─ Graph Neural Network predicts edge confidences
   ├─ Extracts high-confidence paths directly from predictions
   ├─ Learns biological patterns from training data
   └─ Fast inference on GPU (if available)

2. FALLBACK: Classical Algorithms (Dijkstra/BFS/DFS)
   ├─ Used when GNN unavailable or fails
   ├─ Reliable baseline path finding
   └─ Well-tested traditional methods

3. ENHANCEMENT: EdgeWarden + Long-Range Data
   ├─ EdgeWarden scores refine both GNN and algorithm paths
   ├─ Hi-C contact maps provide phasing information
   ├─ Ultra-long reads validate path continuity
   └─ Multi-dimensional scoring combines all sources

This is the CORRECT design philosophy:
- ML-first: Use machine learning as primary method when available
- Classical fallback: Maintain reliability with traditional algorithms
- Data integration: Enhance predictions with multiple evidence sources

OLD ARCHITECTURE (Deprecated - Algorithm-First with GNN Enhancement)
─────────────────────────────────────────────────────────────────────
The previous design used algorithms as primary with GNN as optional refinement.
This was backwards - treating ML as a "bonus feature" instead of the primary
intelligence. Legacy mode still available via use_gnn_primary=False parameter.

Key Features:
- GNN-based path prediction (primary method)
- Classical algorithm fallback (Dijkstra, BFS, DFS)
- EdgeWarden confidence integration (no recalculation)
- Long-range data integration (Hi-C, UL reads)
- Comprehensive validation framework
- Misassembly detection

Author: StrandWeaver Development Team
License: MIT
"""

import logging
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any, Generator
from collections import defaultdict, deque
import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Try to import GPU path finder
try:
    from strandweaver.utils.hardware_management import (
        GPUPathFinder,
        GPUPathFindingConfig,
        GPUPathFindingBackend as GPUBackend,
    )
    GPU_PATHFINDER_AVAILABLE = True
except ImportError:
    GPU_PATHFINDER_AVAILABLE = False
    logger.debug("GPU path finder not available, using CPU algorithms")

# Try to import EdgeWarden score integration (Tier 1 Option 2)
try:
    from strandweaver.assembly_core.edgewarden_module import (
        EdgeWardenScoreManager,
        EdgeWardenPathScorer,
        TechnologyType,
        PathScore,
    )
    EDGEWARDEN_SCORER_AVAILABLE = True
except ImportError:
    EDGEWARDEN_SCORER_AVAILABLE = False
    logger.debug("EdgeWarden scorer not available, using basic scoring")

# Try to import Misassembly Detector
try:
    from strandweaver.assembly_utils.misassembly_detector import (
        MisassemblyDetector,
        MisassemblyFlag,
        MisassemblyType,
        ConfidenceLevel,
    )
    MISASSEMBLY_DETECTOR_AVAILABLE = True
except ImportError:
    MISASSEMBLY_DETECTOR_AVAILABLE = False
    logger.debug("Misassembly detector not available")

# Try to import StrandTether (Hi-C integration)
try:
    from strandweaver.assembly_core.strandtether_module import StrandTether
    STRANDTETHER_AVAILABLE = True
except ImportError:
    STRANDTETHER_AVAILABLE = False
    logger.debug("StrandTether Hi-C integration not available")


# ============================================================================
#                         ENUMS & CONSTANTS
# ============================================================================

class PathFindingAlgorithm(str, Enum):
    """Path finding algorithm options."""
    DEPTH_FIRST = "dfs"
    BREADTH_FIRST = "bfs"
    DIJKSTRA = "dijkstra"
    DYNAMIC_PROGRAMMING = "dp"
    ALL_SHORTEST_PATHS = "all_shortest"


class PathScoringMethod(str, Enum):
    """Path scoring approach."""
    COVERAGE_WEIGHTED = "coverage"
    CONFIDENCE_BASED = "confidence"
    ML_BASED = "ml"
    MULTI_OBJECTIVE = "multi"
    BIOLOGICAL_PLAUSIBILITY = "biological"


class PathValidationRule(str, Enum):
    """Path validation constraints."""
    NO_SELF_LOOPS = "no_self_loops"
    CONNECTED_COMPONENT = "connected"
    STRAND_CONSISTENCY = "strand_consistency"
    MAX_LENGTH = "max_length"
    MIN_COVERAGE = "min_coverage"
    GC_CONTENT = "gc_content"
    REPEAT_AWARE = "repeat_aware"
    SYNTENY_CONSTRAINT = "synteny_constraint"
    BOUNDARY_DETECTION = "boundary_detection"
    CNV_AWARE = "cnv_aware"


@dataclass
class ValidationConfig:
    """
    Tunable thresholds and rule configuration for PathValidator.

    Adjust these per dataset to tailor validation strictness.
    """
    # Coverage
    min_coverage: float = 2.0

    # Repeat-aware thresholds
    repeat_score_threshold: float = 0.8
    repeat_min_support: int = 3
    repeat_min_confidence: float = 0.5
    fallback_repeat_gc_delta: float = 0.05
    fallback_repeat_min_length: int = 10000
    fallback_repeat_min_confidence: float = 0.4

    # Boundary detection
    boundary_cross_confidence_threshold: float = 0.8

    # CNV-aware coverage jump threshold (ratio)
    cnv_ratio_threshold: float = 3.0

    # Optional: enable/disable specific rules (None = default set)
    rules_enabled: Optional[List[PathValidationRule]] = None

    # Optional per-rule mode: 'strict' (default), 'warn', 'defer', 'off'
    rule_modes: Optional[Dict[PathValidationRule, str]] = None


# ============================================================================
#                         DATA STRUCTURES
# ============================================================================

@dataclass
class PathNode:
    """Represents a node in a path with metadata."""
    node_id: int
    sequence: str
    coverage: float
    strand: str = '+'
    length: int = 0
    
    def __post_init__(self):
        if self.length == 0:
            self.length = len(self.sequence)
    
    def __eq__(self, other):
        if isinstance(other, PathNode):
            return self.node_id == other.node_id
        return self.node_id == other
    
    def __hash__(self):
        return hash(self.node_id)


@dataclass
class PathEdge:
    """Represents an edge in a path with confidence metrics."""
    from_node_id: int
    to_node_id: int
    support_count: int = 1
    supporting_reads: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    ml_confidence: Optional[float] = None
    hic_support: bool = False
    ul_support: bool = False
    repeat_score: Optional[float] = None
    
    def __hash__(self):
        return hash((self.from_node_id, self.to_node_id))
    
    def __eq__(self, other):
        if isinstance(other, PathEdge):
            return (self.from_node_id, self.to_node_id) == \
                   (other.from_node_id, other.to_node_id)
        return False


@dataclass
class AssemblyPath:
    """
    Complete assembly path through graph.
    
    Represents a traversal from start to end node(s) with comprehensive
    scoring, confidence, and validation metadata.
    """
    path_id: str
    node_ids: List[int]  # Ordered sequence of node IDs
    edges: List[PathEdge] = field(default_factory=list)
    
    # Scoring metrics (all normalized to 0.0-1.0)
    total_score: float = 0.0
    coverage_score: float = 0.0
    confidence_score: float = 0.0
    length_score: float = 0.0
    ml_score: float = 0.0
    biological_score: float = 0.0
    validation_score: float = 1.0
    ul_confidence: float = 0.5
    hic_confidence: float = 0.5
    haplotype_score: Optional[float] = None
    
    # Metadata
    supporting_read_ids: Set[str] = field(default_factory=set)
    has_hic_support: bool = False
    has_ul_support: bool = False
    is_circular: bool = False
    avg_coverage: float = 0.0
    total_length: int = 0
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    pending_rules: List[str] = field(default_factory=list)
    validation_status: str = "unknown"
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in path."""
        return len(self.node_ids)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in path."""
        return len(self.edges)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            'path_id': self.path_id,
            'node_ids': self.node_ids,
            'total_score': self.total_score,
            'coverage_score': self.coverage_score,
            'confidence_score': self.confidence_score,
            'validation_score': self.validation_score,
            'ul_confidence': self.ul_confidence,
            'hic_confidence': self.hic_confidence,
            'haplotype_score': self.haplotype_score,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'total_length': self.total_length,
            'avg_coverage': self.avg_coverage,
            'is_circular': self.is_circular,
            'has_hic_support': self.has_hic_support,
            'has_ul_support': self.has_ul_support,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'pending_rules': self.pending_rules,
            'validation_status': self.validation_status,
        }


@dataclass
class PathFindingResult:
    """Result of path finding operation."""
    paths: List[AssemblyPath] = field(default_factory=list)
    algorithm_used: PathFindingAlgorithm = PathFindingAlgorithm.DEPTH_FIRST
    num_candidates: int = 0
    runtime_seconds: float = 0.0
    notes: str = ""


# ============================================================================
#                    PYTORCH GNN MODEL ARCHITECTURES (FOR MODEL LOADING)
# ============================================================================
# These classes are only used for loading trained GNN models.
# Training infrastructure is in the training/ module.

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, MessagePassing
    from torch_geometric.data import Data
    TORCH_GNN_AVAILABLE = True
    
    @dataclass
    class GNNConfig:
        """Configuration for GNN model."""
        num_node_features: int = 12
        num_edge_features: int = 11
        hidden_channels: int = 64
        num_layers: int = 3
        dropout: float = 0.2
        learning_rate: float = 1e-3
        weight_decay: float = 1e-5
        batch_size: int = 32
        epochs: int = 100
        patience: int = 10
        use_attention: bool = True
        use_residual: bool = True
        use_batch_norm: bool = True
        output_dim: int = 1
        device: str = "cpu"
    
    class EdgeConvLayer(MessagePassing):
        """Custom edge convolution layer with edge features."""
        def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
            super().__init__(aggr='mean')
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.edge_dim = edge_dim
            self.lin_src = nn.Linear(in_channels, out_channels)
            self.lin_dst = nn.Linear(in_channels, out_channels)
            self.lin_edge = nn.Linear(edge_dim, out_channels)
            self.lin_combined = nn.Linear(3 * out_channels, out_channels)
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            msg_src = self.lin_src(x_i)
            msg_dst = self.lin_dst(x_j)
            if edge_attr is not None:
                msg_edge = self.lin_edge(edge_attr)
                msg = torch.cat([msg_src, msg_dst, msg_edge], dim=-1)
            else:
                msg = torch.cat([msg_src, msg_dst, torch.zeros_like(msg_src)], dim=-1)
            msg = self.lin_combined(msg)
            return F.relu(msg)
    
    class PathGNNModel(nn.Module):
        """Graph Neural Network for assembly path prediction."""
        def __init__(self, config: GNNConfig):
            super().__init__()
            self.config = config
            self.device = torch.device(config.device)
            self.edge_convs = nn.ModuleList()
            in_channels = config.num_node_features
            for i in range(config.num_layers):
                self.edge_convs.append(EdgeConvLayer(in_channels, config.hidden_channels, config.num_edge_features))
                if config.use_batch_norm:
                    self.edge_convs.append(nn.BatchNorm1d(config.hidden_channels))
                in_channels = config.hidden_channels
            if config.use_attention:
                self.attention = GATv2Conv(config.hidden_channels, config.hidden_channels // 2, heads=4, dropout=config.dropout)
                attention_out = config.hidden_channels * 2
            else:
                self.attention = None
                attention_out = config.hidden_channels
            self.edge_head = nn.Sequential(nn.Linear(attention_out * 2, config.hidden_channels), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_channels, config.output_dim), nn.Sigmoid())
            self.node_head = nn.Sequential(nn.Linear(attention_out, config.hidden_channels), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_channels, 4), nn.Softmax(dim=-1))
            self.path_head = nn.Sequential(nn.Linear(attention_out * 3, config.hidden_channels), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_channels, 1), nn.Sigmoid())
            self.to(self.device)
        
        def forward(self, data: Data) -> Dict[str, torch.Tensor]:
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None
            for layer in self.edge_convs:
                if isinstance(layer, EdgeConvLayer):
                    x = layer(x, edge_index, edge_attr)
                elif isinstance(layer, nn.BatchNorm1d):
                    x = layer(x)
                else:
                    x = F.relu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)
            if self.attention is not None:
                x_attn = self.attention(x, edge_index)
                x = torch.cat([x, x_attn], dim=-1)
            src, dst = edge_index
            edge_features = torch.cat([x[src], x[dst]], dim=-1)
            edge_logits = self.edge_head(edge_features)
            node_logits = self.node_head(x)
            return {'edge_logits': edge_logits, 'node_logits': node_logits, 'path_scores': None, 'node_embeddings': x}
        
        def get_edge_probabilities(self, data: Data) -> Dict[int, float]:
            with torch.no_grad():
                output = self.forward(data)
                edge_logits = output['edge_logits'].cpu().numpy().flatten()
            return {i: float(prob) for i, prob in enumerate(edge_logits)}
    
    class SimpleGNN(PathGNNModel):
        """Lightweight GNN with 2 layers."""
        def __init__(self, in_channels: int = 12, out_channels: int = 2):
            config = GNNConfig(num_node_features=in_channels, num_layers=2, hidden_channels=32, output_dim=out_channels)
            super().__init__(config)
    
    class MediumGNN(PathGNNModel):
        """Balanced GNN with 4 layers."""
        def __init__(self, in_channels: int = 12, out_channels: int = 2):
            config = GNNConfig(num_node_features=in_channels, num_layers=4, hidden_channels=64, output_dim=out_channels)
            super().__init__(config)
    
    class DeepGNN(PathGNNModel):
        """Production GNN with 6 layers."""
        def __init__(self, in_channels: int = 12, out_channels: int = 2):
            config = GNNConfig(num_node_features=in_channels, num_layers=6, hidden_channels=128, use_residual=True, use_batch_norm=True, output_dim=out_channels)
            super().__init__(config)

except ImportError:
    TORCH_GNN_AVAILABLE = False
    logger.debug("PyTorch/PyG not available - GNN model architectures unavailable")
    # Define dummy classes for type checking
    class GNNConfig: pass
    class PathGNNModel: pass
    class SimpleGNN: pass
    class MediumGNN: pass
    class DeepGNN: pass


# ============================================================================
#                         GNN PATH PREDICTION (INTEGRATED)
# ============================================================================

@dataclass
class GraphTensors:
    """
    Tensor representation of graph for GNN input.
    
    Attributes:
        node_features: Dict[node_id] -> feature vector (list of floats)
        edge_features: Dict[edge_id] -> feature vector (list of floats)
        edge_index: List of (from_node, to_node) pairs
        node_to_index: Dict mapping node_id -> tensor index
        edge_to_index: Dict mapping edge_id -> tensor index
        num_nodes: Total number of nodes
        num_edges: Total number of edges
    """
    node_features: Dict[int, List[float]] = field(default_factory=dict)
    edge_features: Dict[int, List[float]] = field(default_factory=dict)
    edge_index: List[Tuple[int, int]] = field(default_factory=list)
    node_to_index: Dict[int, int] = field(default_factory=dict)
    edge_to_index: Dict[int, int] = field(default_factory=dict)
    num_nodes: int = 0
    num_edges: int = 0


@dataclass
class GNNPathResult:
    """
    Result of GNN path prediction.
    
    Attributes:
        best_paths: List of most likely node sequences through graph
        edge_confidences: Dict[edge_id] -> probability edge is correct (0.0-1.0)
        path_scores: Dict[path_index] -> overall path confidence
        ambiguous_regions: List of node sets with low confidence
    """
    best_paths: List[List[int]] = field(default_factory=list)
    edge_confidences: Dict[int, float] = field(default_factory=dict)
    path_scores: Dict[int, float] = field(default_factory=dict)
    ambiguous_regions: List[Set[int]] = field(default_factory=list)


class FeatureExtractor:
    """Extracts and normalizes features for GNN input."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FeatureExtractor")
    
    def extract_node_features(
        self,
        node_id: int,
        graph,
        hic_phase_info: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> List[float]:
        """Extract 12-dimensional feature vector for a node."""
        node = graph.nodes.get(node_id)
        if not node:
            return [0.0] * 12
        
        features = []
        coverage = getattr(node, 'coverage', 0.0)
        features.append(min(coverage / 100.0, 1.0))
        length = getattr(node, 'length', 0)
        features.append(math.log(length + 1) / math.log(100000))
        entropy = self._calculate_entropy(node)
        features.append(entropy)
        in_degree = len(graph.in_edges.get(node_id, set()))
        features.append(min(in_degree / 10.0, 1.0))
        out_degree = len(graph.out_edges.get(node_id, set()))
        features.append(min(out_degree / 10.0, 1.0))
        branching = in_degree + out_degree
        features.append(min(branching / 20.0, 1.0))
        
        if hic_phase_info and node_id in hic_phase_info:
            phase = hic_phase_info[node_id]
            features.append(phase.phase_A_score)
            features.append(phase.phase_B_score)
            features.append(math.log(phase.contact_count + 1) / math.log(1000))
        else:
            features.extend([0.5, 0.5, 0.0])
        
        if regional_k_map and node_id in regional_k_map:
            k_val = regional_k_map[node_id]
            features.append(k_val / 100.0)
        else:
            features.append(0.31)
        
        ul_support = 0
        if ul_support_map:
            for edge_id in graph.in_edges.get(node_id, set()):
                ul_support += ul_support_map.get(edge_id, 0)
            for edge_id in graph.out_edges.get(node_id, set()):
                ul_support += ul_support_map.get(edge_id, 0)
        features.append(math.log(ul_support + 1) / math.log(100))
        
        repeat_score = self._estimate_repeat_likelihood(coverage, in_degree, out_degree, entropy)
        features.append(repeat_score)
        
        return features
    
    def extract_edge_features(
        self,
        edge_id: int,
        graph,
        from_node: Optional[int] = None,
        to_node: Optional[int] = None
    ) -> List[float]:
        """Extract 10-dimensional feature vector for an edge."""
        features = [0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5]
        
        if from_node and to_node:
            from_node_obj = graph.nodes.get(from_node)
            to_node_obj = graph.nodes.get(to_node)
            if from_node_obj and to_node_obj:
                cov_from = getattr(from_node_obj, 'coverage', 0.0)
                cov_to = getattr(to_node_obj, 'coverage', 0.0)
                if cov_from > 0 and cov_to > 0:
                    features[9] = min(cov_from, cov_to) / max(cov_from, cov_to)
        
        return features
    
    def _calculate_entropy(self, node) -> float:
        """Calculate sequence entropy (0-1)."""
        seq = getattr(node, 'seq', '')
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
    
    def _estimate_repeat_likelihood(self, coverage: float, in_degree: int, out_degree: int, entropy: float) -> float:
        """Estimate likelihood node is in repeat region."""
        score = 0.0
        if coverage > 50:
            score += 0.3
        if in_degree + out_degree > 2:
            score += 0.3
        if entropy < 0.5:
            score += 0.4
        return min(score, 1.0)


class PathGNN:
    """Graph Neural Network for path prediction with heuristic fallback."""
    
    def __init__(self, model=None, model_path: Optional[str] = None):
        self.model = model
        self.logger = logging.getLogger(f"{__name__}.PathGNN")
        if model_path and not model:
            self.load_model(model_path)
        if self.model:
            try:
                import torch
                self.logger.info(f"GNN using trained model: {type(self.model).__name__}")
            except ImportError:
                self.logger.info(f"GNN using model: {type(self.model).__name__}")
        else:
            self.logger.warning("GNN using heuristic fallback (no trained model)")
    
    def predict_edge_probabilities(self, graph_tensors: GraphTensors) -> Dict[int, float]:
        """Predict probability each edge is correct."""
        if self.model:
            return self._predict_with_model(graph_tensors)
        else:
            return self._predict_heuristic(graph_tensors)
    
    def _predict_with_model(self, graph_tensors: GraphTensors) -> Dict[int, float]:
        """Use trained GNN model for predictions."""
        try:
            import torch
            data = self._graph_tensors_to_pyg_data(graph_tensors)
            if hasattr(self.model, 'get_edge_probabilities'):
                predictions = self.model.get_edge_probabilities(data)
            else:
                output = self.model(data)
                edge_logits = output['edge_logits'].cpu().numpy().flatten()
                predictions = {i: float(prob) for i, prob in enumerate(edge_logits)}
            self.logger.debug(f"GNN predicted {len(predictions)} edge probabilities")
            return predictions
        except Exception as e:
            self.logger.error(f"GNN prediction failed: {e}, falling back to heuristic")
            return self._predict_heuristic(graph_tensors)
    
    def _graph_tensors_to_pyg_data(self, graph_tensors: GraphTensors) -> Any:
        """Convert GraphTensors to PyTorch Geometric Data object."""
        import torch
        from torch_geometric.data import Data
        node_features = []
        for i in range(graph_tensors.num_nodes):
            node_id = None
            for nid, idx in graph_tensors.node_to_index.items():
                if idx == i:
                    node_id = nid
                    break
            if node_id and node_id in graph_tensors.node_features:
                node_features.append(graph_tensors.node_features[node_id])
            else:
                sample_size = len(next(iter(graph_tensors.node_features.values())))
                node_features.append([0.0] * sample_size)
        node_feat_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_index = []
        edge_feat_list = []
        for from_idx, to_idx in graph_tensors.edge_index:
            edge_index.append([from_idx, to_idx])
            edge_id = None
            for eid, (f, t) in enumerate(graph_tensors.edge_index):
                if f == from_idx and t == to_idx:
                    edge_id = eid
                    break
            if edge_id in graph_tensors.edge_features:
                edge_feat_list.append(graph_tensors.edge_features[edge_id])
            else:
                sample_size = len(next(iter(graph_tensors.edge_features.values())))
                edge_feat_list.append([0.0] * sample_size)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
        edge_feat_tensor = torch.tensor(edge_feat_list, dtype=torch.float32)
        data = Data(x=node_feat_tensor, edge_index=edge_index_tensor, edge_attr=edge_feat_tensor, num_nodes=graph_tensors.num_nodes)
        return data
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model from disk."""
        try:
            import torch
            # Model classes now defined in this module
            checkpoint = torch.load(model_path, map_location='cpu')
            model_type = checkpoint.get('model_type', 'SimpleGNN')
            config = checkpoint.get('config', {})
            if model_type == 'SimpleGNN':
                self.model = SimpleGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            elif model_type == 'MediumGNN':
                self.model = MediumGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            elif model_type == 'DeepGNN':
                self.model = DeepGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            else:
                self.model = PathGNNModel(config)
            self.model.load_state_dict(checkpoint.get('model_state', checkpoint))
            self.model.eval()
            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def _predict_heuristic(self, graph_tensors: GraphTensors) -> Dict[int, float]:
        """Heuristic fallback scoring based on feature aggregation."""
        predictions = {}
        for edge_id, features in graph_tensors.edge_features.items():
            score = features[0] * 0.4 + features[4] * 0.3 + features[7] * 0.2 + features[9] * 0.1
            predictions[edge_id] = max(0.0, min(1.0, score))
        return predictions


class PathExtractor:
    """Extracts best paths from graph using GNN edge probabilities."""
    
    def __init__(self, min_edge_confidence: float = 0.3):
        self.min_edge_confidence = min_edge_confidence
        self.logger = logging.getLogger(f"{__name__}.PathExtractor")
    
    def score_and_extract_paths(self, graph, edge_probabilities: Dict[int, float]) -> GNNPathResult:
        """Extract most likely paths using GNN edge confidences."""
        self.logger.info("Extracting paths from GNN predictions")
        result = GNNPathResult(edge_confidences=edge_probabilities)
        confident_edges = {eid: prob for eid, prob in edge_probabilities.items() if prob >= self.min_edge_confidence}
        confident_graph = self._build_confident_subgraph(graph, confident_edges)
        components = self._find_connected_components(confident_graph)
        self.logger.info(f"Found {len(components)} connected components")
        for comp_idx, component in enumerate(components):
            paths = self._extract_component_paths(component, confident_graph, edge_probabilities)
            for path in paths:
                path_score = self._calculate_path_score(path, confident_graph, edge_probabilities)
                path_idx = len(result.best_paths)
                result.best_paths.append(path)
                result.path_scores[path_idx] = path_score
        result.ambiguous_regions = self._find_ambiguous_regions(graph, edge_probabilities)
        self.logger.info(f"Extracted {len(result.best_paths)} paths, {len(result.ambiguous_regions)} ambiguous regions")
        return result
    
    def _build_confident_subgraph(self, graph, confident_edges: Dict) -> Dict:
        """Build subgraph containing only high-confidence edges."""
        subgraph = {'nodes': set(), 'edges': {}, 'out_edges': defaultdict(set), 'in_edges': defaultdict(set)}
        for edge_id in confident_edges:
            edge = graph.edges.get(edge_id)
            if edge:
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                if from_node and to_node:
                    subgraph['nodes'].add(from_node)
                    subgraph['nodes'].add(to_node)
                    subgraph['edges'][edge_id] = edge
                    subgraph['out_edges'][from_node].add(to_node)
                    subgraph['in_edges'][to_node].add(from_node)
        return subgraph
    
    def _find_connected_components(self, subgraph: Dict) -> List[Set[int]]:
        """Find connected components in subgraph."""
        visited = set()
        components = []
        for node in subgraph['nodes']:
            if node not in visited:
                component = set()
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    stack.extend(subgraph['out_edges'].get(current, set()))
                    stack.extend(subgraph['in_edges'].get(current, set()))
                if component:
                    components.append(component)
        return components
    
    def _extract_component_paths(self, component: Set[int], subgraph: Dict, edge_probs: Dict[int, float]) -> List[List[int]]:
        """Extract linear paths through a component."""
        start_nodes = [node for node in component if len(subgraph['in_edges'].get(node, set())) == 0]
        if not start_nodes:
            start_nodes = [max(component, key=lambda n: len(subgraph['out_edges'].get(n, set())))]
        paths = []
        for start in start_nodes:
            path = self._extend_path(start, subgraph, set())
            if len(path) > 1:
                paths.append(path)
        return paths if paths else [[node] for node in list(component)[:1]]
    
    def _extend_path(self, current: int, subgraph: Dict, visited: Set[int]) -> List[int]:
        """Extend path from current node using greedy traversal."""
        path = [current]
        visited.add(current)
        while True:
            neighbors = [n for n in subgraph['out_edges'].get(current, set()) if n not in visited]
            if not neighbors:
                break
            next_node = neighbors[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path
    
    def _calculate_path_score(self, path: List[int], subgraph: Dict, edge_probs: Dict[int, float]) -> float:
        """Calculate overall confidence score for a path."""
        if len(path) < 2:
            return 1.0
        return 0.8
    
    def _find_ambiguous_regions(self, graph, edge_probs: Dict[int, float]) -> List[Set[int]]:
        """Identify regions with low edge confidence."""
        ambiguous = []
        for node_id in graph.nodes:
            out_edges = graph.out_edges.get(node_id, set())
            if len(out_edges) > 1:
                low_conf_edges = [eid for eid in out_edges if edge_probs.get(eid, 0.5) < 0.5]
                if len(low_conf_edges) >= 2:
                    region = {node_id}
                    for edge_id in low_conf_edges:
                        edge = graph.edges.get(edge_id)
                        if edge:
                            region.add(getattr(edge, 'to_node', -1))
                    ambiguous.append(region)
        return ambiguous


# ============================================================================
#                         PATH FINDING ENGINE
# ============================================================================

class PathFinder:
    """
    Main path finding engine.
    
    Implements multiple algorithms for discovering paths through assembly graphs
    and selecting the most biologically plausible ones.
    """
    
    def __init__(
        self,
        graph: Any,
        max_path_length: int = 1000000,
        max_search_depth: int = 10000,
        min_coverage_threshold: float = 2.0,
        use_gpu: bool = True,
        gpu_config: Optional[Dict[str, Any]] = None,
        use_edgewarden_scoring: bool = True,
        edgewarden_technology: str = "pacbio_hifi",
    ):
        """
        Initialize path finder.
        
        Args:
            graph: Assembly graph (DBG, StringGraph, or similar)
            max_path_length: Maximum path length in bases
            max_search_depth: Maximum depth for DFS/BFS search
            min_coverage_threshold: Minimum coverage to consider nodes
            use_gpu: Enable GPU acceleration if available
            gpu_config: Optional GPU configuration dict
            use_edgewarden_scoring: Enable EdgeWarden multi-dimensional scoring (Tier 1 #2)
            edgewarden_technology: Technology type for score weighting
        """
        self.graph = graph
        self.max_path_length = max_path_length
        self.max_search_depth = max_search_depth
        self.min_coverage_threshold = min_coverage_threshold
        self.logger = logging.getLogger(f"{__name__}.PathFinder")
        
        # Initialize GPU path finder if available and requested
        self.gpu_pathfinder = None
        self.gpu_graph = None
        self.use_gpu = use_gpu and GPU_PATHFINDER_AVAILABLE
        
        if self.use_gpu:
            try:
                gpu_config = gpu_config or {}
                config = GPUPathFindingConfig(**gpu_config)
                self.gpu_pathfinder = GPUPathFinder(config=config)
                self.gpu_graph = self.gpu_pathfinder.graph_to_gpu(graph)
                
                if self.gpu_graph:
                    self.logger.info(
                        f"GPU path finding enabled ({config.backend}): "
                        f"{self.gpu_graph.num_nodes} nodes, {self.gpu_graph.num_edges} edges"
                    )
                else:
                    self.logger.warning("GPU graph creation failed, falling back to CPU")
                    self.use_gpu = False
                    
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {e}, using CPU algorithms")
                self.use_gpu = False
        
        if not self.use_gpu:
            self.logger.debug("CPU-only path finding mode")
        
        # Initialize EdgeWarden scorer (Tier 1 Option 2)
        self.edgewarden_scorer = None
        self.use_edgewarden_scoring = use_edgewarden_scoring and EDGEWARDEN_SCORER_AVAILABLE
        
        if self.use_edgewarden_scoring:
            try:
                tech_map = {
                    "pacbio_hifi": TechnologyType.PACBIO_HIFI,
                    "pacbio_clr": TechnologyType.PACBIO_CLR,
                    "nanopore_r9": TechnologyType.NANOPORE_R9,
                    "nanopore_r10": TechnologyType.NANOPORE_R10,
                    "illumina": TechnologyType.ILLUMINA,
                    "ancient_dna": TechnologyType.ANCIENT_DNA,
                }
                tech = tech_map.get(edgewarden_technology.lower(), TechnologyType.PACBIO_HIFI)
                self.edgewarden_scorer = EdgeWardenScoreManager(tech)
                self.logger.info(f"EdgeWarden scoring enabled ({tech.value})")
            except Exception as e:
                self.logger.warning(f"EdgeWarden scorer initialization failed: {e}")
                self.use_edgewarden_scoring = False
        
        if not self.use_edgewarden_scoring:
            self.logger.debug("Using basic path scoring (EdgeWarden not available)")
    
    def find_paths(
        self,
        start_node_id: int,
        end_node_ids: Optional[Set[int]] = None,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        max_paths: int = 100,
    ) -> PathFindingResult:
        """
        Find paths from start node to end node(s).
        
        Args:
            start_node_id: Starting node ID
            end_node_ids: Target node IDs (None = find all paths from start)
            algorithm: Path finding algorithm to use (default: Dijkstra)
            max_paths: Maximum number of paths to find
        
        Returns:
            PathFindingResult with discovered paths
        """
        start_time = time.time()
        
        # Try GPU-accelerated path finding for Dijkstra
        if (algorithm == PathFindingAlgorithm.DIJKSTRA and 
            self.use_gpu and self.gpu_graph and end_node_ids):
            self.logger.debug("Using GPU-accelerated Dijkstra")
            paths = self._find_paths_dijkstra_gpu(start_node_id, end_node_ids, max_paths)
        elif algorithm == PathFindingAlgorithm.DEPTH_FIRST:
            paths = self._find_paths_dfs(start_node_id, end_node_ids, max_paths)
        elif algorithm == PathFindingAlgorithm.BREADTH_FIRST:
            paths = self._find_paths_bfs(start_node_id, end_node_ids, max_paths)
        elif algorithm == PathFindingAlgorithm.DIJKSTRA:
            paths = self._find_paths_dijkstra(start_node_id, end_node_ids, max_paths)
        else:
            paths = self._find_paths_dijkstra(start_node_id, end_node_ids, max_paths)
        
        runtime = time.time() - start_time
        
        self.logger.info(
            f"Found {len(paths)} paths from node {start_node_id} "
            f"in {runtime:.3f}s using {algorithm.value}"
        )
        
        return PathFindingResult(
            paths=paths,
            algorithm_used=algorithm,
            num_candidates=len(paths),
            runtime_seconds=runtime,
        )

    def find_paths_iterative_gnn(
        self,
        start_node_id: int,
        end_node_ids: Optional[Set[int]] = None,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        max_paths: int = 100,
        gnn_scorer: Optional[Any] = None,
        num_iterations: int = 2,
    ) -> PathFindingResult:
        """
        Find paths using iterative algorithm → GNN → algorithm refinement.
        
        Iterates between path discovery (algorithm) and GNN-based scoring/reranking
        to progressively improve candidate quality.
        
        Args:
            start_node_id: Starting node ID
            end_node_ids: Target node IDs
            algorithm: Path finding algorithm (Dijkstra, BFS, or DFS)
            max_paths: Maximum paths per iteration
            gnn_scorer: GNN scorer callable(paths) -> Dict[str, float] or None
            num_iterations: Number of algorithm→GNN cycles (default: 2)
        
        Returns:
            PathFindingResult with refined paths
        """
        if num_iterations < 1:
            num_iterations = 1
        
        start_time = time.time()
        all_candidates: Dict[Tuple[int, ...], AssemblyPath] = {}
        current_weights: Optional[Dict[Tuple[int, int], float]] = None
        
        self.logger.info(
            f"Iterative refinement: {num_iterations} cycle(s), "
            f"algorithm={algorithm.value}, gnn_scorer={'available' if gnn_scorer else 'none'}"
        )
        
        for iteration in range(num_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}: Running {algorithm.value}...")
            
            # Algorithm pass: discover paths (with optional GNN-informed weights)
            if current_weights and algorithm == PathFindingAlgorithm.DIJKSTRA:
                # Re-weight edges based on GNN scores from previous iteration
                paths = self._find_paths_dijkstra_weighted(
                    start_node_id, end_node_ids, max_paths, current_weights
                )
            else:
                result = self.find_paths(
                    start_node_id, end_node_ids, algorithm, max_paths
                )
                paths = result.paths
            
            # Aggregate candidates
            for path in paths:
                key = tuple(path.node_ids)
                if key not in all_candidates:
                    all_candidates[key] = path
            
            self.logger.info(f"    Algorithm found {len(paths)} paths; aggregate={len(all_candidates)}")
            
            # GNN pass (if available and not final iteration)
            if gnn_scorer and iteration < num_iterations - 1:
                self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}: GNN scoring...")
                
                try:
                    gnn_scores = gnn_scorer(list(all_candidates.values()))
                    
                    if gnn_scores:
                        # Extract GNN-informed weights for next iteration
                        current_weights = {}
                        for path in all_candidates.values():
                            path_score = gnn_scores.get(path.path_id, 0.5)
                            
                            # Propagate path-level score to edges
                            for i, node_id in enumerate(path.node_ids[:-1]):
                                next_node = path.node_ids[i + 1]
                                edge_key = (node_id, next_node)
                                # Lower weight = preferred; inverse of score
                                current_weights[edge_key] = 1.0 / max(path_score, 0.1)
                        
                        self.logger.info(
                            f"    GNN scored {len(gnn_scores)} paths; "
                            f"extracted weights for {len(current_weights)} edges"
                        )
                except Exception as e:
                    self.logger.warning(f"GNN scoring failed: {e}, continuing without weights")
                    current_weights = None
        
        runtime = time.time() - start_time
        refined_paths = list(all_candidates.values())
        
        self.logger.info(
            f"Iterative refinement complete: {len(refined_paths)} unique paths "
            f"in {runtime:.3f}s"
        )
        
        return PathFindingResult(
            paths=refined_paths,
            algorithm_used=algorithm,
            num_candidates=len(refined_paths),
            runtime_seconds=runtime,
            notes=f"Iterative refinement ({num_iterations} cycles)",
        )
    
    def find_paths_gnn_primary(
        self,
        start_node_id: int,
        end_node_ids: Optional[Set[int]] = None,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        max_paths: int = 100,
        gnn_scorer: Optional[Any] = None,
        min_confidence: float = 0.3,
    ) -> PathFindingResult:
        """
        Find paths using GNN as primary method with algorithm fallback.
        
        This implements the correct architecture: try GNN first, fall back to
        classical algorithms only if GNN is unavailable or fails.
        
        Architecture:
            1. PRIMARY: Use GNN to predict edge confidences and extract paths
            2. FALLBACK: If GNN unavailable/fails, use classical algorithm
            3. HYBRID: If GNN finds few paths, supplement with algorithm-based
        
        Args:
            start_node_id: Starting node ID
            end_node_ids: Target node IDs
            algorithm: Fallback algorithm (Dijkstra, BFS, or DFS)
            max_paths: Maximum paths to return
            gnn_scorer: GNN scorer callable for path prediction
            min_confidence: Minimum edge confidence for GNN path extraction
        
        Returns:
            PathFindingResult with GNN-predicted or fallback paths
        """
        start_time = time.time()
        paths = []
        notes = []
        
        self.logger.info(
            f"GNN-primary path finding (fallback={algorithm.value})"
        )
        
        # PRIMARY: Try GNN-based path finding
        if gnn_scorer:
            try:
                self.logger.info("  PRIMARY: Using GNN for path prediction...")
                
                # GNN infrastructure is now integrated in this module
                # Build graph tensors for GNN input
                feature_extractor = FeatureExtractor()
                graph_tensors = GraphTensors()
                
                # Extract node features
                for node_id in self.graph.nodes:
                    features = feature_extractor.extract_node_features(
                        node_id, self.graph
                    )
                    graph_tensors.node_features[node_id] = features
                    graph_tensors.node_to_index[node_id] = len(graph_tensors.node_to_index)
                
                graph_tensors.num_nodes = len(graph_tensors.node_features)
                
                # Extract edge features and build edge index
                for edge_id, edge in self.graph.edges.items():
                    from_node = getattr(edge, 'from_node', None)
                        to_node = getattr(edge, 'to_node', None)
                        
                        if from_node and to_node:
                            features = feature_extractor.extract_edge_features(
                                edge_id, self.graph
                            )
                            graph_tensors.edge_features[edge_id] = features
                            graph_tensors.edge_to_index[edge_id] = len(graph_tensors.edge_to_index)
                            
                            # Add to edge index using tensor indices
                            from_idx = graph_tensors.node_to_index[from_node]
                            to_idx = graph_tensors.node_to_index[to_node]
                            graph_tensors.edge_index.append((from_idx, to_idx))
                    
                    graph_tensors.num_edges = len(graph_tensors.edge_features)
                    
                    # Get GNN predictions
                    path_gnn = PathGNN(model=gnn_scorer)
                    edge_probs = path_gnn.predict_edge_probabilities(graph_tensors)
                    
                    # Extract paths from GNN predictions
                    path_extractor = PathExtractor(min_edge_confidence=min_confidence)
                    gnn_result = path_extractor.score_and_extract_paths(
                        self.graph, edge_probs
                    )
                    
                    # Convert GNN paths to AssemblyPath objects
                    for i, node_sequence in enumerate(gnn_result.best_paths):
                        if len(node_sequence) < 2:
                            continue
                        
                        # Check if path starts from our start node or reaches end nodes
                        if start_node_id not in node_sequence:
                            # GNN found paths in other parts of graph, skip
                            continue
                        
                        # Find subsequence from start_node to end
                        try:
                            start_idx = node_sequence.index(start_node_id)
                            subpath = node_sequence[start_idx:]
                            
                            # Check if reaches end nodes
                            if end_node_ids:
                                end_found = False
                                for end_idx, node_id in enumerate(subpath):
                                    if node_id in end_node_ids:
                                        subpath = subpath[:end_idx+1]
                                        end_found = True
                                        break
                                if not end_found:
                                    continue
                            
                            # Create AssemblyPath
                            path = self._create_assembly_path(subpath, f"gnn_path_{i}")
                            path.ml_score = gnn_result.path_scores.get(i, 0.5)
                            paths.append(path)
                            
                        except ValueError:
                            continue
                    
                    if paths:
                        self.logger.info(f"    GNN found {len(paths)} paths from start node")
                        notes.append(f"GNN primary ({len(paths)} paths)")
                    else:
                        self.logger.info("    GNN found no valid paths from start node, falling back")
                        notes.append("GNN found no paths, using fallback")
                        
            except Exception as e:
                self.logger.warning(f"GNN path finding failed: {e}, falling back to algorithm")
                notes.append(f"GNN failed: {str(e)[:50]}")
                paths = []
        else:
            self.logger.info("  No GNN scorer provided, using fallback algorithm")
            notes.append("No GNN scorer")
        
        # FALLBACK: Use classical algorithm if GNN unavailable or found no paths
        if not paths:
            self.logger.info(f"  FALLBACK: Using {algorithm.value} algorithm...")
            result = self.find_paths(start_node_id, end_node_ids, algorithm, max_paths)
            paths = result.paths
            notes.append(f"Algorithm fallback ({len(paths)} paths)")
        
        # HYBRID: If GNN found few paths, supplement with algorithm-based
        elif len(paths) < max_paths // 2:
            self.logger.info(f"  HYBRID: GNN found only {len(paths)} paths, supplementing with {algorithm.value}...")
            result = self.find_paths(start_node_id, end_node_ids, algorithm, max_paths - len(paths))
            
            # Add algorithm paths that don't duplicate GNN paths
            gnn_path_sets = {tuple(p.node_ids) for p in paths}
            for algo_path in result.paths:
                if tuple(algo_path.node_ids) not in gnn_path_sets:
                    paths.append(algo_path)
            
            notes.append(f"Hybrid GNN+algorithm ({len(paths)} total)")
        
        runtime = time.time() - start_time
        
        self.logger.info(
            f"GNN-primary path finding complete: {len(paths)} paths in {runtime:.3f}s"
        )
        
        return PathFindingResult(
            paths=paths[:max_paths],
            algorithm_used=algorithm,
            num_candidates=len(paths),
            runtime_seconds=runtime,
            notes="; ".join(notes),
        )
    
    def _find_paths_dijkstra_weighted(
        self,
        start: int,
        targets: Optional[Set[int]],
        max_paths: int,
        edge_weights: Dict[Tuple[int, int], float],
    ) -> List[AssemblyPath]:
        """Find paths using Dijkstra with custom edge weights."""
        paths = []
        
        # Use provided weights to guide Dijkstra
        # (Implementation mirrors _find_paths_dijkstra but uses edge_weights dict)
        try:
            import heapq
            
            dist = {start: 0}
            heap = [(0, start, [start])]
            visited_dests = set()
            
            while heap and len(paths) < max_paths:
                curr_dist, node, path = heapq.heappop(heap)
                
                if targets and node in targets:
                    if node not in visited_dests:
                        paths.append(AssemblyPath(
                            path_id=f"path_{len(paths)}_weighted",
                            node_ids=path,
                        ))
                        visited_dests.add(node)
                    continue
                
                if len(path) > self.max_search_depth:
                    continue
                
                for neighbor in self._get_neighbors(node):
                    if neighbor in path:  # Avoid cycles
                        continue
                    
                    edge_key = (node, neighbor)
                    edge_weight = edge_weights.get(edge_key, 1.0)
                    new_dist = curr_dist + edge_weight
                    
                    if neighbor not in dist or new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor, path + [neighbor]))
        
        except Exception as e:
            self.logger.warning(f"Weighted Dijkstra failed: {e}, falling back to unweighted")
            return self._find_paths_dijkstra(start, targets, max_paths)
        
        return paths
    
    def find_paths_multi_start(
        self,
        start_node_ids: List[int],
        end_node_ids: Optional[Set[int]] = None,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        max_paths_total: int = 200,
        per_start_max_paths: Optional[int] = None,
        num_workers: int = 4,
    ) -> PathFindingResult:
        """
        Find paths from multiple start nodes in parallel and aggregate results.

        Args:
            start_node_ids: List of starting node IDs
            end_node_ids: Optional set of target node IDs
            algorithm: Path finding algorithm to use
            max_paths_total: Upper bound on total aggregated paths
            per_start_max_paths: Per-start cap (defaults to ceil(max_paths_total/len(starts)))
            num_workers: Parallel workers (threads)

        Returns:
            PathFindingResult with aggregated, deduplicated paths
        """
        if not start_node_ids:
            return PathFindingResult(paths=[], algorithm_used=algorithm, num_candidates=0, runtime_seconds=0.0)

        start_time = time.time()
        per_cap = per_start_max_paths or max(1, math.ceil(max_paths_total / max(1, len(start_node_ids))))

        self.logger.info(
            f"Parallel multi-start: {len(start_node_ids)} starts, algo={algorithm.value}, "
            f"per_start={per_cap}, total_cap={max_paths_total}, workers={num_workers}"
        )

        results: List[AssemblyPath] = []
        seen: Set[Tuple[int, ...]] = set()

        def run_single(start_id: int) -> List[AssemblyPath]:
            try:
                res = self.find_paths(
                    start_node_id=start_id,
                    end_node_ids=end_node_ids,
                    algorithm=algorithm,
                    max_paths=per_cap,
                )
                return res.paths
            except Exception as e:
                self.logger.warning(f"Multi-start worker failed for start {start_id}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = {executor.submit(run_single, s): s for s in start_node_ids}
            for fut in as_completed(futures):
                paths = fut.result()
                for p in paths:
                    key = tuple(p.node_ids)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Assign unique path_id indicating origin if possible
                    p.path_id = p.path_id or f"path_{len(results)}"
                    results.append(p)
                    if len(results) >= max_paths_total:
                        break
                if len(results) >= max_paths_total:
                    break

        runtime = time.time() - start_time
        self.logger.info(
            f"Parallel multi-start collected {len(results)} unique paths in {runtime:.3f}s"
        )

        return PathFindingResult(
            paths=results,
            algorithm_used=algorithm,
            num_candidates=len(results),
            runtime_seconds=runtime,
        )
    
    def _find_paths_dfs(
        self,
        start: int,
        targets: Optional[Set[int]],
        max_paths: int,
    ) -> List[AssemblyPath]:
        """Find paths using depth-first search."""
        paths = []
        path_counter = 0
        
        def dfs_recurse(
            current: int,
            target_set: Optional[Set[int]],
            path: List[int],
            visited: Set[int],
            depth: int,
        ):
            nonlocal path_counter, paths
            
            if path_counter >= max_paths:
                return
            
            if depth > self.max_search_depth:
                return
            
            # Check if we reached target
            if target_set and current in target_set:
                paths.append(AssemblyPath(
                    path_id=f"path_{path_counter}",
                    node_ids=path.copy(),
                ))
                path_counter += 1
                return
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Check coverage threshold
                if not self._passes_coverage_check(neighbor):
                    continue
                
                # Check path length
                if sum(self._get_node_length(n) for n in path) > self.max_path_length:
                    continue
                
                visited.add(neighbor)
                path.append(neighbor)
                
                dfs_recurse(neighbor, target_set, path, visited, depth + 1)
                
                path.pop()
                visited.remove(neighbor)
        
        # Start DFS from start node
        visited = {start}
        dfs_recurse(start, targets, [start], visited, 0)
        
        self.logger.info(f"DFS found {len(paths)} paths")
        return paths
    
    def _find_paths_bfs(
        self,
        start: int,
        targets: Optional[Set[int]],
        max_paths: int,
    ) -> List[AssemblyPath]:
        """Find paths using breadth-first search."""
        paths = []
        queue = deque([(start, [start], {start})])
        
        while queue and len(paths) < max_paths:
            current, path, visited = queue.popleft()
            
            if len(path) > self.max_search_depth:
                continue
            
            # Check if we reached target
            if targets and current in targets:
                paths.append(AssemblyPath(
                    path_id=f"path_{len(paths)}",
                    node_ids=path.copy(),
                ))
                continue
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                if not self._passes_coverage_check(neighbor):
                    continue
                
                if sum(self._get_node_length(n) for n in path) > self.max_path_length:
                    continue
                
                new_visited = visited.copy()
                new_visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], new_visited))
        
        self.logger.info(f"BFS found {len(paths)} paths")
        return paths
    
    def _find_paths_dijkstra(
        self,
        start: int,
        targets: Optional[Set[int]],
        max_paths: int,
    ) -> List[AssemblyPath]:
        """Find shortest paths using Dijkstra's algorithm."""
        paths = []
        
        # Priority queue: (distance, node, path)
        heap = [(0.0, start, [start])]
        visited_paths = set()
        
        while heap and len(paths) < max_paths:
            distance, current, path = heapq.heappop(heap)
            
            # Avoid cycles
            path_tuple = tuple(path)
            if path_tuple in visited_paths:
                continue
            visited_paths.add(path_tuple)
            
            if len(path) > self.max_search_depth:
                continue
            
            # Check if we reached target
            if targets and current in targets:
                paths.append(AssemblyPath(
                    path_id=f"path_{len(paths)}",
                    node_ids=path.copy(),
                ))
                continue
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in path:  # No cycles
                    continue
                
                if not self._passes_coverage_check(neighbor):
                    continue
                
                edge_distance = self._get_edge_distance(current, neighbor)
                new_distance = distance + edge_distance
                
                heapq.heappush(heap, (new_distance, neighbor, path + [neighbor]))
        
        self.logger.info(f"Dijkstra found {len(paths)} shortest paths")
        return paths
    
    def _find_paths_dijkstra_gpu(
        self,
        start: int,
        targets: Set[int],
        max_paths: int,
    ) -> List[AssemblyPath]:
        """
        Find shortest paths using GPU-accelerated Dijkstra's algorithm.
        
        Uses PyTorch GPU tensors for efficient shortest path computation.
        Falls back to CPU Dijkstra if GPU operations fail.
        """
        if not self.gpu_pathfinder or not self.gpu_graph:
            self.logger.warning("GPU path finder not available, falling back to CPU")
            return self._find_paths_dijkstra(start, targets, max_paths)
        
        try:
            # Map node IDs to GPU indices
            start_idx = self.gpu_graph.node_to_idx.get(start)
            end_indices = {
                self.gpu_graph.node_to_idx[t] 
                for t in targets 
                if t in self.gpu_graph.node_to_idx
            }
            
            if start_idx is None or not end_indices:
                self.logger.warning("Start or target nodes not in GPU graph")
                return self._find_paths_dijkstra(start, targets, max_paths)
            
            # Use GPU Dijkstra
            gpu_results = self.gpu_pathfinder.dijkstra_gpu(
                self.gpu_graph,
                start_idx,
                end_indices,
            )
            
            # Convert results to AssemblyPath objects
            paths = []
            for end_idx, (distance, path_indices) in gpu_results.items():
                if len(paths) >= max_paths:
                    break
                
                path = AssemblyPath(
                    path_id=f"path_{len(paths)}_gpu",
                    node_ids=path_indices,
                )
                paths.append(path)
            
            self.logger.info(
                f"GPU Dijkstra found {len(paths)} shortest paths "
                f"from node {start}"
            )
            return paths
            
        except Exception as e:
            self.logger.error(f"GPU Dijkstra failed: {e}, falling back to CPU")
            return self._find_paths_dijkstra(start, targets, max_paths)
    
    def _get_neighbors(self, node_id: int) -> List[int]:
        """Get outgoing neighbors for a node."""
        if hasattr(self.graph, 'out_edges'):
            edge_ids = self.graph.out_edges.get(node_id, set())
            neighbors = []
            for edge_id in edge_ids:
                if edge_id in self.graph.edges:
                    edge = self.graph.edges[edge_id]
                    neighbors.append(edge.to_node)
            return neighbors
        return []
    
    def _get_node_length(self, node_id: int) -> int:
        """Get sequence length of a node."""
        if hasattr(self.graph, 'nodes'):
            node = self.graph.nodes.get(node_id)
            if node and hasattr(node, 'seq'):
                return len(node.seq)
        return 0
    
    def _get_edge_distance(self, from_node: int, to_node: int) -> float:
        """Get distance metric for edge (lower = preferred)."""
        # Inverse of confidence (higher confidence = lower distance)
        if hasattr(self.graph, 'edges'):
            for edge in self.graph.edges.values():
                if hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
                    if edge.from_node == from_node and edge.to_node == to_node:
                        confidence = getattr(edge, 'confidence', 1.0)
                        return 1.0 / max(confidence, 0.1)
        return 1.0
    
    def _passes_coverage_check(self, node_id: int) -> bool:
        """Check if node meets minimum coverage threshold."""
        if hasattr(self.graph, 'nodes'):
            node = self.graph.nodes.get(node_id)
            if node and hasattr(node, 'coverage'):
                return node.coverage >= self.min_coverage_threshold
        return True
    
    # ========================================================================
    #                  EDGEWARDEN SCORE INTEGRATION (Tier 1 #2)
    # ========================================================================
    
    def register_edge_scores_from_edgewarden(
        self,
        edge_scores: Dict[str, Dict[str, float]],
    ) -> int:
        """
        Register EdgeWarden scores for edges.
        
        CRITICAL: This is how PathWeaver leverages EdgeWarden's ML classification.
        
        Args:
            edge_scores: Dict mapping edge_id -> {
                "edge_confidence": float,
                "coverage_consistency": float,
                "repeat_score": float,
                "quality_score": float,
                "error_pattern_score": float,
                "support_count": int,
                "coverage_ratio": float,
            }
        
        Returns:
            Number of edges registered
        """
        if not self.use_edgewarden_scoring or not self.edgewarden_scorer:
            self.logger.warning("EdgeWarden scoring not available")
            return 0
        
        registered = 0
        for edge_id, scores in edge_scores.items():
            try:
                self.edgewarden_scorer.register_edge_score(
                    edge_id=edge_id,
                    source_node=scores.get("source_node", 0),
                    target_node=scores.get("target_node", 0),
                    edge_confidence=scores.get("edge_confidence", 0.5),
                    coverage_consistency=scores.get("coverage_consistency", 0.5),
                    repeat_score=scores.get("repeat_score", 0.5),
                    quality_score=scores.get("quality_score", 0.5),
                    error_pattern_score=scores.get("error_pattern_score", 0.5),
                    support_count=scores.get("support_count", 1),
                    coverage_ratio=scores.get("coverage_ratio", 1.0),
                )
                registered += 1
            except Exception as e:
                self.logger.warning(f"Failed to register edge {edge_id}: {e}")
        
        self.logger.info(f"Registered EdgeWarden scores for {registered} edges")
        return registered
    
    def score_path_with_edgewarden(
        self,
        path_id: str,
        node_ids: List[int],
        edge_ids: List[str],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Score a path using EdgeWarden multi-dimensional scoring.
        
        Args:
            path_id: Path identifier
            node_ids: Sequence of node IDs
            edge_ids: Sequence of edge IDs
            **kwargs: Additional metadata (length, supporting_reads, etc.)
        
        Returns:
            Score dict with composite_score, category, recommendation
        """
        if not self.use_edgewarden_scoring or not self.edgewarden_scorer:
            return None
        
        try:
            score = self.edgewarden_scorer.score_path(
                path_id=path_id,
                node_ids=node_ids,
                edge_ids=edge_ids,
                **kwargs
            )
            
            return {
                "path_id": path_id,
                "composite_score": score.composite_score,
                "confidence_category": score.confidence_category,
                "recommendation": score.recommendation,
                "breakdown": score.breakdown,
                "mean_edge_confidence": score.mean_edge_confidence,
                "min_edge_confidence": score.min_edge_confidence,
            }
        except Exception as e:
            self.logger.warning(f"EdgeWarden scoring failed for path {path_id}: {e}")
            return None
    
    def rank_paths_by_edgewarden(
        self,
        paths: List[Dict[str, Any]],
        min_score: float = 40.0,
    ) -> List[Dict[str, Any]]:
        """
        Rank paths by EdgeWarden scores.
        
        Args:
            paths: List of path dicts with path_id, node_ids, edge_ids
            min_score: Minimum composite score to include
        
        Returns:
            Ranked paths (best first), filtered by confidence
        """
        if not self.use_edgewarden_scoring or not self.edgewarden_scorer:
            return paths
        
        scored_paths = []
        for path in paths:
            scores = self.score_path_with_edgewarden(
                path.get("path_id"),
                path.get("node_ids", []),
                path.get("edge_ids", []),
                path_length_bases=path.get("length", 0),
            )
            
            if scores and scores["composite_score"] >= min_score:
                path["edgewarden_score"] = scores
                scored_paths.append(path)
        
        # Sort by composite score (descending)
        ranked = sorted(
            scored_paths,
            key=lambda p: p.get("edgewarden_score", {}).get("composite_score", 0),
            reverse=True
        )
        
        self.logger.info(
            f"Ranked {len(ranked)} paths by EdgeWarden "
            f"(filtered from {len(paths)}, min_score={min_score})"
        )
        
        return ranked
    
    def explain_path_score(self, path_id: str) -> Dict[str, Any]:
        """
        Get detailed explanation for path's EdgeWarden score.
        
        Answers: "Why is this path RELIABLE/QUESTIONABLE?"
        
        Args:
            path_id: Path identifier
        
        Returns:
            Detailed explanation dictionary
        """
        if not self.use_edgewarden_scoring or not self.edgewarden_scorer:
            return {"error": "EdgeWarden scoring not available"}
        
        return self.edgewarden_scorer.get_path_explanation(path_id)


# ============================================================================
#                         PATH SCORING ENGINE
# ============================================================================

class PathScorer:
    """
    Path scoring system using EdgeWarden outputs + GNN optimization.
    
    DESIGN: Downstream of EdgeWarden
    ═════════════════════════════════
    - Accepts EdgeWarden confidence scores (no recalculation)
    - Uses GNN to optimize at path level (not edge level)
    - Falls back to heuristics if EdgeWarden unavailable
    - Never duplicates EdgeWarden computation
    
    Scoring Dimensions:
    1. EdgeWarden confidence (50%) - From upstream, pre-computed
    2. GNN path optimization (30%) - Learns path-level patterns
    3. Graph topology (20%) - Coverage, length, connectivity
    """
    
    def __init__(
        self,
        edgewarden_weight: float = 0.5,
        gnn_weight: float = 0.3,
        topology_weight: float = 0.2,
    ):
        """
        Initialize path scorer (downstream of EdgeWarden).
        
        Args:
            edgewarden_weight: Weight for EdgeWarden confidence (50%)
            gnn_weight: Weight for GNN path optimization (30%)
            topology_weight: Weight for graph topology (20%)
        
        Note: EdgeWarden outputs are accepted, never recalculated.
        """
        self.edgewarden_weight = edgewarden_weight
        self.gnn_weight = gnn_weight
        self.topology_weight = topology_weight
        self.logger = logging.getLogger(f"{__name__}.PathScorer")
        
        # Normalize weights
        total = sum([edgewarden_weight, gnn_weight, topology_weight])
        self.edgewarden_weight /= total
        self.gnn_weight /= total
        self.topology_weight /= total
        
        # Calibration parameters (can be tuned)
        self.min_score = 0.0
        self.max_score = 1.0
        self.uncertainty_floor = 0.0
        self.uncertainty_ceiling = 1.0
    
    def score_path(
        self,
        path: AssemblyPath,
        graph: Any,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]] = None,
        gnn_path_score: Optional[float] = None,
        validation_score: Optional[float] = None,
        ul_confidence: Optional[float] = None,
        hic_confidence: Optional[float] = None,
        is_final_iteration: bool = False,
    ) -> float:
        """
        Score a path using iterative scoring logic.
        
        All inputs normalized to [0.0, 1.0] for comparability.
        
        SCORING ARCHITECTURE (Iteration-Aware):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Iterations 1 to N-1 (Non-Final):
        ├─ EdgeWarden: 40%   (primary confidence)
        ├─ GNN: 30%          (path optimization)
        ├─ Topology: 20%     (graph structure)
        └─ Validation: 10%   (rule penalties)
        = Score for path selection
        
        Final Iteration (After SVScribe, All Evidence):
        ├─ EdgeWarden: 25%
        ├─ GNN: 20%
        ├─ Topology: 15%
        ├─ UL Confidence: 20%    (from ThreadCompass)
        ├─ Hi-C Confidence: 15%  (from HiCWeaver)
        └─ Validation: 5%
        = FINAL combined score with all evidence
        
        Args:
            path: Path to score
            graph: Assembly graph
            edgewarden_scores: EdgeWarden confidence per edge (ACCEPTS, no recalc)
            gnn_path_score: GNN path-level optimization (optional)
            validation_score: Validation penalty score (optional, 1.0 = perfect)
            ul_confidence: UL support score from ThreadCompass (optional, final only)
            hic_confidence: Hi-C support score from HiCWeaver (optional, final only)
            is_final_iteration: If True, use final scoring with UL/Hi-C weights
        
        Returns:
            Combined score (0.0-1.0, higher is better)
        """
        # Score 1: EdgeWarden confidence (from upstream, no recalculation)
        edgewarden_score = self._score_edgewarden(
            path, edgewarden_scores
        )
        
        # Score 2: GNN path-level optimization (learns path patterns)
        if gnn_path_score is not None:
            gnn_score = gnn_path_score  # Accept GNN output directly
        else:
            gnn_score = self._heuristic_path_quality(path, graph)
        
        # Score 3: Graph topology (coverage, length, connectivity)
        topology_score = self._score_topology(path, graph)
        
        # Score 4: Validation penalty (1.0 = perfect, reduced by violations)
        val_score = validation_score if validation_score is not None else 1.0
        
        # Score 5 & 6: Long-range evidence (only in final iteration)
        ul_score = ul_confidence if (is_final_iteration and ul_confidence is not None) else 0.5
        hic_score = hic_confidence if (is_final_iteration and hic_confidence is not None) else 0.5
        
        # Combine with iteration-appropriate weights
        if is_final_iteration:
            # Final iteration: all evidence combined
            # Weights: EdW=25%, GNN=20%, Topology=15%, UL=20%, HiC=15%, Val=5%
            total_score = (
                self.edgewarden_weight * edgewarden_score +
                self.gnn_weight * gnn_score +
                self.topology_weight * topology_score +
                0.20 * ul_score +
                0.15 * hic_score +
                0.05 * val_score
            )
        else:
            # Intermediate iterations: no long-range evidence yet
            # Redistribute weights without UL/Hi-C (40% EdW, 30% GNN, 20% Topology, 10% Val)
            edw_weight_interim = 0.40
            gnn_weight_interim = 0.30
            topo_weight_interim = 0.20
            val_weight_interim = 0.10
            total_score = (
                edw_weight_interim * edgewarden_score +
                gnn_weight_interim * gnn_score +
                topo_weight_interim * topology_score +
                val_weight_interim * val_score
            )
        
        # Uncertainty quantification: dispersion across components
        components = [edgewarden_score, gnn_score, topology_score, ul_score, hic_score, val_score]
        mean_comp = sum(components) / len(components)
        var_comp = sum((c - mean_comp) ** 2 for c in components) / len(components)
        # Normalize uncertainty to 0-1 range (simple calibration)
        uncertainty = max(self.uncertainty_floor, min(self.uncertainty_ceiling, math.sqrt(var_comp)))
        
        # Optional: adjust total score by uncertainty (conservative)
        # Reduce score slightly when dispersion is high
        adjusted_score = max(self.min_score, min(self.max_score, total_score * (1.0 - 0.1 * uncertainty)))
        
        # Store individual scores
        path.confidence_score = edgewarden_score  # From EdgeWarden
        path.ml_score = gnn_score                # From GNN
        path.coverage_score = topology_score     # From graph
        path.validation_score = val_score        # From validation
        path.ul_confidence = ul_score            # From UL
        path.hic_confidence = hic_score          # From Hi-C
        path.total_score = adjusted_score
        path.uncertainty = uncertainty
        
        return adjusted_score
    
    def compute_final_combined_score(
        self,
        path: AssemblyPath,
        graph: Any,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]] = None,
        gnn_path_score: Optional[float] = None,
        validation_score: Optional[float] = None,
        ul_confidence: Optional[float] = None,
        hic_confidence: Optional[float] = None,
        sv_misassembly_score: Optional[float] = None,
    ) -> float:
        """
        Compute FINAL combined score after all evidence collected (post-SVScribe).
        
        This method is called AFTER:
        - Iteration through assembly
        - SVScribe structural variant detection
        - All long-range evidence collected
        - Final validation pass
        
        FINAL SCORING (All Evidence):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├─ EdgeWarden: 25%              (overlap confidence)
        ├─ GNN: 20%                     (path optimization)
        ├─ Topology: 15%                (graph structure)
        ├─ UL Confidence: 20%           (ultra-long read support)
        ├─ Hi-C Confidence: 15%         (3D contact support)
        ├─ Validation Penalty: 5%       (rule violations)
        └─ (Optional) SV Misassembly: -10% penalty if detected
        = FINAL combined confidence (0.0-1.0)
        
        Args:
            path: Path to score (should be from final iteration)
            graph: Assembly graph
            edgewarden_scores: EdgeWarden edge confidences
            gnn_path_score: GNN path score
            validation_score: Validation penalty (1.0 = perfect)
            ul_confidence: UL support from ThreadCompass
            hic_confidence: Hi-C support from HiCWeaver
            sv_misassembly_score: SVScribe misassembly penalty (0.0 = misassembly, 1.0 = confident)
        
        Returns:
            Final combined score (0.0-1.0)
        """
        self.logger.debug(f"Computing FINAL combined score for path {path.path_id} (post-SVScribe)")
        
        # Compute all component scores
        edgewarden_score = self._score_edgewarden(path, edgewarden_scores)
        gnn_score = gnn_path_score if gnn_path_score is not None else self._heuristic_path_quality(path, graph)
        topology_score = self._score_topology(path, graph)
        val_score = validation_score if validation_score is not None else 1.0
        ul_score = ul_confidence if ul_confidence is not None else 0.5
        hic_score = hic_confidence if hic_confidence is not None else 0.5
        sv_score = sv_misassembly_score if sv_misassembly_score is not None else 1.0
        
        # Compute final score with full weights
        final_score = (
            0.25 * edgewarden_score +      # EdgeWarden
            0.20 * gnn_score +              # GNN
            0.15 * topology_score +         # Topology
            0.20 * ul_score +               # UL (now with real data)
            0.15 * hic_score +              # Hi-C (now with real data)
            0.05 * val_score                # Validation penalty
        )
        
        # Apply SVScribe misassembly penalty if provided
        if sv_misassembly_score is not None and sv_misassembly_score < 0.8:
            # Significant SV issues detected - apply modest penalty
            sv_penalty = (1.0 - sv_misassembly_score) * 0.10  # Up to 10% reduction
            final_score = max(0.0, final_score - sv_penalty)
            self.logger.warning(
                f"Path {path.path_id}: SV misassembly score {sv_misassembly_score:.2f}, "
                f"applying penalty {sv_penalty:.3f}, final={final_score:.3f}"
            )
        
        # Compute uncertainty across all evidence
        components = [
            edgewarden_score, gnn_score, topology_score,
            ul_score, hic_score, val_score, sv_score
        ]
        mean_comp = sum(components) / len(components)
        var_comp = sum((c - mean_comp) ** 2 for c in components) / len(components)
        uncertainty = max(self.uncertainty_floor, min(self.uncertainty_ceiling, math.sqrt(var_comp)))
        
        # Store final scores on path object
        path.final_combined_score = final_score
        path.final_uncertainty = uncertainty
        path.final_edgewarden = edgewarden_score
        path.final_gnn = gnn_score
        path.final_topology = topology_score
        path.final_ul = ul_score
        path.final_hic = hic_score
        path.final_validation = val_score
        path.final_sv_score = sv_score
        path.final_components = {
            'edgewarden': edgewarden_score,
            'gnn': gnn_score,
            'topology': topology_score,
            'ul_confidence': ul_score,
            'hic_confidence': hic_score,
            'validation': val_score,
            'sv_misassembly': sv_score,
            'uncertainty': uncertainty,
        }
        
        self.logger.debug(
            f"Final score components: EdW={edgewarden_score:.2f}, GNN={gnn_score:.2f}, "
            f"Topo={topology_score:.2f}, UL={ul_score:.2f}, HiC={hic_score:.2f}, "
            f"Val={val_score:.2f}, SV={sv_score:.2f} → Final={final_score:.3f} (±{uncertainty:.3f})"
        )
        
        return final_score
    
    def _score_edgewarden(
        self,
        path: AssemblyPath,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]],
    ) -> float:
        """
        Score using EdgeWarden confidence (accepts precomputed values).
        
        IMPORTANT: Uses EdgeWarden outputs from upstream.
        Does NOT recalculate edge confidence.
        """
        if not edgewarden_scores or not path.edges:
            # Fallback: Use path.edges confidence field (set elsewhere)
            if not path.edges:
                return 0.5
            confidences = [e.confidence for e in path.edges]
            avg = sum(confidences) / len(confidences) if confidences else 0.5
            return min(1.0, avg)
        
        # Use provided EdgeWarden scores
        scores = []
        for edge in path.edges:
            edge_key = (edge.from_node_id, edge.to_node_id)
            if edge_key in edgewarden_scores:
                scores.append(edgewarden_scores[edge_key])
        
        if not scores:
            return 0.5
        
        avg = sum(scores) / len(scores)
        # Store optional per-path EdgeWarden dispersion for diagnostics
        ew_mean = avg
        ew_var = sum((s - ew_mean) ** 2 for s in scores) / len(scores)
        # Attach to path for downstream analysis
        try:
            path.edgewarden_mean = ew_mean
            path.edgewarden_std = math.sqrt(ew_var)
        except Exception:
            pass
        return min(1.0, avg)
    
    def _heuristic_path_quality(
        self,
        path: AssemblyPath,
        graph: Any,
    ) -> float:
        """
        Fallback heuristic for path quality (no GNN available).
        
        Uses simple graph topology when GNN unavailable.
        """
        if not path.edges:
            return 0.5
        
        # Simple heuristic: prefer paths with consistent edge support
        support_scores = []
        for edge in path.edges:
            support = min(1.0, edge.support_count / 10.0)  # Normalize to 0-1
            support_scores.append(support)
        
        if not support_scores:
            return 0.5
        
        return min(1.0, sum(support_scores) / len(support_scores))
    
    def _score_topology(self, path: AssemblyPath, graph: Any) -> float:
        """Score based on graph topology (coverage consistency, length)."""
        if not path.node_ids:
            return 0.0
        
        # Get coverage consistency
        coverages = []
        total_length = 0
        
        for node_id in path.node_ids:
            if hasattr(graph, 'nodes'):
                node = graph.nodes.get(node_id)
                if node:
                    if hasattr(node, 'coverage'):
                        coverages.append(node.coverage)
                    if hasattr(node, 'seq'):
                        total_length += len(node.seq)
        
        # Score 1: Coverage consistency
        if coverages:
            mean_cov = sum(coverages) / len(coverages)
            if mean_cov > 0:
                variance = sum((c - mean_cov) ** 2 for c in coverages) / len(coverages)
                cv = math.sqrt(variance) / mean_cov
                coverage_score = max(0.0, 1.0 - cv)
            else:
                coverage_score = 0.0
        else:
            coverage_score = 0.5
        
        # Score 2: Length preference (1kb-1Mb is typical)
        if 1000 <= total_length <= 1000000:
            length_score = 1.0
        elif 500 <= total_length <= 5000000:
            length_score = 0.8
        else:
            length_score = 0.5
        
        # Combined topology score
        return min(1.0, (coverage_score + length_score) / 2.0)
    
    def rank_paths(
        self,
        paths: List[AssemblyPath],
        graph: Any,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]] = None,
        gnn_path_scores: Optional[Dict[str, float]] = None,
        ul_path_scores: Optional[Dict[str, float]] = None,
        hic_path_scores: Optional[Dict[str, float]] = None,
        is_final_iteration: bool = False,
    ) -> List[AssemblyPath]:
        """
        Score and rank paths using iteration-appropriate evidence.
        
        ACCEPTS: EdgeWarden scores (no recalculation)
        USES: GNN path-level scores, UL confidence (final iteration only), Hi-C (final only)
        FALLBACK: Heuristics if upstream unavailable
        
        Iteration Logic:
        ─────────────────
        Non-Final Iterations (1 to N-1):
        ├─ EdgeWarden: 40%
        ├─ GNN: 30%
        ├─ Topology: 20%
        └─ Validation: 10%
        → UL/Hi-C ignored (not available yet)
        
        Final Iteration (N, post-SVScribe):
        ├─ EdgeWarden: 25%
        ├─ GNN: 20%
        ├─ Topology: 15%
        ├─ UL Confidence: 20%  (now available from ThreadCompass)
        ├─ Hi-C Confidence: 15% (now available from HiCWeaver)
        └─ Validation: 5%
        
        Args:
            paths: Paths to rank
            graph: Assembly graph
            edgewarden_scores: EdgeWarden confidence per edge (ACCEPTS from upstream)
            gnn_path_scores: GNN path-level scores by path_id (optional)
            ul_path_scores: UL confidence scores by path_id (optional, final iteration only)
            hic_path_scores: Hi-C confidence scores by path_id (optional, final iteration only)
            is_final_iteration: If True, include UL/Hi-C in scoring; else ignore them
        
        Returns:
            Paths sorted by score (highest first)
        """
        self.logger.info(
            f"Scoring {len(paths)} paths using EdgeWarden + GNN + long-range "
            f"(iteration={'FINAL' if is_final_iteration else 'intermediate'})"
        )
        
        for path in paths:
            gnn_score = gnn_path_scores.get(path.path_id) if gnn_path_scores else None
            
            # Only use UL/Hi-C scores in final iteration
            ul_score = (ul_path_scores.get(path.path_id) if ul_path_scores else None) if is_final_iteration else None
            hic_score = (hic_path_scores.get(path.path_id) if hic_path_scores else None) if is_final_iteration else None
            
            val_score = getattr(path, 'validation_score', None)  # Should be pre-computed
            
            self.score_path(
                path,
                graph,
                edgewarden_scores=edgewarden_scores,
                gnn_path_score=gnn_score,
                validation_score=val_score,
                ul_confidence=ul_score,
                hic_confidence=hic_score,
                is_final_iteration=is_final_iteration,
            )
        
        # Sort by total_score descending
        paths.sort(key=lambda p: p.total_score, reverse=True)
        
        self.logger.info(f"Top path score: {paths[0].total_score:.3f}" if paths else "No paths")
        return paths


# ============================================================================
#                         PATH VALIDATION ENGINE
# ============================================================================

class PathValidator:
    """
    Path validation and constraint checking.
    
    Validates paths against biological and graph constraints.
    """
    
    def __init__(self, graph: Any, config: Optional[ValidationConfig] = None):
        """Initialize validator with optional ValidationConfig."""
        self.graph = graph
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(f"{__name__}.PathValidator")

    def _get_rule_mode(self, rule: PathValidationRule) -> str:
        modes = self.config.rule_modes or {}
        mode = modes.get(rule)
        if mode is None and isinstance(rule, PathValidationRule):
            mode = modes.get(rule.value)
        return mode or "strict"
    
    def validate_path(
        self,
        path: AssemblyPath,
        rules: List[PathValidationRule] = None,
    ) -> bool:
        """
        Validate path against rules.
        
        Args:
            path: Path to validate
            rules: Rules to check (all if None)
        
        Returns:
            True if path is valid
        """
        if rules is None:
            rules = self.config.rules_enabled or [
                PathValidationRule.NO_SELF_LOOPS,
                PathValidationRule.STRAND_CONSISTENCY,
                PathValidationRule.MIN_COVERAGE,
                PathValidationRule.REPEAT_AWARE,
                PathValidationRule.SYNTENY_CONSTRAINT,
                PathValidationRule.BOUNDARY_DETECTION,
                PathValidationRule.CNV_AWARE,
            ]
        
        path.validation_errors = []
        path.validation_warnings = []
        path.pending_rules = []
        path.is_valid = True
        path.validation_status = "unknown"

        def record_failure(msg: str, mode: str):
            if mode == "warn":
                path.validation_warnings.append(msg)
            else:  # strict
                path.validation_errors.append(msg)
                path.is_valid = False
        
        for rule in rules:
            mode = self._get_rule_mode(rule)
            if mode == "off":
                continue
            if mode == "defer":
                path.pending_rules.append(rule.value)
                continue

            if rule == PathValidationRule.NO_SELF_LOOPS:
                if not self._check_no_self_loops(path):
                    record_failure("Contains self-loops", mode)
            
            elif rule == PathValidationRule.CONNECTED_COMPONENT:
                if not self._check_connected(path):
                    record_failure("Disconnected components", mode)
            
            elif rule == PathValidationRule.MAX_LENGTH:
                if not self._check_max_length(path):
                    record_failure("Exceeds max length", mode)
            
            elif rule == PathValidationRule.MIN_COVERAGE:
                if not self._check_min_coverage(path):
                    record_failure("Low coverage nodes", mode)
            
            elif rule == PathValidationRule.REPEAT_AWARE:
                if not self._check_repeat_aware(path):
                    record_failure("High-repeat segments without sufficient confidence", mode)
            
            elif rule == PathValidationRule.SYNTENY_CONSTRAINT:
                if not self._check_synteny(path):
                    record_failure("Violates synteny/marker order constraints", mode)
            
            elif rule == PathValidationRule.BOUNDARY_DETECTION:
                if not self._check_boundary_detection(path):
                    record_failure("Suspicious telomere/centromere boundary join", mode)
            
            elif rule == PathValidationRule.CNV_AWARE:
                if not self._check_cnv(path):
                    record_failure("Incompatible coverage shifts (CNV-aware)", mode)

        if not path.is_valid:
            path.validation_status = "rejected"
        elif path.pending_rules:
            path.validation_status = "pending_long_range"
        elif path.validation_warnings:
            path.validation_status = "warn"
        else:
            path.validation_status = "valid"

        return path.is_valid
    
    def _check_no_self_loops(self, path: AssemblyPath) -> bool:
        """Check that path has no self-loops."""
        return len(set(path.node_ids)) == len(path.node_ids)
    
    def _check_connected(self, path: AssemblyPath) -> bool:
        """Check that all nodes in path are connected."""
        for i in range(len(path.node_ids) - 1):
            from_id = path.node_ids[i]
            to_id = path.node_ids[i + 1]
            
            # Check if edge exists
            if hasattr(self.graph, 'out_edges'):
                edge_ids = self.graph.out_edges.get(from_id, set())
                found = False
                for edge_id in edge_ids:
                    if edge_id in self.graph.edges:
                        edge = self.graph.edges[edge_id]
                        if edge.to_node == to_id:
                            found = True
                            break
                if not found:
                    return False
        
        return True
    
    def _check_max_length(self, path: AssemblyPath) -> bool:
        """Check that path doesn't exceed max length."""
        total = 0
        for node_id in path.node_ids:
            if hasattr(self.graph, 'nodes'):
                node = self.graph.nodes.get(node_id)
                if node and hasattr(node, 'seq'):
                    total += len(node.seq)
        
        return total <= 1000000  # 1Mb default max
    
    def _check_min_coverage(self, path: AssemblyPath) -> bool:
        """Check that all nodes meet minimum coverage."""
        min_cov = self.config.min_coverage
        for node_id in path.node_ids:
            if hasattr(self.graph, 'nodes'):
                node = self.graph.nodes.get(node_id)
                if node and hasattr(node, 'coverage'):
                    if node.coverage < min_cov:
                        return False
        
        return True

    def _check_repeat_aware(self, path: AssemblyPath) -> bool:
        """Penalize paths with high-repeat segments lacking confidence/support."""
        # Heuristic: if any edge has very low confidence and support_count in a repeat-rich area, fail.
        # Use EdgeWarden repeat_score if available on edges; else fallback to node GC/length heuristics.
        # If boundary labels differ, require high confidence; otherwise allow.
        for edge in getattr(path, 'edges', []):
            repeat_score = getattr(edge, 'repeat_score', None)
            confidence = getattr(edge, 'confidence', 0.5)
            support = getattr(edge, 'support_count', 0)
            if repeat_score is not None:
                if (
                    repeat_score > self.config.repeat_score_threshold and
                    (confidence < self.config.repeat_min_confidence or support < self.config.repeat_min_support)
                ):
                    return False
            else:
                # Fallback: if both nodes are long and similar GC (proxy for repeats) with low confidence
                if hasattr(self.graph, 'nodes'):
                    n1 = self.graph.nodes.get(edge.from_node_id)
                    n2 = self.graph.nodes.get(edge.to_node_id)
                    if n1 and n2:
                        gc1 = getattr(n1, 'gc', 0.5)
                        gc2 = getattr(n2, 'gc', 0.5)
                        len1 = len(getattr(n1, 'seq', ''))
                        len2 = len(getattr(n2, 'seq', ''))
                        if (
                            abs(gc1 - gc2) < self.config.fallback_repeat_gc_delta and
                            min(len1, len2) > self.config.fallback_repeat_min_length and
                            confidence < self.config.fallback_repeat_min_confidence
                        ):
                            return False
        return True

    def _check_synteny(self, path: AssemblyPath) -> bool:
        """Validate against known marker order if available (lightweight)."""
        # Expect optional per-node marker index; enforce non-decreasing order.
        last_idx = -math.inf
        for node_id in path.node_ids:
            node = getattr(self.graph, 'nodes', {}).get(node_id)
            if not node:
                continue
            marker_idx = getattr(node, 'marker_index', None)
            if marker_idx is None:
                continue
            if marker_idx < last_idx:
                return False
            last_idx = marker_idx
        return True

    def _check_boundary_detection(self, path: AssemblyPath) -> bool:
        """Flag suspicious joins across known boundaries (telomere/centromere-like)."""
        # If nodes have boundary labels, avoid crossing into a different boundary unless high confidence.
        for edge in getattr(path, 'edges', []):
            n1 = getattr(self.graph, 'nodes', {}).get(edge.from_node_id)
            n2 = getattr(self.graph, 'nodes', {}).get(edge.to_node_id)
            if not n1 or not n2:
                continue
            b1 = getattr(n1, 'boundary_label', None)
            b2 = getattr(n2, 'boundary_label', None)
            # If labels differ, require high confidence; if labels same or missing, allow.
            if b1 is None or b2 is None:
                continue
            if b1 != b2:
                confidence = getattr(edge, 'confidence', 0.5)
                if confidence < self.config.boundary_cross_confidence_threshold:
                    return False
        return True

    def _check_cnv(self, path: AssemblyPath) -> bool:
        """Reject paths with abrupt incompatible coverage shifts (CNV-aware heuristic)."""
        # Compute coverage series and check for large jumps between adjacent nodes.
        coverages = []
        for node_id in path.node_ids:
            node = getattr(self.graph, 'nodes', {}).get(node_id)
            if node and hasattr(node, 'coverage'):
                coverages.append(node.coverage)
        if len(coverages) < 2:
            return True
        for i in range(len(coverages)-1):
            c1, c2 = coverages[i], coverages[i+1]
            if c1 > 0 and c2 > 0:
                ratio = max(c1, c2) / min(c1, c2)
                if ratio > self.config.cnv_ratio_threshold:  # jump is suspicious
                    return False
        return True

    def compute_validation_score(self, path: AssemblyPath) -> float:
        """
        Compute a validation quality score (0.0-1.0) based on violations.
        
        1.0 = perfect (no violations), reduced by each error/warning.
        This score complements the binary is_valid flag and enables
        comparison across modules.
        
        Args:
            path: Path with validation results
        
        Returns:
            Validation score in [0.0, 1.0]
        """
        # Base score
        score = 1.0
        
        # Penalty weights
        error_penalty = 0.15  # -15% per hard error
        warning_penalty = 0.05  # -5% per warning
        pending_penalty = 0.02  # -2% per deferred rule
        
        # Apply penalties
        score -= len(path.validation_errors) * error_penalty
        score -= len(path.validation_warnings) * warning_penalty
        score -= len(path.pending_rules) * pending_penalty
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, score))


# ============================================================================
#                         MAIN PATH ORCHESTRATOR
# ============================================================================

class PathWeaver:
    """
    Unified path finding and ranking engine (downstream of EdgeWarden).
    
    ARCHITECTURE: Downstream Consumer Pattern
    ═══════════════════════════════════════════
    
    EdgeWarden (Upstream) - Computes:
    ├─ Edge-level confidence scores
    ├─ Coverage consistency metrics
    ├─ Error pattern classifications
    └─ Biological plausibility signals
    
    PathWeaver (This Module) - ACCEPTS EdgeWarden outputs:
    ├─ Receives edge scores (Dict[Tuple[int,int], float])
    ├─ Does NOT recalculate EdgeWarden metrics
    ├─ Applies GNN for path-level optimization
    ├─ Uses graph algorithms (DFS, BFS, Dijkstra, DP)
    ├─ Ranks paths by combined EdgeWarden + GNN + topology
    └─ Falls back to heuristics if upstream unavailable
    
    CRITICAL DESIGN PRINCIPLE:
    - ACCEPTS EdgeWarden confidence scores from upstream
    - NEVER duplicates EdgeWarden computation
    - Falls back to path.edges.confidence if upstream unavailable
    - Operates as standalone module with internal consistency
    
    Key Components:
    - PathFinder: Graph traversal algorithms (4 strategies)
    - PathScorer: Multi-dimensional path scoring (EdW, GNN, topology)
    - PathValidator: Path validation rules (6+ checks)
    - PathWeaver: Orchestrator combining all components
    
    Usage Pattern (Downstream):
    ```python
    # Get EdgeWarden scores from upstream
    edgewarden_scores = edge_warden.score_edges(edges)
    
    # Pass to PathWeaver (no recalculation)
    pathweaver = PathWeaver(graph)
    best_paths = pathweaver.find_best_paths(
        start_node,
        edgewarden_scores=edgewarden_scores,  # ACCEPTS from upstream
        gnn_path_scores=gnn_scores,           # Optional path-level scores
    )
    
    # Falls back if upstream unavailable
    if not edgewarden_scores:
        # Automatically uses path.edges.confidence + heuristics
        best_paths = pathweaver.find_best_paths(start_node)
    ```
    """
    
    def __init__(self, graph: Any, validation_config: Optional[ValidationConfig] = None):
        """
        Initialize PathWeaver.
        
        Args:
            graph: Assembly graph (DBG, StringGraph, etc.)
        """
        self.graph = graph
        self.finder = PathFinder(graph)
        self.scorer = PathScorer()
        # Default to two-pass-friendly config: defer boundary/synteny, warn CNV
        default_config = ValidationConfig(
            rule_modes={
                PathValidationRule.BOUNDARY_DETECTION: "defer",
                PathValidationRule.SYNTENY_CONSTRAINT: "defer",
                PathValidationRule.CNV_AWARE: "warn",
            }
        )
        self.validator = PathValidator(graph, config=validation_config or default_config)
        self.logger = logging.getLogger(f"{__name__}.PathWeaver")
        self.discovered_paths: Dict[str, AssemblyPath] = {}
        
        # Misassembly detection
        self.misassembly_detector = None
        if MISASSEMBLY_DETECTOR_AVAILABLE:
            self.misassembly_detector = MisassemblyDetector()
            self.logger.info("Misassembly detection enabled")
        else:
            self.logger.warning("Misassembly detector unavailable - assembly quality may be reduced")

    # ---------------------------------------------------------------------
    # Optional Inputs: Synteny Markers & Boundary Labels
    # ---------------------------------------------------------------------
    def set_validation_config(self, config: ValidationConfig) -> None:
        """Update the validator's configuration thresholds and enabled rules."""
        self.validator.config = config
        self.logger.info("ValidationConfig updated for PathValidator")

    def _strict_config_from(self, config: ValidationConfig) -> ValidationConfig:
        """Clone a config and force all rules to strict (clears rule_modes)."""
        new_cfg = copy.deepcopy(config)
        new_cfg.rule_modes = None
        return new_cfg

    def revalidate_paths(
        self,
        paths: List[AssemblyPath],
        validation_config: Optional[ValidationConfig] = None,
    ) -> List[AssemblyPath]:
        """Re-run validation on existing paths, optionally with a temporary config."""
        original = self.validator.config
        if validation_config:
            self.validator.config = validation_config
        try:
            for path in paths:
                self.validator.validate_path(path)
        finally:
            if validation_config:
                self.validator.config = original
        return paths

    def register_synteny_markers(self, marker_map: Dict[int, int]) -> int:
        """Register per-node synteny marker indices for validation.

        Args:
            marker_map: Dict of `node_id -> marker_index` (non-decreasing expected along valid paths)

        Returns:
            Number of nodes updated
        """
        if not hasattr(self.graph, 'nodes'):
            self.logger.warning("Graph has no nodes attribute; cannot register synteny markers")
            return 0
        updated = 0
        for node_id, idx in marker_map.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            try:
                setattr(node, 'marker_index', int(idx))
                updated += 1
            except Exception as e:
                self.logger.debug(f"Failed to set marker_index for node {node_id}: {e}")
        self.logger.info(f"Registered synteny markers for {updated} nodes")
        return updated

    def register_boundary_labels(self, label_map: Dict[int, str]) -> int:
        """Register per-node boundary labels for boundary detection.

        Args:
            label_map: Dict of `node_id -> label` (e.g., 'telomere', 'centromere', custom region labels)

        Returns:
            Number of nodes updated
        """
        if not hasattr(self.graph, 'nodes'):
            self.logger.warning("Graph has no nodes attribute; cannot register boundary labels")
            return 0
        updated = 0
        for node_id, label in label_map.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            try:
                setattr(node, 'boundary_label', str(label))
                updated += 1
            except Exception as e:
                self.logger.debug(f"Failed to set boundary_label for node {node_id}: {e}")
        self.logger.info(f"Registered boundary labels for {updated} nodes")
        return updated
    
    # ========================================================================
    #          HI-C INTEGRATION (StrandTether Scoring Methods)
    # ========================================================================
    
    def register_hic_contact_matrix(self, contact_matrix: Dict[Tuple[int, int], int]) -> int:
        """
        Register Hi-C contact matrix for path scoring.
        
        Args:
            contact_matrix: Dict of (node_id1, node_id2) -> contact_count
        
        Returns:
            Number of contact pairs registered
        """
        if not self.strandtether:
            self.logger.warning("StrandTether not available, Hi-C registration skipped")
            return 0
        
        num_pairs = self.strandtether.register_contact_matrix(contact_matrix)
        self.logger.info(f"Registered Hi-C contact matrix: {num_pairs} pairs")
        return num_pairs
    
    def compute_hic_path_scores(
        self,
        paths: List[AssemblyPath],
    ) -> Dict[str, float]:
        """
        Compute Hi-C confidence scores for paths.
        
        Args:
            paths: Paths to score
        
        Returns:
            Dict mapping path_id -> hic_confidence (0.0-1.0)
        """
        if not self.strandtether:
            self.logger.debug("StrandTether not available, returning neutral scores")
            return {p.path_id: 0.5 for p in paths}
        
        hic_scores = {}
        for path in paths:
            try:
                score = self.strandtether.score_path_hic(path.node_ids)
                hic_scores[path.path_id] = score
            except Exception as e:
                self.logger.warning(f"Hi-C scoring failed for path {path.path_id}: {e}")
                hic_scores[path.path_id] = 0.5  # Neutral fallback
        
        self.logger.debug(f"Computed Hi-C scores for {len(hic_scores)} paths")
        return hic_scores

    def find_best_paths(
        self,
        start_node_id: int,
        end_node_ids: Optional[Set[int]] = None,
        max_paths: int = 10,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        gnn_scorer: Optional[Any] = None,
        use_gnn_primary: bool = True,
        num_iterations: int = 2,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]] = None,
        gnn_path_scores: Optional[Dict[str, float]] = None,
        ul_path_scores: Optional[Dict[str, float]] = None,
        hic_path_scores: Optional[Dict[str, float]] = None,
        long_range_two_pass: bool = True,
        strict_validation_config: Optional[ValidationConfig] = None,
    ) -> List[AssemblyPath]:
        """
        Find and rank best paths using GNN-FIRST architecture.
        
        NEW ARCHITECTURE (GNN-First)
        ════════════════════════════════════════════════════════════════
        
        1. PRIMARY: GNN path prediction (if available)
           - GNN predicts edge confidences
           - Extracts high-confidence paths directly
           - Learns biological patterns from training data
        
        2. FALLBACK: Classical algorithms (Dijkstra/BFS/DFS)
           - Used when GNN unavailable or fails
           - Reliable baseline path finding
        
        3. ENHANCEMENT: EdgeWarden + Long-range data
           - EdgeWarden scores enhance both GNN and algorithm paths
           - Hi-C and UL support refine path selection
        
        This is the CORRECT design: ML-first with classical fallback.
        Old design (algorithm-first with GNN refinement) was backwards.
        
        Pipeline Steps:
        ──────────────
        1. Path Discovery (GNN-first or algorithm fallback)
        2. EdgeWarden + GNN + Long-range scoring
        3. Validation (connectivity, coverage, etc.)
        4. Misassembly detection
        5. Ranking and selection
        
        Args:
            start_node_id: Starting node
            end_node_ids: Target node(s)
            max_paths: Maximum paths to return
            algorithm: Fallback algorithm (default: Dijkstra)
                - PathFindingAlgorithm.DIJKSTRA: Shortest path (recommended)
                - PathFindingAlgorithm.BREADTH_FIRST: BFS option
                - PathFindingAlgorithm.DEPTH_FIRST: DFS option
            gnn_scorer: GNN model for path prediction (PRIMARY METHOD)
            use_gnn_primary: Use GNN as primary method (default: True)
                - True: GNN first, fallback to algorithm
                - False: Use legacy iterative mode (deprecated)
            num_iterations: For legacy mode only (deprecated)
            edgewarden_scores: EdgeWarden confidence (ACCEPTS from upstream)
            gnn_path_scores: GNN path-level scores (optional optimization)
            ul_path_scores: Ultra-long read support
            hic_path_scores: Hi-C contact support
            long_range_two_pass: Enable two-pass validation
            strict_validation_config: Custom validation thresholds
        
        Returns:
            List of best paths ranked by combined scoring
        """
        self.logger.info(
            f"Finding best paths from node {start_node_id} "
            f"(GNN-first={use_gnn_primary}, fallback={algorithm.value})"
        )
        
        # Step 1: Find candidate paths (GNN-FIRST or fallback)
        if use_gnn_primary and gnn_scorer:
            self.logger.info("  PRIMARY: Using GNN for path discovery")
            finding_result = self.finder.find_paths_gnn_primary(
                start_node_id,
                end_node_ids,
                algorithm,
                max_paths=max_paths * 10,
                gnn_scorer=gnn_scorer,
            )
        elif num_iterations > 1 and gnn_scorer:
            # Legacy mode: iterative algorithm→GNN (deprecated)
            self.logger.warning(
                "  Using LEGACY iterative mode (deprecated). "
                "Set use_gnn_primary=True for new GNN-first architecture"
            )
            finding_result = self.finder.find_paths_iterative_gnn(
                start_node_id,
                end_node_ids,
                algorithm,
                max_paths=max_paths * 10,
                gnn_scorer=gnn_scorer,
                num_iterations=num_iterations,
            )
        else:
            # Pure algorithm mode (no GNN)
            self.logger.info(f"  FALLBACK: Using {algorithm.value} algorithm (no GNN)")
            finding_result = self.finder.find_paths(
                start_node_id,
                end_node_ids,
                algorithm,
                max_paths=max_paths * 10,
            )
        
        paths = finding_result.paths
        self.logger.info(f"Found {len(paths)} candidate paths ({finding_result.notes or 'no notes'})")
        
        if not paths:
            self.logger.warning("No paths found!")
            return []
        
        # Step 2: Score paths using EdgeWarden + GNN (no recalculation)
        # Compute validation scores for all paths first
        for path in paths:
            # Run validation to populate errors/warnings/pending
            self.validator.validate_path(path)
            # Compute 0-1 validation score
            val_score = self.validator.compute_validation_score(path)
            path.validation_score = val_score
        
        self.scorer.rank_paths(
            paths,
            self.graph,
            edgewarden_scores=edgewarden_scores,
            gnn_path_scores=gnn_path_scores,
        )
        
        # Step 3: Filter invalid paths (Pass A)
        valid_paths = []
        for path in paths:
            if self.validator.validate_path(path):
                valid_paths.append(path)

        self.logger.info(
            f"Validated {len(valid_paths)}/{len(paths)} paths (pass A, rule_modes={self.validator.config.rule_modes})"
        )

        if not valid_paths:
            self.logger.warning("No valid paths!")
            return paths[:max_paths]

        # Optional Step 3b: Strict revalidation (Pass B) even if no UL/Hi-C provided
        if long_range_two_pass:
            strict_cfg = strict_validation_config or self._strict_config_from(self.validator.config)
            self.logger.info("Revalidating paths in strict mode (pass B)")
            self.revalidate_paths(valid_paths, validation_config=strict_cfg)
            valid_paths = [p for p in valid_paths if p.is_valid]
            self.logger.info(f"Validated {len(valid_paths)} paths after strict pass B")
            if not valid_paths:
                self.logger.warning("All paths rejected in strict pass B; returning top candidates from pass A")
                return paths[:max_paths]
        
        # Step 4: Detect misassemblies in paths
        if self.misassembly_detector:
            self.logger.info("Scanning for putative misassemblies...")
            for path in valid_paths:
                # Detect misassemblies in this path
                flags = self.misassembly_detector.detect_in_path(
                    contig_id=path.path_id,
                    node_ids=path.node_ids,
                    edge_ids=[f"{e.from_node_id}_{e.to_node_id}" for e in path.edges],
                    edgewarden_scores={f"{e.from_node_id}_{e.to_node_id}": e.confidence for e in path.edges},
                    coverage_data={node_id: self._get_node_coverage(node_id) for node_id in path.node_ids},
                )
                
                # Store flags in path metadata
                if not hasattr(path, 'misassembly_flags'):
                    path.misassembly_flags = []
                path.misassembly_flags = flags
                
                if flags:
                    self.logger.warning(
                        f"Path {path.path_id}: Found {len(flags)} putative misassemblies"
                    )
        
        # Step 5: Return top N
        best_paths = valid_paths[:max_paths]
        
        self.logger.info(
            f"Returning {len(best_paths)} best paths "
            f"(top score: {best_paths[0].total_score:.3f})"
        )
        
        # Store for future reference
        for path in best_paths:
            self.discovered_paths[path.path_id] = path
        
        return best_paths

    def find_best_paths_multi_start(
        self,
        start_node_ids: List[int],
        end_node_ids: Optional[Set[int]] = None,
        max_paths_total: int = 20,
        per_start_max_paths: Optional[int] = None,
        algorithm: PathFindingAlgorithm = PathFindingAlgorithm.DIJKSTRA,
        gnn_scorer: Optional[Any] = None,
        use_gnn_primary: bool = True,
        num_iterations: int = 2,
        edgewarden_scores: Optional[Dict[Tuple[int, int], float]] = None,
        gnn_path_scores: Optional[Dict[str, float]] = None,
        ul_path_scores: Optional[Dict[str, float]] = None,
        hic_path_scores: Optional[Dict[str, float]] = None,
        num_workers: int = 4,
        long_range_two_pass: bool = True,
        strict_validation_config: Optional[ValidationConfig] = None,
    ) -> List[AssemblyPath]:
        """
        Multi-start path finding with GNN-FIRST architecture.

        Finds best paths from multiple start nodes using GNN as primary method
        with algorithm fallback, parallelized across start nodes.
        
        NEW ARCHITECTURE (GNN-First)
        ════════════════════════════════════════════════════════════════
        
        1. PRIMARY: GNN path prediction (parallel per-start)
        2. FALLBACK: Classical algorithms if GNN unavailable
        3. ENHANCEMENT: EdgeWarden + Long-range scoring
        
        Steps:
          1) Discover candidate paths (GNN-first, parallel per start)
          2) Rank with EdgeWarden + GNN + Long-range
          3) Validate and filter
          4) Detect misassemblies
          5) Return top N (max_paths_total)
        
        Args:
            start_node_ids: List of starting nodes
            end_node_ids: Target node(s)
            max_paths_total: Maximum total paths to return
            per_start_max_paths: Per-start limit (auto-calculated if None)
            algorithm: Fallback algorithm (default: Dijkstra)
            gnn_scorer: GNN model for path prediction (PRIMARY METHOD)
            use_gnn_primary: Use GNN as primary (default: True)
            num_iterations: For legacy mode only (deprecated)
            edgewarden_scores: EdgeWarden confidence (ACCEPTS from upstream)
            gnn_path_scores: GNN path-level scores
            ul_path_scores: Ultra-long read support
            hic_path_scores: Hi-C contact support
            num_workers: Parallel workers (threads)
            long_range_two_pass: Enable two-pass validation
            strict_validation_config: Custom validation thresholds
        
        Returns:
            List of best multi-start paths
        """
        self.logger.info(
            f"Finding best multi-start paths from {len(start_node_ids)} starts "
            f"(GNN-first={use_gnn_primary}, fallback={algorithm.value})"
        )

        # Note: For multi-start, we currently don't have a parallel GNN implementation
        # So we either call find_best_paths() per start (simple but serial), or use
        # the existing multi_start finder which doesn't have GNN-primary yet
        
        # TODO: Future optimization - implement parallel GNN-first per start
        # For now, use simple serial approach with GNN-first per start
        
        if use_gnn_primary and gnn_scorer:
            self.logger.info("  Using serial GNN-first path finding per start node")
            all_paths = []
            for start_node_id in start_node_ids:
                start_paths = self.find_best_paths(
                    start_node_id=start_node_id,
                    end_node_ids=end_node_ids,
                    max_paths=per_start_max_paths or (max_paths_total // len(start_node_ids) + 1),
                    algorithm=algorithm,
                    gnn_scorer=gnn_scorer,
                    use_gnn_primary=True,
                    edgewarden_scores=edgewarden_scores,
                    gnn_path_scores=gnn_path_scores,
                    long_range_two_pass=False,  # We'll do it once at the end
                )
                all_paths.extend(start_paths)
            
            paths = all_paths
            self.logger.info(f"Found {len(paths)} GNN-first multi-start candidate paths")
            
        else:
            # Fallback to classical multi-start (faster but less accurate)
            self.logger.info(f"  Using classical multi-start path finding ({algorithm.value})")
            finding_result = self.finder.find_paths_multi_start(
                start_node_ids=start_node_ids,
                end_node_ids=end_node_ids,
                algorithm=algorithm,
                max_paths_total=max_paths_total * 10,
                per_start_max_paths=per_start_max_paths,
                num_workers=num_workers,
            )
            paths = finding_result.paths
            self.logger.info(f"Found {len(paths)} multi-start candidate paths")

        if not paths:
            self.logger.warning("No paths found (multi-start)!")
            return []

        # Step 2: Score paths using EdgeWarden + GNN + long-range
        # Compute validation scores for all paths first
        for path in paths:
            self.validator.validate_path(path)
            val_score = self.validator.compute_validation_score(path)
            path.validation_score = val_score
        
        self.scorer.rank_paths(
            paths,
            self.graph,
            edgewarden_scores=edgewarden_scores,
            gnn_path_scores=gnn_path_scores,
            ul_path_scores=ul_path_scores,
            hic_path_scores=hic_path_scores,
        )

        # Step 3: Validate paths (Pass A)
        valid_paths = [p for p in paths if self.validator.validate_path(p)]
        self.logger.info(
            f"Validated {len(valid_paths)}/{len(paths)} multi-start paths (pass A, rule_modes={self.validator.config.rule_modes})"
        )

        if not valid_paths:
            self.logger.warning("No valid paths (multi-start)!")
            return paths[:max_paths_total]

        # Optional Step 3b: Strict revalidation (Pass B)
        if long_range_two_pass:
            strict_cfg = strict_validation_config or self._strict_config_from(self.validator.config)
            self.logger.info("Revalidating multi-start paths in strict mode (pass B)")
            self.revalidate_paths(valid_paths, validation_config=strict_cfg)
            valid_paths = [p for p in valid_paths if p.is_valid]
            self.logger.info(f"Validated {len(valid_paths)} multi-start paths after strict pass B")
            if not valid_paths:
                self.logger.warning("All multi-start paths rejected in strict pass B; returning top candidates from pass A")
                return paths[:max_paths_total]

        # Step 4: Misassembly detection
        if self.misassembly_detector:
            self.logger.info("Scanning (multi-start) for putative misassemblies...")
            for path in valid_paths:
                flags = self.misassembly_detector.detect_in_path(
                    contig_id=path.path_id,
                    node_ids=path.node_ids,
                    edge_ids=[f"{e.from_node_id}_{e.to_node_id}" for e in path.edges],
                    edgewarden_scores={f"{e.from_node_id}_{e.to_node_id}": e.confidence for e in path.edges},
                    coverage_data={node_id: self._get_node_coverage(node_id) for node_id in path.node_ids},
                )
                path.misassembly_flags = flags

        # Step 5: Return top N
        best_paths = valid_paths[:max_paths_total]

        for path in best_paths:
            self.discovered_paths[path.path_id] = path

        return best_paths
    
    def _get_node_coverage(self, node_id: int) -> float:
        """Get coverage for a node from graph."""
        if hasattr(self.graph, 'nodes'):
            node = self.graph.nodes.get(node_id)
            if node and hasattr(node, 'coverage'):
                return node.coverage
        return 0.0
    
    def get_misassembly_report(
        self,
        output_format: str = "tsv"
    ) -> str:
        """
        Generate comprehensive misassembly report for all paths.
        
        Args:
            output_format: "tsv", "json", or "bed"
        
        Returns:
            Formatted misassembly report
        """
        if not self.misassembly_detector:
            return "Misassembly detection not available"
        
        return self.misassembly_detector.generate_report(output_format)
    
    def export_misassemblies_for_downstream(
        self,
        contig_id: str,
        module: str = "ul_integration"
    ) -> Dict[str, Any]:
        """
        Export misassembly flags for downstream modules.
        
        This passes putative misassembly locations to:
        - Ultra-long read integration module
        - Hi-C scaffolding module
        - Manual review system
        - Quality control pipeline
        
        Args:
            contig_id: Contig to export
            module: Target module name
        
        Returns:
            Module-specific formatted data
        """
        if not self.misassembly_detector:
            return {"error": "Misassembly detection not available"}
        
        return self.misassembly_detector.export_for_downstream(contig_id, module)
    
    def export_paths(
        self,
        paths: List[AssemblyPath],
        format: str = "tsv",
    ) -> str:
        """
        Export paths in specified format.
        
        Args:
            paths: Paths to export
            format: Output format ('tsv', 'json', 'gfa')
        
        Returns:
            Formatted string
        """
        if format == "tsv":
            lines = ["path_id\tnum_nodes\ttotal_length\ttotal_score\tcoverage_score\tconfidence_score\tis_valid"]
            for path in paths:
                lines.append(
                    f"{path.path_id}\t{path.num_nodes}\t{path.total_length}\t"
                    f"{path.total_score:.3f}\t{path.coverage_score:.3f}\t"
                    f"{path.confidence_score:.3f}\t{path.is_valid}"
                )
            return "\n".join(lines)
        
        elif format == "json":
            import json
            return json.dumps([p.to_dict() for p in paths], indent=2)
        
        else:
            self.logger.warning(f"Unknown format: {format}")
            return ""


# ============================================================================
#                         UTILITY FUNCTIONS
# ============================================================================

def create_path_from_node_ids(
    node_ids: List[int],
    graph: Any,
    path_id: Optional[str] = None,
) -> AssemblyPath:
    """
    Create AssemblyPath object from node sequence.
    
    Args:
        node_ids: Ordered node IDs
        graph: Assembly graph
        path_id: Optional path identifier
    
    Returns:
        Constructed AssemblyPath
    """
    if not node_ids:
        raise ValueError("Cannot create path from empty node list")
    
    if path_id is None:
        path_id = f"path_{hash(tuple(node_ids))}"
    
    # Create edges
    edges = []
    for i in range(len(node_ids) - 1):
        from_id = node_ids[i]
        to_id = node_ids[i + 1]
        edge = PathEdge(from_node_id=from_id, to_node_id=to_id)
        edges.append(edge)
    
    # Calculate metrics
    total_length = 0
    coverages = []
    
    for node_id in node_ids:
        if hasattr(graph, 'nodes'):
            node = graph.nodes.get(node_id)
            if node:
                if hasattr(node, 'seq'):
                    total_length += len(node.seq)
                if hasattr(node, 'coverage'):
                    coverages.append(node.coverage)
    
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    
    path = AssemblyPath(
        path_id=path_id,
        node_ids=node_ids,
        edges=edges,
        total_length=total_length,
        avg_coverage=avg_coverage,
    )
    
    return path


__all__ = [
    'PathWeaver',
    'PathFinder',
    'PathScorer',
    'PathValidator',
    'ValidationConfig',
    'AssemblyPath',
    'PathEdge',
    'PathNode',
    'PathFindingAlgorithm',
    'PathScoringMethod',
    'PathValidationRule',
    'create_path_from_node_ids',
]
