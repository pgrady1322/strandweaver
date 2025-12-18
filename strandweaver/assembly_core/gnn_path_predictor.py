#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GNN Path Prediction Engine for StrandWeaver.

This module uses Graph Neural Networks to predict the most likely true paths
through ambiguous assembly graphs. The GNN learns from node/edge features
to assign confidence scores to edges, enabling intelligent path selection.

The module provides:
1. Feature tensor construction for GNN input
2. GNN model interface (prediction only, training is external)
3. Path scoring and extraction using GNN probabilities
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import math
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    """
    Extracts and normalizes features for GNN input.
    """
    
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
        """
        Extract feature vector for a node.
        
        Features (12 dimensions):
        - Coverage (normalized)
        - Node length (log-scaled)
        - Sequence entropy
        - In-degree
        - Out-degree
        - Branching factor (in + out)
        - Hi-C phase A score
        - Hi-C phase B score
        - Hi-C contact count (log-scaled)
        - Recommended k (normalized)
        - UL support (log-scaled)
        - Repeat likelihood
        """
        node = graph.nodes.get(node_id)
        if not node:
            return [0.0] * 12
        
        features = []
        
        # Coverage (normalize by dividing by 100)
        coverage = getattr(node, 'coverage', 0.0)
        features.append(min(coverage / 100.0, 1.0))
        
        # Length (log-scaled, normalize by dividing by log(100000))
        length = getattr(node, 'length', 0)
        features.append(math.log(length + 1) / math.log(100000))
        
        # Entropy (already 0-1)
        entropy = self._calculate_entropy(node)
        features.append(entropy)
        
        # In-degree (normalize by dividing by 10)
        in_degree = len(graph.in_edges.get(node_id, set()))
        features.append(min(in_degree / 10.0, 1.0))
        
        # Out-degree (normalize by dividing by 10)
        out_degree = len(graph.out_edges.get(node_id, set()))
        features.append(min(out_degree / 10.0, 1.0))
        
        # Branching factor
        branching = in_degree + out_degree
        features.append(min(branching / 20.0, 1.0))
        
        # Hi-C phasing
        if hic_phase_info and node_id in hic_phase_info:
            phase = hic_phase_info[node_id]
            features.append(phase.phase_A_score)
            features.append(phase.phase_B_score)
            features.append(math.log(phase.contact_count + 1) / math.log(1000))
        else:
            features.extend([0.5, 0.5, 0.0])
        
        # Recommended k
        if regional_k_map and node_id in regional_k_map:
            k_val = regional_k_map[node_id]
            features.append(k_val / 100.0)  # Normalize by max k
        else:
            features.append(0.31)  # Default k=31
        
        # UL support (aggregate from edges)
        ul_support = 0
        if ul_support_map:
            for edge_id in graph.in_edges.get(node_id, set()):
                ul_support += ul_support_map.get(edge_id, 0)
            for edge_id in graph.out_edges.get(node_id, set()):
                ul_support += ul_support_map.get(edge_id, 0)
        features.append(math.log(ul_support + 1) / math.log(100))
        
        # Repeat likelihood
        repeat_score = self._estimate_repeat_likelihood(
            coverage, in_degree, out_degree, entropy
        )
        features.append(repeat_score)
        
        return features
    
    def extract_edge_features(
        self,
        edge_id: int,
        from_node: int,
        to_node: int,
        graph,
        ai_annotations: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> List[float]:
        """
        Extract feature vector for an edge.
        
        Features (10 dimensions):
        - AI score_true
        - AI score_repeat
        - AI score_chimeric
        - AI confidence
        - Hi-C weight
        - Hi-C cis contacts (log-scaled)
        - Hi-C trans contacts (log-scaled)
        - UL support count (log-scaled)
        - Regional k consistency
        - Coverage consistency
        """
        features = []
        
        # AI annotations
        if ai_annotations and edge_id in ai_annotations:
            ai = ai_annotations[edge_id]
            features.append(ai.score_true)
            features.append(ai.score_repeat)
            features.append(ai.score_chimeric)
            features.append(ai.confidence)
        else:
            features.extend([0.5, 0.0, 0.0, 0.5])
        
        # Hi-C support
        if hic_edge_support and edge_id in hic_edge_support:
            hic = hic_edge_support[edge_id]
            features.append(hic.hic_weight)
            features.append(math.log(hic.cis_contacts + 1) / math.log(100))
            features.append(math.log(hic.trans_contacts + 1) / math.log(100))
        else:
            features.extend([0.5, 0.0, 0.0])
        
        # UL support
        ul_count = ul_support_map.get(edge_id, 0) if ul_support_map else 0
        features.append(math.log(ul_count + 1) / math.log(50))
        
        # Regional k consistency
        if regional_k_map and from_node in regional_k_map and to_node in regional_k_map:
            k_from = regional_k_map[from_node]
            k_to = regional_k_map[to_node]
            k_consistency = 1.0 - abs(k_from - k_to) / max(k_from, k_to)
            features.append(k_consistency)
        else:
            features.append(1.0)
        
        # Coverage consistency
        from_node_obj = graph.nodes.get(from_node)
        to_node_obj = graph.nodes.get(to_node)
        if from_node_obj and to_node_obj:
            cov_from = getattr(from_node_obj, 'coverage', 0.0)
            cov_to = getattr(to_node_obj, 'coverage', 0.0)
            if cov_from > 0 and cov_to > 0:
                cov_consistency = min(cov_from, cov_to) / max(cov_from, cov_to)
                features.append(cov_consistency)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
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
        
        return entropy / 2.0  # Normalize to 0-1
    
    def _estimate_repeat_likelihood(
        self,
        coverage: float,
        in_degree: int,
        out_degree: int,
        entropy: float
    ) -> float:
        """Estimate likelihood node is in repeat region."""
        score = 0.0
        
        # High coverage
        if coverage > 50:
            score += 0.3
        
        # High branching
        if in_degree + out_degree > 2:
            score += 0.3
        
        # Low entropy
        if entropy < 0.5:
            score += 0.4
        
        return min(score, 1.0)


class PathGNN:
    """
    Graph Neural Network for path prediction.
    
    Wrapper around trained PathGNNModel (PyTorch Geometric) with fallback
    to heuristic scoring if model unavailable or PyTorch not installed.
    """
    
    def __init__(self, model=None, model_path: Optional[str] = None):
        """
        Initialize GNN.
        
        Args:
            model: Optional trained GNN model with predict() method
            model_path: Optional path to trained model file to load
        """
        self.model = model
        self.logger = logging.getLogger(f"{__name__}.PathGNN")
        
        # Load model from path if provided
        if model_path and not model:
            self.load_model(model_path)
        
        if self.model and TORCH_AVAILABLE:
            self.logger.info(f"GNN using trained model: {type(self.model).__name__}")
        elif self.model:
            self.logger.info(f"GNN using model: {type(self.model).__name__}")
        else:
            self.logger.warning("GNN using heuristic fallback (no trained model available)")
    
    def predict_edge_probabilities(
        self,
        graph_tensors: GraphTensors
    ) -> Dict[int, float]:
        """
        Predict probability each edge is correct.
        
        Args:
            graph_tensors: Graph in tensor format
        
        Returns:
            Dict[edge_id] -> probability (0.0-1.0)
        """
        if self.model:
            # Use trained model
            return self._predict_with_model(graph_tensors)
        else:
            # Fallback: heuristic scoring
            return self._predict_heuristic(graph_tensors)
    
    def _predict_with_model(self, graph_tensors: GraphTensors) -> Dict[int, float]:
        """Use trained GNN model for predictions."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, falling back to heuristic")
            return self._predict_heuristic(graph_tensors)
        
        try:
            # Convert graph tensors to PyG Data format
            data = self._graph_tensors_to_pyg_data(graph_tensors)
            
            # Get predictions from model
            if hasattr(self.model, 'get_edge_probabilities'):
                predictions = self.model.get_edge_probabilities(data)
            else:
                # Fallback for models without this method
                output = self.model(data)
                edge_logits = output['edge_logits'].cpu().numpy().flatten()
                predictions = {i: float(prob) for i, prob in enumerate(edge_logits)}
            
            self.logger.debug(f"GNN predicted {len(predictions)} edge probabilities")
            return predictions
            
        except Exception as e:
            self.logger.error(f"GNN prediction failed: {e}, falling back to heuristic")
            return self._predict_heuristic(graph_tensors)
    
    def _graph_tensors_to_pyg_data(self, graph_tensors: GraphTensors) -> Union["Data", Dict]:
        """Convert GraphTensors to PyTorch Geometric Data object."""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available, cannot convert to PyG format")
            return {}
        
        import torch
        import numpy as np
        
        # Build node feature matrix
        node_features = []
        for i in range(graph_tensors.num_nodes):
            # Find node by tensor index
            node_id = None
            for nid, idx in graph_tensors.node_to_index.items():
                if idx == i:
                    node_id = nid
                    break
            
            if node_id and node_id in graph_tensors.node_features:
                node_features.append(graph_tensors.node_features[node_id])
            else:
                # Default feature vector (zeros)
                sample_size = len(next(iter(graph_tensors.node_features.values())))
                node_features.append([0.0] * sample_size)
        
        node_feat_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edge list and edge features
        edge_index = []
        edge_feat_list = []
        
        for from_idx, to_idx in graph_tensors.edge_index:
            edge_index.append([from_idx, to_idx])
            
            # Find corresponding edge_id
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
        
        # Create PyG Data object
        from torch_geometric.data import Data
        data = Data(
            x=node_feat_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_feat_tensor,
            num_nodes=graph_tensors.num_nodes
        )
        
        return data
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model file
        
        Returns:
            True if successful, False otherwise
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available, cannot load model")
            return False
        
        try:
            import torch
            from strandweaver.assembly_utils.gnn_models import PathGNNModel, SimpleGNN
            
            # Load model state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Determine model type from checkpoint
            model_type = checkpoint.get('model_type', 'SimpleGNN')
            config = checkpoint.get('config', {})
            
            # Instantiate model
            if model_type == 'SimpleGNN':
                from strandweaver.assembly_utils.gnn_models import SimpleGNN
                self.model = SimpleGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            elif model_type == 'MediumGNN':
                from strandweaver.assembly_utils.gnn_models import MediumGNN
                self.model = MediumGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            elif model_type == 'DeepGNN':
                from strandweaver.assembly_utils.gnn_models import DeepGNN
                self.model = DeepGNN(config.get('in_channels', 12), config.get('out_channels', 2))
            else:
                self.model = PathGNNModel(config)
            
            # Load state
            self.model.load_state_dict(checkpoint.get('model_state', checkpoint))
            self.model.eval()
            
            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: str, config: Optional[Dict] = None) -> bool:
        """
        Save current model to disk.
        
        Args:
            model_path: Path to save model
            config: Optional config dict to save with model
        
        Returns:
            True if successful, False otherwise
        """
        if not self.model or not TORCH_AVAILABLE:
            self.logger.error("No model available to save")
            return False
        
        try:
            import torch
            
            checkpoint = {
                'model_type': type(self.model).__name__,
                'model_state': self.model.state_dict(),
                'config': config or {}
            }
            
            torch.save(checkpoint, model_path)
            self.logger.info(f"Saved {type(self.model).__name__} to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {e}")
            return False
    
    def _predict_heuristic(self, graph_tensors: GraphTensors) -> Dict[int, float]:
        """
        Heuristic fallback scoring based on feature aggregation.
        
        Combines:
        - AI score_true (40%)
        - Hi-C weight (30%)
        - UL support (20%)
        - Coverage consistency (10%)
        """
        predictions = {}
        
        for edge_id, features in graph_tensors.edge_features.items():
            # Features: [ai_true, ai_repeat, ai_chimeric, ai_conf, 
            #            hic_weight, hic_cis, hic_trans, ul_support, 
            #            k_consistency, cov_consistency]
            
            score = 0.0
            
            # AI score_true (index 0)
            score += features[0] * 0.4
            
            # Hi-C weight (index 4)
            score += features[4] * 0.3
            
            # UL support (index 7)
            score += features[7] * 0.2
            
            # Coverage consistency (index 9)
            score += features[9] * 0.1
            
            predictions[edge_id] = max(0.0, min(1.0, score))
        
        return predictions


class PathExtractor:
    """
    Extracts best paths from graph using GNN edge probabilities.
    """
    
    def __init__(self, min_edge_confidence: float = 0.3):
        self.min_edge_confidence = min_edge_confidence
        self.logger = logging.getLogger(f"{__name__}.PathExtractor")
    
    def score_and_extract_paths(
        self,
        graph,
        edge_probabilities: Dict[int, float]
    ) -> GNNPathResult:
        """
        Extract most likely paths using GNN edge confidences.
        
        Algorithm:
        1. Filter edges below confidence threshold
        2. Find connected components
        3. Extract linear paths through each component
        4. Score paths by average edge confidence
        
        Args:
            graph: Assembly graph
            edge_probabilities: Dict[edge_id] -> confidence
        
        Returns:
            GNNPathResult with best paths and scores
        """
        self.logger.info("Extracting paths from GNN predictions")
        
        result = GNNPathResult(edge_confidences=edge_probabilities)
        
        # Filter high-confidence edges
        confident_edges = {
            edge_id: prob for edge_id, prob in edge_probabilities.items()
            if prob >= self.min_edge_confidence
        }
        
        # Build adjacency for confident edges only
        confident_graph = self._build_confident_subgraph(graph, confident_edges)
        
        # Find connected components
        components = self._find_connected_components(confident_graph)
        
        self.logger.info(f"Found {len(components)} connected components")
        
        # Extract paths from each component
        for comp_idx, component in enumerate(components):
            paths = self._extract_component_paths(
                component, confident_graph, edge_probabilities
            )
            
            for path in paths:
                path_score = self._calculate_path_score(
                    path, confident_graph, edge_probabilities
                )
                
                path_idx = len(result.best_paths)
                result.best_paths.append(path)
                result.path_scores[path_idx] = path_score
        
        # Identify ambiguous regions
        result.ambiguous_regions = self._find_ambiguous_regions(
            graph, edge_probabilities
        )
        
        self.logger.info(
            f"Extracted {len(result.best_paths)} paths, "
            f"{len(result.ambiguous_regions)} ambiguous regions"
        )
        
        return result
    
    def _build_confident_subgraph(self, graph, confident_edges: Dict) -> Dict:
        """Build subgraph containing only high-confidence edges."""
        subgraph = {
            'nodes': set(),
            'edges': {},
            'out_edges': defaultdict(set),
            'in_edges': defaultdict(set)
        }
        
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
                    
                    # Add neighbors
                    stack.extend(subgraph['out_edges'].get(current, set()))
                    stack.extend(subgraph['in_edges'].get(current, set()))
                
                if component:
                    components.append(component)
        
        return components
    
    def _extract_component_paths(
        self,
        component: Set[int],
        subgraph: Dict,
        edge_probs: Dict[int, float]
    ) -> List[List[int]]:
        """Extract linear paths through a component."""
        # Find start nodes (in-degree 0 or low in-degree)
        start_nodes = [
            node for node in component
            if len(subgraph['in_edges'].get(node, set())) == 0
        ]
        
        if not start_nodes:
            # Circular or complex topology, pick highest-degree node
            start_nodes = [max(component, key=lambda n: len(subgraph['out_edges'].get(n, set())))]
        
        paths = []
        for start in start_nodes:
            path = self._extend_path(start, subgraph, set())
            if len(path) > 1:
                paths.append(path)
        
        return paths if paths else [[node] for node in list(component)[:1]]
    
    def _extend_path(
        self,
        current: int,
        subgraph: Dict,
        visited: Set[int]
    ) -> List[int]:
        """Extend path from current node using greedy traversal."""
        path = [current]
        visited.add(current)
        
        while True:
            neighbors = [
                n for n in subgraph['out_edges'].get(current, set())
                if n not in visited
            ]
            
            if not neighbors:
                break
            
            # Pick first unvisited neighbor (could be weighted by edge prob)
            next_node = neighbors[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path
    
    def _calculate_path_score(
        self,
        path: List[int],
        subgraph: Dict,
        edge_probs: Dict[int, float]
    ) -> float:
        """Calculate overall confidence score for a path."""
        if len(path) < 2:
            return 1.0
        
        scores = []
        for i in range(len(path) - 1):
            # Find edge between consecutive nodes
            # (In real implementation, would look up edge_id properly)
            scores.append(0.8)  # Placeholder
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _find_ambiguous_regions(
        self,
        graph,
        edge_probs: Dict[int, float]
    ) -> List[Set[int]]:
        """Identify regions with low edge confidence."""
        ambiguous = []
        
        # Find nodes with multiple low-confidence outgoing edges
        for node_id in graph.nodes:
            out_edges = graph.out_edges.get(node_id, set())
            if len(out_edges) > 1:
                low_conf_edges = [
                    edge_id for edge_id in out_edges
                    if edge_probs.get(edge_id, 0.5) < 0.5
                ]
                
                if len(low_conf_edges) >= 2:
                    # This is an ambiguous branch point
                    region = {node_id}
                    for edge_id in low_conf_edges:
                        edge = graph.edges.get(edge_id)
                        if edge:
                            region.add(getattr(edge, 'to_node', -1))
                    ambiguous.append(region)
        
        return ambiguous


def build_graph_tensors(
    graph,
    hic_phase_info: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None
) -> GraphTensors:
    """
    Convert assembly graph to tensor format for GNN.
    
    Args:
        graph: Assembly graph (DBG or StringGraph)
        hic_phase_info: Hi-C phasing information
        ai_annotations: AI edge annotations
        hic_edge_support: Hi-C edge support
        regional_k_map: Regional k recommendations
        ul_support_map: UL support counts
    
    Returns:
        GraphTensors ready for GNN input
    """
    extractor = FeatureExtractor()
    tensors = GraphTensors()
    
    # Extract node features
    for node_id in graph.nodes:
        features = extractor.extract_node_features(
            node_id, graph, hic_phase_info, regional_k_map, ul_support_map
        )
        tensors.node_features[node_id] = features
        tensors.node_to_index[node_id] = len(tensors.node_to_index)
    
    tensors.num_nodes = len(tensors.node_features)
    
    # Extract edge features
    for edge_id, edge in graph.edges.items():
        from_node = getattr(edge, 'from_node', None)
        to_node = getattr(edge, 'to_node', None)
        
        if from_node and to_node:
            features = extractor.extract_edge_features(
                edge_id, from_node, to_node, graph,
                ai_annotations, hic_edge_support, regional_k_map, ul_support_map
            )
            tensors.edge_features[edge_id] = features
            tensors.edge_to_index[edge_id] = len(tensors.edge_to_index)
            tensors.edge_index.append((from_node, to_node))
    
    tensors.num_edges = len(tensors.edge_features)
    
    return tensors


def predict_paths_with_gnn(
    graph,
    gnn_model=None,
    hic_phase_info: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None,
    min_edge_confidence: float = 0.3
) -> GNNPathResult:
    """
    Main entry point for GNN path prediction.
    
    Args:
        graph: Assembly graph
        gnn_model: Optional trained GNN model
        ... (feature sources)
        min_edge_confidence: Minimum edge confidence threshold
    
    Returns:
        GNNPathResult with predicted paths and confidences
    """
    # Build tensors
    tensors = build_graph_tensors(
        graph, hic_phase_info, ai_annotations,
        hic_edge_support, regional_k_map, ul_support_map
    )
    
    # Predict edge probabilities
    gnn = PathGNN(model=gnn_model)
    edge_probs = gnn.predict_edge_probabilities(tensors)
    
    # Extract paths
    extractor = PathExtractor(min_edge_confidence=min_edge_confidence)
    result = extractor.score_and_extract_paths(graph, edge_probs)
    
    return result
