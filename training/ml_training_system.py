"""
ML Training System for StrandWeaver

This module consolidates all ML model interfaces, implementations, and training
infrastructure into a single comprehensive system.

Components:
1. Model Interfaces (Base Classes & Protocols)
   - BaseMLModel - Abstract base for all models
   - EdgeAIModel - Edge classification (EdgeWarden)
   - PathGNNModel - GNN path prediction (PathWeaver)
   - DiploidAIModel - Haplotype assignment (Haplotype Detangler)
   - ULRoutingAIModel - UL read routing (ThreadCompass)
   - SVAIModel - Structural variant detection

2. Model Implementations
   - Concrete implementations for each model type
   - Technology-specific variants
   - Ensemble methods

3. Training Infrastructure
   - ModelTrainer - Training orchestration
   - ModelEvaluator - Evaluation and metrics
   - TrainingConfig - Configuration management
   - Model registry and factory methods

Usage:
    # Create model from registry
    model = create_model('edge_classifier', technology='hifi')
    
    # Train model
    trainer = ModelTrainer(model, config)
    metrics = trainer.train(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(X_test, y_test)

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation (Consolidated Phase 3)
"""

from __future__ import annotations
import logging
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# ML framework imports (conditionally imported)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
#                    PART 1: MODEL INTERFACES & BASE CLASSES
# ============================================================================

# ============================================================================
#                           BASE MODEL INTERFACE
# ============================================================================

class BaseMLModel(ABC):
    """
    Abstract base class for all StrandWeaver ML models.
    
    Provides common interface for training, prediction, and serialization.
    """
    
    def __init__(self, model_name: str, version: str = "v1.0"):
        """
        Initialize base model.
        
        Args:
            model_name: Human-readable model name
            version: Model version string
        """
        self.model_name = model_name
        self.version = version
        self.is_trained = False
        self.training_metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            **kwargs: Training hyperparameters
        
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, input_data) -> Any:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input features
        
        Returns:
            Predictions (format depends on model)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        pass
    
    def evaluate(self, test_data) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test dataset
        
        Returns:
            Evaluation metrics
        """
        # Default implementation - subclasses can override
        logger.warning(f"{self.model_name}: Using default evaluation")
        return {"accuracy": 0.0}


# ============================================================================
#                       1. EDGE AI - OVERLAP CLASSIFIER
# ============================================================================

@dataclass
class EdgePrediction:
    """Prediction for a single edge."""
    edge_id: Tuple[str, str]  # (source, target)
    label: str  # TRUE, ALLELIC, REPEAT, SV_BREAK, CHIMERIC, UNKNOWN
    confidence: float  # 0.0 - 1.0
    class_probabilities: Dict[str, float] = field(default_factory=dict)


class EdgeAIModel(BaseMLModel):
    """
    Interface for overlap/edge classification models.
    
    Classifies edges in the assembly graph as:
    - TRUE: Legitimate overlap from same haplotype
    - ALLELIC: Overlap between homologous loci (different haplotypes)
    - REPEAT: Both reads from repeat regions
    - SV_BREAK: Edge crosses structural variant breakpoint
    - CHIMERIC: False edge between distant loci
    """
    
    def __init__(self, model_name: str = "EdgeAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.num_classes = 6  # TRUE, ALLELIC, REPEAT, SV_BREAK, CHIMERIC, UNKNOWN
        self.feature_dim = 17  # As defined in feature_builder.py
    
    @abstractmethod
    def predict_single_edge(self, features: Dict[str, float]) -> EdgePrediction:
        """
        Classify a single edge.
        
        Args:
            features: Edge feature dictionary (17 features)
        
        Returns:
            EdgePrediction with label and confidence
        """
        pass
    
    def predict(self, edges: List[Dict[str, float]]) -> List[EdgePrediction]:
        """
        Classify multiple edges.
        
        Args:
            edges: List of edge feature dictionaries
        
        Returns:
            List of EdgePrediction objects
        """
        return [self.predict_single_edge(edge) for edge in edges]
    
    @abstractmethod
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over edge classes.
        
        Args:
            features: Edge feature dictionary
        
        Returns:
            Dictionary mapping class names to probabilities
        """
        pass


# ============================================================================
#                   2. PATH GNN - GRAPH PATH PREDICTOR
# ============================================================================

@dataclass
class GraphTensors:
    """Graph structure in tensor format."""
    node_features: np.ndarray  # N × F
    edge_index: np.ndarray  # 2 × E (COO format)
    edge_features: np.ndarray  # E × F_e
    num_nodes: int = 0
    num_edges: int = 0


@dataclass
class PathPrediction:
    """Predicted path through graph."""
    path_nodes: List[str]  # Ordered node IDs
    path_score: float  # Overall path quality score
    edge_probabilities: List[float]  # Probability for each edge in path
    alternative_paths: List[List[str]] = field(default_factory=list)


class PathGNNModel(BaseMLModel):
    """
    Interface for GNN-based path prediction models.
    
    Uses graph neural networks to:
    - Learn node and edge embeddings
    - Predict correct traversal paths
    - Score edge correctness in context
    - Find optimal assembly paths
    """
    
    def __init__(self, model_name: str = "PathGNN", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.node_feature_dim = 32
        self.edge_feature_dim = 16
        self.embedding_dim = 64
    
    @abstractmethod
    def predict_edge_probabilities(
        self,
        graph: GraphTensors,
        source_node: str,
        candidate_targets: List[str]
    ) -> Dict[str, float]:
        """
        Predict probability of edges from source to each candidate target.
        
        Args:
            graph: Graph structure with node/edge features
            source_node: Source node ID
            candidate_targets: List of potential target nodes
        
        Returns:
            Dictionary mapping target node IDs to edge probabilities
        """
        pass
    
    @abstractmethod
    def find_best_path(
        self,
        graph: GraphTensors,
        start_node: str,
        end_node: str,
        max_length: int = 100
    ) -> PathPrediction:
        """
        Find highest-scoring path between two nodes.
        
        Args:
            graph: Graph structure
            start_node: Starting node ID
            end_node: Ending node ID
            max_length: Maximum path length
        
        Returns:
            PathPrediction with best path and score
        """
        pass
    
    def predict(self, graph: GraphTensors) -> List[PathPrediction]:
        """
        Predict all high-confidence paths in graph.
        
        Args:
            graph: Graph structure
        
        Returns:
            List of predicted paths
        """
        # Default implementation - find paths between all node pairs
        logger.info(f"{self.model_name}: Predicting paths in graph with {graph.num_nodes} nodes")
        return []


# ============================================================================
#              3. DIPLOID AI - HAPLOTYPE ASSIGNMENT
# ============================================================================

@dataclass
class HaplotypePrediction:
    """Haplotype assignment for a node."""
    node_id: str
    haplotype: str  # 'A', 'B', 'BOTH' (shared), 'REPEAT', 'UNKNOWN'
    confidence: float
    haplotype_scores: Dict[str, float] = field(default_factory=dict)


class DiploidAIModel(BaseMLModel):
    """
    Interface for diploid haplotype assignment models.
    
    Assigns assembly graph nodes to haplotypes using:
    - GNN embeddings
    - Hi-C contact patterns
    - Coverage signals
    - Allelic markers
    - Repeat content
    """
    
    def __init__(self, model_name: str = "DiploidAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.num_haplotypes = 5  # A, B, BOTH, REPEAT, UNKNOWN
        self.feature_dim = 42  # 32 GNN + 10 signals
    
    @abstractmethod
    def predict_single_node(self, features: Dict[str, float]) -> HaplotypePrediction:
        """
        Assign haplotype to a single node.
        
        Args:
            features: Node feature dictionary (42D)
        
        Returns:
            HaplotypePrediction with assignment and confidence
        """
        pass
    
    def predict(self, nodes: List[Dict[str, float]]) -> List[HaplotypePrediction]:
        """
        Assign haplotypes to multiple nodes.
        
        Args:
            nodes: List of node feature dictionaries
        
        Returns:
            List of HaplotypePrediction objects
        """
        return [self.predict_single_node(node) for node in nodes]
    
    @abstractmethod
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over haplotypes.
        
        Args:
            features: Node feature dictionary
        
        Returns:
            Dictionary mapping haplotype labels to probabilities
        """
        pass
    
    @abstractmethod
    def disentangle_graph(
        self,
        graph: GraphTensors,
        node_features: List[Dict[str, float]]
    ) -> Tuple[List[str], List[str]]:
        """
        Partition entire graph into two haplotypes.
        
        Args:
            graph: Graph structure
            node_features: Features for all nodes
        
        Returns:
            Tuple of (haplotype_A_nodes, haplotype_B_nodes)
        """
        pass


# ============================================================================
#              4. UL ROUTING AI - ULTRALONG READ ROUTING
# ============================================================================

@dataclass
class RoutePrediction:
    """Predicted route for an ultralong read."""
    read_id: str
    path: List[str]  # Ordered node IDs
    route_score: float  # 0.0 - 1.0
    confidence: float
    alternative_routes: List[List[str]] = field(default_factory=list)


class ULRoutingAIModel(BaseMLModel):
    """
    Interface for ultralong read routing models.
    
    Routes ultralong reads through assembly graphs using:
    - Path topology features
    - Coverage consistency
    - Sequence identity
    - Graph branching patterns
    """
    
    def __init__(self, model_name: str = "ULRoutingAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.feature_dim = 12  # As defined in feature_builder.py
    
    @abstractmethod
    def score_route(self, features: Dict[str, float]) -> float:
        """
        Score a candidate route for an ultralong read.
        
        Args:
            features: Route feature dictionary (12D)
        
        Returns:
            Route quality score (0.0 - 1.0)
        """
        pass
    
    @abstractmethod
    def find_best_route(
        self,
        read_id: str,
        graph: GraphTensors,
        candidate_nodes: List[str],
        read_sequence: str
    ) -> RoutePrediction:
        """
        Find best route for an ultralong read through the graph.
        
        Args:
            read_id: Read identifier
            graph: Assembly graph
            candidate_nodes: Nodes the read might traverse
            read_sequence: Read sequence for alignment
        
        Returns:
            RoutePrediction with best path and score
        """
        pass
    
    def predict(
        self,
        reads: List[Tuple[str, str]],  # (read_id, sequence)
        graph: GraphTensors
    ) -> List[RoutePrediction]:
        """
        Route multiple ultralong reads.
        
        Args:
            reads: List of (read_id, sequence) tuples
            graph: Assembly graph
        
        Returns:
            List of RoutePrediction objects
        """
        predictions = []
        for read_id, sequence in reads:
            # Default: use all nodes as candidates
            candidate_nodes = [f"node_{i}" for i in range(graph.num_nodes)]
            pred = self.find_best_route(read_id, graph, candidate_nodes, sequence)
            predictions.append(pred)
        return predictions


# ============================================================================
#              5. SV AI - STRUCTURAL VARIANT DETECTION
# ============================================================================

@dataclass
class SVPrediction:
    """Predicted structural variant."""
    region_id: str
    chrom: str
    start: int
    end: int
    sv_type: str  # deletion, insertion, inversion, duplication, translocation
    sv_size: int
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)  # Coverage, Hi-C, etc.


class SVAIModel(BaseMLModel):
    """
    Interface for structural variant detection models.
    
    Detects SVs in assembly graphs using:
    - Coverage patterns (drops, spikes)
    - Graph branching complexity
    - Hi-C contact disruptions
    - Ultralong read support
    - Sequence composition
    """
    
    def __init__(self, model_name: str = "SVAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.feature_dim = 14  # As defined in feature_builder.py
        self.sv_types = ['deletion', 'insertion', 'inversion', 'duplication', 'translocation', 'none']
    
    @abstractmethod
    def predict_single_region(self, features: Dict[str, float]) -> SVPrediction:
        """
        Detect SV in a single genomic region.
        
        Args:
            features: Region feature dictionary (14D)
        
        Returns:
            SVPrediction with SV type and confidence
        """
        pass
    
    def predict(self, regions: List[Dict[str, float]]) -> List[SVPrediction]:
        """
        Detect SVs in multiple regions.
        
        Args:
            regions: List of region feature dictionaries
        
        Returns:
            List of SVPrediction objects
        """
        return [self.predict_single_region(region) for region in regions]
    
    @abstractmethod
    def predict_sv_type(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over SV types.
        
        Args:
            features: Region feature dictionary
        
        Returns:
            Dictionary mapping SV types to probabilities
        """
        pass
    
    @abstractmethod
    def scan_genome(
        self,
        graph: GraphTensors,
        coverage: Dict[str, float],
        window_size: int = 1000
    ) -> List[SVPrediction]:
        """
        Scan entire genome for structural variants.
        
        Args:
            graph: Assembly graph
            coverage: Coverage information per node
            window_size: Window size for scanning
        
        Returns:
            List of detected SVs
        """
        pass


# ============================================================================
#                       MODEL FACTORY AND REGISTRY
# ============================================================================

class ModelRegistry:
    """
    Registry for ML model implementations.
    
    Allows registration and retrieval of model implementations.
    """
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: type):
        """
        Register a model implementation.
        
        Args:
            model_type: Model type identifier
            model_class: Model class to register
        """
        cls._models[model_type] = model_class
        logger.info(f"Registered model: {model_type} -> {model_class.__name__}")
    
    @classmethod
    def get(cls, model_type: str) -> type:
        """
        Get a registered model class.
        
        Args:
            model_type: Model type identifier
        
        Returns:
            Model class
        
        Raises:
            KeyError: If model type not registered
        """
        if model_type not in cls._models:
            raise KeyError(f"Model type '{model_type}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[model_type]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of registered model types."""
        return list(cls._models.keys())


def create_model(model_type: str, **kwargs) -> BaseMLModel:
    """
    Factory function to create ML models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    
    Example:
        >>> edge_model = create_model('edge_ai', version='v2.0')
        >>> predictions = edge_model.predict(edge_features)
    """
    model_class = ModelRegistry.get(model_type)
    return model_class(**kwargs)


# ============================================================================
#                       TRAINING UTILITIES
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    device: str = "cpu"  # or "cuda"
    random_seed: int = 42


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class ModelTrainer:
    """
    Generic trainer for StrandWeaver ML models.
    
    Handles common training loop, checkpointing, and logging.
    """
    
    def __init__(self, model: BaseMLModel, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train(self, train_data, val_data=None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {self.model.model_name} for {self.config.num_epochs} epochs")
        
        # Delegate to model's train method
        results = self.model.train(train_data, val_data, config=self.config)
        
        logger.info(f"Training complete: {results}")
        return results
    
    def save_checkpoint(self, epoch: int, path: Path):
        """Save training checkpoint."""
        checkpoint_path = Path(path) / f"{self.model.model_name}_epoch_{epoch}.pt"
        self.model.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


# ============================================================================
#                       EVALUATION UTILITIES
# ============================================================================

class ModelEvaluator:
    """
    Evaluator for StrandWeaver ML models.
    
    Computes standard metrics for each model type.
    """
    
    @staticmethod
    def evaluate_classification(
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate classification model.
        
        Args:
            predictions: Predicted labels
            ground_truth: True labels
        
        Returns:
            Dictionary of metrics (accuracy, precision, recall, F1)
        """
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return {
            'accuracy': accuracy,
            'num_samples': len(predictions),
            'num_correct': correct
        }
    
    @staticmethod
    def evaluate_path_prediction(
        predicted_paths: List[List[str]],
        ground_truth_paths: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate path prediction model.
        
        Args:
            predicted_paths: Predicted paths
            ground_truth_paths: True paths
        
        Returns:
            Dictionary of metrics (path accuracy, node precision, etc.)
        """
        exact_matches = sum(1 for p, gt in zip(predicted_paths, ground_truth_paths) if p == gt)
        path_accuracy = exact_matches / len(predicted_paths) if predicted_paths else 0.0
        
        return {
            'path_accuracy': path_accuracy,
            'exact_matches': exact_matches,
            'num_paths': len(predicted_paths)
        }


# ============================================================================
#              6. ADAPTIVE KMER AI - K-MER SELECTION FOR CORRECTION
# ============================================================================

@dataclass
class KmerPrediction:
    """Predicted optimal k-mer size for correction."""
    read_id: str
    region_start: int
    region_end: int
    optimal_k: int  # Predicted k-mer size (e.g., 15, 17, 19, 21, 25, 31)
    confidence: float  # 0.0-1.0
    context_features: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)  # over_correction_risk, under_correction_risk


@dataclass
class ReadContext:
    """Sequence context for k-mer selection."""
    read_id: str
    sequence: str
    quality_scores: Optional[List[int]] = None
    technology: str = "ont"  # ont, pacbio, illumina
    region_start: int = 0
    region_end: Optional[int] = None


class AdaptiveKmerAIModel(BaseMLModel):
    """
    Interface for adaptive k-mer selection models.
    
    Predicts optimal k-mer size for read correction based on:
    - Sequence context (homopolymers, STRs, low-complexity)
    - Technology-specific error profiles
    - Quality score distributions
    - Local sequence entropy
    - Over-correction risk
    
    This replaces hard-coded k-mer choices with learned optimal values.
    """
    
    def __init__(self, model_name: str = "AdaptiveKmerAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.supported_k_values = [15, 17, 19, 21, 25, 31, 41, 51]
        self.feature_dim = 32  # Context features for k-mer selection
    
    @abstractmethod
    def predict_optimal_k(self, context: ReadContext) -> KmerPrediction:
        """
        Predict optimal k-mer size for a read/region.
        
        Args:
            context: Read sequence and metadata
        
        Returns:
            KmerPrediction with optimal k and confidence
        """
        pass
    
    @abstractmethod
    def extract_context_features(self, context: ReadContext) -> Dict[str, float]:
        """
        Extract features from sequence context.
        
        Features include:
        - Homopolymer run lengths (mean, max, variance)
        - STR density (di, tri, tetra-nucleotide repeats)
        - Low-complexity score (Shannon entropy, GC content)
        - Quality score statistics (mean, min, variance)
        - Technology-specific error rates
        
        Args:
            context: Read sequence and metadata
        
        Returns:
            Feature dictionary (32D)
        """
        pass
    
    @abstractmethod
    def assess_correction_risk(self, context: ReadContext, k: int) -> Dict[str, float]:
        """
        Assess risk of over/under-correction for given k.
        
        Args:
            context: Read sequence and metadata
            k: K-mer size to assess
        
        Returns:
            Risk scores: {
                'over_correction_risk': 0.0-1.0,
                'under_correction_risk': 0.0-1.0,
                'confidence': 0.0-1.0
            }
        """
        pass
    
    def predict(self, contexts: List[ReadContext]) -> List[KmerPrediction]:
        """
        Predict optimal k for multiple reads/regions.
        
        Args:
            contexts: List of read contexts
        
        Returns:
            List of KmerPrediction objects
        """
        return [self.predict_optimal_k(ctx) for ctx in contexts]
    
    @abstractmethod
    def predict_k_distribution(self, context: ReadContext) -> Dict[int, float]:
        """
        Get probability distribution over k-mer sizes.
        
        Args:
            context: Read sequence and metadata
        
        Returns:
            Dictionary mapping k values to probabilities
        """
        pass


# ============================================================================
#         7. BASE ERROR CLASSIFIER AI - PER-BASE ERROR CLASSIFICATION
# ============================================================================

@dataclass
class BaseErrorPrediction:
    """Per-base error classification."""
    read_id: str
    position: int
    base: str
    error_class: str  # 'correct', 'error', 'ambiguous'
    confidence: float  # 0.0-1.0
    error_probabilities: Dict[str, float] = field(default_factory=dict)  # correct, error, ambiguous
    suggested_correction: Optional[str] = None  # If error_class == 'error'


@dataclass
class BaseContext:
    """Context for per-base error classification."""
    read_id: str
    position: int
    base: str
    quality_score: Optional[int] = None
    left_context: str = ""  # 5-10 bases before
    right_context: str = ""  # 5-10 bases after
    kmer_coverage: Optional[int] = None  # Coverage of k-mer containing this base
    technology: str = "ont"


class BaseErrorClassifierAIModel(BaseMLModel):
    """
    Interface for per-base error classification models.
    
    Classifies each base as:
    - 'correct': High confidence correct base
    - 'error': High confidence error (needs correction)
    - 'ambiguous': Uncertain (skip correction or use cautious approach)
    
    Benefits:
    - Selective correction (only fix high-confidence errors)
    - Better priors for k-mer weighting
    - Avoid miscorrection in ancient DNA
    - Faster polishing (skip correct regions)
    
    Model types:
    - Lightweight CNN on one-hot encoded k-mers
    - Transformer with learned base embeddings
    - Gradient boosting on engineered features
    """
    
    def __init__(self, model_name: str = "BaseErrorClassifierAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.error_classes = ['correct', 'error', 'ambiguous']
        self.feature_dim = 48  # Features for base classification
    
    @abstractmethod
    def classify_base(self, context: BaseContext) -> BaseErrorPrediction:
        """
        Classify a single base as correct/error/ambiguous.
        
        Args:
            context: Base and surrounding context
        
        Returns:
            BaseErrorPrediction with classification and confidence
        """
        pass
    
    @abstractmethod
    def extract_base_features(self, context: BaseContext) -> Dict[str, float]:
        """
        Extract features for base classification.
        
        Features include:
        - Quality score
        - K-mer coverage (if available)
        - Homopolymer context (length, position in run)
        - Dinucleotide context
        - GC content in window
        - Entropy of surrounding sequence
        - Technology-specific error signatures
        
        Args:
            context: Base and surrounding context
        
        Returns:
            Feature dictionary (48D)
        """
        pass
    
    def predict(self, contexts: List[BaseContext]) -> List[BaseErrorPrediction]:
        """
        Classify multiple bases.
        
        Args:
            contexts: List of base contexts
        
        Returns:
            List of BaseErrorPrediction objects
        """
        return [self.classify_base(ctx) for ctx in contexts]
    
    @abstractmethod
    def classify_read(
        self,
        read_id: str,
        sequence: str,
        quality_scores: Optional[List[int]] = None,
        kmer_coverage: Optional[Dict[int, int]] = None
    ) -> List[BaseErrorPrediction]:
        """
        Classify all bases in a read.
        
        Args:
            read_id: Read identifier
            sequence: Read sequence
            quality_scores: Optional quality scores
            kmer_coverage: Optional k-mer coverage map
        
        Returns:
            List of BaseErrorPrediction for each position
        """
        pass
    
    @abstractmethod
    def get_error_mask(
        self,
        sequence: str,
        confidence_threshold: float = 0.8
    ) -> List[bool]:
        """
        Get binary mask of likely errors.
        
        Args:
            sequence: Read sequence
            confidence_threshold: Minimum confidence for error classification
        
        Returns:
            Boolean list (True = likely error, False = likely correct/ambiguous)
        """
        pass
    
    @abstractmethod
    def suggest_corrections(
        self,
        read_id: str,
        sequence: str,
        quality_scores: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """
        Suggest corrections for likely errors.
        
        Args:
            read_id: Read identifier
            sequence: Read sequence
            quality_scores: Optional quality scores
        
        Returns:
            Dictionary mapping position -> suggested base
        """
        pass
    
    @abstractmethod
    def get_correction_priority(
        self,
        predictions: List[BaseErrorPrediction]
    ) -> List[Tuple[int, float]]:
        """
        Get correction priority order.
        
        Returns positions sorted by error confidence (high to low).
        Useful for selective correction strategies.
        
        Args:
            predictions: List of base predictions
        
        Returns:
            List of (position, error_confidence) tuples, sorted by confidence
        """
        pass




# ============================================================================
#                    PART 2: MODEL IMPLEMENTATIONS
# ============================================================================

# ============================================================================
#                    1. EDGE AI - OVERLAP CLASSIFIER
# ============================================================================

class XGBoostEdgeAI(EdgeAIModel):
    """
    XGBoost-based overlap/edge classifier.
    
    Features (17D):
    - overlap_length, overlap_identity, read1_length, read2_length
    - coverage_r1, coverage_r2, gc_content_r1, gc_content_r2
    - repeat_fraction_r1, repeat_fraction_r2, kmer_diversity_r1, kmer_diversity_r2
    - branching_factor_r1, branching_factor_r2, hic_support
    - mapping_quality_r1, mapping_quality_r2
    
    Labels: TRUE, ALLELIC, REPEAT, SV_BREAK, CHIMERIC, UNKNOWN
    """
    
    def __init__(self, model_name: str = "XGBoostEdgeAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {
            'TRUE': 0, 'ALLELIC': 1, 'REPEAT': 2,
            'SV_BREAK': 3, 'CHIMERIC': 4, 'UNKNOWN': 5
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost classifier.
        
        Args:
            train_data: List of (features_dict, label) tuples
            val_data: Optional validation data
            **kwargs: XGBoost parameters
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and labels
        X_train = np.array([self._dict_to_array(x[0]) for x in train_data])
        y_train = np.array([self.label_encoder[x[1].upper() if isinstance(x[1], str) else x[1]] for x in train_data])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': self.num_classes,
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'n_estimators': kwargs.get('n_estimators', 100),
            'random_state': kwargs.get('random_seed', 42),
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        if val_data:
            X_val = np.array([self._dict_to_array(x[0]) for x in val_data])
            y_val = np.array([self.label_encoder[x[1].upper() if isinstance(x[1], str) else x[1]] for x in val_data])
            X_val = self.scaler.transform(X_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=kwargs.get('verbose', 10)
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            logger.info(f"Validation accuracy: {accuracy:.4f}")
        else:
            self.model.fit(X_train, y_train, verbose=kwargs.get('verbose', 10))
            accuracy = 0.0
        
        self.is_trained = True
        self.training_metadata = {
            'num_train_examples': len(train_data),
            'num_val_examples': len(val_data) if val_data else 0,
            'val_accuracy': accuracy,
            'params': params
        }
        
        return {'val_accuracy': accuracy, 'num_epochs': params['n_estimators']}
    
    def predict_single_edge(self, features: Dict[str, float]) -> EdgePrediction:
        """Classify a single edge."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Convert features to array
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X)[0]
        label_idx = np.argmax(proba)
        label = self.label_decoder[label_idx]
        confidence = proba[label_idx]
        
        # Build class probabilities
        class_probs = {self.label_decoder[i]: float(p) for i, p in enumerate(proba)}
        
        return EdgePrediction(
            edge_id=('unknown', 'unknown'),
            label=label,
            confidence=confidence,
            class_probabilities=class_probs
        )
    
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability distribution over edge classes."""
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[0]
        return {self.label_decoder[i]: float(p) for i, p in enumerate(proba)}
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained,
            'version': self.version
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.label_decoder = metadata['label_decoder']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = metadata['is_trained']
        self.version = metadata['version']
        
        logger.info(f"Loaded {self.model_name} from {path}")
    
    def _dict_to_array(self, features) -> np.ndarray:
        """Convert feature dictionary or list to numpy array."""
        # Handle both list and dict formats
        if isinstance(features, (list, np.ndarray)):
            return np.array(features)
        
        # Dictionary format: Expected feature order (17 features)
        feature_names = [
            'overlap_length', 'overlap_identity', 'read1_length', 'read2_length',
            'coverage_r1', 'coverage_r2', 'gc_content_r1', 'gc_content_r2',
            'repeat_fraction_r1', 'repeat_fraction_r2',
            'kmer_diversity_r1', 'kmer_diversity_r2',
            'branching_factor_r1', 'branching_factor_r2',
            'hic_support', 'mapping_quality_r1', 'mapping_quality_r2'
        ]
        return np.array([features.get(name, 0.0) for name in feature_names])


# ============================================================================
#              3. DIPLOID AI - HAPLOTYPE ASSIGNMENT
# ============================================================================

class XGBoostDiploidAI(DiploidAIModel):
    """
    XGBoost-based haplotype classifier.
    
    Features (42D):
    - 32D GNN node embeddings
    - 10D signal features:
      * coverage, gc_content, repeat_fraction
      * kmer_diversity, branching_factor
      * hic_contact_density, allele_frequency
      * heterozygosity, phase_consistency, mappability
    
    Labels: A, B, BOTH (shared), REPEAT, UNKNOWN
    """
    
    def __init__(self, model_name: str = "XGBoostDiploidAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {
            'A': 0, 'B': 1, 'BOTH': 2, 'REPEAT': 3, 'UNKNOWN': 4
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost classifier for haplotype assignment.
        
        Args:
            train_data: List of (features_dict, label) tuples
            val_data: Optional validation data
            **kwargs: XGBoost parameters
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and labels
        X_train = np.array([self._dict_to_array(x[0]) for x in train_data])
        y_train = np.array([self.label_encoder[x[1].upper() if isinstance(x[1], str) else x[1]] for x in train_data])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': self.num_haplotypes,
            'max_depth': kwargs.get('max_depth', 8),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'n_estimators': kwargs.get('n_estimators', 200),
            'random_state': kwargs.get('random_seed', 42),
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        if val_data:
            X_val = np.array([self._dict_to_array(x[0]) for x in val_data])
            y_val = np.array([self.label_encoder[x[1].upper() if isinstance(x[1], str) else x[1]] for x in val_data])
            X_val = self.scaler.transform(X_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=kwargs.get('verbose', 10)
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='weighted', zero_division=0
            )
            
            logger.info(f"Validation accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            self.model.fit(X_train, y_train, verbose=kwargs.get('verbose', 10))
            accuracy = precision = recall = f1 = 0.0
        
        self.is_trained = True
        self.training_metadata = {
            'num_train_examples': len(train_data),
            'num_val_examples': len(val_data) if val_data else 0,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'params': params
        }
        
        return {
            'val_accuracy': accuracy,
            'val_f1': f1,
            'num_epochs': params['n_estimators']
        }
    
    def predict_single_node(self, features: Dict[str, float]) -> HaplotypePrediction:
        """Assign haplotype to a single node."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Convert features to array
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X)[0]
        label_idx = np.argmax(proba)
        label = self.label_decoder[label_idx]
        confidence = proba[label_idx]
        
        # Build haplotype scores
        hap_scores = {self.label_decoder[i]: float(p) for i, p in enumerate(proba)}
        
        return HaplotypePrediction(
            node_id='unknown',
            haplotype=label,
            confidence=confidence,
            haplotype_scores=hap_scores
        )
    
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability distribution over haplotypes."""
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[0]
        return {self.label_decoder[i]: float(p) for i, p in enumerate(proba)}
    
    def disentangle_graph(
        self,
        graph: GraphTensors,
        node_features: List[Dict[str, float]]
    ) -> Tuple[List[str], List[str]]:
        """
        Partition entire graph into two haplotypes.
        
        Args:
            graph: Graph structure
            node_features: Features for all nodes
        
        Returns:
            Tuple of (haplotype_A_nodes, haplotype_B_nodes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Predict haplotypes for all nodes
        predictions = self.predict(node_features)
        
        hap_a_nodes = []
        hap_b_nodes = []
        
        for i, pred in enumerate(predictions):
            node_id = f"node_{i}"
            if pred.haplotype == 'A':
                hap_a_nodes.append(node_id)
            elif pred.haplotype == 'B':
                hap_b_nodes.append(node_id)
            # Skip BOTH, REPEAT, UNKNOWN
        
        logger.info(f"Disentangled graph: {len(hap_a_nodes)} hap A, {len(hap_b_nodes)} hap B")
        return (hap_a_nodes, hap_b_nodes)
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained,
            'version': self.version
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.label_decoder = metadata['label_decoder']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = metadata['is_trained']
        self.version = metadata['version']
        
        logger.info(f"Loaded {self.model_name} from {path}")
    
    def _dict_to_array(self, features) -> np.ndarray:
        """Convert feature dictionary or list to numpy array."""
        # Handle both list and dict formats
        if isinstance(features, (list, np.ndarray)):
            return np.array(features)
        
        # Dictionary format: 32D GNN embeddings + 10D signals
        gnn_features = [features.get(f'gnn_dim_{i}', 0.0) for i in range(32)]
        signal_features = [
            features.get('coverage', 0.0),
            features.get('gc_content', 0.0),
            features.get('repeat_fraction', 0.0),
            features.get('kmer_diversity', 0.0),
            features.get('branching_factor', 0.0),
            features.get('hic_contact_density', 0.0),
            features.get('allele_frequency', 0.0),
            features.get('heterozygosity', 0.0),
            features.get('phase_consistency', 0.0),
            features.get('mappability', 0.0)
        ]
        return np.array(gnn_features + signal_features)


# ============================================================================
#              5. SV AI - STRUCTURAL VARIANT DETECTION
# ============================================================================

class XGBoostSVAI(SVAIModel):
    """
    XGBoost-based structural variant detector.
    
    Features (14D):
    - coverage_mean, coverage_std, coverage_median
    - gc_content, repeat_fraction, kmer_diversity
    - branching_complexity, hic_disruption_score
    - ul_support, mapping_quality
    - region_length, breakpoint_precision
    - allele_balance, phase_switch_rate
    
    SV Types: deletion, insertion, inversion, duplication, translocation, none
    """
    
    def __init__(self, model_name: str = "XGBoostSVAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {
            'deletion': 0, 'insertion': 1, 'inversion': 2,
            'duplication': 3, 'translocation': 4, 'none': 5
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost classifier for SV detection.
        
        Args:
            train_data: List of (features_dict, sv_type) tuples
            val_data: Optional validation data
            **kwargs: XGBoost parameters
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and labels
        X_train = np.array([self._dict_to_array(x[0]) for x in train_data])
        y_train = np.array([self.label_encoder[x[1].lower() if isinstance(x[1], str) else x[1]] for x in train_data])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.sv_types),
            'max_depth': kwargs.get('max_depth', 7),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'n_estimators': kwargs.get('n_estimators', 150),
            'random_state': kwargs.get('random_seed', 42),
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'scale_pos_weight': kwargs.get('scale_pos_weight', 2.0)  # Handle class imbalance
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        if val_data:
            X_val = np.array([self._dict_to_array(x[0]) for x in val_data])
            y_val = np.array([self.label_encoder[x[1].lower() if isinstance(x[1], str) else x[1]] for x in val_data])
            X_val = self.scaler.transform(X_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=kwargs.get('verbose', 10)
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            logger.info(f"Validation accuracy: {accuracy:.4f}")
        else:
            self.model.fit(X_train, y_train, verbose=kwargs.get('verbose', 10))
            accuracy = 0.0
        
        self.is_trained = True
        self.training_metadata = {
            'num_train_examples': len(train_data),
            'num_val_examples': len(val_data) if val_data else 0,
            'val_accuracy': accuracy,
            'params': params
        }
        
        return {'val_accuracy': accuracy, 'num_epochs': params['n_estimators']}
    
    def predict_single_region(self, features: Dict[str, float]) -> SVPrediction:
        """Detect SV in a single genomic region."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Convert features to array
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X)[0]
        sv_type_idx = np.argmax(proba)
        sv_type = self.label_decoder[sv_type_idx]
        confidence = proba[sv_type_idx]
        
        return SVPrediction(
            region_id='unknown',
            chrom='unknown',
            start=0,
            end=0,
            sv_type=sv_type,
            sv_size=features.get('region_length', 0),
            confidence=confidence,
            evidence={
                'coverage_mean': features.get('coverage_mean', 0.0),
                'hic_disruption': features.get('hic_disruption_score', 0.0),
                'ul_support': features.get('ul_support', 0.0)
            }
        )
    
    def predict_sv_type(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability distribution over SV types."""
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[0]
        return {self.label_decoder[i]: float(p) for i, p in enumerate(proba)}
    
    def scan_genome(
        self,
        graph: GraphTensors,
        coverage: Dict[str, float],
        window_size: int = 1000
    ) -> List[SVPrediction]:
        """
        Scan entire genome for structural variants.
        
        Args:
            graph: Assembly graph
            coverage: Coverage information per node
            window_size: Window size for scanning
        
        Returns:
            List of detected SVs
        """
        logger.info(f"Scanning genome for SVs (window size: {window_size})")
        
        # Placeholder implementation
        # In real implementation, would slide windows across graph
        svs = []
        
        logger.info(f"Detected {len(svs)} structural variants")
        return svs
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained,
            'version': self.version
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.label_decoder = metadata['label_decoder']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = metadata['is_trained']
        self.version = metadata['version']
        
        logger.info(f"Loaded {self.model_name} from {path}")
    
    def _dict_to_array(self, features) -> np.ndarray:
        """Convert feature dictionary or list to numpy array."""
        # Handle both list and dict formats
        if isinstance(features, (list, np.ndarray)):
            return np.array(features)
        
        # Dictionary format
        feature_names = [
            'coverage_mean', 'coverage_std', 'coverage_median',
            'gc_content', 'repeat_fraction', 'kmer_diversity',
            'branching_complexity', 'hic_disruption_score',
            'ul_support', 'mapping_quality',
            'region_length', 'breakpoint_precision',
            'allele_balance', 'phase_switch_rate'
        ]
        return np.array([features.get(name, 0.0) for name in feature_names])


# ============================================================================
#              2. PATH GNN - GRAPH NEURAL NETWORK
# ============================================================================

class SimplePathGNN(PathGNNModel):
    """
    Simplified path prediction using XGBoost on graph features.
    
    This is a lightweight alternative to full GNN that uses:
    - Aggregated node/edge features
    - Path topology metrics
    - Coverage consistency along paths
    
    For production use, consider implementing with PyTorch Geometric or DGL.
    """
    
    def __init__(self, model_name: str = "SimplePathGNN", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.edge_scorer = None  # XGBoost model for edge scoring
        self.scaler = StandardScaler()
    
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train path prediction model.
        
        Args:
            train_data: List of (edge_features, label) tuples
                       where label is 0 (not in correct path) or 1 (in correct path)
            val_data: Optional validation data
            **kwargs: XGBoost parameters
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} edge examples")
        
        # Extract features and labels
        X_train = np.array([self._edge_features_to_array(x[0]) for x in train_data])
        y_train = np.array([x[1] for x in train_data])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters for binary classification
        params = {
            'objective': 'binary:logistic',
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'n_estimators': kwargs.get('n_estimators', 100),
            'random_state': kwargs.get('random_seed', 42),
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }
        
        # Train model
        self.edge_scorer = xgb.XGBClassifier(**params)
        
        if val_data:
            X_val = np.array([self._edge_features_to_array(x[0]) for x in val_data])
            y_val = np.array([x[1] for x in val_data])
            X_val = self.scaler.transform(X_val)
            
            self.edge_scorer.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=kwargs.get('verbose', 10)
            )
            
            # Evaluate
            y_pred = self.edge_scorer.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            logger.info(f"Validation accuracy: {accuracy:.4f}")
        else:
            self.edge_scorer.fit(X_train, y_train, verbose=kwargs.get('verbose', 10))
            accuracy = 0.0
        
        self.is_trained = True
        self.training_metadata = {
            'num_train_examples': len(train_data),
            'num_val_examples': len(val_data) if val_data else 0,
            'val_accuracy': accuracy,
            'params': params
        }
        
        return {'val_accuracy': accuracy, 'num_epochs': params['n_estimators']}
    
    def predict_edge_probabilities(
        self,
        graph: GraphTensors,
        source_node: str,
        candidate_targets: List[str]
    ) -> Dict[str, float]:
        """
        Predict probability of edges from source to each candidate target.
        
        Args:
            graph: Graph structure with node/edge features
            source_node: Source node ID
            candidate_targets: List of potential target nodes
        
        Returns:
            Dictionary mapping target node IDs to edge probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        probabilities = {}
        
        for target in candidate_targets:
            # Extract edge features (simplified - would use actual graph topology)
            edge_features = self._extract_edge_features(graph, source_node, target)
            
            # Score edge
            X = self._edge_features_to_array(edge_features).reshape(1, -1)
            X = self.scaler.transform(X)
            prob = self.edge_scorer.predict_proba(X)[0][1]  # Probability of class 1 (correct edge)
            
            probabilities[target] = float(prob)
        
        return probabilities
    
    def find_best_path(
        self,
        graph: GraphTensors,
        start_node: str,
        end_node: str,
        max_length: int = 100
    ) -> PathPrediction:
        """
        Find highest-scoring path between two nodes using greedy search.
        
        Args:
            graph: Graph structure
            start_node: Starting node ID
            end_node: Ending node ID
            max_length: Maximum path length
        
        Returns:
            PathPrediction with best path and score
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Greedy path search
        path = [start_node]
        current = start_node
        path_probs = []
        
        for _ in range(max_length):
            if current == end_node:
                break
            
            # Get neighbors (simplified - would use actual graph adjacency)
            neighbors = self._get_neighbors(graph, current)
            
            if not neighbors:
                break
            
            # Score all edges from current node
            edge_probs = self.predict_edge_probabilities(graph, current, neighbors)
            
            # Choose highest probability edge
            best_neighbor = max(edge_probs.items(), key=lambda x: x[1])
            path.append(best_neighbor[0])
            path_probs.append(best_neighbor[1])
            current = best_neighbor[0]
        
        # Calculate overall path score
        path_score = np.mean(path_probs) if path_probs else 0.0
        
        return PathPrediction(
            path_nodes=path,
            path_score=path_score,
            edge_probabilities=path_probs,
            alternative_paths=[]
        )
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.edge_scorer.save_model(str(path.with_suffix('.json')))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained,
            'version': self.version
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        self.edge_scorer = xgb.XGBClassifier()
        self.edge_scorer.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = metadata['is_trained']
        self.version = metadata['version']
        
        logger.info(f"Loaded {self.model_name} from {path}")
    
    def _edge_features_to_array(self, features) -> np.ndarray:
        """Convert edge feature dictionary or list to numpy array."""
        if isinstance(features, (list, np.ndarray)):
            return np.array(features)
        
        # Dictionary format: 16D edge features
        feature_names = [
            'overlap_length', 'overlap_identity', 'coverage_consistency',
            'gc_similarity', 'repeat_match', 'branching_score',
            'path_support', 'hic_contact', 'mapping_quality',
            'kmer_match', 'sequence_complexity', 'orientation_score',
            'distance_score', 'topology_score', 'ul_support', 'sv_evidence'
        ]
        return np.array([features.get(name, 0.0) for name in feature_names])
    
    def _extract_edge_features(self, graph: GraphTensors, source: str, target: str) -> Dict[str, float]:
        """Extract features for a specific edge (simplified stub)."""
        # In production, would extract from graph.edge_features based on edge index
        return {
            'overlap_length': 1000.0,
            'overlap_identity': 0.95,
            'coverage_consistency': 0.9,
            'gc_similarity': 0.8,
            'repeat_match': 0.1,
            'branching_score': 2.0,
            'path_support': 5.0,
            'hic_contact': 1.0,
            'mapping_quality': 40.0,
            'kmer_match': 0.85,
            'sequence_complexity': 0.7,
            'orientation_score': 1.0,
            'distance_score': 0.9,
            'topology_score': 0.8,
            'ul_support': 3.0,
            'sv_evidence': 0.0
        }
    
    def _get_neighbors(self, graph: GraphTensors, node: str) -> List[str]:
        """Get neighboring nodes (simplified stub)."""
        # In production, would use graph.edge_index to find actual neighbors
        return []


# ============================================================================
#              4. UL ROUTING AI - ULTRALONG READ ROUTING
# ============================================================================

class XGBoostULRoutingAI(ULRoutingAIModel):
    """
    XGBoost-based ultralong read routing model.
    
    Features (12D):
    - path_length, num_branches, coverage_mean, coverage_std
    - sequence_identity, mapping_quality, num_gaps, gap_size_mean
    - kmer_consistency, orientation_consistency, ul_span, route_complexity
    """
    
    def __init__(self, model_name: str = "XGBoostULRoutingAI", version: str = "v1.0"):
        super().__init__(model_name, version)
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, train_data, val_data=None, **kwargs) -> Dict[str, Any]:
        """
        Train UL routing regression model.
        
        Args:
            train_data: List of (features, route_score) tuples
                       where route_score is 0.0-1.0 quality score
            val_data: Optional validation data
            **kwargs: XGBoost parameters
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and scores
        X_train = np.array([self._dict_to_array(x[0]) for x in train_data])
        y_train = np.array([x[1] for x in train_data])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters for regression
        params = {
            'objective': 'reg:squarederror',
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'n_estimators': kwargs.get('n_estimators', 100),
            'random_state': kwargs.get('random_seed', 42),
            'tree_method': 'hist',
            'eval_metric': 'rmse'
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        
        if val_data:
            X_val = np.array([self._dict_to_array(x[0]) for x in val_data])
            y_val = np.array([x[1] for x in val_data])
            X_val = self.scaler.transform(X_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=kwargs.get('verbose', 10)
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mae = np.mean(np.abs(y_val - y_pred))
            
            logger.info(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        else:
            self.model.fit(X_train, y_train, verbose=kwargs.get('verbose', 10))
            rmse = mae = 0.0
        
        self.is_trained = True
        self.training_metadata = {
            'num_train_examples': len(train_data),
            'num_val_examples': len(val_data) if val_data else 0,
            'val_rmse': rmse,
            'val_mae': mae,
            'params': params
        }
        
        return {'val_rmse': float(rmse), 'val_mae': float(mae), 'num_epochs': params['n_estimators']}
    
    def score_route(self, features: Dict[str, float]) -> float:
        """
        Score a candidate route for an ultralong read.
        
        Args:
            features: Route feature dictionary (12D)
        
        Returns:
            Route quality score (0.0 - 1.0)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        X = self._dict_to_array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        score = self.model.predict(X)[0]
        
        # Clip to [0, 1] range
        return float(np.clip(score, 0.0, 1.0))
    
    def find_best_route(
        self,
        read_id: str,
        graph: GraphTensors,
        candidate_nodes: List[str],
        read_sequence: str
    ) -> RoutePrediction:
        """
        Find best route for an ultralong read through the graph.
        
        Args:
            read_id: Read identifier
            graph: Assembly graph
            candidate_nodes: Nodes the read might traverse
            read_sequence: Read sequence for alignment
        
        Returns:
            RoutePrediction with best path and score
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # Generate candidate routes (simplified - would use actual graph traversal)
        candidate_routes = self._generate_candidate_routes(graph, candidate_nodes, read_sequence)
        
        if not candidate_routes:
            return RoutePrediction(
                read_id=read_id,
                path=[],
                route_score=0.0,
                confidence=0.0,
                alternative_routes=[]
            )
        
        # Score each route
        route_scores = []
        for route in candidate_routes:
            features = self._extract_route_features(route, read_sequence)
            score = self.score_route(features)
            route_scores.append((route, score))
        
        # Sort by score
        route_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Best route
        best_route, best_score = route_scores[0]
        alternatives = [r[0] for r in route_scores[1:3]]  # Top 2 alternatives
        
        return RoutePrediction(
            read_id=read_id,
            path=best_route,
            route_score=best_score,
            confidence=best_score,
            alternative_routes=alternatives
        )
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained,
            'version': self.version
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = metadata['is_trained']
        self.version = metadata['version']
        
        logger.info(f"Loaded {self.model_name} from {path}")
    
    def _dict_to_array(self, features) -> np.ndarray:
        """Convert feature dictionary or list to numpy array."""
        if isinstance(features, (list, np.ndarray)):
            return np.array(features)
        
        # Dictionary format: 12D route features
        feature_names = [
            'path_length', 'num_branches', 'coverage_mean', 'coverage_std',
            'sequence_identity', 'mapping_quality', 'num_gaps', 'gap_size_mean',
            'kmer_consistency', 'orientation_consistency', 'ul_span', 'route_complexity'
        ]
        return np.array([features.get(name, 0.0) for name in feature_names])
    
    def _generate_candidate_routes(self, graph: GraphTensors, nodes: List[str], sequence: str) -> List[List[str]]:
        """Generate candidate routes through graph (simplified stub)."""
        # In production, would use actual graph traversal and alignment
        if len(nodes) >= 3:
            return [nodes[:3], nodes[1:4] if len(nodes) >= 4 else nodes[:2]]
        return [nodes] if nodes else []
    
    def _extract_route_features(self, route: List[str], sequence: str) -> Dict[str, float]:
        """Extract features for a route (simplified stub)."""
        return {
            'path_length': len(route),
            'num_branches': 2,
            'coverage_mean': 30.0,
            'coverage_std': 5.0,
            'sequence_identity': 0.95,
            'mapping_quality': 40.0,
            'num_gaps': 1,
            'gap_size_mean': 50.0,
            'kmer_consistency': 0.9,
            'orientation_consistency': 1.0,
            'ul_span': 50000.0,
            'route_complexity': 1.5
        }


# ============================================================================
#                       MODEL REGISTRATION
# ============================================================================

# Import read correction models
from .read_correction_models import XGBoostAdaptiveKmerAI, XGBoostBaseErrorClassifierAI

# Register all models in the registry
ModelRegistry.register('edge_ai', XGBoostEdgeAI)
ModelRegistry.register('path_gnn', SimplePathGNN)
ModelRegistry.register('diploid_ai', XGBoostDiploidAI)
ModelRegistry.register('ul_routing', XGBoostULRoutingAI)
ModelRegistry.register('sv_ai', XGBoostSVAI)
ModelRegistry.register('adaptive_kmer', XGBoostAdaptiveKmerAI)
ModelRegistry.register('base_error_classifier', XGBoostBaseErrorClassifierAI)

logger.info("Registered ML models: edge_ai, path_gnn, diploid_ai, ul_routing, sv_ai, adaptive_kmer, base_error_classifier")



# ============================================================================
#                    PART 3: TRAINING INFRASTRUCTURE
# ============================================================================

class KmerModelTrainer:
    """
    Trains XGBoost models to predict optimal k-mer sizes.
    
    Strategy:
    - Separate model for each stage (DBG, UL overlap, extension, polish)
    - Regression task (predict k value)
    - Cross-validation for robust evaluation
    - Hyperparameter tuning via grid search
    
    Example:
        trainer = KmerModelTrainer(output_dir='models')
        trainer.train_from_csv('training_data/labeled_data.csv')
        trainer.save_models()
    """
    
    def __init__(self,
                 output_dir: Path,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            output_dir: Directory to save trained models
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Trained models (one per stage)
        self.models = {}
        
        # Training history
        self.training_history = {
            'dbg': {},
            'ul_overlap': {},
            'extension': {},
            'polish': {},
        }
    
    def train_from_csv(self, data_path: Path):
        """
        Train all models from labeled CSV data.
        
        Expected CSV columns:
        - feat_* columns (features)
        - optimal_dbg_k (target)
        - optimal_ul_overlap_k (target)
        - optimal_extension_k (target)
        - optimal_polish_k (target)
        """
        logger.info(f"Loading training data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Extract features
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        X = df[feature_cols].values
        
        logger.info(f"Training with {len(X)} samples, {len(feature_cols)} features")
        
        # Train model for each stage
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            target_col = f'optimal_{stage}_k'
            
            if target_col not in df.columns:
                logger.warning(f"No target column {target_col}, skipping {stage}")
                continue
            
            y = df[target_col].values
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {stage} k predictor")
            logger.info(f"{'='*60}")
            
            model = self._train_single_model(X, y, stage)
            self.models[stage] = model
    
    def _train_single_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           stage: str) -> xgb.XGBRegressor:
        """
        Train XGBoost model for a single stage.
        
        Uses grid search for hyperparameter tuning.
        """
        logger.info(f"Target range: [{y.min()}, {y.max()}], mean: {y.mean():.1f}")
        
        # Define hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        # Grid search with cross-validation
        logger.info("Running grid search for optimal hyperparameters...")
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=-1,
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Evaluate with cross-validation
        self._evaluate_model(best_model, X, y, stage)
        
        # Train final model on all data
        logger.info("Training final model on full dataset...")
        best_model.fit(X, y)
        
        # Feature importance
        self._log_feature_importance(best_model, stage)
        
        return best_model
    
    def _evaluate_model(self,
                       model: xgb.XGBRegressor,
                       X: np.ndarray,
                       y: np.ndarray,
                       stage: str):
        """Evaluate model with cross-validation."""
        logger.info(f"\nCross-validation evaluation ({self.cv_folds} folds):")
        
        # MAE (Mean Absolute Error)
        mae_scores = -cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error'
        )
        
        # R² score
        r2_scores = cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='r2'
        )
        
        # RMSE (Root Mean Squared Error)
        mse_scores = -cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(mse_scores)
        
        logger.info(f"  MAE:  {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
        logger.info(f"  RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        logger.info(f"  R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
        
        # Store in history
        self.training_history[stage] = {
            'mae_mean': float(mae_scores.mean()),
            'mae_std': float(mae_scores.std()),
            'rmse_mean': float(rmse_scores.mean()),
            'rmse_std': float(rmse_scores.std()),
            'r2_mean': float(r2_scores.mean()),
            'r2_std': float(r2_scores.std()),
            'cv_folds': self.cv_folds,
        }
        
        # Check if MAE is acceptable (within 5 of optimal)
        if mae_scores.mean() > 5:
            logger.warning(f"⚠️  MAE > 5 - model may need improvement")
        else:
            logger.info(f"✓ MAE < 5 - good prediction accuracy!")
    
    def _log_feature_importance(self, model: xgb.XGBRegressor, stage: str):
        """Log and save feature importance."""
        importance = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        logger.info(f"\nTop 10 most important features for {stage}:")
        for i, idx in enumerate(indices[:10], 1):
            logger.info(f"  {i}. Feature {idx}: {importance[idx]:.4f}")
        
        # Save to history
        self.training_history[stage]['feature_importance'] = {
            int(i): float(imp) for i, imp in enumerate(importance)
        }
    
    def save_models(self):
        """Save trained models and training history."""
        # Save each model
        for stage, model in self.models.items():
            model_path = self.output_dir / f'{stage}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {stage} model to {model_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save metadata
        metadata = {
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'stages': list(self.models.keys()),
        }
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_models(self):
        """Load previously trained models."""
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            model_path = self.output_dir / f'{stage}_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[stage] = pickle.load(f)
                logger.info(f"Loaded {stage} model from {model_path}")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load training history
        history_path = self.output_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict k values for all stages.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Dict mapping stage → predictions
        """
        predictions = {}
        for stage, model in self.models.items():
            predictions[stage] = model.predict(X)
        return predictions
    
    def generate_report(self) -> str:
        """Generate a summary report of training results."""
        report = []
        report.append("=" * 70)
        report.append("ADAPTIVE K-MER PREDICTION: TRAINING REPORT")
        report.append("=" * 70)
        report.append("")
        
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            if stage not in self.training_history:
                continue
            
            hist = self.training_history[stage]
            
            report.append(f"\n{stage.upper()} K PREDICTION")
            report.append("-" * 40)
            report.append(f"MAE:  {hist['mae_mean']:.2f} ± {hist['mae_std']:.2f}")
            report.append(f"RMSE: {hist['rmse_mean']:.2f} ± {hist['rmse_std']:.2f}")
            report.append(f"R²:   {hist['r2_mean']:.3f} ± {hist['r2_std']:.3f}")
            
            # Success criteria
            if hist['mae_mean'] < 5:
                report.append("✓ PASSED: MAE < 5 (within 5 of optimal k)")
            else:
                report.append("⚠️  WARNING: MAE > 5 (may need more training data)")
            
            if hist['r2_mean'] > 0.7:
                report.append("✓ PASSED: R² > 0.7 (good predictive power)")
            else:
                report.append("⚠️  WARNING: R² < 0.7 (weak predictions)")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """Example training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train adaptive k-mer prediction models')
    parser.add_argument('--data', required=True, help='Path to labeled training data CSV')
    parser.add_argument('--output', default='models', help='Output directory for models')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models
    trainer = KmerModelTrainer(
        output_dir=Path(args.output),
        cv_folds=args.cv_folds
    )
    
    trainer.train_from_csv(Path(args.data))
    trainer.save_models()
    
    # Print report
    print("\n" + trainer.generate_report())


if __name__ == '__main__':
    main()
