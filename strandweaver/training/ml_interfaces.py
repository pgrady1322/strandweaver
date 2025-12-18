"""
ML model interfaces for StrandWeaver training.

This module defines abstract base classes and interfaces for all ML models
used in the StrandWeaver assembly pipeline. These interfaces enable:
- Standardized model APIs
- Easy model swapping and comparison
- Consistent prediction formats
- Model serialization/deserialization
- Training and evaluation workflows

The 7 ML subsystems:
1. EdgeAI - Overlap classification
2. PathGNN - Graph path prediction
3. DiploidAI - Haplotype assignment
4. ULRoutingAI - Ultralong read routing
5. SVAI - Structural variant detection
6. AdaptiveKmerAI - ML-guided k-mer selection for read correction
7. BaseErrorClassifierAI - Per-base error classification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


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

