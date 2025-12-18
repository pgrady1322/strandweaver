"""
ML model implementations for StrandWeaver.

This module contains concrete implementations of all 7 ML models:
1. EdgeAI - XGBoost classifier for overlap classification
2. PathGNN - Graph neural network for path prediction
3. DiploidAI - XGBoost classifier for haplotype assignment
4. ULRoutingAI - Regression model for ultralong read routing
5. SVAI - XGBoost classifier for structural variant detection
6. AdaptiveKmerAI - XGBoost classifier for adaptive k-mer selection
7. BaseErrorClassifierAI - XGBoost classifier for per-base error classification

Each model uses the interfaces defined in ml_interfaces.py.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json

# ML libraries
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    logging.warning("scikit-learn not installed. Install with: pip install scikit-learn")

# Import interfaces
from .ml_interfaces import (
    EdgeAIModel, EdgePrediction,
    PathGNNModel, PathPrediction, GraphTensors,
    DiploidAIModel, HaplotypePrediction,
    ULRoutingAIModel, RoutePrediction,
    SVAIModel, SVPrediction,
    AdaptiveKmerAIModel, KmerPrediction, ReadContext,
    BaseErrorClassifierAIModel, BaseErrorPrediction, BaseContext,
    TrainingConfig, TrainingMetrics,
    ModelRegistry
)

logger = logging.getLogger(__name__)


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
