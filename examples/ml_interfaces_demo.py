"""
ML interfaces demonstration.

This script shows how to implement and use the ML model interfaces
defined in ml_interfaces.py. It includes:
- Simple reference implementations
- Model registration
- Training and prediction workflows
- Evaluation examples
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.ml_interfaces import (
    EdgeAIModel,
    PathGNNModel,
    DiploidAIModel,
    ULRoutingAIModel,
    SVAIModel,
    EdgePrediction,
    HaplotypePrediction,
    SVPrediction,
    ModelRegistry,
    create_model,
    TrainingConfig,
    ModelTrainer,
    ModelEvaluator,
    GraphTensors
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================================================
#                   SIMPLE REFERENCE IMPLEMENTATIONS
# ============================================================================

class SimpleEdgeAI(EdgeAIModel):
    """
    Simple rule-based EdgeAI implementation for demonstration.
    
    Uses basic heuristics instead of actual ML.
    """
    
    def __init__(self):
        super().__init__(model_name="SimpleEdgeAI", version="v0.1")
        self.is_trained = True  # No training needed for rule-based
    
    def predict_single_edge(self, features: dict) -> EdgePrediction:
        """Classify edge using simple rules."""
        # Extract key features
        identity = features.get('identity', 0.0)
        position_dist = features.get('position_distance', 0)
        is_repeat = features.get('is_repeat_region', 0)
        
        # Simple classification rules
        if is_repeat > 0.5:
            label = 'repeat'
            confidence = 0.7
        elif identity > 0.95 and position_dist < 1000:
            label = 'true'
            confidence = 0.9
        elif identity > 0.90 and position_dist < 10000:
            label = 'allelic'
            confidence = 0.7
        elif position_dist > 100000:
            label = 'chimeric'
            confidence = 0.8
        else:
            label = 'unknown'
            confidence = 0.5
        
        # Mock probabilities
        probs = {
            'true': 0.8 if label == 'true' else 0.1,
            'allelic': 0.7 if label == 'allelic' else 0.1,
            'repeat': 0.7 if label == 'repeat' else 0.1,
            'sv_break': 0.1,
            'chimeric': 0.8 if label == 'chimeric' else 0.1,
            'unknown': 0.5 if label == 'unknown' else 0.1
        }
        
        return EdgePrediction(
            edge_id=("source", "target"),
            label=label,
            confidence=confidence,
            class_probabilities=probs
        )
    
    def predict_proba(self, features: dict) -> dict:
        """Get probability distribution."""
        pred = self.predict_single_edge(features)
        return pred.class_probabilities
    
    def train(self, train_data, val_data=None, **kwargs):
        """No training needed for rule-based model."""
        return {'status': 'Rule-based model - no training needed'}
    
    def save(self, path: Path):
        """No model to save."""
        logger.info(f"Rule-based model - nothing to save")
    
    def load(self, path: Path):
        """No model to load."""
        logger.info(f"Rule-based model - nothing to load")


class SimpleDiploidAI(DiploidAIModel):
    """
    Simple haplotype assignment implementation.
    
    Uses coverage and Hi-C ratios for basic assignment.
    """
    
    def __init__(self):
        super().__init__(model_name="SimpleDiploidAI", version="v0.1")
        self.is_trained = True
    
    def predict_single_node(self, features: dict) -> HaplotypePrediction:
        """Assign haplotype using simple rules."""
        # Extract key features (assuming dict format)
        feature_array = features.get('features', [0] * 42)
        
        # Extract Hi-C ratio (index 34 in 42D vector)
        hic_ratio = feature_array[34] if len(feature_array) > 34 else 0.5
        
        # Simple assignment based on Hi-C ratio
        if hic_ratio > 0.7:
            haplotype = 'A'
            confidence = 0.8
        elif hic_ratio < 0.3:
            haplotype = 'B'
            confidence = 0.8
        elif abs(hic_ratio - 0.5) < 0.1:
            haplotype = 'BOTH'
            confidence = 0.6
        else:
            haplotype = 'UNKNOWN'
            confidence = 0.4
        
        scores = {
            'A': hic_ratio,
            'B': 1.0 - hic_ratio,
            'BOTH': 1.0 - abs(hic_ratio - 0.5) * 2,
            'REPEAT': 0.1,
            'UNKNOWN': 0.3
        }
        
        return HaplotypePrediction(
            node_id=features.get('node_id', 'unknown'),
            haplotype=haplotype,
            confidence=confidence,
            haplotype_scores=scores
        )
    
    def predict_proba(self, features: dict) -> dict:
        """Get probability distribution."""
        pred = self.predict_single_node(features)
        return pred.haplotype_scores
    
    def disentangle_graph(self, graph, node_features):
        """Partition graph into haplotypes."""
        hap_A = []
        hap_B = []
        
        for node_feat in node_features:
            pred = self.predict_single_node(node_feat)
            if pred.haplotype == 'A':
                hap_A.append(node_feat.get('node_id', ''))
            elif pred.haplotype == 'B':
                hap_B.append(node_feat.get('node_id', ''))
        
        return hap_A, hap_B
    
    def train(self, train_data, val_data=None, **kwargs):
        """No training needed."""
        return {'status': 'Rule-based model - no training needed'}
    
    def save(self, path: Path):
        """No model to save."""
        pass
    
    def load(self, path: Path):
        """No model to load."""
        pass


class SimpleSVAI(SVAIModel):
    """
    Simple SV detection implementation.
    
    Uses coverage patterns for basic SV detection.
    """
    
    def __init__(self):
        super().__init__(model_name="SimpleSVAI", version="v0.1")
        self.is_trained = True
    
    def predict_single_region(self, features: dict) -> SVPrediction:
        """Detect SV using simple rules."""
        # Extract key features
        feature_array = features.get('features', [0] * 14)
        
        # Coverage features (indices 0-3)
        cov_drop = feature_array[2] if len(feature_array) > 2 else 0.0
        cov_spike = feature_array[3] if len(feature_array) > 3 else 0.0
        
        # Determine SV type
        if cov_drop > 0.7:
            sv_type = 'deletion'
            confidence = 0.8
        elif cov_spike > 0.7:
            sv_type = 'insertion'
            confidence = 0.7
        elif feature_array[6] > 0.7:  # graph_complexity
            sv_type = 'inversion'
            confidence = 0.6
        else:
            sv_type = 'none'
            confidence = 0.5
        
        return SVPrediction(
            region_id=features.get('region_id', 'unknown'),
            chrom=features.get('chrom', 'chr1'),
            start=features.get('start', 0),
            end=features.get('end', 1000),
            sv_type=sv_type,
            sv_size=features.get('sv_size', 0),
            confidence=confidence,
            evidence={'coverage_drop': cov_drop, 'coverage_spike': cov_spike}
        )
    
    def predict_sv_type(self, features: dict) -> dict:
        """Get SV type probabilities."""
        pred = self.predict_single_region(features)
        return {
            'deletion': 0.8 if pred.sv_type == 'deletion' else 0.1,
            'insertion': 0.7 if pred.sv_type == 'insertion' else 0.1,
            'inversion': 0.6 if pred.sv_type == 'inversion' else 0.1,
            'duplication': 0.1,
            'translocation': 0.1,
            'none': 0.5 if pred.sv_type == 'none' else 0.2
        }
    
    def scan_genome(self, graph, coverage, window_size=1000):
        """Scan genome for SVs."""
        svs = []
        # Mock scanning
        for i in range(3):
            sv = SVPrediction(
                region_id=f"sv_{i}",
                chrom="chr1",
                start=i * window_size,
                end=(i + 1) * window_size,
                sv_type='deletion' if i % 2 == 0 else 'insertion',
                sv_size=500,
                confidence=0.7
            )
            svs.append(sv)
        return svs
    
    def train(self, train_data, val_data=None, **kwargs):
        """No training needed."""
        return {'status': 'Rule-based model - no training needed'}
    
    def save(self, path: Path):
        """No model to save."""
        pass
    
    def load(self, path: Path):
        """No model to load."""
        pass


# ============================================================================
#                           DEMONSTRATION
# ============================================================================

def main():
    """Run ML interfaces demonstration."""
    
    print("=" * 80)
    print("ML INTERFACES DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Step 1: Register model implementations
    print("STEP 1: Registering model implementations...")
    print("-" * 80)
    
    ModelRegistry.register('edge_ai', SimpleEdgeAI)
    ModelRegistry.register('diploid_ai', SimpleDiploidAI)
    ModelRegistry.register('sv_ai', SimpleSVAI)
    
    print(f"✅ Registered models: {ModelRegistry.list_models()}")
    print()
    
    # Step 2: Create models using factory
    print("STEP 2: Creating models...")
    print("-" * 80)
    
    edge_model = create_model('edge_ai')
    diploid_model = create_model('diploid_ai')
    sv_model = create_model('sv_ai')
    
    print(f"✅ Created 3 models:")
    print(f"   - {edge_model.model_name} {edge_model.version}")
    print(f"   - {diploid_model.model_name} {diploid_model.version}")
    print(f"   - {sv_model.model_name} {sv_model.version}")
    print()
    
    # Step 3: EdgeAI prediction
    print("STEP 3: EdgeAI - Overlap classification...")
    print("-" * 80)
    
    # Mock edge features
    edge_features = {
        'identity': 0.96,
        'position_distance': 500,
        'is_repeat_region': 0,
        'coverage_ratio': 0.95,
        'gc_similarity': 0.99
    }
    
    edge_pred = edge_model.predict_single_edge(edge_features)
    print(f"Edge classification:")
    print(f"  Label: {edge_pred.label}")
    print(f"  Confidence: {edge_pred.confidence:.2f}")
    print(f"  Class probabilities:")
    for cls, prob in sorted(edge_pred.class_probabilities.items(), key=lambda x: -x[1])[:3]:
        print(f"    {cls}: {prob:.3f}")
    print()
    
    # Step 4: DiploidAI prediction
    print("STEP 4: DiploidAI - Haplotype assignment...")
    print("-" * 80)
    
    # Mock node features (42D vector)
    node_features = {
        'node_id': 'node_1',
        'features': np.random.randn(42).tolist()
    }
    node_features['features'][34] = 0.8  # Set Hi-C ratio to favor haplotype A
    
    hap_pred = diploid_model.predict_single_node(node_features)
    print(f"Haplotype assignment:")
    print(f"  Node: {hap_pred.node_id}")
    print(f"  Haplotype: {hap_pred.haplotype}")
    print(f"  Confidence: {hap_pred.confidence:.2f}")
    print(f"  Haplotype scores:")
    for hap, score in sorted(hap_pred.haplotype_scores.items(), key=lambda x: -x[1])[:3]:
        print(f"    {hap}: {score:.3f}")
    print()
    
    # Step 5: SVAI prediction
    print("STEP 5: SVAI - Structural variant detection...")
    print("-" * 80)
    
    # Mock SV features (14D vector)
    sv_features = {
        'region_id': 'region_chr1_1000-2000',
        'chrom': 'chr1',
        'start': 1000,
        'end': 2000,
        'features': [30.0, 5.0, 0.8, 0.1] + [0.0] * 10,  # High coverage drop
        'sv_size': 500
    }
    
    sv_pred = sv_model.predict_single_region(sv_features)
    print(f"SV detection:")
    print(f"  Region: {sv_pred.region_id}")
    print(f"  Location: {sv_pred.chrom}:{sv_pred.start}-{sv_pred.end}")
    print(f"  SV type: {sv_pred.sv_type}")
    print(f"  SV size: {sv_pred.sv_size} bp")
    print(f"  Confidence: {sv_pred.confidence:.2f}")
    print(f"  Evidence: {sv_pred.evidence}")
    print()
    
    # Step 6: Batch predictions
    print("STEP 6: Batch predictions...")
    print("-" * 80)
    
    # Batch edge predictions
    edges = [
        {'identity': 0.96, 'position_distance': 500, 'is_repeat_region': 0},
        {'identity': 0.85, 'position_distance': 150000, 'is_repeat_region': 0},
        {'identity': 0.92, 'position_distance': 2000, 'is_repeat_region': 1}
    ]
    
    edge_preds = edge_model.predict(edges)
    print(f"Classified {len(edge_preds)} edges:")
    for i, pred in enumerate(edge_preds):
        print(f"  Edge {i+1}: {pred.label} (conf: {pred.confidence:.2f})")
    print()
    
    # Step 7: Graph disentanglement
    print("STEP 7: Graph disentanglement...")
    print("-" * 80)
    
    # Mock graph with nodes
    mock_graph = GraphTensors(
        node_features=np.random.randn(10, 32),
        edge_index=np.array([[0, 1, 2], [1, 2, 3]]),
        edge_features=np.random.randn(3, 16),
        num_nodes=10,
        num_edges=3
    )
    
    # Create node features with varying Hi-C ratios
    mock_nodes = []
    for i in range(10):
        feat = np.random.randn(42).tolist()
        feat[34] = 0.8 if i < 5 else 0.2  # First 5 to hap A, rest to hap B
        mock_nodes.append({'node_id': f'node_{i}', 'features': feat})
    
    hap_A, hap_B = diploid_model.disentangle_graph(mock_graph, mock_nodes)
    print(f"Graph partitioned into haplotypes:")
    print(f"  Haplotype A: {len(hap_A)} nodes - {hap_A[:3]}...")
    print(f"  Haplotype B: {len(hap_B)} nodes - {hap_B[:3]}...")
    print()
    
    # Step 8: Genome scanning
    print("STEP 8: Genome-wide SV scanning...")
    print("-" * 80)
    
    svs = sv_model.scan_genome(mock_graph, {}, window_size=1000)
    print(f"Detected {len(svs)} structural variants:")
    for sv in svs:
        print(f"  {sv.region_id}: {sv.sv_type} at {sv.chrom}:{sv.start}-{sv.end} (conf: {sv.confidence:.2f})")
    print()
    
    # Step 9: Model evaluation
    print("STEP 9: Model evaluation...")
    print("-" * 80)
    
    # Mock predictions and ground truth
    predictions = ['true', 'allelic', 'repeat', 'true', 'chimeric']
    ground_truth = ['true', 'true', 'repeat', 'true', 'chimeric']
    
    metrics = ModelEvaluator.evaluate_classification(predictions, ground_truth)
    print(f"Classification metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Correct: {metrics['num_correct']}/{metrics['num_samples']}")
    print()
    
    # Step 10: Training configuration
    print("STEP 10: Training configuration...")
    print("-" * 80)
    
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        early_stopping_patience=10,
        device='cpu'
    )
    
    print(f"Training configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    print()
    
    print("=" * 80)
    print("ML INTERFACES DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Demonstrated 3 model interfaces (EdgeAI, DiploidAI, SVAI)")
    print("  ✅ Showed model registration and factory patterns")
    print("  ✅ Performed single and batch predictions")
    print("  ✅ Demonstrated graph disentanglement")
    print("  ✅ Showed SV scanning capabilities")
    print("  ✅ Evaluated model performance")
    print()
    print("Next steps:")
    print("  1. Implement actual ML models (PyTorch, scikit-learn, etc.)")
    print("  2. Train models on generated synthetic datasets")
    print("  3. Evaluate on held-out test sets")
    print("  4. Deploy trained models to production pipeline")
    print("  5. Create training corpus orchestrator")


if __name__ == "__main__":
    main()
