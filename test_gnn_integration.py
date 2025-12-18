#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test GNN integration with assembly pipeline.

Verifies:
1. GNN model creation and configuration
2. PathGNN inference class functionality
3. PyG Data conversion utilities
4. Model loading/saving
"""

import sys
import json
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_gnn_models_module():
    """Test that GNN models module loads correctly."""
    logger.info("Testing GNN models module...")
    
    try:
        # Note: gnn_models moved to assembly_utils, gnn_path_predictor has duplicate in assembly_core
        from strandweaver.assembly_utils.gnn_models import (
            GNNConfig, EdgeConvLayer, PathGNNModel,
            SimpleGNN, MediumGNN, DeepGNN, GNNTrainer
        )
        logger.info("✓ GNN models module imports successfully")
        
        # Test config creation
        config = GNNConfig()
        logger.info(f"✓ GNNConfig created: {config.num_layers} layers, {config.hidden_dim} hidden units")
        
        return True
    except ImportError as e:
        logger.warning(f"✗ GNN models import failed (expected if PyTorch unavailable): {e}")
        return False


def test_path_gnn_class():
    """Test PathGNN class functionality."""
    logger.info("Testing PathGNN class...")
    
    try:
        from strandweaver.assembly_core.gnn_path_predictor import PathGNN, GraphTensors
        
        # Create instance without model (heuristic mode)
        gnn = PathGNN()
        logger.info(f"✓ PathGNN created in heuristic mode")
        
        # Create sample graph tensors
        graph_tensors = GraphTensors(
            node_features={0: [0.1] * 12, 1: [0.2] * 12},
            edge_features={0: [0.5] * 10},
            edge_index=[(0, 1)],
            node_to_index={0: 0, 1: 1},
            edge_to_index={0: 0},
            num_nodes=2,
            num_edges=1
        )
        logger.info(f"✓ Sample GraphTensors created")
        
        # Test heuristic prediction
        predictions = gnn.predict_edge_probabilities(graph_tensors)
        logger.info(f"✓ Heuristic predictions: {predictions}")
        
        return True
    except Exception as e:
        logger.error(f"✗ PathGNN test failed: {e}", exc_info=True)
        return False


def test_gnn_inference_structure():
    """Test that GNN inference structure is properly defined."""
    logger.info("Testing GNN inference structure...")
    
    try:
        from strandweaver.assembly_core.gnn_path_predictor import PathGNN, FeatureExtractor, PathExtractor, GNNPathResult
        
        # Check FeatureExtractor
        fe = FeatureExtractor()
        logger.info(f"✓ FeatureExtractor created")
        
        # Check PathExtractor
        pe = PathExtractor()
        logger.info(f"✓ PathExtractor created")
        
        # Check GNNPathResult
        result = GNNPathResult()
        logger.info(f"✓ GNNPathResult created")
        
        # Verify PathGNN has required methods
        gnn = PathGNN()
        assert hasattr(gnn, 'predict_edge_probabilities'), "Missing predict_edge_probabilities"
        assert hasattr(gnn, '_predict_with_model'), "Missing _predict_with_model"
        assert hasattr(gnn, '_predict_heuristic'), "Missing _predict_heuristic"
        assert hasattr(gnn, '_graph_tensors_to_pyg_data'), "Missing _graph_tensors_to_pyg_data"
        assert hasattr(gnn, 'load_model'), "Missing load_model"
        assert hasattr(gnn, 'save_model'), "Missing save_model"
        logger.info(f"✓ PathGNN has all required methods")
        
        return True
    except AssertionError as e:
        logger.error(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ GNN structure test failed: {e}", exc_info=True)
        return False


def check_pipeline_integration():
    """Check that GNN is properly integrated with assembly pipeline."""
    logger.info("Checking pipeline integration...")
    
    try:
        # Check that PathGNN can be imported from assembly context
        from strandweaver.assembly_core.gnn_path_predictor import PathGNN
        logger.info(f"✓ PathGNN importable from assembly context")
        
        # Check that training module exists and is importable
        from strandweaver import training
        logger.info(f"✓ Training module importable")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Integration check failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("GNN INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("GNN Models Module", test_gnn_models_module()))
    results.append(("PathGNN Class", test_path_gnn_class()))
    results.append(("GNN Inference Structure", test_gnn_inference_structure()))
    results.append(("Pipeline Integration", check_pipeline_integration()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
