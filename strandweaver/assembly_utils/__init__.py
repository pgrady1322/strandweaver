"""
StrandWeaver v0.1.0

Assembly Utilities module for StrandWeaver.

This module provides utility functions for assembly tasks:
- Graph cleanup and validation
- Misassembly detection
- Overlap filtering with AI
- General assembly utilities

Note: GNN models now integrated into pathweaver_module in assembly_core.
Note: PreprocessingCoordinator now integrated into PipelineOrchestrator in utils.
"""

# GNN models consolidated into pathweaver_module
# from ..assembly_core.pathweaver_module import (
#     PathGNNModel, SimpleGNN, MediumGNN, DeepGNN, GNNConfig,
#     PathGNN, GraphTensors, FeatureExtractor, PathExtractor, GNNPathResult
# )

from .graph_cleanup import GraphCleaner
from .misassembly_detector import MisassemblyDetector
# overlap_ai_filter archived - use EdgeWarden from assembly_core instead

__all__ = [
    # GNN Models (now in pathweaver_module)
    "PathGNNModel",
    "SimpleGNN",
    "MediumGNN",
    "DeepGNN",
    # GNN Path Predictor
    "PathGNN",
    "GraphTensors",
    "FeatureExtractor",
    "PathExtractor",
    "GNNPathResult",
    # Utilities
    "GraphCleaner",
    "MisassemblyDetector",
    "OverlapFilter",
]
