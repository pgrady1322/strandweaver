"""
Assembly Utilities module for StrandWeaver.

This module provides utility functions and AI models for assembly tasks:
- GNN models for path prediction
- Graph cleanup and validation
- Misassembly detection
- Overlap filtering with AI
- General assembly utilities
"""

from .gnn_models import (
    PathGNNModel,
    SimpleGNN,
    MediumGNN,
    DeepGNN
)

from .gnn_path_predictor import (
    PathGNN,
    GraphTensors,
    FeatureExtractor,
    PathExtractor,
    GNNPathResult
)

from .graph_cleanup import GraphCleaner
from .misassembly_detector import MisassemblyDetector
from .overlap_ai_filter import OverlapFilter
from .utilities import (
    PreprocessingCoordinator,
    AssemblyUtils
)

__all__ = [
    # GNN Models
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
    "PreprocessingCoordinator",
    "AssemblyUtils",
]
