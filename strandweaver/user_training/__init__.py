#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

User training subpackage — end-user model retraining.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from .training_config import (
    UserGenomeConfig,
    UserReadConfig,
    UserTrainingConfig,
    GraphTrainingConfig,
    ReadType,
    Ploidy
)

# config_based_workflow depends on the 'training' backend (strandweaver
# dev branch only).  The module itself is now safely importable thanks to guarded
# imports, but we still wrap re-export here for extra resilience.
try:
    from .config_based_workflow import (
        generate_custom_training_data,
        TrainingDataGenerator
    )
except Exception:  # pragma: no cover — only if module itself is broken
    generate_custom_training_data = None  # type: ignore[assignment]
    TrainingDataGenerator = None  # type: ignore[assignment]

# graph_training_data is pure Python (no backend dependency for the core
# data structures).  The generate_graph_training_data entry point requires
# backend SimulatedRead objects at runtime, but the module is importable.
try:
    from .graph_training_data import (
        generate_graph_training_data,
        detect_overlaps,
        build_overlap_graph,
        label_graph,
        SyntheticGraph,
        ReadInfo,
    )
except Exception:  # pragma: no cover
    generate_graph_training_data = None  # type: ignore[assignment]
    detect_overlaps = None  # type: ignore[assignment]
    build_overlap_graph = None  # type: ignore[assignment]
    label_graph = None  # type: ignore[assignment]
    SyntheticGraph = None  # type: ignore[assignment]
    ReadInfo = None  # type: ignore[assignment]

# train_models requires numpy/xgboost/sklearn at *call* time but the
# module itself is importable (all heavy deps are guarded).
try:
    from .train_models import (
        train_all_models,
        ModelTrainingConfig,
        load_trained_model,
    )
except Exception:  # pragma: no cover
    train_all_models = None  # type: ignore[assignment]
    ModelTrainingConfig = None  # type: ignore[assignment]
    load_trained_model = None  # type: ignore[assignment]

__all__ = [
    'UserGenomeConfig',
    'UserReadConfig',
    'UserTrainingConfig',
    'GraphTrainingConfig',
    'ReadType',
    'Ploidy',
    'generate_custom_training_data',
    'TrainingDataGenerator',
    'generate_graph_training_data',
    'detect_overlaps',
    'build_overlap_graph',
    'label_graph',
    'SyntheticGraph',
    'ReadInfo',
    'train_all_models',
    'ModelTrainingConfig',
    'load_trained_model',
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
