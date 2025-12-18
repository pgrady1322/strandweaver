"""
Utilities module for StrandWeaver.

This module provides core utilities for the assembly pipeline:
- Pipeline orchestration and coordination
- Checkpoint management
- Device management (CPU/GPU)
- GPU acceleration core functions
"""

from .pipeline import PipelineOrchestrator
from .pipeline_orchestrator import AssemblyOrchestrator
from .checkpoints import CheckpointManager
from .device import get_optimal_device, DeviceManager
from .gpu_core import (
    GPUKmerCounter,
    GPUGraphBuilder,
    GPUContactMapBuilder,
    GPUPhaser,
    GPUAnchorMapper,
    get_gpu_info,
)

__all__ = [
    # Pipeline
    "PipelineOrchestrator",
    "AssemblyOrchestrator",
    # Checkpoints
    "CheckpointManager",
    # Device management
    "get_optimal_device",
    "DeviceManager",
    # GPU acceleration
    "GPUKmerCounter",
    "GPUGraphBuilder",
    "GPUContactMapBuilder",
    "GPUPhaser",
    "GPUAnchorMapper",
    "get_gpu_info",
]
