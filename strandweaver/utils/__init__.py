"""
Utilities module for StrandWeaver.

This module provides core utilities for the assembly pipeline:
- Pipeline orchestration and coordination (unified master orchestrator)
- Checkpoint management
- Hardware management (CPU/GPU) - consolidated device and acceleration
- GPU acceleration core functions
"""

from .pipeline import (
    PipelineOrchestrator,
    AssemblyResult,
    KmerPrediction,
    PreprocessingStats,
    PreprocessingResult,
)
from .checkpoints import CheckpointManager
from .hardware_management import (
    # GPU Backend (explicit selection for HPC)
    GPUBackend,
    set_gpu_backend,
    get_gpu_backend,
    get_gpu_info,
    # PyTorch Device Detection (auto-detection for ML/local use)
    get_optimal_device,
    get_device_info,
    print_device_info,
    setup_pytorch_device,
    optimize_for_device,
    get_batch_size_for_device,
    DeviceManager,
    # GPU Acceleration Classes
    GPUKmerCounter,
    GPUGraphBuilder,
    GPUContactMapBuilder,
    GPUPhaser,
    GPUAnchorMapper,
)

__all__ = [
    # Pipeline (unified master orchestrator)
    "PipelineOrchestrator",
    "AssemblyResult",
    "KmerPrediction",
    "PreprocessingStats",
    "PreprocessingResult",
    # Checkpoints
    "CheckpointManager",
    # GPU Backend Management (explicit for HPC)
    "GPUBackend",
    "set_gpu_backend",
    "get_gpu_backend",
    "get_gpu_info",
    # PyTorch Device Detection (auto-detect for ML/local)
    "get_optimal_device",
    "get_device_info",
    "print_device_info",
    "setup_pytorch_device",
    "optimize_for_device",
    "get_batch_size_for_device",
    "DeviceManager",
    # GPU Acceleration
    "GPUKmerCounter",
    "GPUGraphBuilder",
    "GPUContactMapBuilder",
    "GPUPhaser",
    "GPUAnchorMapper",
]
