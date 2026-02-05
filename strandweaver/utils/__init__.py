"""
StrandWeaver v0.1.0

Utilities module for StrandWeaver.

This module provides core utilities for the assembly pipeline:
- Pipeline orchestration and coordination (unified master orchestrator)
- Checkpoint management
- Hardware management (CPU/GPU) - consolidated device and acceleration
- GPU acceleration core functions
"""

# Import order matters to avoid circular imports
# Import hardware_management and checkpoints first (no dependencies)
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
    # Note: GPUPhaser and GPUAnchorMapper don't exist in hardware_management
    # They are implemented as GPUSpectralPhaser and GPUAnchorFinder
)

# Lazy import of pipeline to break circular dependency
# pipeline depends on assembly_core which depends on hardware_management
# which triggers this __init__.py
def __getattr__(name):
    """Lazy import of pipeline module to break circular dependency."""
    if name in ('PipelineOrchestrator', 'AssemblyResult', 'KmerPrediction', 
                'PreprocessingStats', 'PreprocessingResult'):
        from .pipeline import (
            PipelineOrchestrator,
            AssemblyResult,
            KmerPrediction,
            PreprocessingStats,
            PreprocessingResult,
        )
        globals().update({
            'PipelineOrchestrator': PipelineOrchestrator,
            'AssemblyResult': AssemblyResult,
            'KmerPrediction': KmerPrediction,
            'PreprocessingStats': PreprocessingStats,
            'PreprocessingResult': PreprocessingResult,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
]
