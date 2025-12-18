#!/usr/bin/env python3
"""
GPU Setup Verification Script

Tests GPU availability and configuration for StrandWeaver training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def check_pytorch():
    """Check PyTorch installation and GPU support."""
    try:
        import torch
        logger.info(f"✅ PyTorch installed: {torch.__version__}")
        
        # Check MPS
        if torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon MPS backend available")
            try:
                # Test MPS with simple operation
                x = torch.randn(100, 100, device='mps')
                y = torch.randn(100, 100, device='mps')
                z = torch.matmul(x, y)
                logger.info("✅ MPS operations working correctly")
                return 'mps'
            except Exception as e:
                logger.warning(f"⚠️ MPS available but test failed: {e}")
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"✅ NVIDIA CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   GPUs: {torch.cuda.device_count()}")
            return 'cuda'
        
        logger.warning("⚠️ No GPU backend available, will use CPU")
        return 'cpu'
        
    except ImportError:
        logger.error("❌ PyTorch not installed: pip install torch")
        return None


def check_cupy():
    """Check CuPy installation (optional for NVIDIA GPUs)."""
    try:
        import cupy as cp
        logger.info(f"✅ CuPy installed: {cp.__version__}")
        
        # Test CuPy
        try:
            device = cp.cuda.Device(0)
            logger.info(f"✅ CuPy GPU accessible: {device.name}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ CuPy installed but GPU not accessible: {e}")
            return False
            
    except ImportError:
        logger.info("ℹ️ CuPy not installed (optional for NVIDIA GPUs)")
        return False


def check_strandweaver_gpu():
    """Check StrandWeaver GPU components."""
    try:
        from strandweaver.utils.gpu_core import GPUKmerCounter, get_gpu_info
        from strandweaver.utils.gpu_core import (
            GPUGraphBuilder,
            GPUKmerExtractor,
            GPUSequenceAligner
        )
        
        logger.info("✅ StrandWeaver GPU modules imported successfully")
        
        # Check GPU availability
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            logger.info(f"✅ GPU detected: {gpu_info['info']}")
        else:
            logger.warning(f"⚠️ GPU unavailable: {gpu_info['info']}")
        
        # Test graph builder initialization
        try:
            builder = GPUGraphBuilder(k=21, use_gpu=True)
            if builder.use_gpu:
                logger.info(f"✅ GPUGraphBuilder initialized with {builder.backend_type.upper()} backend")
            else:
                logger.warning("⚠️ GPUGraphBuilder initialized but using CPU")
        except Exception as e:
            logger.error(f"❌ GPUGraphBuilder failed: {e}")
        
        # Test k-mer extractor initialization
        try:
            extractor = GPUKmerExtractor(k=21, use_gpu=True)
            if extractor.use_gpu:
                logger.info(f"✅ GPUKmerExtractor initialized with {extractor.backend_type.upper()} backend")
            else:
                logger.warning("⚠️ GPUKmerExtractor initialized but using CPU")
        except Exception as e:
            logger.error(f"❌ GPUKmerExtractor failed: {e}")
        
        # Test k-mer counter initialization
        try:
            counter = GPUKmerCounter(k_size=21, use_gpu=True)
            if counter.use_gpu:
                logger.info(f"✅ GPUKmerCounter initialized with {counter.backend_type.upper()} backend")
            else:
                logger.warning("⚠️ GPUKmerCounter initialized but using CPU")
        except Exception as e:
            logger.error(f"❌ GPUKmerCounter failed: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ StrandWeaver GPU modules not available: {e}")
        return False


def check_lightgbm():
    """Check LightGBM installation for k-mer selector."""
    try:
        import lightgbm as lgb
        logger.info(f"✅ LightGBM installed: {lgb.__version__}")
        return True
    except ImportError:
        logger.error("❌ LightGBM not installed: pip install lightgbm")
        return False


def print_recommendations(backend):
    """Print recommendations based on detected backend."""
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    if backend == 'mps':
        logger.info("✅ Apple Silicon MPS detected - optimal configuration")
        logger.info("\nFor training:")
        logger.info("  python scripts/generate_assembly_training_data.py \\")
        logger.info("      --scenario repeat_heavy \\")
        logger.info("      --output-dir training_data/assembly_ai \\")
        logger.info("      --num-workers 8 \\")
        logger.info("      --use-gpu")
        logger.info("\nMemory: Recommended 16+ GB RAM")
        logger.info("Workers: 4-8 for repeat_heavy, 8-16 for balanced")
        
    elif backend == 'cuda':
        logger.info("✅ NVIDIA CUDA detected - optimal configuration")
        logger.info("\nFor training:")
        logger.info("  python scripts/generate_assembly_training_data.py \\")
        logger.info("      --scenario repeat_heavy \\")
        logger.info("      --output-dir training_data/assembly_ai \\")
        logger.info("      --num-workers 16 \\")
        logger.info("      --use-gpu")
        logger.info("\nVRAM: Recommended 8+ GB")
        logger.info("Workers: 8-32 depending on VRAM")
        
    elif backend == 'cpu':
        logger.info("⚠️ CPU-only mode - training will be slower")
        logger.info("\nFor training:")
        logger.info("  python scripts/generate_assembly_training_data.py \\")
        logger.info("      --scenario balanced \\")  # Recommend smaller scenario
        logger.info("      --output-dir training_data/assembly_ai \\")
        logger.info("      --num-workers 4")  # No --use-gpu flag
        logger.info("\nConsider: Use 'simple' or 'balanced' scenarios")
        logger.info("Workers: 2-4 to avoid memory issues")
        
    else:
        logger.error("❌ PyTorch not available - install required")
        logger.info("\nInstall PyTorch:")
        logger.info("  pip install torch torchvision")


def main():
    """Run all GPU verification checks."""
    logger.info("="*60)
    logger.info("StrandWeaver GPU Configuration Check")
    logger.info("="*60 + "\n")
    
    # Check PyTorch
    logger.info("[1/5] Checking PyTorch...")
    backend = check_pytorch()
    print()
    
    # Check CuPy (optional)
    logger.info("[2/5] Checking CuPy (optional)...")
    check_cupy()
    print()
    
    # Check LightGBM
    logger.info("[3/5] Checking LightGBM...")
    check_lightgbm()
    print()
    
    # Check StrandWeaver GPU components
    logger.info("[4/5] Checking StrandWeaver GPU components...")
    check_strandweaver_gpu()
    print()
    
    # Print recommendations
    logger.info("[5/5] Generating recommendations...")
    print_recommendations(backend)
    
    logger.info("\n" + "="*60)
    logger.info("Verification complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
