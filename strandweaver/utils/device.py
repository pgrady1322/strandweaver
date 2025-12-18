"""
Hardware device detection and management for StrandWeaver.

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
"""

import logging
from typing import Optional, Tuple
import platform

logger = logging.getLogger(__name__)


def get_optimal_device(
    prefer_gpu: bool = False, 
    gpu_device: int = 0,
    backend: Optional[str] = None
) -> Tuple[str, str]:
    """
    Detect and return the optimal compute device.
    
    Priority (when backend='auto' or None):
    1. MPS (Apple Silicon GPU) if available and requested
    2. CUDA (NVIDIA GPU) if available and requested
    3. CPU (fallback)
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        gpu_device: GPU device ID (for multi-GPU systems)
        backend: Specific backend to use ('auto', 'mps', 'cuda', 'cpu', or None)
                 If specified, forces that backend (falls back to CPU if unavailable)
    
    Returns:
        Tuple of (device_type, device_string)
        - device_type: 'mps', 'cuda', or 'cpu'
        - device_string: PyTorch device string (e.g., 'cuda:0', 'mps', 'cpu')
    """
    # Force CPU if requested
    if backend == 'cpu' or (not prefer_gpu and backend is None):
        logger.info("CPU mode (default)")
        return ('cpu', 'cpu')
    
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, falling back to CPU")
        return ('cpu', 'cpu')
    
    # Force MPS if requested
    if backend == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info(f"Apple Silicon GPU (MPS backend) - user requested")
            logger.info(f"  Platform: {platform.processor()}")
            logger.info(f"  Device: mps")
            return ('mps', 'mps')
        else:
            logger.warning("MPS backend requested but not available, falling back to CPU")
            return ('cpu', 'cpu')
    
    # Force CUDA if requested
    if backend == 'cuda':
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(gpu_device)
            device_str = f'cuda:{gpu_device}'
            logger.info(f"NVIDIA GPU (CUDA backend) - user requested")
            logger.info(f"  Device {gpu_device}: {device_name}")
            logger.info(f"  Device string: {device_str}")
            return ('cuda', device_str)
        else:
            logger.warning("CUDA backend requested but not available, falling back to CPU")
            return ('cpu', 'cpu')
    
    # Auto-detect (backend='auto' or None with prefer_gpu=True)
    # Check for Apple Silicon MPS (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info(f"Apple Silicon GPU detected (MPS backend)")
        logger.info(f"  Platform: {platform.processor()}")
        logger.info(f"  Device: mps")
        return ('mps', 'mps')
    
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(gpu_device)
        device_str = f'cuda:{gpu_device}'
        logger.info(f"NVIDIA GPU detected (CUDA backend)")
        logger.info(f"  Device {gpu_device}: {device_name}")
        logger.info(f"  Device string: {device_str}")
        return ('cuda', device_str)
    
    # Fallback to CPU
    logger.warning("GPU requested but no GPU backend available (CUDA/MPS)")
    logger.info("Falling back to CPU")
    return ('cpu', 'cpu')


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_available': False,
        'pytorch_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'mps_available': False,
        'mps_built': False,
        'recommended_device': 'cpu',
    }
    
    try:
        import torch
        info['pytorch_available'] = True
        info['pytorch_version'] = torch.__version__
        
        # CUDA info
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_devices'] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
            info['recommended_device'] = 'cuda'
        
        # MPS info (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            info['mps_built'] = torch.backends.mps.is_built()
            info['mps_available'] = torch.backends.mps.is_available()
            if info['mps_available']:
                info['recommended_device'] = 'mps'
    
    except ImportError:
        pass
    
    return info


def print_device_info():
    """Print detailed device information to console."""
    info = get_device_info()
    
    print("="*60)
    print("Hardware Device Information")
    print("="*60)
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Python: {info['python_version']}")
    print()
    
    if info['pytorch_available']:
        print(f"PyTorch: {info['pytorch_version']}")
        print()
        
        # CUDA
        if info['cuda_available']:
            print("✓ NVIDIA CUDA Available")
            print(f"  CUDA Version: {info['cuda_version']}")
            print(f"  Device Count: {info['cuda_device_count']}")
            for i, device in enumerate(info['cuda_devices']):
                print(f"    Device {i}: {device}")
        else:
            print("✗ NVIDIA CUDA Not Available")
        print()
        
        # MPS
        if info['mps_built']:
            if info['mps_available']:
                print("✓ Apple Silicon MPS Available")
                print("  Backend: Metal Performance Shaders")
                print(f"  Device: {info['processor']}")
            else:
                print("⚠ Apple Silicon MPS Built but Not Available")
        else:
            print("✗ Apple Silicon MPS Not Available")
        print()
        
        print(f"Recommended Device: {info['recommended_device'].upper()}")
    else:
        print("✗ PyTorch Not Installed")
        print("  Install with: pip install torch>=2.0.0")
    
    print("="*60)


def setup_pytorch_device(config: dict) -> Tuple[str, str]:
    """
    Setup PyTorch device based on configuration.
    
    Args:
        config: Pipeline configuration dictionary
    
    Returns:
        Tuple of (device_type, device_string)
    """
    use_gpu = config.get('hardware', {}).get('use_gpu', False)
    gpu_device = config.get('hardware', {}).get('gpu_device', 0)
    
    device_type, device_str = get_optimal_device(prefer_gpu=use_gpu, gpu_device=gpu_device)
    
    return device_type, device_str


def optimize_for_device(model, device_str: str):
    """
    Optimize PyTorch model for specific device.
    
    Args:
        model: PyTorch model
        device_str: Device string ('cuda:0', 'mps', 'cpu')
    
    Returns:
        Model moved to device
    """
    import torch
    
    model = model.to(device_str)
    
    # Device-specific optimizations
    if device_str.startswith('cuda'):
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimizations enabled (cuDNN benchmark)")
    
    elif device_str == 'mps':
        # MPS optimizations
        logger.info("MPS (Apple Silicon) acceleration enabled")
        # Note: MPS doesn't have equivalent of cudnn.benchmark yet
    
    else:
        # CPU optimizations
        torch.set_num_threads(config.get('hardware', {}).get('threads', 1) if 'config' in globals() else 1)
        logger.info("CPU mode with optimized threading")
    
    return model


def get_batch_size_for_device(device_type: str, base_batch_size: int = 32) -> int:
    """
    Get recommended batch size for device.
    
    Args:
        device_type: 'cuda', 'mps', or 'cpu'
        base_batch_size: Base batch size for GPU
    
    Returns:
        Recommended batch size
    """
    if device_type == 'cuda':
        # CUDA: Use base batch size (can be large)
        return base_batch_size
    
    elif device_type == 'mps':
        # MPS: Slightly smaller batch size (unified memory)
        return max(16, base_batch_size // 2)
    
    else:
        # CPU: Much smaller batch size
        return max(4, base_batch_size // 8)


if __name__ == '__main__':
    # Run diagnostics
    print_device_info()
    
    print("\nDevice Selection Tests:")
    print("-" * 60)
    
    # Test CPU mode
    device_type, device_str = get_optimal_device(prefer_gpu=False)
    print(f"CPU mode: {device_type} ({device_str})")
    
    # Test GPU mode
    device_type, device_str = get_optimal_device(prefer_gpu=True)
    print(f"GPU mode: {device_type} ({device_str})")
    
    print("-" * 60)
