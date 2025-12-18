#!/usr/bin/env python3
"""
Training Data Preparation and Validation

Checks that your training setup is complete and correct before running training.
Provides diagnostics and fixes for common issues.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def check_environment() -> bool:
    """Check Python environment and required packages."""
    logger.info("\n" + "="*70)
    logger.info("CHECKING ENVIRONMENT")
    logger.info("="*70)
    
    all_good = True
    
    # Python version
    import sys
    logger.info(f"✓ Python {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        logger.error("✗ Python 3.8+ required")
        all_good = False
    
    # PyTorch
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        logger.info(f"  GPU available: {cuda_available}")
        if cuda_available:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("✗ PyTorch not installed: pip install torch")
        all_good = False
    
    # NumPy
    try:
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__}")
    except ImportError:
        logger.error("✗ NumPy not installed: pip install numpy")
        all_good = False
    
    # scikit-learn
    try:
        import sklearn
        logger.info(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        logger.error("✗ scikit-learn not installed: pip install scikit-learn")
        all_good = False
    
    # transformers
    try:
        import transformers
        logger.info(f"✓ transformers {transformers.__version__}")
    except ImportError:
        logger.error("✗ transformers not installed: pip install transformers")
        all_good = False
    
    return all_good


def check_baseline_model(baseline_path: Path) -> bool:
    """Check if baseline model exists and is loadable."""
    logger.info("\n" + "="*70)
    logger.info("CHECKING BASELINE MODEL")
    logger.info("="*70)
    
    if not baseline_path.exists():
        logger.error(f"✗ Model not found: {baseline_path}")
        logger.info("  Expected at: models/error_predictor_v2.pt")
        logger.info("  To train baseline: python3 scripts/train_models/train_base_error_predictor_v2.py")
        return False
    
    logger.info(f"✓ Model file exists: {baseline_path}")
    logger.info(f"  Size: {baseline_path.stat().st_size / 1e6:.1f} MB")
    
    # Try to load
    try:
        import torch
        checkpoint = torch.load(baseline_path, map_location='cpu')
        logger.info(f"✓ Model file is readable")
        
        if 'model_state_dict' in checkpoint:
            logger.info(f"✓ Contains model_state_dict")
            num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() if isinstance(p, torch.Tensor))
            logger.info(f"  Parameters: {num_params:,}")
        
        if 'accuracy' in checkpoint:
            logger.info(f"  Training accuracy: {checkpoint['accuracy']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        return False


def check_training_data(data_dir: Path) -> Dict:
    """Check training data directory and contents."""
    logger.info("\n" + "="*70)
    logger.info("CHECKING TRAINING DATA")
    logger.info("="*70)
    
    result = {
        'exists': False,
        'batch_files': 0,
        'total_examples': 0,
        'technologies': set(),
        'example_batches': {},
    }
    
    if not data_dir.exists():
        logger.error(f"✗ Data directory not found: {data_dir}")
        logger.info("  Training data should be in:")
        logger.info("    training_data/read_correction_v2/base_error/")
        logger.info("  Files should be: base_error_*_batch_*.pkl")
        return result
    
    logger.info(f"✓ Data directory exists: {data_dir}")
    result['exists'] = True
    
    # Find batch files
    batch_files = sorted(data_dir.glob('base_error_*_batch_*.pkl'))
    result['batch_files'] = len(batch_files)
    
    if not batch_files:
        logger.error(f"✗ No batch files found in {data_dir}")
        logger.error("  Expected files matching: base_error_*_batch_*.pkl")
        return result
    
    logger.info(f"✓ Found {len(batch_files)} batch files")
    
    # Sample first few batches to understand structure
    for i, pkl_file in enumerate(batch_files[:3]):
        try:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                batch_size = len(batch_data)
                result['total_examples'] += batch_size
                
                # Sample first item to understand structure
                if batch_size > 0:
                    sample = batch_data[0]
                    logger.info(f"\n  Batch {i+1}: {pkl_file.name}")
                    logger.info(f"    Size: {batch_size} examples")
                    
                    if isinstance(sample, tuple) and len(sample) == 2:
                        base_ctx, label = sample
                        logger.info(f"    Format: (BaseContext, label)")
                        
                        # Check for technology field
                        if hasattr(base_ctx, 'technology'):
                            tech = base_ctx.technology
                            result['technologies'].add(tech)
                            logger.info(f"    Technology: {tech}")
                        
                        # Store example
                        result['example_batches'][pkl_file.name] = {
                            'size': batch_size,
                            'label_sample': label,
                        }
        except Exception as e:
            logger.error(f"  ✗ Error loading {pkl_file.name}: {e}")
    
    # Calculate total for all batches
    result['total_examples'] = len(batch_files) * (result['total_examples'] // max(1, min(3, len(batch_files))))
    
    logger.info(f"\n  Total examples: ~{result['total_examples']:,}")
    logger.info(f"  Technologies found: {', '.join(sorted(result['technologies']))}")
    
    if result['total_examples'] < 1000:
        logger.warning(f"⚠ Only {result['total_examples']:,} examples (recommend 10k+)")
    
    return result


def check_output_directories(output_dirs: Dict) -> bool:
    """Check and create output directories."""
    logger.info("\n" + "="*70)
    logger.info("CHECKING OUTPUT DIRECTORIES")
    logger.info("="*70)
    
    all_good = True
    
    for name, path in output_dirs.items():
        path_obj = Path(path)
        
        if path_obj.exists():
            logger.info(f"✓ {name}: {path}")
        else:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ {name}: Created {path}")
            except Exception as e:
                logger.error(f"✗ {name}: Cannot create {path} - {e}")
                all_good = False
    
    return all_good


def create_training_summary(
    baseline_path: Path,
    data_dir: Path,
    output_dirs: Dict,
    is_ready: bool,
) -> Path:
    """Create a summary report."""
    summary = {
        'status': 'READY' if is_ready else 'INCOMPLETE',
        'baseline_model': {
            'path': str(baseline_path),
            'exists': baseline_path.exists(),
        },
        'training_data': {
            'path': str(data_dir),
            'exists': data_dir.exists(),
            'batch_files': len(list(data_dir.glob('base_error_*_batch_*.pkl'))) if data_dir.exists() else 0,
        },
        'output_directories': output_dirs,
    }
    
    summary_path = Path(__file__).parent / 'training_readiness_report.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


def main():
    logger.info("\n╔════════════════════════════════════════════════════════════════════════╗")
    logger.info("║          TRAINING DATA PREPARATION AND VALIDATION                      ║")
    logger.info("╚════════════════════════════════════════════════════════════════════════╝")
    
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    baseline_path = repo_root / 'models' / 'error_predictor_v2.pt'
    data_dir = repo_root / 'training_data' / 'read_correction_v2' / 'base_error'
    
    output_dirs = {
        'tech_specific_heads': str(repo_root / 'models' / 'tech_specific_heads'),
        'student_models': str(repo_root / 'models' / 'student_models'),
    }
    
    # Run checks
    checks = {
        'environment': check_environment(),
        'baseline': check_baseline_model(baseline_path),
        'training_data': check_training_data(data_dir)['exists'],
        'output_dirs': check_output_directories(output_dirs),
    }
    
    is_ready = all(checks.values())
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("READINESS SUMMARY")
    logger.info("="*70)
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    summary_path = create_training_summary(baseline_path, data_dir, output_dirs, is_ready)
    logger.info(f"\n✓ Report saved: {summary_path}")
    
    if is_ready:
        logger.info("\n" + "="*70)
        logger.info("✅ READY TO TRAIN!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("  1. Per-tech heads:")
        logger.info("     python3 scripts/train_models/train_tech_specific_heads.py --device cuda")
        logger.info("\n  2. Knowledge distillation (MLP):")
        logger.info("     python3 scripts/train_models/train_knowledge_distillation.py --student-type mlp --device cuda")
        logger.info("\n  3. Knowledge distillation (CNN):")
        logger.info("     python3 scripts/train_models/train_knowledge_distillation.py --student-type cnn --device cuda")
        return 0
    else:
        logger.info("\n" + "="*70)
        logger.error("❌ ISSUES DETECTED - FIX BEFORE TRAINING")
        logger.info("="*70)
        logger.info("\nSee detailed messages above for each check.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
