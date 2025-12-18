#!/usr/bin/env python3
"""
CLI wrapper for assembly AI training data generation.

Usage:
    python scripts/generate_assembly_training_data.py --scenario balanced --output-dir training_data/assembly_ai
"""

import argparse
import logging
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.full_corpus_generator import (
    generate_training_corpus,
    list_scenarios,
    get_scenario_info,
    estimate_generation_time
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for assembly AI models (5 models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios:
  - simple: Quick test (10 genomes × 100kb, ~2 min)
  - balanced: Production training (100 genomes × 1Mb, ~20 min)
  - repeat_heavy: Repeat-rich (50 genomes × 2Mb, 60% repeats, ~30 min)
  - sv_dense: SV-focused (50 genomes × 1Mb, 10× SV density, ~15 min)
  - diploid_focus: High heterozygosity (100 genomes × 1Mb, 2% het, ~25 min)
  - ultra_long_focus: UL-optimized (30 genomes × 5Mb, 50× UL, ~40 min)

Examples:
  # Quick test
  python scripts/generate_assembly_training_data.py --scenario simple --output-dir test_data
  
  # Production dataset
  python scripts/generate_assembly_training_data.py --scenario balanced --output-dir training_data/assembly_ai --num-workers 8
"""
    )
    
    parser.add_argument(
        '--scenario',
        choices=list_scenarios(),
        required=True,
        help='Training scenario to generate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for training data'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show scenario info and exit'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.use_gpu:
        try:
            from strandweaver.utils.device import get_optimal_device
            device_type, device_str = get_optimal_device(prefer_gpu=True)
            logger.info(f"Using device: {device_type.upper()} ({device_str})")
            if device_type == 'cpu':
                logger.warning("GPU requested but not available, using CPU")
        except ImportError:
            logger.warning("Device detection not available, using CPU")
            device_type, device_str = 'cpu', 'cpu'
    else:
        device_type, device_str = 'cpu', 'cpu'
        logger.info("Using device: CPU (default)")
    
    # Show info if requested
    if args.info:
        info = get_scenario_info(args.scenario)
        estimate = estimate_generation_time(args.scenario)
        
        print(f"\nScenario: {args.scenario}")
        print("=" * 60)
        print(f"Genome size: {info['genome_size']:,} bp")
        print(f"Number of genomes: {info['num_genomes']}")
        print(f"Illumina coverage: {info['illumina_coverage']}×")
        print(f"HiFi coverage: {info['hifi_coverage']}×")
        print(f"ONT coverage: {info['ont_coverage']}×")
        print(f"Ultra-long coverage: {info['ul_coverage']}×")
        print(f"Hi-C coverage: {info['hic_coverage']}×")
        print(f"\nEstimated time: {estimate['minutes']:.1f} minutes ({estimate['hours']:.2f} hours)")
        print(f"Estimated disk: {estimate['disk_gb']:.2f} GB")
        print()
        return 0
    
    # Generate training data
    logger.info(f"Generating assembly AI training data with scenario: {args.scenario}")
    logger.info(f"Output directory: {args.output_dir}")
    
    stats = generate_training_corpus(
        scenario=args.scenario,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    logger.info("Generation complete!")
    logger.info(f"Total genomes: {stats.get('total_genomes', 'N/A')}")
    
    return 0


if __name__ == '__main__':
    exit(main())
