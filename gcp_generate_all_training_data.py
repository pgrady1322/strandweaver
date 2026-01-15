#!/usr/bin/env python3
"""
Generate production training data for all 5 StrandWeaver AI modules.

This script is designed to run on a GCP VM with:
  - n1-highmem-16 (16 vCPUs, 104 GB RAM)
  - NVIDIA T4 GPU (16 GB VRAM)
  - 500 GB SSD storage

It generates training data for 5 scenarios, each with 100 genomes × 10 Mb.

Usage:
    # Run all scenarios sequentially
    python gcp_generate_all_training_data.py --all --workers 12
    
    # Run specific scenario
    python gcp_generate_all_training_data.py --scenario balanced --workers 12
    
    # Dry run (estimate time/resources)
    python gcp_generate_all_training_data.py --all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add strandweaver to path
sys.path.insert(0, str(Path(__file__).parent))

from strandweaver.training import (
    generate_training_corpus,
    SCENARIOS,
    list_scenarios,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Estimated runtimes per scenario (on n1-highmem-16 with T4)
SCENARIO_ESTIMATES = {
    'balanced': 10.0,  # hours
    'repeat_heavy': 12.5,
    'sv_dense': 10.0,
    'diploid_focus': 12.0,
    'ultra_long_focus': 15.0,
}


def estimate_scenario_time(scenario: str) -> float:
    """Estimate runtime in hours for a scenario."""
    return SCENARIO_ESTIMATES.get(scenario, 10.0)


def estimate_scenario_size(scenario: str) -> float:
    """Estimate output size in GB for a scenario."""
    config = SCENARIOS[scenario]
    
    # Rough estimate: 
    # - Genomes: 2× genome size (haploid A + B)
    # - Reads: ~10× genome size (all technologies)
    # - Features: ~5× genome size
    # Total: ~17× genome size
    total_bp = config.num_genomes * config.genome_size
    estimated_gb = (total_bp * 17) / 1e9
    
    return estimated_gb


def show_dry_run(scenarios: list):
    """Show estimates for scenarios without running."""
    logger.info("="*80)
    logger.info("DRY RUN: Training Data Generation Estimates")
    logger.info("="*80)
    logger.info("")
    
    total_time = 0.0
    total_size = 0.0
    
    for scenario in scenarios:
        config = SCENARIOS[scenario]
        time_hrs = estimate_scenario_time(scenario)
        size_gb = estimate_scenario_size(scenario)
        
        logger.info(f"{scenario}:")
        logger.info(f"  Configuration:")
        logger.info(f"    - {config.num_genomes} genomes × {config.genome_size:,} bp")
        logger.info(f"    - Total genome data: {config.num_genomes * config.genome_size / 1e9:.1f} Gb")
        logger.info(f"  Estimated runtime: {time_hrs:.1f} hours")
        logger.info(f"  Estimated output: {size_gb:.1f} GB")
        logger.info("")
        
        total_time += time_hrs
        total_size += size_gb
    
    logger.info("-"*80)
    logger.info(f"TOTAL for {len(scenarios)} scenarios:")
    logger.info(f"  Runtime: {total_time:.1f} hours ({total_time/24:.1f} days)")
    logger.info(f"  Output: {total_size:.1f} GB")
    logger.info(f"  Cost (n1-highmem-16 + T4): ${total_time * 1.50:.2f}")
    logger.info("")
    
    # Show completion time
    if total_time < 24:
        logger.info(f"If started now: Complete in {total_time:.1f} hours")
    else:
        logger.info(f"If started now: Complete in {total_time/24:.1f} days")
    
    end_time = datetime.now() + timedelta(hours=total_time)
    logger.info(f"Estimated completion: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


def run_scenario(scenario: str, output_base: str, workers: int, use_gpu: bool):
    """Run training data generation for one scenario."""
    logger.info("")
    logger.info("="*80)
    logger.info(f"STARTING SCENARIO: {scenario}")
    logger.info("="*80)
    logger.info("")
    
    config = SCENARIOS[scenario]
    estimated_time = estimate_scenario_time(scenario)
    estimated_size = estimate_scenario_size(scenario)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Genomes: {config.num_genomes} × {config.genome_size:,} bp")
    logger.info(f"  - Workers: {workers}")
    logger.info(f"  - GPU: {'Yes' if use_gpu else 'No'}")
    logger.info(f"  - Estimated time: {estimated_time:.1f} hours")
    logger.info(f"  - Estimated size: {estimated_size:.1f} GB")
    logger.info("")
    
    start_time = time.time()
    
    try:
        result = generate_training_corpus(
            scenario=scenario,
            output_dir=output_base,
            num_processes=workers,
        )
        
        elapsed = (time.time() - start_time) / 3600  # hours
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"COMPLETED: {scenario}")
        logger.info("="*80)
        logger.info(f"  Runtime: {elapsed:.2f} hours")
        logger.info(f"  Genomes: {result.get('num_genomes', 0)}")
        logger.info(f"  Graph nodes: {result.get('total_graph_nodes', 0):,}")
        logger.info(f"  Graph edges: {result.get('total_graph_edges', 0):,}")
        logger.info(f"  Labels: {result.get('total_labels', 0):,}")
        
        # Check output size
        output_path = Path(output_base) / scenario
        if output_path.exists():
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            logger.info(f"  Output size: {total_size / 1e9:.2f} GB")
        
        logger.info("")
        
        return True
        
    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        logger.error("")
        logger.error("="*80)
        logger.error(f"FAILED: {scenario}")
        logger.error("="*80)
        logger.error(f"  Error: {e}")
        logger.error(f"  Runtime before failure: {elapsed:.2f} hours")
        logger.error("")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate production training data on GCP VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production scenarios (all 100 genomes × 10 Mb):
  - balanced: Standard training data
  - repeat_heavy: 60% repeat density
  - sv_dense: 10× structural variant density
  - diploid_focus: 2% heterozygosity
  - ultra_long_focus: 50× ultra-long coverage

Estimated total time: 50-60 hours
Estimated total cost: $75-90 (n1-highmem-16 + T4)

Examples:
  # Dry run
  python gcp_generate_all_training_data.py --all --dry-run
  
  # Run all scenarios
  python gcp_generate_all_training_data.py --all --workers 12
  
  # Run specific scenario
  python gcp_generate_all_training_data.py --scenario balanced --workers 12
"""
    )
    
    parser.add_argument(
        '--scenario',
        choices=list_scenarios(),
        help='Generate specific scenario'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all production scenarios'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default='training_data',
        help='Base output directory (default: training_data)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        help='Number of parallel workers (default: 12)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show estimates without running'
    )
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.all:
        scenarios = ['balanced', 'repeat_heavy', 'sv_dense', 'diploid_focus', 'ultra_long_focus']
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.error("Must specify --scenario or --all")
    
    # Dry run
    if args.dry_run:
        show_dry_run(scenarios)
        return
    
    # Setup GPU
    use_gpu = not args.no_gpu
    if use_gpu:
        logger.info("Checking GPU availability...")
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✓ MPS GPU detected (Apple Silicon)")
            else:
                logger.warning("GPU requested but not available, using CPU")
                use_gpu = False
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            use_gpu = False
    
    logger.info("")
    logger.info("="*80)
    logger.info("PRODUCTION TRAINING DATA GENERATION")
    logger.info("="*80)
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"GPU: {'Yes' if use_gpu else 'No'}")
    logger.info(f"Output: {args.output.absolute()}")
    logger.info("")
    
    # Run scenarios
    overall_start = time.time()
    results = {}
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"[{i}/{len(scenarios)}] Starting {scenario}...")
        success = run_scenario(scenario, str(args.output), args.workers, use_gpu)
        results[scenario] = success
        
        if not success:
            logger.error(f"Scenario {scenario} failed. Continuing with remaining scenarios...")
    
    overall_elapsed = (time.time() - overall_start) / 3600
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total runtime: {overall_elapsed:.2f} hours")
    logger.info("")
    logger.info("Results:")
    for scenario, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {scenario}: {status}")
    logger.info("")
    
    num_success = sum(results.values())
    logger.info(f"Completed: {num_success}/{len(scenarios)} scenarios")
    
    if num_success == len(scenarios):
        logger.info("")
        logger.info("🎉 All scenarios completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Compress output: tar -czf training_data.tar.gz training_data/")
        logger.info("  2. Download: gcloud compute scp VM_NAME:~/training_data.tar.gz .")
        logger.info("  3. Stop VM: gcloud compute instances stop VM_NAME")
        logger.info("  4. Begin Phase 5.4: ML model training")
    else:
        logger.warning("")
        logger.warning(f"⚠️  {len(scenarios) - num_success} scenarios failed")
        logger.warning("Review error logs above and re-run failed scenarios")


if __name__ == '__main__':
    main()
