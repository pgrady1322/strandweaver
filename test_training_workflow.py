#!/usr/bin/env python3
"""
Quick test of training workflow before running on GCP.

This test runs a minimal scenario to verify all components work.
"""

import sys
from pathlib import Path
import logging
import shutil

# Add strandweaver to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def show_production_configs():
    """Show production-scale configurations without importing full module."""
    logger.info("="*80)
    logger.info("PRODUCTION TRAINING CONFIGURATIONS")
    logger.info("="*80)
    logger.info("")
    logger.info("All production scenarios now use:")
    logger.info("  - Genome size: 10 Mb (10× increase from before)")
    logger.info("  - Num genomes: 100 (standardized across all)")
    logger.info("  - Total: 1 Gb per scenario")
    logger.info("")
    
    scenarios_info = {
        'balanced': {
            'time': '8-12 hours',
            'features': 'Standard balanced training',
        },
        'repeat_heavy': {
            'time': '10-15 hours',
            'features': '60% repeats, k=51',
        },
        'sv_dense': {
            'time': '8-12 hours',
            'features': '10× SV density, max SV=500kb',
        },
        'diploid_focus': {
            'time': '10-14 hours',
            'features': '2% heterozygosity, Hi-C=40×',
        },
        'ultra_long_focus': {
            'time': '12-18 hours',
            'features': 'UL=50× coverage',
        },
    }
    
    for scenario, info in scenarios_info.items():
        logger.info(f"  {scenario}:")
        logger.info(f"    - 100 genomes × 10,000,000 bp")
        logger.info(f"    - Estimated time on n1-highmem-16: {info['time']}")
        logger.info(f"    - Special features: {info['features']}")
        logger.info("")
    
    logger.info("Total estimated time for all 5 scenarios: ~50-70 hours")
    logger.info("Cost estimate on GCP n1-highmem-16 + T4: $75-$105 total")
    logger.info("")
    logger.info("Optimization tips:")
    logger.info("  1. Run scenarios in parallel on separate VMs")
    logger.info("  2. Use preemptible VMs for 70% discount")
    logger.info("  3. Start with 'balanced' to verify everything works")


def show_scenario_info():
    """Show all available scenarios."""
    logger.info("="*80)
    logger.info("AVAILABLE TRAINING SCENARIOS")
    logger.info("="*80)
    logger.info("")
    
    logger.info("Test scenarios (for quick verification):")
    logger.info("  simple:")
    logger.info("    - Genomes: 10 × 100,000 bp")
    logger.info("    - Purpose: Quick workflow test (2-5 min)")
    logger.info("")
    logger.info("  fast_balanced:")
    logger.info("    - Genomes: 20 × 500,000 bp")
    logger.info("    - Purpose: Small-scale test (10-15 min)")
    logger.info("")
    
    logger.info("Production scenarios (for model training):")
    show_production_configs()


def test_imports():
    """Test that imports work without circular dependency."""
    logger.info("="*80)
    logger.info("TESTING IMPORTS")
    logger.info("="*80)
    logger.info("")
    
    try:
        logger.info("Importing training module...")
        from strandweaver.training import (
            generate_training_corpus,
            SCENARIOS,
            list_scenarios,
        )
        logger.info("✓ Training module imported successfully")
        logger.info(f"✓ Found {len(list_scenarios())} scenarios")
        logger.info(f"✓ Available: {', '.join(list_scenarios())}")
        logger.info("")
        return True
    except Exception as e:
        logger.error("✗ Import failed!")
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        logger.info("")
        logger.info("This needs to be fixed before running on GCP.")
        logger.info("Likely issue: Circular import in assembly_core modules")
        return False


def test_simple_scenario():
    """Test the simplest scenario to verify workflow."""
    logger.info("="*80)
    logger.info("TESTING TRAINING WORKFLOW - SIMPLE SCENARIO")
    logger.info("="*80)
    logger.info("")
    logger.info("This will generate 1 genome × 100kb to test all components.")
    logger.info("Expected time: 2-5 minutes")
    logger.info("")
    
    # First test imports
    if not test_imports():
        return False
    
    # Now import what we need
    from strandweaver.training import generate_training_corpus
    
    # Clean up any previous test
    test_dir = Path('test_output')
    if test_dir.exists():
        logger.info(f"Cleaning up previous test: {test_dir}")
        shutil.rmtree(test_dir)
    
    try:
        # Override to use just 1 genome for quick test
        result = generate_training_corpus(
            scenario='simple',
            output_dir='test_output',
            num_processes=4,
            num_genomes=1  # Override to just 1 genome
        )
        
        logger.info("")
        logger.info("="*80)
        logger.info("TEST SUCCESSFUL!")
        logger.info("="*80)
        logger.info("")
        logger.info("Results:")
        logger.info(f"  - Genomes processed: {result.get('num_genomes', 0)}")
        logger.info(f"  - Total nodes: {result.get('total_graph_nodes', 0)}")
        logger.info(f"  - Total edges: {result.get('total_graph_edges', 0)}")
        logger.info(f"  - Total labels: {result.get('total_labels', 0)}")
        logger.info(f"  - Output: {test_dir.absolute()}")
        logger.info("")
        
        # Check output structure
        if test_dir.exists():
            logger.info("Output structure:")
            for item in sorted(test_dir.rglob('*'))[:20]:  # Limit to first 20
                if item.is_file():
                    size = item.stat().st_size / (1024*1024)  # MB
                    logger.info(f"  {item.relative_to(test_dir)}: {size:.2f} MB")
        
        logger.info("")
        logger.info("✓ Training workflow is working correctly!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review test_output/ directory")
        logger.info("  2. Run 'python test_training_workflow.py --show-production' to see production configs")
        logger.info("  3. Run on GCP VM for full training data generation")
        
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error("TEST FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Please fix the error before running on GCP.")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test training workflow and show scenario information",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--show-scenarios', action='store_true', 
                       help='Show all available scenarios')
    parser.add_argument('--show-production', action='store_true',
                       help='Show production-scale configurations and estimates')
    parser.add_argument('--test-imports', action='store_true',
                       help='Test that imports work correctly')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test (1 genome × 100kb)')
    
    args = parser.parse_args()
    
    if args.show_scenarios:
        show_scenario_info()
    elif args.show_production:
        show_production_configs()
    elif args.test_imports:
        success = test_imports()
        sys.exit(0 if success else 1)
    elif args.test:
        success = test_simple_scenario()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        logger.info("")
        logger.info("Common usage:")
        logger.info("  python test_training_workflow.py --test-imports       # Test imports")
        logger.info("  python test_training_workflow.py --show-production    # Show production configs")
        logger.info("  python test_training_workflow.py --test               # Run quick test")


