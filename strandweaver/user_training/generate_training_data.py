#!/usr/bin/env python3
"""
Command-line interface for user-configurable training data generation.

Example usage:

# Basic usage with defaults (diploid, HiFi only)
python generate_training_data.py --genome-size 1000000 --num-genomes 10 --output training_data/test

# Specify multiple read types
python generate_training_data.py \
  --genome-size 5000000 \
  --num-genomes 50 \
  --read-types hifi ont ultra_long hic \
  --coverage 30 20 10 15 \
  --output training_data/multi_tech

# Custom genome parameters
python generate_training_data.py \
  --genome-size 2000000 \
  --num-genomes 20 \
  --gc-content 0.45 \
  --repeat-density 0.50 \
  --snp-rate 0.002 \
  --sv-density 0.00005 \
  --output training_data/repeat_rich

Author: StrandWeaver User Training Infrastructure
Date: February 2026
"""

import argparse
import sys
import logging
from pathlib import Path

from strandweaver.user_training import (
    UserGenomeConfig,
    UserReadConfig,
    UserTrainingConfig,
    ReadType,
    Ploidy,
    generate_custom_training_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate custom training data for StrandWeaver ML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic HiFi dataset
  %(prog)s --genome-size 1000000 --num-genomes 10 --output data/test
  
  # Multi-technology with custom coverage
  %(prog)s --genome-size 5000000 --num-genomes 50 \\
    --read-types hifi ont ultra_long hic \\
    --coverage 30 20 10 15 \\
    --output data/multi_tech
  
  # Repeat-rich genomes
  %(prog)s --genome-size 2000000 --num-genomes 20 \\
    --repeat-density 0.60 --gc-content 0.35 \\
    --output data/repeat_rich
        """
    )
    
    # Genome parameters
    genome_group = parser.add_argument_group('Genome Parameters')
    genome_group.add_argument('--genome-size', type=int, required=True,
                              help='Genome size in base pairs (e.g., 1000000 for 1Mb)')
    genome_group.add_argument('--num-genomes', type=int, required=True,
                              help='Number of independent genomes to generate')
    genome_group.add_argument('--gc-content', type=float, default=0.42,
                              help='GC content fraction (0-1, default: 0.42)')
    genome_group.add_argument('--repeat-density', type=float, default=0.30,
                              help='Repeat density fraction (0-1, default: 0.30)')
    genome_group.add_argument('--ploidy', type=str, default='diploid',
                              choices=['haploid', 'diploid', 'triploid', 'tetraploid'],
                              help='Ploidy level (default: diploid)')
    genome_group.add_argument('--snp-rate', type=float, default=0.001,
                              help='SNP rate per bp (default: 0.001)')
    genome_group.add_argument('--indel-rate', type=float, default=0.0001,
                              help='Indel rate per bp (default: 0.0001)')
    genome_group.add_argument('--sv-density', type=float, default=0.00001,
                              help='SV density per bp (default: 0.00001)')
    genome_group.add_argument('--sv-types', type=str, nargs='+',
                              default=['deletion', 'insertion', 'inversion', 'duplication'],
                              choices=['deletion', 'insertion', 'inversion', 'duplication', 'translocation'],
                              help='SV types to include (default: del ins inv dup)')
    genome_group.add_argument('--centromeres', type=int, default=1,
                              help='Number of centromeric regions (default: 1)')
    genome_group.add_argument('--seed', type=int, default=None,
                              help='Random seed for reproducibility (default: random)')
    
    # Read parameters
    read_group = parser.add_argument_group('Read Parameters')
    read_group.add_argument('--read-types', type=str, nargs='+',
                            default=['hifi'],
                            choices=['illumina', 'hifi', 'ont', 'ultra_long', 'hic', 'ancient_dna'],
                            help='Read types to generate (default: hifi)')
    read_group.add_argument('--coverage', type=float, nargs='+',
                            default=None,
                            help='Coverage for each read type (default: 30 for all)')
    read_group.add_argument('--error-rates', type=float, nargs='+',
                            default=None,
                            help='Error rates for each read type (default: tech-specific)')
    
    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--output', '-o', type=str, required=True,
                              help='Output directory for training data')
    output_group.add_argument('--num-workers', type=int, default=4,
                              help='Number of parallel workers (default: 4)')
    output_group.add_argument('--no-labels', action='store_true',
                              help='Skip ground-truth label generation')
    output_group.add_argument('--no-compress', action='store_true',
                              help='Do not compress output files')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create genome configuration
    genome_config = UserGenomeConfig(
        genome_size=args.genome_size,
        num_genomes=args.num_genomes,
        gc_content=args.gc_content,
        repeat_density=args.repeat_density,
        ploidy=Ploidy[args.ploidy.upper()],
        snp_rate=args.snp_rate,
        indel_rate=args.indel_rate,
        sv_density=args.sv_density,
        sv_types=args.sv_types,
        centromere_count=args.centromeres,
        random_seed=args.seed
    )
    
    # Create read configurations
    read_configs = []
    num_read_types = len(args.read_types)
    
    # Parse coverage (use 30 as default if not specified)
    if args.coverage is None:
        coverages = [30.0] * num_read_types
    elif len(args.coverage) == 1:
        coverages = args.coverage * num_read_types
    elif len(args.coverage) != num_read_types:
        logger.error(f"Number of coverage values ({len(args.coverage)}) must match number of read types ({num_read_types})")
        sys.exit(1)
    else:
        coverages = args.coverage
    
    # Parse error rates (use None for tech-specific defaults)
    if args.error_rates is None:
        error_rates = [0.01] * num_read_types  # Will be overridden by tech defaults
    elif len(args.error_rates) == 1:
        error_rates = args.error_rates * num_read_types
    elif len(args.error_rates) != num_read_types:
        logger.error(f"Number of error rates ({len(args.error_rates)}) must match number of read types ({num_read_types})")
        sys.exit(1)
    else:
        error_rates = args.error_rates
    
    # Create read configs
    for read_type, coverage, error_rate in zip(args.read_types, coverages, error_rates):
        read_configs.append(UserReadConfig(
            read_type=ReadType(read_type),
            coverage=coverage,
            error_rate=error_rate
        ))
    
    # Create full training configuration
    training_config = UserTrainingConfig(
        genome_config=genome_config,
        read_configs=read_configs,
        output_dir=Path(args.output),
        num_workers=args.num_workers,
        generate_labels=not args.no_labels,
        compress_output=not args.no_compress
    )
    
    # Display configuration
    logger.info("="*80)
    logger.info("TRAINING DATA GENERATION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Genome size: {args.genome_size:,} bp")
    logger.info(f"Number of genomes: {args.num_genomes}")
    logger.info(f"Ploidy: {args.ploidy}")
    logger.info(f"GC content: {args.gc_content:.2%}")
    logger.info(f"Repeat density: {args.repeat_density:.2%}")
    logger.info(f"SNP rate: {args.snp_rate}")
    logger.info(f"SV density: {args.sv_density}")
    logger.info(f"Read types: {', '.join(args.read_types)}")
    logger.info(f"Coverage: {', '.join(f'{c}x' for c in coverages)}")
    logger.info(f"Output: {args.output}")
    logger.info("="*80)
    
    # Generate training data
    try:
        summary = generate_custom_training_data(training_config)
        
        logger.info("\n" + "="*80)
        logger.info("GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Genomes generated: {summary['num_genomes_generated']}")
        logger.info(f"Total time: {summary['generation_time_human']}")
        logger.info(f"Output directory: {args.output}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training data generation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
