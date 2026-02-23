#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

CLI for user-configurable training data generation — synthetic genomes and
simulated reads for training or retraining StrandWeaver ML models.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import argparse
import sys
import logging
from pathlib import Path

from strandweaver.user_training import (
    UserGenomeConfig,
    UserReadConfig,
    UserTrainingConfig,
    GraphTrainingConfig,
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

# Human-readable labels for summary output
_TECH_LABELS = {
    'illumina': 'Illumina PE',
    'hifi': 'PacBio HiFi',
    'ont': 'ONT',
    'ultra_long': 'ONT Ultra-long',
    'hic': 'Hi-C',
    'ancient_dna': 'Ancient DNA',
}


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for StrandWeaver ML models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # 10 diploid genomes (1 Mb), HiFi only at 30×
  %(prog)s --genome-size 1000000 -o data/test

  # 50 genomes, three technologies
  %(prog)s --genome-size 5000000 -n 50 \\
      --read-types hifi ont hic --coverage 30 20 15 \\
      -o data/multi_tech

  # Repeat-rich, low-GC organism
  %(prog)s --genome-size 2000000 -n 20 \\
      --gc-content 0.35 --repeat-density 0.60 \\
      -o data/repeat_rich
        """
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--genome-size', type=int, required=True,
        help='Simulated genome size in bp (e.g. 1000000 for 1 Mb)')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Output directory for generated training data')

    # ── Genome characteristics ────────────────────────────────────────────
    parser.add_argument(
        '--gc-content', type=float, default=0.42,
        help='GC content as a fraction, 0–1 (default: 0.42)')
    parser.add_argument(
        '--repeat-density', type=float, default=0.30,
        help='Fraction of the genome that is repetitive, 0–1 (default: 0.30)')

    # ── Read types & coverage ─────────────────────────────────────────────
    parser.add_argument(
        '--read-types', type=str, nargs='+', default=['hifi'],
        choices=['illumina', 'hifi', 'ont', 'ultra_long', 'hic', 'ancient_dna'],
        help='Sequencing technologies to simulate (default: hifi)')
    parser.add_argument(
        '--coverage', type=float, nargs='+', default=None,
        help=('Coverage for each --read-type, in the same order. '
              'A single value is broadcast to all types (default: 30)'))

    # ── Dataset size & reproducibility ────────────────────────────────────
    parser.add_argument(
        '-n', '--num-genomes', type=int, default=10,
        help='Number of independent genomes to generate (default: 10)')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility')

    # ── Advanced genome parameters ────────────────────────────────────────
    adv = parser.add_argument_group(
        'Advanced Genome Parameters',
        'Fine-tune variation and structure (sensible defaults provided)')
    adv.add_argument(
        '--ploidy', type=str, default='diploid',
        choices=['haploid', 'diploid', 'triploid', 'tetraploid'],
        help='Ploidy level (default: diploid)')
    adv.add_argument(
        '--snp-rate', type=float, default=0.001,
        help='SNP rate per bp between haplotypes (default: 0.001)')
    adv.add_argument(
        '--indel-rate', type=float, default=0.0001,
        help='Small indel rate per bp (default: 0.0001)')
    adv.add_argument(
        '--sv-density', type=float, default=0.00001,
        help='Structural variant density per bp (default: 0.00001)')
    adv.add_argument(
        '--sv-types', type=str, nargs='+',
        default=['deletion', 'insertion', 'inversion', 'duplication'],
        choices=['deletion', 'insertion', 'inversion', 'duplication', 'translocation'],
        help='SV types to include (default: del ins inv dup)')
    adv.add_argument(
        '--centromeres', type=int, default=1,
        help='Number of centromeric regions per genome (default: 1)')

    # ── Advanced read parameters ──────────────────────────────────────────
    radv = parser.add_argument_group(
        'Advanced Read Parameters',
        'Override technology-specific defaults (one value per --read-type)')
    radv.add_argument(
        '--error-rates', type=float, nargs='+', default=None,
        help='Error rate for each read type (default: tech-specific)')

    # ── Graph training data ────────────────────────────────────────────────
    gadv = parser.add_argument_group(
        'Graph Training Data',
        'Generate labelled assembly graphs for model training')
    gadv.add_argument(
        '--graph-training', action='store_true', default=False,
        help='Enable graph training data generation (builds synthetic '
             'overlap graphs with ground-truth labels for EdgeAI, '
             'DiploidAI, PathGNN, UL Routing, and SV Detection)')
    gadv.add_argument(
        '--min-overlap-bp', type=int, default=500,
        help='Minimum overlap length (bp) for graph edges (default: 500)')
    gadv.add_argument(
        '--min-overlap-identity', type=float, default=0.90,
        help='Minimum overlap identity for graph edges (default: 0.90)')
    gadv.add_argument(
        '--noise-edge-fraction', type=float, default=0.10,
        help='Fraction of noise (false) edges to inject for '
             'negative-class training (default: 0.10)')
    gadv.add_argument(
        '--no-noise-edges', action='store_true', default=False,
        help='Disable noise edge injection')
    gadv.add_argument(
        '--no-gfa', action='store_true', default=False,
        help='Skip GFA file export')
    gadv.add_argument(
        '--graph-max-coverage', type=float, default=None,
        help='Subsample reads to this coverage before graph '
             'construction (saves RAM for large genomes)')

    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)

    # ── Resolve coverage list ─────────────────────────────────────────────
    num_techs = len(args.read_types)
    if args.coverage is None:
        coverages = [30.0] * num_techs
    elif len(args.coverage) == 1:
        coverages = args.coverage * num_techs
    elif len(args.coverage) != num_techs:
        logger.error(
            f"Number of --coverage values ({len(args.coverage)}) must match "
            f"number of --read-types ({num_techs})")
        return 1
    else:
        coverages = args.coverage

    # ── Resolve error-rate list ───────────────────────────────────────────
    if args.error_rates is None:
        error_rates = [None] * num_techs          # use tech-specific defaults
    elif len(args.error_rates) == 1:
        error_rates = args.error_rates * num_techs
    elif len(args.error_rates) != num_techs:
        logger.error(
            f"Number of --error-rates values ({len(args.error_rates)}) must "
            f"match number of --read-types ({num_techs})")
        return 1
    else:
        error_rates = args.error_rates

    # ── Build config objects ──────────────────────────────────────────────
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
        random_seed=args.seed,
    )

    read_configs = [
        UserReadConfig(read_type=ReadType(rt), coverage=cov, error_rate=er)
        for rt, cov, er in zip(args.read_types, coverages, error_rates)
    ]

    # ── Build graph training config (if enabled) ─────────────────────────
    graph_config = None
    if args.graph_training:
        graph_config = GraphTrainingConfig(
            enabled=True,
            min_overlap_bp=args.min_overlap_bp,
            min_overlap_identity=args.min_overlap_identity,
            add_noise_edges=not args.no_noise_edges,
            noise_edge_fraction=args.noise_edge_fraction,
            export_gfa=not args.no_gfa,
            max_coverage_for_graph=args.graph_max_coverage,
        )

    training_config = UserTrainingConfig(
        genome_config=genome_config,
        read_configs=read_configs,
        output_dir=Path(args.output),
        graph_config=graph_config,
    )

    # ── Display summary ───────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("STRANDWEAVER — TRAINING DATA GENERATION")
    logger.info("=" * 72)
    logger.info(f"  Genome size      : {args.genome_size:>12,} bp")
    logger.info(f"  Num genomes      : {args.num_genomes:>12}")
    logger.info(f"  Ploidy           :  {args.ploidy}")
    logger.info(f"  GC content       : {args.gc_content:>11.0%}")
    logger.info(f"  Repeat density   : {args.repeat_density:>11.0%}")
    logger.info(f"  SNP rate         :  {args.snp_rate}")
    logger.info(f"  Indel rate       :  {args.indel_rate}")
    logger.info(f"  SV density       :  {args.sv_density}")
    logger.info(f"  SV types         :  {', '.join(args.sv_types)}")
    techs_str = ", ".join(
        f"{_TECH_LABELS.get(rt, rt)} {cov:.0f}×"
        for rt, cov in zip(args.read_types, coverages))
    logger.info(f"  Read types       :  {techs_str}")
    logger.info(f"  Seed             :  {args.seed if args.seed is not None else 'random'}")
    logger.info(f"  Output           :  {args.output}")
    if args.graph_training:
        logger.info(f"  Graph training   :  ENABLED")
        logger.info(f"    Min overlap    : {args.min_overlap_bp:>8} bp")
        logger.info(f"    Min identity   : {args.min_overlap_identity:>7.0%}")
        logger.info(f"    Noise edges    :  {'off' if args.no_noise_edges else f'{args.noise_edge_fraction:.0%}'}")
        logger.info(f"    GFA export     :  {'no' if args.no_gfa else 'yes'}")
    else:
        logger.info(f"  Graph training   :  disabled (use --graph-training to enable)")
    logger.info("=" * 72)

    # ── Generate ──────────────────────────────────────────────────────────
    try:
        summary = generate_custom_training_data(training_config)

        logger.info("")
        logger.info("=" * 72)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 72)
        logger.info(f"  Genomes generated : {summary['num_genomes_generated']}")
        logger.info(f"  Total time        : {summary['generation_time_human']}")
        logger.info(f"  Output directory  : {args.output}")
        logger.info("=" * 72)
        return 0

    except Exception as e:
        logger.error(f"Error during training data generation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
