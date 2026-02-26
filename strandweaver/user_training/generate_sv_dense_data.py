#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Generate SV-dense training data for SVScribe improvement.

Produces training batches with 5–25× higher SV density than the defaults,
varied SV size distributions, and multiple SV type mixes to give the
SVScribe model ≥ 1,000 examples per class.

Usage:
  # Default: 5 batches × 40 genomes, SV density sweep from 1e-4 to 5e-4
  python -m strandweaver.user_training.generate_sv_dense_data -o sv_dense_output/

  # Quick smoke test (2 batches × 5 genomes)
  python -m strandweaver.user_training.generate_sv_dense_data \
      -o sv_dense_output/ --num-genomes 5 --num-batches 2

  # Custom density range
  python -m strandweaver.user_training.generate_sv_dense_data \
      -o sv_dense_output/ --sv-density-min 2e-4 --sv-density-max 5e-4

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  BATCH CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

# Each batch varies SV density, size distribution, and SV type mix to
# produce a diverse training set.  The goal is ≥ 1,000 examples per SV
# type from the combined batches.

DEFAULT_BATCHES = [
    {
        'name': 'sv_dense_1e4',
        'sv_density': 1e-4,   # 100 SVs per Mb (10× default)
        'sv_types': ['deletion', 'insertion', 'inversion', 'duplication'],
        'genome_size': 2_000_000,
        'repeat_density': 0.30,
        'gc_content': 0.42,
        'desc': 'Moderate SV density (10× baseline), all 4 types',
    },
    {
        'name': 'sv_dense_2e4',
        'sv_density': 2e-4,   # 200 SVs per Mb (20× default)
        'sv_types': ['deletion', 'insertion', 'inversion', 'duplication'],
        'genome_size': 1_500_000,
        'repeat_density': 0.35,
        'gc_content': 0.40,
        'desc': 'High SV density (20× baseline), higher repeat content',
    },
    {
        'name': 'sv_dense_3e4',
        'sv_density': 3e-4,   # 300 SVs per Mb (30× default)
        'sv_types': ['deletion', 'insertion', 'inversion', 'duplication'],
        'genome_size': 1_000_000,
        'repeat_density': 0.25,
        'gc_content': 0.45,
        'desc': 'Very high SV density (30× baseline), GC-rich',
    },
    {
        'name': 'sv_dense_inv_dup_focus',
        'sv_density': 4e-4,
        'sv_types': ['inversion', 'duplication'],
        'genome_size': 1_000_000,
        'repeat_density': 0.30,
        'gc_content': 0.42,
        'desc': 'Focused on inversions + duplications (weakest classes)',
    },
    {
        'name': 'sv_dense_ins_focus',
        'sv_density': 5e-4,
        'sv_types': ['insertion'],
        'genome_size': 1_000_000,
        'repeat_density': 0.30,
        'gc_content': 0.42,
        'desc': 'Focused on insertions only (weakest class, F1=0.319)',
    },
]


def _build_cli_args(
    batch: Dict[str, Any],
    output_dir: Path,
    num_genomes: int,
    coverage: float,
    read_types: List[str],
    seed: Optional[int],
) -> List[str]:
    """Build CLI arguments for generate_training_data.py for one batch."""
    batch_dir = output_dir / batch['name']

    args = [
        '--genome-size', str(batch['genome_size']),
        '-n', str(num_genomes),
        '-o', str(batch_dir),
        '--gc-content', str(batch['gc_content']),
        '--repeat-density', str(batch['repeat_density']),
        '--sv-density', str(batch['sv_density']),
        '--sv-types', *batch['sv_types'],
        '--read-types', *read_types,
        '--coverage', str(coverage),
        '--graph-training',
    ]

    if seed is not None:
        args.extend(['--seed', str(seed)])

    return args


def generate_sv_dense_batches(
    output_dir: Path,
    num_genomes: int = 40,
    coverage: float = 30.0,
    read_types: Optional[List[str]] = None,
    batches: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = 42,
    sv_density_min: Optional[float] = None,
    sv_density_max: Optional[float] = None,
    num_batches: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate multiple SV-dense training data batches.

    Parameters
    ----------
    output_dir : Path
        Root output directory. Each batch creates a subdirectory.
    num_genomes : int
        Genomes per batch (default 40).
    coverage : float
        Coverage depth (default 30.0).
    read_types : list of str
        Sequencing technologies (default: ['hifi']).
    batches : list of dict or None
        Custom batch configs. None uses DEFAULT_BATCHES.
    seed : int or None
        Base random seed. Each batch offsets by batch index.
    sv_density_min : float or None
        If set, override batch SV densities with a linear sweep.
    sv_density_max : float or None
        Upper bound of density sweep (used with sv_density_min).
    num_batches : int or None
        If set with density sweep, number of batches to generate.
    dry_run : bool
        If True, only print planned commands without executing.

    Returns
    -------
    dict
        Summary with per-batch results.
    """
    if read_types is None:
        read_types = ['hifi']

    if batches is None:
        batches = list(DEFAULT_BATCHES)

    # Apply density sweep if requested
    if sv_density_min is not None and sv_density_max is not None:
        n = num_batches or len(batches)
        if n == 1:
            densities = [sv_density_min]
        else:
            step = (sv_density_max - sv_density_min) / (n - 1)
            densities = [sv_density_min + i * step for i in range(n)]

        batches = []
        for i, d in enumerate(densities):
            batches.append({
                'name': f'sv_dense_sweep_{i:02d}',
                'sv_density': d,
                'sv_types': ['deletion', 'insertion', 'inversion', 'duplication'],
                'genome_size': 1_500_000,
                'repeat_density': 0.30,
                'gc_content': 0.42,
                'desc': f'Density sweep batch {i}: {d:.1e} SVs/bp',
            })
    elif num_batches is not None:
        batches = batches[:num_batches]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'num_batches': len(batches),
        'num_genomes_per_batch': num_genomes,
        'coverage': coverage,
        'read_types': read_types,
        'batches': {},
    }

    t_start = time.time()

    for batch_idx, batch in enumerate(batches):
        batch_seed = (seed + batch_idx * 1000) if seed is not None else None
        cli_args = _build_cli_args(
            batch, output_dir, num_genomes, coverage, read_types, batch_seed,
        )

        # Expected SV counts
        expected_svs_per_genome = int(batch['sv_density'] * batch['genome_size'])
        expected_total_svs = expected_svs_per_genome * num_genomes
        per_type_estimate = expected_total_svs // max(len(batch['sv_types']), 1)

        logger.info("")
        logger.info("=" * 62)
        logger.info("  Batch %d/%d: %s", batch_idx + 1, len(batches), batch['name'])
        logger.info("  %s", batch['desc'])
        logger.info("  SV density    : %.1e SVs/bp (%d SVs per %d bp genome)",
                     batch['sv_density'], expected_svs_per_genome, batch['genome_size'])
        logger.info("  SV types      : %s", ', '.join(batch['sv_types']))
        logger.info("  Expected SVs  : ~%d total (~%d per type)",
                     expected_total_svs, per_type_estimate)
        logger.info("  Genomes       : %d", num_genomes)
        logger.info("  CLI args      : %s", ' '.join(cli_args))
        logger.info("=" * 62)

        batch_result = {
            'config': batch,
            'expected_total_svs': expected_total_svs,
            'expected_per_type': per_type_estimate,
            'cli_args': cli_args,
        }

        if dry_run:
            logger.info("  [DRY RUN] Skipping execution")
            batch_result['status'] = 'dry_run'
        else:
            try:
                from strandweaver.user_training.generate_training_data import main as gen_main
                logger.info("  Generating training data...")
                t_batch = time.time()
                gen_main(cli_args)
                elapsed = round(time.time() - t_batch, 1)
                batch_result['status'] = 'success'
                batch_result['elapsed_seconds'] = elapsed
                logger.info("  ✓ Batch complete in %.1fs", elapsed)
            except Exception as exc:
                logger.error("  ✗ Batch failed: %s", exc)
                batch_result['status'] = 'failed'
                batch_result['error'] = str(exc)

        report['batches'][batch['name']] = batch_result

    total_elapsed = round(time.time() - t_start, 1)
    report['total_elapsed_seconds'] = total_elapsed

    # Estimate total SV examples across all batches
    total_sv_estimate = sum(
        b.get('expected_total_svs', 0) for b in report['batches'].values()
    )
    report['total_sv_estimate'] = total_sv_estimate

    # Save report
    report_path = output_dir / 'sv_dense_generation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  SV-Dense Data Generation Complete")
    logger.info("  Total time     : %.1fs", total_elapsed)
    logger.info("  Batches        : %d", len(batches))
    logger.info("  ~Total SV rows : %d (target: ≥1,000 per type)", total_sv_estimate)
    logger.info("  Report         : %s", report_path)
    logger.info("=" * 62)

    return report


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='generate-sv-dense-data',
        description='Generate SV-dense training data for SVScribe improvement.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Default: 5 batches × 40 genomes
              %(prog)s -o sv_dense_output/

              # Quick smoke test
              %(prog)s -o sv_dense_output/ --num-genomes 5 --num-batches 2

              # Custom density sweep (10 batches from 5e-5 to 5e-4)
              %(prog)s -o sv_dense_output/ \\
                  --sv-density-min 5e-5 --sv-density-max 5e-4 --num-batches 10

              # Dry run (print planned commands only)
              %(prog)s -o sv_dense_output/ --dry-run

              # After generation, train SVScribe on the combined data:
              python -m strandweaver.user_training.train_models \\
                  --data-dir sv_dense_output/ --output-dir trained_models/ \\
                  --models sv_ai
        """),
    )

    parser.add_argument('-o', '--output', required=True,
                        help='Root output directory for all batches')
    parser.add_argument('--num-genomes', type=int, default=40,
                        help='Genomes per batch (default: 40)')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Limit to N batches from the default set (default: all 5)')
    parser.add_argument('--coverage', type=float, default=30.0,
                        help='Coverage depth (default: 30.0)')
    parser.add_argument('--read-types', nargs='+', default=['hifi'],
                        choices=['hifi', 'ont', 'illumina', 'ultra_long', 'hic'],
                        help='Read technologies (default: hifi)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    # Density sweep mode
    sweep = parser.add_argument_group('Density Sweep Mode',
                                      'Generate batches with linearly spaced SV densities')
    sweep.add_argument('--sv-density-min', type=float, default=None,
                       help='Minimum SV density for sweep (e.g. 5e-5)')
    sweep.add_argument('--sv-density-max', type=float, default=None,
                       help='Maximum SV density for sweep (e.g. 5e-4)')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print planned commands without executing')

    args = parser.parse_args()

    report = generate_sv_dense_batches(
        output_dir=Path(args.output),
        num_genomes=args.num_genomes,
        coverage=args.coverage,
        read_types=args.read_types,
        seed=args.seed,
        sv_density_min=args.sv_density_min,
        sv_density_max=args.sv_density_max,
        num_batches=args.num_batches,
        dry_run=args.dry_run,
    )

    # Final summary
    succeeded = sum(1 for b in report['batches'].values() if b.get('status') == 'success')
    failed = sum(1 for b in report['batches'].values() if b.get('status') == 'failed')
    dry = sum(1 for b in report['batches'].values() if b.get('status') == 'dry_run')

    if dry:
        print(f"\n  Dry run: {dry} batches planned (use without --dry-run to execute)\n")
    elif failed:
        print(f"\n  ⚠ {succeeded} succeeded, {failed} failed — check logs\n")
        sys.exit(1)
    else:
        print(f"\n  ✓ All {succeeded} batches completed successfully\n")


if __name__ == '__main__':
    main()

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
