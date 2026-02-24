#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Training subpackage — ML model training infrastructure.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from .synthetic_data_generator import (
    GenomeConfig,
    DiploidGenome,
    StructuralVariant,
    SVType,
    SimulatedRead,
    SimulatedReadPair,
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    AncientDNAConfig,
    generate_diploid_genome,
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_fastq,
    write_paired_fastq,
)

__all__ = [
    'GenomeConfig',
    'DiploidGenome',
    'StructuralVariant',
    'SVType',
    'SimulatedRead',
    'SimulatedReadPair',
    'IlluminaConfig',
    'HiFiConfig',
    'ONTConfig',
    'ULConfig',
    'HiCConfig',
    'AncientDNAConfig',
    'generate_diploid_genome',
    'simulate_illumina_reads',
    'simulate_long_reads',
    'simulate_hic_reads',
    'write_fastq',
    'write_paired_fastq',
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
