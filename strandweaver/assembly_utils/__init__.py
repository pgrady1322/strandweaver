#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Assembly Utilities — graph cleanup, misassembly detection, chromosome
classification, gene annotation, Hi-C graph alignment, and QV estimation.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from .graph_cleanup import GraphCleanupEngine, clean_graph, CleanedGraphResult, Bubble
from .misassembly_detector import (
    MisassemblyDetector, MisassemblyFlag, MisassemblyType,
    ConfidenceLevel, CorrectionStrategy
)
from .chromosome_classifier import (
    ChromosomeClassifier, ChromosomePrefilter, GeneContentClassifier,
    AdvancedChromosomeFeatures, ChromosomeClassification
)
from .gene_annotation import (
    BlastAnnotator, AugustusPredictor, BUSCOAnalyzer,
    BlastHit, Gene, BUSCOResult, find_orfs
)
from .hic_graph_aligner import (
    HiCGraphAligner, HiCReadPair, HiCAlignment, align_hic_reads_to_graph
)
from .qv_estimator import (
    QVEstimator, ContigQV, AssemblyQV, estimate_assembly_qv
)

__all__ = [
    # Graph cleanup & phasing
    "GraphCleanupEngine",
    "clean_graph",
    "CleanedGraphResult",
    "Bubble",
    # Misassembly detection
    "MisassemblyDetector",
    "MisassemblyFlag",
    "MisassemblyType",
    "ConfidenceLevel",
    "CorrectionStrategy",
    # Chromosome classification
    "ChromosomeClassifier",
    "ChromosomePrefilter",
    "GeneContentClassifier",
    "AdvancedChromosomeFeatures",
    "ChromosomeClassification",
    # Gene annotation
    "BlastAnnotator",
    "AugustusPredictor",
    "BUSCOAnalyzer",
    "BlastHit",
    "Gene",
    "BUSCOResult",
    "find_orfs",
    # Hi-C graph alignment
    "HiCGraphAligner",
    "HiCReadPair",
    "HiCAlignment",
    "align_hic_reads_to_graph",
    # QV estimation
    "QVEstimator",
    "ContigQV",
    "AssemblyQV",
    "estimate_assembly_qv",
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
