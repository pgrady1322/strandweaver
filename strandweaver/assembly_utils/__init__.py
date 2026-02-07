"""
StrandWeaver v0.1.0

Assembly Utilities module for StrandWeaver.

This module provides utility functions for assembly tasks:
- Graph cleanup and phasing-aware validation
- Misassembly detection and flagging
- Chromosome classification (microchromosome identification)
- Gene annotation (BLAST, Augustus, BUSCO, ORF)
- Hi-C read alignment to assembly graphs

Author: StrandWeaver Development Team
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
]
