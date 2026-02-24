#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Assembly core subpackage — graph engines and ML modules.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from .illumina_olc_contig_module import (
    ContigBuilder,
    Overlap,
    OverlapGraph,
    PairedEndInfo,
    build_contigs_from_reads
)

from .dbg_engine_module import (
    build_dbg_from_long_reads,
    DBGGraph,
    DBGNode,
    KmerGraph,
    KmerNode,
    KmerEdge,
    Anchor
)

from .string_graph_engine_module import (
    build_string_graph_from_dbg_and_ul,
    StringGraph,
    ULAnchor,
    LongReadOverlay
)

from .edgewarden_module import EdgeWarden
from .pathweaver_module import PathWeaver
from .threadcompass_module import ThreadCompass
from .strandtether_module import StrandTether
from .pathweaver_module import (
    PathGNN, FeatureExtractor, PathExtractor, GraphTensors, GNNPathResult,
    PathGNNModel, SimpleGNN, MediumGNN, DeepGNN, GNNConfig
)
from .svscribe_module import SVScribe  # Fixed: was SVDetector
from .haplotype_detangler_module import HaplotypeDetangler, Ploidy

# ULRouter is an alias for ThreadCompass (the actual implementation)
# ThreadCompass provides the complete UL read routing functionality
ULRouter = ThreadCompass

__all__ = [
    # Assembly functions
    "assemble",
    "build_contigs_from_reads",
    "build_dbg_from_long_reads",
    "build_string_graph_from_dbg_and_ul",
    # Core classes
    "ContigBuilder",
    "Overlap",
    "OverlapGraph",
    "PairedEndInfo",
    "DBGGraph",
    "DBGNode",
    "StringGraph",
    "ULAnchor",
    # AI-powered modules
    "EdgeWarden",
    "PathWeaver",
    "ThreadCompass",
    "StrandTether",
    "PathGNN",
    "ULRouter",
    "SVScribe",  # Fixed: was SVDetector
    "HaplotypeDetangler",
    "Ploidy",
    # Data structures
    "KmerGraph",
    "KmerNode",
    "KmerEdge",
    "LongReadOverlay",
    "Anchor",
]


def assemble(reads_file, output_file, technology='auto', **kwargs):
    """
    Convenience function for graph-based assembly.
    
    Thin wrapper around the DBG engine: builds a De Bruijn graph from
    the input reads and writes assembled contigs to *output_file*.
    For the full pipeline (error profiling, correction, scaffolding,
    finishing) use ``strandweaver pipeline`` or
    :class:`~strandweaver.utils.pipeline.PipelineOrchestrator`.
    
    Args:
        reads_file: Path to input reads (FASTA/FASTQ, optionally gzipped)
        output_file: Path to output FASTA file
        technology: Sequencing technology hint ('auto', 'illumina', 'ont',
                    'pacbio').  Affects default k-mer size.
        **kwargs: Forwarded to ``build_dbg_from_long_reads``
                  (e.g. ``base_k``, ``min_coverage``)
    
    Returns:
        dict with basic assembly statistics
    """
    from pathlib import Path
    from ..io_utils import read_fastq, read_fasta, write_fasta
    
    reads_path = Path(reads_file)
    suffix = reads_path.name.replace('.gz', '').rsplit('.', 1)[-1].lower()
    if suffix in ('fq', 'fastq'):
        reads = list(read_fastq(reads_path))
    else:
        reads = list(read_fasta(reads_path))
    
    base_k = kwargs.pop('base_k', 31)
    min_coverage = kwargs.pop('min_coverage', 2)
    
    graph = build_dbg_from_long_reads(reads, base_k=base_k,
                                       min_coverage=min_coverage, **kwargs)
    
    # Extract contigs from graph
    contigs = []
    for node_id, node in graph.nodes.items():
        if len(node.sequence) >= kwargs.get('min_contig_length', 500):
            from ..io_utils import SeqRead
            contigs.append(SeqRead(
                id=f"contig_{node_id}",
                sequence=node.sequence,
                quality=None
            ))
    
    write_fasta(contigs, Path(output_file))
    
    total_bp = sum(len(c.sequence) for c in contigs)
    return {
        'num_contigs': len(contigs),
        'total_bases': total_bp,
        'graph_nodes': len(graph.nodes),
        'graph_edges': len(graph.edges),
    }

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
