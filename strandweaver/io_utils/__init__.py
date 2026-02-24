#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

I/O utilities subpackage — file reading and writing.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

# Core data structures and file I/O
from .io_core import (
    SeqRead,
    ReadPair,
    ReadCollection,
    read_fastq,
    read_fastq_pairs,
    write_fastq,
    count_fastq_reads,
    get_fastq_stats,
    read_fasta,
    write_fasta,
    count_fasta_sequences,
    get_fasta_stats,
)

# Technology handling imports
from ..preprocessing.read_classification_utility import (
    ReadTechnology,
    ReadRole,
    ReadTypeProfile,
    READ_PROFILES,
    classify_read_type,
    get_read_profile,
    infer_technology_from_length,
    parse_technology,
    validate_technology_specs,
    auto_detect_technology,
    detect_technologies,
    format_technology_summary,
    FlowCellType,
    BasecallerType,
    BasecallerAccuracy,
    NanoporeMetadata,
    parse_nanopore_metadata,
    is_longbow_available,
    detect_nanopore_metadata_with_longbow,
    detect_from_fastq_headers,
)

# Assembly export functions
from .assembly_export import (
    # Graph export
    export_graph_to_gfa,
    export_coverage_csv,
    export_for_bandageng,
    validate_gfa_file,
    # Graph import
    load_graph_from_gfa,
    # Assembly export (NEW)
    write_contigs_fasta,
    write_scaffolds_fasta,
    export_assembly_stats,
    export_paths_tsv,
    # Utilities
    generate_unitig_name,
    parse_unitig_name,
    get_node_name_mapping,
    get_reverse_node_mapping,
)

# User input functions
from .user_input import (
    # Data structures
    UserPathCorrection,
    ReconstructedScaffold,
    ContigBreak,
    ForcedJoin,
    # Path corrections
    parse_user_corrections_tsv,
    reconstruct_scaffolds_from_user_paths,
    apply_user_scaffolds_to_string_graph,
    validate_user_corrections,
    export_scaffolds_to_tsv,
    get_scaffold_statistics,
    # Manual edits (NEW)
    load_contig_breaks,
    load_forced_joins,
    load_exclusion_list,
    apply_manual_edits,
)

__all__ = [
    # Core data structures
    "SeqRead",
    "ReadPair",
    "ReadCollection",
    
    # Technology types (from preprocessing)
    "ReadTechnology",
    "ReadRole",
    "ReadTypeProfile",
    "READ_PROFILES",
    
    # Read type utilities
    "classify_read_type",
    "get_read_profile",
    "infer_technology_from_length",
    
    # Technology validation and parsing
    "parse_technology",
    "validate_technology_specs",
    "auto_detect_technology",
    "detect_technologies",
    "format_technology_summary",
    
    # FASTQ I/O
    "read_fastq",
    "read_fastq_pairs",
    "write_fastq",
    "count_fastq_reads",
    "get_fastq_stats",
    
    # FASTA I/O
    "read_fasta",
    "write_fasta",
    "count_fasta_sequences",
    "get_fasta_stats",
    
    # Nanopore metadata
    "FlowCellType",
    "BasecallerType",
    "BasecallerAccuracy",
    "NanoporeMetadata",
    "parse_nanopore_metadata",
    
    # LongBow integration
    "is_longbow_available",
    "detect_nanopore_metadata_with_longbow",
    "detect_from_fastq_headers",
    
    # Assembly export
    "export_graph_to_gfa",
    "export_coverage_csv",
    "export_for_bandageng",
    "validate_gfa_file",
    "write_contigs_fasta",
    "write_scaffolds_fasta",
    "export_assembly_stats",
    "export_paths_tsv",
    "generate_unitig_name",
    "parse_unitig_name",
    "get_node_name_mapping",
    "get_reverse_node_mapping",
    
    # User input
    "UserPathCorrection",
    "ReconstructedScaffold",
    "ContigBreak",
    "ForcedJoin",
    "parse_user_corrections_tsv",
    "reconstruct_scaffolds_from_user_paths",
    "apply_user_scaffolds_to_string_graph",
    "validate_user_corrections",
    "export_scaffolds_to_tsv",
    "get_scaffold_statistics",
    "load_contig_breaks",
    "load_forced_joins",
    "load_exclusion_list",
    "apply_manual_edits",
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
