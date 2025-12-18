"""
Read I/O module for StrandWeaver.

Handles reading and writing sequencing reads with support for multiple
technologies and automatic read type classification.

CONSOLIDATED MODULES:
- io_core_module.py: Core read structures, FASTQ/FASTA I/O, ReadCollection
- technology_handling_core_module.py: Technology types, validation, ONT metadata, LongBow
"""

# Core data structures from io_core_module
from .io_core_module import (
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

# Technology handling from technology_handling_core_module
from .technology_handling_core_module import (
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

__all__ = [
    # Core data structures
    "SeqRead",
    "ReadPair",
    "ReadCollection",
    
    # Technology types
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
]
