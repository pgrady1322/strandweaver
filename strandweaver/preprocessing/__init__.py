#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing Module for StrandWeaver.

Comprehensive preprocessing pipeline including error correction, profiling, 
read classification, and adaptive k-mer selection.

Consolidated modules:
- errorsmith_module.py: Technology-aware error correction
- read_error_profiling_utility.py: Error profiling and k-mer analysis
- kweaver_module.py: K-Weaver AI-powered adaptive k-mer selector
- read_classification.py: Read technology classification (moved from io_utils)

Main Components:
    - BaseCorrector: Abstract base class for all correctors
    - CorrectionStats: Statistics tracking for corrections
    - Correction Strategies: K-mer, quality-aware, consensus algorithms
    - Technology-specific correctors: ONT, Illumina, PacBio, Ancient DNA
    - Error profiling and pattern analysis
    - AI-powered k-mer selection (K-Weaver)
    - Read technology classification and metadata detection
    - Adapter detection and trimming
    - Visualization and reporting tools
"""

from .errorsmith_module import (
    # Base infrastructure
    BaseCorrector,
    get_corrector,
    CorrectionStats,
    
    # Correction strategies
    BloomFilter,
    KmerSpectrum,
    KmerCorrector,
    QualityAwareCorrector,
    ConsensusCorrector,
    
    # Technology-specific correctors
    HomopolymerDetector,
    ONTCorrector,
    PacBioCorrector,
    IlluminaCorrector,
    AncientDNACorrector,
    
    # Adapter detection
    AdapterMatch,
    AdapterDetector,
    detect_and_trim_adapters,
    ILLUMINA_ADAPTERS,
    
    # Visualization
    ErrorProfile,
    ErrorVisualizer,
    collect_kmer_spectrum,
)

from .read_error_profiling_utility import (
    # Error types and patterns
    ErrorType,
    ErrorPattern,
    PositionalErrorProfile,
    ErrorPatternAnalyzer,
    # K-mer analysis
    KmerAnalyzer,
    # Main profiler
    ErrorProfiler,
)

from .kweaver_module import (
    # Feature extraction
    FeatureExtractor,
    ReadFeatures,
    extract_features_from_file,
    # K-mer prediction
    KWeaverPredictor,
    KmerPrediction,
    # Backward compatibility
    AdaptiveKmerPredictor,
)

from .read_classification_utility import (
    # Technology types
    ReadTechnology,
    ReadRole,
    ReadTypeProfile,
    READ_PROFILES,
    # Classification
    classify_read_type,
    get_read_profile,
    infer_technology_from_length,
    parse_technology,
    validate_technology_specs,
    auto_detect_technology,
    detect_technologies,
    format_technology_summary,
    # Nanopore metadata
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
    # Base infrastructure
    'BaseCorrector',
    'get_corrector',
    'CorrectionStats',
    
    # Correction strategies
    'BloomFilter',
    'KmerSpectrum',
    'KmerCorrector',
    'QualityAwareCorrector',
    'ConsensusCorrector',
    
    # Technology-specific correctors
    'HomopolymerDetector',
    'ONTCorrector',
    'PacBioCorrector',
    'IlluminaCorrector',
    'AncientDNACorrector',
    
    # Adapter detection
    'AdapterMatch',
    'AdapterDetector',
    'detect_and_trim_adapters',
    'ILLUMINA_ADAPTERS',
    
    # Visualization
    'ErrorProfile',
    'ErrorVisualizer',
    'collect_kmer_spectrum',
    
    # Error profiling
    'ErrorType',
    'ErrorPattern',
    'PositionalErrorProfile',
    'ErrorPatternAnalyzer',
    'KmerAnalyzer',
    'ErrorProfiler',
    
    # AI-powered K-Weaver module
    'KWeaverPredictor',
    'KmerPrediction',
    'FeatureExtractor',
    'ReadFeatures',
    'extract_features_from_file',
    'AdaptiveKmerPredictor',  # Backward compatibility
    
    # Read technology classification (moved from io_utils)
    'ReadTechnology',
    'ReadRole',
    'ReadTypeProfile',
    'READ_PROFILES',
    'classify_read_type',
    'get_read_profile',
    'infer_technology_from_length',
    'parse_technology',
    'validate_technology_specs',
    'auto_detect_technology',
    'detect_technologies',
    'format_technology_summary',
    'FlowCellType',
    'BasecallerType',
    'BasecallerAccuracy',
    'NanoporeMetadata',
    'parse_nanopore_metadata',
    'is_longbow_available',
    'detect_nanopore_metadata_with_longbow',
    'detect_from_fastq_headers',
]
