#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read Correction Module for StrandWeaver.

Consolidated modules for improved usability:
- errorsmith_module.py: Technology-aware error correction
- read_error_profiling_utility.py: Error profiling and k-mer analysis
- adaptive_kmer.py: AI-powered adaptive k-mer selection
- feature_extraction.py: Feature extraction for ML models

Main Components:
    - BaseCorrector: Abstract base class for all correctors
    - CorrectionStats: Statistics tracking for corrections
    - Correction Strategies: K-mer, quality-aware, consensus algorithms
    - Technology-specific correctors: ONT, Illumina, PacBio, Ancient DNA
    - Error profiling and pattern analysis
    - AI-powered k-mer selection
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

from .adaptive_kmer import (
    AdaptiveKmerSelector,
    AdaptiveKmerPredictor,
)

from .feature_extraction import (
    FeatureExtractor,
    ReadFeatures,
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
    
    # AI-powered modules
    'AdaptiveKmerSelector',
    'AdaptiveKmerPredictor',
    'FeatureExtractor',
    'ReadFeatures',
]
