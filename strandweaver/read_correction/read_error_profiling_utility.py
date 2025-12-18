#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read Error Profiling Utility for StrandWeaver.

Consolidated module for analyzing sequencing errors using k-mer analysis.
Provides detailed error profiles for technology-specific correction without
requiring a reference genome.

SECTIONS:
1. Error Type Definitions and Pattern Classification
2. K-mer Spectrum Analysis
3. Main Error Profiler
"""

# =============================================================================
# SECTION 1: ERROR TYPE DEFINITIONS AND PATTERN CLASSIFICATION
# =============================================================================
# Source: error_patterns.py

import json
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple, Set, TYPE_CHECKING
from collections import Counter, defaultdict
from datetime import datetime

from ..io import read_fastq, write_fastq, ReadCollection, ReadTechnology, SeqRead

if TYPE_CHECKING:
    from ..io import NanoporeMetadata


class ErrorType(Enum):
    """Types of sequencing errors."""
    SUBSTITUTION = "substitution"  # Base change (A->G, C->T, etc.)
    INSERTION = "insertion"        # Extra base inserted
    DELETION = "deletion"          # Base deleted
    HOMOPOLYMER = "homopolymer"    # Error in homopolymer run
    DAMAGE = "damage"              # Ancient DNA damage (C->T, G->A)
    UNKNOWN = "unknown"


@dataclass
class ErrorPattern:
    """
    Describes a specific error pattern.
    
    Attributes:
        error_type: Type of error
        from_base: Original base(s)
        to_base: Observed base(s)
        count: Number of times observed
        quality_scores: Quality scores when error observed
        positions: Positions in reads where observed (0-based)
    """
    error_type: ErrorType
    from_base: str
    to_base: str
    count: int = 0
    quality_scores: List[int] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    
    def add_observation(self, position: int, quality: Optional[int] = None):
        """Add an observation of this error pattern."""
        self.count += 1
        self.positions.append(position)
        if quality is not None:
            self.quality_scores.append(quality)
    
    def average_quality(self) -> float:
        """Get average quality score for this error."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)
    
    def position_bias(self) -> Dict[str, float]:
        """
        Calculate position bias (5' vs 3' end).
        
        Returns:
            Dictionary with 'five_prime', 'middle', 'three_prime' fractions
        """
        if not self.positions:
            return {'five_prime': 0.0, 'middle': 0.0, 'three_prime': 0.0}
        
        # Assume positions are normalized to [0, 1] or use actual positions
        five_prime = sum(1 for p in self.positions if p < 5)
        three_prime = sum(1 for p in self.positions if p > len(self.positions) - 5)
        middle = len(self.positions) - five_prime - three_prime
        
        total = len(self.positions)
        return {
            'five_prime': five_prime / total,
            'middle': middle / total,
            'three_prime': three_prime / total
        }


@dataclass
class PositionalErrorProfile:
    """
    Error profile with position-specific information.
    
    Useful for identifying position-dependent errors like ancient DNA damage.
    """
    # Position -> error type -> count
    position_errors: Dict[int, Dict[ErrorType, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    # Total observations per position
    position_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_error(self, position: int, error_type: ErrorType):
        """Record an error at a specific position."""
        self.position_errors[position][error_type] += 1
        self.position_counts[position] += 1
    
    def get_error_rate(self, position: int, error_type: Optional[ErrorType] = None) -> float:
        """
        Get error rate at a specific position.
        
        Args:
            position: Position in read (0-based)
            error_type: Specific error type (None = all errors)
        
        Returns:
            Error rate (0-1)
        """
        if position not in self.position_counts or self.position_counts[position] == 0:
            return 0.0
        
        if error_type is None:
            # Total error rate at position
            total_errors = sum(self.position_errors[position].values())
        else:
            total_errors = self.position_errors[position].get(error_type, 0)
        
        return total_errors / self.position_counts[position]
    
    def get_damage_profile(self, window_size: int = 10) -> Dict[int, float]:
        """
        Get ancient DNA damage profile (C->T at 5' end).
        
        Args:
            window_size: Number of positions to consider from 5' end
        
        Returns:
            Dictionary mapping position to C->T damage rate
        """
        damage_profile = {}
        
        for pos in range(window_size):
            if pos in self.position_counts:
                damage_rate = self.get_error_rate(pos, ErrorType.DAMAGE)
                damage_profile[pos] = damage_rate
            else:
                damage_profile[pos] = 0.0
        
        return damage_profile
    
    def to_dict(self) -> Dict:
        """Export to dictionary format."""
        return {
            'position_errors': {
                pos: {et.value: count for et, count in errors.items()}
                for pos, errors in self.position_errors.items()
            },
            'position_counts': dict(self.position_counts),
            'damage_profile': self.get_damage_profile()
        }


class ErrorPatternAnalyzer:
    """Analyzes error patterns from aligned sequences or k-mer analysis."""
    
    def __init__(self):
        """Initialize error pattern analyzer."""
        # Error type -> (from_base, to_base) -> ErrorPattern
        self.patterns: Dict[ErrorType, Dict[tuple, ErrorPattern]] = defaultdict(dict)
        self.positional_profile = PositionalErrorProfile()
    
    def record_substitution(self, from_base: str, to_base: str, 
                          position: int, quality: Optional[int] = None):
        """Record a substitution error."""
        # Check if it's ancient DNA damage
        is_damage = (from_base == 'C' and to_base == 'T') or (from_base == 'G' and to_base == 'A')
        error_type = ErrorType.DAMAGE if is_damage and position < 10 else ErrorType.SUBSTITUTION
        
        key = (from_base, to_base)
        if key not in self.patterns[error_type]:
            self.patterns[error_type][key] = ErrorPattern(
                error_type=error_type,
                from_base=from_base,
                to_base=to_base
            )
        
        self.patterns[error_type][key].add_observation(position, quality)
        self.positional_profile.add_error(position, error_type)
    
    def record_insertion(self, inserted_base: str, position: int, 
                        quality: Optional[int] = None, context: Optional[str] = None):
        """Record an insertion error."""
        # Check if it's in a homopolymer run
        is_homopolymer = context and len(set(context)) == 1
        error_type = ErrorType.HOMOPOLYMER if is_homopolymer else ErrorType.INSERTION
        
        key = ("", inserted_base)
        if key not in self.patterns[error_type]:
            self.patterns[error_type][key] = ErrorPattern(
                error_type=error_type,
                from_base="",
                to_base=inserted_base
            )
        
        self.patterns[error_type][key].add_observation(position, quality)
        self.positional_profile.add_error(position, error_type)
    
    def record_deletion(self, deleted_base: str, position: int, 
                       quality: Optional[int] = None, context: Optional[str] = None):
        """Record a deletion error."""
        # Check if it's in a homopolymer run
        is_homopolymer = context and len(set(context)) == 1
        error_type = ErrorType.HOMOPOLYMER if is_homopolymer else ErrorType.DELETION
        
        key = (deleted_base, "")
        if key not in self.patterns[error_type]:
            self.patterns[error_type][key] = ErrorPattern(
                error_type=error_type,
                from_base=deleted_base,
                to_base=""
            )
        
        self.patterns[error_type][key].add_observation(position, quality)
        self.positional_profile.add_error(position, error_type)
    
    def get_all_patterns(self) -> List[ErrorPattern]:
        """Get all recorded error patterns."""
        all_patterns = []
        for error_type_patterns in self.patterns.values():
            all_patterns.extend(error_type_patterns.values())
        return all_patterns
    
    def get_patterns_by_type(self, error_type: ErrorType) -> List[ErrorPattern]:
        """Get patterns for a specific error type."""
        if error_type not in self.patterns:
            return []
        return list(self.patterns[error_type].values())
    
    def get_most_common_errors(self, n: int = 10) -> List[ErrorPattern]:
        """Get the N most common error patterns."""
        all_patterns = self.get_all_patterns()
        return sorted(all_patterns, key=lambda p: p.count, reverse=True)[:n]
    
    def to_dict(self) -> Dict:
        """Export analysis to dictionary."""
        return {
            'patterns': {
                error_type.value: {
                    f"{pattern.from_base}->{pattern.to_base}": {
                        'count': pattern.count,
                        'average_quality': pattern.average_quality(),
                        'position_bias': pattern.position_bias()
                    }
                    for pattern in patterns.values()
                }
                for error_type, patterns in self.patterns.items()
            },
            'positional_profile': self.positional_profile.to_dict()
        }


# =============================================================================
# SECTION 2: K-MER SPECTRUM ANALYSIS
# =============================================================================
# Source: kmer_analysis.py

@dataclass
class KmerSpectrum:
    """
    K-mer frequency spectrum.
    
    Attributes:
        k: K-mer size
        kmer_counts: Dictionary mapping k-mer to count
        histogram: Coverage histogram (coverage -> number of k-mers)
        total_kmers: Total number of k-mers observed
        unique_kmers: Number of unique k-mers
    """
    k: int
    kmer_counts: Dict[str, int] = field(default_factory=dict)
    histogram: Dict[int, int] = field(default_factory=dict)
    total_kmers: int = 0
    unique_kmers: int = 0
    
    # Automatically determined thresholds
    error_threshold: int = 0  # K-mers below this are likely errors
    solid_threshold: int = 0  # K-mers above this are likely correct
    estimated_coverage: float = 0.0  # Peak coverage
    
    def add_kmer(self, kmer: str):
        """Add a k-mer observation."""
        kmer = kmer.upper()
        if len(kmer) != self.k:
            return
        
        # Use canonical k-mer (lexicographically smaller of forward/reverse)
        canonical = self.get_canonical(kmer)
        
        self.kmer_counts[canonical] = self.kmer_counts.get(canonical, 0) + 1
        self.total_kmers += 1
    
    def build_histogram(self):
        """Build coverage histogram from k-mer counts."""
        self.histogram = Counter(self.kmer_counts.values())
        self.unique_kmers = len(self.kmer_counts)
    
    def find_thresholds(self):
        """
        Automatically determine error and solid k-mer thresholds.
        
        Uses k-mer spectrum to identify:
        - Error threshold: Valley between error peak (low coverage) and genomic peak
        - Solid threshold: Start of genomic peak (high coverage k-mers)
        - Estimated coverage: Mode of genomic peak
        """
        if not self.histogram:
            self.build_histogram()
        
        # Get coverage values sorted
        coverages = sorted(self.histogram.keys())
        if len(coverages) < 2:
            # Not enough data
            self.error_threshold = 1
            self.solid_threshold = 2
            self.estimated_coverage = max(coverages) if coverages else 1.0
            return
        
        # Find peaks using simple derivative
        counts = [self.histogram[cov] for cov in coverages]
        
        # Find first minimum (valley between error and genomic peaks)
        # This is typically around 3-5x for most datasets
        valley_idx = 0
        for i in range(1, len(counts) - 1):
            if counts[i] < counts[i-1] and counts[i] < counts[i+1]:
                valley_idx = i
                break
        
        if valley_idx == 0:
            # No clear valley, use heuristic
            valley_idx = min(len(counts) // 3, 5)
        
        self.error_threshold = coverages[valley_idx]
        
        # Find genomic peak (maximum after valley)
        peak_idx = valley_idx
        max_count = counts[valley_idx]
        for i in range(valley_idx, len(counts)):
            if counts[i] > max_count:
                max_count = counts[i]
                peak_idx = i
        
        self.estimated_coverage = coverages[peak_idx]
        
        # Solid threshold is typically half the peak coverage
        self.solid_threshold = max(self.error_threshold + 1, int(self.estimated_coverage * 0.5))
    
    def is_error_kmer(self, kmer: str) -> bool:
        """Check if k-mer is likely an error (low coverage)."""
        canonical = self.get_canonical(kmer)
        count = self.kmer_counts.get(canonical, 0)
        return count > 0 and count < self.error_threshold
    
    def is_solid_kmer(self, kmer: str) -> bool:
        """Check if k-mer is solid (high coverage, likely correct)."""
        canonical = self.get_canonical(kmer)
        count = self.kmer_counts.get(canonical, 0)
        return count >= self.solid_threshold
    
    def get_kmer_count(self, kmer: str) -> int:
        """Get count for a k-mer."""
        canonical = self.get_canonical(kmer)
        return self.kmer_counts.get(canonical, 0)
    
    @staticmethod
    def get_canonical(kmer: str) -> str:
        """
        Get canonical k-mer (lexicographically smaller of forward/RC).
        
        This reduces memory by treating forward and reverse complement as same.
        """
        rc = KmerSpectrum.reverse_complement(kmer)
        return min(kmer, rc)
    
    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Get reverse complement of sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))
    
    def estimate_genome_size(self) -> int:
        """
        Estimate genome size from k-mer spectrum.
        
        Uses formula: genome_size ≈ total_unique_kmers / coverage
        """
        if self.estimated_coverage == 0:
            self.find_thresholds()
        
        if self.estimated_coverage == 0:
            return 0
        
        # Count k-mers in genomic peak (not errors)
        genomic_kmers = sum(
            count * self.histogram.get(count, 0)
            for count in self.histogram.keys()
            if count >= self.solid_threshold
        )
        
        return int(genomic_kmers / self.estimated_coverage)
    
    def to_dict(self) -> Dict:
        """Export spectrum to dictionary."""
        return {
            'k': self.k,
            'total_kmers': self.total_kmers,
            'unique_kmers': self.unique_kmers,
            'error_threshold': self.error_threshold,
            'solid_threshold': self.solid_threshold,
            'estimated_coverage': self.estimated_coverage,
            'estimated_genome_size': self.estimate_genome_size(),
            'histogram': dict(self.histogram)
        }


class KmerAnalyzer:
    """
    Analyzes k-mers for error detection and correction.
    
    This is the core of the error profiling system. It builds k-mer spectra,
    identifies erroneous k-mers, and estimates error rates.
    """
    
    def __init__(self, k: int = 21, solid_threshold: Optional[int] = None):
        """
        Initialize k-mer analyzer.
        
        Args:
            k: K-mer size (default 21, good for most genomes)
            solid_threshold: Manual solid k-mer threshold (None = auto-detect)
        """
        self.k = k
        self.spectrum = KmerSpectrum(k=k)
        self.manual_solid_threshold = solid_threshold
    
    def build_spectrum_from_reads(self, reads: Iterator, sample_size: Optional[int] = None):
        """
        Build k-mer spectrum from reads.
        
        Args:
            reads: Iterator of SeqRead objects
            sample_size: Maximum number of reads to process (None = all)
        """
        read_count = 0
        
        for read in reads:
            if sample_size and read_count >= sample_size:
                break
            
            # Extract all k-mers from read
            for kmer in self.extract_kmers(read.sequence):
                self.spectrum.add_kmer(kmer)
            
            read_count += 1
        
        # Build histogram and find thresholds
        self.spectrum.build_histogram()
        self.spectrum.find_thresholds()
        
        # Override with manual threshold if provided
        if self.manual_solid_threshold is not None:
            self.spectrum.solid_threshold = self.manual_solid_threshold
    
    def extract_kmers(self, sequence: str) -> List[str]:
        """Extract all k-mers from a sequence."""
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            # Skip k-mers with N
            if 'N' not in kmer:
                kmers.append(kmer)
        return kmers
    
    def identify_errors_in_read(self, sequence: str) -> List[int]:
        """
        Identify likely error positions in a read using k-mer coverage.
        
        A position is likely an error if:
        - K-mer containing it has low coverage (< error_threshold)
        - But k-mers with one substitution have high coverage (>= solid_threshold)
        
        Returns:
            List of error positions (0-based)
        """
        error_positions = []
        
        kmers = self.extract_kmers(sequence)
        
        for i, kmer in enumerate(kmers):
            kmer_count = self.spectrum.get_kmer_count(kmer)
            
            # If this k-mer has low coverage, it might contain an error
            if kmer_count < self.spectrum.error_threshold:
                # Check if any single-base correction creates a solid k-mer
                has_solid_neighbor = False
                
                for pos in range(len(kmer)):
                    for base in ['A', 'C', 'G', 'T']:
                        if base == kmer[pos]:
                            continue
                        
                        corrected = kmer[:pos] + base + kmer[pos+1:]
                        if self.spectrum.is_solid_kmer(corrected):
                            has_solid_neighbor = True
                            # Record error position in read
                            error_positions.append(i + pos)
                            break
                    
                    if has_solid_neighbor:
                        break
        
        # Remove duplicates and sort
        return sorted(set(error_positions))
    
    def estimate_error_rate(self, reads: Iterator, sample_size: int = 10000) -> float:
        """
        Estimate overall error rate using k-mer analysis.
        
        Args:
            reads: Iterator of SeqRead objects
            sample_size: Number of reads to sample
        
        Returns:
            Estimated error rate (0-1)
        """
        total_bases = 0
        total_errors = 0
        read_count = 0
        
        for read in reads:
            if read_count >= sample_size:
                break
            
            errors = self.identify_errors_in_read(read.sequence)
            total_errors += len(errors)
            total_bases += len(read.sequence)
            read_count += 1
        
        if total_bases == 0:
            return 0.0
        
        return total_errors / total_bases
    
    def get_error_kmer_neighbors(self, kmer: str) -> List[str]:
        """
        Get all single-substitution neighbors of a k-mer.
        
        Returns only neighbors that are solid k-mers.
        """
        solid_neighbors = []
        
        for i in range(len(kmer)):
            for base in ['A', 'C', 'G', 'T']:
                if base == kmer[i]:
                    continue
                
                neighbor = kmer[:i] + base + kmer[i+1:]
                if self.spectrum.is_solid_kmer(neighbor):
                    solid_neighbors.append(neighbor)
        
        return solid_neighbors
    
    def suggest_correction(self, kmer: str) -> Optional[str]:
        """
        Suggest correction for an error k-mer.
        
        Returns the solid k-mer neighbor with highest coverage,
        or None if no solid neighbors exist.
        """
        neighbors = self.get_error_kmer_neighbors(kmer)
        
        if not neighbors:
            return None
        
        # Return neighbor with highest coverage
        return max(neighbors, key=lambda k: self.spectrum.get_kmer_count(k))
    
    def to_dict(self) -> Dict:
        """Export analyzer state to dictionary."""
        return {
            'k': self.k,
            'spectrum': self.spectrum.to_dict()
        }


# =============================================================================
# SECTION 3: MAIN ERROR PROFILER
# =============================================================================
# Source: profiler.py

@dataclass
class ErrorProfile:
    """
    Complete error profile for a read dataset.
    
    Attributes:
        technology: Sequencing technology
        total_reads: Number of reads analyzed
        total_bases: Total bases analyzed
        error_rate: Overall error rate (0-1)
        kmer_spectrum: K-mer spectrum analysis
        error_patterns: Detailed error patterns
        positional_profile: Position-specific error rates
        quality_metrics: Quality score statistics
        ont_metadata: Optional Nanopore-specific metadata
        created_at: Timestamp of profile creation
    """
    technology: ReadTechnology
    total_reads: int = 0
    total_bases: int = 0
    error_rate: float = 0.0
    kmer_spectrum: Optional[Dict] = None
    error_patterns: Dict = field(default_factory=dict)
    positional_profile: Optional[Dict] = None
    quality_metrics: Dict = field(default_factory=dict)
    ont_metadata: Optional[Dict] = None  # Stored as dict for JSON serialization
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Export profile to dictionary."""
        result = {
            'technology': self.technology.value,
            'total_reads': self.total_reads,
            'total_bases': self.total_bases,
            'error_rate': self.error_rate,
            'kmer_spectrum': self.kmer_spectrum,
            'error_patterns': self.error_patterns,
            'positional_profile': self.positional_profile,
            'quality_metrics': self.quality_metrics,
            'created_at': self.created_at
        }
        
        # Include ONT metadata if present
        if self.ont_metadata:
            result['ont_metadata'] = self.ont_metadata
        
        return result
    
    def save(self, filepath: Path):
        """Save profile to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ErrorProfile':
        """Load profile from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert technology string back to enum
        data['technology'] = ReadTechnology(data['technology'])
        
        return cls(**data)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "="*70,
            "Error Profile Summary",
            "="*70,
            f"Technology: {self.technology.name}",
            f"Reads analyzed: {self.total_reads:,}",
            f"Total bases: {self.total_bases:,}",
            f"Estimated error rate: {self.error_rate:.4%}",
            ""
        ]
        
        # Add ONT metadata if present
        if self.ont_metadata:
            lines.extend([
                "Nanopore Metadata:",
                f"  Flow cell: {self.ont_metadata.get('flow_cell', 'N/A')}",
                f"  Basecaller: {self.ont_metadata.get('basecaller', 'N/A')}",
                f"  Accuracy: {self.ont_metadata.get('accuracy', 'N/A').upper()}",
                f"  Expected error rate: {self.ont_metadata.get('expected_error_rate', 0):.2%}",
                f"  Source: {self.ont_metadata.get('source', 'unknown')}",
                ""
            ])
        
        if self.kmer_spectrum:
            lines.extend([
                "K-mer Analysis:",
                f"  K-mer size: {self.kmer_spectrum.get('k', 'N/A')}",
                f"  Total k-mers: {self.kmer_spectrum.get('total_kmers', 0):,}",
                f"  Unique k-mers: {self.kmer_spectrum.get('unique_kmers', 0):,}",
                f"  Estimated coverage: {self.kmer_spectrum.get('estimated_coverage', 0):.1f}x",
                f"  Estimated genome size: {self.kmer_spectrum.get('estimated_genome_size', 0):,} bp",
                ""
            ])
        
        if self.quality_metrics:
            lines.extend([
                "Quality Metrics:",
                f"  Mean quality: {self.quality_metrics.get('mean_quality', 0):.1f}",
                f"  Median quality: {self.quality_metrics.get('median_quality', 0):.1f}",
                f"  Min quality: {self.quality_metrics.get('min_quality', 0):.1f}",
                f"  Max quality: {self.quality_metrics.get('max_quality', 0):.1f}",
                ""
            ])
        
        lines.append("="*70)
        
        return "\n".join(lines)


class ErrorProfiler:
    """
    Main error profiler for StrandWeaver.
    
    Uses k-mer analysis as the primary method for error detection,
    providing accurate error rate estimation without requiring a reference genome.
    """
    
    def __init__(self, 
                 k: int = 21,
                 sample_size: Optional[int] = None,
                 min_quality: int = 0,
                 enable_positional_analysis: bool = True):
        """
        Initialize error profiler.
        
        Args:
            k: K-mer size for analysis (default 21)
            sample_size: Number of reads to sample (None = all reads)
            min_quality: Minimum quality score to consider (for filtering)
            enable_positional_analysis: Whether to perform position-specific analysis
        """
        self.k = k
        self.sample_size = sample_size
        self.min_quality = min_quality
        self.enable_positional_analysis = enable_positional_analysis
    
    def profile(self, 
                reads_file: Path,
                technology: ReadTechnology,
                output_file: Optional[Path] = None,
                ont_metadata: Optional['NanoporeMetadata'] = None) -> ErrorProfile:
        """
        Profile errors in a read dataset.
        
        Args:
            reads_file: Path to input reads (FASTQ format)
            technology: Sequencing technology
            output_file: Optional path to save profile JSON
            ont_metadata: Optional Nanopore metadata for improved profiling
        
        Returns:
            ErrorProfile object
        """
        print(f"Profiling errors in {reads_file}")
        print(f"Technology: {technology.name}")
        print(f"K-mer size: {self.k}")
        
        # Display ONT metadata if provided
        if ont_metadata:
            print(f"ONT Metadata: {ont_metadata}")
        
        # Step 1: Build k-mer spectrum
        print("\nStep 1: Building k-mer spectrum...")
        kmer_analyzer = KmerAnalyzer(k=self.k)
        
        # Adjust k-mer threshold based on ONT metadata
        if ont_metadata:
            threshold_modifier = ont_metadata.get_kmer_threshold_modifier()
            kmer_analyzer.spectrum.error_threshold = int(
                kmer_analyzer.spectrum.error_threshold * threshold_modifier
            )
            print(f"  Adjusted k-mer threshold for {ont_metadata.accuracy.value.upper()} accuracy: {kmer_analyzer.spectrum.error_threshold}")
        
        reads_iter = read_fastq(reads_file, technology=technology, sample_size=self.sample_size)
        kmer_analyzer.build_spectrum_from_reads(reads_iter)
        
        print(f"  Total k-mers: {kmer_analyzer.spectrum.total_kmers:,}")
        print(f"  Unique k-mers: {kmer_analyzer.spectrum.unique_kmers:,}")
        print(f"  Estimated coverage: {kmer_analyzer.spectrum.estimated_coverage:.1f}x")
        print(f"  Error threshold: {kmer_analyzer.spectrum.error_threshold}")
        print(f"  Solid threshold: {kmer_analyzer.spectrum.solid_threshold}")
        
        # Step 2: Estimate error rate
        print("\nStep 2: Estimating error rate...")
        reads_iter = read_fastq(reads_file, technology=technology, sample_size=self.sample_size)
        error_rate = kmer_analyzer.estimate_error_rate(reads_iter, sample_size=min(10000, self.sample_size or 10000))
        
        print(f"  Estimated error rate: {error_rate:.4%}")
        
        # Step 3: Analyze error patterns
        print("\nStep 3: Analyzing error patterns...")
        pattern_analyzer = ErrorPatternAnalyzer()
        
        reads_iter = read_fastq(reads_file, technology=technology, sample_size=self.sample_size)
        total_reads, total_bases = self._analyze_patterns(
            reads_iter, 
            kmer_analyzer, 
            pattern_analyzer
        )
        
        print(f"  Reads analyzed: {total_reads:,}")
        print(f"  Total bases: {total_bases:,}")
        
        # Step 4: Calculate quality metrics
        print("\nStep 4: Calculating quality metrics...")
        reads_iter = read_fastq(reads_file, technology=technology, sample_size=self.sample_size)
        quality_metrics = self._calculate_quality_metrics(reads_iter)
        
        print(f"  Mean quality: {quality_metrics['mean_quality']:.1f}")
        print(f"  Median quality: {quality_metrics['median_quality']:.1f}")
        
        # Step 5: Build error profile
        print("\nStep 5: Building error profile...")
        
        # Convert ont_metadata to dict if present
        ont_metadata_dict = None
        if ont_metadata:
            ont_metadata_dict = ont_metadata.to_dict()
        
        profile = ErrorProfile(
            technology=technology,
            total_reads=total_reads,
            total_bases=total_bases,
            error_rate=error_rate,
            kmer_spectrum=kmer_analyzer.to_dict()['spectrum'],
            error_patterns=pattern_analyzer.to_dict(),
            positional_profile=pattern_analyzer.positional_profile.to_dict() if self.enable_positional_analysis else None,
            quality_metrics=quality_metrics,
            ont_metadata=ont_metadata_dict
        )
        
        # Save if output file specified
        if output_file:
            profile.save(output_file)
            print(f"\n✅ Profile saved to {output_file}")
        
        print("\n" + profile.summary())
        
        return profile
    
    def _analyze_patterns(self,
                         reads: Iterator[SeqRead],
                         kmer_analyzer: KmerAnalyzer,
                         pattern_analyzer: ErrorPatternAnalyzer) -> tuple:
        """
        Analyze error patterns in reads using k-mer analysis.
        
        Returns:
            Tuple of (total_reads, total_bases)
        """
        total_reads = 0
        total_bases = 0
        
        for read in reads:
            total_reads += 1
            total_bases += len(read.sequence)
            
            # Identify error positions using k-mers
            error_positions = kmer_analyzer.identify_errors_in_read(read.sequence)
            
            # Analyze each error
            for pos in error_positions:
                if pos >= len(read.sequence):
                    continue
                
                # Get k-mer containing this position
                kmer_start = max(0, pos - kmer_analyzer.k + 1)
                kmer_end = min(len(read.sequence), pos + kmer_analyzer.k)
                
                for k_pos in range(kmer_start, kmer_end - kmer_analyzer.k + 1):
                    kmer = read.sequence[k_pos:k_pos + kmer_analyzer.k]
                    
                    # Try to determine error type by finding correction
                    correction = kmer_analyzer.suggest_correction(kmer)
                    
                    if correction:
                        # Find the differing position
                        for i in range(len(kmer)):
                            if kmer[i] != correction[i]:
                                # This is a substitution error
                                quality = read.get_average_quality() if read.quality else None
                                pattern_analyzer.record_substitution(
                                    from_base=correction[i],  # True base
                                    to_base=kmer[i],          # Observed base
                                    position=pos,
                                    quality=quality
                                )
                                break
        
        return total_reads, total_bases
    
    def _calculate_quality_metrics(self, reads: Iterator[SeqRead]) -> Dict:
        """Calculate quality score statistics."""
        qualities = []
        
        for read in reads:
            if read.quality:
                avg_q = read.get_average_quality()
                if avg_q > 0:
                    qualities.append(avg_q)
        
        if not qualities:
            return {
                'mean_quality': 0.0,
                'median_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0
            }
        
        return {
            'mean_quality': float(np.mean(qualities)),
            'median_quality': float(np.median(qualities)),
            'min_quality': float(np.min(qualities)),
            'max_quality': float(np.max(qualities))
        }
    
    def profile_collection(self,
                          collection: ReadCollection,
                          output_dir: Path) -> Dict[ReadTechnology, ErrorProfile]:
        """
        Profile errors for all technologies in a read collection.
        
        Args:
            collection: ReadCollection with diverse read types
            output_dir: Directory to save individual profiles
        
        Returns:
            Dictionary mapping technology to ErrorProfile
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles = {}
        
        # Get statistics to see which technologies are present
        stats = collection.get_statistics()
        
        for tech_name, tech_stats in stats['by_technology'].items():
            try:
                technology = ReadTechnology(tech_name)
            except ValueError:
                continue
            
            print(f"\n{'='*70}")
            print(f"Profiling {technology.name} reads")
            print(f"{'='*70}")
            
            # Get reads for this technology
            reads = collection.get_reads_by_technology(technology)
            
            if not reads:
                print(f"No reads found for {technology.name}, skipping")
                continue
            
            # Create temporary FASTQ for profiling
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as tmp:
                temp_path = Path(tmp.name)
            
            try:
                # Write reads to temp file
                write_fastq(reads, temp_path)
                
                # Profile
                output_file = output_dir / f"{technology.value}_error_profile.json"
                profile = self.profile(temp_path, technology, output_file)
                profiles[technology] = profile
                
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
        
        return profiles
