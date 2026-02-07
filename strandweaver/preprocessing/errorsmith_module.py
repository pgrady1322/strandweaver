"""
StrandWeaver v0.1.0

ErrorSmith: Unified Error Correction Module for StrandWeaver

This module consolidates all error correction functionality into a single,
comprehensive module for improved usability and clarity. ErrorSmith provides
technology-specific error correction strategies for multiple sequencing platforms.

Consolidated from 7 separate modules:
- corrector.py: Base correction infrastructure and factory functions
- stats.py: Correction statistics tracking
- strategies.py: Core correction algorithms (k-mer, quality-aware, consensus)
- tech_correctors.py: Technology-specific correctors (ONT, PacBio, Illumina, Ancient DNA)
- adapters.py: Adapter detection and trimming
- visualizations.py: Error correction visualization and reporting
- __init__.py: Module exports

Architecture:
    Section 1: Base Corrector Infrastructure
    Section 2: Correction Statistics
    Section 3: Correction Strategies (Bloom Filter, K-mer, Quality-aware, Consensus)
    Section 4: Technology-Specific Correctors
    Section 5: Adapter Detection and Trimming
    Section 6: Visualization and Reporting

Usage:
    from strandweaver.preprocessing import get_corrector, CorrectionStats
    
    # Create technology-specific corrector
    corrector = get_corrector("ont", error_profile, k_size=21)
    
    # Correct reads
    corrected_read = corrector.correct_read(read)
    
    # Batch correction with statistics
    stats = corrector.correct_reads("input.fastq", "output.fastq")
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from abc import ABC, abstractmethod
import re
import math
import hashlib
import os
import logging

logger = logging.getLogger(__name__)

# Third-party visualization imports (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# StrandWeaver imports
from ..io_utils import SeqRead, read_fastq, write_fastq
from .read_classification_utility import ReadTechnology


# Set publication-quality defaults if available
if VISUALIZATION_AVAILABLE:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)


# ============================================================================
# SECTION 1: BASE CORRECTOR INFRASTRUCTURE
# ============================================================================
# Core abstract base class for all correctors and factory function


class BaseCorrector(ABC):
    """
    Abstract base class for all error correctors.
    
    Provides common interface and functionality for technology-specific
    correctors. Subclasses must implement correct_read() method.
    """
    
    def __init__(
        self,
        error_profile: Optional[Dict[str, Any]] = None,
        collect_viz_data: bool = False
    ):
        """
        Initialize base corrector.
        
        Args:
            error_profile: Optional error profile for guided correction
            collect_viz_data: Whether to collect visualization data
        """
        self.error_profile = error_profile
        self.collect_viz_data = collect_viz_data
        self.stats = CorrectionStats()
        
        # Visualization data collection
        if collect_viz_data:
            self.viz_data = ErrorProfile()
        else:
            self.viz_data = None
    
    @abstractmethod
    def correct_read(self, read: SeqRead) -> SeqRead:
        """
        Correct a single read.
        
        Args:
            read: SeqRead to correct
            
        Returns:
            Corrected SeqRead
        """
        pass
    
    def correct_reads(
        self,
        input_file: str,
        output_file: str,
        max_reads: Optional[int] = None,
        verbose: bool = True
    ) -> CorrectionStats:
        """
        Correct reads from input file and write to output file.
        
        Args:
            input_file: Path to input FASTQ file
            output_file: Path to output FASTQ file
            max_reads: Maximum number of reads to process (None = all)
            verbose: Print progress information
            
        Returns:
            CorrectionStats with summary of corrections
        """
        reads_processed = 0
        corrected_reads = []
        
        # Read input
        if verbose:
            print(f"Reading reads from {input_file}...")
        
        for read in read_fastq(input_file):
            if max_reads and reads_processed >= max_reads:
                break
            
            # Track quality before
            if read.quality and self.viz_data:
                self.viz_data.quality_before.extend(read.quality)
            
            # Correct read
            corrected = self.correct_read(read)
            
            # Track quality after
            if corrected.quality and self.viz_data:
                self.viz_data.quality_after.extend(corrected.quality)
            
            # Collect visualization data if requested
            if self.collect_viz_data and self.viz_data:
                self._collect_read_visualization_data(read, corrected)
            
            corrected_reads.append(corrected)
            reads_processed += 1
            
            if verbose and reads_processed % 10000 == 0:
                print(f"  Processed {reads_processed:,} reads...")
        
        # Write output
        if verbose:
            print(f"Writing corrected reads to {output_file}...")
        
        write_fastq(corrected_reads, output_file)
        
        if verbose:
            print(f"✓ Completed: {reads_processed:,} reads corrected")
            print(self.stats.summary())
        
        return self.stats
    
    def _collect_read_visualization_data(
        self,
        original: SeqRead,
        corrected: SeqRead
    ):
        """
        Collect visualization data from read correction.
        
        Args:
            original: Original read
            corrected: Corrected read
        """
        if not self.viz_data:
            return
        
        # Update counters
        self.viz_data.total_reads += 1
        self.viz_data.total_bases += len(original.sequence)
        
        # Track read length
        self.viz_data.read_lengths.append(len(original.sequence))
        
        # Calculate GC content
        gc_count = original.sequence.count('G') + original.sequence.count('C')
        gc_content = gc_count / len(original.sequence) if len(original.sequence) > 0 else 0
        
        # Compare sequences to find corrections
        was_corrected = original.sequence != corrected.sequence
        
        if was_corrected:
            self.viz_data.gc_content_errors.append(gc_content)
            self.viz_data.total_corrections += 1
        else:
            self.viz_data.gc_content_correct.append(gc_content)


def get_corrector(
    technology: str,
    error_profile: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseCorrector:
    """
    Factory function to get appropriate corrector for technology.
    
    Args:
        technology: Technology type ("ont", "pacbio", "illumina", "ancient_dna")
        error_profile: Optional error profile for guided correction
        **kwargs: Additional arguments passed to corrector constructor
        
    Returns:
        Technology-specific corrector instance
        
    Raises:
        ValueError: If technology not recognized
    """
    tech_lower = technology.lower()
    
    # Map technology strings to corrector classes
    if tech_lower in ["ont", "ont_regular", "ont_ultralong", "nanopore"]:
        return ONTCorrector(error_profile=error_profile, **kwargs)
    
    elif tech_lower in ["pacbio", "pacbio_hifi", "hifi"]:
        return PacBioCorrector(error_profile=error_profile, **kwargs)
    
    elif tech_lower in ["illumina", "paired_end", "single_end"]:
        return IlluminaCorrector(error_profile=error_profile, **kwargs)
    
    elif tech_lower in ["ancient_dna", "ancient", "adna"]:
        return AncientDNACorrector(error_profile=error_profile, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown technology: {technology}. "
            f"Supported: ont, pacbio, illumina, ancient_dna"
        )


# ============================================================================
# SECTION 2: CORRECTION STATISTICS
# ============================================================================
# Statistics tracking for correction operations


@dataclass
class CorrectionStats:
    """
    Statistics for error correction operations.
    
    Tracks the number and types of corrections made, quality improvements,
    and other metrics useful for evaluating correction performance.
    """
    
    # Read-level statistics
    reads_processed: int = 0
    reads_corrected: int = 0  # Reads with at least one correction
    
    # Base-level statistics
    bases_corrected: int = 0
    total_bases: int = 0
    
    # Correction types
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    
    # Position tracking
    corrections_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Quality tracking
    quality_before_mean: Optional[float] = None
    quality_before_median: Optional[float] = None
    quality_after_mean: Optional[float] = None
    quality_after_median: Optional[float] = None
    
    # Temporary quality accumulators
    _quality_before: List[float] = field(default_factory=list, repr=False)
    _quality_after: List[float] = field(default_factory=list, repr=False)
    
    def record_correction(self, correction_type: str, position: int):
        """
        Record a single correction.
        
        Args:
            correction_type: Type of correction ('substitution', 'insertion', 'deletion')
            position: Position in read where correction was made
        """
        self.bases_corrected += 1
        self.corrections_by_position[position] += 1
        
        if correction_type == 'substitution':
            self.substitutions += 1
        elif correction_type == 'insertion':
            self.insertions += 1
        elif correction_type == 'deletion':
            self.deletions += 1
    
    def record_read(
        self,
        was_corrected: bool,
        num_bases: int,
        quality_before: Optional[List[int]] = None,
        quality_after: Optional[List[int]] = None
    ):
        """
        Record processing of a read.
        
        Args:
            was_corrected: Whether the read had any corrections
            num_bases: Number of bases in the read
            quality_before: Quality scores before correction
            quality_after: Quality scores after correction
        """
        self.reads_processed += 1
        self.total_bases += num_bases
        
        if was_corrected:
            self.reads_corrected += 1
        
        # Track quality scores
        if quality_before:
            self._quality_before.extend(quality_before)
        
        if quality_after:
            self._quality_after.extend(quality_after)
    
    def finalize(self):
        """Compute final statistics from accumulated data."""
        if self._quality_before:
            self.quality_before_mean = sum(self._quality_before) / len(self._quality_before)
            sorted_before = sorted(self._quality_before)
            self.quality_before_median = sorted_before[len(sorted_before) // 2]
        
        if self._quality_after:
            self.quality_after_mean = sum(self._quality_after) / len(self._quality_after)
            sorted_after = sorted(self._quality_after)
            self.quality_after_median = sorted_after[len(sorted_after) // 2]
    
    def get_correction_rate(self) -> float:
        """
        Get the correction rate (percentage of bases corrected).
        
        Returns:
            Correction rate as percentage (0-100)
        """
        if self.total_bases == 0:
            return 0.0
        return (self.bases_corrected / self.total_bases) * 100
    
    def summary(self) -> str:
        """
        Get a human-readable summary of correction statistics.
        
        Returns:
            Multi-line string with summary
        """
        self.finalize()
        
        lines = [
            "\n" + "=" * 60,
            "CORRECTION STATISTICS SUMMARY",
            "=" * 60,
            f"Reads processed:       {self.reads_processed:,}",
            f"Reads corrected:       {self.reads_corrected:,} ({self.reads_corrected/self.reads_processed*100:.1f}%)" if self.reads_processed > 0 else "Reads corrected:       0",
            f"Total bases:           {self.total_bases:,}",
            f"Bases corrected:       {self.bases_corrected:,} ({self.get_correction_rate():.3f}%)",
            "",
            "Correction Types:",
            f"  Substitutions:       {self.substitutions:,}",
            f"  Insertions:          {self.insertions:,}",
            f"  Deletions:           {self.deletions:,}",
        ]
        
        if self.quality_before_mean is not None:
            lines.extend([
                "",
                "Quality Improvement:",
                f"  Mean before:         Q{self.quality_before_mean:.1f}",
                f"  Mean after:          Q{self.quality_after_mean:.1f}",
                f"  Improvement:         {self.quality_after_mean - self.quality_before_mean:+.1f}",
            ])
        
        lines.append("=" * 60 + "\n")
        
        return "\n".join(lines)


# ============================================================================
# SECTION 3: CORRECTION STRATEGIES
# ============================================================================
# Core correction algorithms: Bloom Filter, K-mer, Quality-aware, Consensus


class BloomFilter:
    """
    Space-efficient probabilistic data structure for k-mer membership testing.
    
    Uses multiple hash functions to achieve O(1) membership queries with
    configurable false positive rate. False negatives are impossible.
    
    Particularly useful for large k-mer sets where exact hash tables
    would consume too much memory.
    """
    
    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter.
        
        Args:
            expected_elements: Expected number of elements to store
            false_positive_rate: Desired false positive rate (0-1)
        """
        # Calculate optimal bit array size
        # m = -(n * ln(p)) / (ln(2)^2)
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate
        
        n = expected_elements
        p = false_positive_rate
        
        self.size = int(-(n * math.log(p)) / (math.log(2) ** 2))
        
        # Calculate optimal number of hash functions
        # k = (m / n) * ln(2)
        self.num_hashes = int((self.size / n) * math.log(2))
        self.num_hashes = max(1, self.num_hashes)  # At least 1 hash function
        
        # Bit array
        self.bit_array = [False] * self.size
        self.items_added = 0
    
    def _hash(self, item: str, seed: int) -> int:
        """
        Generate hash value for item with given seed.
        
        Args:
            item: Item to hash
            seed: Seed for hash function
            
        Returns:
            Hash value in range [0, size)
        """
        # Use MD5 with seed for hash function
        h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(h, 16) % self.size
    
    def add(self, item: str):
        """
        Add item to Bloom filter.
        
        Args:
            item: Item to add (typically a k-mer string)
        """
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            self.bit_array[idx] = True
        
        self.items_added += 1
    
    def contains(self, item: str) -> bool:
        """
        Check if item might be in the set.
        
        Args:
            item: Item to check
            
        Returns:
            True if item might be present (with possible false positive),
            False if item is definitely not present
        """
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            if not self.bit_array[idx]:
                return False  # Definitely not present
        
        return True  # Might be present (or false positive)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Bloom filter."""
        bits_set = sum(self.bit_array)
        fill_ratio = bits_set / self.size if self.size > 0 else 0
        
        # Calculate actual false positive rate
        k = self.num_hashes
        n = self.items_added
        m = self.size
        
        if m > 0 and n > 0:
            actual_fpr = (1 - math.exp(-k * n / m)) ** k
        else:
            actual_fpr = 0.0
        
        return {
            'size': self.size,
            'num_hashes': self.num_hashes,
            'items_added': self.items_added,
            'bits_set': bits_set,
            'fill_ratio': fill_ratio,
            'estimated_fpr': actual_fpr
        }


class KmerSpectrum:
    """
    K-mer spectrum for identifying solid (correct) vs error k-mers.
    
    Solid k-mers appear frequently in the read set and are likely correct.
    Error k-mers appear infrequently and likely contain sequencing errors.
    """
    
    def __init__(self, k_size: int = 21, min_freq: int = 2):
        """
        Initialize k-mer spectrum.
        
        Args:
            k_size: Length of k-mers
            min_freq: Minimum frequency for a k-mer to be considered solid
        """
        self.k_size = k_size
        self.min_freq = min_freq
        self.kmer_counts: Dict[str, int] = defaultdict(int)
        self.solid_kmers: Set[str] = set()
        self.bloom_filter: Optional[BloomFilter] = None
    
    def add_sequence(self, sequence: str):
        """
        Add a sequence to the k-mer spectrum.
        
        Args:
            sequence: DNA sequence to extract k-mers from
        """
        for i in range(len(sequence) - self.k_size + 1):
            kmer = sequence[i:i + self.k_size]
            if len(kmer) == self.k_size:  # Ensure full k-mer
                self.kmer_counts[kmer] += 1
    
    def build_solid_kmers(self, use_bloom: bool = True):
        """
        Identify solid k-mers based on frequency threshold.
        
        Args:
            use_bloom: Whether to build a Bloom filter for fast lookups
        """
        self.solid_kmers = {
            kmer for kmer, count in self.kmer_counts.items()
            if count >= self.min_freq
        }
        
        # Build Bloom filter for fast membership testing
        if use_bloom and len(self.solid_kmers) > 0:
            # Estimate false positive rate based on set size
            # For smaller sets, use lower FPR
            if len(self.solid_kmers) < 10000:
                fpr = 0.001  # 0.1% for small sets
            else:
                fpr = 0.01   # 1% for large sets
            
            self.bloom_filter = BloomFilter(
                expected_elements=len(self.solid_kmers),
                false_positive_rate=fpr
            )
            
            # Add all solid k-mers to Bloom filter
            for kmer in self.solid_kmers:
                self.bloom_filter.add(kmer)
    
    def is_solid(self, kmer: str) -> bool:
        """
        Check if a k-mer is solid (high frequency).
        
        Uses Bloom filter for fast pre-filtering before exact lookup.
        """
        # Fast Bloom filter check first (if available)
        if self.bloom_filter:
            if not self.bloom_filter.contains(kmer):
                # Definitely not solid
                return False
        
        # Exact lookup in set
        return kmer in self.solid_kmers
    
    def get_count(self, kmer: str) -> int:
        """Get count of a k-mer."""
        return self.kmer_counts.get(kmer, 0)


class KmerCorrector:
    """
    K-mer based error correction strategy.
    
    Uses k-mer spectrum to identify and correct errors. Error positions
    are identified by k-mers with low frequency, and corrections are
    attempted by replacing bases to form solid k-mers.
    """
    
    def __init__(
        self,
        k_size: int = 21,
        min_kmer_freq: int = 2,
        max_corrections: int = 10
    ):
        """
        Initialize k-mer corrector.
        
        Args:
            k_size: K-mer size
            min_kmer_freq: Minimum frequency for solid k-mer
            max_corrections: Maximum corrections per read
        """
        self.k_size = k_size
        self.min_kmer_freq = min_kmer_freq
        self.max_corrections = max_corrections
        self.spectrum = KmerSpectrum(k_size, min_kmer_freq)
    
    def build_spectrum(self, sequences: List[str]):
        """
        Build k-mer spectrum from sequences.
        
        Args:
            sequences: List of DNA sequences
        """
        for seq in sequences:
            self.spectrum.add_sequence(seq)
        
        self.spectrum.build_solid_kmers()
    
    def correct_sequence(
        self,
        sequence: str,
        quality: Optional[List[int]] = None
    ) -> Tuple[str, List[Tuple[int, str, str]]]:
        """
        Correct errors in a sequence using k-mer spectrum.
        
        Args:
            sequence: DNA sequence to correct
            quality: Optional quality scores (prioritize low quality bases)
            
        Returns:
            Tuple of (corrected_sequence, corrections_list)
            where corrections_list contains (position, original_base, new_base)
        """
        corrected = list(sequence)
        corrections = []
        
        # Find error positions (k-mers that are not solid)
        error_positions = self._find_error_positions(sequence)
        
        if quality:
            # Prioritize low quality positions
            error_positions = sorted(
                error_positions,
                key=lambda pos: quality[pos] if pos < len(quality) else 0
            )
        
        # Attempt corrections
        for pos in error_positions[:self.max_corrections]:
            correction = self._correct_position(corrected, pos)
            
            if correction:
                new_base = correction
                original_base = corrected[pos]
                corrected[pos] = new_base
                corrections.append((pos, original_base, new_base))
        
        return ''.join(corrected), corrections
    
    def _find_error_positions(self, sequence: str) -> List[int]:
        """
        Find positions likely to contain errors based on k-mer spectrum.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            List of positions that may contain errors
        """
        error_positions = set()
        
        for i in range(len(sequence) - self.k_size + 1):
            kmer = sequence[i:i + self.k_size]
            
            if not self.spectrum.is_solid(kmer):
                # This k-mer is not solid, mark all positions in it
                for j in range(i, i + self.k_size):
                    error_positions.add(j)
        
        return sorted(list(error_positions))
    
    def _correct_position(self, sequence: List[str], pos: int) -> Optional[str]:
        """
        Try to correct a single position.
        
        Args:
            sequence: Current sequence (as list of characters)
            pos: Position to correct
            
        Returns:
            Corrected base if correction found, None otherwise
        """
        original_base = sequence[pos]
        bases = ['A', 'C', 'G', 'T']
        
        best_base = None
        best_score = -1
        
        for base in bases:
            if base == original_base:
                continue
            
            # Try this substitution
            sequence[pos] = base
            
            # Count how many solid k-mers we create
            score = 0
            for i in range(max(0, pos - self.k_size + 1), min(len(sequence) - self.k_size + 1, pos + 1)):
                kmer = ''.join(sequence[i:i + self.k_size])
                if len(kmer) == self.k_size and self.spectrum.is_solid(kmer):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_base = base
        
        # Restore original
        sequence[pos] = original_base
        
        # Only return correction if it creates at least one solid k-mer
        if best_score > 0:
            return best_base
        
        return None


class QualityAwareCorrector:
    """
    Quality-score aware correction strategy.
    
    Uses quality scores to guide correction decisions. Low quality bases
    are more likely to be errors and are corrected more aggressively.
    """
    
    def __init__(self, min_quality: int = 10, quality_threshold: int = 20):
        """
        Initialize quality-aware corrector.
        
        Args:
            min_quality: Minimum quality to keep a base
            quality_threshold: Quality threshold for aggressive correction
        """
        self.min_quality = min_quality
        self.quality_threshold = quality_threshold
    
    def should_correct(self, quality: int, evidence_strength: float) -> bool:
        """
        Determine if a position should be corrected based on quality.
        
        Args:
            quality: Quality score of the base
            evidence_strength: Strength of correction evidence (0-1)
            
        Returns:
            True if correction should be made
        """
        # Very low quality: correct with weak evidence
        if quality < self.min_quality:
            return evidence_strength > 0.3
        
        # Medium quality: need moderate evidence
        if quality < self.quality_threshold:
            return evidence_strength > 0.6
        
        # High quality: need strong evidence
        return evidence_strength > 0.9
    
    def adjust_quality(self, original_quality: int, corrected: bool) -> int:
        """
        Adjust quality score after correction.
        
        Args:
            original_quality: Original quality score
            corrected: Whether base was corrected
            
        Returns:
            Adjusted quality score
        """
        if corrected:
            # Corrected bases get moderate quality
            return max(self.quality_threshold, min(original_quality + 10, 40))
        else:
            # Uncorrected bases keep original quality
            return original_quality


class ConsensusCorrector:
    """
    Consensus-based correction strategy.
    
    Uses multiple overlapping reads to build consensus and correct errors.
    This is particularly useful for high-coverage datasets.
    """
    
    def __init__(self, min_coverage: int = 3, consensus_threshold: float = 0.6):
        """
        Initialize consensus corrector.
        
        Args:
            min_coverage: Minimum coverage required for consensus
            consensus_threshold: Fraction of reads that must agree
        """
        self.min_coverage = min_coverage
        self.consensus_threshold = consensus_threshold
    
    def build_consensus(
        self,
        pileup: List[Tuple[str, int]]
    ) -> Tuple[str, float]:
        """
        Build consensus from pileup of bases and qualities.
        
        Args:
            pileup: List of (base, quality) tuples at a position
            
        Returns:
            Tuple of (consensus_base, confidence)
        """
        if not pileup:
            return 'N', 0.0
        
        # Count bases weighted by quality
        base_scores = defaultdict(float)
        total_score = 0.0
        
        for base, quality in pileup:
            score = 10 ** (quality / 10)  # Phred score to probability
            base_scores[base] += score
            total_score += score
        
        # Find most common base
        best_base = max(base_scores.items(), key=lambda x: x[1])
        consensus = best_base[0]
        confidence = best_base[1] / total_score if total_score > 0 else 0.0
        
        return consensus, confidence
    
    def correct_with_consensus(
        self,
        sequence: str,
        quality: List[int],
        overlapping_reads: List[Tuple[str, List[int], int]]
    ) -> Tuple[str, List[int]]:
        """
        Correct sequence using consensus from overlapping reads.
        
        Args:
            sequence: Sequence to correct
            quality: Quality scores
            overlapping_reads: List of (seq, qual, offset) for overlapping reads
            
        Returns:
            Tuple of (corrected_sequence, corrected_quality)
        """
        corrected = list(sequence)
        corrected_qual = list(quality)
        
        for i in range(len(sequence)):
            # Build pileup at this position
            pileup = [(sequence[i], quality[i])]
            
            for other_seq, other_qual, offset in overlapping_reads:
                other_pos = i - offset
                if 0 <= other_pos < len(other_seq):
                    pileup.append((other_seq[other_pos], other_qual[other_pos]))
            
            # Get consensus if we have enough coverage
            if len(pileup) >= self.min_coverage:
                consensus, confidence = self.build_consensus(pileup)
                
                if confidence >= self.consensus_threshold:
                    corrected[i] = consensus
                    # Adjust quality based on consensus confidence
                    corrected_qual[i] = int(min(40, confidence * 40))
        
        return ''.join(corrected), corrected_qual


# ============================================================================
# SECTION 4: TECHNOLOGY-SPECIFIC CORRECTORS
# ============================================================================
# ONT, PacBio, Illumina, and Ancient DNA correctors


class HomopolymerDetector:
    """
    Detect and analyze homopolymer runs in sequences.
    
    Homopolymers are the primary error source in ONT sequencing.
    """
    
    def __init__(self, min_length: int = 3):
        """
        Initialize homopolymer detector.
        
        Args:
            min_length: Minimum length to consider a homopolymer
        """
        self.min_length = min_length
    
    def find_homopolymers(self, sequence: str) -> List[Tuple[int, int, str, int]]:
        """
        Find all homopolymer runs in a sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            List of (start_pos, end_pos, base, length) tuples
        """
        homopolymers = []
        i = 0
        
        while i < len(sequence):
            base = sequence[i]
            run_start = i
            run_length = 1
            
            # Extend the run
            while i + 1 < len(sequence) and sequence[i + 1] == base:
                run_length += 1
                i += 1
            
            # Record if meets minimum length
            if run_length >= self.min_length:
                homopolymers.append((run_start, i + 1, base, run_length))
            
            i += 1
        
        return homopolymers
    
    def is_homopolymer_region(self, sequence: str, start: int, end: int) -> bool:
        """
        Check if a region is a homopolymer.
        
        Args:
            sequence: DNA sequence
            start: Start position
            end: End position
            
        Returns:
            True if region is homopolymer
        """
        if end - start < self.min_length:
            return False
        
        region = sequence[start:end]
        return len(set(region)) == 1


class ONTCorrector(BaseCorrector):
    """
    ONT-specific error corrector.
    
    Leverages ONT metadata (flow cell type, basecaller, accuracy mode) to
    apply technology-specific correction strategies with appropriate parameters.
    
    Key features:
    - Homopolymer compression/expansion correction
    - Flow cell-aware error rate expectations
    - Basecaller model integration (Guppy/Dorado HAC/SUP)
    - Quality score recalibration
    
    Error characteristics:
    - Indel-biased (especially in homopolymers)
    - ~5-15% raw error rate (R9), ~3-8% (R10)
    - Systematic homopolymer errors
    - Context-dependent error patterns
    """
    
    def __init__(
        self,
        error_profile: Optional[Dict[str, Any]] = None,
        k_size: int = 21,
        min_kmer_freq: int = 3,
        max_corrections: int = 10,
        homopolymer_correction: bool = True,
        collect_viz_data: bool = False
    ):
        """
        Initialize ONT corrector.
        
        Args:
            error_profile: Optional error profile for guided correction
            k_size: K-mer size for correction
            min_kmer_freq: Minimum k-mer frequency to be considered solid
            max_corrections: Maximum corrections per read
            homopolymer_correction: Enable homopolymer-specific correction
            collect_viz_data: Whether to collect visualization data
        """
        super().__init__(error_profile, collect_viz_data)
        
        self.k_size = k_size
        self.min_kmer_freq = min_kmer_freq
        self.max_corrections = max_corrections
        self.homopolymer_correction = homopolymer_correction
        
        # Initialize k-mer corrector
        self.kmer_corrector = KmerCorrector(
            k_size=k_size,
            min_kmer_freq=min_kmer_freq,
            max_corrections=max_corrections
        )
        
        # Initialize homopolymer detector
        if homopolymer_correction:
            self.homopolymer_detector = HomopolymerDetector(min_length=3)
        else:
            self.homopolymer_detector = None
    
    def correct_read(self, read: SeqRead) -> SeqRead:
        """
        Correct a single ONT read.
        
        Args:
            read: SeqRead to correct
            
        Returns:
            Corrected SeqRead
        """
        sequence = read.sequence
        quality = read.quality
        
        # Find homopolymer regions if enabled
        homopolymer_regions = set()
        if self.homopolymer_correction and self.homopolymer_detector:
            homopolymers = self.homopolymer_detector.find_homopolymers(sequence)
            for start, end, base, length in homopolymers:
                homopolymer_regions.update(range(start, end))
                
                # Track homopolymer stats if collecting viz data
                if self.viz_data:
                    self.viz_data.homopolymer_lengths.append(length)
        
        # K-mer based correction
        corrected_seq, corrections = self.kmer_corrector.correct_sequence(
            sequence, quality
        )
        
        # Update statistics
        was_corrected = corrected_seq != sequence
        self.stats.record_read(
            was_corrected=was_corrected,
            num_bases=len(sequence),
            quality_before=quality,
            quality_after=quality  # Quality unchanged for now
        )
        
        for pos, old_base, new_base in corrections:
            if old_base != new_base:
                # Determine correction type
                if len(old_base) == len(new_base) == 1:
                    self.stats.record_correction('substitution', pos)
                elif len(old_base) < len(new_base):
                    self.stats.record_correction('insertion', pos)
                else:
                    self.stats.record_correction('deletion', pos)
        
        # Create corrected read
        return SeqRead(
            id=read.id,
            sequence=corrected_seq,
            quality=quality,
            technology=read.technology,
            metadata=read.metadata
        )


class PacBioCorrector(BaseCorrector):
    """
    PacBio HiFi-specific error corrector.
    
    PacBio HiFi reads are already high-quality (Q30+, 99.9% accuracy),
    so correction must be conservative to avoid introducing errors.
    
    Key features:
    - Conservative correction (avoid over-correction)
    - Quality-weighted scoring (trust Q30+ bases)
    - Early stopping to preserve high-quality regions
    - Chemistry-aware parameters
    
    Error characteristics:
    - Very low error rate (0.5-1%)
    - Indel-biased but less extreme than ONT
    - No systematic homopolymer errors (CCS corrects them)
    - Uniform error distribution
    """
    
    def __init__(
        self,
        error_profile: Optional[Dict[str, Any]] = None,
        k_size: int = 31,
        min_kmer_freq: int = 3,
        max_corrections: int = 5,
        min_quality_threshold: int = 30,
        collect_viz_data: bool = False
    ):
        """
        Initialize PacBio corrector.
        
        Args:
            error_profile: Optional error profile for guided correction
            k_size: K-mer size (larger for HiFi due to longer reads)
            min_kmer_freq: Minimum k-mer frequency
            max_corrections: Maximum corrections per read (conservative)
            min_quality_threshold: Only correct bases below this quality
            collect_viz_data: Whether to collect visualization data
        """
        super().__init__(error_profile, collect_viz_data)
        
        self.k_size = k_size
        self.min_kmer_freq = min_kmer_freq
        self.max_corrections = max_corrections
        self.min_quality_threshold = min_quality_threshold
        
        # Initialize k-mer corrector
        self.kmer_corrector = KmerCorrector(
            k_size=k_size,
            min_kmer_freq=min_kmer_freq,
            max_corrections=max_corrections
        )
        
        # Initialize quality-aware corrector
        self.quality_corrector = QualityAwareCorrector(
            min_quality=min_quality_threshold,
            quality_threshold=min_quality_threshold
        )
    
    def correct_read(self, read: SeqRead) -> SeqRead:
        """
        Correct a single PacBio HiFi read conservatively.
        
        Args:
            read: SeqRead to correct
            
        Returns:
            Corrected SeqRead
        """
        sequence = read.sequence
        quality = read.quality
        
        # Skip correction for very high quality reads
        if quality:
            # Convert ASCII quality to Phred scores if needed
            if isinstance(quality, str):
                phred_scores = [ord(q) - 33 for q in quality]
                avg_quality = sum(phred_scores) / len(phred_scores)
            else:
                avg_quality = sum(quality) / len(quality) if quality else 0
            
            if avg_quality >= 40:  # Q40+ reads, skip correction
                return read
        
        # K-mer based correction
        corrected_seq, corrections = self.kmer_corrector.correct_sequence(
            sequence, quality
        )
        
        # Update statistics
        was_corrected = corrected_seq != sequence
        self.stats.record_read(
            was_corrected=was_corrected,
            num_bases=len(sequence),
            quality_before=quality,
            quality_after=quality
        )
        
        for pos, old_base, new_base in corrections:
            if old_base != new_base:
                if len(old_base) == len(new_base) == 1:
                    self.stats.record_correction('substitution', pos)
                elif len(old_base) < len(new_base):
                    self.stats.record_correction('insertion', pos)
                else:
                    self.stats.record_correction('deletion', pos)
        
        # Create corrected read
        return SeqRead(
            id=read.id,
            sequence=corrected_seq,
            quality=quality,
            technology=read.technology,
            metadata=read.metadata
        )


class IlluminaCorrector(BaseCorrector):
    """
    Illumina-specific error corrector.
    
    Illumina sequencing has substitution-heavy errors with position-dependent
    quality degradation, especially at 3' ends.
    
    Key features:
    - Substitution-focused correction
    - Position-aware quality weighting (3' end degradation)
    - GGC motif error correction
    - Adapter trimming integration
    
    Error characteristics:
    - Substitution-biased (4:1 ratio vs indels)
    - Quality degrades toward 3' end
    - Specific error motifs (e.g., GGC → GGT)
    - Homogeneous error distribution early in read
    """
    
    def __init__(
        self,
        error_profile: Optional[Dict[str, Any]] = None,
        k_size: int = 25,
        min_kmer_freq: int = 3,
        max_corrections: int = 10,
        trim_adapters: bool = True,
        collect_viz_data: bool = False
    ):
        """
        Initialize Illumina corrector.
        
        Args:
            error_profile: Optional error profile for guided correction
            k_size: K-mer size
            min_kmer_freq: Minimum k-mer frequency
            max_corrections: Maximum corrections per read
            trim_adapters: Detect and trim adapters
            collect_viz_data: Whether to collect visualization data
        """
        super().__init__(error_profile, collect_viz_data)
        
        self.k_size = k_size
        self.min_kmer_freq = min_kmer_freq
        self.max_corrections = max_corrections
        self.trim_adapters = trim_adapters
        
        # Initialize k-mer corrector
        self.kmer_corrector = KmerCorrector(
            k_size=k_size,
            min_kmer_freq=min_kmer_freq,
            max_corrections=max_corrections
        )
        
        # Initialize adapter detector if enabled
        if trim_adapters:
            self.adapter_detector = AdapterDetector()
        else:
            self.adapter_detector = None
    
    def correct_read(self, read: SeqRead) -> SeqRead:
        """
        Correct a single Illumina read.
        
        Args:
            read: SeqRead to correct
            
        Returns:
            Corrected SeqRead
        """
        sequence = read.sequence
        quality = read.quality
        
        # Trim adapters if enabled
        if self.trim_adapters and self.adapter_detector:
            trimmed_read, adapter_match = self.adapter_detector.trim_adapter(read)
            sequence = trimmed_read.sequence
            quality = trimmed_read.quality
        
        # K-mer based correction
        corrected_seq, corrections = self.kmer_corrector.correct_sequence(
            sequence, quality
        )
        
        # Update statistics
        was_corrected = corrected_seq != sequence
        self.stats.record_read(
            was_corrected=was_corrected,
            num_bases=len(sequence),
            quality_before=quality,
            quality_after=quality
        )
        
        for pos, old_base, new_base in corrections:
            if old_base != new_base:
                if len(old_base) == len(new_base) == 1:
                    self.stats.record_correction('substitution', pos)
                elif len(old_base) < len(new_base):
                    self.stats.record_correction('insertion', pos)
                else:
                    self.stats.record_correction('deletion', pos)
        
        # Create corrected read
        return SeqRead(
            id=read.id,
            sequence=corrected_seq,
            quality=quality,
            technology=read.technology,
            metadata=read.metadata
        )


class AncientDNACorrector(BaseCorrector):
    """
    Ancient DNA-specific error corrector.
    
    Ancient DNA has characteristic post-mortem damage patterns (deamination)
    that must be distinguished from true variants during correction.
    
    Key features:
    - Damage-aware correction (C→T at 5', G→A at 3')
    - Position-dependent damage modeling
    - UDG treatment detection
    - Conservative correction in damage zones
    - Damage profile estimation
    
    Error characteristics:
    - C→T deamination at 5' end (cytosine → uracil → thymine)
    - G→A deamination at 3' end (complement of C→T)
    - Damage concentrated within ~10bp of read ends
    - Exponential decay of damage with distance from ends
    """
    
    def __init__(
        self,
        error_profile: Optional[Dict[str, Any]] = None,
        k_size: int = 21,
        min_kmer_freq: int = 3,
        max_corrections: int = 8,
        damage_5p_rate: float = 0.05,
        damage_3p_rate: float = 0.05,
        damage_decay_length: int = 10,
        collect_viz_data: bool = False
    ):
        """
        Initialize ancient DNA corrector.
        
        Args:
            error_profile: Optional error profile for guided correction
            k_size: K-mer size
            min_kmer_freq: Minimum k-mer frequency
            max_corrections: Maximum corrections per read
            damage_5p_rate: C→T deamination rate at 5' end
            damage_3p_rate: G→A deamination rate at 3' end
            damage_decay_length: Distance from read end where damage occurs (bp)
            collect_viz_data: Whether to collect visualization data
        """
        super().__init__(error_profile, collect_viz_data)
        
        self.k_size = k_size
        self.min_kmer_freq = min_kmer_freq
        self.max_corrections = max_corrections
        self.damage_5p_rate = damage_5p_rate
        self.damage_3p_rate = damage_3p_rate
        self.damage_decay_length = damage_decay_length
        
        # Initialize k-mer corrector
        self.kmer_corrector = KmerCorrector(
            k_size=k_size,
            min_kmer_freq=min_kmer_freq,
            max_corrections=max_corrections
        )
    
    def correct_read(self, read: SeqRead) -> SeqRead:
        """
        Correct a single ancient DNA read with damage awareness.
        
        Args:
            read: SeqRead to correct
            
        Returns:
            Corrected SeqRead
        """
        sequence = read.sequence
        quality = read.quality
        
        # Identify damage zones
        damage_zones = set()
        
        # 5' end damage zone
        for i in range(min(self.damage_decay_length, len(sequence))):
            damage_zones.add(i)
        
        # 3' end damage zone
        for i in range(min(self.damage_decay_length, len(sequence))):
            pos = len(sequence) - 1 - i
            damage_zones.add(pos)
        
        # Track damage if collecting viz data
        if self.viz_data:
            for i in range(min(self.damage_decay_length, len(sequence))):
                # 5' C→T
                if sequence[i] == 'T':
                    self.viz_data.damage_5p_by_position[i] = self.viz_data.damage_5p_by_position.get(i, 0) + 1
                
                # 3' G→A
                pos = len(sequence) - 1 - i
                if pos >= 0 and sequence[pos] == 'A':
                    self.viz_data.damage_3p_by_position[i] = self.viz_data.damage_3p_by_position.get(i, 0) + 1
        
        # K-mer based correction
        corrected_seq, corrections = self.kmer_corrector.correct_sequence(
            sequence, quality
        )
        
        # Revert corrections in damage zones if they match damage patterns
        if corrected_seq != sequence:
            corrected_list = list(corrected_seq)
            
            for i in damage_zones:
                if i < len(sequence) and i < len(corrected_seq):
                    # Check if correction reverses likely damage
                    if sequence[i] == 'T' and corrected_list[i] == 'C':
                        # Potential C→T damage at 5' end
                        if i < self.damage_decay_length:
                            corrected_list[i] = 'T'  # Keep damage
                    
                    elif sequence[i] == 'A' and corrected_list[i] == 'G':
                        # Potential G→A damage at 3' end
                        if i >= len(sequence) - self.damage_decay_length:
                            corrected_list[i] = 'A'  # Keep damage
            
            corrected_seq = ''.join(corrected_list)
        
        # Update statistics
        was_corrected = corrected_seq != sequence
        self.stats.record_read(
            was_corrected=was_corrected,
            num_bases=len(sequence),
            quality_before=quality,
            quality_after=quality
        )
        
        for pos, old_base, new_base in corrections:
            if old_base != new_base:
                if len(old_base) == len(new_base) == 1:
                    self.stats.record_correction('substitution', pos)
                elif len(old_base) < len(new_base):
                    self.stats.record_correction('insertion', pos)
                else:
                    self.stats.record_correction('deletion', pos)
        
        # Create corrected read
        return SeqRead(
            id=read.id,
            sequence=corrected_seq,
            quality=quality,
            technology=read.technology,
            metadata=read.metadata
        )


# ============================================================================
# SECTION 5: ADAPTER DETECTION AND TRIMMING
# ============================================================================
# Adapter detection and removal for Illumina reads


# Common Illumina adapter sequences
ILLUMINA_ADAPTERS = {
    # TruSeq adapters
    "TruSeq_Universal": "AGATCGGAAGAG",
    "TruSeq_Index": "GATCGGAAGAGCACACGTCTGAACTCCAGTCAC",
    
    # Nextera adapters
    "Nextera_Transposase_1": "CTGTCTCTTATACACATCT",
    "Nextera_Transposase_2": "CTGTCTCTTATACACATCTCCGAGCCCACGAGAC",
    "Nextera_Transposase_3": "CTGTCTCTTATACACATCTGACGCTGCCGACGA",
    
    # Small RNA adapters
    "SmallRNA_3prime": "TGGAATTCTCGGGTGCCAAGG",
    
    # Universal adapters (partial sequences commonly found)
    "Universal_partial_1": "AGATCGGAAGAGC",
    "Universal_partial_2": "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT",

    # Illumina other
    "PCRFree_Adapter": "CTGTCTCTTATACACATCT+ATGTGTATAAGAGACA",
}


@dataclass
class AdapterMatch:
    """Information about an adapter match in a read."""
    adapter_name: str
    adapter_seq: str
    position: int  # Start position in read
    match_length: int  # Length of matched sequence
    is_3prime: bool  # True if adapter at 3' end
    
    def __str__(self) -> str:
        end = "3'" if self.is_3prime else "5'"
        return f"{self.adapter_name} at {end} (pos {self.position}, {self.match_length}bp)"


class AdapterDetector:
    """
    Detect and trim adapter sequences from reads.
    
    Uses semi-global alignment to find adapters, allowing for
    mismatches and partial adapter sequences.
    """
    
    def __init__(
        self,
        min_overlap: int = 8,
        max_error_rate: float = 0.1,
        min_adapter_length: int = 8,
        check_both_ends: bool = True,
        adapters: Optional[Dict[str, str]] = None
    ):
        """
        Initialize adapter detector.
        
        Args:
            min_overlap: Minimum overlap between read and adapter
            max_error_rate: Maximum error rate for adapter match (0.1 = 10%)
            min_adapter_length: Minimum adapter length to search for
            check_both_ends: Check both 3' and 5' ends of reads
            adapters: Custom adapter dictionary (uses defaults if None)
        """
        self.min_overlap = min_overlap
        self.max_error_rate = max_error_rate
        self.min_adapter_length = min_adapter_length
        self.check_both_ends = check_both_ends
        
        # Use provided adapters or defaults
        self.adapters = adapters if adapters is not None else ILLUMINA_ADAPTERS
        
        # Filter adapters by minimum length
        self.adapters = {
            name: seq for name, seq in self.adapters.items()
            if len(seq) >= min_adapter_length
        }
        
        # Statistics
        self.reads_processed = 0
        self.adapters_found = 0
        self.bases_trimmed = 0
    
    def detect_adapter(self, read: SeqRead) -> Optional[AdapterMatch]:
        """
        Detect adapter in a single read.
        
        Args:
            read: SeqRead to check for adapters
            
        Returns:
            AdapterMatch if adapter found, None otherwise
        """
        sequence = read.sequence
        
        # Check 3' end (most common)
        match = self._check_3prime(sequence)
        if match:
            return match
        
        # Check 5' end if requested
        if self.check_both_ends:
            match = self._check_5prime(sequence)
            if match:
                return match
        
        return None
    
    def _check_3prime(self, sequence: str) -> Optional[AdapterMatch]:
        """Check for adapter at 3' end of sequence."""
        # Check last N bases where N is reasonable for adapter
        search_length = min(len(sequence), 50)  # Check last 50bp
        read_suffix = sequence[-search_length:]
        
        for adapter_name, adapter_seq in self.adapters.items():
            # Try to find adapter at end of read
            match = self._find_overlap(read_suffix, adapter_seq)
            if match:
                # Convert local position to read position
                position = len(sequence) - search_length + match[0]
                return AdapterMatch(
                    adapter_name=adapter_name,
                    adapter_seq=adapter_seq,
                    position=position,
                    match_length=match[1],
                    is_3prime=True
                )
        
        return None
    
    def _check_5prime(self, sequence: str) -> Optional[AdapterMatch]:
        """Check for adapter at 5' end of sequence."""
        # Check first N bases
        search_length = min(len(sequence), 50)
        read_prefix = sequence[:search_length]
        
        for adapter_name, adapter_seq in self.adapters.items():
            # Try to find adapter at start of read
            match = self._find_overlap_prefix(read_prefix, adapter_seq)
            if match:
                return AdapterMatch(
                    adapter_name=adapter_name,
                    adapter_seq=adapter_seq,
                    position=match[0],
                    match_length=match[1],
                    is_3prime=False
                )
        
        return None
    
    def _find_overlap(
        self, read_suffix: str, adapter_seq: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find overlap between read suffix and adapter.
        
        Returns (position, length) if found, None otherwise.
        """
        # Try different overlap lengths
        for overlap_len in range(len(adapter_seq), self.min_overlap - 1, -1):
            adapter_prefix = adapter_seq[:overlap_len]
            
            # Search for this adapter prefix in read suffix
            for i in range(len(read_suffix) - overlap_len + 1):
                read_segment = read_suffix[i:i + overlap_len]
                
                # Calculate mismatches
                mismatches = sum(
                    1 for a, b in zip(read_segment, adapter_prefix) if a != b
                )
                error_rate = mismatches / overlap_len
                
                if error_rate <= self.max_error_rate:
                    return (i, overlap_len)
        
        return None
    
    def _find_overlap_prefix(
        self, read_prefix: str, adapter_seq: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find overlap between read prefix and adapter (5' end).
        
        Returns (position, length) if found, None otherwise.
        """
        # For 5' adapters, check if adapter suffix matches read prefix
        for overlap_len in range(len(adapter_seq), self.min_overlap - 1, -1):
            adapter_suffix = adapter_seq[-overlap_len:]
            
            # Check if read starts with adapter suffix
            if len(read_prefix) >= overlap_len:
                read_segment = read_prefix[:overlap_len]
                
                mismatches = sum(
                    1 for a, b in zip(read_segment, adapter_suffix) if a != b
                )
                error_rate = mismatches / overlap_len
                
                if error_rate <= self.max_error_rate:
                    return (0, overlap_len)
        
        return None
    
    def trim_adapter(self, read: SeqRead) -> Tuple[SeqRead, Optional[AdapterMatch]]:
        """
        Detect and trim adapter from read.
        
        Args:
            read: SeqRead to trim
            
        Returns:
            (trimmed_read, adapter_match) tuple
            If no adapter found, returns (original_read, None)
        """
        self.reads_processed += 1
        
        # Detect adapter
        match = self.detect_adapter(read)
        
        if match is None:
            return read, None
        
        # Trim adapter
        self.adapters_found += 1
        
        if match.is_3prime:
            # Trim from 3' end
            trimmed_seq = read.sequence[:match.position]
            trimmed_qual = read.quality[:match.position] if read.quality else None
            self.bases_trimmed += len(read.sequence) - match.position
        else:
            # Trim from 5' end
            trim_end = match.position + match.match_length
            trimmed_seq = read.sequence[trim_end:]
            trimmed_qual = read.quality[trim_end:] if read.quality else None
            self.bases_trimmed += trim_end
        
        # Create trimmed read
        trimmed_read = SeqRead(
            id=read.id,
            sequence=trimmed_seq,
            quality=trimmed_qual,
            technology=read.technology,
            metadata=read.metadata
        )
        
        return trimmed_read, match
    
    def get_stats(self) -> Dict[str, float]:
        """Get adapter detection statistics."""
        adapter_rate = (
            self.adapters_found / self.reads_processed
            if self.reads_processed > 0 else 0.0
        )
        avg_trim_length = (
            self.bases_trimmed / self.adapters_found
            if self.adapters_found > 0 else 0.0
        )
        
        return {
            "reads_processed": self.reads_processed,
            "adapters_found": self.adapters_found,
            "adapter_rate": adapter_rate,
            "bases_trimmed": self.bases_trimmed,
            "avg_trim_length": avg_trim_length
        }


def detect_and_trim_adapters(
    reads: List[SeqRead],
    min_overlap: int = 8,
    max_error_rate: float = 0.1,
    verbose: bool = False
) -> Tuple[List[SeqRead], Dict[str, float]]:
    """
    Convenience function to detect and trim adapters from reads.
    
    Args:
        reads: List of reads to process
        min_overlap: Minimum adapter overlap
        max_error_rate: Maximum error rate for matching
        verbose: Print progress information
        
    Returns:
        (trimmed_reads, statistics) tuple
    """
    detector = AdapterDetector(
        min_overlap=min_overlap,
        max_error_rate=max_error_rate
    )
    
    trimmed_reads = []
    for read in reads:
        trimmed, _ = detector.trim_adapter(read)
        trimmed_reads.append(trimmed)
    
    return trimmed_reads, detector.get_stats()


# ============================================================================
# SECTION 6: VISUALIZATION AND REPORTING
# ============================================================================
# Comprehensive error correction visualization and reporting


@dataclass
class ErrorProfile:
    """Stores error statistics for visualization."""
    
    # Position-based error tracking
    errors_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    corrections_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Error type tracking
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    
    # Corrections by type
    substitutions_corrected: int = 0
    insertions_corrected: int = 0
    deletions_corrected: int = 0
    
    # Homopolymer tracking
    homopolymer_lengths: List[int] = field(default_factory=list)
    homopolymer_errors: List[int] = field(default_factory=list)
    homopolymer_corrections: List[int] = field(default_factory=list)
    
    # Quality tracking
    quality_before: List[float] = field(default_factory=list)
    quality_after: List[float] = field(default_factory=list)
    
    # K-mer spectrum
    kmer_frequencies: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Ancient DNA specific
    damage_5p_by_position: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    damage_3p_by_position: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    
    # GC content tracking
    gc_content_errors: List[float] = field(default_factory=list)
    gc_content_correct: List[float] = field(default_factory=list)
    
    # Read length tracking
    read_lengths: List[int] = field(default_factory=list)
    
    # Metadata
    total_reads: int = 0
    total_bases: int = 0
    total_errors: int = 0
    total_corrections: int = 0


class ErrorVisualizer:
    """Creates comprehensive error correction visualizations."""
    
    def __init__(self, output_dir: str = "correction_reports"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        if not VISUALIZATION_AVAILABLE:
            print("⚠️  Warning: Matplotlib not available. Visualizations will be disabled.")
            print("   Install with: pip install matplotlib seaborn")
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_summary_report(
        self,
        profile: ErrorProfile,
        technology: str,
        prefix: str = "correction"
    ) -> str:
        """Create text summary report."""
        filepath = os.path.join(self.output_dir, f"{prefix}_summary.txt")
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ERROR CORRECTION SUMMARY REPORT - {technology.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total reads processed:        {profile.total_reads:,}\n")
            f.write(f"Total bases processed:        {profile.total_bases:,}\n")
            f.write(f"Total errors detected:        {profile.total_errors:,}\n")
            f.write(f"Total corrections applied:    {profile.total_corrections:,}\n")
            
            if profile.total_bases > 0:
                error_rate = (profile.total_errors / profile.total_bases) * 100
                f.write(f"Error rate:                   {error_rate:.3f}%\n")
            
            if profile.total_errors > 0:
                correction_rate = (profile.total_corrections / profile.total_errors) * 100
                f.write(f"Correction rate:              {correction_rate:.1f}%\n")
            
            f.write("\n")
            
            # Error types
            f.write("ERROR TYPES:\n")
            f.write("-" * 80 + "\n")
            total_typed_errors = profile.substitutions + profile.insertions + profile.deletions
            if total_typed_errors > 0:
                f.write(f"Substitutions:                {profile.substitutions:,} "
                       f"({profile.substitutions/total_typed_errors*100:.1f}%)\n")
                f.write(f"Insertions:                   {profile.insertions:,} "
                       f"({profile.insertions/total_typed_errors*100:.1f}%)\n")
                f.write(f"Deletions:                    {profile.deletions:,} "
                       f"({profile.deletions/total_typed_errors*100:.1f}%)\n")
            else:
                f.write("No error type data available\n")
            
            f.write("\n")
            
            # Correction types
            f.write("CORRECTIONS BY TYPE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Substitutions corrected:      {profile.substitutions_corrected:,}\n")
            f.write(f"Insertions corrected:         {profile.insertions_corrected:,}\n")
            f.write(f"Deletions corrected:          {profile.deletions_corrected:,}\n")
            f.write("\n")
            
            # Quality improvement
            if profile.quality_before and profile.quality_after and NUMPY_AVAILABLE:
                f.write("QUALITY IMPROVEMENT:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean quality before:          Q{np.mean(profile.quality_before):.1f}\n")
                f.write(f"Mean quality after:           Q{np.mean(profile.quality_after):.1f}\n")
                improvement = np.mean(profile.quality_after) - np.mean(profile.quality_before)
                f.write(f"Quality improvement:          {improvement:+.1f}\n")
                f.write("\n")
            
            # Homopolymer analysis
            if profile.homopolymer_lengths and NUMPY_AVAILABLE:
                f.write("HOMOPOLYMER ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total homopolymers:           {len(profile.homopolymer_lengths):,}\n")
                f.write(f"Mean length:                  {np.mean(profile.homopolymer_lengths):.1f} bp\n")
                f.write(f"Max length:                   {max(profile.homopolymer_lengths)} bp\n")
                if profile.homopolymer_corrections:
                    f.write(f"Homopolymers corrected:       {sum(profile.homopolymer_corrections):,}\n")
                f.write("\n")
            
            # Read lengths
            if profile.read_lengths and NUMPY_AVAILABLE:
                f.write("READ LENGTH STATISTICS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean read length:             {np.mean(profile.read_lengths):.0f} bp\n")
                f.write(f"Median read length:           {np.median(profile.read_lengths):.0f} bp\n")
                f.write(f"Min read length:              {min(profile.read_lengths)} bp\n")
                f.write(f"Max read length:              {max(profile.read_lengths)} bp\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            if VISUALIZATION_AVAILABLE:
                f.write("Visualizations generated in the same directory.\n")
            else:
                f.write("Install matplotlib and seaborn for visualizations.\n")
            f.write("=" * 80 + "\n")
        
        return filepath


def collect_kmer_spectrum(kmer_counts: Dict[str, int], profile: ErrorProfile):
    """
    Collect k-mer frequency spectrum from k-mer counter.
    
    Args:
        kmer_counts: Dictionary of k-mer -> count
        profile: ErrorProfile to update
    """
    frequency_dist = Counter(kmer_counts.values())
    profile.kmer_frequencies = dict(frequency_dist)


# ============================================================================
# BATCH PROCESSING FUNCTIONS (Nextflow Integration)
# ============================================================================

def profile_technology(
    reads_file: str,
    technology: str,
    k_size: int = 21,
    sample_size: int = 100000,
    threads: int = 1
) -> Dict[str, Any]:
    """
    Profile sequencing errors for a specific technology.
    
    This function analyzes reads to build an error profile including k-mer
    spectrum, error rate estimates, and common error patterns. Used by
    Nextflow batch processing for the sequential profiling stage.
    
    Args:
        reads_file: Path to FASTQ file
        technology: 'hifi', 'ont', or 'illumina'
        k_size: K-mer size for spectrum analysis
        sample_size: Number of reads to sample for profiling
        threads: Number of threads to use
    
    Returns:
        Dict with error profile:
        {
            'technology': str,
            'k_size': int,
            'kmer_spectrum': dict,
            'error_rate': float,
            'mean_quality': float,
            'read_count': int,
            'base_count': int,
            'gc_content': float
        }
    """
    from ..io_utils import read_fastq
    
    logger.info(f"Profiling {technology} reads from {reads_file}")
    
    # Initialize k-mer spectrum
    kmer_counts = Counter()
    quality_scores = []
    read_lengths = []
    total_reads = 0
    total_bases = 0
    gc_count = 0
    
    # Sample reads and build profile
    for read in read_fastq(reads_file):
        if total_reads >= sample_size:
            break
        
        seq = read.sequence
        read_lengths.append(len(seq))
        total_bases += len(seq)
        gc_count += seq.count('G') + seq.count('C')
        
        # Extract k-mers
        for i in range(len(seq) - k_size + 1):
            kmer = seq[i:i + k_size]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                kmer_counts[kmer] += 1
        
        # Collect quality scores (convert from ASCII to Phred scores)
        if read.quality:
            # Convert ASCII quality scores to Phred scores (Phred+33 encoding)
            phred_scores = [ord(q) - 33 for q in read.quality]
            quality_scores.extend(phred_scores)
        
        total_reads += 1
    
    # Calculate statistics
    mean_quality = np.mean(quality_scores) if quality_scores else 30.0
    error_rate = 10 ** (-mean_quality / 10) if quality_scores else 0.01
    gc_content = gc_count / total_bases if total_bases > 0 else 0.5
    
    # Build k-mer spectrum (frequency distribution)
    frequency_dist = Counter(kmer_counts.values())
    kmer_spectrum = {str(k): v for k, v in sorted(frequency_dist.items())}
    
    profile = {
        'technology': technology,
        'k_size': k_size,
        'kmer_spectrum': kmer_spectrum,
        'error_rate': float(error_rate),
        'mean_quality': float(mean_quality),
        'read_count': total_reads,
        'base_count': total_bases,
        'gc_content': float(gc_content),
        'mean_read_length': float(np.mean(read_lengths)) if read_lengths else 0.0
    }
    
    logger.info(f"Profile complete: {total_reads} reads, error rate={error_rate:.4f}")
    
    return profile


def correct_batch(
    reads_file: str,
    technology: str,
    error_profile: Dict[str, Any],
    output_file: str,
    threads: int = 1
) -> None:
    """
    Correct errors in a batch of reads.
    
    This function applies technology-specific error correction to a batch
    of reads using a pre-computed error profile. Used by Nextflow for
    parallel batch correction.
    
    Args:
        reads_file: Input FASTQ batch
        technology: Technology type ('hifi', 'ont', 'illumina')
        error_profile: Pre-computed error profile from profile_technology()
        output_file: Output corrected FASTQ
        threads: Number of threads to use
    """
    from ..io_utils import read_fastq, write_fastq
    
    logger.info(f"Correcting {technology} reads batch: {Path(reads_file).name}")
    
    # Create corrector for technology
    corrector = get_corrector(
        technology=technology,
        error_profile=error_profile,
        k_size=error_profile.get('k_size', 21)
    )
    
    # Correct reads
    corrected_reads = []
    for read in read_fastq(reads_file):
        corrected = corrector.correct_read(read)
        corrected_reads.append(corrected)
    
    # Write output
    write_fastq(corrected_reads, output_file)
    
    logger.info(f"Corrected {len(corrected_reads)} reads → {output_file}")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Section 1: Base Corrector Infrastructure
    'BaseCorrector',
    'get_corrector',
    
    # Section 2: Correction Statistics
    'CorrectionStats',
    
    # Section 3: Correction Strategies
    'BloomFilter',
    'KmerSpectrum',
    'KmerCorrector',
    'QualityAwareCorrector',
    'ConsensusCorrector',
    
    # Section 4: Technology-Specific Correctors
    'HomopolymerDetector',
    'ONTCorrector',
    'PacBioCorrector',
    'IlluminaCorrector',
    'AncientDNACorrector',
    
    # Section 5: Adapter Detection
    'AdapterMatch',
    'AdapterDetector',
    'detect_and_trim_adapters',
    'ILLUMINA_ADAPTERS',
    
    # Section 6: Visualization
    'ErrorProfile',
    'ErrorVisualizer',
    'collect_kmer_spectrum',
    
    # Section 7: Batch Processing (Nextflow)
    'profile_technology',
    'correct_batch',
]
