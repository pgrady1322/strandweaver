#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core I/O module for StrandWeaver.

Consolidated module containing:
- Core read data structures (SeqRead, ReadPair)
- FASTQ file I/O operations
- FASTA file I/O operations
- ReadCollection for organizing reads by technology and role

This module handles reading and writing sequencing reads with support
for multiple technologies and automatic read type classification.
"""

# =============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# =============================================================================

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, TextIO, Union, Tuple, List, Dict, Any
from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# =============================================================================
# SECTION 2: CORE READ DATA STRUCTURES
# =============================================================================
# Source: read.py

@dataclass
class SeqRead:
    """
    Sequencing read with metadata.
    
    Attributes:
        id: Read identifier
        sequence: DNA sequence
        quality: Quality scores (Phred+33 encoding)
        technology: Sequencing technology
        length: Read length (computed)
        metadata: Additional metadata
        is_corrected: Whether read has been error-corrected
        correction_confidence: Confidence score for correction (0-1)
        roles: Assembly roles this read can fulfill
    """
    id: str
    sequence: str
    quality: Optional[str] = None
    technology: 'ReadTechnology' = None  # Forward reference, imported from technology module
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_corrected: bool = False
    correction_confidence: float = 0.0
    
    def __post_init__(self):
        """Initialize computed fields."""
        # Import here to avoid circular dependency
        from .technology_handling_core_module import ReadTechnology, classify_read_type
        
        # Set default technology if None
        if self.technology is None:
            self.technology = ReadTechnology.UNKNOWN
        
        # Ensure sequence is uppercase
        self.sequence = self.sequence.upper()
        
        # Auto-classify ONT reads from ONT_REGULAR to ONT_ULTRALONG if appropriate
        # BUT respect explicit ONT_ULTRALONG designation (don't downgrade)
        if self.technology.name == 'ONT_REGULAR':
            # Only auto-upgrade ONT_REGULAR to ONT_ULTRALONG if long enough
            self.technology = classify_read_type(len(self.sequence), self.technology)
        # If user explicitly set ONT_ULTRALONG, respect it even for shorter reads
    
    @property
    def length(self) -> int:
        """Get read length."""
        return len(self.sequence)
    
    @property
    def profile(self):
        """Get read type profile."""
        from .technology_handling_core_module import get_read_profile
        return get_read_profile(self.technology)
    
    @property
    def roles(self) -> List['ReadRole']:
        """Get assembly roles this read can fulfill."""
        return self.profile.roles
    
    def can_fulfill_role(self, role: 'ReadRole') -> bool:
        """
        Check if this read can fulfill a specific role.
        
        Args:
            role: Assembly role to check
        
        Returns:
            True if read can fulfill this role
        """
        return role in self.roles
    
    def is_long_read(self) -> bool:
        """Check if this is a long read (>10kb)."""
        return self.profile.is_long_read()
    
    def is_ultra_long(self) -> bool:
        """Check if this is an ultra-long read (>100kb for ONT)."""
        return self.profile.is_ultra_long()
    
    def is_short_read(self) -> bool:
        """Check if this is a short read (<1kb)."""
        return self.length < 1000
    
    def get_average_quality(self) -> Optional[float]:
        """
        Calculate average quality score.
        
        Returns:
            Average Phred quality score, or None if no quality data
        """
        if not self.quality:
            return None
        
        # Convert Phred+33 to numeric scores
        scores = [ord(c) - 33 for c in self.quality]
        return sum(scores) / len(scores) if scores else None
    
    def get_gc_content(self) -> float:
        """
        Calculate GC content.
        
        Returns:
            GC content as fraction (0-1)
        """
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return gc_count / self.length if self.length > 0 else 0.0
    
    def reverse_complement(self) -> 'SeqRead':
        """
        Get reverse complement of this read.
        
        Returns:
            New SeqRead with reverse complement sequence
        """
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        rc_seq = ''.join(complement.get(base, 'N') for base in reversed(self.sequence))
        rc_qual = self.quality[::-1] if self.quality else None
        
        return SeqRead(
            id=f"{self.id}_RC",
            sequence=rc_seq,
            quality=rc_qual,
            technology=self.technology,
            metadata={**self.metadata, 'is_reverse_complement': True},
            is_corrected=self.is_corrected,
            correction_confidence=self.correction_confidence
        )
    
    def to_fastq_string(self) -> str:
        """
        Convert to FASTQ format string.
        
        Returns:
            FASTQ format string (4 lines)
        """
        quality = self.quality if self.quality else 'I' * self.length  # Default quality
        return f"@{self.id}\n{self.sequence}\n+\n{quality}\n"
    
    def to_fasta_string(self) -> str:
        """
        Convert to FASTA format string.
        
        Returns:
            FASTA format string (2 lines)
        """
        return f">{self.id}\n{self.sequence}\n"
    
    def __repr__(self) -> str:
        """String representation."""
        tech_value = self.technology.value if hasattr(self.technology, 'value') else str(self.technology)
        return (f"SeqRead(id='{self.id}', length={self.length}, "
                f"tech={tech_value}, corrected={self.is_corrected})")
    
    def __len__(self) -> int:
        """Length of read."""
        return self.length


@dataclass
class ReadPair:
    """
    Paired-end read pair.
    
    Attributes:
        read1: Forward read (R1)
        read2: Reverse read (R2)
        insert_size: Insert size (distance between reads)
    """
    read1: SeqRead
    read2: SeqRead
    insert_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate read pair."""
        # Ensure both reads have same technology
        if self.read1.technology != self.read2.technology:
            raise ValueError(
                f"Read pair technology mismatch: {self.read1.technology} vs {self.read2.technology}"
            )
    
    @property
    def technology(self):
        """Get technology for this read pair."""
        return self.read1.technology
    
    @property
    def total_bases(self) -> int:
        """Get total bases in both reads."""
        return self.read1.length + self.read2.length
    
    def is_proper_pair(self, max_insert: int = 1000) -> bool:
        """
        Check if this is a proper pair.
        
        Args:
            max_insert: Maximum expected insert size
        
        Returns:
            True if proper pair
        """
        if self.insert_size is None:
            return True  # Unknown insert size
        
        return 0 < self.insert_size <= max_insert
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ReadPair(R1={self.read1.length}bp, R2={self.read2.length}bp, insert={self.insert_size})"


# =============================================================================
# SECTION 3: FILE UTILITIES
# =============================================================================
# Helper functions for file handling with automatic gzip detection

def is_gzipped(filepath: Union[str, Path]) -> bool:
    """
    Check if file is gzip compressed.
    
    Args:
        filepath: Path to file
    
    Returns:
        True if file is gzipped
    """
    filepath = Path(filepath)
    return filepath.suffix in ('.gz', '.gzip')


def open_file(filepath: Union[str, Path], mode: str = 'r') -> TextIO:
    """
    Open file with automatic gzip detection.
    
    Args:
        filepath: Path to file
        mode: File mode ('r' or 'w')
    
    Returns:
        File handle
    """
    filepath = Path(filepath)
    
    if is_gzipped(filepath):
        if 'r' in mode:
            return gzip.open(filepath, 'rt')
        else:
            return gzip.open(filepath, 'wt')
    else:
        return open(filepath, mode)


# =============================================================================
# SECTION 4: FASTQ FILE I/O
# =============================================================================
# Source: fastq.py

def read_fastq(
    filepath: Union[str, Path],
    technology: Optional['ReadTechnology'] = None,
    sample_size: Optional[int] = None,
    min_length: int = 0,
    max_length: Optional[int] = None,
    min_quality: float = 0.0,
    infer_tech_from_length: bool = False
) -> Iterator[SeqRead]:
    """
    Read FASTQ file and yield SeqRead objects.
    
    Args:
        filepath: Path to FASTQ file (can be gzipped)
        technology: Sequencing technology (if None, must infer or use unknown)
        sample_size: Maximum number of reads to yield (None = all)
        min_length: Minimum read length filter
        max_length: Maximum read length filter (None = no limit)
        min_quality: Minimum average quality score filter
        infer_tech_from_length: Automatically infer technology from read length
    
    Yields:
        SeqRead objects
    
    Examples:
        >>> # Read all Illumina reads
        >>> for read in read_fastq("reads.fq", technology=ReadTechnology.ILLUMINA):
        ...     print(read.id, read.length)
        
        >>> # Read ONT reads, automatically classify ultra-long
        >>> for read in read_fastq("ont.fq", technology=ReadTechnology.ONT_REGULAR):
        ...     if read.is_ultra_long():
        ...         print(f"Ultra-long read: {read.id}, {read.length}bp")
        
        >>> # Sample first 10k reads
        >>> reads = list(read_fastq("large.fq", sample_size=10000))
    """
    from .technology_handling_core_module import ReadTechnology, infer_technology_from_length
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"FASTQ file not found: {filepath}")
    
    count = 0
    
    with open_file(filepath, 'r') as handle:
        for record in SeqIO.parse(handle, "fastq"):
            # Check sample size limit
            if sample_size and count >= sample_size:
                break
            
            # Get sequence and quality
            sequence = str(record.seq)
            quality = "".join(chr(q + 33) for q in record.letter_annotations.get("phred_quality", []))
            
            # Apply length filters
            length = len(sequence)
            if length < min_length:
                continue
            if max_length and length > max_length:
                continue
            
            # Infer technology if requested
            if infer_tech_from_length:
                tech = infer_technology_from_length(length, quality)
            elif technology:
                tech = technology
            else:
                tech = ReadTechnology.UNKNOWN
            
            # Create SeqRead
            read = SeqRead(
                id=record.id,
                sequence=sequence,
                quality=quality if quality else None,
                technology=tech,
                metadata={'description': record.description}
            )
            
            # Apply quality filter
            if min_quality > 0:
                avg_qual = read.get_average_quality()
                if avg_qual is None or avg_qual < min_quality:
                    continue
            
            yield read
            count += 1


def read_fastq_pairs(
    filepath_r1: Union[str, Path],
    filepath_r2: Union[str, Path],
    technology: 'ReadTechnology' = None,
    sample_size: Optional[int] = None,
    min_length: int = 0,
    max_insert: int = 1000
) -> Iterator[Tuple[SeqRead, SeqRead]]:
    """
    Read paired-end FASTQ files.
    
    Args:
        filepath_r1: Path to R1 FASTQ file
        filepath_r2: Path to R2 FASTQ file
        technology: Sequencing technology
        sample_size: Maximum number of pairs to yield
        min_length: Minimum read length
        max_insert: Maximum expected insert size
    
    Yields:
        Tuples of (read1, read2)
    """
    from .technology_handling_core_module import ReadTechnology
    
    if technology is None:
        technology = ReadTechnology.ILLUMINA
    
    r1_iter = read_fastq(filepath_r1, technology, sample_size, min_length)
    r2_iter = read_fastq(filepath_r2, technology, sample_size, min_length)
    
    for read1, read2 in zip(r1_iter, r2_iter):
        # Verify read IDs match (allow for /1 /2 suffixes)
        id1 = read1.id.rstrip('/1').rstrip('.1')
        id2 = read2.id.rstrip('/2').rstrip('.2')
        
        if id1 != id2:
            raise ValueError(f"Read ID mismatch in paired files: {read1.id} vs {read2.id}")
        
        # Mark as paired
        read1.metadata['is_paired'] = True
        read1.metadata['pair'] = 'R1'
        read2.metadata['is_paired'] = True
        read2.metadata['pair'] = 'R2'
        
        yield read1, read2


def write_fastq(
    reads: Iterator[SeqRead],
    filepath: Union[str, Path],
    compress: bool = False
) -> int:
    """
    Write SeqRead objects to FASTQ file.
    
    Args:
        reads: Iterator of SeqRead objects
        filepath: Output FASTQ file path
        compress: Whether to gzip compress output
    
    Returns:
        Number of reads written
    """
    filepath = Path(filepath)
    
    # Add .gz extension if compressing
    if compress and not is_gzipped(filepath):
        filepath = Path(str(filepath) + '.gz')
    
    # Create output directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    with open_file(filepath, 'w') as handle:
        for read in reads:
            # Convert SeqRead to BioPython SeqRecord
            quality_scores = [ord(c) - 33 for c in read.quality] if read.quality else [30] * read.length
            
            record = SeqRecord(
                seq=Seq(read.sequence),  # Must be a Seq object, not a string
                id=read.id,
                description="",
                letter_annotations={"phred_quality": quality_scores}
            )
            
            SeqIO.write(record, handle, "fastq")
            count += 1
    
    return count


def count_fastq_reads(filepath: Union[str, Path]) -> int:
    """
    Count number of reads in FASTQ file.
    
    Args:
        filepath: Path to FASTQ file
    
    Returns:
        Number of reads
    """
    count = 0
    
    with open_file(filepath, 'r') as handle:
        for _ in SeqIO.parse(handle, "fastq"):
            count += 1
    
    return count


def get_fastq_stats(filepath: Union[str, Path], sample_size: int = 10000) -> dict:
    """
    Get statistics about FASTQ file.
    
    Args:
        filepath: Path to FASTQ file
        sample_size: Number of reads to sample for stats
    
    Returns:
        Dictionary with statistics
    """
    lengths = []
    qualities = []
    gc_contents = []
    
    for read in read_fastq(filepath, sample_size=sample_size):
        lengths.append(read.length)
        
        avg_qual = read.get_average_quality()
        if avg_qual is not None:
            qualities.append(avg_qual)
        
        gc_contents.append(read.get_gc_content())
    
    if not lengths:
        return {}
    
    return {
        'num_reads_sampled': len(lengths),
        'mean_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_quality': sum(qualities) / len(qualities) if qualities else None,
        'mean_gc_content': sum(gc_contents) / len(gc_contents) if gc_contents else None,
    }


# =============================================================================
# SECTION 5: FASTA FILE I/O
# =============================================================================
# Source: fasta.py

def read_fasta(
    filepath: Union[str, Path],
    technology: Optional['ReadTechnology'] = None,
    min_length: int = 0,
    max_length: Optional[int] = None
) -> Iterator[SeqRead]:
    """
    Read FASTA file and yield SeqRead objects.
    
    Args:
        filepath: Path to FASTA file (can be gzipped)
        technology: Sequencing technology
        min_length: Minimum sequence length filter
        max_length: Maximum sequence length filter
    
    Yields:
        SeqRead objects (without quality scores)
    """
    from .technology_handling_core_module import ReadTechnology
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")
    
    # Determine file opener
    if filepath.suffix in ('.gz', '.gzip'):
        handle = gzip.open(filepath, 'rt')
    else:
        handle = open(filepath, 'r')
    
    try:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            length = len(sequence)
            
            # Apply filters
            if length < min_length:
                continue
            if max_length and length > max_length:
                continue
            
            yield SeqRead(
                id=record.id,
                sequence=sequence,
                quality=None,  # FASTA has no quality scores
                technology=technology or ReadTechnology.UNKNOWN,
                metadata={'description': record.description}
            )
    finally:
        handle.close()


def write_fasta(
    reads: Iterator[SeqRead],
    filepath: Union[str, Path],
    compress: bool = False,
    line_width: int = 80
) -> int:
    """
    Write SeqRead objects to FASTA file.
    
    Args:
        reads: Iterator of SeqRead objects
        filepath: Output FASTA file path
        compress: Whether to gzip compress output
        line_width: Number of bases per line (0 = no wrapping)
    
    Returns:
        Number of sequences written
    """
    filepath = Path(filepath)
    
    # Add .gz extension if compressing
    if compress and filepath.suffix not in ('.gz', '.gzip'):
        filepath = Path(str(filepath) + '.gz')
    
    # Create output directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file opener
    if compress or filepath.suffix in ('.gz', '.gzip'):
        handle = gzip.open(filepath, 'wt')
    else:
        handle = open(filepath, 'w')
    
    count = 0
    
    try:
        for read in reads:
            # Write header
            handle.write(f">{read.id}\n")
            
            # Write sequence (with optional line wrapping)
            if line_width > 0:
                for i in range(0, len(read.sequence), line_width):
                    handle.write(read.sequence[i:i+line_width] + '\n')
            else:
                handle.write(read.sequence + '\n')
            
            count += 1
    finally:
        handle.close()
    
    return count


def count_fasta_sequences(filepath: Union[str, Path]) -> int:
    """
    Count number of sequences in FASTA file.
    
    Args:
        filepath: Path to FASTA file
    
    Returns:
        Number of sequences
    """
    count = sum(1 for _ in read_fasta(filepath))
    return count


def get_fasta_stats(filepath: Union[str, Path]) -> dict:
    """
    Get statistics about FASTA file.
    
    Args:
        filepath: Path to FASTA file
    
    Returns:
        Dictionary with statistics including N50
    """
    lengths = []
    gc_contents = []
    
    for read in read_fasta(filepath):
        lengths.append(read.length)
        gc_contents.append(read.get_gc_content())
    
    if not lengths:
        return {}
    
    # Sort lengths for N50 calculation
    sorted_lengths = sorted(lengths, reverse=True)
    total_length = sum(sorted_lengths)
    half_length = total_length / 2
    
    # Calculate N50
    cumulative = 0
    n50 = 0
    l50 = 0
    
    for i, length in enumerate(sorted_lengths, 1):
        cumulative += length
        if cumulative >= half_length:
            n50 = length
            l50 = i
            break
    
    return {
        'num_sequences': len(lengths),
        'total_length': total_length,
        'mean_length': total_length / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'n50': n50,
        'l50': l50,
        'mean_gc_content': sum(gc_contents) / len(gc_contents) if gc_contents else None,
    }


# =============================================================================
# SECTION 6: READ COLLECTION MANAGEMENT
# =============================================================================
# Source: collection.py

class ReadCollection:
    """
    Collection of reads organized by technology and role.
    
    This class manages different read types and enables:
    - Separation of ONT ultra-long reads for scaffolding
    - Grouping reads by technology for batch correction
    - Filtering reads by role for specific assembly steps
    """
    
    def __init__(self):
        """Initialize empty read collection."""
        # Main storage: technology -> list of reads
        self._reads_by_tech: Dict = defaultdict(list)
        
        # Index by role for quick filtering
        self._reads_by_role: Dict = defaultdict(list)
        
        # Statistics
        self._total_reads = 0
        self._total_bases = 0
    
    def add_read(self, read: SeqRead):
        """
        Add a read to the collection.
        
        Args:
            read: SeqRead to add
        """
        # Add to technology index
        self._reads_by_tech[read.technology].append(read)
        
        # Add to role indices
        for role in read.roles:
            self._reads_by_role[role].append(read)
        
        # Update statistics
        self._total_reads += 1
        self._total_bases += read.length
    
    def add_reads_from_file(
        self,
        filepath: Path,
        technology: 'ReadTechnology',
        file_format: str = 'fastq',
        sample_size: Optional[int] = None
    ):
        """
        Add reads from a file.
        
        Args:
            filepath: Path to reads file
            technology: Technology type for these reads
            file_format: File format ('fastq' or 'fasta')
            sample_size: Maximum number of reads to add
        """
        if file_format == 'fastq':
            reader = read_fastq(filepath, technology=technology, sample_size=sample_size)
        elif file_format == 'fasta':
            reader = read_fasta(filepath, technology=technology)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        for read in reader:
            self.add_read(read)
    
    def get_reads_by_technology(self, technology: 'ReadTechnology') -> List[SeqRead]:
        """
        Get all reads of a specific technology.
        
        Args:
            technology: Read technology type
        
        Returns:
            List of reads
        """
        return self._reads_by_tech.get(technology, [])
    
    def get_reads_by_role(self, role: 'ReadRole') -> List[SeqRead]:
        """
        Get all reads that can fulfill a specific role.
        
        Args:
            role: Assembly role
        
        Returns:
            List of reads that can fulfill this role
        """
        return self._reads_by_role.get(role, [])
    
    def get_technologies(self) -> List['ReadTechnology']:
        """
        Get list of all technologies present in the collection.
        
        Returns:
            List of ReadTechnology values that have reads
        """
        return [tech for tech, reads in self._reads_by_tech.items() if reads]
    
    def get_stats_by_technology(self) -> Dict:
        """
        Get detailed statistics for each technology.
        
        Returns:
            Dictionary mapping technology to statistics dict with:
            - count: Number of reads
            - total_bases: Total base pairs
            - mean_length: Average read length
            - min_length: Shortest read
            - max_length: Longest read
            - mean_quality: Average quality score (if available)
        """
        stats = {}
        
        for tech, reads in self._reads_by_tech.items():
            if not reads:
                continue
            
            lengths = [r.length for r in reads]
            total_bases = sum(lengths)
            
            # Calculate mean quality for reads with quality scores
            qualities = [r.get_average_quality() for r in reads if r.get_average_quality() is not None]
            mean_quality = sum(qualities) / len(qualities) if qualities else None
            
            stats[tech] = {
                'count': len(reads),
                'total_bases': total_bases,
                'mean_length': total_bases / len(reads),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'mean_quality': mean_quality,
            }
        
        return stats
    
    def get_long_reads(self, min_length: int = 10000) -> List[SeqRead]:
        """
        Get all long reads above a length threshold.
        
        Args:
            min_length: Minimum length threshold
        
        Returns:
            List of long reads
        """
        long_reads = []
        for reads in self._reads_by_tech.values():
            long_reads.extend([r for r in reads if r.length >= min_length])
        return long_reads
    
    def get_ultra_long_reads(self) -> List[SeqRead]:
        """
        Get all ultra-long ONT reads (>100kb).
        
        These are particularly useful for scaffolding.
        
        Returns:
            List of ultra-long reads
        """
        from .technology_handling_core_module import ReadTechnology
        return self._reads_by_tech.get(ReadTechnology.ONT_ULTRALONG, [])
    
    def get_short_reads(self, max_length: int = 1000) -> List[SeqRead]:
        """
        Get all short reads below a length threshold.
        
        Args:
            max_length: Maximum length threshold
        
        Returns:
            List of short reads
        """
        short_reads = []
        for reads in self._reads_by_tech.values():
            short_reads.extend([r for r in reads if r.length <= max_length])
        return short_reads
    
    def get_corrected_reads(self) -> List[SeqRead]:
        """
        Get all reads that have been error-corrected.
        
        Returns:
            List of corrected reads
        """
        corrected = []
        for reads in self._reads_by_tech.values():
            corrected.extend([r for r in reads if r.is_corrected])
        return corrected
    
    def get_uncorrected_reads(self) -> List[SeqRead]:
        """
        Get all reads that have NOT been error-corrected.
        
        Returns:
            List of uncorrected reads
        """
        uncorrected = []
        for reads in self._reads_by_tech.values():
            uncorrected.extend([r for r in reads if not r.is_corrected])
        return uncorrected
    
    def get_statistics(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with statistics by technology and role
        """
        stats = {
            'total_reads': self._total_reads,
            'total_bases': self._total_bases,
            'mean_length': self._total_bases / self._total_reads if self._total_reads > 0 else 0,
            'by_technology': {},
            'by_role': {},
            'ultra_long_count': len(self.get_ultra_long_reads()),
            'corrected_count': len(self.get_corrected_reads()),
        }
        
        # Technology breakdown
        for tech, reads in self._reads_by_tech.items():
            if reads:
                total_bases = sum(r.length for r in reads)
                tech_value = tech.value if hasattr(tech, 'value') else str(tech)
                stats['by_technology'][tech_value] = {
                    'count': len(reads),
                    'total_bases': total_bases,
                    'mean_length': total_bases / len(reads),
                    'min_length': min(r.length for r in reads),
                    'max_length': max(r.length for r in reads),
                }
        
        # Role breakdown
        for role, reads in self._reads_by_role.items():
            role_value = role.value if hasattr(role, 'value') else str(role)
            stats['by_role'][role_value] = {
                'count': len(reads),
                'total_bases': sum(r.length for r in reads),
            }
        
        return stats
    
    def split_ont_reads(self) -> Tuple[List[SeqRead], List[SeqRead]]:
        """
        Split ONT reads into regular and ultra-long.
        
        Returns:
            Tuple of (regular_ont_reads, ultra_long_reads)
        """
        from .technology_handling_core_module import ReadTechnology
        regular = self._reads_by_tech.get(ReadTechnology.ONT_REGULAR, [])
        ultra_long = self._reads_by_tech.get(ReadTechnology.ONT_ULTRALONG, [])
        
        return regular, ultra_long
    
    def write_to_file(
        self,
        filepath: Path,
        technology: Optional['ReadTechnology'] = None,
        role: Optional['ReadRole'] = None,
        file_format: str = 'fastq',
        compress: bool = False
    ) -> int:
        """
        Write reads to file.
        
        Args:
            filepath: Output file path
            technology: Filter by technology (None = all)
            role: Filter by role (None = all)
            file_format: Output format ('fastq' or 'fasta')
            compress: Whether to compress output
        
        Returns:
            Number of reads written
        """
        # Collect reads based on filters
        if technology:
            reads = self.get_reads_by_technology(technology)
        elif role:
            reads = self.get_reads_by_role(role)
        else:
            # All reads
            reads = []
            for read_list in self._reads_by_tech.values():
                reads.extend(read_list)
        
        # Write to file
        if file_format == 'fastq':
            return write_fastq(iter(reads), filepath, compress)
        elif file_format == 'fasta':
            return write_fasta(iter(reads), filepath, compress)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def __len__(self) -> int:
        """Total number of reads in collection."""
        return self._total_reads
    
    def __repr__(self) -> str:
        """String representation."""
        tech_counts = {}
        for tech, reads in self._reads_by_tech.items():
            tech_value = tech.value if hasattr(tech, 'value') else str(tech)
            tech_counts[tech_value] = len(reads)
        return f"ReadCollection(total={self._total_reads}, by_tech={tech_counts})"
