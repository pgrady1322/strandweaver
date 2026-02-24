#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Read classification — technology and quality tier assignment.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

# =============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# =============================================================================

import logging
import subprocess
import json
import shutil
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 2: READ TYPE DEFINITIONS
# =============================================================================

class ReadTechnology(Enum):
    """Sequencing technology types."""
    ILLUMINA = "illumina"
    ANCIENT_DNA = "ancient"
    ONT_REGULAR = "ont"
    ONT_ULTRALONG = "ont_ultralong"
    PACBIO_HIFI = "pacbio"
    HI_C = "hic"  # Hi-C proximity ligation (for scaffolding only)
    UNKNOWN = "unknown"


class ReadRole(Enum):
    """Role of reads in assembly pipeline."""
    ERROR_CORRECTION = "correction"  # Used for error correction only
    ASSEMBLY = "assembly"            # Used for assembly
    SCAFFOLDING = "scaffolding"      # Used for scaffolding (e.g., ultra-long ONT)
    POLISHING = "polishing"          # Used for final polishing


@dataclass
class ReadTypeProfile:
    """
    Profile describing a specific read type's characteristics.
    
    Attributes:
        technology: Sequencing technology
        error_rate: Typical error rate (0-1)
        typical_length: Typical read length (bp)
        min_length_threshold: Minimum length to qualify for this type
        max_length_threshold: Maximum length for this type (None = unlimited)
        error_profile: Dominant error types (substitution, insertion, deletion)
        quality_encoding: Quality score encoding (phred33, phred64, etc.)
        roles: Assembly roles this read type can fulfill
    """
    technology: ReadTechnology
    error_rate: float
    typical_length: int
    min_length_threshold: int
    max_length_threshold: Optional[int]
    error_profile: str
    quality_encoding: str
    roles: List[ReadRole]
    
    def is_long_read(self) -> bool:
        """Check if this is a long-read technology."""
        return self.typical_length >= 10000
    
    def is_ultra_long(self) -> bool:
        """Check if this qualifies as ultra-long."""
        return self.technology == ReadTechnology.ONT_ULTRALONG


# Define standard read type profiles
READ_PROFILES = {
    ReadTechnology.ILLUMINA: ReadTypeProfile(
        technology=ReadTechnology.ILLUMINA,
        error_rate=0.001,  # ~0.1% error rate
        typical_length=150,
        min_length_threshold=50,
        max_length_threshold=500,
        error_profile="substitution",
        quality_encoding="phred33",
        roles=[ReadRole.ERROR_CORRECTION, ReadRole.ASSEMBLY, ReadRole.POLISHING]
    ),
    
    ReadTechnology.ANCIENT_DNA: ReadTypeProfile(
        technology=ReadTechnology.ANCIENT_DNA,
        error_rate=0.05,  # ~5% error rate (with damage)
        typical_length=50,
        min_length_threshold=30,
        max_length_threshold=150,
        error_profile="substitution_damage",  # C->T deamination
        quality_encoding="phred33",
        roles=[ReadRole.ASSEMBLY]
    ),
    
    ReadTechnology.ONT_REGULAR: ReadTypeProfile(
        technology=ReadTechnology.ONT_REGULAR,
        error_rate=0.05,  # ~5% error rate (modern ONT)
        typical_length=15000,
        min_length_threshold=1000,
        max_length_threshold=80000,  # < 80kb = regular ONT
        error_profile="insertion_deletion_homopolymer",
        quality_encoding="phred33",
        roles=[ReadRole.ASSEMBLY, ReadRole.SCAFFOLDING]
    ),
    
    ReadTechnology.ONT_ULTRALONG: ReadTypeProfile(
        technology=ReadTechnology.ONT_ULTRALONG,
        error_rate=0.05,  # ~5% error rate (same as regular ONT)
        typical_length=100000,
        min_length_threshold=80000,  # >= 80kb = ultra-long (accepts down to 80kb)
        max_length_threshold=None,  # No upper limit
        error_profile="insertion_deletion_homopolymer",
        quality_encoding="phred33",
        roles=[ReadRole.SCAFFOLDING, ReadRole.ASSEMBLY]  # Scaffolding priority
    ),
    
    ReadTechnology.PACBIO_HIFI: ReadTypeProfile(
        technology=ReadTechnology.PACBIO_HIFI,
        error_rate=0.001,  # ~0.1% error rate (HiFi)
        typical_length=15000,
        min_length_threshold=1000,
        max_length_threshold=None,
        error_profile="random",
        quality_encoding="phred33",
        roles=[ReadRole.ASSEMBLY, ReadRole.POLISHING]
    ),
    
    ReadTechnology.HI_C: ReadTypeProfile(
        technology=ReadTechnology.HI_C,
        error_rate=0.001,  # ~0.1% (Illumina-based)
        typical_length=150,
        min_length_threshold=50,
        max_length_threshold=500,
        error_profile="substitution",
        quality_encoding="phred33",
        roles=[ReadRole.SCAFFOLDING]  # ONLY for scaffolding - not corrected
    ),
}


def classify_read_type(length: int, technology: ReadTechnology) -> ReadTechnology:
    """
    Classify a read into specific type based on length and base technology.
    
    This is important for ONT reads where ultra-long reads need special handling.
    
    Args:
        length: Read length in base pairs
        technology: Base technology type
    
    Returns:
        Specific read technology classification
    
    Examples:
        >>> classify_read_type(50000, ReadTechnology.ONT_REGULAR)
        ReadTechnology.ONT_REGULAR
        >>> classify_read_type(150000, ReadTechnology.ONT_REGULAR)
        ReadTechnology.ONT_ULTRALONG
        >>> classify_read_type(90000, ReadTechnology.ONT_REGULAR)
        ReadTechnology.ONT_ULTRALONG
    """
    # Special handling for ONT reads - distinguish ultra-long
    if technology in (ReadTechnology.ONT_REGULAR, ReadTechnology.ONT_ULTRALONG):
        if length >= READ_PROFILES[ReadTechnology.ONT_ULTRALONG].min_length_threshold:
            return ReadTechnology.ONT_ULTRALONG
        else:
            return ReadTechnology.ONT_REGULAR
    
    # For other technologies, return as-is
    return technology


def get_read_profile(technology: ReadTechnology) -> ReadTypeProfile:
    """
    Get the profile for a specific read technology.
    
    Args:
        technology: Read technology type
    
    Returns:
        Read type profile
    """
    return READ_PROFILES.get(technology, READ_PROFILES[ReadTechnology.ILLUMINA])


def infer_technology_from_length(length: int, quality_scores: Optional[str] = None) -> ReadTechnology:
    """
    Infer sequencing technology from read length and quality scores.
    
    This is a heuristic approach when technology is not explicitly specified.
    
    Args:
        length: Read length in base pairs
        quality_scores: Optional quality score string
    
    Returns:
        Inferred read technology
    """
    # Very short reads - likely Illumina or ancient DNA
    if length < 200:
        # Ancient DNA is typically < 100bp with low quality
        if quality_scores and sum(ord(c) - 33 for c in quality_scores) / length < 25:
            return ReadTechnology.ANCIENT_DNA
        return ReadTechnology.ILLUMINA
    
    # Medium reads (200-1000bp) - could be long Illumina
    elif length < 1000:
        return ReadTechnology.ILLUMINA
    
    # Long reads (1kb-80kb) - ONT or PacBio
    elif length < 80000:
        # High quality long reads - likely PacBio HiFi
        if quality_scores and sum(ord(c) - 33 for c in quality_scores) / length > 30:
            return ReadTechnology.PACBIO_HIFI
        # Lower quality - likely ONT
        return ReadTechnology.ONT_REGULAR
    
    # Ultra-long reads (>=80kb) - ONT ultra-long
    else:
        return ReadTechnology.ONT_ULTRALONG


# =============================================================================
# SECTION 3: TECHNOLOGY VALIDATION AND PARSING
# =============================================================================

# Technology name mappings (handles common variations)
TECHNOLOGY_ALIASES = {
    # Illumina variations
    'illumina': ReadTechnology.ILLUMINA,
    'ill': ReadTechnology.ILLUMINA,
    'short': ReadTechnology.ILLUMINA,
    
    # Ancient DNA variations
    'ancient': ReadTechnology.ANCIENT_DNA,
    'ancient_dna': ReadTechnology.ANCIENT_DNA,
    'adna': ReadTechnology.ANCIENT_DNA,
    
    # ONT variations
    'ont': ReadTechnology.ONT_REGULAR,
    'nanopore': ReadTechnology.ONT_REGULAR,
    'minion': ReadTechnology.ONT_REGULAR,
    'promethion': ReadTechnology.ONT_REGULAR,
    
    # ONT ultra-long variations
    'ont_ultralong': ReadTechnology.ONT_ULTRALONG,
    'ont_ul': ReadTechnology.ONT_ULTRALONG,
    'ultralong': ReadTechnology.ONT_ULTRALONG,
    
    # PacBio variations
    'pacbio': ReadTechnology.PACBIO_HIFI,
    'pacbio_hifi': ReadTechnology.PACBIO_HIFI,
    'hifi': ReadTechnology.PACBIO_HIFI,
    'ccs': ReadTechnology.PACBIO_HIFI,
    
    # Hi-C variations
    'hic': ReadTechnology.HI_C,
    'hi-c': ReadTechnology.HI_C,
    'hic_reads': ReadTechnology.HI_C,
    'proximity_ligation': ReadTechnology.HI_C,
}


def parse_technology(tech_str: str) -> ReadTechnology:
    """
    Parse a technology string into ReadTechnology enum.
    
    Handles various naming conventions and provides helpful error messages.
    
    Args:
        tech_str: Technology string (e.g., "illumina", "ONT", "HiFi")
    
    Returns:
        ReadTechnology enum value
    
    Raises:
        ValueError: If technology string is not recognized
    
    Examples:
        >>> parse_technology("illumina")
        ReadTechnology.ILLUMINA
        >>> parse_technology("ONT")
        ReadTechnology.ONT_REGULAR
        >>> parse_technology("hifi")
        ReadTechnology.PACBIO_HIFI
    """
    # Normalize to lowercase
    tech_lower = tech_str.lower().strip()
    
    # Check if it's in our alias mapping
    if tech_lower in TECHNOLOGY_ALIASES:
        return TECHNOLOGY_ALIASES[tech_lower]
    
    # Not recognized - provide helpful error
    raise ValueError(
        f"Unknown technology: '{tech_str}'\n\n"
        f"Supported technologies:\n"
        f"  • illumina (or: ill, short)\n"
        f"  • ancient (or: ancient_dna, adna)\n"
        f"  • ont (or: nanopore, minion, promethion)\n"
        f"  • ont_ultralong (or: ont_ul, ultralong)\n"
        f"  • pacbio (or: pacbio_hifi, hifi, ccs)\n"
        f"  • hic (or: hi-c, hic_reads, proximity_ligation)\n"
        f"  • auto (auto-detect from read characteristics)"
    )


def validate_technology_specs(
    reads_files: List[str],
    technologies: List[str]
) -> List[ReadTechnology]:
    """
    Validate and convert technology specifications.
    
    Ensures technology list matches reads files and converts strings to enums.
    
    Args:
        reads_files: List of read file paths
        technologies: List of technology strings (or empty for auto-detect)
    
    Returns:
        List of ReadTechnology enum values (one per reads file)
    
    Raises:
        ValueError: If validation fails (mismatched counts, invalid names)
    """
    # If no technologies specified, return auto-detect for all files
    if not technologies:
        return [ReadTechnology.UNKNOWN] * len(reads_files)
    
    # Validate count matches
    if len(technologies) != len(reads_files):
        raise ValueError(
            f"Technology count mismatch:\n"
            f"  Reads files: {len(reads_files)}\n"
            f"  Technologies: {len(technologies)}\n\n"
            f"You must specify one --technology per --reads file, in the same order.\n\n"
            f"Example:\n"
            f"  strandweaver pipeline -r illumina.fq -r ont.fq \\\n"
            f"      --technology illumina --technology ont -o output/"
        )
    
    # Parse each technology string
    parsed = []
    for i, tech_str in enumerate(technologies, 1):
        if tech_str.lower() == 'auto':
            parsed.append(ReadTechnology.UNKNOWN)
        else:
            try:
                parsed.append(parse_technology(tech_str))
            except ValueError as e:
                raise ValueError(f"Error in technology #{i}: {e}")
    
    return parsed


def auto_detect_technology(reads_file: str, sample_size: int = 100) -> ReadTechnology:
    """
    Auto-detect technology from read file characteristics.
    
    Samples reads and infers technology based on length and quality.
    
    Args:
        reads_file: Path to FASTQ/FASTA file
        sample_size: Number of reads to sample for detection
    
    Returns:
        Inferred ReadTechnology
    """
    reads_path = Path(reads_file)
    
    # Only works for FASTQ files (need quality scores for best detection)
    if reads_path.suffix.lower() not in ['.fastq', '.fq', '.gz']:
        # For FASTA, just use length-based heuristic
        from .io_core_module import read_fasta
        lengths = []
        for i, read in enumerate(read_fasta(reads_file)):
            if i >= sample_size:
                break
            lengths.append(len(read.sequence))
        
        if not lengths:
            return ReadTechnology.UNKNOWN
        
        avg_length = sum(lengths) / len(lengths)
        return infer_technology_from_length(int(avg_length))
    
    # For FASTQ, use both length and quality
    from .io_core_module import read_fastq
    detections = []
    
    for i, read in enumerate(read_fastq(reads_file)):
        if i >= sample_size:
            break
        
        tech = infer_technology_from_length(
            len(read.sequence),
            read.quality
        )
        detections.append(tech)
    
    if not detections:
        return ReadTechnology.UNKNOWN
    
    # Return most common detection
    most_common = Counter(detections).most_common(1)[0][0]
    return most_common


def detect_technologies(
    reads_files: List[str],
    technologies: List[ReadTechnology],
    sample_size: int = 100
) -> List[ReadTechnology]:
    """
    Detect technologies for files marked as UNKNOWN.
    
    Args:
        reads_files: List of read file paths
        technologies: List of ReadTechnology (may contain UNKNOWN)
        sample_size: Number of reads to sample per file
    
    Returns:
        List of ReadTechnology with UNKNOWN replaced by detections
    """
    result = []
    
    for reads_file, tech in zip(reads_files, technologies):
        if tech == ReadTechnology.UNKNOWN:
            print(f"  Auto-detecting technology for: {reads_file}")
            detected = auto_detect_technology(reads_file, sample_size)
            print(f"    → Detected: {detected.value}")
            result.append(detected)
        else:
            result.append(tech)
    
    return result


def format_technology_summary(
    reads_files: List[str],
    technologies: List[ReadTechnology]
) -> str:
    """
    Format a human-readable summary of reads and technologies.
    
    Args:
        reads_files: List of read file paths
        technologies: List of ReadTechnology enums
    
    Returns:
        Formatted summary string
    """
    lines = [f"Input Files ({len(reads_files)}):"]
    
    for i, (reads, tech) in enumerate(zip(reads_files, technologies), 1):
        tech_name = tech.value.upper()
        lines.append(f"  {i}. {reads}")
        lines.append(f"     Technology: {tech_name}")
    
    return "\n".join(lines)


# =============================================================================
# SECTION 4: NANOPORE METADATA STRUCTURES
# =============================================================================

class FlowCellType(Enum):
    """Known Nanopore flow cell types."""
    R9_4_1 = "R9.4.1"
    R9_5 = "R9.5"
    R10_3 = "R10.3"
    R10_4 = "R10.4"
    R10_4_1 = "R10.4.1"
    UNKNOWN = "unknown"


class BasecallerType(Enum):
    """Known Nanopore basecallers."""
    GUPPY = "guppy"
    DORADO = "dorado"
    BONITO = "bonito"
    UNKNOWN = "unknown"


class BasecallerAccuracy(Enum):
    """Basecaller accuracy modes."""
    SUP = "sup"      # Super accurate (slowest, most accurate)
    HAC = "hac"      # High accuracy (balanced)
    FAST = "fast"    # Fast mode (fastest, least accurate)
    UNKNOWN = "unknown"


# Expected error rates and characteristics for different ONT configurations
# Format: (flow_cell, accuracy) -> (expected_error_rate, homopolymer_error_rate)
#
# SOURCES AND CITATIONS:
# - Shafin et al. 2020 (Nature Biotechnology): Nanopolish accuracy benchmarks
# - Logsdon et al. 2020 (Science): T2T-CHM13 assembly with R9.4.1 data
# - Nurk et al. 2022 (Science): Complete T2T human genome (R9.4.1 + R10.3)
# - ONT Q20+ Chemistry (2022): R10.4 benchmarks showing modal Q23-Q25
# - ONT Q30 Chemistry (2023): R10.4.1 benchmarks showing modal Q25-Q30
# - Wang et al. 2021 (Genome Research): Comprehensive ONT error profiling
#
# NOTES:
# - Values marked [PUBLISHED] are from peer-reviewed papers or ONT benchmarks
# - Values marked [EXTRAPOLATED] are computed from published data using known relationships
# - Homopolymer error rates are typically 3-5x higher for R9 chemistry, 2-3x for R10
# - FAST mode is typically 3-5x worse than SUP mode
# - Modal accuracy (most common read quality) used, not mean/median
#
ONT_ERROR_PROFILES: Dict[Tuple[FlowCellType, BasecallerAccuracy], Tuple[float, float]] = {
    # R9.4.1 flow cell
    # [PUBLISHED] Shafin et al. 2020: Guppy SUP achieves ~Q18-Q20 (1.0-1.6% error)
    # [PUBLISHED] Logsdon et al. 2020: R9.4.1 SUP data used for CHM13, ~1.2% modal
    (FlowCellType.R9_4_1, BasecallerAccuracy.SUP): (0.012, 0.045),  # 1.2% overall, 4.5% homopolymer
    
    # [PUBLISHED] Wang et al. 2021: Guppy HAC achieves ~Q15-Q16 (2.5-3.2% error)
    (FlowCellType.R9_4_1, BasecallerAccuracy.HAC): (0.028, 0.10),   # 2.8% overall, 10% homopolymer
    
    # [EXTRAPOLATED] FAST mode ~4x worse than SUP, limited published data
    (FlowCellType.R9_4_1, BasecallerAccuracy.FAST): (0.05, 0.18),   # 5% overall, 18% homopolymer
    
    # R9.5 flow cell (similar chemistry to R9.4.1, minor improvements)
    # [EXTRAPOLATED] ~10% improvement over R9.4.1 based on ONT specifications
    (FlowCellType.R9_5, BasecallerAccuracy.SUP): (0.011, 0.040),
    (FlowCellType.R9_5, BasecallerAccuracy.HAC): (0.025, 0.090),
    (FlowCellType.R9_5, BasecallerAccuracy.FAST): (0.045, 0.16),
    
    # R10.3 flow cell (intermediate chemistry, before Q20+)
    # [PUBLISHED] Nurk et al. 2022: R10.3 data showed ~0.6-0.8% modal accuracy
    # [EXTRAPOLATED] Intermediate between R9.4.1 and R10.4 Q20+ chemistry
    (FlowCellType.R10_3, BasecallerAccuracy.SUP): (0.007, 0.025),   # 0.7% overall, 2.5% homopolymer
    (FlowCellType.R10_3, BasecallerAccuracy.HAC): (0.018, 0.055),
    (FlowCellType.R10_3, BasecallerAccuracy.FAST): (0.035, 0.11),
    
    # R10.4 flow cell (Q20+ chemistry)
    # [PUBLISHED] ONT Q20+ Chemistry (2022): Modal Q23-Q25 (0.3-0.5% error)
    # [PUBLISHED] Multiple community benchmarks confirm ~0.4% modal for SUP
    (FlowCellType.R10_4, BasecallerAccuracy.SUP): (0.004, 0.012),   # 0.4% overall, 1.2% homopolymer
    
    # [EXTRAPOLATED] HAC ~2.5x worse than SUP based on R9.4.1 ratios
    (FlowCellType.R10_4, BasecallerAccuracy.HAC): (0.010, 0.028),
    
    # [EXTRAPOLATED] FAST ~4x worse than SUP
    (FlowCellType.R10_4, BasecallerAccuracy.FAST): (0.016, 0.045),
    
    # R10.4.1 flow cell (Q30 chemistry, latest generation)
    # [PUBLISHED] ONT Q30 Chemistry (2023): Modal Q28-Q30 (0.1-0.16% error)
    # [PUBLISHED] Community benchmarks show ~0.15% modal for Dorado SUP
    (FlowCellType.R10_4_1, BasecallerAccuracy.SUP): (0.0015, 0.0045),  # 0.15% overall, 0.45% homopolymer
    
    # [EXTRAPOLATED] HAC ~2.5x worse than SUP
    (FlowCellType.R10_4_1, BasecallerAccuracy.HAC): (0.004, 0.011),
    
    # [EXTRAPOLATED] FAST ~4x worse than SUP
    (FlowCellType.R10_4_1, BasecallerAccuracy.FAST): (0.006, 0.018),
}

# Default conservative estimates for unknown configurations
DEFAULT_ONT_ERROR_PROFILE = (0.02, 0.05)  # 2% overall, 5% homopolymer


@dataclass
class NanoporeMetadata:
    """
    Metadata about Nanopore sequencing run.
    
    This information is used to optimize error profiling and correction
    parameters based on known characteristics of different flow cell
    and basecaller combinations.
    
    Attributes:
        flow_cell: Flow cell type (e.g., "R9.4.1", "R10.4.1")
        basecaller: Basecaller used (e.g., "guppy", "dorado")
        accuracy: Basecaller accuracy mode ("sup", "hac", "fast")
        basecaller_version: Optional basecaller version string
        source: How metadata was obtained ("manual", "longbow", "inferred")
    """
    flow_cell: FlowCellType
    basecaller: BasecallerType
    accuracy: BasecallerAccuracy
    basecaller_version: Optional[str] = None
    source: str = "manual"
    
    def get_expected_error_rates(self) -> Tuple[float, float]:
        """
        Get expected error rates for this configuration.
        
        Returns:
            Tuple of (overall_error_rate, homopolymer_error_rate)
        """
        key = (self.flow_cell, self.accuracy)
        return ONT_ERROR_PROFILES.get(key, DEFAULT_ONT_ERROR_PROFILE)
    
    def get_kmer_threshold_modifier(self) -> float:
        """
        Get k-mer solid threshold modifier based on accuracy.
        
        Higher accuracy = lower threshold (can trust lower-frequency k-mers)
        Lower accuracy = higher threshold (need higher frequency for confidence)
        
        Returns:
            Multiplier for base k-mer threshold (1.0 = no change)
        """
        if self.accuracy == BasecallerAccuracy.SUP:
            return 0.7  # Can trust k-mers at 70% of default threshold
        elif self.accuracy == BasecallerAccuracy.HAC:
            return 1.0  # Use default threshold
        elif self.accuracy == BasecallerAccuracy.FAST:
            return 1.5  # Need higher frequency for confidence
        else:
            return 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'flow_cell': self.flow_cell.value,
            'basecaller': self.basecaller.value,
            'accuracy': self.accuracy.value,
            'basecaller_version': self.basecaller_version,
            'source': self.source,
            'expected_error_rate': self.get_expected_error_rates()[0],
            'expected_homopolymer_error_rate': self.get_expected_error_rates()[1]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NanoporeMetadata':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            flow_cell=FlowCellType(data['flow_cell']),
            basecaller=BasecallerType(data['basecaller']),
            accuracy=BasecallerAccuracy(data['accuracy']),
            basecaller_version=data.get('basecaller_version'),
            source=data.get('source', 'manual')
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [
            f"Flow cell: {self.flow_cell.value}",
            f"Basecaller: {self.basecaller.value}",
            f"Accuracy: {self.accuracy.value.upper()}"
        ]
        
        if self.basecaller_version:
            parts.append(f"Version: {self.basecaller_version}")
        
        error_rate, hp_error_rate = self.get_expected_error_rates()
        parts.append(f"Expected error rate: {error_rate:.2%}")
        parts.append(f"Expected homopolymer error rate: {hp_error_rate:.2%}")
        
        return " | ".join(parts)


def parse_nanopore_metadata(
    flow_cell: Optional[str] = None,
    basecaller: Optional[str] = None,
    accuracy: Optional[str] = None,
    basecaller_version: Optional[str] = None
) -> Optional[NanoporeMetadata]:
    """
    Parse and validate Nanopore metadata from string inputs.
    
    Args:
        flow_cell: Flow cell type (e.g., "R9.4.1", "r10.4.1")
        basecaller: Basecaller name (e.g., "guppy", "dorado")
        accuracy: Accuracy mode (e.g., "sup", "hac", "fast")
        basecaller_version: Optional version string
    
    Returns:
        NanoporeMetadata object if all required fields provided, None otherwise
    
    Raises:
        ValueError: If invalid values provided
    """
    # If nothing provided, return None (will use standard ONT profiling)
    if not any([flow_cell, basecaller, accuracy]):
        return None
    
    # If partial info, that's an error
    if not all([flow_cell, basecaller, accuracy]):
        missing = []
        if not flow_cell: missing.append("--ont-flowcell")
        if not basecaller: missing.append("--ont-basecaller")
        if not accuracy: missing.append("--ont-accuracy")
        raise ValueError(
            f"Incomplete ONT metadata. Missing: {', '.join(missing)}. "
            "Either provide all three flags, use --ont-detect, or omit for standard ONT profiling."
        )
    
    # Parse flow cell
    flow_cell_clean = flow_cell.upper().replace('R', 'R')  # Normalize
    try:
        flow_cell_enum = FlowCellType(flow_cell_clean)
    except ValueError:
        valid = [fc.value for fc in FlowCellType if fc != FlowCellType.UNKNOWN]
        raise ValueError(
            f"Unknown flow cell type: '{flow_cell}'. "
            f"Valid options: {', '.join(valid)}"
        )
    
    # Parse basecaller
    basecaller_clean = basecaller.lower()
    try:
        basecaller_enum = BasecallerType(basecaller_clean)
    except ValueError:
        valid = [bc.value for bc in BasecallerType if bc != BasecallerType.UNKNOWN]
        raise ValueError(
            f"Unknown basecaller: '{basecaller}'. "
            f"Valid options: {', '.join(valid)}"
        )
    
    # Parse accuracy
    accuracy_clean = accuracy.lower()
    try:
        accuracy_enum = BasecallerAccuracy(accuracy_clean)
    except ValueError:
        valid = [acc.value for acc in BasecallerAccuracy if acc != BasecallerAccuracy.UNKNOWN]
        raise ValueError(
            f"Unknown accuracy mode: '{accuracy}'. "
            f"Valid options: {', '.join(valid)}"
        )
    
    return NanoporeMetadata(
        flow_cell=flow_cell_enum,
        basecaller=basecaller_enum,
        accuracy=accuracy_enum,
        basecaller_version=basecaller_version,
        source="manual"
    )


# =============================================================================
# SECTION 5: LONGBOW INTEGRATION
# =============================================================================

def is_longbow_available() -> bool:
    """
    Check if LongBow is installed and available.
    
    Returns:
        True if LongBow executable is found in PATH
    """
    return shutil.which('longbow') is not None


def run_longbow(reads_file: Path, sample_size: int = 1000) -> Optional[Dict]:
    """
    Run LongBow to detect basecaller information from reads.
    
    Args:
        reads_file: Path to FASTQ file
        sample_size: Number of reads to sample for detection
    
    Returns:
        Dictionary with LongBow results, or None if detection failed
    
    Raises:
        RuntimeError: If LongBow is not installed
        subprocess.CalledProcessError: If LongBow execution fails
    """
    if not is_longbow_available():
        raise RuntimeError(
            "LongBow is not installed or not found in PATH. "
            "Install with: pip install longbow-ont\n"
            "Or provide ONT metadata manually using:\n"
            "  --ont-flowcell, --ont-basecaller, --ont-accuracy"
        )
    
    try:
        # Run LongBow with JSON output
        # Note: Adjust command based on actual LongBow CLI
        # This is a placeholder - need to verify actual LongBow interface
        cmd = [
            'longbow',
            'detect',
            '--input', str(reads_file),
            '--sample-size', str(sample_size),
            '--output-format', 'json'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=60  # 1 minute timeout
        )
        
        # Parse JSON output
        return json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("LongBow detection timed out after 60 seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LongBow execution failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LongBow output: {e}")


def parse_longbow_output(longbow_result: Dict) -> NanoporeMetadata:
    """
    Parse LongBow output into NanoporeMetadata.
    
    Args:
        longbow_result: Dictionary from LongBow JSON output
    
    Returns:
        NanoporeMetadata object
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Note: This is placeholder code - actual LongBow output format needs verification
    # Adjust field names based on real LongBow output
    
    # Extract basecaller info
    basecaller_str = longbow_result.get('basecaller', '').lower()
    if 'guppy' in basecaller_str:
        basecaller = BasecallerType.GUPPY
    elif 'dorado' in basecaller_str:
        basecaller = BasecallerType.DORADO
    elif 'bonito' in basecaller_str:
        basecaller = BasecallerType.BONITO
    else:
        basecaller = BasecallerType.UNKNOWN
    
    # Extract accuracy mode
    model_str = longbow_result.get('model', '').lower()
    if 'sup' in model_str or 'super' in model_str:
        accuracy = BasecallerAccuracy.SUP
    elif 'hac' in model_str or 'high' in model_str:
        accuracy = BasecallerAccuracy.HAC
    elif 'fast' in model_str:
        accuracy = BasecallerAccuracy.FAST
    else:
        accuracy = BasecallerAccuracy.UNKNOWN
    
    # Extract flow cell type
    flowcell_str = longbow_result.get('flowcell', '').upper()
    
    # Try to match known flow cell patterns
    if 'R9.4.1' in flowcell_str or 'FLO-MIN106' in flowcell_str:
        flow_cell = FlowCellType.R9_4_1
    elif 'R10.4.1' in flowcell_str:
        flow_cell = FlowCellType.R10_4_1
    elif 'R10.4' in flowcell_str or 'FLO-MIN114' in flowcell_str:
        flow_cell = FlowCellType.R10_4
    elif 'R10.3' in flowcell_str:
        flow_cell = FlowCellType.R10_3
    elif 'R9.5' in flowcell_str:
        flow_cell = FlowCellType.R9_5
    else:
        # Try to extract version number with regex
        match = re.search(r'R(\d+\.?\d*\.?\d*)', flowcell_str)
        if match:
            version = match.group(1)
            # Try exact match
            try:
                flow_cell = FlowCellType(f"R{version}")
            except ValueError:
                flow_cell = FlowCellType.UNKNOWN
        else:
            flow_cell = FlowCellType.UNKNOWN
    
    # Get basecaller version if available
    basecaller_version = longbow_result.get('basecaller_version')
    
    return NanoporeMetadata(
        flow_cell=flow_cell,
        basecaller=basecaller,
        accuracy=accuracy,
        basecaller_version=basecaller_version,
        source="longbow"
    )


def detect_nanopore_metadata_with_longbow(
    reads_file: Path,
    sample_size: int = 1000
) -> Optional[NanoporeMetadata]:
    """
    Auto-detect Nanopore metadata using LongBow.
    
    This is a high-level wrapper that:
    1. Checks if LongBow is available
    2. Runs LongBow detection
    3. Parses results into NanoporeMetadata
    
    Args:
        reads_file: Path to FASTQ file
        sample_size: Number of reads to sample
    
    Returns:
        NanoporeMetadata object, or None if detection failed
    """
    if not is_longbow_available():
        print("⚠️  LongBow not found - install with: pip install longbow-ont")
        print("    Falling back to standard ONT profiling")
        return None
    
    try:
        print("🔍 Running LongBow to detect ONT metadata...")
        longbow_result = run_longbow(reads_file, sample_size)
        
        if not longbow_result:
            print("⚠️  LongBow detection returned no results")
            return None
        
        metadata = parse_longbow_output(longbow_result)
        
        print(f"✓ LongBow detection complete:")
        print(f"  {metadata}")
        
        return metadata
        
    except RuntimeError as e:
        print(f"⚠️  LongBow detection failed: {e}")
        print("    Falling back to standard ONT profiling")
        return None
    except Exception as e:
        print(f"⚠️  Unexpected error during LongBow detection: {e}")
        print("    Falling back to standard ONT profiling")
        return None


def detect_from_fastq_headers(reads_file: Path, max_reads: int = 100) -> Optional[NanoporeMetadata]:
    """
    Attempt to detect metadata from FASTQ headers.
    
    Many ONT FASTQ files include basecaller information in read headers.
    This is a fallback when LongBow is not available.
    
    Args:
        reads_file: Path to FASTQ file
        max_reads: Maximum number of reads to check
    
    Returns:
        NanoporeMetadata if detected, None otherwise
    """
    # This is a simplified fallback
    # Real implementation would parse @ lines for basecaller info
    # Format varies: @read_id basecall=guppy model=dna_r9.4.1_450bps_sup.cfg
    
    try:
        basecaller = None
        accuracy = None
        flow_cell = None
        basecaller_version = None
        
        with open(reads_file, 'r') as f:
            reads_checked = 0
            for line in f:
                if line.startswith('@'):
                    # Parse header
                    if 'guppy' in line.lower():
                        basecaller = BasecallerType.GUPPY
                    elif 'dorado' in line.lower():
                        basecaller = BasecallerType.DORADO
                    
                    if 'sup' in line.lower():
                        accuracy = BasecallerAccuracy.SUP
                    elif 'hac' in line.lower():
                        accuracy = BasecallerAccuracy.HAC
                    elif 'fast' in line.lower():
                        accuracy = BasecallerAccuracy.FAST
                    
                    if 'r9.4.1' in line.lower():
                        flow_cell = FlowCellType.R9_4_1
                    elif 'r10.4.1' in line.lower():
                        flow_cell = FlowCellType.R10_4_1
                    elif 'r10.4' in line.lower():
                        flow_cell = FlowCellType.R10_4
                    
                    reads_checked += 1
                    if reads_checked >= max_reads:
                        break
                    
                    # If we found everything, stop early
                    if all([basecaller, accuracy, flow_cell]):
                        break
        
        # Return metadata only if we found all required fields
        if all([basecaller, accuracy, flow_cell]):
            return NanoporeMetadata(
                flow_cell=flow_cell,
                basecaller=basecaller,
                accuracy=accuracy,
                basecaller_version=basecaller_version,
                source="inferred"
            )
        
        return None
        
    except Exception:
        return None


def classify_read_technology(read_length: int, quality_scores: Optional[List[int]] = None) -> ReadTechnology:
    """
    Classify read technology based on read characteristics.
    
    Args:
        read_length: Length of the read in base pairs
        quality_scores: Optional quality scores for the read
        
    Returns:
        Classified ReadTechnology
    """
    # Ultra-long reads (>50kb)
    if read_length >= 50000:
        return ReadTechnology.ONT_ULTRALONG
    
    # Long reads (>1kb)
    if read_length >= 1000:
        # Could be ONT regular or PacBio HiFi
        # Use quality scores if available to distinguish
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            # HiFi typically has Q20+ (99% accuracy)
            if avg_quality >= 20:
                return ReadTechnology.PACBIO_HIFI
        return ReadTechnology.ONT_REGULAR
    
    # Short reads (<1kb)
    if read_length < 1000:
        # Could be Illumina or ancient DNA
        if quality_scores:
            # Ancient DNA typically has lower quality and damage patterns
            # For now, default to Illumina
            logger.debug(
                "Ancient DNA vs Illumina distinction not yet implemented; "
                "defaulting to Illumina for short reads with quality scores"
            )
        return ReadTechnology.ILLUMINA
    
    return ReadTechnology.UNKNOWN


def detect_technology_from_header(header: str) -> str:
    """
    Detect sequencing technology from FASTQ header.
    
    Args:
        header: FASTQ header line (starting with @)
        
    Returns:
        Detected technology as string (lowercase)
    """
    header_lower = header.lower()
    
    # Oxford Nanopore indicators (including runid pattern)
    if any(x in header_lower for x in ['ont', 'nanopore', 'minion', 'promethion', 'guppy', 'dorado', 'runid=', 'ch=', 'start_time=']):
        return 'ont'
    
    # PacBio indicators (including m64011 pattern)
    if any(x in header_lower for x in ['pacbio', 'hifi', 'ccs']) or (header_lower.startswith('@m') and '/ccs' in header_lower):
        return 'pacbio'
    
    # Illumina indicators (including SRR accessions and machine IDs)
    illumina_patterns = ['illumina', 'nextseq', 'novaseq', 'hiseq', 'miseq']
    illumina_prefixes = ['@srr', '@err', '@drr', '@m00', '@k00', '@a00', '@d00', '@e00', '@ns', '@vl', '@mn', '@nb']
    if any(x in header_lower for x in illumina_patterns) or any(header_lower.startswith(x) for x in illumina_prefixes):
        return 'illumina'
    
    # Hi-C indicators
    if any(x in header_lower for x in ['hic', 'hi-c', 'omni-c', 'dovetail']):
        return 'hic'
    
    return 'unknown'


def parse_quality_scores(quality_string: str, encoding: str = 'phred33') -> List[int]:
    """
    Parse quality scores from FASTQ quality string.
    
    Args:
        quality_string: Quality string from FASTQ
        encoding: Quality encoding (phred33 or phred64)
        
    Returns:
        List of integer quality scores
    """
    offset = 33 if encoding == 'phred33' else 64
    return [ord(char) - offset for char in quality_string]


def has_low_quality_bases(quality_scores, threshold: int = 20, max_low_quality_fraction: float = 0.1) -> bool:
    """
    Check if a read has too many low-quality bases.
    
    Args:
        quality_scores: List of quality scores (ints) or quality string
        threshold: Quality score threshold (default Q20)
        max_low_quality_fraction: Maximum fraction of bases below threshold
        
    Returns:
        True if read has too many low-quality bases
    """
    if not quality_scores:
        return False
    
    # Handle string input (convert to ints)
    if isinstance(quality_scores, str):
        quality_scores = parse_quality_scores(quality_scores)
    
    low_quality_count = sum(1 for q in quality_scores if q < threshold)
    low_quality_fraction = low_quality_count / len(quality_scores)
    
    return low_quality_fraction > max_low_quality_fraction

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
