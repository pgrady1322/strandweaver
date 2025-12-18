"""
Feature extraction from sequencing reads for ML-based assembly optimization.

This module extracts comprehensive features from FASTQ/FASTA files that are used
to predict optimal k-mer sizes for different assembly stages.

Key features extracted:
- Read length statistics (mean, median, N50, distribution)
- Base quality metrics (mean Q, error rate estimation)
- Coverage depth estimation
- GC content and sequence composition
- Genome size estimation (via k-mer counting)
- Read type detection (HiFi/ONT/Illumina)
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gzip

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReadFeatures:
    """
    Comprehensive features extracted from sequencing reads.
    
    These features are used as input to the adaptive k-mer predictor.
    All features are normalized and scaled appropriately for ML models.
    """
    
    # Read length statistics
    mean_read_length: float
    median_read_length: float
    read_length_n50: float
    min_read_length: int
    max_read_length: int
    read_length_std: float
    
    # Quality metrics
    mean_base_quality: float
    median_base_quality: float
    estimated_error_rate: float
    
    # Coverage and depth
    total_bases: int
    num_reads: int
    estimated_genome_size: Optional[int]
    estimated_coverage: Optional[float]
    
    # Sequence composition
    gc_content: float
    gc_std: float  # GC variance across reads
    
    # Read type characteristics
    read_type: str  # 'hifi', 'ont', 'illumina', 'unknown'
    is_paired_end: bool
    
    # K-mer spectrum features (optional, for genome complexity)
    kmer_spectrum_peak: Optional[int] = None  # Peak coverage in k-mer histogram
    kmer_diversity: Optional[float] = None    # Unique k-mers / total k-mers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for ML input."""
        return asdict(self)
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to numpy array for ML models.
        
        Returns all 19 features in same order as CSV columns:
        1. mean_read_length
        2. median_read_length
        3. read_length_n50
        4. min_read_length
        5. max_read_length
        6. read_length_std
        7. mean_base_quality
        8. median_base_quality
        9. estimated_error_rate
        10. total_bases
        11. num_reads
        12. estimated_genome_size
        13. estimated_coverage
        14. gc_content
        15. gc_std
        16. read_type (encoded: hifi=0, ont=1, illumina=2)
        17. is_paired_end (bool as int)
        18. kmer_spectrum_peak
        19. kmer_diversity
        """
        # Encode read_type as numeric
        read_type_encoding = {
            'hifi': 0,
            'ont': 1,
            'illumina': 2,
            'unknown': 3
        }
        
        return np.array([
            self.mean_read_length,
            self.median_read_length,
            self.read_length_n50,
            self.min_read_length,
            self.max_read_length,
            self.read_length_std,
            self.mean_base_quality,
            self.median_base_quality,
            self.estimated_error_rate,
            self.total_bases,
            self.num_reads,
            self.estimated_genome_size if self.estimated_genome_size else 0,
            self.estimated_coverage if self.estimated_coverage else 0.0,
            self.gc_content,
            self.gc_std,
            read_type_encoding.get(self.read_type, 3),
            1.0 if self.is_paired_end else 0.0,
            self.kmer_spectrum_peak if self.kmer_spectrum_peak else 0,
            self.kmer_diversity if self.kmer_diversity else 0.0,
        ])


class FeatureExtractor:
    """
    Extracts ML features from sequencing read files.
    
    Supports FASTQ and FASTA formats (gzipped or uncompressed).
    Can subsample large files for faster feature extraction.
    
    Example:
        extractor = FeatureExtractor(subsample=10000)
        features = extractor.extract_from_file('reads.fastq.gz')
        print(f"Detected {features.read_type} reads")
        print(f"Estimated coverage: {features.estimated_coverage:.1f}x")
    """
    
    def __init__(self, 
                 subsample: Optional[int] = None,
                 kmer_size: int = 21,
                 estimate_genome_size: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            subsample: If set, only analyze this many reads (for speed)
            kmer_size: K-mer size for genome size estimation
            estimate_genome_size: Whether to estimate genome size (slower)
        """
        self.subsample = subsample
        self.kmer_size = kmer_size
        self.estimate_genome_size = estimate_genome_size
    
    def extract_from_file(self, reads_file: Path) -> ReadFeatures:
        """
        Extract features from a FASTQ/FASTA file.
        
        Args:
            reads_file: Path to reads file (FASTQ/FASTA, optionally gzipped)
        
        Returns:
            ReadFeatures object with comprehensive read statistics
        """
        reads_file = Path(reads_file)
        if not reads_file.exists():
            raise FileNotFoundError(f"Reads file not found: {reads_file}")
        
        logger.info(f"Extracting features from {reads_file.name}")
        
        # Parse reads
        sequences, qualities = self._parse_reads(reads_file)
        
        if self.subsample and len(sequences) > self.subsample:
            logger.info(f"Subsampling {self.subsample} reads from {len(sequences)}")
            indices = np.random.choice(len(sequences), self.subsample, replace=False)
            sequences = [sequences[i] for i in indices]
            if qualities:
                qualities = [qualities[i] for i in indices]
        
        # Extract features
        length_features = self._extract_length_features(sequences)
        quality_features = self._extract_quality_features(qualities)
        composition_features = self._extract_composition_features(sequences)
        coverage_features = self._extract_coverage_features(sequences)
        read_type = self._detect_read_type(
            length_features, 
            quality_features, 
            len(sequences)
        )
        
        return ReadFeatures(
            # Length stats
            mean_read_length=length_features['mean'],
            median_read_length=length_features['median'],
            read_length_n50=length_features['n50'],
            min_read_length=length_features['min'],
            max_read_length=length_features['max'],
            read_length_std=length_features['std'],
            
            # Quality stats
            mean_base_quality=quality_features['mean_q'],
            median_base_quality=quality_features['median_q'],
            estimated_error_rate=quality_features['error_rate'],
            
            # Coverage stats
            total_bases=coverage_features['total_bases'],
            num_reads=len(sequences),
            estimated_genome_size=coverage_features.get('genome_size'),
            estimated_coverage=coverage_features.get('coverage'),
            
            # Composition stats
            gc_content=composition_features['gc_mean'],
            gc_std=composition_features['gc_std'],
            
            # Read type
            read_type=read_type,
            is_paired_end=self._detect_paired_end(reads_file),
        )
    
    def _parse_reads(self, 
                     reads_file: Path) -> Tuple[List[str], Optional[List[str]]]:
        """
        Parse sequences and quality strings from FASTQ/FASTA.
        
        Returns:
            (sequences, qualities) - qualities is None for FASTA
        """
        sequences = []
        qualities = [] if reads_file.suffix.lower() in ['.fastq', '.fq'] or \
                         reads_file.suffixes[-2:] == ['.fastq', '.gz'] or \
                         reads_file.suffixes[-2:] == ['.fq', '.gz'] else None
        
        open_func = gzip.open if reads_file.suffix == '.gz' else open
        mode = 'rt' if reads_file.suffix == '.gz' else 'r'
        
        with open_func(reads_file, mode) as f:
            if qualities is not None:
                # FASTQ format
                while True:
                    header = f.readline()
                    if not header:
                        break
                    seq = f.readline().strip()
                    f.readline()  # + line
                    qual = f.readline().strip()
                    
                    if seq:
                        sequences.append(seq)
                        qualities.append(qual)
            else:
                # FASTA format
                current_seq = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(''.join(current_seq))
                            current_seq = []
                    else:
                        current_seq.append(line)
                if current_seq:
                    sequences.append(''.join(current_seq))
        
        logger.info(f"Parsed {len(sequences)} reads")
        return sequences, qualities
    
    def _extract_length_features(self, sequences: List[str]) -> Dict:
        """Extract read length statistics."""
        if len(sequences) == 0:
            raise ValueError("No sequences found in input file")
        
        lengths = np.array([len(seq) for seq in sequences])
        
        # Calculate N50
        sorted_lengths = np.sort(lengths)[::-1]
        cumsum = np.cumsum(sorted_lengths)
        n50_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        n50 = int(sorted_lengths[n50_idx])
        
        return {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'n50': n50,
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'std': float(np.std(lengths)),
        }
    
    def _extract_quality_features(self, 
                                  qualities: Optional[List[str]]) -> Dict:
        """Extract base quality statistics from FASTQ quality strings."""
        if qualities is None or len(qualities) == 0:
            # FASTA file - assume high quality
            return {
                'mean_q': 40.0,
                'median_q': 40.0,
                'error_rate': 0.0001,
            }
        
        # Convert Phred+33 to quality scores
        all_quals = []
        for qual_str in qualities:
            quals = [ord(c) - 33 for c in qual_str]
            all_quals.extend(quals)
        
        all_quals = np.array(all_quals)
        mean_q = float(np.mean(all_quals))
        median_q = float(np.median(all_quals))
        
        # Estimate error rate from quality scores
        # Q = -10 * log10(P_error)  =>  P_error = 10^(-Q/10)
        error_rate = float(np.mean(10 ** (-all_quals / 10)))
        
        return {
            'mean_q': mean_q,
            'median_q': median_q,
            'error_rate': error_rate,
        }
    
    def _extract_composition_features(self, sequences: List[str]) -> Dict:
        """Extract GC content and sequence composition features."""
        gc_contents = []
        
        for seq in sequences:
            seq_upper = seq.upper()
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            total = len(seq)
            gc_contents.append(gc_count / total if total > 0 else 0.0)
        
        gc_contents = np.array(gc_contents)
        
        return {
            'gc_mean': float(np.mean(gc_contents)),
            'gc_std': float(np.std(gc_contents)),
        }
    
    def _extract_coverage_features(self, sequences: List[str]) -> Dict:
        """Estimate genome size and coverage depth."""
        total_bases = sum(len(seq) for seq in sequences)
        
        result = {'total_bases': total_bases}
        
        if self.estimate_genome_size and len(sequences) > 0:
            # Estimate genome size using k-mer counting (basic method)
            # More sophisticated methods (GenomeScope) can be added later
            genome_size = self._estimate_genome_size_simple(sequences)
            result['genome_size'] = genome_size
            result['coverage'] = total_bases / genome_size if genome_size > 0 else None
        
        return result
    
    def _estimate_genome_size_simple(self, sequences: List[str]) -> int:
        """
        Simple genome size estimation using unique k-mer counting.
        
        This is a basic estimator. For production, consider:
        - GenomeScope-style estimation from k-mer histograms
        - Correction for sequencing errors
        - Handling of heterozygosity
        """
        # Count unique k-mers
        kmers = set()
        for seq in sequences[:min(len(sequences), 10000)]:  # Sample for speed
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = seq[i:i+self.kmer_size]
                if 'N' not in kmer:
                    kmers.add(kmer)
        
        # Rough estimate: genome size ≈ unique k-mers
        # This underestimates for repetitive genomes but works for rough estimates
        estimated_size = len(kmers)
        
        # Scale up if we subsampled
        if len(sequences) > 10000:
            scale_factor = len(sequences) / 10000
            estimated_size = int(estimated_size * scale_factor)
        
        logger.info(f"Estimated genome size: {estimated_size:,} bp")
        return estimated_size
    
    def _detect_read_type(self, 
                         length_features: Dict,
                         quality_features: Dict,
                         num_reads: int) -> str:
        """
        Detect read technology (HiFi, ONT, Illumina) from features.
        
        Rules:
        - HiFi: 10-25kb, Q20+ (error rate < 0.01)
        - ONT: >20kb, Q12+ (error rate ~0.05-0.15)
        - Illumina: <500bp, Q30+ (error rate < 0.001)
        """
        mean_len = length_features['mean']
        error_rate = quality_features['error_rate']
        
        if mean_len > 20000:
            # Long reads
            if error_rate < 0.01:
                return 'hifi'
            else:
                return 'ont'
        elif mean_len < 500:
            # Short reads
            return 'illumina'
        elif 5000 <= mean_len <= 25000 and error_rate < 0.01:
            # Medium-length high-accuracy
            return 'hifi'
        else:
            return 'unknown'
    
    def _detect_paired_end(self, reads_file: Path) -> bool:
        """
        Detect if reads are paired-end from filename conventions.
        
        Looks for: _R1/_R2, _1/_2, .1/.2 suffixes
        """
        name = reads_file.stem
        if reads_file.suffix == '.gz':
            name = Path(name).stem  # Remove .fastq/.fq too
        
        paired_patterns = ['_R1', '_R2', '_1', '_2', '.1', '.2']
        return any(pattern in name for pattern in paired_patterns)


def extract_features_from_file(reads_file: Path, 
                               subsample: Optional[int] = None) -> ReadFeatures:
    """
    Convenience function to extract features from a reads file.
    
    Args:
        reads_file: Path to FASTQ/FASTA file
        subsample: Optional number of reads to subsample
    
    Returns:
        ReadFeatures object
    
    Example:
        features = extract_features_from_file('reads.fastq.gz', subsample=10000)
        print(f"Coverage: {features.estimated_coverage:.1f}x")
    """
    extractor = FeatureExtractor(subsample=subsample)
    return extractor.extract_from_file(reads_file)
