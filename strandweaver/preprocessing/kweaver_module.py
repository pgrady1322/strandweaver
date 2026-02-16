#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

K-Weaver — AI-powered adaptive k-mer selector. Dynamically selects optimal
k-mer sizes for each assembly stage based on read characteristics.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pickle
import gzip

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

# Minimum read-length N50 (bp) for UL overlap k to be reliable.
# Below this threshold, reads are too short to benefit from ultra-long
# overlap detection; using the UL k value is likely to cause problems
# (mis-joins, false overlaps, fragmented string graph).
UL_MIN_N50 = 50_000


@dataclass
class KmerPrediction:
    """Predicted k-mer sizes for different assembly stages."""
    dbg_k: int                          # De Bruijn graph k-mer
    ul_overlap_k: int                   # UL read overlap alignment k-mer (advisory; UL mapper uses anchor_k=15)
    extension_k: int                    # Path extension k-mer
    polish_k: int                       # Final polishing k-mer
    dbg_confidence: float = 1.0         # Confidence in DBG k choice (0-1)
    ul_confidence: float = 1.0          # UL mapping confidence (primary name)
    extension_confidence: float = 1.0   # Extension k confidence
    polish_confidence: float = 1.0      # Polish k confidence
    reasoning: Optional[str] = None     # Explanation of k choice
    ul_applicable: bool = True          # Whether UL k is meaningful for these reads
    read_n50: Optional[float] = None    # Read N50 used for UL applicability check

    @property
    def ul_overlap_confidence(self) -> float:
        """Alias for ul_confidence."""
        return self.ul_confidence

    @property
    def ul_warning(self) -> Optional[str]:
        """Human-readable warning when UL k is not applicable.

        Returns None if UL k is reliable, otherwise a string explaining
        why the predicted UL k should not be trusted.
        """
        if self.ul_applicable:
            return None
        n50_str = f"{self.read_n50:,.0f}" if self.read_n50 else "unknown"
        return (
            f"UL overlap k={self.ul_overlap_k} is advisory only — reads "
            f"have N50={n50_str} bp (need ≥{UL_MIN_N50:,} bp for reliable "
            f"ultra-long overlap detection). Using shorter reads in the UL "
            f"slot will likely cause mis-joins and fragmented scaffolds."
        )

    def get_primary_k(self) -> int:
        """Get primary k-mer (for DBG) from prediction."""
        return self.dbg_k

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dbg_k': self.dbg_k,
            'ul_overlap_k': self.ul_overlap_k,
            'extension_k': self.extension_k,
            'polish_k': self.polish_k,
            'dbg_confidence': self.dbg_confidence,
            'ul_confidence': self.ul_confidence,
            'extension_confidence': self.extension_confidence,
            'polish_confidence': self.polish_confidence,
            'reasoning': self.reasoning,
            'ul_applicable': self.ul_applicable,
            'ul_warning': self.ul_warning,
        }
    
    def get_all_ks(self) -> Dict[str, int]:
        """Get all k-mer sizes as dict."""
        return {
            'dbg': self.dbg_k,
            'ul_overlap': self.ul_overlap_k,
            'extension': self.extension_k,
            'polish': self.polish_k,
        }


# ============================================================================
# Feature Extraction Components
# ============================================================================

@dataclass
class ReadFeatures:
    """
    Comprehensive features extracted from sequencing reads.
    
    These features are used as input to K-Weaver's ML models for predicting
    optimal k-mer sizes. All features are normalized and scaled appropriately.
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
        Genome size estimation using k-mer histogram peak method.
        
        Uses the modal k-mer frequency (excluding singletons) to estimate
        average coverage depth, then divides total k-mer mass by coverage.
        This handles repetitive genomes much better than unique k-mer counting
        (G19 fix).
        
        Algorithm:
          1. Count all k-mers across sampled reads
          2. Build frequency histogram (how many k-mers appear 1×, 2×, etc.)
          3. Find peak frequency (mode) excluding singletons (freq=1, likely errors)
          4. Genome size ≈ sum of all k-mer counts / peak_frequency
          
        Falls back to unique k-mer counting if histogram is too sparse.
        """
        # Count k-mers
        kmer_counts: Dict[str, int] = defaultdict(int)
        sample_limit = min(len(sequences), 10000)
        for seq in sequences[:sample_limit]:
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = seq[i:i+self.kmer_size]
                if 'N' not in kmer:
                    kmer_counts[kmer] += 1
        
        if len(kmer_counts) == 0:
            logger.warning("No k-mers found for genome size estimation")
            return 1
        
        # Build frequency histogram (exclude singletons as likely errors)
        freq_histogram: Dict[int, int] = defaultdict(int)
        for count in kmer_counts.values():
            freq_histogram[count] += 1
        
        # Find modal frequency (peak of histogram, excluding freq=1)
        non_singleton_freqs = {f: n for f, n in freq_histogram.items() if f >= 2}
        
        if non_singleton_freqs:
            # Peak = frequency with the most k-mers (approximates coverage depth)
            peak_freq = max(non_singleton_freqs, key=non_singleton_freqs.get)
            
            # Total k-mer mass (sum of all counts)
            total_kmer_mass = sum(kmer_counts.values())
            
            # Genome size ≈ total mass / coverage depth
            estimated_size = int(total_kmer_mass / peak_freq)
        else:
            # All k-mers are singletons — fall back to unique count
            estimated_size = len(kmer_counts)
            logger.warning("All k-mers are singletons; genome size estimate unreliable")
        
        # Scale up if we subsampled reads
        if len(sequences) > sample_limit:
            scale_factor = len(sequences) / sample_limit
            estimated_size = int(estimated_size * scale_factor)
        
        logger.info(f"Estimated genome size: {estimated_size:,} bp (k-mer histogram peak method)")
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


# ============================================================================
# K-Weaver Prediction Components
# ============================================================================
# Note: KmerPrediction is imported from utils.pipeline at the top of the file


class KWeaverPredictor:
    """
    K-Weaver: AI-powered adaptive k-mer size predictor.
    
    Predicts optimal k-mer sizes for each assembly stage using ML models trained on:
    - DBG k: Optimized for graph connectivity and bubble frequency
    - UL overlap k: Optimized for spanning accuracy and specificity
    - Extension k: Optimized for gap closure and mis-join avoidance
    - Polish k: Optimized for base quality improvement
    
    Falls back to technology-aware rules when ML models are unavailable.
    
    Example:
        # Quick prediction from file
        predictor = KWeaverPredictor()
        prediction = predictor.predict_from_file('reads.fastq')
        
        # Extract features first (more control)
        extractor = FeatureExtractor()
        features = extractor.extract_from_file('reads.fastq')
        prediction = predictor.predict(features)
        
        # Use predictions
        dbg_k = prediction.dbg_k
        ul_k = prediction.ul_overlap_k
    """
    
    def __init__(self, model_dir: Optional[Path] = None, use_ml: bool = True):
        """
        Initialize K-Weaver predictor with trained models.
        
        Args:
            model_dir: Directory containing trained XGBoost models.
                      If None, uses default models from package data.
            use_ml: If True, attempt to load ML models. If False or loading fails,
                   falls back to rule-based predictions.
        """
        self.model_dir = model_dir
        self.use_ml = use_ml
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """
        Load trained XGBoost models for each stage.
        
        Looks for models in:
        1. Specified model_dir
        2. Package default location (strandweaver/ai/training/trained_models/)
        3. Falls back to rule-based if not found
        """
        if not self.use_ml:
            logger.info("ML models disabled - using rule-based predictions")
            self.models = None
            return
        
        logger.info("Loading K-Weaver ML models...")
        
        # Determine model directory
        if self.model_dir:
            model_path = Path(self.model_dir)
        else:
            # Default: look in package data
            model_path = Path(__file__).parent / 'training' / 'trained_models'
        
        # Check if models exist
        model_files = {
            'dbg': model_path / 'dbg_model.pkl',
            'ul_overlap': model_path / 'ul_overlap_model.pkl',
            'extension': model_path / 'extension_model.pkl',
            'polish': model_path / 'polish_model.pkl',
        }
        
        # Try to load each model
        loaded_count = 0
        for stage, model_file in model_files.items():
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[stage] = pickle.load(f)
                    logger.info(f"  ✓ Loaded {stage} model from {model_file}")
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load {stage} model: {e}")
            else:
                logger.debug(f"  Model not found: {model_file}")
        
        if loaded_count == 0:
            logger.warning("No trained models found - using rule-based defaults")
            logger.info(f"  Looked in: {model_path}")
            logger.info("  To train models, run: python workflow_train_models.py")
            self.models = None
        elif loaded_count < 4:
            logger.warning(f"Only {loaded_count}/4 models loaded - some predictions will use rules")
        else:
            logger.info(f"✓ Successfully loaded all 4 K-Weaver models from {model_path}")
    
    def predict(self, features: ReadFeatures) -> KmerPrediction:
        """
        Predict optimal k-mer sizes for all assembly stages.
        
        Args:
            features: ReadFeatures object from FeatureExtractor
        
        Returns:
            KmerPrediction with k values for each stage
        """
        if self.models is None or len(self.models) == 0:
            # Use rule-based defaults if no models loaded
            logger.debug("Using rule-based k-mer prediction (no ML models)")
            return self._predict_rule_based(features)
        
        # Use trained ML models
        logger.debug("Using ML-based k-mer prediction")
        return self._predict_ml_based(features)
    
    def _predict_rule_based(self, features: ReadFeatures) -> KmerPrediction:
        """
        Rule-based k-mer selection (fallback until ML models trained).

        Rules based on industry best practices:
        - HiFi reads: DBG=31, UL_overlap=501, ext=55, polish=77
        - ONT reads: DBG=21, UL_overlap=501, ext=41, polish=55
        - Illumina: DBG=31, UL_overlap=N/A, ext=55, polish=77

        UL overlap k is only marked as *applicable* when the read N50
        exceeds ``UL_MIN_N50`` (50 Kb).  Shorter reads will still get
        a predicted UL k (for the rare case where a user intentionally
        feeds non-UL reads into the UL slot), but ``ul_applicable``
        will be False and ``ul_confidence`` will be reduced.
        """
        read_type = features.read_type
        mean_len = features.mean_read_length
        error_rate = features.estimated_error_rate
        read_n50 = features.read_length_n50

        # UL applicability: need N50 ≥ 50 Kb for reliable ultra-long overlap
        ul_ok = read_n50 >= UL_MIN_N50

        logger.info(f"Using rule-based k-mer selection for {read_type} reads")
        if not ul_ok:
            logger.info(
                f"  UL overlap k is advisory only (read N50={read_n50:,.0f} bp "
                f"< {UL_MIN_N50:,} bp threshold)"
            )

        if read_type == 'hifi':
            # High-quality long reads - can use larger k
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=501,
                extension_k=55,
                polish_k=77,
                dbg_confidence=0.8,
                ul_confidence=0.8 if ul_ok else 0.3,
                extension_confidence=0.8,
                polish_confidence=0.8,
                ul_applicable=ul_ok,
                read_n50=read_n50,
            )
        elif read_type == 'ont':
            # Higher error rate - need smaller k for DBG
            return KmerPrediction(
                dbg_k=21 if error_rate > 0.05 else 31,
                ul_overlap_k=501,
                extension_k=41,
                polish_k=55,
                dbg_confidence=0.7,
                ul_confidence=0.8 if ul_ok else 0.3,
                extension_confidence=0.7,
                polish_confidence=0.7,
                ul_applicable=ul_ok,
                read_n50=read_n50,
            )
        elif read_type == 'illumina':
            # Short reads - moderate k
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=31,  # No ultralong reads
                extension_k=55,
                polish_k=77,
                dbg_confidence=0.8,
                ul_confidence=0.2,  # Very low — Illumina can never be UL
                extension_confidence=0.8,
                polish_confidence=0.8,
                ul_applicable=False,
                read_n50=read_n50,
            )
        else:
            # Unknown - conservative defaults
            logger.warning(f"Unknown read type, using conservative k values")
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=501,
                extension_k=41,
                polish_k=55,
                dbg_confidence=0.5,
                ul_confidence=0.3,
                extension_confidence=0.5,
                polish_confidence=0.5,
                ul_applicable=False,
                read_n50=read_n50,
            )
    
    def _predict_ml_based(self, features: ReadFeatures) -> KmerPrediction:
        """
        ML-based k-mer prediction using trained XGBoost models.
        
        Uses separate trained models for each assembly stage.
        Falls back to rule-based for any missing models.
        """
        # Convert features to numpy array
        X = features.to_feature_vector().reshape(1, -1)
        
        # Get rule-based prediction as fallback
        rule_based = self._predict_rule_based(features)
        
        # Predict k for each stage (use rule-based if model missing)
        k_dbg = int(self.models['dbg'].predict(X)[0]) if 'dbg' in self.models else rule_based.dbg_k
        k_ul = int(self.models['ul_overlap'].predict(X)[0]) if 'ul_overlap' in self.models else rule_based.ul_overlap_k
        k_ext = int(self.models['extension'].predict(X)[0]) if 'extension' in self.models else rule_based.extension_k
        k_pol = int(self.models['polish'].predict(X)[0]) if 'polish' in self.models else rule_based.polish_k
        
        # Confidence scores
        # For now, use high confidence for ML predictions, lower for fallbacks
        dbg_conf = 0.95 if 'dbg' in self.models else rule_based.dbg_confidence
        ul_conf = 0.95 if 'ul_overlap' in self.models else rule_based.ul_confidence
        ext_conf = 0.95 if 'extension' in self.models else rule_based.extension_confidence
        pol_conf = 0.95 if 'polish' in self.models else rule_based.polish_confidence

        # UL applicability — reads need N50 ≥ 50 Kb to benefit from UL overlap
        read_n50 = features.read_length_n50
        ul_ok = read_n50 >= UL_MIN_N50
        if not ul_ok:
            ul_conf = min(ul_conf, 0.3)
            logger.info(
                f"  UL overlap k={k_ul} is advisory only (read N50="
                f"{read_n50:,.0f} bp < {UL_MIN_N50:,} bp threshold)"
            )

        logger.info(f"ML predictions: DBG={k_dbg}, UL={k_ul}, ext={k_ext}, pol={k_pol}")

        return KmerPrediction(
            dbg_k=k_dbg,
            ul_overlap_k=k_ul,
            extension_k=k_ext,
            polish_k=k_pol,
            dbg_confidence=dbg_conf,
            ul_confidence=ul_conf,
            extension_confidence=ext_conf,
            polish_confidence=pol_conf,
            ul_applicable=ul_ok,
            read_n50=read_n50,
        )
    
    def predict_single_k(self, features: ReadFeatures, stage: str) -> int:
        """
        Predict k for a single assembly stage.
        
        Args:
            features: ReadFeatures object
            stage: One of 'dbg', 'ul_overlap', 'extension', 'polish'
        
        Returns:
            Optimal k-mer size for that stage
        """
        prediction = self.predict(features)
        k_dict = prediction.to_dict()
        
        if stage not in k_dict:
            raise ValueError(f"Invalid stage: {stage}. "
                           f"Must be one of {list(k_dict.keys())}")
        
        return k_dict[stage]
    
    def get_default_prediction(self, technology: str) -> KmerPrediction:
        """
        Get default k-mer prediction for a specific technology.
        
        Args:
            technology: Technology name ('hifi', 'ont', 'illumina', etc.)
            
        Returns:
            KmerPrediction with default values for that technology
        """
        from ..preprocessing.read_classification_utility import ReadTechnology
        
        # Map string to ReadTechnology if needed
        tech_map = {
            'hifi': 'hifi',
            'pacbio': 'hifi',
            'ont': 'ont',
            'nanopore': 'ont',
            'illumina': 'illumina',
        }
        
        normalized_tech = tech_map.get(technology.lower(), 'unknown')
        
        # Create minimal features with just the technology
        class MinimalFeatures:
            def __init__(self, tech):
                self.read_type = tech
                self.mean_read_length = 15000 if tech in ['hifi', 'ont'] else 150
                self.estimated_error_rate = 0.01 if tech == 'hifi' else 0.05 if tech == 'ont' else 0.001
                # Conservative N50 defaults — callers with real reads should
                # use predict() / predict_from_file() which compute real N50
                self.read_length_n50 = 15000 if tech == 'hifi' else 20000 if tech == 'ont' else 150
        
        features = MinimalFeatures(normalized_tech)
        return self._predict_rule_based(features)
    
    def predict_from_file(self, reads_file: Path) -> KmerPrediction:
        """
        Convenience method: extract features and predict k values from reads file.
        
        Args:
            reads_file: Path to FASTQ or FASTA file
        
        Returns:
            KmerPrediction with k values for each stage
        """
        logger.info(f"Extracting features from {reads_file}")
        extractor = FeatureExtractor()
        features = extractor.extract_from_file(reads_file)
        
        logger.info(f"Predicting k-mer sizes for {features.read_type} reads")
        return self.predict(features)


# ============================================================================
# Convenience Functions
# ============================================================================

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


def extract_read_features(reads_file: Path, subsample: Optional[int] = None) -> ReadFeatures:
    """
    Extract features from a read file. Convenience wrapper for extract_features_from_file.
    
    Args:
        reads_file: Path to FASTQ/FASTA file
        subsample: Number of reads to sample (None = all reads)
        
    Returns:
        ReadFeatures object with comprehensive read statistics
    """
    return extract_features_from_file(reads_file, subsample=subsample)


# ============================================================================
# Batch Processing Functions (Nextflow Integration)
# ============================================================================

def extract_kmers_batch(
    reads_file: str,
    k: int,
    output_table: str,
    threads: int = 1
) -> None:
    """
    Extract k-mers from read batch for huge genome mode.
    
    This function extracts k-mers from a batch of reads and serializes
    them to a binary format for later aggregation. Used by Nextflow for
    parallel k-mer extraction in --huge mode.
    
    Args:
        reads_file: Input FASTQ batch
        k: K-mer size
        output_table: Output k-mer table (binary format)
        threads: Number of threads to use
    """
    from collections import Counter
    import pickle
    from ..io_utils import read_fastq
    
    logger.info(f"Extracting {k}-mers from batch: {Path(reads_file).name}")
    
    # Count k-mers in batch
    kmer_counts = Counter()
    read_count = 0
    
    for read in read_fastq(reads_file):
        seq = read.sequence
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                kmer_counts[kmer] += 1
        read_count += 1
    
    # Serialize to binary format
    output_path = Path(output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(dict(kmer_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Extracted {len(kmer_counts)} unique {k}-mers from {read_count} reads")


AdaptiveKmerPredictor = KWeaverPredictor

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
