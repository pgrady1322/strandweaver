#!/usr/bin/env python3
"""
StrandWeaver Assembly Utilities Module.

This module contains utility classes and functions used across the assembly pipeline:

1. PreprocessingCoordinator: Integration of K-Weaver and ErrorSmith
   - K-Weaver: Selects optimal k-mer size for error correction and DBG
   - ErrorSmith: Corrects reads using technology-specific error patterns
   - Output: Corrected reads with k-mer parameters for downstream assembly

2. Additional utilities (to be added as needed):
   - Common data structures
   - Shared helper functions
   - Pipeline coordination utilities
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class KmerPrediction:
    """Predicted k-mer sizes for different assembly stages."""
    dbg_k: int                          # De Bruijn graph k-mer
    ul_overlap_k: int                   # UL read overlap alignment k-mer
    extension_k: int                    # Path extension k-mer
    polish_k: int                       # Final polishing k-mer
    dbg_confidence: float               # Confidence in DBG k choice (0-1)
    ul_overlap_confidence: float        # UL mapping confidence
    extension_confidence: float         # Extension k confidence
    polish_confidence: float            # Polish k confidence
    reasoning: Optional[str] = None     # Explanation of k choice
    
    def get_primary_k(self) -> int:
        """Get primary k-mer (for DBG) from prediction."""
        return self.dbg_k
    
    def get_all_ks(self) -> Dict[str, int]:
        """Get all k-mer sizes as dict."""
        return {
            'dbg': self.dbg_k,
            'ul_overlap': self.ul_overlap_k,
            'extension': self.extension_k,
            'polish': self.polish_k,
        }


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing run."""
    input_reads: int
    corrected_reads: int
    correction_rate: float               # Fraction of reads modified
    bases_corrected: int
    bases_removed: int
    bases_inserted: int
    reads_discarded: int                 # Low-quality reads removed
    
    dbg_k_selected: int
    ul_k_selected: int
    extension_k_selected: int
    polish_k_selected: int
    
    mean_quality_before: float
    mean_quality_after: float
    
    # Timing
    profiling_time_sec: float
    correction_time_sec: float
    k_selection_time_sec: float
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Preprocessing Summary:\n"
            f"  Input: {self.input_reads:,} reads\n"
            f"  Corrected: {self.corrected_reads:,} reads ({100*self.correction_rate:.1f}%)\n"
            f"  Bases corrected: {self.bases_corrected:,} (+{self.bases_inserted:,} ins, -{self.bases_removed:,} del)\n"
            f"  K-mers selected: DBG={self.dbg_k_selected}, UL={self.ul_k_selected}, "
            f"extend={self.extension_k_selected}, polish={self.polish_k_selected}\n"
            f"  Quality: {self.mean_quality_before:.1f} → {self.mean_quality_after:.1f}\n"
            f"  Time: {self.profiling_time_sec:.1f}s profile, {self.correction_time_sec:.1f}s correct, "
            f"{self.k_selection_time_sec:.1f}s select"
        )


@dataclass
class PreprocessingResult:
    """Complete preprocessing pipeline result."""
    corrected_reads_path: Path          # Path to corrected reads file
    kmer_prediction: KmerPrediction     # Predicted k-mers
    stats: PreprocessingStats
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'corrected_reads_path': str(self.corrected_reads_path),
            'kmer_prediction': {
                'dbg_k': self.kmer_prediction.dbg_k,
                'ul_overlap_k': self.kmer_prediction.ul_overlap_k,
                'extension_k': self.kmer_prediction.extension_k,
                'polish_k': self.kmer_prediction.polish_k,
                'confidences': {
                    'dbg': self.kmer_prediction.dbg_confidence,
                    'ul_overlap': self.kmer_prediction.ul_overlap_confidence,
                    'extension': self.kmer_prediction.extension_confidence,
                    'polish': self.kmer_prediction.polish_confidence,
                },
                'reasoning': self.kmer_prediction.reasoning,
            },
            'stats': {
                'input_reads': self.stats.input_reads,
                'corrected_reads': self.stats.corrected_reads,
                'correction_rate': self.stats.correction_rate,
                'bases_corrected': self.stats.bases_corrected,
                'bases_removed': self.stats.bases_removed,
                'bases_inserted': self.stats.bases_inserted,
                'reads_discarded': self.stats.reads_discarded,
                'k_selections': {
                    'dbg': self.stats.dbg_k_selected,
                    'ul': self.stats.ul_k_selected,
                    'extension': self.stats.extension_k_selected,
                    'polish': self.stats.polish_k_selected,
                },
                'quality': {
                    'before': self.stats.mean_quality_before,
                    'after': self.stats.mean_quality_after,
                },
                'timing': {
                    'profiling_sec': self.stats.profiling_time_sec,
                    'correction_sec': self.stats.correction_time_sec,
                    'k_selection_sec': self.stats.k_selection_time_sec,
                }
            },
            'metadata': self.metadata,
        }


class PreprocessingCoordinator:
    """
    Orchestrates preprocessing: K-Weaver + ErrorSmith + output.
    
    Flow:
    1. Profile reads (error patterns, coverage, quality)
    2. Predict optimal k-mers (K-Weaver)
    3. Correct reads (ErrorSmith)
    4. Output corrected reads + k-mer predictions
    """
    
    def __init__(
        self,
        technology: str = "ont_r10",
        use_ai_k_selection: bool = True,
        use_ai_error_correction: bool = True,
        enable_gpu: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            technology: Sequencing technology (ont_r10, pacbio_hifi, illumina, etc)
            use_ai_k_selection: Use K-Weaver AI for k-mer prediction
            use_ai_error_correction: Use ErrorSmith AI for error correction
            enable_gpu: Enable GPU acceleration if available
        """
        self.technology = technology
        self.use_ai_k_selection = use_ai_k_selection
        self.use_ai_error_correction = use_ai_error_correction
        self.enable_gpu = enable_gpu
        self.logger = logging.getLogger(f"{__name__}.PreprocessingCoordinator")
    
    def run_preprocessing(
        self,
        input_reads_path: Path,
        output_dir: Path,
        min_read_length: int = 1000,
        min_quality_score: float = 5.0
    ) -> PreprocessingResult:
        """
        Run complete preprocessing pipeline.
        
        Args:
            input_reads_path: Path to input reads (FASTQ)
            output_dir: Output directory for corrected reads
            min_read_length: Minimum read length to keep
            min_quality_score: Minimum mean quality score
            
        Returns:
            PreprocessingResult with corrected reads and k-mer predictions
        """
        import time
        from strandweaver.errors import ErrorProfiler
        from strandweaver.read_correction.adaptive_kmer import AdaptiveKmerPredictor
        
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Preprocessing Pipeline")
        self.logger.info(f"  Input: {input_reads_path}")
        self.logger.info(f"  Technology: {self.technology}")
        self.logger.info(f"  AI K-selection: {self.use_ai_k_selection}")
        self.logger.info(f"  AI Error correction: {self.use_ai_error_correction}")
        self.logger.info("=" * 80)
        
        # Step 1: Profile errors
        self.logger.info("\n[1/3] Profiling errors...")
        profile_start = time.time()
        
        error_profiler = ErrorProfiler(k=21)
        error_profile = error_profiler.profile(
            reads_file=input_reads_path,
            technology=self.technology.upper() if hasattr(self.technology, 'upper') else self.technology
        )
        
        profile_time = time.time() - profile_start
        self.logger.info(f"  ✓ Profiling complete ({profile_time:.1f}s)")
        self.logger.info(f"    Error rate: {error_profile.error_rate:.2%}")
        self.logger.info(f"    Estimated coverage: {error_profile.estimated_coverage:.1f}×")
        
        # Step 2: Predict optimal k-mers
        self.logger.info("\n[2/3] Predicting optimal k-mers (K-Weaver)...")
        k_select_start = time.time()
        
        kmer_predictor = AdaptiveKmerPredictor(technology=self.technology, use_ai=self.use_ai_k_selection)
        
        # Create ReadContext from profile
        from strandweaver.training.ml_interfaces import ReadContext
        read_context = ReadContext(
            read_id="profile",
            technology=self.technology,
            read_length=int(error_profile.mean_read_length),
            gc_content=error_profile.gc_content,
            mean_quality=error_profile.mean_quality_score,
            error_rate=error_profile.error_rate,
            homopolymer_length=int(error_profile.max_homopolymer_length),
            coverage=error_profile.estimated_coverage
        )
        
        kmer_prediction = kmer_predictor.predict(read_context)
        k_select_time = time.time() - k_select_start
        
        self.logger.info(f"  ✓ K-mer prediction complete ({k_select_time:.1f}s)")
        self.logger.info(f"    DBG k: {kmer_prediction.dbg_k} (confidence: {kmer_prediction.dbg_confidence:.2f})")
        self.logger.info(f"    UL overlap k: {kmer_prediction.ul_overlap_k}")
        self.logger.info(f"    Extension k: {kmer_prediction.extension_k}")
        self.logger.info(f"    Polish k: {kmer_prediction.polish_k}")
        
        # Step 3: Correct reads
        self.logger.info("\n[3/3] Correcting reads (ErrorSmith)...")
        correct_start = time.time()
        
        from strandweaver.correction import get_corrector
        corrector = get_corrector(
            technology=self.technology,
            error_profile=error_profile,
            k_size=kmer_prediction.dbg_k,  # Use predicted k for correction
            use_ai=self.use_ai_error_correction,
            enable_gpu=self.enable_gpu
        )
        
        corrected_reads_path = output_dir / "corrected_reads.fastq"
        correction_stats = corrector.correct_file(
            input_reads_path,
            corrected_reads_path,
            min_read_length=min_read_length,
            min_quality_score=min_quality_score
        )
        
        correct_time = time.time() - correct_start
        self.logger.info(f"  ✓ Error correction complete ({correct_time:.1f}s)")
        self.logger.info(f"    Corrected: {correction_stats['corrected_reads']:,} reads")
        self.logger.info(f"    Bases modified: {correction_stats['bases_corrected']:,}")
        self.logger.info(f"    Discarded: {correction_stats['reads_discarded']:,} low-quality")
        
        # Step 4: Compile statistics
        total_time = time.time() - start_time
        
        stats = PreprocessingStats(
            input_reads=correction_stats.get('input_reads', 0),
            corrected_reads=correction_stats.get('corrected_reads', 0),
            correction_rate=correction_stats.get('correction_rate', 0.0),
            bases_corrected=correction_stats.get('bases_corrected', 0),
            bases_removed=correction_stats.get('bases_removed', 0),
            bases_inserted=correction_stats.get('bases_inserted', 0),
            reads_discarded=correction_stats.get('reads_discarded', 0),
            dbg_k_selected=kmer_prediction.dbg_k,
            ul_k_selected=kmer_prediction.ul_overlap_k,
            extension_k_selected=kmer_prediction.extension_k,
            polish_k_selected=kmer_prediction.polish_k,
            mean_quality_before=error_profile.mean_quality_score,
            mean_quality_after=correction_stats.get('mean_quality_after', error_profile.mean_quality_score),
            profiling_time_sec=profile_time,
            correction_time_sec=correct_time,
            k_selection_time_sec=k_select_time,
        )
        
        # Create result object
        result = PreprocessingResult(
            corrected_reads_path=corrected_reads_path,
            kmer_prediction=kmer_prediction,
            stats=stats,
            metadata={
                'error_profile': error_profile.to_dict() if hasattr(error_profile, 'to_dict') else {},
                'total_time_sec': total_time,
                'technology': self.technology,
            }
        )
        
        # Save result metadata
        result_json = output_dir / "preprocessing_result.json"
        with open(result_json, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(stats.summary())
        self.logger.info("=" * 80)
        self.logger.info(f"\nResult saved to: {result_json}")
        
        return result
    
    def get_kmer_for_stage(
        self,
        preprocessing_result: PreprocessingResult,
        stage: str
    ) -> int:
        """
        Get k-mer size for a specific assembly stage.
        
        Args:
            preprocessing_result: Result from run_preprocessing
            stage: Assembly stage ('dbg', 'ul_overlap', 'extension', 'polish')
            
        Returns:
            Recommended k-mer size
        """
        pred = preprocessing_result.kmer_prediction
        
        stage_map = {
            'dbg': pred.dbg_k,
            'ul_overlap': pred.ul_overlap_k,
            'ul': pred.ul_overlap_k,  # Alias
            'extension': pred.extension_k,
            'polish': pred.polish_k,
        }
        
        if stage not in stage_map:
            self.logger.warning(f"Unknown stage '{stage}'; defaulting to DBG k={pred.dbg_k}")
            return pred.dbg_k
        
        return stage_map[stage]
    
    def check_preprocessing_quality(
        self,
        result: PreprocessingResult,
        min_corrected_fraction: float = 0.8,
        min_quality_improvement: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Check if preprocessing quality is acceptable.
        
        Args:
            result: PreprocessingResult
            min_corrected_fraction: Minimum fraction of reads corrected
            min_quality_improvement: Minimum quality score improvement
            
        Returns:
            Tuple of (is_acceptable, explanation_message)
        """
        issues = []
        
        # Check correction rate
        corr_frac = result.stats.corrected_reads / max(result.stats.input_reads, 1)
        if corr_frac < min_corrected_fraction:
            issues.append(f"Low correction rate: {100*corr_frac:.1f}% < {100*min_corrected_fraction:.1f}% threshold")
        
        # Check quality improvement
        quality_gain = result.stats.mean_quality_after - result.stats.mean_quality_before
        if quality_gain < min_quality_improvement:
            issues.append(f"Small quality gain: {quality_gain:.1f} < {min_quality_improvement:.1f} threshold")
        
        # Check k-mer selection confidence
        min_k_conf = min(
            result.kmer_prediction.dbg_confidence,
            result.kmer_prediction.ul_overlap_confidence
        )
        if min_k_conf < 0.6:
            issues.append(f"Low k-mer confidence: {min_k_conf:.2f} < 0.6")
        
        # Determine acceptability
        if not issues:
            return True, "✓ Preprocessing quality acceptable"
        else:
            return False, "⚠ Preprocessing issues:\n  " + "\n  ".join(issues)
