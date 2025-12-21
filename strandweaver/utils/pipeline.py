"""
StrandWeaver Master Pipeline Orchestrator.

This is the unified coordinator that manages the complete assembly pipeline:
- Preprocessing: Error profiling, k-mer prediction (K-Weaver), read correction (ErrorSmith)
- Assembly: Technology-specific routing (OLC, DBG, String Graph, Hi-C scaffolding)
- Finishing: Polishing, gap filling, final output

Consolidates three former coordinators:
1. PreprocessingCoordinator (from utilities.py) - K-Weaver + ErrorSmith integration
2. AssemblyOrchestrator (from pipeline_orchestrator.py) - Assembly routing by read type
3. PipelineOrchestrator (this file) - End-to-end pipeline with checkpointing

Key features:
- Single entry point for complete pipeline execution
- Checkpoint support for resume capability
- AI model loading and management
- Technology-specific processing (Illumina, HiFi, ONT, Ancient DNA)
- Deterministic assembly flows based on read type
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field
import time
import math

from ..io_utils import SeqRead, read_fastq, write_fastq, read_fasta, write_fasta
from ..preprocessing import ErrorProfiler
from ..preprocessing import (
    ONTCorrector,
    PacBioCorrector,
    IlluminaCorrector,
    AncientDNACorrector,
)
from ..assembly_core.dbg_engine_module import build_dbg_from_long_reads, DBGGraph
from ..assembly_core.string_graph_engine_module import (
    build_string_graph_from_dbg_and_ul,
    StringGraph,
    ULAnchor,
    LongReadOverlay,
)
from ..assembly_core.illumina_olc_contig_module import ContigBuilder
from ..assembly_core.dbg_engine_module import (
    KmerGraph,
    KmerNode,
    KmerEdge,
)
from ..assembly_core.edgewarden_module import EdgeWarden
from ..assembly_core.pathweaver_module import PathWeaver
from ..assembly_core.haplotype_detangler_module import HaplotypeDetangler
from ..assembly_core.threadcompass_module import ThreadCompass
from ..assembly_core.svscribe_module import SVScribe


# ============================================================================
# Data Structures (from PreprocessingCoordinator)
# ============================================================================

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


@dataclass
class AssemblyResult:
    """
    Result of assembly pipeline.
    
    Attributes:
        dbg: De Bruijn graph
        string_graph: String graph with UL overlay
        contigs: Final assembled contigs
        scaffolds: Hi-C scaffolds (if Hi-C data provided)
        stats: Assembly statistics
    """
    dbg: Optional[DBGGraph] = None
    string_graph: Optional[StringGraph] = None
    contigs: List = None
    scaffolds: List = None
    stats: Dict[str, Any] = None


# ============================================================================
# Master Pipeline Orchestrator (Unified Coordinator)
# ============================================================================

class PipelineOrchestrator:
    """
    Master orchestrator for the complete StrandWeaver pipeline.
    
    This unified coordinator consolidates three former coordinators:
    1. PreprocessingCoordinator - K-Weaver + ErrorSmith preprocessing
    2. AssemblyOrchestrator - Technology-specific assembly routing
    3. PipelineOrchestrator - End-to-end pipeline execution
    
    Manages:
    - Preprocessing: Error profiling, k-mer prediction, read correction
    - Assembly: Technology-specific flows (Illumina/HiFi/ONT/Ancient/Mixed)
      * Illumina: OLC → DBG → String Graph (if UL) → Hi-C
      * HiFi/ONT: DBG → String Graph (if UL) → Hi-C
      * Ancient: DBG → String Graph (if UL) → Hi-C
    - Finishing: Polishing, gap filling, final output
    - Pipeline control: Checkpointing, recovery, AI model loading
    
    Key principle: String graph ALWAYS follows DBG when UL reads available.
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[Path] = None):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration dictionary
            checkpoint_dir: Directory for checkpoints
        """
        self.config = config
        self.output_dir = Path(config['runtime']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir or (self.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_level = getattr(logging, config['output']['logging']['level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / config['output']['logging']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Pipeline steps (from config)
        self.steps = config['pipeline']['steps']
        
        # Runtime state
        self.state = {
            'current_step': None,
            'completed_steps': [],
            'read_files': config['runtime']['reads'],  # Store paths, not reads
            'technologies': config['runtime']['technologies'],
            'kmer_prediction': None,  # K-Weaver predictions for pipeline stages
            'corrected_files': {},  # {technology: corrected_file_path}
            'assembly_result': None,
            'final_contigs': None,
        }
        
        # AI models (lazy loaded)
        self._ai_models = {}
    
    def run(self, start_from: Optional[str] = None, resume: bool = False):
        """
        Run the complete pipeline.
        
        Args:
            start_from: Step to start from (None = start from beginning)
            resume: Resume from last checkpoint
        
        Returns:
            Pipeline execution summary
        """
        self.logger.info("="*60)
        self.logger.info("Starting StrandWeaver Pipeline")
        self.logger.info("="*60)
        
        # Determine starting step
        if resume:
            start_step = self._find_last_checkpoint()
            self.logger.info(f"Resuming from checkpoint: {start_step}")
        elif start_from:
            start_step = start_from
            self.logger.info(f"Starting from step: {start_step}")
        else:
            start_step = self.steps[0]
            self.logger.info(f"Starting from beginning: {start_step}")
        
        # Execute steps
        start_index = self.steps.index(start_step)
        for i, step in enumerate(self.steps[start_index:], start=start_index):
            self.state['current_step'] = step
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STEP {i+1}/{len(self.steps)}: {step.upper()}")
            self.logger.info(f"{'='*60}")
            
            try:
                self._execute_step(step)
                self.state['completed_steps'].append(step)
                self._create_checkpoint(step)
                
            except Exception as e:
                self.logger.error(f"Step {step} failed: {e}", exc_info=True)
                raise
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Pipeline Complete!")
        self.logger.info("="*60)
        
        return {
            "status": "success",
            "steps_completed": len(self.state['completed_steps']),
            "output_dir": str(self.output_dir),
        }
    
    def _execute_step(self, step: str):
        """Execute a single pipeline step."""
        if step == 'kweaver':
            self._step_kweaver()
        elif step == 'profile':
            self._step_profile()
        elif step == 'correct':
            self._step_correct()
        elif step == 'assemble':
            self._step_assemble()
        elif step == 'finish':
            self._step_finish()
        else:
            raise ValueError(f"Unknown step: {step}")
    
    def _step_kweaver(self):
        """K-Weaver: Predict optimal k-mer sizes for all pipeline stages."""
        self.logger.info("Running K-Weaver k-mer prediction...")
        
        from ..preprocessing import KWeaverPredictor, FeatureExtractor
        
        # Check if AI models should be used
        use_ai = self.config['ai']['enabled'] and \
                 self.config['ai']['correction'].get('adaptive_kmer', {}).get('enabled', False)
        
        # Initialize K-Weaver
        predictor = KWeaverPredictor(use_ml=use_ai)
        
        # Extract features from first input file (representative sample)
        first_file = Path(self.state['read_files'][0])
        self.logger.info(f"  Extracting features from: {first_file.name}")
        
        extractor = FeatureExtractor()
        features = extractor.extract_from_file(
            first_file,
            sample_size=min(100000, self.config['profiling']['sample_size'])
        )
        
        # Predict optimal k-mers for all stages
        kmer_prediction = predictor.predict(features)
        
        # Log predictions
        self.logger.info("✓ K-mer predictions complete:")
        self.logger.info(f"  DBG construction: k={kmer_prediction.dbg_k} (confidence: {kmer_prediction.dbg_confidence:.2f})")
        self.logger.info(f"  UL overlaps: k={kmer_prediction.ul_overlap_k} (confidence: {kmer_prediction.ul_overlap_confidence:.2f})")
        self.logger.info(f"  Extension: k={kmer_prediction.extension_k} (confidence: {kmer_prediction.extension_confidence:.2f})")
        self.logger.info(f"  Polish: k={kmer_prediction.polish_k} (confidence: {kmer_prediction.polish_confidence:.2f})")
        
        if kmer_prediction.reasoning:
            self.logger.info(f"  Reasoning: {kmer_prediction.reasoning}")
        
        # Save predictions to file
        kmer_path = self.output_dir / "kmer_predictions.json"
        with open(kmer_path, 'w') as f:
            import json
            json.dump(kmer_prediction.to_dict(), f, indent=2)
        
        # Store in state for use throughout pipeline
        self.state['kmer_prediction'] = kmer_prediction
        self.logger.info(f"  Saved to: {kmer_path}")
    
    def _step_profile(self):
        """Error profiling step."""
        self.logger.info("Profiling sequencing errors...")
        
        # Get k-mer prediction from K-Weaver (if available)
        kmer_prediction = self.state.get('kmer_prediction')
        profile_k = kmer_prediction.dbg_k if kmer_prediction else 21  # Default fallback
        
        self.logger.info(f"  Using k={profile_k} for error profiling (from K-Weaver)")
        
        # Sample reads from files (streaming) - don't load all into memory
        sample_size = self.config['profiling']['sample_size']
        sampled_reads = []
        total_reads = 0
        
        for reads_file in self.state['read_files']:
            reads_path = Path(reads_file)
            read_count = 0
            
            # Stream reads from file
            for read in self._read_file_streaming(reads_path):
                if len(sampled_reads) < sample_size:
                    sampled_reads.append(read)
                read_count += 1
                
                # Stop if we have enough samples
                if len(sampled_reads) >= sample_size:
                    break
            
            total_reads += read_count
            self.logger.info(f"  Sampled from {reads_file}: {read_count:,} reads")
            
            if len(sampled_reads) >= sample_size:
                break
        
        self.logger.info(f"Profiling {len(sampled_reads)} reads (sampled from {total_reads:,} total)")
        
        # Run profiler with K-Weaver k-mer size
        profiler = ErrorProfiler(k=profile_k)
        profile = profiler.profile_errors(
            sampled_reads,
            detect_ancient=self.config['profiling']['ancient_dna']['detect_damage']
        )
        
        # Save profile
        profile_path = self.output_dir / "error_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        self.logger.info(f"✓ Error profile saved: {profile_path}")
        self.logger.info(f"  Technology detected: {profile.get('technology', 'unknown')}")
        self.logger.info(f"  Mean quality: {profile.get('mean_quality', 0):.2f}")
        
        # Don't store reads - only metadata
        self.state['error_profile'] = profile
    
    def _step_correct(self):
        """Error correction step."""
        self.logger.info("Correcting sequencing errors...")
        
        # Load AI models if enabled
        use_ai = self.config['ai']['enabled']
        ai_models = self._load_ai_models() if use_ai else None
        
        # Get K-Weaver k-mer predictions
        kmer_prediction = self.state.get('kmer_prediction')
        
        # Process each input file independently by technology
        # This minimizes read combining/separating operations
        technologies = self.state['technologies']
        read_files = self.state['read_files']
        
        for i, (reads_file, tech) in enumerate(zip(read_files, technologies)):
            self.logger.info(f"\n  Processing file {i+1}/{len(read_files)}: {Path(reads_file).name} ({tech})")
            
            # Select technology-specific corrector with K-Weaver k-mer
            if tech == 'illumina':
                # Use polish_k for Illumina correction (precision focus)
                correction_k = kmer_prediction.polish_k if kmer_prediction else self.config['correction']['illumina']['kmer_size']
                corrector = IlluminaCorrector(
                    kmer_size=correction_k,
                    ml_k_model=ai_models.get('adaptive_kmer') if use_ai else None
                )
                self.logger.info(f"    Using k={correction_k} for Illumina correction")
            elif tech == 'ancient':
                corrector = AncientDNACorrector(
                    ml_damage_model=ai_models.get('base_error_classifier') if use_ai else None
                )
            elif tech == 'ont' or tech == 'ont_ultralong':
                # Use extension_k for ONT correction (balance accuracy/length)
                correction_k = kmer_prediction.extension_k if kmer_prediction else 41
                corrector = ONTCorrector(
                    kmer_size=correction_k,
                    ml_error_model=ai_models.get('base_error_classifier') if use_ai else None
                )
                self.logger.info(f"    Using k={correction_k} for ONT correction")
            elif tech == 'pacbio':
                # Use polish_k for PacBio correction (high precision)
                correction_k = kmer_prediction.polish_k if kmer_prediction else 55
                corrector = PacBioCorrector(
                    kmer_size=correction_k,
                    ml_error_model=ai_models.get('base_error_classifier') if use_ai else None
                )
                self.logger.info(f"    Using k={correction_k} for PacBio correction")
            else:
                # Auto-detect or unknown - use generic
                corrector = IlluminaCorrector()
            
            # Correct reads in streaming fashion
            input_path = Path(reads_file)
            output_path = self.output_dir / f"corrected_{tech}_{i}.fastq"
            
            corrected_count = 0
            with open(output_path, 'w') as out_f:
                for read in self._read_file_streaming(input_path):
                    # Correct individual read
                    corrected = corrector.correct_read(read)
                    if corrected:
                        # Write immediately to output file
                        self._write_read_to_file(corrected, out_f)
                        corrected_count += 1
            
            self.logger.info(f"    ✓ Corrected {corrected_count:,} reads → {output_path.name}")
            
            # Store corrected file path by technology
            if tech not in self.state['corrected_files']:
                self.state['corrected_files'][tech] = []
            self.state['corrected_files'][tech].append(str(output_path))
    
    def _step_assemble(self):
        """Assembly step."""
        self.logger.info("Assembling contigs...")
        
        # Load AI models if enabled
        use_ai = self.config['ai']['enabled']
        ai_models = self._load_ai_models() if use_ai else None
        
        # Determine primary technology for assembly strategy
        technologies = self.state['technologies']
        primary_tech = max(set(technologies), key=technologies.count)
        
        self.logger.info(f"  Primary technology: {primary_tech}")
        self.logger.info(f"  AI-powered assembly: {use_ai}")
        
        # Prepare corrected read files by category
        corrected_files = self.state['corrected_files']
        
        # Separate by read type for assembly (without loading into memory)
        long_read_files = []
        ul_read_files = []
        illumina_read_files = []
        
        for tech, files in corrected_files.items():
            if tech == 'ont' or tech == 'pacbio':
                long_read_files.extend(files)
            elif tech == 'ont_ultralong':
                ul_read_files.extend(files)
            elif tech == 'illumina':
                illumina_read_files.extend(files)
        
        # Prepare Hi-C data if enabled
        hic_data = None
        if self.config['scaffolding']['hic']['enabled']:
            hic_data = {
                'r1': self.config['runtime'].get('hic_r1'), 
                'r2': self.config['runtime'].get('hic_r2')
            }
        
        # Get K-Weaver k-mer predictions for assembly
        kmer_prediction = self.state.get('kmer_prediction')
        
        # Run technology-specific assembly pipeline
        # Pass file paths instead of loaded reads
        assembly_result = self._run_assembly_pipeline(
            read_type=primary_tech.lower(),
            illumina_files=illumina_read_files,
            long_read_files=long_read_files,
            ul_read_files=ul_read_files,
            hic_data=hic_data,
            kmer_prediction=kmer_prediction,
            ml_k_model=ai_models.get('adaptive_kmer') if use_ai else None
        )
        
        # Save contigs
        contigs_path = self.output_dir / "contigs.fasta"
        self._save_reads(assembly_result.contigs, contigs_path)
        
        # Save assembly graph
        if assembly_result.graph:
            graph_path = self.output_dir / "assembly_graph.gfa"
            self._save_graph(assembly_result.graph, graph_path)
        
        self.logger.info(f"✓ Assembly complete")
        self.logger.info(f"  Contigs: {len(assembly_result.contigs)}")
        self.logger.info(f"  Total bases: {sum(len(c.seq) for c in assembly_result.contigs):,}")
        self.logger.info(f"  Output: {contigs_path}")
        
        # Save contigs
        contigs_path = self.output_dir / "contigs.fasta"
        self._save_reads(assembly_result.contigs, contigs_path)
        
        # Save assembly graph
        if assembly_result.dbg:
            graph_path = self.output_dir / "assembly_graph.gfa"
            self._save_graph(assembly_result.dbg, graph_path)
        
        self.logger.info(f"✓ Assembly complete")
        self.logger.info(f"  Contigs: {len(assembly_result.contigs)}")
        self.logger.info(f"  Total bases: {sum(len(c.seq) for c in assembly_result.contigs):,}")
        self.logger.info(f"  Output: {contigs_path}")
        
        self.state['assembly_result'] = assembly_result
        self.state['final_contigs'] = assembly_result.contigs
    
    # ========================================================================
    # Assembly Orchestrator Methods (integrated from pipeline_orchestrator.py)
    # ========================================================================
    
    def _run_assembly_pipeline(
        self,
        read_type: str,
        illumina_files: Optional[List[str]] = None,
        long_read_files: Optional[List[str]] = None,
        ul_read_files: Optional[List[str]] = None,
        hic_data: Optional[Any] = None,
        kmer_prediction: Optional[Any] = None,
        ml_k_model: Optional[Any] = None
    ) -> AssemblyResult:
        """
        Orchestrate the full assembly flow based on read_type.
        
        Now uses streaming file-based processing to avoid memory issues.
        
        Args:
            read_type: One of "illumina", "hifi", "ont", "ancient", "mixed"
            illumina_files: List of corrected Illumina read file paths
            long_read_files: List of corrected long read file paths (HiFi, ONT)
            ul_read_files: List of corrected ultra-long read file paths
            hic_data: Optional Hi-C data for scaffolding
            ml_k_model: Optional ML model for regional k-mer selection
        
        Returns:
            AssemblyResult containing DBG, string graph, contigs, scaffolds
        
        Pipeline flows:
            Illumina:  OLC → DBG → String Graph → Hi-C
            HiFi:      DBG → String Graph → Hi-C
            ONT:       DBG → String Graph (+ UL overlay) → Hi-C
            Ancient:   DBG → String Graph → Hi-C
            Mixed:     DBG → String Graph → Hi-C
        """
        self.logger.info(f"Starting assembly pipeline for read_type={read_type}")
        
        result = AssemblyResult(stats={})
        
        # Determine pipeline flow based on read_type
        if read_type == "illumina":
            result = self._run_illumina_pipeline(
                illumina_files, long_read_files, ul_read_files,
                hic_data, ml_k_model
            )
        
        elif read_type == "hifi" or read_type == "pacbio":
            result = self._run_hifi_pipeline(
                long_read_files, ul_read_files,
                hic_data, ml_k_model
            )
        
        elif read_type == "ont" or read_type == "ont_r10" or read_type == "ont_ultralong":
            result = self._run_ont_pipeline(
                long_read_files, ul_read_files,
                hic_data, ml_k_model
            )
        
        elif read_type == "ancient":
            result = self._run_ancient_pipeline(
                long_read_files, ul_read_files, hic_data, ml_k_model
            )
        
        elif read_type == "mixed":
            result = self._run_mixed_pipeline(
                illumina_files, long_read_files, ul_read_files,
                hic_data, ml_k_model
            )
        
        else:
            raise ValueError(
                f"Unknown read_type: {read_type}. "
                f"Must be one of: illumina, hifi, ont, ancient, mixed"
            )
        
        self.logger.info("Assembly pipeline complete")
        return result
    
    def _run_illumina_pipeline(
        self,
        illumina_files: Optional[List[str]],
        long_read_files: Optional[List[str]],
        ul_read_files: Optional[List[str]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Illumina pipeline with correct module ordering.
        
        CORRECT FLOW:
        1. OLC → artificial long reads
        2. DBG engine → raw graph
        3. EdgeWarden → filter low-quality edges
        4. PathWeaver → select optimal paths
        5. String Graph engine → UL overlay (if UL reads present)
        6. ThreadCompass → route UL reads (if UL reads present)
        7. Hi-C scaffolding → proximity-guided scaffolds
        8. Haplotype Detangler → phase the graph
        9. Iteration cycle (EdgeWarden → PathWeaver → String Graph → ThreadCompass)
        10. SVScribe → detect structural variants
        11. Finalize → extract contigs/scaffolds
        
        NOTE: Haplotype Detangler runs AFTER first complete iteration, not before.
        """
        self.logger.info("Running Illumina pipeline: OLC → DBG → EdgeWarden → PathWeaver → String Graph → ThreadCompass → Hi-C → Phasing → Iteration → SVScribe")
        
        result = AssemblyResult(stats={'read_type': 'illumina'})
        
        # Step 1: OLC assembly to generate artificial long reads from Illumina files
        # NOTE: For OLC, we load Illumina reads into memory (they're short, OLC output is smaller)
        self.logger.info("Step 1: Running OLC to generate artificial long reads")
        illumina_reads = []
        if illumina_files:
            for illumina_file in illumina_files:
                illumina_reads.extend(self._read_file_streaming(Path(illumina_file)))
        
        olc_long_reads = self._run_olc(illumina_reads) if illumina_reads else []
        result.stats['olc_contigs'] = len(olc_long_reads)
        
        # Step 2: Build DBG from artificial long reads (NO phasing yet)
        self.logger.info("Step 2: Building DBG from OLC-derived long reads")
        base_k = self.config.get('dbg_k', 31)
        min_coverage = self.config.get('min_coverage', 2)
        
        result.dbg = build_dbg_from_long_reads(
            olc_long_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 3: EdgeWarden - Filter low-quality edges
        self.logger.info("Step 3: Applying EdgeWarden to filter edges")
        use_edge_ai = self.config.get('assembly', {}).get('graph', {}).get('use_edge_ai', True)
        edge_warden = EdgeWarden(use_ai=use_edge_ai and self.config['ai']['enabled'])
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 4: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 4: Applying PathWeaver for path selection")
        use_path_ai = self.config.get('assembly', {}).get('dbg', {}).get('use_path_gnn', True)
        path_weaver = PathWeaver(use_ai=use_path_ai and self.config['ai']['enabled'])
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = path_weaver.get_path_count()
        
        # Step 5: Build string graph (if UL reads available)
        # Load UL reads from files if present
        ul_reads = []
        if ul_read_files:
            self.logger.info("Step 5: Loading UL reads for string graph overlay")
            for ul_file in ul_read_files:
                ul_reads.extend(self._read_file_streaming(Path(ul_file)))
        
        if ul_reads:
            self.logger.info("Step 5: Building string graph with UL overlay")
            ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
            
            # Step 6: ThreadCompass - Route UL reads through graph
            self.logger.info("Step 6: Applying ThreadCompass for UL routing")
            use_ul_ai = self.config.get('assembly', {}).get('string_graph', {}).get('use_ul_routing_ai', True)
            thread_compass = ThreadCompass(use_ai=use_ul_ai and self.config['ai']['enabled'])
            result.string_graph = thread_compass.route_ul_reads(
                result.string_graph,
                ul_reads  # Already loaded above
            )
            result.stats['ul_reads_routed'] = thread_compass.get_routed_count()
        else:
            self.logger.info("Step 5-6: No UL reads; skipping string graph overlay and ThreadCompass")
            thread_compass = None  # For iteration cycle
        
        # Step 7: Hi-C proximity edge addition (if available)
        if hic_data:
            self.logger.info("Step 7: Adding Hi-C proximity edges to graph")
            # Hi-C adds long-range connectivity edges to the graph
            # NO contig extraction - work directly on graph structure
            graph_to_annotate = result.string_graph or result.dbg
            result.dbg = self._add_hic_edges_to_graph(graph_to_annotate, hic_data)
            if result.string_graph:
                result.string_graph.dbg = result.dbg  # Update underlying DBG
            result.stats['hic_edges_added'] = self._count_hic_edges(result.dbg)
            result.stats['hic_coverage_pct'] = self._calculate_hic_coverage(result.dbg)
        
        # Step 8: Haplotype Detangler - Phase the graph (AFTER first iteration)
        # Uses Hi-C connectivity clusters from proximity edges already in graph
        self.logger.info("Step 8: Applying Haplotype Detangler for phasing")
        use_diploid_ai = self.config.get('assembly', {}).get('diploid', {}).get('use_diploid_ai', True)
        haplotype_detangler = HaplotypeDetangler(use_ai=use_diploid_ai and self.config['ai']['enabled'])
        
        graph_to_phase = result.string_graph or result.dbg
        phasing_result = haplotype_detangler.phase_graph(
            graph_to_phase,
            use_hic_edges=True  # Hi-C edges already integrated into graph
        )
        result.stats['haplotypes_detected'] = phasing_result.num_haplotypes
        result.stats['phasing_confidence'] = phasing_result.confidence
        
        # Step 9: Iteration cycle (refine graph with phasing information)
        max_iterations = self.config.get('correction', {}).get('max_iterations', 3)
        # Reduce iterations if AI enabled (more accurate per iteration)
        if self.config['ai']['enabled']:
            max_iterations = min(max_iterations, 2)
        
        self.logger.info(f"Step 9: Running {max_iterations} refinement iterations")
        for iteration in range(max_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
            
            # Re-apply EdgeWarden with phasing context
            result.dbg = edge_warden.filter_graph(
                result.dbg,
                phasing_info=phasing_result
            )
            
            # Re-apply PathWeaver with phasing context
            result.dbg = path_weaver.resolve_paths(
                result.dbg,
                phasing_info=phasing_result
            )
            
            # Rebuild string graph if UL reads present
            if ul_reads:
                result.string_graph = build_string_graph_from_dbg_and_ul(
                    result.dbg,
                    ul_anchors,
                    min_support=self.config.get('min_ul_support', 2)
                )
                
                # Re-route UL reads
                if thread_compass:  # Only if ThreadCompass was initialized
                    result.string_graph = thread_compass.route_ul_reads(
                        result.string_graph,
                        ul_reads,
                        phasing_info=phasing_result
                    )
            
            result.stats[f'iteration_{iteration + 1}_edges'] = len(result.dbg.edges)
        
        # Step 10: SVScribe - Detect structural variants
        # Distinguishes sequence evidence (DBG/UL) vs proximity evidence (Hi-C edges)
        self.logger.info("Step 10: Running SVScribe for structural variant detection")
        use_sv_ai = self.config.get('assembly', {}).get('sv_detection', {}).get('use_sv_ai', True)
        sv_scribe = SVScribe(use_ai=use_sv_ai and self.config['ai']['enabled'])
        
        sv_calls = sv_scribe.detect_svs(
            graph=result.string_graph or result.dbg,
            ul_routes=thread_compass.get_routes() if (thread_compass and ul_reads) else None,
            distinguish_edge_types=True,  # Separate sequence vs Hi-C evidence
            phasing_info=phasing_result
        )
        result.stats['sv_calls'] = len(sv_calls)
        result.stats['sv_types'] = sv_scribe.get_sv_type_counts()
        
        # Step 11: Extract final contigs from refined graph
        self.logger.info("Step 11: Extracting final contigs from refined graph")
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 12: Build scaffolds by traversing graph with Hi-C edge priority
        if hic_data:
            self.logger.info("Step 12: Building scaffolds from graph (prioritizing Hi-C edges)")
            result.scaffolds = self._extract_scaffolds_from_graph(
                result.string_graph or result.dbg,
                prefer_hic_edges=True,  # Follow Hi-C edges for long-range connections
                phasing_info=phasing_result  # Haplotype-aware scaffolding
            )
            result.stats['num_scaffolds'] = len(result.scaffolds)
            result.stats['scaffold_n50'] = self._calculate_n50([len(s.seq) for s in result.scaffolds])
        
        return result
    
    def _run_hifi_pipeline(
        self,
        hifi_read_files: Optional[List[str]],
        ul_read_files: Optional[List[str]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        HiFi pipeline with correct module ordering.
        
        CORRECT FLOW:
        1. DBG engine → raw graph (skip OLC for HiFi)
        2. EdgeWarden → filter low-quality edges
        3. PathWeaver → select optimal paths
        4. String Graph engine → UL overlay (if UL reads present)
        5. ThreadCompass → route UL reads (if UL reads present)
        6. Hi-C scaffolding → proximity-guided scaffolds
        7. Haplotype Detangler → phase the graph
        8. Iteration cycle (EdgeWarden → PathWeaver → String Graph → ThreadCompass)
        9. SVScribe → detect structural variants
        10. Finalize → extract contigs/scaffolds
        """
        self.logger.info("Running HiFi pipeline: DBG → EdgeWarden → PathWeaver → String Graph → ThreadCompass → Hi-C → Phasing → Iteration → SVScribe")
        
        result = AssemblyResult(stats={'read_type': 'hifi'})
        
        # Step 1: Load HiFi reads from files and build DBG
        hifi_reads = []
        if hifi_read_files:
            self.logger.info(f"Step 1: Loading HiFi reads from {len(hifi_read_files)} files")
            for hifi_file in hifi_read_files:
                hifi_reads.extend(self._read_file_streaming(Path(hifi_file)))
        
        self.logger.info(f"Step 1: Building DBG from {len(hifi_reads)} HiFi reads")
        base_k = self.config.get('dbg_k', 51)  # Higher k for HiFi
        min_coverage = self.config.get('min_coverage', 2)
        
        result.dbg = build_dbg_from_long_reads(
            hifi_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 2: EdgeWarden - Filter low-quality edges
        self.logger.info("Step 2: Applying EdgeWarden to filter edges")
        use_edge_ai = self.config.get('assembly', {}).get('graph', {}).get('use_edge_ai', True)
        edge_warden = EdgeWarden(use_ai=use_edge_ai and self.config['ai']['enabled'])
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 3: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 3: Applying PathWeaver for path selection")
        use_path_ai = self.config.get('assembly', {}).get('dbg', {}).get('use_path_gnn', True)
        path_weaver = PathWeaver(use_ai=use_path_ai and self.config['ai']['enabled'])
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = path_weaver.get_path_count()
        
        # Step 4: Build string graph (if UL reads available)
        # Load UL reads from files if present
        ul_reads = []
        if ul_read_files:
            self.logger.info("Step 4: Loading UL reads for string graph overlay")
            for ul_file in ul_read_files:
                ul_reads.extend(self._read_file_streaming(Path(ul_file)))
        
        if ul_reads:
            self.logger.info("Step 4: Building string graph with UL overlay")
            ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
            
            # Step 5: ThreadCompass - Route UL reads through graph
            self.logger.info("Step 5: Applying ThreadCompass for UL routing")
            use_ul_ai = self.config.get('assembly', {}).get('string_graph', {}).get('use_ul_routing_ai', True)
            thread_compass = ThreadCompass(use_ai=use_ul_ai and self.config['ai']['enabled'])
            result.string_graph = thread_compass.route_ul_reads(
                result.string_graph,
                ul_reads  # Already loaded above
            )
            result.stats['ul_reads_routed'] = thread_compass.get_routed_count()
        else:
            self.logger.info("Step 4-5: No UL reads; skipping string graph overlay and ThreadCompass")
            thread_compass = None  # For iteration cycle
        
        # Step 6: Hi-C proximity edge addition (if available)
        if hic_data:
            self.logger.info("Step 6: Adding Hi-C proximity edges to graph")
            graph_to_annotate = result.string_graph or result.dbg
            result.dbg = self._add_hic_edges_to_graph(graph_to_annotate, hic_data)
            if result.string_graph:
                result.string_graph.dbg = result.dbg
            result.stats['hic_edges_added'] = self._count_hic_edges(result.dbg)
            result.stats['hic_coverage_pct'] = self._calculate_hic_coverage(result.dbg)
        
        # Step 7: Haplotype Detangler - Phase using Hi-C connectivity clusters (HiFi)
        self.logger.info("Step 7: Applying Haplotype Detangler for phasing")
        use_diploid_ai = self.config.get('assembly', {}).get('diploid', {}).get('use_diploid_ai', True)
        haplotype_detangler = HaplotypeDetangler(use_ai=use_diploid_ai and self.config['ai']['enabled'])
        
        graph_to_phase = result.string_graph or result.dbg
        phasing_result = haplotype_detangler.phase_graph(
            graph_to_phase,
            use_hic_edges=True  # Use Hi-C edges for connectivity clustering
        )
        result.stats['haplotypes_detected'] = phasing_result.num_haplotypes
        result.stats['phasing_confidence'] = phasing_result.confidence
        
        # Step 8: Iteration cycle (refine graph with phasing information)
        max_iterations = self.config.get('correction', {}).get('max_iterations', 3)
        if self.config['ai']['enabled']:
            max_iterations = min(max_iterations, 2)
        
        self.logger.info(f"Step 8: Running {max_iterations} refinement iterations")
        for iteration in range(max_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
            
            result.dbg = edge_warden.filter_graph(result.dbg, phasing_info=phasing_result)
            result.dbg = path_weaver.resolve_paths(result.dbg, phasing_info=phasing_result)
            
            if ul_reads:
                result.string_graph = build_string_graph_from_dbg_and_ul(
                    result.dbg, ul_anchors,
                    min_support=self.config.get('min_ul_support', 2)
                )
                if thread_compass:  # Only if ThreadCompass was initialized
                    result.string_graph = thread_compass.route_ul_reads(
                        result.string_graph,
                        ul_reads,
                        phasing_info=phasing_result
                    )
            
            result.stats[f'iteration_{iteration + 1}_edges'] = len(result.dbg.edges)
        
        # Step 9: SVScribe - Detect structural variants (distinguish edge types - HiFi)
        self.logger.info("Step 9: Running SVScribe for structural variant detection")
        use_sv_ai = self.config.get('assembly', {}).get('sv_detection', {}).get('use_sv_ai', True)
        sv_scribe = SVScribe(use_ai=use_sv_ai and self.config['ai']['enabled'])
        
        sv_calls = sv_scribe.detect_svs(
            graph=result.string_graph or result.dbg,
            ul_routes=thread_compass.get_routes() if (thread_compass and ul_reads) else None,
            distinguish_edge_types=True,  # Sequence vs Hi-C evidence
            phasing_info=phasing_result
        )
        result.stats['sv_calls'] = len(sv_calls)
        result.stats['sv_types'] = sv_scribe.get_sv_type_counts()
        
        # Step 10: Extract final contigs from refined graph (HiFi)
        self.logger.info("Step 10: Extracting final contigs from refined graph")
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 11: Build scaffolds prioritizing Hi-C edges (HiFi)
        if hic_data:
            self.logger.info("Step 11: Building scaffolds from graph (prioritizing Hi-C edges)")
            result.scaffolds = self._extract_scaffolds_from_graph(
                result.string_graph or result.dbg,
                prefer_hic_edges=True,
                phasing_info=phasing_result
            )
            result.stats['num_scaffolds'] = len(result.scaffolds)
            result.stats['scaffold_n50'] = self._calculate_n50([len(s.seq) for s in result.scaffolds])
        
        return result
    
    def _run_ont_pipeline(
        self,
        ont_read_files: Optional[List[str]],
        ul_read_files: Optional[List[str]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        ONT pipeline with correct module ordering.
        
        CORRECT FLOW:
        1. DBG engine → raw graph (skip OLC for ONT)
        2. EdgeWarden → filter low-quality edges
        3. PathWeaver → select optimal paths
        4. String Graph engine → UL overlay (essential for ONT)
        5. ThreadCompass → route UL reads
        6. Hi-C scaffolding → proximity-guided scaffolds
        7. Haplotype Detangler → phase the graph
        8. Iteration cycle (EdgeWarden → PathWeaver → String Graph → ThreadCompass)
        9. SVScribe → detect structural variants
        10. Finalize → extract contigs/scaffolds
        """
        self.logger.info("Running ONT pipeline: DBG → EdgeWarden → PathWeaver → String Graph + UL → ThreadCompass → Hi-C → Phasing → Iteration → SVScribe")
        
        result = AssemblyResult(stats={'read_type': 'ont'})
        
        # Step 1: Load ONT reads from files and build DBG
        ont_reads = []
        if ont_read_files:
            self.logger.info(f"Step 1: Loading ONT reads from {len(ont_read_files)} files")
            for ont_file in ont_read_files:
                ont_reads.extend(self._read_file_streaming(Path(ont_file)))
        
        self.logger.info(f"Step 1: Building DBG from {len(ont_reads)} ONT reads")
        base_k = self.config.get('dbg_k', 41)  # Medium k for ONT
        min_coverage = self.config.get('min_coverage', 3)  # Higher for noisy ONT
        
        result.dbg = build_dbg_from_long_reads(
            ont_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 2: EdgeWarden - Filter low-quality edges (critical for noisy ONT)
        self.logger.info("Step 2: Applying EdgeWarden to filter edges")
        use_edge_ai = self.config.get('assembly', {}).get('graph', {}).get('use_edge_ai', True)
        edge_warden = EdgeWarden(use_ai=use_edge_ai and self.config['ai']['enabled'])
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 3: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 3: Applying PathWeaver for path selection")
        use_path_ai = self.config.get('assembly', {}).get('dbg', {}).get('use_path_gnn', True)
        path_weaver = PathWeaver(use_ai=use_path_ai and self.config['ai']['enabled'])
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = path_weaver.get_path_count()
        
        # Step 4: Build string graph (UL overlay is critical for ONT)
        # Load UL reads from files if present
        ul_reads = []
        if ul_read_files:
            self.logger.info("Step 4: Loading UL reads for string graph overlay (essential for ONT)")
            for ul_file in ul_read_files:
                ul_reads.extend(self._read_file_streaming(Path(ul_file)))
        
        if ul_reads:
            self.logger.info("Step 4: Building string graph with UL overlay (essential for ONT)")
            ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
            
            # Step 5: ThreadCompass - Route UL reads through graph
            self.logger.info("Step 5: Applying ThreadCompass for UL routing")
            use_ul_ai = self.config.get('assembly', {}).get('string_graph', {}).get('use_ul_routing_ai', True)
            thread_compass = ThreadCompass(use_ai=use_ul_ai and self.config['ai']['enabled'])
            result.string_graph = thread_compass.route_ul_reads(
                result.string_graph,
                ul_reads  # Already loaded above
            )
            result.stats['ul_reads_routed'] = thread_compass.get_routed_count()
        else:
            self.logger.warning("Step 4-5: No UL reads provided for ONT assembly - may have low contiguity")
            thread_compass = None  # For iteration cycle
        
        # Step 6: Hi-C proximity edge addition (if available)
        if hic_data:
            self.logger.info("Step 6: Adding Hi-C proximity edges to graph")
            graph_to_annotate = result.string_graph or result.dbg
            result.dbg = self._add_hic_edges_to_graph(graph_to_annotate, hic_data)
            if result.string_graph:
                result.string_graph.dbg = result.dbg
            result.stats['hic_edges_added'] = self._count_hic_edges(result.dbg)
            result.stats['hic_coverage_pct'] = self._calculate_hic_coverage(result.dbg)
        
        # Step 7: Haplotype Detangler - Phase using Hi-C connectivity clusters
        self.logger.info("Step 7: Applying Haplotype Detangler for phasing")
        use_diploid_ai = self.config.get('assembly', {}).get('diploid', {}).get('use_diploid_ai', True)
        haplotype_detangler = HaplotypeDetangler(use_ai=use_diploid_ai and self.config['ai']['enabled'])
        
        graph_to_phase = result.string_graph or result.dbg
        phasing_result = haplotype_detangler.phase_graph(
            graph_to_phase,
            use_hic_edges=True  # Use Hi-C edges for connectivity clustering
        )
        result.stats['haplotypes_detected'] = phasing_result.num_haplotypes
        result.stats['phasing_confidence'] = phasing_result.confidence
        
        # Step 8: Iteration cycle (refine graph with phasing information)
        max_iterations = self.config.get('correction', {}).get('max_iterations', 3)
        if self.config['ai']['enabled']:
            max_iterations = min(max_iterations, 2)
        
        self.logger.info(f"Step 8: Running {max_iterations} refinement iterations")
        for iteration in range(max_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
            
            result.dbg = edge_warden.filter_graph(result.dbg, phasing_info=phasing_result)
            result.dbg = path_weaver.resolve_paths(result.dbg, phasing_info=phasing_result)
            
            if ul_reads:
                result.string_graph = build_string_graph_from_dbg_and_ul(
                    result.dbg, ul_anchors,
                    min_support=self.config.get('min_ul_support', 2)
                )
                if thread_compass:  # Only if ThreadCompass was initialized
                    result.string_graph = thread_compass.route_ul_reads(
                        result.string_graph,
                        ul_reads,
                        phasing_info=phasing_result
                    )
            
            result.stats[f'iteration_{iteration + 1}_edges'] = len(result.dbg.edges)
        
        # Step 9: SVScribe - Detect structural variants (distinguish edge types)
        self.logger.info("Step 9: Running SVScribe for structural variant detection")
        use_sv_ai = self.config.get('assembly', {}).get('sv_detection', {}).get('use_sv_ai', True)
        sv_scribe = SVScribe(use_ai=use_sv_ai and self.config['ai']['enabled'])
        
        sv_calls = sv_scribe.detect_svs(
            graph=result.string_graph or result.dbg,
            ul_routes=thread_compass.get_routes() if (thread_compass and ul_reads) else None,
            distinguish_edge_types=True,  # Sequence vs Hi-C evidence
            phasing_info=phasing_result
        )
        result.stats['sv_calls'] = len(sv_calls)
        result.stats['sv_types'] = sv_scribe.get_sv_type_counts()
        
        # Step 10: Extract final contigs from refined graph
        self.logger.info("Step 10: Extracting final contigs from refined graph")
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 11: Build scaffolds prioritizing Hi-C edges
        if hic_data:
            self.logger.info("Step 11: Building scaffolds from graph (prioritizing Hi-C edges)")
            result.scaffolds = self._extract_scaffolds_from_graph(
                result.string_graph or result.dbg,
                prefer_hic_edges=True,
                phasing_info=phasing_result
            )
            result.stats['num_scaffolds'] = len(result.scaffolds)
            result.stats['scaffold_n50'] = self._calculate_n50([len(s.seq) for s in result.scaffolds])
        
        return result
    
    def _run_ancient_pipeline(
        self,
        corrected_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Ancient DNA pipeline with correct module ordering.
        
        Assumes preprocessing has already merged/assembled ancient DNA
        fragments into longer "pseudo-long-reads".
        
        Flow identical to HiFi pipeline:
        DBG → EdgeWarden → PathWeaver → String Graph → ThreadCompass → Hi-C → Phasing → Iteration → SVScribe
        """
        self.logger.info("Running Ancient DNA pipeline: DBG → EdgeWarden → PathWeaver → String Graph → ThreadCompass → Hi-C → Phasing → Iteration → SVScribe")
        
        # Ancient DNA uses similar flow to HiFi (same module ordering)
        return self._run_hifi_pipeline(
            corrected_reads, ul_reads, ul_anchors, hic_data, ml_k_model
        )
    
    def _run_mixed_pipeline(
        self,
        corrected_reads: List[SeqRead],
        long_reads: Optional[List[SeqRead]],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Mixed pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Uses best available long reads (prioritize: HiFi > ONT > corrected).
        String graph built IF UL reads available (always follows DBG).
        """
        self.logger.info("Running Mixed pipeline: DBG → String Graph → Hi-C")
        
        # Use long_reads if available, otherwise corrected_reads
        reads_for_dbg = long_reads if long_reads else corrected_reads
        
        return self._run_hifi_pipeline(
            reads_for_dbg, ul_reads, ul_anchors, hic_data, ml_k_model
        )
    
    def _run_olc(self, corrected_reads: List[SeqRead]) -> List[SeqRead]:
        """
        Run OLC assembly to generate artificial long reads from Illumina.
        
        Args:
            corrected_reads: List of corrected Illumina reads (SeqRead objects)
        
        Returns:
            List of assembled contigs as SeqRead objects (artificial long reads)
        """
        self.logger.info("Running OLC assembly")
        
        # Use existing ContigBuilder for OLC
        builder = ContigBuilder(
            k_size=self.config.get('olc_k', 31),
            min_overlap=self.config.get('olc_min_overlap', 50),
            min_contig_length=self.config.get('olc_min_contig', 500),
            use_gpu=False,
            use_adaptive_k=False
        )
        
        olc_contigs = builder.build_contigs(corrected_reads, verbose=False)
        return olc_contigs
    
    def _generate_ul_anchors(self, dbg: DBGGraph, ul_reads: List[SeqRead]) -> List[ULAnchor]:
        """
        Generate UL anchor points by aligning UL reads to DBG nodes.
        
        Uses ULReadMapper with GraphAligner to create high-quality mappings,
        then converts to simplified ULAnchor format for string graph construction.
        
        Args:
            dbg: De Bruijn graph with nodes to anchor to
            ul_reads: List of ultra-long reads (SeqRead objects)
        
        Returns:
            List of ULAnchor objects mapping UL reads to DBG nodes
        
        Algorithm:
        1. Convert DBG to KmerGraph format (required by ULReadMapper)
        2. Use ULReadMapper to align UL reads to graph (k-mer anchors + GraphAligner)
        3. Extract anchor points from ULReadMapping results
        4. Convert to simplified ULAnchor format
        5. Filter by quality metrics
        """
        self.logger.info(f"Generating UL anchors for {len(ul_reads)} reads...")
        
        # Convert DBG to KmerGraph format
        self.logger.debug("Converting DBG to KmerGraph format...")
        kmer_graph = self._dbg_to_kmer_graph(dbg)
        
        # Initialize LongReadOverlay (formerly ULReadMapper)
        mapper = LongReadOverlay(
            min_anchor_length=100,
            min_identity=0.7,
            anchor_k=15,
            min_anchors=3,
            use_mbg=True  # Enable GraphAligner
        )
        
        # Prepare read data as (read_id, sequence) tuples
        ul_read_tuples = [(read.id, read.sequence) for read in ul_reads]
        
        # Map reads to graph (uses batch processing automatically)
        self.logger.info("Mapping UL reads to DBG nodes...")
        mappings = mapper.build_ul_paths(kmer_graph, ul_read_tuples, min_coverage=0.3)
        
        self.logger.info(f"Successfully mapped {len(mappings)}/{len(ul_reads)} UL reads")
        
        # Convert ULReadMapping objects to ULAnchor objects
        anchors = []
        for mapping in mappings:
            # Each node in the path becomes an anchor point
            for i, node_id in enumerate(mapping.path):
                orientation = mapping.orientations[i] if i < len(mapping.orientations) else '+'
                
                # Extract anchor details from mapping.anchors if available
                node_anchors = [a for a in mapping.anchors if a.node_id == node_id]
                
                if node_anchors:
                    # Use actual anchor positions
                    for anchor_obj in node_anchors:
                        anchor = ULAnchor(
                            ul_read_id=mapping.read_id,
                            node_id=str(node_id),
                            read_start=anchor_obj.read_start,
                            read_end=anchor_obj.read_end,
                            node_start=anchor_obj.node_start,
                            strand=anchor_obj.orientation
                        )
                        anchors.append(anchor)
                else:
                    # Create synthetic anchor for this node
                    # (happens when GraphAligner fills gaps between exact anchors)
                    anchor = ULAnchor(
                        ul_read_id=mapping.read_id,
                        node_id=str(node_id),
                        read_start=0,  # Unknown exact position
                        read_end=0,
                        node_start=0,
                        strand=orientation
                    )
                    anchors.append(anchor)
        
        self.logger.info(f"Generated {len(anchors)} UL anchor points from {len(mappings)} read mappings")
        
        return anchors
    
    def _dbg_to_kmer_graph(self, dbg: DBGGraph) -> KmerGraph:
        """
        Convert DBGGraph to KmerGraph format for ULReadMapper.
        
        KmerGraph is the format used by assembly_core.py components.
        This conversion enables reuse of ULReadMapper functionality.
        
        Args:
            dbg: DBGGraph object from dbg_engine.py
        
        Returns:
            KmerGraph object compatible with ULReadMapper
        """
        kmer_graph = KmerGraph()
        
        # Convert nodes
        for node_id, dbg_node in dbg.nodes.items():
            kmer_node = KmerNode(
                id=node_id,
                seq=dbg_node.seq,
                coverage=dbg_node.coverage,
                length=dbg_node.length
            )
            kmer_graph.nodes[node_id] = kmer_node
        
        # Convert edges
        for edge_id, dbg_edge in dbg.edges.items():
            kmer_edge = KmerEdge(
                id=edge_id,
                from_id=dbg_edge.from_id,
                to_id=dbg_edge.to_id,
                coverage=dbg_edge.coverage
            )
            kmer_graph.edges[edge_id] = kmer_edge
        
        return kmer_graph
    
    def _extract_contigs_from_graph(self, graph: Union[DBGGraph, StringGraph]) -> List[SeqRead]:
        """
        Extract linear contig sequences from DBG or string graph.
        
        Traverses graph to find linear paths and generate consensus sequences.
        
        Args:
            graph: DBGGraph or StringGraph object
        
        Returns:
            List of SeqRead objects representing assembled contigs
        """
        self.logger.info("Extracting contigs from graph")
        
        # Placeholder implementation
        if isinstance(graph, DBGGraph):
            # Convert DBG nodes to contigs
            contigs = []
            for node_id, node in graph.nodes.items():
                # Each unitig becomes a contig
                
                # Calculate quality from coverage (log scale)
                # Q = 20 + 10 * log10(coverage + 1), capped at Q40
                avg_qual = min(40, int(20 + 10 * math.log10(node.coverage + 1)))
                quality = chr(avg_qual + 33) * len(node.seq)
                
                contig = SeqRead(
                    id=f"contig_{node_id}",
                    sequence=node.seq,
                    quality=quality,
                    metadata={
                        'source': 'dbg',
                        'node_id': node_id,
                        'coverage': node.coverage,
                        'length': node.length,
                        'recommended_k': node.recommended_k
                    }
                )
                contigs.append(contig)
            return contigs
        
        elif isinstance(graph, StringGraph):
            # Traverse string graph to find paths
            # Placeholder: just return DBG contigs
            return self._extract_contigs_from_graph(graph.dbg)
        
        return []
    
    def _run_hic_scaffolding(
        self,
        contigs: List[SeqRead],
        hic_data: Optional[Any],
        phasing_info: Optional[Any] = None
    ) -> List[SeqRead]:
        """
        Run Hi-C scaffolding on contigs.
        
        TODO: This is a placeholder implementation.
        
        Args:
            contigs: List of assembled contigs (SeqRead objects)
            hic_data: Hi-C read data for proximity ligation
            phasing_info: Optional phasing information from Haplotype Detangler
        
        Returns:
            List of scaffolded contigs (SeqRead objects)
        
        Real implementation would:
        1. Build contig contact matrix from Hi-C alignments
        2. Cluster contigs by contact frequency
        3. Order and orient contigs within scaffolds
        4. Use phasing_info to separate haplotypes if available
        5. Join contigs with N-gaps
        6. Return scaffolded sequences
        """
        self.logger.warning(
            f"Hi-C scaffolding is not yet implemented. "
            f"Returning {len(contigs)} unscaffolded contigs."
        )
        
        if phasing_info:
            self.logger.info(f"Phasing information available: {phasing_info.num_haplotypes} haplotypes detected")
        
        # Placeholder: return contigs unchanged
        return contigs
    
    def _add_hic_edges_to_graph(
        self,
        graph: Union[DBGGraph, StringGraph],
        hic_data: Any
    ) -> Union[DBGGraph, StringGraph]:
        """
        Add Hi-C proximity edges to the graph (Strategy 4: Hi-C as graph annotation).
        
        Hi-C creates NEW edges representing long-range proximity contacts, not modifying
        existing sequence-based edges. These edges enable:
        - Phasing via connectivity clustering
        - SVScribe evidence type distinction
        - Scaffolding by traversing proximity edges
        
        Args:
            graph: DBGGraph or StringGraph to annotate
            hic_data: Hi-C read pairs for proximity ligation analysis
        
        Returns:
            Graph with additional Hi-C proximity edges
        
        Implementation:
        1. Build contact matrix from Hi-C alignments to graph nodes
        2. Identify strong long-range contacts (threshold-based)
        3. Create new edges with type='hic' and proximity scores
        4. Add edges to graph without modifying existing edges
        """
        self.logger.info("Adding Hi-C proximity edges to graph")
        
        # TODO: Full implementation
        # For now, log that Hi-C edge addition is pending
        self.logger.warning(
            "Hi-C edge addition is not yet fully implemented. "
            "Graph structure maintained, but Hi-C edges not added."
        )
        
        # Placeholder: Return graph unchanged
        # Real implementation would:
        # 1. Parse Hi-C read pairs
        # 2. Align to graph nodes
        # 3. Build contact matrix
        # 4. Threshold for significant contacts
        # 5. Add proximity edges with type='hic'
        
        return graph
    
    def _count_hic_edges(self, graph: Union[DBGGraph, StringGraph]) -> int:
        """
        Count number of Hi-C proximity edges in graph.
        
        Args:
            graph: Graph to analyze
        
        Returns:
            Number of edges with type='hic'
        """
        # Placeholder implementation
        count = 0
        if isinstance(graph, DBGGraph):
            for edge in graph.edges:
                if hasattr(edge, 'edge_type') and edge.edge_type == 'hic':
                    count += 1
        elif isinstance(graph, StringGraph) and hasattr(graph, 'dbg'):
            count = self._count_hic_edges(graph.dbg)
        
        return count
    
    def _calculate_hic_coverage(self, graph: Union[DBGGraph, StringGraph]) -> float:
        """
        Calculate percentage of graph nodes with Hi-C edge connections.
        
        Args:
            graph: Graph to analyze
        
        Returns:
            Percentage (0-100) of nodes connected by Hi-C edges
        """
        # Placeholder implementation
        if isinstance(graph, DBGGraph):
            if not graph.nodes:
                return 0.0
            
            nodes_with_hic = set()
            for edge in graph.edges:
                if hasattr(edge, 'edge_type') and edge.edge_type == 'hic':
                    if hasattr(edge, 'source'):
                        nodes_with_hic.add(edge.source)
                    if hasattr(edge, 'target'):
                        nodes_with_hic.add(edge.target)
            
            return (len(nodes_with_hic) / len(graph.nodes)) * 100.0
        
        elif isinstance(graph, StringGraph) and hasattr(graph, 'dbg'):
            return self._calculate_hic_coverage(graph.dbg)
        
        return 0.0
    
    def _extract_scaffolds_from_graph(
        self,
        graph: Union[DBGGraph, StringGraph],
        prefer_hic_edges: bool = True,
        phasing_info: Optional[Any] = None
    ) -> List[SeqRead]:
        """
        Build scaffolds by traversing graph, prioritizing Hi-C proximity edges.
        
        Unlike traditional scaffolding (contig concatenation with gaps), this
        traverses the graph structure using Hi-C edges to connect distant nodes.
        
        Args:
            graph: Graph to traverse
            prefer_hic_edges: If True, follow Hi-C edges for long-range connections
            phasing_info: Optional haplotype information for phased scaffolding
        
        Returns:
            List of scaffold sequences (SeqRead objects)
        
        Algorithm:
        1. Start from high-coverage seed nodes
        2. Traverse edges preferring: sequence edges locally, Hi-C edges for jumps
        3. Use phasing_info to keep scaffolds haplotype-consistent
        4. Generate scaffold sequences from traversed paths
        5. Mark Hi-C junction points in metadata
        """
        self.logger.info("Building scaffolds from graph with Hi-C edge priority")
        
        # TODO: Full implementation
        # For now, extract contigs and return (no Hi-C scaffolding yet)
        self.logger.warning(
            "Hi-C-guided scaffold extraction not yet fully implemented. "
            "Returning contigs instead of scaffolds."
        )
        
        # Placeholder: Return contigs
        # Real implementation would:
        # 1. Find scaffold seed nodes (high coverage, unambiguous)
        # 2. Extend scaffolds using sequence edges
        # 3. Jump gaps using Hi-C edges
        # 4. Respect phasing boundaries
        # 5. Generate scaffold sequences with marked Hi-C junctions
        
        return self._extract_contigs_from_graph(graph)
    
    # ========================================================================
    # Finishing and utility methods
    # ========================================================================
    
    def _step_finish(self):
        """Finishing step (polishing, gap filling, etc.)."""
        self.logger.info("Finishing assembly...")
        
        # Load contigs
        if self.state['final_contigs'] is None:
            contigs_path = self.output_dir / "contigs.fasta"
            if contigs_path.exists():
                contigs = list(read_fasta(contigs_path))
            else:
                raise FileNotFoundError("No contigs found for finishing step")
        else:
            contigs = self.state['final_contigs']
        
        # Polishing (if enabled)
        if self.config['finishing']['polishing']['enabled']:
            self.logger.info("  Polishing contigs...")
            # TODO: Implement polishing
            self.logger.info("  ⚠ Polishing not yet implemented")
        
        # Gap filling (if enabled)
        if self.config['finishing']['gap_filling']['enabled']:
            self.logger.info("  Filling gaps...")
            # TODO: Implement gap filling
            self.logger.info("  ⚠ Gap filling not yet implemented")
        
        # Claude AI finishing (if enabled)
        if self.config['ai']['claude']['use_for_finishing']:
            self.logger.info("  Running Claude AI finishing...")
            # TODO: Implement Claude finishing
            self.logger.info("  ⚠ Claude finishing not yet implemented")
        
        # Save final assembly
        final_path = self.output_dir / "final_assembly.fasta"
        self._save_reads(contigs, final_path)
        
        self.logger.info(f"✓ Finishing complete")
        self.logger.info(f"  Final assembly: {final_path}")
    
    def _create_checkpoint(self, step: str):
        """Create checkpoint after step completion."""
        checkpoint_path = self.checkpoint_dir / f"{step}.checkpoint"
        
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'completed_steps': self.state['completed_steps'],
            'state_files': {
                'reads': str(self.output_dir / "corrected_reads.fastq") if step in ['correct', 'assemble', 'finish'] else None,
                'contigs': str(self.output_dir / "contigs.fasta") if step in ['assemble', 'finish'] else None,
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Checkpoint created: {checkpoint_path}")
    
    def _find_last_checkpoint(self) -> str:
        """Find the last completed checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("*.checkpoint"))
        
        if not checkpoints:
            self.logger.warning("No checkpoints found, starting from beginning")
            return self.steps[0]
        
        # Find most recent checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            checkpoint_data = json.load(f)
        
        completed_step = checkpoint_data['step']
        
        # Resume from next step after completed
        if completed_step in self.steps:
            step_index = self.steps.index(completed_step)
            if step_index + 1 < len(self.steps):
                return self.steps[step_index + 1]
        
        # If last step was completed, start from beginning
        return self.steps[0]
    
    def _read_file_streaming(self, reads_path: Path):
        """Stream reads from file without loading all into memory."""
        if reads_path.suffix in ['.fq', '.fastq']:
            yield from read_fastq(reads_path)
        elif reads_path.suffix in ['.fa', '.fasta']:
            yield from read_fasta(reads_path)
        else:
            # Try FASTQ first, fall back to FASTA
            try:
                yield from read_fastq(reads_path)
            except:
                yield from read_fasta(reads_path)
    
    def _write_read_to_file(self, read: SeqRead, file_handle):
        """Write a single read to an open file handle (FASTQ format)."""
        file_handle.write(f"@{read.id}\n")
        file_handle.write(f"{read.sequence}\n")
        file_handle.write("+\n")
        file_handle.write(f"{read.quality}\n")
    
    def _save_reads(self, reads: List[SeqRead], output_path: Path):
        """Save reads to file."""
        output_format = self.config['output']['format']
        
        if output_format == 'fastq' or output_path.suffix in ['.fq', '.fastq']:
            write_fastq(reads, output_path)
        elif output_format == 'fasta' or output_path.suffix in ['.fa', '.fasta']:
            write_fasta(reads, output_path)
        else:
            # Default to FASTA
            write_fasta(reads, output_path)
    
    def _save_graph(self, graph, output_path: Path):
        """Save assembly graph to GFA format."""
        # TODO: Implement GFA export
        self.logger.debug(f"Graph export to {output_path} not yet implemented")
    
    def _load_ai_models(self) -> Dict[str, Any]:
        """Load AI models (lazy loading)."""
        if self._ai_models:
            return self._ai_models
        
        self.logger.info("Loading AI models...")
        
        # Load correction models
        if self.config['ai']['correction']['adaptive_kmer']['enabled']:
            # TODO: Load actual model
            self._ai_models['adaptive_kmer'] = None
            self.logger.info("  ✓ AdaptiveKmerAI (not yet trained)")
        
        if self.config['ai']['correction']['base_error_classifier']['enabled']:
            # TODO: Load actual model
            self._ai_models['base_error_classifier'] = None
            self.logger.info("  ✓ BaseErrorClassifierAI (not yet trained)")
        
        # Load assembly models
        if self.config['ai']['assembly']['edge_ai']['enabled']:
            # TODO: Load actual model
            self._ai_models['edge_ai'] = None
            self.logger.info("  ✓ EdgeAI (not yet trained)")
        
        if self.config['ai']['assembly']['path_gnn']['enabled']:
            # TODO: Load actual model
            self._ai_models['path_gnn'] = None
            self.logger.info("  ✓ PathGNN (not yet trained)")
        
        if self.config['ai']['assembly']['diploid_ai']['enabled']:
            # TODO: Load actual model
            self._ai_models['diploid_ai'] = None
            self.logger.info("  ✓ DiploidAI (not yet trained)")
        
        if self.config['ai']['assembly']['ul_routing_ai']['enabled']:
            # TODO: Load actual model
            self._ai_models['ul_routing_ai'] = None
            self.logger.info("  ✓ ULRoutingAI (not yet trained)")
        
        if self.config['ai']['assembly']['sv_ai']['enabled']:
            # TODO: Load actual model
            self._ai_models['sv_ai'] = None
            self.logger.info("  ✓ SVAI (not yet trained)")
        
        return self._ai_models

