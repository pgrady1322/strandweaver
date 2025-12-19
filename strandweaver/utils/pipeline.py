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
)
from ..assembly_core.illumina_olc_contig_module import ContigBuilder
from ..assembly_core.dbg_engine_module import (
    KmerGraph,
    KmerNode,
    KmerEdge,
)


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
            'reads': None,
            'corrected_reads': None,
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
        if step == 'profile':
            self._step_profile()
        elif step == 'correct':
            self._step_correct()
        elif step == 'assemble':
            self._step_assemble()
        elif step == 'finish':
            self._step_finish()
        else:
            raise ValueError(f"Unknown step: {step}")
    
    def _step_profile(self):
        """Error profiling step."""
        self.logger.info("Profiling sequencing errors...")
        
        # Load reads
        reads = self._load_reads()
        
        # Sample reads for profiling
        sample_size = self.config['profiling']['sample_size']
        sampled_reads = reads[:sample_size] if len(reads) > sample_size else reads
        
        self.logger.info(f"Profiling {len(sampled_reads)} reads (sampled from {len(reads)})")
        
        # Run profiler
        profiler = ErrorProfiler()
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
        
        # Store reads for next step
        self.state['reads'] = reads
        self.state['error_profile'] = profile
    
    def _step_correct(self):
        """Error correction step."""
        self.logger.info("Correcting sequencing errors...")
        
        # Load reads if not already in memory
        if self.state['reads'] is None:
            reads = self._load_reads()
        else:
            reads = self.state['reads']
        
        # Load AI models if enabled
        use_ai = self.config['ai']['enabled']
        ai_models = self._load_ai_models() if use_ai else None
        
        # Determine technology-specific corrector
        technologies = self.config['runtime']['technologies']
        corrected_reads = []
        
        for tech in set(technologies):
            tech_reads = [r for r, t in zip(reads, technologies) if t == tech]
            
            if tech == 'illumina':
                corrector = IlluminaCorrector(
                    kmer_size=self.config['correction']['illumina']['kmer_size'],
                    ml_k_model=ai_models.get('adaptive_kmer') if use_ai else None
                )
            elif tech == 'ancient':
                corrector = AncientDNACorrector(
                    ml_damage_model=ai_models.get('base_error_classifier') if use_ai else None
                )
            elif tech == 'ont' or tech == 'ont_ultralong':
                corrector = ONTCorrector(
                    ml_error_model=ai_models.get('base_error_classifier') if use_ai else None
                )
            elif tech == 'pacbio':
                corrector = PacBioCorrector(
                    ml_error_model=ai_models.get('base_error_classifier') if use_ai else None
                )
            else:
                # Auto-detect or unknown - use generic
                corrector = IlluminaCorrector()
            
            self.logger.info(f"  Correcting {len(tech_reads)} {tech} reads...")
            tech_corrected = corrector.correct_reads(tech_reads)
            corrected_reads.extend(tech_corrected)
        
        # Save corrected reads
        corrected_path = self.output_dir / "corrected_reads.fastq"
        self._save_reads(corrected_reads, corrected_path)
        
        self.logger.info(f"✓ Corrected {len(corrected_reads)} reads")
        self.logger.info(f"  Output: {corrected_path}")
        
        self.state['corrected_reads'] = corrected_reads
    
    def _step_assemble(self):
        """Assembly step."""
        self.logger.info("Assembling contigs...")
        
        # Load corrected reads
        if self.state['corrected_reads'] is None:
            corrected_path = self.output_dir / "corrected_reads.fastq"
            if corrected_path.exists():
                corrected_reads = list(read_fastq(corrected_path))
            else:
                # Fall back to original reads if correction was skipped
                corrected_reads = self._load_reads()
        else:
            corrected_reads = self.state['corrected_reads']
        
        # Load AI models if enabled
        use_ai = self.config['ai']['enabled']
        ai_models = self._load_ai_models() if use_ai else None
        
        # Detect primary technology for assembly strategy
        technologies = self.config['runtime']['technologies']
        primary_tech = max(set(technologies), key=technologies.count)
        
        # Run assembly using integrated assembly orchestrator methods
        self.logger.info(f"  Primary technology: {primary_tech}")
        self.logger.info(f"  AI-powered assembly: {use_ai}")
        
        # Determine read type for assembly routing
        read_type = primary_tech.lower()
        
        # Prepare HiC data if enabled
        hic_data = None
        if self.config['scaffolding']['hic']['enabled']:
            hic_data = {
                'r1': self.config['runtime'].get('hic_r1'), 
                'r2': self.config['runtime'].get('hic_r2')
            }
        
        # Run technology-specific assembly pipeline
        assembly_result = self._run_assembly_pipeline(
            read_type=read_type,
            corrected_reads=corrected_reads,
            long_reads=None,  # Use corrected_reads as long reads
            ul_reads=None,    # TODO: Extract UL reads if available
            ul_anchors=None,
            hic_data=hic_data,
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
        corrected_reads: List[SeqRead],
        olc_long_reads: Optional[List[SeqRead]] = None,
        long_reads: Optional[List[SeqRead]] = None,
        ul_reads: Optional[List[SeqRead]] = None,
        ul_anchors: Optional[List[ULAnchor]] = None,
        hic_data: Optional[Any] = None,
        ml_k_model: Optional[Any] = None
    ) -> AssemblyResult:
        """
        Orchestrate the full assembly flow based on read_type.
        
        Integrated from AssemblyOrchestrator.run_assembly_pipeline().
        
        Args:
            read_type: One of "illumina", "hifi", "ont", "ancient", "mixed"
            corrected_reads: Error-corrected reads (from preprocessing)
            olc_long_reads: Optional artificial long reads from OLC (for Illumina)
            long_reads: Optional true long reads (HiFi, ONT)
            ul_reads: Optional ultra-long ONT reads (for string graph overlay)
            ul_anchors: Optional pre-computed UL anchors to DBG nodes
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
                corrected_reads, olc_long_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type == "hifi" or read_type == "pacbio":
            result = self._run_hifi_pipeline(
                long_reads or corrected_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type == "ont" or read_type == "ont_r10" or read_type == "ont_ultralong":
            result = self._run_ont_pipeline(
                long_reads or corrected_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type == "ancient":
            result = self._run_ancient_pipeline(
                corrected_reads, ul_reads, ul_anchors, hic_data, ml_k_model
            )
        
        elif read_type == "mixed":
            result = self._run_mixed_pipeline(
                corrected_reads, long_reads, ul_reads, ul_anchors,
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
        corrected_reads: List[SeqRead],
        olc_long_reads: Optional[List[SeqRead]],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Illumina pipeline: OLC → DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Use OLC to generate artificial long reads (if not provided)
        2. Build DBG from artificial long reads
        3. Build string graph IF UL reads available (always follows DBG)
        4. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running Illumina pipeline: OLC → DBG → String Graph → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'illumina'})
        
        # Step 1: OLC assembly to generate artificial long reads
        if olc_long_reads is None:
            self.logger.info("Running OLC to generate artificial long reads")
            olc_long_reads = self._run_olc(corrected_reads)
            result.stats['olc_contigs'] = len(olc_long_reads)
        else:
            self.logger.info(f"Using {len(olc_long_reads)} pre-computed OLC contigs")
        
        # Step 2: Build DBG from artificial long reads
        self.logger.info("Building DBG from OLC-derived long reads")
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
        
        # Step 3: Build string graph
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        else:
            self.logger.info("No UL reads; skipping string graph overlay")
        
        # Step 4: Extract contigs from graph
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 5: Hi-C scaffolding (if available)
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
        return result
    
    def _run_hifi_pipeline(
        self,
        hifi_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        HiFi pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Build DBG directly from HiFi reads (skip OLC)
        2. Build string graph IF UL reads available (always follows DBG)
        3. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running HiFi pipeline: DBG → String Graph → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'hifi'})
        
        # Step 1: Build DBG from HiFi reads
        self.logger.info(f"Building DBG from {len(hifi_reads)} HiFi reads")
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
        
        # Step 2: Build string graph
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        
        # Step 3: Extract contigs
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 4: Hi-C scaffolding
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
        return result
    
    def _run_ont_pipeline(
        self,
        ont_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        ONT pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Build DBG from ONT reads (skip OLC)
        2. Build string graph IF UL reads available (always follows DBG, essential for ONT)
        3. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running ONT pipeline: DBG → String Graph + UL → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'ont'})
        
        # Step 1: Build DBG from ONT reads
        self.logger.info(f"Building DBG from {len(ont_reads)} ONT reads")
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
        
        # Step 2: Build string graph (UL overlay is critical for ONT)
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay (essential for ONT)")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        else:
            self.logger.warning("No UL reads provided for ONT assembly - may have low contiguity")
        
        # Step 3: Extract contigs
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 4: Hi-C scaffolding
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
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
        Ancient DNA pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Assumes preprocessing has already merged/assembled ancient DNA
        fragments into longer "pseudo-long-reads".
        String graph built IF UL reads available (always follows DBG).
        """
        self.logger.info("Running Ancient DNA pipeline: DBG → String Graph → Hi-C")
        
        # Ancient DNA uses similar flow to HiFi
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
        
        # Initialize ULReadMapper
        mapper = ULReadMapper(
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
    
    def _run_hic_scaffolding(self, contigs: List[SeqRead], hic_data: Optional[Any]) -> List[SeqRead]:
        """
        Run Hi-C scaffolding on contigs.
        
        TODO: This is a placeholder implementation.
        
        Args:
            contigs: List of assembled contigs (SeqRead objects)
            hic_data: Hi-C read data for proximity ligation
        
        Returns:
            List of scaffolded contigs (SeqRead objects)
        
        Real implementation would:
        1. Build contig contact matrix from Hi-C alignments
        2. Cluster contigs by contact frequency
        3. Order and orient contigs within scaffolds
        4. Join contigs with N-gaps
        5. Return scaffolded sequences
        """
        self.logger.warning(
            f"Hi-C scaffolding is not yet implemented. "
            f"Returning {len(contigs)} unscaffolded contigs."
        )
        
        # Placeholder: return contigs unchanged
        return contigs
    
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
    
    def _load_reads(self) -> List[SeqRead]:
        """Load input reads from files."""
        all_reads = []
        
        for reads_file in self.config['runtime']['reads']:
            reads_path = Path(reads_file)
            
            if reads_path.suffix in ['.fq', '.fastq']:
                reads = list(read_fastq(reads_path))
            elif reads_path.suffix in ['.fa', '.fasta']:
                reads = list(read_fasta(reads_path))
            else:
                # Try both
                try:
                    reads = list(read_fastq(reads_path))
                except:
                    reads = list(read_fasta(reads_path))
            
            all_reads.extend(reads)
            self.logger.info(f"  Loaded {len(reads):,} reads from {reads_file}")
        
        return all_reads
    
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

