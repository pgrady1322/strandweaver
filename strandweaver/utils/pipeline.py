#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Master Pipeline Orchestrator — coordinates the complete assembly pipeline
from preprocessing through assembly to finishing.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
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
from ..preprocessing.kweaver_module import KmerPrediction  # Import from kweaver to avoid circular import
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
# Lazy import to avoid circular dependency
# from ..assembly_core.illumina_olc_contig_module import ContigBuilder
from ..assembly_core.dbg_engine_module import (
    KmerGraph,
    KmerNode,
    KmerEdge,
)
from ..assembly_core.edgewarden_module import EdgeWarden
from ..assembly_core.pathweaver_module import PathWeaver
from ..assembly_core.haplotype_detangler_module import HaplotypeDetangler
from ..assembly_core.threadcompass_module import ThreadCompass
from ..assembly_core.svscribe_module import SVScribe, svs_to_dict_list
from ..io_utils.assembly_export import (
    export_graph_to_gfa,
    export_assembly_stats,
    export_for_bandageng,
)
from collections import defaultdict
from dataclasses import asdict


# ============================================================================
# Data Structures (from PreprocessingCoordinator)
# ============================================================================
# Note: KmerPrediction is imported from kweaver_module above

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
            'path_weaver': None,  # PathWeaver instance (carries misassembly flags)
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
        elif step == 'misassembly_report':
            self._step_misassembly_report()
        elif step == 'classify_chromosomes':
            self._step_classify_chromosomes()
        else:
            raise ValueError(f"Unknown step: {step}")
    
    def _step_kweaver(self):
        """K-Weaver: Predict optimal k-mer sizes for all pipeline stages."""
        self.logger.info("Running K-Weaver k-mer prediction...")
        
        from ..preprocessing import KWeaverPredictor, FeatureExtractor
        
        # Check if AI models should be used
        use_ai = self.config['ai']['enabled'] and \
                 self.config['ai']['correction'].get('adaptive_kmer', {}).get('enabled', False)
        
        # Initialize K-Weaver with model_dir if configured
        kweaver_model_dir = self._resolve_sub_model_path(
            explicit_path=self.config['ai']['correction']['adaptive_kmer'].get('model_path'),
            model_dir=self._resolve_model_dir(),
            sub_folder='kweaver',
        )
        predictor = KWeaverPredictor(model_dir=kweaver_model_dir, use_ml=use_ai)
        
        # Extract features from first input file (representative sample)
        first_file = Path(self.state['read_files'][0])
        self.logger.info(f"  Extracting features from: {first_file.name}")
        
        # Initialize extractor with subsampling
        subsample_size = min(100000, self.config['profiling']['sample_size'])
        extractor = FeatureExtractor(subsample=subsample_size)
        features = extractor.extract_from_file(first_file)
        
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
        """
        Error profiling step — profiles EACH technology separately.

        Each sequencing technology has fundamentally different error profiles
        (substitution-dominated for Illumina, indel-dominated for ONT,
        low-error for HiFi). The pipeline generates a per-technology error
        profile and passes each to the corresponding corrector.
        """
        self.logger.info("Profiling sequencing errors (per-technology)...")

        # Get k-mer prediction from K-Weaver (if available)
        kmer_prediction = self.state.get('kmer_prediction')
        profile_k = kmer_prediction.dbg_k if kmer_prediction else 21  # Default fallback

        self.logger.info(f"  Using k={profile_k} for error profiling (from K-Weaver)")

        sample_size = self.config['profiling']['sample_size']

        # Convert string → ReadTechnology enum
        from strandweaver.preprocessing.read_classification_utility import ReadTechnology
        tech_map = {
            'illumina': ReadTechnology.ILLUMINA,
            'ancient': ReadTechnology.ANCIENT_DNA,
            'ont': ReadTechnology.ONT_REGULAR,
            'ont_ultralong': ReadTechnology.ONT_ULTRALONG,
            'pacbio': ReadTechnology.PACBIO_HIFI,
            'hic': ReadTechnology.HI_C,
            'auto': ReadTechnology.UNKNOWN,
        }

        # ----------------------------------------------------------------
        # Profile each (read_file, technology) pair independently
        # ----------------------------------------------------------------
        error_profiles: Dict[str, Any] = {}          # keyed by technology string
        read_files = self.state['read_files']
        technologies = self.state['technologies']

        for i, (reads_file, tech_str) in enumerate(zip(read_files, technologies)):
            # Skip Hi-C reads — they are proximity ligation, not error-corrected
            if tech_str == 'hic':
                self.logger.info(f"  Skipping Hi-C file {Path(reads_file).name} (not error-profiled)")
                continue

            reads_path = Path(reads_file)
            technology = tech_map.get(tech_str, ReadTechnology.UNKNOWN)

            self.logger.info(f"\n  Profiling file {i+1}/{len(read_files)}: "
                             f"{reads_path.name} ({tech_str})")

            # Sample reads from this file
            sampled_reads = []
            read_count = 0
            for read in self._read_file_streaming(reads_path):
                if len(sampled_reads) < sample_size:
                    sampled_reads.append(read)
                read_count += 1
                if len(sampled_reads) >= sample_size:
                    break

            self.logger.info(f"    Sampled {len(sampled_reads):,} / {read_count:,} reads")

            # Run profiler
            profiler = ErrorProfiler(k=profile_k, sample_size=sample_size)
            profile_path = self.output_dir / f"error_profile_{tech_str}_{i}.json"

            profile = profiler.profile(
                reads_file=reads_path,
                technology=technology,
                output_file=profile_path,
            )

            self.logger.info(f"    ✓ Profile saved: {profile_path.name}")
            self.logger.info(f"      Technology: {profile.get('technology', 'unknown')}")
            self.logger.info(f"      Mean quality: {profile.get('mean_quality', 0):.2f}")

            # Store profile keyed by tech string.
            # If multiple files share a technology, keep the last profile
            # (they should have similar error characteristics).
            error_profiles[tech_str] = profile

        # Store all per-technology profiles
        self.state['error_profiles'] = error_profiles

        # Keep single 'error_profile' as the first profile
        if error_profiles:
            self.state['error_profile'] = next(iter(error_profiles.values()))
        else:
            self.state['error_profile'] = {}

        self.logger.info(f"\n✓ Error profiling complete — "
                         f"{len(error_profiles)} technology profile(s) generated")
    
    def _step_correct(self):
        """Error correction step — uses per-technology error profiles."""
        self.logger.info("Correcting sequencing errors...")
        
        # Load AI models if enabled
        use_ai = self.config['ai']['enabled']
        ai_models = self._load_ai_models() if use_ai else None
        
        # Get K-Weaver k-mer predictions
        kmer_prediction = self.state.get('kmer_prediction')
        
        # Per-technology error profiles from _step_profile()
        error_profiles = self.state.get('error_profiles', {})
        
        # Process each input file independently by technology
        # This minimizes read combining/separating operations
        technologies = self.state['technologies']
        read_files = self.state['read_files']
        
        # Track Illumina paired-end relationship through correction
        paired_idx = self.config.get('runtime', {}).get('illumina_paired_indices')
        
        for i, (reads_file, tech) in enumerate(zip(read_files, technologies)):
            # Determine R1/R2 label for Illumina paired-end files
            pe_label = ''
            if tech == 'illumina' and paired_idx is not None:
                if i == paired_idx[0]:
                    pe_label = '_R1'
                elif i == paired_idx[1]:
                    pe_label = '_R2'
            self.logger.info(f"\n  Processing file {i+1}/{len(read_files)}: {Path(reads_file).name} ({tech}{pe_label})")
            
            # Look up per-technology error profile (from _step_profile)
            tech_profile = error_profiles.get(tech, {})
            if tech_profile:
                self.logger.info(f"    Using {tech} error profile "
                                 f"(mean_quality={tech_profile.get('mean_quality', 0):.2f})")
            else:
                self.logger.info(f"    No error profile for {tech} — using defaults")
            
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
                    k_size=correction_k
                )
                self.logger.info(f"    Using k={correction_k} for ONT correction")
            elif tech == 'pacbio':
                # Use polish_k for PacBio correction (high precision)
                correction_k = kmer_prediction.polish_k if kmer_prediction else 55
                corrector = PacBioCorrector(
                    k_size=correction_k
                )
                self.logger.info(f"    Using k={correction_k} for PacBio correction")
            else:
                # Auto-detect or unknown - use generic
                corrector = IlluminaCorrector()
            
            # Correct reads in streaming fashion
            input_path = Path(reads_file)
            output_path = self.output_dir / f"corrected_{tech}{pe_label}_{i}.fastq"
            
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
        
        # Preserve Illumina paired-end info for downstream consumers
        if paired_idx is not None:
            self.state['illumina_paired_corrected'] = {
                'r1': str(self.output_dir / f"corrected_illumina_R1_{paired_idx[0]}.fastq"),
                'r2': str(self.output_dir / f"corrected_illumina_R2_{paired_idx[1]}.fastq"),
            }
            self.logger.info(f"  Illumina paired-end files preserved: "
                             f"R1={Path(self.state['illumina_paired_corrected']['r1']).name}, "
                             f"R2={Path(self.state['illumina_paired_corrected']['r2']).name}")
    
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
        
        # ----------------------------------------------------------------
        # Fallback: if correction was skipped (--skip-correction), use the
        # original raw input reads so the assembly step has data to work with.
        # ----------------------------------------------------------------
        if not corrected_files:
            self.logger.warning(
                "No corrected files found (correction may have been skipped). "
                "Falling back to raw input reads for assembly."
            )
            raw_reads = self.state['read_files']
            raw_techs = self.state['technologies']
            corrected_files = {}
            for path, tech in zip(raw_reads, raw_techs):
                tech_key = tech.lower()
                if tech_key not in corrected_files:
                    corrected_files[tech_key] = []
                corrected_files[tech_key].append(str(path))
        
        # Separate by read type for assembly (without loading into memory)
        long_read_files = []
        ul_read_files = []
        illumina_read_files = []
        
        for tech, files in corrected_files.items():
            if tech in ('ont', 'pacbio', 'hifi', 'ont_r10'):
                long_read_files.extend(files)
            elif tech == 'ont_ultralong':
                ul_read_files.extend(files)
            elif tech == 'illumina':
                illumina_read_files.extend(files)
            else:
                # Unknown tech — treat as long reads by default
                long_read_files.extend(files)
        
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
        
        # Save assembly graph (use string_graph if available, otherwise dbg)
        final_graph = assembly_result.string_graph or assembly_result.dbg
        if final_graph:
            graph_path = self.output_dir / "assembly_graph.gfa"
            self._save_graph(final_graph, graph_path)
        
        self.logger.info(f"✓ Assembly complete")
        self.logger.info(f"  Contigs: {len(assembly_result.contigs)}")
        self.logger.info(f"  Total bases: {sum(len(c.sequence) for c in assembly_result.contigs):,}")
        self.logger.info(f"  Output: {contigs_path}")
        
        self.state['assembly_result'] = assembly_result
        self.state['final_contigs'] = assembly_result.contigs
        # Store the assembly graph for downstream steps (classify_chromosomes)
        self.state['graph'] = assembly_result.string_graph or assembly_result.dbg
    
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
        # Use K-Weaver prediction for optimal k-mer size
        kmer_prediction = self.state.get('kmer_prediction')
        base_k = kmer_prediction.dbg_k if kmer_prediction else self.config.get('dbg_k', 31)
        min_coverage = self.config.get('min_coverage', 2)
        self.logger.info(f"  Using k={base_k} for DBG construction (from K-Weaver)")
        
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
        ew_model_dir = (self._ai_models.get('edge_ai') or 'models/edgewarden')
        edge_warden = EdgeWarden(technology='illumina', model_dir=ew_model_dir)
        edge_warden.load_models()
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 4: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 4: Applying PathWeaver for path selection")
        path_weaver = PathWeaver(graph=result.dbg)
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = len(path_weaver.discovered_paths)
        self.state['path_weaver'] = path_weaver  # Preserve for misassembly reporting
        
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
        # Calculate average phasing confidence
        if phasing_result.confidence_scores:
            result.stats['phasing_confidence'] = sum(phasing_result.confidence_scores.values()) / len(phasing_result.confidence_scores)
        else:
            result.stats['phasing_confidence'] = 0.0
        
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
        
        # Export SV calls to JSON
        if sv_calls:
            self.logger.info(f"Exporting {len(sv_calls)} structural variant calls")
            sv_path = self.output_dir / "sv_calls.json"
            with open(sv_path, 'w') as f:
                json.dump(svs_to_dict_list(sv_calls), f, indent=2)
            self.logger.info(f"Saved SV calls to {sv_path}")
        
        # Export phasing information
        if phasing_result:
            self.logger.info("Exporting phasing information")
            phasing_path = self.output_dir / "phasing_info.json"
            with open(phasing_path, 'w') as f:
                # Convert dataclass to dict, handling non-serializable types
                phasing_dict = {
                    'num_haplotypes': phasing_result.num_haplotypes,
                    'node_assignments': phasing_result.node_assignments,
                    'confidence_scores': phasing_result.confidence_scores,
                    'metadata': {k: (list(v) if isinstance(v, set) else v) 
                                 for k, v in phasing_result.metadata.items()}
                }
                json.dump(phasing_dict, f, indent=2)
            self.logger.info(f"Saved phasing info to {phasing_path}")
        
        # Export assembly graph to GFA format
        self.logger.info("Exporting assembly graph to GFA format")
        gfa_path = self.output_dir / "assembly_graph.gfa"
        export_graph_to_gfa(
            result.string_graph or result.dbg,
            gfa_path,
            include_sequence=True
        )
        self.logger.info(f"Saved graph to {gfa_path}")
        
        # Step 11: Extract final contigs from refined graph
        self.logger.info("Step 11: Extracting final contigs from refined graph")
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Export assembly statistics
        self.logger.info("Calculating assembly statistics")
        stats_path = self.output_dir / "assembly_stats.json"
        contigs_list = [(c.id, c.sequence) for c in result.contigs]
        export_assembly_stats(
            result.string_graph or result.dbg,
            stats_path,
            contigs=contigs_list
        )
        self.logger.info(f"Saved statistics to {stats_path}")
        
        # Export coverage data for BandageNG visualization
        self.logger.info("Exporting coverage data for BandageNG")
        graph_for_export = result.string_graph or result.dbg
        # Extract coverage directly from graph nodes (already computed during assembly)
        long_coverage = {node_id: node.coverage for node_id, node in graph_for_export.nodes.items()}
        ul_coverage = None  # UL coverage would need separate tracking if needed
        hic_coverage = self._calculate_hic_support_per_node(graph_for_export) if hic_data else None
        edge_scores = self._extract_edge_scores(graph_for_export)
        
        export_for_bandageng(
            graph_for_export,
            output_prefix=self.output_dir / "assembly",
            long_read_coverage=long_coverage,
            ul_read_coverage=ul_coverage,
            hic_support=hic_coverage,
            edge_quality_scores=edge_scores
        )
        self.logger.info("Exported coverage and edge score CSVs for BandageNG")
        
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
            
            # Save scaffolds to FASTA
            if result.scaffolds:
                self.logger.info(f"Saving {len(result.scaffolds)} scaffolds")
                scaffolds_path = self.output_dir / "scaffolds.fasta"
                self._save_reads(result.scaffolds, scaffolds_path)
                self.logger.info(f"Saved scaffolds to {scaffolds_path}")
        
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
        # Use K-Weaver prediction for optimal k-mer size
        kmer_prediction = self.state.get('kmer_prediction')
        base_k = kmer_prediction.dbg_k if kmer_prediction else self.config.get('dbg_k', 51)
        min_coverage = self.config.get('min_coverage', 2)
        self.logger.info(f"  Using k={base_k} for DBG construction (from K-Weaver)")
        
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
        ew_model_dir = (self._ai_models.get('edge_ai') or 'models/edgewarden')
        edge_warden = EdgeWarden(technology='pacbio_hifi', model_dir=ew_model_dir)
        edge_warden.load_models()
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 3: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 3: Applying PathWeaver for path selection")
        path_weaver = PathWeaver(graph=result.dbg)
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = len(path_weaver.discovered_paths)
        self.state['path_weaver'] = path_weaver  # Preserve for misassembly reporting
        
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
        # Calculate average phasing confidence
        if phasing_result.confidence_scores:
            result.stats['phasing_confidence'] = sum(phasing_result.confidence_scores.values()) / len(phasing_result.confidence_scores)
        else:
            result.stats['phasing_confidence'] = 0.0
        
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
        
        # Export SV calls to JSON
        if sv_calls:
            self.logger.info(f"Exporting {len(sv_calls)} structural variant calls")
            sv_path = self.output_dir / "sv_calls.json"
            with open(sv_path, 'w') as f:
                json.dump(svs_to_dict_list(sv_calls), f, indent=2)
            self.logger.info(f"Saved SV calls to {sv_path}")
        
        # Export phasing information
        if phasing_result:
            self.logger.info("Exporting phasing information")
            phasing_path = self.output_dir / "phasing_info.json"
            with open(phasing_path, 'w') as f:
                # Convert dataclass to dict, handling non-serializable types
                phasing_dict = {
                    'num_haplotypes': phasing_result.num_haplotypes,
                    'node_assignments': phasing_result.node_assignments,
                    'confidence_scores': phasing_result.confidence_scores,
                    'metadata': {k: (list(v) if isinstance(v, set) else v) 
                                 for k, v in phasing_result.metadata.items()}
                }
                json.dump(phasing_dict, f, indent=2)
            self.logger.info(f"Saved phasing info to {phasing_path}")
        
        # Export assembly graph to GFA format
        self.logger.info("Exporting assembly graph to GFA format")
        gfa_path = self.output_dir / "assembly_graph.gfa"
        export_graph_to_gfa(
            result.string_graph or result.dbg,
            gfa_path,
            include_sequence=True
        )
        self.logger.info(f"Saved graph to {gfa_path}")
        
        # Step 10: Extract final contigs from refined graph (HiFi)
        self.logger.info("Step 10: Extracting final contigs from refined graph")
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Export assembly statistics
        self.logger.info("Calculating assembly statistics")
        stats_path = self.output_dir / "assembly_stats.json"
        contigs_list = [(c.id, c.sequence) for c in result.contigs]
        export_assembly_stats(
            result.string_graph or result.dbg,
            stats_path,
            contigs=contigs_list
        )
        self.logger.info(f"Saved statistics to {stats_path}")
        
        # Export coverage data for BandageNG visualization
        self.logger.info("Exporting coverage data for BandageNG")
        graph_for_export = result.string_graph or result.dbg
        # Extract coverage directly from graph nodes (already computed during assembly)
        long_coverage = {node_id: node.coverage for node_id, node in graph_for_export.nodes.items()}
        ul_coverage = None  # UL coverage would need separate tracking if needed
        hic_coverage = self._calculate_hic_support_per_node(graph_for_export) if hic_data else None
        edge_scores = self._extract_edge_scores(graph_for_export)
        
        export_for_bandageng(
            graph_for_export,
            output_prefix=self.output_dir / "assembly",
            long_read_coverage=long_coverage,
            ul_read_coverage=ul_coverage,
            hic_support=hic_coverage,
            edge_quality_scores=edge_scores
        )
        self.logger.info("Exported coverage and edge score CSVs for BandageNG")
        
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
            
            # Save scaffolds to FASTA
            if result.scaffolds:
                self.logger.info(f"Saving {len(result.scaffolds)} scaffolds")
                scaffolds_path = self.output_dir / "scaffolds.fasta"
                self._save_reads(result.scaffolds, scaffolds_path)
                self.logger.info(f"Saved scaffolds to {scaffolds_path}")
        
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
        # Use K-Weaver prediction for optimal k-mer size
        kmer_prediction = self.state.get('kmer_prediction')
        base_k = kmer_prediction.dbg_k if kmer_prediction else self.config.get('dbg_k', 41)
        min_coverage = self.config.get('min_coverage', 3)  # Higher for noisy ONT
        self.logger.info(f"  Using k={base_k} for DBG construction (from K-Weaver)")
        
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
        ew_model_dir = (self._ai_models.get('edge_ai') or 'models/edgewarden')
        edge_warden = EdgeWarden(technology='nanopore_r10', model_dir=ew_model_dir)
        edge_warden.load_models()
        result.dbg = edge_warden.filter_graph(result.dbg)
        result.stats['edges_after_warden'] = len(result.dbg.edges)
        
        # Step 3: PathWeaver - Select optimal paths through graph
        self.logger.info("Step 3: Applying PathWeaver for path selection")
        path_weaver = PathWeaver(graph=result.dbg)
        result.dbg = path_weaver.resolve_paths(result.dbg)
        result.stats['paths_selected'] = len(path_weaver.discovered_paths)
        self.state['path_weaver'] = path_weaver  # Preserve for misassembly reporting
        
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
        # Calculate average phasing confidence
        if phasing_result.confidence_scores:
            result.stats['phasing_confidence'] = sum(phasing_result.confidence_scores.values()) / len(phasing_result.confidence_scores)
        else:
            result.stats['phasing_confidence'] = 0.0
        
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
            corrected_reads, ul_reads, hic_data, ml_k_model
        )
    
    def _run_mixed_pipeline(
        self,
        corrected_reads: List[SeqRead],
        long_reads: Optional[List[SeqRead]],
        ul_reads: Optional[List[SeqRead]],
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
            reads_for_dbg, ul_reads, hic_data, ml_k_model
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
        
        # Lazy import to avoid circular dependency
        from ..assembly_core.illumina_olc_contig_module import ContigBuilder
        
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
        Applies minimum contig length filter from config.
        
        Args:
            graph: DBGGraph or StringGraph object
        
        Returns:
            List of SeqRead objects representing assembled contigs
        """
        self.logger.info("Extracting contigs from graph")
        
        # Get minimum contig length from config (default: 0 = keep all)
        min_len = self.config.get('runtime', {}).get('min_contig_length', 0)
        
        if isinstance(graph, DBGGraph):
            # Convert DBG nodes to contigs
            contigs = []
            filtered_count = 0
            filtered_bp = 0
            for node_id, node in graph.nodes.items():
                # Skip contigs below minimum length threshold
                if min_len > 0 and node.length < min_len:
                    filtered_count += 1
                    filtered_bp += node.length
                    continue
                
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
            
            if filtered_count > 0:
                self.logger.info(f"  Filtered {filtered_count} contigs below {min_len}bp "
                                 f"({filtered_bp:,} bp removed)")
            
            return contigs
        
        elif isinstance(graph, StringGraph):
            # Traverse string graph to find paths
            # Placeholder: just return DBG contigs
            return self._extract_contigs_from_graph(graph.dbg)
        
        return []
    
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
                     Can be: (r1_path, r2_path) tuple for FASTQ pairs, or
                     single BAM/SAM path for pre-aligned reads
        
        Returns:
            Graph with additional Hi-C proximity edges
        
        Implementation:
        1. Build contact matrix from Hi-C alignments to graph nodes
        2. Identify strong long-range contacts (threshold-based)
        3. Create new edges with type='hic' and proximity scores
        4. Add edges to graph without modifying existing edges
        """
        from ..assembly_utils.hic_graph_aligner import align_hic_reads_to_graph
        from ..assembly_core.strandtether_module import StrandTether
        
        self.logger.info("Adding Hi-C proximity edges to graph")
        
        if hic_data is None:
            self.logger.warning("No Hi-C data provided, skipping Hi-C edge addition")
            return graph
        
        # Get Hi-C configuration
        hic_config = self.config.get('hic', {})
        min_contacts = hic_config.get('min_contacts', 3)
        k = hic_config.get('k', 21)
        min_matches = hic_config.get('min_matches', 3)
        sample_size = hic_config.get('sample_size', None)
        min_contact_threshold = hic_config.get('min_contact_threshold', 2)
        
        try:
            # Step 1: Parse and align Hi-C reads to graph nodes
            self.logger.info("  Parsing and aligning Hi-C reads...")
            hic_pairs = align_hic_reads_to_graph(
                hic_data=hic_data,
                graph=graph,
                k=k,
                min_matches=min_matches,
                sample_size=sample_size
            )
            
            if not hic_pairs:
                self.logger.warning("No Hi-C pairs aligned to graph, skipping edge addition")
                return graph
            
            self.logger.info(f"  Aligned {len(hic_pairs)} Hi-C read pairs")
            
            # Step 2: Build contact map using StrandTether
            self.logger.info("  Building Hi-C contact map...")
            tether = StrandTether(min_contact_threshold=min_contact_threshold)
            contact_map = tether.build_contact_map(hic_pairs)
            
            # Persist contact map for downstream steps (chromosome classification)
            self.state['hic_contact_map'] = contact_map
            
            total_contacts = len(contact_map.contacts)
            self.logger.info(f"  Contact map built: {total_contacts} unique contacts")
            
            if total_contacts == 0:
                self.logger.warning("No contacts found in contact map")
                return graph
            
            # Step 3: Add Hi-C edges to graph
            self.logger.info(f"  Adding Hi-C edges (min_contacts={min_contacts})...")
            
            # Get maximum contact count for normalization
            max_contact_count = max(contact_map.contacts.values())
            
            # Track edges added
            edges_added = 0
            
            # Working with DBGGraph
            if isinstance(graph, DBGGraph):
                for (node1, node2), count in contact_map.contacts.items():
                    # Skip if below threshold
                    if count < min_contacts:
                        continue
                    
                    # Skip if nodes don't exist in graph
                    if node1 not in graph.nodes or node2 not in graph.nodes:
                        continue
                    
                    # Calculate normalized proximity score (0-1)
                    proximity_score = count / max_contact_count
                    
                    # Create Hi-C edge
                    hic_edge = self._create_hic_edge(
                        source=node1,
                        target=node2,
                        edge_type='hic',
                        proximity_score=proximity_score,
                        contact_count=count
                    )
                    
                    # Add edge to graph
                    graph.edges.append(hic_edge)
                    edges_added += 1
                
                self.logger.info(f"  ✓ Added {edges_added} Hi-C edges to graph")
            
            # Working with StringGraph (add to underlying DBG)
            elif isinstance(graph, StringGraph) and hasattr(graph, 'dbg'):
                self.logger.info("  Applying Hi-C edges to underlying DBG...")
                graph.dbg = self._add_hic_edges_to_graph(graph.dbg, hic_data)
                edges_added = self._count_hic_edges(graph.dbg)
                self.logger.info(f"  ✓ Added {edges_added} Hi-C edges to StringGraph's DBG")
            
            else:
                self.logger.warning(f"Unsupported graph type: {type(graph)}")
                return graph
            
            # Step 4: Log statistics
            if edges_added > 0:
                coverage = self._calculate_hic_coverage(graph)
                self.logger.info(f"  Hi-C coverage: {coverage:.1f}% of nodes connected")
            
            return graph
        
        except Exception as e:
            self.logger.error(f"Error adding Hi-C edges: {e}")
            self.logger.warning("Continuing without Hi-C edges")
            return graph
    
    def _create_hic_edge(
        self,
        source: int,
        target: int,
        edge_type: str,
        proximity_score: float,
        contact_count: int
    ):
        """
        Create a Hi-C proximity edge compatible with graph structure.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Edge type (should be 'hic')
            proximity_score: Normalized proximity score (0-1)
            contact_count: Raw Hi-C contact count
        
        Returns:
            Edge object compatible with DBGGraph/StringGraph
        """
        # Create a simple object with required attributes
        # This should match the structure used by DBGGraph edges
        class HiCEdge:
            def __init__(self, source, target, edge_type, proximity_score, contact_count):
                self.source = source
                self.target = target
                self.edge_type = edge_type
                self.proximity_score = proximity_score
                self.contact_count = contact_count
                self.quality_score = proximity_score  # Alias for consistency
                self.confidence = proximity_score
                
                # Additional metadata
                self.metadata = {
                    'hic_contact_count': contact_count,
                    'proximity_score': proximity_score
                }
            
            def __repr__(self):
                return (f"HiCEdge({self.source}->{self.target}, "
                       f"contacts={self.contact_count}, score={self.proximity_score:.3f})")
        
        return HiCEdge(source, target, edge_type, proximity_score, contact_count)
    
    def _count_hic_edges(self, graph: Union[DBGGraph, StringGraph]) -> int:
        """
        Count number of Hi-C proximity edges in graph.
        
        Args:
            graph: Graph to analyze
        
        Returns:
            Number of edges with type='hic'
        """
        count = 0
        if isinstance(graph, DBGGraph):
            for _edge_id, edge in graph.edges.items():
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
        if isinstance(graph, DBGGraph):
            if not graph.nodes:
                return 0.0
            
            nodes_with_hic = set()
            for _edge_id, edge in graph.edges.items():
                if hasattr(edge, 'edge_type') and edge.edge_type == 'hic':
                    source_id = edge.source if hasattr(edge, 'source') else edge.from_id
                    target_id = edge.target if hasattr(edge, 'target') else edge.to_id
                    nodes_with_hic.add(source_id)
                    nodes_with_hic.add(target_id)
            
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
        
        # Get scaffolding configuration
        hic_config = self.config.get('hic', {})
        gap_size = hic_config.get('gap_size', 100)
        min_scaffold_length = hic_config.get('min_scaffold_length', 1000)
        
        # Check if there are Hi-C edges in the graph
        hic_edge_count = self._count_hic_edges(graph)
        
        if hic_edge_count == 0 or not prefer_hic_edges:
            self.logger.warning(
                f"No Hi-C edges found (count={hic_edge_count}) or Hi-C disabled. "
                "Returning contigs instead of scaffolds."
            )
            return self._extract_contigs_from_graph(graph)
        
        self.logger.info(f"  Found {hic_edge_count} Hi-C edges for scaffolding")
        
        # Work with DBGGraph (extract from StringGraph if needed)
        if isinstance(graph, StringGraph) and hasattr(graph, 'dbg'):
            working_graph = graph.dbg
        elif isinstance(graph, DBGGraph):
            working_graph = graph
        else:
            self.logger.warning(f"Unsupported graph type: {type(graph)}, returning contigs")
            return self._extract_contigs_from_graph(graph)
        
        # Track visited nodes
        visited = set()
        scaffolds = []
        
        # Step 1: Get seed nodes (high coverage, unvisited)
        node_coverage = {}
        for node_id, node in working_graph.nodes.items():
            if hasattr(node, 'coverage'):
                node_coverage[node_id] = node.coverage
            elif hasattr(node, 'count'):
                node_coverage[node_id] = node.count
            else:
                node_coverage[node_id] = 1.0
        
        # Sort by coverage (descending)
        seed_nodes = sorted(node_coverage.keys(), key=lambda n: node_coverage[n], reverse=True)
        
        self.logger.info(f"  Traversing from {len(seed_nodes)} potential seed nodes...")
        
        # Step 2: Build scaffolds by traversing from each seed
        scaffold_count = 0
        scaffolds_by_haplotype = {'hapA': [], 'hapB': [], 'unphased': []}
        
        for seed in seed_nodes:
            if seed in visited:
                continue
            
            # Traverse with Hi-C priority
            path, junction_scores = self._traverse_with_hic_priority(
                graph=working_graph,
                start=seed,
                visited=visited,
                phasing_info=phasing_info
            )
            
            # Only create scaffold if path has at least 2 nodes
            node_ids = [p for p in path if isinstance(p, int)]
            if len(node_ids) >= 2:
                # Build scaffold sequence
                scaffold_seq, hic_junctions = self._build_scaffold_sequence(
                    graph=working_graph,
                    path=path,
                    default_gap_size=gap_size
                )
                
                # Calculate scaffold confidence
                confidence = self._calculate_scaffold_confidence(
                    path=path,
                    graph=working_graph,
                    hic_junction_scores=junction_scores
                )
                
                # Check minimum length
                if len(scaffold_seq) >= min_scaffold_length:
                    scaffold_count += 1
                    
                    # Determine haplotype
                    haplotype = None
                    if phasing_info:
                        seed_hap = self._get_node_haplotype(seed, phasing_info)
                        if seed_hap == 0:
                            haplotype = 'hapA'
                        elif seed_hap == 1:
                            haplotype = 'hapB'
                    
                    if haplotype is None:
                        haplotype = 'unphased'
                    
                    scaffold = SeqRead(
                        id=f"scaffold_{scaffold_count}",
                        sequence=scaffold_seq,
                        quality="~" * len(scaffold_seq),
                        metadata={
                            'node_path': node_ids,
                            'hic_scaffolded': True,
                            'hic_junctions': hic_junctions,
                            'num_nodes': len(node_ids),
                            'num_hic_junctions': len(hic_junctions),
                            'confidence': confidence,
                            'seed_node': seed,
                            'haplotype': haplotype
                        }
                    )
                    scaffolds.append(scaffold)
                    scaffolds_by_haplotype[haplotype].append(scaffold)
        
        self.logger.info(f"  ✓ Built {len(scaffolds)} scaffolds using Hi-C edges")
        
        # Report statistics
        if scaffolds:
            total_length = sum(len(s.sequence) for s in scaffolds)
            avg_length = total_length / len(scaffolds)
            n50 = self._calculate_n50([len(s.sequence) for s in scaffolds])
            
            # Calculate average confidence
            confidences = [s.metadata['confidence'] for s in scaffolds]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Count Hi-C junctions
            total_junctions = sum(s.metadata['num_hic_junctions'] for s in scaffolds)
            
            self.logger.info(f"  Total length: {total_length:,} bp")
            self.logger.info(f"  Average length: {avg_length:,.0f} bp")
            self.logger.info(f"  Scaffold N50: {n50:,} bp")
            self.logger.info(f"  Total Hi-C junctions: {total_junctions}")
            self.logger.info(f"  Average confidence: {avg_confidence:.2f}")
            
            # Report haplotype breakdown if phasing available
            if phasing_info:
                n_hapA = len(scaffolds_by_haplotype['hapA'])
                n_hapB = len(scaffolds_by_haplotype['hapB'])
                n_unphased = len(scaffolds_by_haplotype['unphased'])
                
                self.logger.info(f"  Haplotype breakdown:")
                self.logger.info(f"    HapA: {n_hapA} scaffolds")
                self.logger.info(f"    HapB: {n_hapB} scaffolds")
                self.logger.info(f"    Unphased: {n_unphased} scaffolds")
                
                # Optionally save haplotype-specific files
                if self.config.get('hic', {}).get('save_haplotype_scaffolds', True):
                    self._save_haplotype_scaffolds(scaffolds_by_haplotype)
        
        return scaffolds
    
    def _save_haplotype_scaffolds(self, scaffolds_by_haplotype: Dict[str, List[SeqRead]]):
        """
        Save haplotype-specific scaffold files.
        
        Args:
            scaffolds_by_haplotype: Dict mapping haplotype to scaffold lists
        """
        for haplotype, scaffolds in scaffolds_by_haplotype.items():
            if not scaffolds:
                continue
            
            filename = f"scaffolds_{haplotype}.fasta"
            output_path = self.output_dir / filename
            
            self._save_reads(scaffolds, output_path)
            self.logger.info(f"  Saved {len(scaffolds)} {haplotype} scaffolds to {filename}")
    
    def _traverse_with_hic_priority(
        self,
        graph,
        start: int,
        visited: set,
        phasing_info: Optional[Any] = None
    ) -> Tuple[List[Union[int, str, Tuple[str, int, float]]], List[float]]:
        """
        Traverse graph from start node, prioritizing Hi-C edges.
        
        Args:
            graph: DBGGraph to traverse
            start: Starting node ID
            visited: Set of already visited nodes
            phasing_info: Optional phasing information
        
        Returns:
            Tuple of:
            - List of node IDs and ('gap', gap_size, score) tuples representing the path
            - List of Hi-C junction confidence scores
        """
        path = [start]
        visited.add(start)
        current = start
        junction_scores = []  # Track Hi-C junction quality
        
        max_iterations = 10000  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Find edges from current node
            hic_edges = []      # (target, score, edge_obj)
            sequence_edges = [] # (target, edge_obj)
            
            for edge in graph.edges:
                # Check if edge starts from current node
                if not hasattr(edge, 'source') or edge.source != current:
                    continue
                
                target = edge.target
                
                # Skip if target already visited
                if target in visited or target not in graph.nodes:
                    continue
                
                # Check phasing compatibility
                if phasing_info and not self._phasing_compatible(current, target, phasing_info):
                    continue
                
                # Categorize edge by type
                if hasattr(edge, 'edge_type') and edge.edge_type == 'hic':
                    # Hi-C edge
                    score = edge.proximity_score if hasattr(edge, 'proximity_score') else 0.5
                    hic_edges.append((target, score, edge))
                else:
                    # Sequence edge (sequence or ultra-long)
                    sequence_edges.append((target, edge))
            
            # Decide next node: Priority = sequence edges > Hi-C edges
            if sequence_edges:
                # Follow sequence edge (direct connection)
                next_node = sequence_edges[0][0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            elif hic_edges:
                # Follow Hi-C edge (long-range jump)
                # Sort by score and take best
                hic_edges.sort(key=lambda x: x[1], reverse=True)
                next_node, score, edge = hic_edges[0]
                
                # Estimate gap size for this Hi-C junction
                current_node = graph.nodes[current]
                target_node = graph.nodes[next_node]
                gap_size = self._estimate_gap_size(edge, current_node, target_node)
                
                # Mark with gap before Hi-C jump (include gap size and score)
                path.append(('gap', gap_size, score))
                path.append(next_node)
                
                # Track junction quality
                junction_scores.append(score)
                
                visited.add(next_node)
                current = next_node
            
            else:
                # No more edges to follow
                break
        
        return path, junction_scores
    
    def _build_scaffold_sequence(
        self,
        graph,
        path: List[Union[int, Tuple[str, int, float]]],
        default_gap_size: int = 100
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build scaffold sequence from node path with variable N-gaps at Hi-C junctions.
        
        Args:
            graph: DBGGraph
            path: List of node IDs and ('gap', gap_size, score) tuples
            default_gap_size: Default gap size for old-style 'gap' markers
        
        Returns:
            Tuple of:
            - scaffold_sequence: Full sequence with N-gaps
            - junction_info: List of dicts with gap position, size, and confidence
        """
        parts = []
        hic_junctions = []
        current_pos = 0
        
        for element in path:
            if isinstance(element, tuple) and element[0] == 'gap':
                # New format: ('gap', gap_size, score)
                gap_size = element[1]
                score = element[2]
                
                # Insert N-gap
                gap = 'N' * gap_size
                parts.append(gap)
                hic_junctions.append({
                    'position': current_pos,
                    'gap_size': gap_size,
                    'confidence': score
                })
                current_pos += gap_size
            
            elif element == 'gap':
                # Old format: simple 'gap' marker
                gap = 'N' * default_gap_size
                parts.append(gap)
                hic_junctions.append({
                    'position': current_pos,
                    'gap_size': default_gap_size,
                    'confidence': 0.5  # Unknown
                })
                current_pos += default_gap_size
            
            else:
                # Add node sequence
                node = graph.nodes[element]
                seq = node.seq if hasattr(node, 'seq') else node.sequence
                parts.append(seq)
                current_pos += len(seq)
        
        scaffold_seq = ''.join(parts)
        return scaffold_seq, hic_junctions
    
    def _estimate_gap_size(
        self,
        edge: Any,
        node1: Any,
        node2: Any,
        default_gap: int = 100
    ) -> int:
        """
        Estimate gap size for Hi-C junction based on contact frequency and graph distance.
        
        Strategy:
        - High contact frequency (>0.7) → small gap (100 N)
        - Medium contact frequency (0.3-0.7) → medium gap (500 N)
        - Low contact frequency (<0.3) → large gap (1000 N)
        - Adjust based on coverage discontinuity
        
        Args:
            edge: Hi-C edge connecting the nodes
            node1: Source node
            node2: Target node
            default_gap: Default gap size if no information available
        
        Returns:
            Estimated gap size in base pairs
        """
        # Get contact frequency/proximity score
        if hasattr(edge, 'proximity_score'):
            contact_freq = edge.proximity_score
        elif hasattr(edge, 'weight'):
            # Normalize weight to 0-1 range (assume max weight ~100)
            contact_freq = min(edge.weight / 100.0, 1.0)
        else:
            contact_freq = 0.5  # Unknown, assume medium
        
        # Base gap size from contact frequency
        if contact_freq > 0.7:
            base_gap = 100    # Strong contact, likely close
        elif contact_freq > 0.4:
            base_gap = 500    # Medium contact
        elif contact_freq > 0.2:
            base_gap = 1000   # Weak contact
        else:
            base_gap = 5000   # Very weak contact, large gap
        
        # Adjust for coverage discontinuity
        cov1 = node1.coverage if hasattr(node1, 'coverage') else node1.count if hasattr(node1, 'count') else 1.0
        cov2 = node2.coverage if hasattr(node2, 'coverage') else node2.count if hasattr(node2, 'count') else 1.0
        
        if cov1 > 0 and cov2 > 0:
            cov_ratio = max(cov1, cov2) / min(cov1, cov2)
            if cov_ratio > 2.0:
                # Large coverage difference suggests a real gap
                base_gap = int(base_gap * 1.5)
        
        # Cap at reasonable maximum
        return min(base_gap, 10000)
    
    def _calculate_scaffold_confidence(
        self,
        path: List[Union[int, str]],
        graph: Any,
        hic_junction_scores: List[float]
    ) -> float:
        """
        Calculate confidence score for a scaffold based on Hi-C junction quality.
        
        Args:
            path: Scaffold path (nodes and gaps)
            graph: Assembly graph
            hic_junction_scores: Contact scores at each Hi-C junction
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not hic_junction_scores:
            # No Hi-C junctions, just contigs stitched together
            return 1.0
        
        # Average junction quality
        avg_score = sum(hic_junction_scores) / len(hic_junction_scores)
        
        # Penalty for many junctions (more junctions = more uncertainty)
        num_junctions = len(hic_junction_scores)
        junction_penalty = 1.0 - (num_junctions * 0.05)  # 5% penalty per junction
        junction_penalty = max(junction_penalty, 0.5)  # Cap at 50% penalty
        
        # Combined confidence
        confidence = avg_score * junction_penalty
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_node_haplotype(
        self,
        node_id: int,
        phasing_info: Optional[Any]
    ) -> Optional[int]:
        """
        Get haplotype assignment for a node.
        
        Args:
            node_id: Node identifier
            phasing_info: Phasing information from HaplotypeDetangler
        
        Returns:
            0 (hapA), 1 (hapB), or None (unphased/ambiguous)
        """
        if phasing_info is None:
            return None
        
        # PhasingResult format: node_assignments dict
        if hasattr(phasing_info, 'node_assignments'):
            assignments = phasing_info.node_assignments
        elif isinstance(phasing_info, dict):
            assignments = phasing_info
        else:
            return None
        
        phase = assignments.get(node_id)
        
        # Return None for unassigned/ambiguous (-1, 'unassigned', etc.)
        if phase in [None, -1, 'unassigned']:
            return None
        
        return phase
    
    def _phasing_compatible(
        self,
        node1: int,
        node2: int,
        phasing_info: Optional[Any]
    ) -> bool:
        """
        Check if two nodes are phasing-compatible (same haplotype or ambiguous).
        
        Args:
            node1: First node ID
            node2: Second node ID
            phasing_info: Phasing information (from HaplotypeDetangler)
        
        Returns:
            True if nodes can be in same scaffold, False otherwise
        """
        if phasing_info is None:
            return True  # No phasing info, allow all connections
        
        phase1 = self._get_node_haplotype(node1, phasing_info)
        phase2 = self._get_node_haplotype(node2, phasing_info)
        
        # If either is unphased, allow
        if phase1 is None or phase2 is None:
            return True
        
        # Otherwise, must match
        return phase1 == phase2
    
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
            self.logger.warning(
                "Polishing was requested but is not yet implemented. "
                "Contigs will proceed unpolished."
            )
        
        # Gap filling (if enabled)
        if self.config['finishing']['gap_filling']['enabled']:
            self.logger.warning(
                "Gap filling was requested but is not yet implemented. "
                "Gaps will remain unfilled in the final assembly."
            )
        
        # Apply minimum contig length filter
        min_len = self.config.get('runtime', {}).get('min_contig_length', 0)
        if min_len > 0:
            before_count = len(contigs)
            contigs = [c for c in contigs if len(c.sequence) >= min_len]
            removed = before_count - len(contigs)
            if removed > 0:
                self.logger.info(f"  Filtered {removed} contigs below {min_len}bp")
        
        # Save final assembly
        final_path = self.output_dir / "final_assembly.fasta"
        self._save_reads(contigs, final_path)
        
        total_bp = sum(len(c.sequence) for c in contigs)
        self.logger.info(f"✓ Finishing complete")
        self.logger.info(f"  Final contigs: {len(contigs)} ({total_bp:,} bp)")
        self.logger.info(f"  Final assembly: {final_path}")
    
    def _step_misassembly_report(self):
        """
        Generate misassembly report from PathWeaver's detection flags.
        
        Writes TSV and BED files to the output directory for downstream
        analysis and genome-browser visualization.
        """
        self.logger.info("Generating misassembly report...")
        
        path_weaver = self.state.get('path_weaver')
        if path_weaver is None:
            self.logger.warning("No PathWeaver instance available — "
                                "misassembly detection requires the assemble step to run first. Skipping.")
            return
        
        # Check if misassembly detector is wired up
        if not hasattr(path_weaver, 'misassembly_detector') or path_weaver.misassembly_detector is None:
            self.logger.info("Misassembly detector not available in PathWeaver — skipping report")
            return
        
        # Get configuration
        misassembly_config = self.config.get('misassembly_report', {})
        min_confidence = misassembly_config.get('min_confidence', 'MEDIUM')
        output_formats = misassembly_config.get('formats', ['tsv', 'bed'])
        
        # Adjust detector confidence threshold if configured
        from ..assembly_utils.misassembly_detector import ConfidenceLevel
        conf_map = {
            'HIGH': ConfidenceLevel.HIGH,
            'MEDIUM': ConfidenceLevel.MEDIUM,
            'LOW': ConfidenceLevel.LOW,
        }
        if min_confidence.upper() in conf_map:
            path_weaver.misassembly_detector.min_confidence = conf_map[min_confidence.upper()]
        
        # Count total flags across all contigs
        total_flags = sum(
            len(flags) for flags in path_weaver.misassembly_detector.detected_flags.values()
        )
        
        if total_flags == 0:
            self.logger.info("✓ No misassemblies detected — assembly looks clean")
            # Still write empty files so downstream tools don't fail on missing files
        
        # Write reports in each requested format
        files_written = []
        
        if 'tsv' in output_formats:
            tsv_path = self.output_dir / "misassembly_report.tsv"
            try:
                report = path_weaver.get_misassembly_report(output_format='tsv')
                with open(tsv_path, 'w') as f:
                    f.write(report)
                files_written.append(tsv_path)
            except Exception as e:
                self.logger.warning(f"Failed to write TSV report: {e}")
        
        if 'bed' in output_formats:
            bed_path = self.output_dir / "misassembly_report.bed"
            try:
                report = path_weaver.get_misassembly_report(output_format='bed')
                # Add BED track header for genome browser
                header = 'track name="Misassemblies" description="StrandWeaver putative misassembly regions" itemRgb=On\n'
                with open(bed_path, 'w') as f:
                    f.write(header)
                    f.write(report)
                files_written.append(bed_path)
            except Exception as e:
                self.logger.warning(f"Failed to write BED report: {e}")
        
        if 'json' in output_formats:
            json_path = self.output_dir / "misassembly_report.json"
            try:
                report = path_weaver.get_misassembly_report(output_format='json')
                with open(json_path, 'w') as f:
                    f.write(report)
                files_written.append(json_path)
            except Exception as e:
                self.logger.warning(f"Failed to write JSON report: {e}")
        
        # Log summary
        if total_flags > 0:
            # Break down by confidence
            high = medium = low = 0
            for flags in path_weaver.misassembly_detector.detected_flags.values():
                for flag in flags:
                    if flag.confidence.value == 'HIGH':
                        high += 1
                    elif flag.confidence.value == 'MEDIUM':
                        medium += 1
                    else:
                        low += 1
            
            self.logger.info(f"✓ Misassembly report complete")
            self.logger.info(f"  Total flags: {total_flags} "
                             f"(HIGH={high}, MEDIUM={medium}, LOW={low})")
            contigs_flagged = len(path_weaver.misassembly_detector.detected_flags)
            self.logger.info(f"  Contigs with flags: {contigs_flagged}")
        else:
            self.logger.info(f"✓ Misassembly report complete (no flags)")
        
        for p in files_written:
            self.logger.info(f"  Output: {p}")
    
    def _step_classify_chromosomes(self):
        """
        Classify unconnected scaffolds to identify potential microchromosomes.
        
        Uses multi-tier classification:
        - Tier 1: Fast pre-filtering (length, coverage, GC, connectivity, telomere detection)
        - Tier 2: Gene content analysis (BLAST homology search)
        - Tier 3: Advanced features (Hi-C patterns, synteny) - optional
        """
        # Check if enabled
        if not self.config.get('chromosome_classification', {}).get('enabled', False):
            self.logger.info("Chromosome classification disabled, skipping")
            return
        
        self.logger.info("Classifying potential chromosomes and microchromosomes...")
        
        from ..assembly_utils.chromosome_classifier import ChromosomeClassifier
        
        # Get configuration
        chrom_config = self.config.get('chromosome_classification', {})
        advanced = chrom_config.get('advanced', False)
        
        # ====================================================================
        # Pre-flight check: verify external tools are available
        # ====================================================================
        gene_method = chrom_config.get('gene_detection_method', 'orf')
        tool_checks = {
            'blast': ('blastx', 'Install NCBI BLAST+: conda install -c bioconda blast'),
            'augustus': ('augustus', 'Install Augustus: conda install -c bioconda augustus'),
            'busco': ('busco', 'Install BUSCO: conda install -c bioconda busco'),
        }
        
        if gene_method in tool_checks:
            tool_cmd, install_hint = tool_checks[gene_method]
            import shutil
            if shutil.which(tool_cmd) is None:
                self.logger.error(
                    f"Gene detection method '{gene_method}' requires '{tool_cmd}' "
                    f"but it was not found on $PATH.\n"
                    f"  → {install_hint}\n"
                    f"  → Or use --gene-detection-method orf (no external tools needed)"
                )
                raise RuntimeError(
                    f"Required tool '{tool_cmd}' not found for "
                    f"--gene-detection-method {gene_method}. "
                    f"Install it or switch to --gene-detection-method orf."
                )
        
        # Load scaffolds (prefer scaffolds > final_assembly > contigs)
        scaffolds_path = self.output_dir / "scaffolds.fasta"
        if not scaffolds_path.exists():
            scaffolds_path = self.output_dir / "final_assembly.fasta"
        if not scaffolds_path.exists():
            scaffolds_path = self.output_dir / "contigs.fasta"
        if not scaffolds_path.exists():
            self.logger.warning("No scaffolds, final assembly, or contigs found — skipping classification")
            return
        
        self.logger.info(f"  Loading scaffolds from {scaffolds_path.name}...")
        scaffolds = list(read_fasta(scaffolds_path))
        self.logger.info(f"  Loaded {len(scaffolds)} scaffolds")
        
        # Ensure gene_detection_method is set (default to 'orf' for no external deps)
        if 'gene_detection_method' not in chrom_config:
            chrom_config['gene_detection_method'] = 'orf'
        
        self.logger.info(f"  Gene detection method: {chrom_config['gene_detection_method']}")
        if chrom_config['gene_detection_method'] == 'orf':
            self.logger.info("  (Built-in ORF finder — no external tools required)")
        
        # Initialize classifier
        classifier = ChromosomeClassifier(config=chrom_config, advanced=advanced)
        
        # Get graph and Hi-C data if available
        graph = self.state.get('graph')
        contact_map = self.state.get('hic_contact_map')
        
        # Run classification
        results = classifier.classify_scaffolds(
            scaffolds=scaffolds,
            graph=graph,
            contact_map=contact_map
        )
        
        # Export results
        output_format = chrom_config.get('output_format', 'json')
        output_path = self.output_dir / f"chromosome_classification.{output_format}"
        
        classifier.export_results(results, output_path, format=output_format)
        
        # Annotate graph for BandageNG if requested
        if chrom_config.get('annotate_graph', True) and graph:
            self._annotate_graph_with_classifications(graph, results)
        
        # Summary statistics
        high_conf = sum(1 for r in results if r.classification == 'HIGH_CONFIDENCE_CHROMOSOME')
        likely = sum(1 for r in results if r.classification == 'LIKELY_CHROMOSOME')
        possible = sum(1 for r in results if r.classification == 'POSSIBLE_CHROMOSOME')
        
        self.logger.info(f"✓ Chromosome classification complete")
        self.logger.info(f"  Results: {output_path}")
        self.logger.info(f"  High confidence: {high_conf}")
        self.logger.info(f"  Likely: {likely}")
        self.logger.info(f"  Possible: {possible}")
    
    def _annotate_graph_with_classifications(self, graph, classifications: List):
        """
        Add chromosome classifications as node annotations in graph.
        
        This enables visualization in BandageNG with color-coded nodes.
        """
        # Create classification map
        class_map = {c.scaffold_id: c for c in classifications}
        
        # Add annotations to nodes
        for node_id, node in graph.nodes.items():
            # Try to match node to classification
            scaffold_name = f"scaffold_{node_id}"
            if scaffold_name in class_map:
                c = class_map[scaffold_name]
                
                # Add metadata
                if not hasattr(node, 'metadata'):
                    node.metadata = {}
                
                node.metadata['chromosome_class'] = c.classification
                node.metadata['chromosome_prob'] = c.probability
                node.metadata['is_chromosome'] = 'CHROMOSOME' in c.classification
        
        self.logger.info("  Graph annotated with chromosome classifications")
    
    # ========================================================================
    # Checkpoint and utility methods
    # ========================================================================
    
    def _create_checkpoint(self, step: str):
        """Create checkpoint after step completion."""
        checkpoint_path = self.checkpoint_dir / f"{step}.checkpoint"
        
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'completed_steps': list(self.state['completed_steps']),
            'state_files': {
                'reads': str(self.output_dir / "corrected_reads.fastq") if step in ['correct', 'assemble', 'finish'] else None,
                'contigs': str(self.output_dir / "contigs.fasta") if step in ['assemble', 'finish'] else None,
            },
            # Persist serializable state so resume can restore it
            'pipeline_state': {
                'read_files': self.state.get('read_files'),
                'technologies': self.state.get('technologies'),
                'kmer_prediction': self.state.get('kmer_prediction'),
                'corrected_files': self.state.get('corrected_files', {}),
            },
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Checkpoint created: {checkpoint_path}")
    
    def _find_last_checkpoint(self) -> str:
        """Find the last completed checkpoint and restore pipeline state."""
        checkpoints = sorted(self.checkpoint_dir.glob("*.checkpoint"))
        
        if not checkpoints:
            self.logger.warning("No checkpoints found, starting from beginning")
            return self.steps[0]
        
        # Find most recent checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            checkpoint_data = json.load(f)
        
        completed_step = checkpoint_data['step']
        
        # Restore pipeline state from checkpoint
        if 'pipeline_state' in checkpoint_data:
            saved = checkpoint_data['pipeline_state']
            self.state['completed_steps'] = checkpoint_data.get('completed_steps', [])
            self.state['read_files'] = saved.get('read_files', self.state['read_files'])
            self.state['technologies'] = saved.get('technologies', self.state['technologies'])
            self.state['kmer_prediction'] = saved.get('kmer_prediction')
            self.state['corrected_files'] = saved.get('corrected_files', {})
            self.logger.info(f"Restored pipeline state from checkpoint ({len(self.state['completed_steps'])} steps completed)")
        
        # Resume from next step after completed
        if completed_step in self.steps:
            step_index = self.steps.index(completed_step)
            if step_index + 1 < len(self.steps):
                return self.steps[step_index + 1]
        
        # If last step was completed, start from beginning
        return self.steps[0]
    
    def _read_file_streaming(self, reads_path: Path):
        """Stream reads from file without loading all into memory."""
        # Strip .gz/.gzip suffix to inspect the real format extension
        # (read_fastq and read_fasta handle gzip decompression natively)
        effective_path = reads_path
        if reads_path.suffix.lower() in ['.gz', '.gzip']:
            effective_path = Path(reads_path.stem)  # e.g. reads.fastq.gz → reads.fastq

        if effective_path.suffix in ['.fq', '.fastq']:
            yield from read_fastq(reads_path)
        elif effective_path.suffix in ['.fa', '.fasta']:
            yield from read_fasta(reads_path)
        else:
            # Try FASTQ first, fall back to FASTA
            try:
                yield from read_fastq(reads_path)
            except Exception:
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
    
    def _calculate_coverage_from_reads(self, graph: Union[DBGGraph, StringGraph], 
                                       reads: List[SeqRead]) -> Dict[int, float]:
        """
        Calculate per-node coverage from read mappings.
        
        Maps reads to graph nodes using k-mer matching and calculates coverage depth.
        
        Args:
            graph: DBGGraph or StringGraph object
            reads: List of SeqRead objects
        
        Returns:
            Dict mapping node_id to coverage depth (float)
        """
        coverage = defaultdict(float)
        
        if not reads:
            return coverage
        
        k = 21  # Use small k-mer size for mapping
        
        # Sample reads if too many (for performance)
        sample_size = min(10000, len(reads))
        sampled_reads = reads[:sample_size] if len(reads) > sample_size else reads
        
        for read in sampled_reads:
            # Extract k-mers from read
            seq = read.sequence
            if len(seq) < k:
                continue
            
            read_kmers = set(seq[i:i+k] for i in range(len(seq) - k + 1))
            
            # Find matching nodes
            for node_id, node in graph.nodes.items():
                node_seq = node.seq if hasattr(node, 'seq') else ""
                if not node_seq or len(node_seq) < k:
                    continue
                
                # Count k-mer matches
                node_kmers = set(node_seq[i:i+k] for i in range(len(node_seq) - k + 1))
                matches = len(read_kmers & node_kmers)
                
                if matches > 0:
                    # Normalize by node length (k-mer count)
                    node_kmer_count = max(1, len(node_seq) - k + 1)
                    coverage[node_id] += matches / node_kmer_count
        
        # Scale up if we sampled
        if len(reads) > sample_size:
            scale_factor = len(reads) / sample_size
            coverage = {nid: cov * scale_factor for nid, cov in coverage.items()}
        
        return dict(coverage)
    
    def _calculate_hic_support_per_node(self, graph: Union[DBGGraph, StringGraph]) -> Dict[int, float]:
        """
        Calculate per-node Hi-C support from Hi-C edges.
        
        Counts Hi-C contact edges connected to each node.  Used by
        ``export_for_bandageng`` which expects ``dict[int, float]``.
        
        See also ``_calculate_hic_coverage`` which returns a single float
        percentage.
        
        Args:
            graph: DBGGraph or StringGraph object with Hi-C edges
        
        Returns:
            Dict mapping node_id to Hi-C contact count
        """
        hic_support = defaultdict(int)
        
        for edge_id, edge in graph.edges.items():
            if hasattr(edge, 'edge_type') and edge.edge_type == 'hic':
                # Get source and target node IDs
                source_id = edge.source if hasattr(edge, 'source') else edge.from_id
                target_id = edge.target if hasattr(edge, 'target') else edge.to_id
                
                hic_support[source_id] += 1
                hic_support[target_id] += 1
        
        return dict(hic_support)
    
    def _extract_edge_scores(self, graph: Union[DBGGraph, StringGraph]) -> Dict[Tuple[int, int], float]:
        """
        Extract edge quality scores (0-1) for BandageNG visualization.
        
        Gets EdgeWarden quality scores or calculates from edge properties.
        
        Args:
            graph: DBGGraph or StringGraph object
        
        Returns:
            Dict mapping (source_id, target_id) to quality score (0.0-1.0)
        """
        edge_scores = {}
        
        # Iterate over edge objects from the edges dictionary
        for edge_id, edge in graph.edges.items():
            # Get source and target IDs
            source_id = edge.source if hasattr(edge, 'source') else edge.from_id
            target_id = edge.target if hasattr(edge, 'target') else edge.to_id
            
            # Get quality score from edge attributes
            if hasattr(edge, 'quality_score'):
                # EdgeWarden score (already 0-1)
                score = edge.quality_score
            elif hasattr(edge, 'confidence'):
                # Use confidence as score
                score = edge.confidence
            elif hasattr(edge, 'coverage'):
                # Calculate score from coverage (normalize to 0-1)
                # Assume coverage 1-50 maps to 0.0-1.0
                score = min(1.0, edge.coverage / 50.0)
            else:
                # Default score
                score = 0.5
            
            # Clamp to 0-1 range
            score = max(0.0, min(1.0, score))
            
            edge_scores[(source_id, target_id)] = score
        
        return edge_scores
    
    def _calculate_n50(self, lengths: List[int]) -> int:
        """
        Calculate N50 from list of sequence lengths.
        
        N50 is the length L such that 50% of all bases are in sequences of length >= L.
        
        Args:
            lengths: List of sequence lengths
        
        Returns:
            N50 value (int)
        """
        if not lengths:
            return 0
        
        sorted_lengths = sorted(lengths, reverse=True)
        total = sum(sorted_lengths)
        cumsum = 0
        
        for length in sorted_lengths:
            cumsum += length
            if cumsum >= total / 2:
                return length
        
        return 0
    
    def _save_graph(self, graph, output_path: Path):
        """Save assembly graph to GFA format."""
        try:
            export_graph_to_gfa(graph, output_path)
            self.logger.info(f"Assembly graph saved to {output_path}")
        except Exception as e:
            self.logger.warning(f"Graph export failed: {e}")
    
    def _load_ai_models(self) -> Dict[str, Any]:
        """
        Load AI models from disk (lazy, one-shot).

        Resolution order for each model:
          1. Explicit ``model_path`` in the per-model config block
          2. ``<model_dir>/<sub_folder>`` (central model directory)
          3. Package defaults inside the source tree

        If a model file is not found the slot is set to ``None`` and the
        pipeline falls back to classical heuristics for that component.
        """
        if self._ai_models:
            return self._ai_models

        model_dir = self._resolve_model_dir()
        self.logger.info(f"Loading AI models (model_dir={model_dir or 'auto'})...")
        loaded, skipped = 0, 0

        # ------------------------------------------------------------------
        # 1. Correction: adaptive_kmer (KWeaverPredictor pickle files)
        # ------------------------------------------------------------------
        if self.config['ai']['correction']['adaptive_kmer']['enabled']:
            model = self._try_load_pickle_model(
                explicit_path=self.config['ai']['correction']['adaptive_kmer'].get('model_path'),
                model_dir=model_dir,
                sub_folder='kweaver',
                model_name='adaptive_kmer',
            )
            self._ai_models['adaptive_kmer'] = model
            if model is not None:
                loaded += 1
                self.logger.info("  ✓ AdaptiveKmerAI loaded")
            else:
                skipped += 1
                self.logger.warning("  ⚠ AdaptiveKmerAI not found — using classical k-mer selection")

        # ------------------------------------------------------------------
        # 2. Correction: base_error_classifier (pickle)
        # ------------------------------------------------------------------
        if self.config['ai']['correction']['base_error_classifier']['enabled']:
            model = self._try_load_pickle_model(
                explicit_path=self.config['ai']['correction']['base_error_classifier'].get('model_path'),
                model_dir=model_dir,
                sub_folder='error_classifier',
                model_name='base_error_classifier',
            )
            self._ai_models['base_error_classifier'] = model
            if model is not None:
                loaded += 1
                self.logger.info("  ✓ BaseErrorClassifierAI loaded")
            else:
                skipped += 1
                self.logger.warning("  ⚠ BaseErrorClassifierAI not found — using classical correction")

        # ------------------------------------------------------------------
        # 3. Assembly: edge_ai (EdgeWarden per-technology pkl files)
        #    Stored as model_dir path — EdgeWarden loads its own files.
        # ------------------------------------------------------------------
        if self.config['ai']['assembly']['edge_ai']['enabled']:
            ew_dir = self._resolve_sub_model_path(
                explicit_path=self.config['ai']['assembly']['edge_ai'].get('model_path'),
                model_dir=model_dir,
                sub_folder='edgewarden',
            )
            self._ai_models['edge_ai'] = str(ew_dir) if ew_dir and ew_dir.is_dir() else None
            if self._ai_models['edge_ai']:
                loaded += 1
                self.logger.info(f"  ✓ EdgeWardenAI models directory: {ew_dir}")
            else:
                skipped += 1
                self.logger.warning("  ⚠ EdgeWardenAI models not found — using rule-based filtering")

        # ------------------------------------------------------------------
        # 4. Assembly: path_gnn (PyTorch checkpoint)
        # ------------------------------------------------------------------
        if self.config['ai']['assembly']['path_gnn']['enabled']:
            gnn_path = self._resolve_sub_model_path(
                explicit_path=self.config['ai']['assembly']['path_gnn'].get('model_path'),
                model_dir=model_dir,
                sub_folder='pathgnn',
                file_name='pathgnn_model.pt',
            )
            self._ai_models['path_gnn'] = str(gnn_path) if gnn_path and gnn_path.exists() else None
            if self._ai_models['path_gnn']:
                loaded += 1
                self.logger.info(f"  ✓ PathGNN checkpoint: {gnn_path}")
            else:
                skipped += 1
                self.logger.warning("  ⚠ PathGNN model not found — using heuristic path scoring")

        # ------------------------------------------------------------------
        # 5. Assembly: diploid_ai (pickle or torch)
        # ------------------------------------------------------------------
        if self.config['ai']['assembly']['diploid_ai']['enabled']:
            dip_path = self._resolve_sub_model_path(
                explicit_path=self.config['ai']['assembly']['diploid_ai'].get('model_path'),
                model_dir=model_dir,
                sub_folder='diploid',
                file_name='diploid_model.pkl',
            )
            self._ai_models['diploid_ai'] = self._try_load_pickle_file(dip_path) if dip_path else None
            if self._ai_models['diploid_ai']:
                loaded += 1
                self.logger.info("  ✓ DiploidAI loaded")
            else:
                skipped += 1
                self.logger.warning("  ⚠ DiploidAI not found — using heuristic phasing")

        # ------------------------------------------------------------------
        # 6. Assembly: ul_routing_ai (pickle or torch)
        # ------------------------------------------------------------------
        if self.config['ai']['assembly']['ul_routing_ai']['enabled']:
            ul_path = self._resolve_sub_model_path(
                explicit_path=self.config['ai']['assembly']['ul_routing_ai'].get('model_path'),
                model_dir=model_dir,
                sub_folder='ul_routing',
                file_name='ul_routing_model.pkl',
            )
            self._ai_models['ul_routing_ai'] = self._try_load_pickle_file(ul_path) if ul_path else None
            if self._ai_models['ul_routing_ai']:
                loaded += 1
                self.logger.info("  ✓ ULRoutingAI loaded")
            else:
                skipped += 1
                self.logger.warning("  ⚠ ULRoutingAI not found — using heuristic UL routing")

        # ------------------------------------------------------------------
        # 7. Assembly: sv_ai (pickle or torch)
        # ------------------------------------------------------------------
        if self.config['ai']['assembly']['sv_ai']['enabled']:
            sv_path = self._resolve_sub_model_path(
                explicit_path=self.config['ai']['assembly']['sv_ai'].get('model_path'),
                model_dir=model_dir,
                sub_folder='sv_detector',
                file_name='sv_detector_model.pkl',
            )
            self._ai_models['sv_ai'] = self._try_load_pickle_file(sv_path) if sv_path else None
            if self._ai_models['sv_ai']:
                loaded += 1
                self.logger.info("  ✓ SVDetectorAI loaded")
            else:
                skipped += 1
                self.logger.warning("  ⚠ SVDetectorAI not found — using heuristic SV detection")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        total = loaded + skipped
        self.logger.info(f"AI model loading complete: {loaded}/{total} loaded, "
                         f"{skipped}/{total} using classical fallback")
        if skipped > 0 and loaded == 0:
            self.logger.warning(
                "No trained AI models found. The pipeline will run in fully "
                "classical mode. Train models with `strandweaver train` and "
                "set --model-dir to the output directory."
            )

        return self._ai_models

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> Optional[Path]:
        """Return the central model directory from config, or None."""
        raw = self.config['ai'].get('model_dir')
        if raw:
            p = Path(raw)
            if p.is_dir():
                return p
            self.logger.warning(f"Configured model_dir does not exist: {p}")
        return None

    def _resolve_sub_model_path(
        self,
        explicit_path: Optional[str],
        model_dir: Optional[Path],
        sub_folder: str,
        file_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Resolve a model file / directory path.

        Priority: explicit_path > model_dir/sub_folder[/file_name]
        """
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                return p
        if model_dir:
            base = model_dir / sub_folder
            if file_name:
                candidate = base / file_name
                if candidate.exists():
                    return candidate
            elif base.is_dir():
                return base
        return None

    def _try_load_pickle_model(
        self,
        explicit_path: Optional[str],
        model_dir: Optional[Path],
        sub_folder: str,
        model_name: str,
    ) -> Optional[Any]:
        """
        Try to load a pickle-serialised model.

        Looks for ``<model_name>.pkl`` inside the resolved directory.
        """
        resolved = self._resolve_sub_model_path(
            explicit_path=explicit_path,
            model_dir=model_dir,
            sub_folder=sub_folder,
        )
        if resolved is None:
            return None

        # If resolved is a file, load it directly
        if resolved.is_file():
            return self._try_load_pickle_file(resolved)

        # If resolved is a directory, look for the canonical file
        candidate = resolved / f"{model_name}.pkl"
        if candidate.exists():
            return self._try_load_pickle_file(candidate)

        # Also check for any .pkl in the directory (single-model folders)
        pkl_files = list(resolved.glob("*.pkl"))
        if len(pkl_files) == 1:
            return self._try_load_pickle_file(pkl_files[0])

        return None

    @staticmethod
    def _try_load_pickle_file(path: Optional[Path]) -> Optional[Any]:
        """Safely load a single pickle file, returning None on any error."""
        if path is None or not Path(path).exists():
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load model from {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Model saving helpers (for use after training)
    # ------------------------------------------------------------------

    def save_all_models(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save all currently loaded AI models to *output_dir*.

        Produces the canonical layout expected by ``_load_ai_models()``:

            output_dir/
              kweaver/adaptive_kmer.pkl
              error_classifier/base_error_classifier.pkl
              edgewarden/edgewarden_<tech>.pkl  (+ scaler_<tech>.pkl)
              pathgnn/pathgnn_model.pt
              diploid/diploid_model.pkl
              ul_routing/ul_routing_model.pkl
              sv_detector/sv_detector_model.pkl

        Returns:
            Dict mapping model name → saved path (or skip reason).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report: Dict[str, str] = {}

        for name, sub, ext in [
            ('adaptive_kmer',        'kweaver',          'pkl'),
            ('base_error_classifier','error_classifier',  'pkl'),
            ('diploid_ai',           'diploid',           'pkl'),
            ('ul_routing_ai',        'ul_routing',        'pkl'),
            ('sv_ai',                'sv_detector',       'pkl'),
        ]:
            model = self._ai_models.get(name)
            if model is None:
                report[name] = 'skipped (not loaded)'
                continue
            dest = out / sub
            dest.mkdir(parents=True, exist_ok=True)
            fpath = dest / f"{name}.{ext}"
            try:
                with open(fpath, 'wb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                report[name] = str(fpath)
                self.logger.info(f"  Saved {name} → {fpath}")
            except Exception as e:
                report[name] = f'FAILED: {e}'
                self.logger.error(f"  Failed to save {name}: {e}")

        # EdgeWarden — delegate to its own save_models()
        ew_dir_str = self._ai_models.get('edge_ai')
        if ew_dir_str:
            report['edge_ai'] = f'models dir: {ew_dir_str} (save via EdgeWarden.save_models())'
        else:
            report['edge_ai'] = 'skipped (not loaded)'

        # PathGNN — torch checkpoint
        gnn_path = self._ai_models.get('path_gnn')
        if gnn_path:
            report['path_gnn'] = f'checkpoint: {gnn_path} (save via torch.save())'
        else:
            report['path_gnn'] = 'skipped (not loaded)'

        self.logger.info(f"Model save report: {json.dumps(report, indent=2)}")
        return report

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
