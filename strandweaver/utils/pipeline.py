"""
Pipeline orchestrator for StrandWeaver.

Coordinates the execution of all assembly steps with checkpoint support.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json
import pickle
from datetime import datetime

from ..io import SeqRead, read_fastq, write_fastq, read_fasta, write_fasta
from ..read_correction import ErrorProfiler
from ..read_correction import (
    ONTCorrector,
    PacBioCorrector,
    IlluminaCorrector,
    AncientDNACorrector,
)
from .pipeline_orchestrator import AssemblyOrchestrator


class PipelineOrchestrator:
    """
    Orchestrate the complete assembly pipeline.
    
    Manages:
    - Step execution order
    - Checkpoint creation and recovery
    - Inter-step file passing
    - Error handling
    - AI model loading
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
        
        # Initialize assembly orchestrator
        orchestrator = AssemblyOrchestrator(
            ml_edge_model=ai_models.get('edge_ai') if use_ai else None,
            ml_path_model=ai_models.get('path_gnn') if use_ai else None,
            ml_diploid_model=ai_models.get('diploid_ai') if use_ai else None,
            ml_ul_model=ai_models.get('ul_routing_ai') if use_ai else None,
            ml_sv_model=ai_models.get('sv_ai') if use_ai else None,
        )
        
        # Run assembly
        self.logger.info(f"  Primary technology: {primary_tech}")
        self.logger.info(f"  AI-powered assembly: {use_ai}")
        
        assembly_result = orchestrator.run_assembly_pipeline(
            corrected_reads=corrected_reads,
            technology=primary_tech,
            hic_data={'r1': self.config['runtime'].get('hic_r1'), 
                     'r2': self.config['runtime'].get('hic_r2')} if self.config['scaffolding']['hic']['enabled'] else None,
            output_dir=self.output_dir
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
        
        self.state['assembly_result'] = assembly_result
        self.state['final_contigs'] = assembly_result.contigs
    
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

