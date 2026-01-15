# StrandWeaver Polishing Integration Proposal
**Date**: December 28, 2025  
**Status**: 📋 Proposal  
**Target**: Complete finishing module for production-ready assemblies

---

## Executive Summary

This proposal outlines a comprehensive polishing integration for StrandWeaver that leverages proven tools (Arrow, Medaka, Racon) with intelligent technology routing, iterative refinement, and quality assessment. The design draws on your T2T-Polish experience while integrating seamlessly with StrandWeaver's existing pipeline architecture.

**Key Goals**:
1. ✅ Technology-specific polishing (PacBio → Arrow, ONT → Medaka, Universal → Racon)
2. ✅ Iterative polishing with convergence detection
3. ✅ Quality assessment (QV calculation, completeness tracking)
4. ✅ Gap filling integration
5. ✅ Flexible configuration with sensible defaults

**Estimated Timeline**: 1-2 weeks

---

## Current State Analysis

### Existing Implementation
```python
def _step_finish(self):
    """Finishing step (polishing, gap filling, etc.)."""
    # ✅ Load contigs/scaffolds
    # 🔴 Polishing - TODO
    # 🔴 Gap filling - TODO
    # 🔴 Claude AI finishing - TODO (unclear purpose)
    # ✅ Save final assembly
```

### Available Infrastructure
- ✅ Read management (read_fastq, read_fasta, SeqRead objects)
- ✅ Technology detection (ReadTechnology enum, auto-detection)
- ✅ Checkpoint/resume system
- ✅ Logging infrastructure
- ✅ Configuration management
- ✅ Thread management
- ✅ Output directory structure

### Integration with Your T2T-Polish Experience

From `APv4.py`, I can see you have:
- ✅ **Merfin integration** for QV calculation and k-mer based polishing evaluation
- ✅ **Merqury integration** for completeness assessment
- ✅ **GenomeScope2** for k-mer coverage estimation
- ✅ **Jellyfish** k-mer counting
- ✅ **Multi-round polishing** with quality tracking
- ✅ **Resume capability** with intermediate checkpoints
- ✅ **Optimized mode** using fitted histograms for peak coverage

**Proposal**: Integrate these proven patterns into StrandWeaver's finishing module.

---

## Architecture Design

### 1. Polishing Module Structure

```
strandweaver/
├── finishing/
│   ├── __init__.py
│   ├── polisher.py              # Main polishing orchestrator
│   ├── arrow_polisher.py        # PacBio Arrow wrapper
│   ├── medaka_polisher.py       # ONT Medaka wrapper
│   ├── racon_polisher.py        # Universal Racon wrapper
│   ├── quality_assessment.py    # QV calculation (Merfin/Merqury)
│   ├── gap_filler.py            # Gap filling (TGS-GapCloser, etc.)
│   └── kmer_analyzer.py         # K-mer based QC
└── utils/
    └── pipeline.py              # Updated _step_finish()
```

### 2. Main Polishing Orchestrator

```python
# File: strandweaver/finishing/polisher.py

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .arrow_polisher import ArrowPolisher
from .medaka_polisher import MedakaPolisher
from .racon_polisher import RaconPolisher
from .quality_assessment import QualityAssessor
from ..io_utils import SeqRead, read_fasta, write_fasta
from ..preprocessing import ReadTechnology


@dataclass
class PolishingResult:
    """Results from a polishing round."""
    iteration: int
    assembly_path: Path
    qv: Optional[float] = None
    completeness: Optional[float] = None
    error_rate: Optional[float] = None
    num_changes: Optional[int] = None
    elapsed_time: float = 0.0
    converged: bool = False


class AssemblyPolisher:
    """
    Main polishing orchestrator that selects and runs appropriate polishing tools.
    
    Features:
    - Technology-specific tool selection (Arrow/Medaka/Racon)
    - Iterative polishing with convergence detection
    - Quality assessment (QV, completeness)
    - Resume capability
    - Detailed logging and progress tracking
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the polishing orchestrator.
        
        Args:
            config: Configuration dictionary (from defaults.yaml + user overrides)
            output_dir: Output directory for polishing results
            logger: Logger instance
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract polishing configuration
        self.polish_config = config.get('finishing', {}).get('polishing', {})
        
        # Initialize polishing tools
        self.arrow = ArrowPolisher(config, logger=self.logger)
        self.medaka = MedakaPolisher(config, logger=self.logger)
        self.racon = RaconPolisher(config, logger=self.logger)
        
        # Quality assessor (Merfin/Merqury)
        self.quality = QualityAssessor(config, logger=self.logger)
        
    def polish_assembly(
        self,
        assembly: Path,
        reads: List[Path],
        technologies: List[ReadTechnology],
        max_iterations: int = 3,
        convergence_threshold: float = 0.01,
        resume: bool = False
    ) -> List[PolishingResult]:
        """
        Polish assembly using iterative refinement.
        
        Algorithm:
        1. Detect primary technology (PacBio, ONT, or mixed)
        2. Select appropriate polishing tool
        3. Iterate until convergence or max iterations:
           a. Run polishing round
           b. Assess quality (QV, completeness)
           c. Check convergence (QV improvement < threshold)
           d. Save checkpoint
        4. Return polishing history
        
        Args:
            assembly: Path to input assembly FASTA
            reads: List of read file paths
            technologies: List of detected read technologies
            max_iterations: Maximum polishing iterations
            convergence_threshold: QV improvement threshold for convergence
            resume: Whether to resume from checkpoint
        
        Returns:
            List of PolishingResult objects (one per iteration)
        """
        self.logger.info("="*70)
        self.logger.info("POLISHING ASSEMBLY")
        self.logger.info("="*70)
        
        # Check if polishing is enabled
        if not self.polish_config.get('enabled', True):
            self.logger.info("Polishing disabled in configuration")
            return []
        
        # Select polishing tool based on technology
        polisher = self._select_polisher(technologies)
        self.logger.info(f"Selected polisher: {polisher.__class__.__name__}")
        
        # Check if tool is available
        if not polisher.check_availability():
            self.logger.warning(f"{polisher.__class__.__name__} not available in PATH")
            
            # Try fallback to Racon
            if polisher != self.racon:
                self.logger.info("Attempting fallback to Racon...")
                polisher = self.racon
                if not polisher.check_availability():
                    self.logger.error("No polishing tools available (Arrow, Medaka, Racon)")
                    return []
            else:
                self.logger.error("No polishing tools available")
                return []
        
        # Iterative polishing
        results = []
        current_assembly = assembly
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"POLISHING ROUND {iteration}/{max_iterations}")
            self.logger.info(f"{'='*70}")
            
            # Check for resume checkpoint
            checkpoint_path = self.output_dir / f"polishing_round_{iteration}.fasta"
            if resume and checkpoint_path.exists():
                self.logger.info(f"Resume: Loading existing round {iteration} from {checkpoint_path}")
                current_assembly = checkpoint_path
                
                # Try to load previous result
                if len(results) >= iteration - 1:
                    continue
            
            # Run polishing
            result = self._polish_iteration(
                polisher=polisher,
                assembly=current_assembly,
                reads=reads,
                iteration=iteration,
                checkpoint_path=checkpoint_path
            )
            results.append(result)
            
            # Update assembly for next round
            current_assembly = result.assembly_path
            
            # Check convergence
            if self._check_convergence(results, convergence_threshold):
                self.logger.info(f"✓ Polishing converged after {iteration} iterations")
                result.converged = True
                break
        
        # Summary
        self._log_polishing_summary(results)
        
        return results
    
    def _select_polisher(self, technologies: List[ReadTechnology]):
        """
        Select appropriate polishing tool based on read technology.
        
        Priority:
        1. PacBio → Arrow (best for PacBio HiFi)
        2. ONT → Medaka (best for ONT)
        3. Mixed/Other → Racon (universal, works with any long reads)
        
        Args:
            technologies: List of detected technologies
        
        Returns:
            Polisher instance
        """
        tech_set = set(technologies)
        
        # PacBio priority
        if ReadTechnology.PACBIO_HIFI in tech_set or ReadTechnology.PACBIO_CLR in tech_set:
            self.logger.info("Detected PacBio reads → Using Arrow polisher")
            return self.arrow
        
        # ONT priority
        if ReadTechnology.ONT in tech_set or ReadTechnology.ONT_ULTRALONG in tech_set:
            self.logger.info("Detected ONT reads → Using Medaka polisher")
            return self.medaka
        
        # Fallback to universal
        self.logger.info("Using universal Racon polisher")
        return self.racon
    
    def _polish_iteration(
        self,
        polisher,
        assembly: Path,
        reads: List[Path],
        iteration: int,
        checkpoint_path: Path
    ) -> PolishingResult:
        """
        Execute a single polishing iteration.
        
        Args:
            polisher: Polisher instance
            assembly: Current assembly path
            reads: Read files
            iteration: Current iteration number
            checkpoint_path: Where to save polished assembly
        
        Returns:
            PolishingResult with quality metrics
        """
        import time
        start_time = time.time()
        
        # Run polishing
        self.logger.info(f"Running {polisher.__class__.__name__}...")
        polished_assembly = polisher.polish(
            assembly=assembly,
            reads=reads,
            output=checkpoint_path
        )
        
        elapsed = time.time() - start_time
        
        # Assess quality
        self.logger.info("Assessing polished assembly quality...")
        qv, completeness = self.quality.assess(
            assembly=polished_assembly,
            reads=reads
        )
        
        # Create result
        result = PolishingResult(
            iteration=iteration,
            assembly_path=polished_assembly,
            qv=qv,
            completeness=completeness,
            elapsed_time=elapsed
        )
        
        self.logger.info(f"Round {iteration} complete:")
        self.logger.info(f"  QV: {qv:.2f}" if qv else "  QV: N/A")
        self.logger.info(f"  Completeness: {completeness:.4f}%" if completeness else "  Completeness: N/A")
        self.logger.info(f"  Time: {elapsed:.1f}s")
        
        return result
    
    def _check_convergence(
        self,
        results: List[PolishingResult],
        threshold: float
    ) -> bool:
        """
        Check if polishing has converged.
        
        Convergence criteria:
        - QV improvement < threshold (e.g., 0.01)
        - OR QV decreased (overfitting)
        
        Args:
            results: List of polishing results
            threshold: Convergence threshold
        
        Returns:
            True if converged
        """
        if len(results) < 2:
            return False
        
        prev_qv = results[-2].qv
        curr_qv = results[-1].qv
        
        # Can't check convergence without QV
        if prev_qv is None or curr_qv is None:
            return False
        
        improvement = curr_qv - prev_qv
        
        # Converged if improvement is minimal
        if abs(improvement) < threshold:
            self.logger.info(f"Convergence: QV improvement {improvement:.4f} < threshold {threshold}")
            return True
        
        # Converged if QV decreased (overfitting)
        if improvement < 0:
            self.logger.warning(f"Convergence: QV decreased by {-improvement:.4f} (possible overfitting)")
            return True
        
        return False
    
    def _log_polishing_summary(self, results: List[PolishingResult]):
        """Log summary of polishing results."""
        if not results:
            return
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("POLISHING SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Rounds completed: {len(results)}")
        
        # Table header
        self.logger.info(f"\n{'Round':<8} {'QV':<10} {'Completeness':<15} {'Time (s)':<10}")
        self.logger.info("-" * 50)
        
        # Table rows
        for result in results:
            qv_str = f"{result.qv:.2f}" if result.qv else "N/A"
            comp_str = f"{result.completeness:.4f}%" if result.completeness else "N/A"
            time_str = f"{result.elapsed_time:.1f}"
            
            marker = " ✓" if result.converged else ""
            self.logger.info(f"{result.iteration:<8} {qv_str:<10} {comp_str:<15} {time_str:<10}{marker}")
        
        # Final metrics
        final = results[-1]
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"FINAL ASSEMBLY: {final.assembly_path}")
        if final.qv:
            self.logger.info(f"Final QV: {final.qv:.2f}")
        if final.completeness:
            self.logger.info(f"Final Completeness: {final.completeness:.4f}%")
        self.logger.info(f"{'='*70}\n")
```

---

## 3. Technology-Specific Polishers

### 3.1 Arrow Polisher (PacBio)

```python
# File: strandweaver/finishing/arrow_polisher.py

from pathlib import Path
from typing import List, Optional
import subprocess
import shutil
import logging


class ArrowPolisher:
    """
    Arrow polishing for PacBio reads.
    
    Arrow is the recommended polisher for PacBio HiFi and CLR reads.
    Requires: pbmm2 (alignment), samtools, gcpp (Arrow consensus caller)
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config.get('finishing', {}).get('polishing', {}).get('arrow', {})
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool paths
        self.pbmm2 = self.config.get('pbmm2_path', 'pbmm2')
        self.gcpp = self.config.get('gcpp_path', 'gcpp')
        self.samtools = self.config.get('samtools_path', 'samtools')
        
        # Parameters
        self.threads = self.config.get('threads', 8)
        self.min_coverage = self.config.get('min_coverage', 5)
        self.algorithm = self.config.get('algorithm', 'arrow')  # arrow or quiver
    
    def check_availability(self) -> bool:
        """Check if Arrow tools are available."""
        for tool in [self.pbmm2, self.gcpp, self.samtools]:
            if shutil.which(tool) is None:
                self.logger.warning(f"Arrow tool not found: {tool}")
                return False
        return True
    
    def polish(
        self,
        assembly: Path,
        reads: List[Path],
        output: Path
    ) -> Path:
        """
        Polish assembly using Arrow.
        
        Steps:
        1. Align PacBio reads to assembly with pbmm2
        2. Sort and index BAM
        3. Run gcpp (Arrow) for consensus calling
        
        Args:
            assembly: Input assembly FASTA
            reads: List of PacBio read files (FASTQ/BAM)
            output: Output polished assembly path
        
        Returns:
            Path to polished assembly
        """
        self.logger.info(f"Arrow polishing: {assembly}")
        
        work_dir = output.parent
        bam_path = work_dir / f"{output.stem}.aligned.bam"
        sorted_bam = work_dir / f"{output.stem}.aligned.sorted.bam"
        
        # Step 1: Align reads
        self._align_reads(assembly, reads, bam_path)
        
        # Step 2: Sort and index BAM
        self._sort_and_index(bam_path, sorted_bam)
        
        # Step 3: Run Arrow consensus
        self._run_gcpp(assembly, sorted_bam, output)
        
        return output
    
    def _align_reads(self, assembly: Path, reads: List[Path], output_bam: Path):
        """Align PacBio reads to assembly using pbmm2."""
        self.logger.info(f"  Aligning {len(reads)} read file(s) with pbmm2...")
        
        # Concatenate multiple read files if needed
        if len(reads) > 1:
            # Use file list mode or concatenate
            read_list = output_bam.parent / "read_files.fofn"
            with open(read_list, 'w') as f:
                for read_file in reads:
                    f.write(f"{read_file}\n")
            input_arg = f"@{read_list}"
        else:
            input_arg = str(reads[0])
        
        cmd = [
            self.pbmm2,
            'align',
            '--preset', 'CCS',  # Use CCS preset for HiFi
            '--sort',
            '-j', str(self.threads),
            str(assembly),
            input_arg,
            str(output_bam)
        ]
        
        self._run_command(cmd, "pbmm2 alignment")
    
    def _sort_and_index(self, bam: Path, sorted_bam: Path):
        """Sort and index BAM file."""
        self.logger.info("  Sorting and indexing BAM...")
        
        # Sort
        cmd_sort = [
            self.samtools, 'sort',
            '-@', str(self.threads),
            '-o', str(sorted_bam),
            str(bam)
        ]
        self._run_command(cmd_sort, "samtools sort")
        
        # Index
        cmd_index = [self.samtools, 'index', str(sorted_bam)]
        self._run_command(cmd_index, "samtools index")
    
    def _run_gcpp(self, reference: Path, bam: Path, output: Path):
        """Run gcpp (Arrow consensus caller)."""
        self.logger.info(f"  Running gcpp ({self.algorithm} algorithm)...")
        
        cmd = [
            self.gcpp,
            '-j', str(self.threads),
            '--algorithm', self.algorithm,
            '-r', str(reference),
            '-o', str(output),
            str(bam)
        ]
        
        self._run_command(cmd, "gcpp")
    
    def _run_command(self, cmd: List[str], description: str):
        """Execute command with error handling."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"{description} completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description} failed:")
            self.logger.error(f"  Command: {' '.join(cmd)}")
            self.logger.error(f"  Error: {e.stderr}")
            raise
```

### 3.2 Medaka Polisher (ONT)

```python
# File: strandweaver/finishing/medaka_polisher.py

from pathlib import Path
from typing import List, Optional
import subprocess
import shutil
import logging


class MedakaPolisher:
    """
    Medaka polishing for ONT reads.
    
    Medaka is the recommended polisher for Oxford Nanopore reads.
    Requires: medaka (includes minimap2 alignment)
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config.get('finishing', {}).get('polishing', {}).get('medaka', {})
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool path
        self.medaka = self.config.get('medaka_path', 'medaka_consensus')
        
        # Parameters
        self.threads = self.config.get('threads', 8)
        self.model = self.config.get('model', 'r941_min_hac_g507')  # Auto-detect if possible
        self.batch_size = self.config.get('batch_size', 100)
    
    def check_availability(self) -> bool:
        """Check if Medaka is available."""
        if shutil.which(self.medaka) is None:
            self.logger.warning(f"Medaka not found: {self.medaka}")
            return False
        return True
    
    def polish(
        self,
        assembly: Path,
        reads: List[Path],
        output: Path
    ) -> Path:
        """
        Polish assembly using Medaka.
        
        Medaka handles alignment internally, so this is a single command.
        
        Args:
            assembly: Input assembly FASTA
            reads: List of ONT read files (FASTQ)
            output: Output polished assembly path
        
        Returns:
            Path to polished assembly
        """
        self.logger.info(f"Medaka polishing: {assembly}")
        self.logger.info(f"  Model: {self.model}")
        
        work_dir = output.parent / f"{output.stem}_medaka"
        work_dir.mkdir(exist_ok=True)
        
        # Concatenate multiple read files if needed
        if len(reads) > 1:
            concat_reads = work_dir / "all_reads.fastq"
            self._concatenate_reads(reads, concat_reads)
            input_reads = concat_reads
        else:
            input_reads = reads[0]
        
        # Run medaka_consensus
        cmd = [
            self.medaka,
            '-i', str(input_reads),
            '-d', str(assembly),
            '-o', str(work_dir),
            '-t', str(self.threads),
            '-m', self.model,
            '-b', str(self.batch_size)
        ]
        
        self._run_command(cmd, "medaka_consensus")
        
        # Move consensus output to final location
        consensus_file = work_dir / "consensus.fasta"
        if consensus_file.exists():
            shutil.copy(consensus_file, output)
        else:
            raise FileNotFoundError(f"Medaka consensus not found: {consensus_file}")
        
        return output
    
    def _concatenate_reads(self, reads: List[Path], output: Path):
        """Concatenate multiple read files."""
        self.logger.info(f"  Concatenating {len(reads)} read files...")
        with open(output, 'w') as outf:
            for read_file in reads:
                with open(read_file) as inf:
                    outf.write(inf.read())
    
    def _run_command(self, cmd: List[str], description: str):
        """Execute command with error handling."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"{description} completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description} failed:")
            self.logger.error(f"  Command: {' '.join(cmd)}")
            self.logger.error(f"  Error: {e.stderr}")
            raise
```

### 3.3 Racon Polisher (Universal)

```python
# File: strandweaver/finishing/racon_polisher.py

from pathlib import Path
from typing import List, Optional
import subprocess
import shutil
import logging


class RaconPolisher:
    """
    Racon polishing for any long reads (universal).
    
    Racon is a fast, universal long-read consensus caller that works with
    both PacBio and ONT reads when specialized tools aren't available.
    
    Requires: minimap2, racon
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config.get('finishing', {}).get('polishing', {}).get('racon', {})
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool paths
        self.minimap2 = self.config.get('minimap2_path', 'minimap2')
        self.racon = self.config.get('racon_path', 'racon')
        
        # Parameters
        self.threads = self.config.get('threads', 8)
        self.window_length = self.config.get('window_length', 500)
        self.quality_threshold = self.config.get('quality_threshold', 10.0)
    
    def check_availability(self) -> bool:
        """Check if Racon tools are available."""
        for tool in [self.minimap2, self.racon]:
            if shutil.which(tool) is None:
                self.logger.warning(f"Racon tool not found: {tool}")
                return False
        return True
    
    def polish(
        self,
        assembly: Path,
        reads: List[Path],
        output: Path
    ) -> Path:
        """
        Polish assembly using Racon.
        
        Steps:
        1. Align reads to assembly with minimap2
        2. Run racon for consensus calling
        
        Args:
            assembly: Input assembly FASTA
            reads: List of read files (FASTQ)
            output: Output polished assembly path
        
        Returns:
            Path to polished assembly
        """
        self.logger.info(f"Racon polishing: {assembly}")
        
        work_dir = output.parent
        paf_path = work_dir / f"{output.stem}.aligned.paf"
        
        # Concatenate multiple read files if needed
        if len(reads) > 1:
            concat_reads = work_dir / f"{output.stem}_all_reads.fastq"
            self._concatenate_reads(reads, concat_reads)
            input_reads = concat_reads
        else:
            input_reads = reads[0]
        
        # Step 1: Align with minimap2
        self._align_reads(assembly, input_reads, paf_path)
        
        # Step 2: Run Racon
        self._run_racon(input_reads, paf_path, assembly, output)
        
        return output
    
    def _align_reads(self, assembly: Path, reads: Path, output_paf: Path):
        """Align reads to assembly using minimap2."""
        self.logger.info("  Aligning reads with minimap2...")
        
        cmd = [
            self.minimap2,
            '-x', 'map-ont',  # or map-pb for PacBio
            '-t', str(self.threads),
            str(assembly),
            str(reads)
        ]
        
        with open(output_paf, 'w') as outf:
            result = subprocess.run(
                cmd,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        
        self.logger.debug("minimap2 alignment completed")
    
    def _run_racon(
        self,
        reads: Path,
        alignments: Path,
        target: Path,
        output: Path
    ):
        """Run Racon consensus caller."""
        self.logger.info("  Running Racon consensus...")
        
        cmd = [
            self.racon,
            '-t', str(self.threads),
            '-w', str(self.window_length),
            '-q', str(self.quality_threshold),
            str(reads),
            str(alignments),
            str(target)
        ]
        
        with open(output, 'w') as outf:
            result = subprocess.run(
                cmd,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        
        self.logger.debug("Racon completed")
    
    def _concatenate_reads(self, reads: List[Path], output: Path):
        """Concatenate multiple read files."""
        self.logger.info(f"  Concatenating {len(reads)} read files...")
        with open(output, 'w') as outf:
            for read_file in reads:
                with open(read_file) as inf:
                    outf.write(inf.read())
    
    def _run_command(self, cmd: List[str], description: str):
        """Execute command with error handling."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"{description} completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description} failed:")
            self.logger.error(f"  Command: {' '.join(cmd)}")
            self.logger.error(f"  Error: {e.stderr}")
            raise
```

---

## 4. Quality Assessment Integration

```python
# File: strandweaver/finishing/quality_assessment.py

from pathlib import Path
from typing import List, Optional, Tuple
import subprocess
import shutil
import logging
import re


class QualityAssessor:
    """
    Quality assessment using k-mer based methods (Merfin/Merqury).
    
    Integrates patterns from your T2T-Polish APv4.py implementation:
    - Merfin for QV calculation and k-mer spectrum analysis
    - Merqury for completeness assessment
    - GenomeScope2 for k-mer coverage estimation
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config.get('finishing', {}).get('quality_assessment', {})
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool paths
        self.merfin = self.config.get('merfin_path', 'merfin')
        self.merqury = self.config.get('merqury_path', 'merqury.sh')
        self.meryl = self.config.get('meryl_path', 'meryl')
        self.jellyfish = self.config.get('jellyfish_path', 'jellyfish')
        self.genomescope = self.config.get('genomescope_path', 'genomescope2')
        
        # Parameters
        self.kmer_size = self.config.get('kmer_size', 21)
        self.threads = self.config.get('threads', 8)
        self.use_merfin = self.config.get('use_merfin', True)
        self.use_merqury = self.config.get('use_merqury', True)
        
        # Cached k-mer databases
        self.kmer_db = None
        self.fitted_hist = None
        self.kcov = None
    
    def assess(
        self,
        assembly: Path,
        reads: List[Path]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Assess assembly quality.
        
        Returns QV and completeness scores.
        
        Args:
            assembly: Path to assembly FASTA
            reads: List of read files
        
        Returns:
            Tuple of (QV, completeness) or (None, None) if tools unavailable
        """
        # Build k-mer database if needed
        if self.kmer_db is None:
            self.kmer_db = self._build_kmer_db(reads)
        
        qv = None
        completeness = None
        
        # Merfin QV
        if self.use_merfin and shutil.which(self.merfin):
            qv = self._run_merfin(assembly)
        
        # Merqury completeness
        if self.use_merqury and shutil.which(self.merqury):
            completeness = self._run_merqury(assembly)
        
        return qv, completeness
    
    def _build_kmer_db(self, reads: List[Path]) -> Path:
        """Build k-mer database using Meryl."""
        self.logger.info("Building k-mer database...")
        
        output_db = reads[0].parent / f"kmer_db.k{self.kmer_size}.meryl"
        
        if output_db.exists():
            self.logger.info(f"  Using existing k-mer database: {output_db}")
            return output_db
        
        # Concatenate reads if multiple files
        if len(reads) > 1:
            concat_reads = output_db.parent / "all_reads_for_kmers.fastq"
            with open(concat_reads, 'w') as outf:
                for read_file in reads:
                    with open(read_file) as inf:
                        outf.write(inf.read())
            input_reads = concat_reads
        else:
            input_reads = reads[0]
        
        # Run Meryl count
        cmd = [
            self.meryl,
            f'k={self.kmer_size}',
            f'threads={self.threads}',
            'count',
            str(input_reads),
            'output',
            str(output_db)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"  K-mer database built: {output_db}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Meryl failed: {e.stderr}")
            return None
        
        return output_db
    
    def _run_merfin(self, assembly: Path) -> Optional[float]:
        """
        Run Merfin for QV calculation.
        
        Based on your T2T-Polish implementation.
        """
        self.logger.info("  Running Merfin QV calculation...")
        
        hist_out = assembly.parent / f"{assembly.stem}.merfin_hist.txt"
        
        # Merfin histogram command
        cmd = [
            self.merfin, '-hist',
            '-sequence', str(assembly),
            '-readmers', str(self.kmer_db),
            '-peak', str(self.kcov) if self.kcov else '100',
            '-output', str(hist_out)
        ]
        
        if self.fitted_hist:
            cmd += ['-prob', str(self.fitted_hist)]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Parse QV from output
            output = result.stdout or result.stderr
            qv_match = re.search(r'QV[:\s]+([0-9]+\\.?[0-9]*)', output)
            if qv_match:
                qv = float(qv_match.group(1))
                self.logger.info(f"    Merfin QV: {qv:.2f}")
                return qv
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Merfin failed: {e.stderr}")
        
        return None
    
    def _run_merqury(self, assembly: Path) -> Optional[float]:
        """
        Run Merqury for completeness assessment.
        
        Based on your T2T-Polish implementation.
        """
        self.logger.info("  Running Merqury completeness...")
        
        outprefix = assembly.parent / f"{assembly.stem}_merqury"
        
        cmd = [
            'bash', self.merqury,
            str(self.kmer_db),
            str(assembly),
            str(outprefix)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse completeness from .completeness.stats file
            comp_file = Path(f"{outprefix}.completeness.stats")
            if comp_file.exists():
                with open(comp_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            completeness = float(parts[4])
                            self.logger.info(f"    Merqury Completeness: {completeness:.4f}%")
                            return completeness
        
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Merqury failed: {e.stderr}")
        
        return None
```

---

## 5. Gap Filling Integration

```python
# File: strandweaver/finishing/gap_filler.py

from pathlib import Path
from typing import List, Optional
import subprocess
import shutil
import logging


class GapFiller:
    """
    Gap filling for scaffolds with N-gaps.
    
    Supports:
    - TGS-GapCloser (recommended for long reads)
    - LR_Gapcloser
    - Manual gap filling with long read alignments
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config.get('finishing', {}).get('gap_filling', {})
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool selection
        self.tool = self.config.get('tool', 'tgs-gapcloser')  # or 'lr_gapcloser'
        
        # Tool paths
        self.tgs_gapcloser = self.config.get('tgs_gapcloser_path', 'TGS-GapCloser.sh')
        self.lr_gapcloser = self.config.get('lr_gapcloser_path', 'LR_Gapcloser.sh')
        
        # Parameters
        self.threads = self.config.get('threads', 8)
        self.min_gap_size = self.config.get('min_gap_size', 1)  # Fill all gaps
        self.max_gap_size = self.config.get('max_gap_size', 10000)  # Up to 10kb
    
    def check_availability(self) -> bool:
        """Check if gap filling tools are available."""
        if self.tool == 'tgs-gapcloser':
            return shutil.which(self.tgs_gapcloser) is not None
        elif self.tool == 'lr_gapcloser':
            return shutil.which(self.lr_gapcloser) is not None
        return False
    
    def fill_gaps(
        self,
        scaffolds: Path,
        reads: List[Path],
        output: Path
    ) -> Path:
        """
        Fill N-gaps in scaffolds using long reads.
        
        Args:
            scaffolds: Input scaffolds with N-gaps
            reads: List of long read files
            output: Output path for gap-filled scaffolds
        
        Returns:
            Path to gap-filled scaffolds
        """
        self.logger.info(f"Gap filling: {scaffolds}")
        
        # Count gaps before
        n_gaps_before = self._count_gaps(scaffolds)
        self.logger.info(f"  Gaps before: {n_gaps_before}")
        
        if n_gaps_before == 0:
            self.logger.info("  No gaps to fill")
            shutil.copy(scaffolds, output)
            return output
        
        # Run gap filler
        if self.tool == 'tgs-gapcloser':
            self._run_tgs_gapcloser(scaffolds, reads, output)
        elif self.tool == 'lr_gapcloser':
            self._run_lr_gapcloser(scaffolds, reads, output)
        else:
            self.logger.error(f"Unknown gap filling tool: {self.tool}")
            shutil.copy(scaffolds, output)
            return output
        
        # Count gaps after
        n_gaps_after = self._count_gaps(output)
        n_closed = n_gaps_before - n_gaps_after
        closure_rate = (n_closed / n_gaps_before * 100) if n_gaps_before > 0 else 0
        
        self.logger.info(f"  Gaps after: {n_gaps_after}")
        self.logger.info(f"  Gaps closed: {n_closed} ({closure_rate:.1f}%)")
        
        return output
    
    def _run_tgs_gapcloser(self, scaffolds: Path, reads: List[Path], output: Path):
        """Run TGS-GapCloser."""
        self.logger.info("  Running TGS-GapCloser...")
        
        # TGS-GapCloser expects a file list for reads
        read_list = output.parent / "reads.fofn"
        with open(read_list, 'w') as f:
            for read_file in reads:
                f.write(f"{read_file}\n")
        
        cmd = [
            self.tgs_gapcloser,
            '--scaff', str(scaffolds),
            '--reads', str(read_list),
            '--output', str(output.parent),
            '--thread', str(self.threads),
            '--min_match', '100',  # Minimum match length
            '--min_idy', '0.9',    # Minimum identity
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # TGS-GapCloser outputs to scaff.fa by default
            tgs_output = output.parent / "scaff.fa"
            if tgs_output.exists():
                shutil.move(tgs_output, output)
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"TGS-GapCloser failed: {e.stderr}")
            # Copy input to output as fallback
            shutil.copy(scaffolds, output)
    
    def _run_lr_gapcloser(self, scaffolds: Path, reads: List[Path], output: Path):
        """Run LR_Gapcloser."""
        self.logger.info("  Running LR_Gapcloser...")
        
        # Concatenate reads
        concat_reads = output.parent / "all_reads.fasta"
        with open(concat_reads, 'w') as outf:
            for read_file in reads:
                with open(read_file) as inf:
                    outf.write(inf.read())
        
        cmd = [
            self.lr_gapcloser,
            '-i', str(scaffolds),
            '-l', str(concat_reads),
            '-o', str(output),
            '-t', str(self.threads)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"LR_Gapcloser failed: {e.stderr}")
            shutil.copy(scaffolds, output)
    
    def _count_gaps(self, fasta: Path) -> int:
        """Count number of N-gap regions in FASTA."""
        count = 0
        with open(fasta) as f:
            for line in f:
                if not line.startswith('>'):
                    # Count runs of N's
                    count += line.upper().count('NNN')  # At least 3 N's
        return count
```

---

## 6. Updated Pipeline Integration

```python
# File: strandweaver/utils/pipeline.py (_step_finish method)

def _step_finish(self):
    """Finishing step (polishing, gap filling, etc.)."""
    from ..finishing.polisher import AssemblyPolisher
    from ..finishing.gap_filler import GapFiller
    
    self.logger.info("="*70)
    self.logger.info("FINISHING ASSEMBLY")
    self.logger.info("="*70)
    
    # Determine input: scaffolds (if Hi-C) or contigs
    if self.state.get('final_scaffolds') is not None:
        input_assembly = self.state['final_scaffolds']
        assembly_path = self.output_dir / "scaffolds.fasta"
        assembly_type = "scaffolds"
    elif self.state.get('final_contigs') is not None:
        input_assembly = self.state['final_contigs']
        assembly_path = self.output_dir / "contigs.fasta"
        assembly_type = "contigs"
    else:
        # Try loading from files
        scaffolds_path = self.output_dir / "scaffolds.fasta"
        contigs_path = self.output_dir / "contigs.fasta"
        
        if scaffolds_path.exists():
            input_assembly = list(read_fasta(scaffolds_path))
            assembly_path = scaffolds_path
            assembly_type = "scaffolds"
        elif contigs_path.exists():
            input_assembly = list(read_fasta(contigs_path))
            assembly_path = contigs_path
            assembly_type = "contigs"
        else:
            raise FileNotFoundError("No contigs or scaffolds found for finishing step")
    
    self.logger.info(f"Input: {assembly_type} ({len(input_assembly)} sequences)")
    
    # Get read files for polishing
    reads = self._get_read_files_for_polishing()
    technologies = self.state.get('technologies', [])
    
    # ========================================================================
    # POLISHING
    # ========================================================================
    
    polishing_config = self.config.get('finishing', {}).get('polishing', {})
    
    if polishing_config.get('enabled', False):
        polisher = AssemblyPolisher(
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        # Run iterative polishing
        polishing_results = polisher.polish_assembly(
            assembly=assembly_path,
            reads=reads,
            technologies=technologies,
            max_iterations=polishing_config.get('max_iterations', 3),
            convergence_threshold=polishing_config.get('convergence_threshold', 0.01),
            resume=self.state.get('resume', False)
        )
        
        # Update assembly path to polished version
        if polishing_results:
            assembly_path = polishing_results[-1].assembly_path
            self.logger.info(f"✓ Polishing complete: {assembly_path}")
        else:
            self.logger.warning("Polishing was skipped or failed")
    else:
        self.logger.info("Polishing disabled in configuration")
    
    # ========================================================================
    # GAP FILLING
    # ========================================================================
    
    gap_filling_config = self.config.get('finishing', {}).get('gap_filling', {})
    
    if gap_filling_config.get('enabled', False):
        gap_filler = GapFiller(
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        if gap_filler.check_availability():
            gap_filled_path = self.output_dir / "gap_filled.fasta"
            
            gap_filler.fill_gaps(
                scaffolds=assembly_path,
                reads=reads,
                output=gap_filled_path
            )
            
            assembly_path = gap_filled_path
            self.logger.info(f"✓ Gap filling complete: {assembly_path}")
        else:
            self.logger.warning(f"Gap filling tool not available: {gap_filling_config.get('tool')}")
    else:
        self.logger.info("Gap filling disabled in configuration")
    
    # ========================================================================
    # FINAL ASSEMBLY
    # ========================================================================
    
    # Save final assembly
    final_path = self.output_dir / "final_assembly.fasta"
    shutil.copy(assembly_path, final_path)
    
    # Calculate final statistics
    final_assembly = list(read_fasta(final_path))
    total_length = sum(len(seq.sequence) for seq in final_assembly)
    n50 = self._calculate_n50([len(seq.sequence) for seq in final_assembly])
    
    self.logger.info(f"\n{'='*70}")
    self.logger.info("FINISHING COMPLETE")
    self.logger.info(f"{'='*70}")
    self.logger.info(f"Final assembly: {final_path}")
    self.logger.info(f"Sequences: {len(final_assembly)}")
    self.logger.info(f"Total length: {total_length:,} bp")
    self.logger.info(f"N50: {n50:,} bp")
    self.logger.info(f"{'='*70}\n")

def _get_read_files_for_polishing(self) -> List[Path]:
    """Get read files for polishing."""
    reads = []
    
    # Corrected reads (preferred)
    if self.state.get('corrected_files'):
        reads.extend([Path(f) for f in self.state['corrected_files']])
    
    # Original reads (fallback)
    elif self.state.get('raw_files'):
        reads.extend([Path(f) for f in self.state['raw_files']])
    
    # From config
    elif self.config.get('input', {}).get('reads'):
        reads.append(Path(self.config['input']['reads']))
    
    return reads
```

---

## 7. Configuration Updates

Add to `defaults.yaml`:

```yaml
# Finishing configuration
finishing:
  # Polishing
  polishing:
    enabled: true
    max_iterations: 3
    convergence_threshold: 0.01  # QV improvement threshold
    
    # Arrow (PacBio)
    arrow:
      pbmm2_path: pbmm2
      gcpp_path: gcpp
      samtools_path: samtools
      threads: 8
      min_coverage: 5
      algorithm: arrow  # arrow or quiver
    
    # Medaka (ONT)
    medaka:
      medaka_path: medaka_consensus
      threads: 8
      model: r941_min_hac_g507  # Auto-detect recommended
      batch_size: 100
    
    # Racon (Universal)
    racon:
      minimap2_path: minimap2
      racon_path: racon
      threads: 8
      window_length: 500
      quality_threshold: 10.0
  
  # Quality assessment
  quality_assessment:
    enabled: true
    use_merfin: true
    use_merqury: true
    kmer_size: 21
    threads: 8
    merfin_path: merfin
    merqury_path: merqury.sh
    meryl_path: meryl
    jellyfish_path: jellyfish
    genomescope_path: genomescope2
  
  # Gap filling
  gap_filling:
    enabled: false  # Disabled by default (experimental)
    tool: tgs-gapcloser  # or lr_gapcloser
    tgs_gapcloser_path: TGS-GapCloser.sh
    lr_gapcloser_path: LR_Gapcloser.sh
    threads: 8
    min_gap_size: 1
    max_gap_size: 10000
```

---

## 8. Implementation Timeline

### Week 1 (Days 1-5): Core Polishing
- **Day 1**: Implement base `AssemblyPolisher` class
- **Day 2**: Implement `ArrowPolisher` (PacBio)
- **Day 3**: Implement `MedakaPolisher` (ONT)
- **Day 4**: Implement `RaconPolisher` (universal)
- **Day 5**: Testing with real data, bug fixes

### Week 2 (Days 6-10): Quality & Gap Filling
- **Day 6-7**: Implement `QualityAssessor` (Merfin/Merqury integration)
- **Day 8**: Implement `GapFiller`
- **Day 9**: Integrate into `pipeline.py`, update configuration
- **Day 10**: Testing, documentation, benchmarking

---

## 9. Testing Strategy

### Unit Tests
```python
# Test each polisher independently
def test_arrow_polisher():
    # Test with PacBio HiFi reads
    # Verify output quality

def test_medaka_polisher():
    # Test with ONT reads
    # Verify convergence

def test_racon_polisher():
    # Test with mixed reads
    # Verify fallback works
```

### Integration Tests
```python
def test_full_polishing_workflow():
    # Test complete finishing pipeline
    # Verify QV improvement
    # Check convergence detection
```

### Real Data Testing
- **PacBio HiFi**: E. coli, human chr22
- **ONT**: S. cerevisiae, C. elegans
- **Mixed**: Test with both read types

---

## 10. Success Metrics

**Quality Improvements**:
- QV increase: 30-35 → 50+ after polishing
- Completeness: >99.5%
- Error rate: <1 error per 100kb

**Performance**:
- Arrow: ~2-4 hours per Gb
- Medaka: ~4-8 hours per Gb
- Racon: ~1-2 hours per Gb

**Convergence**:
- Most assemblies converge in 2-3 rounds
- QV improvement <0.01 triggers stop

---

## 11. Future Enhancements

### Phase 2 (Optional)
1. **Homopolymer correction** (ONT-specific)
2. **Variant-aware polishing** (preserve heterozygosity)
3. **Hybrid polishing** (Illumina + long reads)
4. **GPU acceleration** (Medaka)
5. **Distributed polishing** (split-apply-combine)

### Phase 3 (Research)
1. **ML-based polishing** (train custom models)
2. **Context-aware error correction**
3. **Structural variant preservation**

---

## 12. Dependencies

**Required Tools**:
- Arrow: `pbmm2`, `gcpp`, `samtools`
- Medaka: `medaka` (includes minimap2)
- Racon: `minimap2`, `racon`
- QC: `merfin`, `merqury`, `meryl`, `jellyfish`, `genomescope2`
- Gap filling: `TGS-GapCloser` or `LR_Gapcloser`

**Python Packages**:
- `pysam` (BAM/CRAM handling)
- `biopython` (FASTA parsing)
- All existing StrandWeaver dependencies

---

## 13. Questions for Discussion

1. **Priority**: Should we implement all 3 polishers initially, or start with Racon (universal)?
2. **Quality assessment**: Required or optional? (adds significant time)
3. **Gap filling**: Include in v1 or defer to v2?
4. **Convergence**: Should we always do 3 rounds, or stop early?
5. **Claude AI finishing**: What is the intended purpose of this feature?

---

## Conclusion

This proposal provides a comprehensive, production-ready polishing system that:
- ✅ Leverages your proven T2T-Polish patterns
- ✅ Integrates seamlessly with StrandWeaver architecture
- ✅ Supports all major long-read technologies
- ✅ Provides quality tracking and convergence detection
- ✅ Enables iterative refinement
- ✅ Maintains checkpoint/resume capability

**Recommendation**: Start with Racon (universal fallback) and quality assessment, then add Arrow/Medaka for technology-specific optimization. Gap filling can be deferred to Phase 2.

**Next Steps**:
1. Review and approve proposal
2. Begin implementation (Week 1: Core polishing)
3. Test with real datasets
4. Benchmark against existing polishing pipelines
5. Integrate with full StrandWeaver pipeline

Would you like me to begin implementation, or would you prefer to discuss any aspects of the proposal first?
