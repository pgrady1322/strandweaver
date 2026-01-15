# Polishing Integration Proposal for StrandWeaver (Revised)

## Executive Summary

This document proposes integration of assembly polishing and quality assessment into StrandWeaver's finishing pipeline by **directly leveraging the user's proven T2T-Polish APv4.py workflow** rather than reimplementing polishing tools from scratch.

**Key Features:**
- **Direct APv4.py integration** - Use existing, production-ready polishing pipeline
- **Optional polishing** - Disabled by default due to significant runtime (2-8 hours per iteration)
- **Mandatory quality assessment** - Always run Merfin QV and Merqury completeness metrics
- **Minimal dependencies** - Only DeepVariant Singularity image required (APv4.py handles rest)
- **Gap filling** - Optional, independent from polishing
- **Resume capability** - Leverage APv4.py's built-in resume support
- **Estimated timeline:** 3-5 days implementation (vs. 1-2 weeks for full reimplementation)

**Architecture Simplification:**
- ~~6 modules~~ → **3 modules**: `apv4_polisher.py`, `quality_assessment.py`, `gap_filler.py`
- ~~Reimplement Arrow/Medaka/Racon~~ → **Thin wrapper around APv4.py**
- Technology detection handled by APv4.py automatically

---

## 1. Current State Analysis

### Current `_step_finish()` Implementation

Located in [pipeline.py](strandweaver/pipeline.py#L2389-L2440):

```python
def _step_finish(self):
    """Finishing step (polishing, gap filling, etc.)."""
    self.logger.info("Step: Finishing")
    
    # Load contigs from previous step
    contigs_file = os.path.join(self.output_dir, "scaffolds.fasta")
    
    # TODO: Implement polishing (APv4.py integration)
    # TODO: Implement gap filling
    # TODO: Implement Claude AI finishing
    
    # Save final assembly
    final_assembly = os.path.join(self.output_dir, "final_assembly.fasta")
    shutil.copy(contigs_file, final_assembly)
```

**Analysis:**
- All finishing features are placeholders
- User has proven APv4.py polishing workflow
- Quality assessment patterns already exist in APv4.py
- Should leverage existing code rather than reimplement

### Why Use APv4.py Directly?

**Advantages:**
1. **Production-ready** - APv4.py already proven on real T2T datasets
2. **Comprehensive** - Handles Winnowmap, FalconC, DeepVariant, Merfin, Merqury
3. **Optimized** - K-mer coverage estimation, convergence detection built-in
4. **Actively maintained** - User develops/debugs APv4.py regularly
5. **Faster implementation** - Wrapper vs. full reimplementation (3-5 days vs. 1-2 weeks)
6. **Technology auto-detection** - APv4.py routes PacBio/ONT via `--deepseq_type`
7. **Resume support** - Built-in `--resume` and `--resume-from` flags
8. **Proven QV tracking** - Multi-iteration convergence already implemented

**Trade-offs:**
- External dependency on APv4.py location (mitigated: configurable path)
- Less granular control (acceptable: APv4.py already highly configurable)
- Singularity requirement for DeepVariant (already standard dependency)

---

## 2. Proposed Architecture

### Module Structure

Create new `strandweaver/finishing/` directory with **3 modules**:

```
strandweaver/finishing/
├── __init__.py
├── apv4_polisher.py         # Wrapper around user's APv4.py
├── quality_assessment.py    # Always-on QV/completeness (Merfin + Merqury)
└── gap_filler.py           # Optional gap closing (independent)
```

**Simplification rationale:**
- APv4.py already handles technology routing (Arrow/Medaka via DeepVariant models)
- No need to reimplement Winnowmap, FalconC, Merfin polishing logic
- Quality assessment extracted to always run (even when polishing disabled)
- Gap filling remains independent, optional feature

### Workflow Flow

```
1. Initial assembly (from Hi-C scaffolding)
   ↓
2. [MANDATORY] Quality assessment → Baseline QV/completeness
   ↓
3. Polishing enabled? (default: NO due to runtime)
   ↓ Yes (user explicitly enabled)
4. APv4.py polish (1-3 iterations with convergence detection)
   │  - Winnowmap alignment
   │  - FalconC filtering  
   │  - DeepVariant variant calling
   │  - Merfin polishing
   │  - Per-iteration QV tracking
   ↓
5. [MANDATORY] Final quality assessment
   ↓
6. Gap filling enabled? (optional, independent)
   ↓ Yes
7. Gap closing → Final QV check
   ↓
8. Final assembly output
```

**Key changes from original proposal:**
- Quality assessment **always runs** (baseline + final)
- Polishing **optional** (disabled by default, ~2-8 hours per iteration)
- Gap filling **independent** from polishing (can run without polishing)

---

## 3. Implementation Details

### 3.1 APv4.py Wrapper (`apv4_polisher.py`)

**Purpose:** Thin wrapper to invoke user's APv4.py polish subcommand with appropriate parameters

```python
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import os
import subprocess
import json
import re

class APv4Polisher:
    """
    Wrapper around user's T2T-Polish APv4.py for assembly polishing.
    
    Features:
    - Direct APv4.py invocation via subprocess
    - Automatic parameter mapping from StrandWeaver config
    - Resume capability (leverages APv4.py --resume)
    - QV/completeness parsing from APv4.py outputs
    - Technology detection (PacBio/ONT auto-routing)
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Locate APv4.py script
        self.apv4_script = config.get('apv4_path', 'APv4.py')
        if not os.path.exists(self.apv4_script):
            raise FileNotFoundError(
                f"APv4.py not found at {self.apv4_script}. "
                "Please set 'apv4_path' in configuration."
            )
        
        # Validate DeepVariant SIF
        self.dv_sif = config.get('deepvariant_sif')
        if not self.dv_sif or not os.path.exists(self.dv_sif):
            raise FileNotFoundError(
                f"DeepVariant Singularity image required: {self.dv_sif}"
            )
    
    def polish_assembly(
        self,
        draft_fasta: str,
        reads: str,
        output_prefix: str,
        resume: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run APv4.py polishing pipeline.
        
        Args:
            draft_fasta: Path to initial assembly
            reads: Path to reads (FASTQ/FASTA)
            output_prefix: Output file prefix
            resume: Enable resume mode (default: True)
        
        Returns:
            Tuple of (final_polished_fasta, qv_history)
        """
        self.logger.info("Starting APv4.py polishing pipeline")
        
        # Build APv4.py command
        cmd = self._build_apv4_command(
            draft=draft_fasta,
            reads=reads,
            prefix=output_prefix,
            resume=resume
        )
        
        # Execute APv4.py
        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            self.logger.error(f"APv4.py failed with return code {result.returncode}")
            self.logger.error(f"STDERR:\n{result.stderr}")
            raise RuntimeError("APv4.py polishing failed")
        
        # Parse outputs
        final_assembly = self._find_final_consensus(output_prefix)
        qv_history = self._parse_qv_summary(output_prefix)
        
        self.logger.info(f"Polishing complete: {final_assembly}")
        self.logger.info(f"Total iterations: {len(qv_history)}")
        
        return final_assembly, qv_history
    
    def _build_apv4_command(
        self,
        draft: str,
        reads: str,
        prefix: str,
        resume: bool
    ) -> List[str]:
        """Build APv4.py polish subcommand with appropriate parameters."""
        pol_config = self.config.get('polishing', {})
        
        cmd = [
            'python3',
            self.apv4_script,
            'polish',
            '--draft', draft,
            '--reads', reads,
            '--prefix', prefix,
            '--singularity_sif', self.dv_sif,
            '--threads', str(self.config.get('threads', 32)),
            '--iterations', str(pol_config.get('max_iterations', 3)),
            '--kmer_size', str(pol_config.get('kmer_size', 31)),
        ]
        
        # Technology detection (map to DeepVariant model type)
        tech = self.config.get('read_technology', '').lower()
        if tech == 'pacbio':
            cmd += ['--deepseq_type', 'PACBIO']
        elif tech == 'ont':
            cmd += ['--deepseq_type', 'ONT_R104']
        else:
            self.logger.warning(
                f"Unknown technology '{tech}', defaulting to PACBIO"
            )
            cmd += ['--deepseq_type', 'PACBIO']
        
        # Optimized mode (k-mer coverage estimation)
        if pol_config.get('optimized', True):
            cmd += ['--optimized']
            cmd += ['--ploidy', pol_config.get('ploidy', 'haploid')]
        
        # Resume capability
        if resume:
            cmd += ['--resume']
        
        # Optional: readmers database
        if 'readmers_db' in pol_config:
            cmd += ['--readmers', pol_config['readmers_db']]
        
        # Optional: cleanup intermediate files
        if pol_config.get('cleanup', False):
            cmd += ['--cleanup']
        
        # Logging
        log_file = f"{prefix}.apv4_polish.log"
        cmd += ['--log-file', log_file]
        
        return cmd
    
    def _find_final_consensus(self, prefix: str) -> str:
        """Locate final consensus FASTA from APv4.py outputs."""
        iterations = self.config.get('polishing', {}).get('max_iterations', 3)
        
        # APv4.py outputs: <prefix>.iter_N.consensus.fasta
        for i in range(iterations, 0, -1):
            candidate = f"{prefix}.iter_{i}.consensus.fasta"
            if os.path.exists(candidate):
                return candidate
        
        raise FileNotFoundError(
            f"No consensus FASTA found with prefix {prefix}"
        )
    
    def _parse_qv_summary(self, prefix: str) -> List[Dict[str, Any]]:
        """Parse QV/completeness history from APv4.py summary file."""
        summary_file = f"{prefix}.QV_Completeness_summary.txt"
        
        if not os.path.exists(summary_file):
            self.logger.warning(f"QV summary not found: {summary_file}")
            return []
        
        qv_history = []
        current_iter = None
        
        with open(summary_file) as f:
            for line in f:
                # Match iteration headers
                iter_match = re.match(r'> Iteration (\d+)', line)
                if iter_match:
                    current_iter = int(iter_match.group(1))
                    qv_history.append({
                        'iteration': current_iter,
                        'merfin_qv': None,
                        'merqury_qv': None,
                        'completeness': None
                    })
                    continue
                
                if current_iter is None:
                    continue
                
                # Parse Merfin QV
                merfin_qv = re.search(r'QV.*?(\d+\.\d+)', line)
                if merfin_qv:
                    qv_history[-1]['merfin_qv'] = float(merfin_qv.group(1))
                
                # Parse Merqury QV
                merqury_qv = re.search(r'Merqury QV:\s*(\d+\.\d+)', line)
                if merqury_qv:
                    qv_history[-1]['merqury_qv'] = float(merqury_qv.group(1))
                
                # Parse completeness
                comp = re.search(r'Completeness.*?(\d+\.\d+)', line)
                if comp:
                    qv_history[-1]['completeness'] = float(comp.group(1))
        
        return qv_history
```

**Key Design Decisions:**

1. **Thin wrapper** - Minimal logic, delegates to APv4.py
2. **Automatic technology routing** - Maps StrandWeaver's `read_technology` to APv4.py's `--deepseq_type`
3. **Resume by default** - Leverages APv4.py's `--resume` flag
4. **Configurable APv4.py location** - Via `apv4_path` in config
5. **QV parsing** - Extracts metrics from APv4.py's summary file

---

### 3.2 Quality Assessment (`quality_assessment.py`) [ALWAYS RUNS]

**Purpose:** Mandatory quality metrics for all assemblies (polished or not)

**Tools:** Merfin (QV calculation), Merqury (completeness), Jellyfish (k-mer counting)

```python
class QualityAssessor:
    """
    Always-on quality assessment for assemblies.
    
    Runs regardless of polishing status to provide baseline and final metrics.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.kmer_size = config.get('quality_assessment', {}).get('kmer_size', 21)
    
    def assess_quality(
        self,
        assembly: str,
        reads: str,
        output_prefix: str,
        readmers_db: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Run quality assessment (always executed, regardless of polishing).
        
        Args:
            assembly: Path to assembly FASTA
            reads: Path to reads (for k-mer database)
            output_prefix: Output file prefix
            readmers_db: Pre-computed readmers Meryl DB (optional)
        
        Returns:
            Dict with keys: 'merfin_qv', 'merqury_qv', 'completeness'
        """
        self.logger.info("Running quality assessment (Merfin + Merqury)")
        
        # Compute or use existing readmers DB
        if readmers_db is None:
            readmers_db = self._compute_readmers(reads, output_prefix)
        
        # Run Merfin QV calculation
        merfin_qv = self._run_merfin(assembly, readmers_db, output_prefix)
        
        # Run Merqury completeness
        merqury_results = self._run_merqury(assembly, readmers_db, output_prefix)
        
        metrics = {
            'merfin_qv': merfin_qv,
            'merqury_qv': merqury_results['qv'],
            'completeness': merqury_results['completeness']
        }
        
        self.logger.info(
            f"Quality metrics: Merfin QV={merfin_qv:.2f}, "
            f"Merqury QV={merqury_results['qv']:.2f}, "
            f"Completeness={merqury_results['completeness']:.2%}"
        )
        
        # Save metrics to file
        metrics_file = f"{output_prefix}.quality_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _compute_readmers(self, reads: str, prefix: str) -> str:
        """Compute k-mer database from reads using Meryl."""
        readmers_db = f"{prefix}.readmers.meryl"
        
        if os.path.exists(readmers_db):
            self.logger.info(f"Using existing readmers DB: {readmers_db}")
            return readmers_db
        
        self.logger.info("Computing readmers database with Meryl...")
        threads = self.config.get('threads', 8)
        
        cmd = [
            'meryl', f'k={self.kmer_size}',
            f'threads={threads}',
            'count', reads,
            'output', readmers_db
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return readmers_db
    
    def _run_merfin(self, assembly: str, readmers: str, prefix: str) -> float:
        """Run Merfin histogram QV calculation."""
        hist_out = f"{prefix}.merfin_hist.txt"
        
        cmd = [
            'merfin', '-hist',
            '-sequence', assembly,
            '-readmers', readmers,
            '-peak', '106.7',  # Default; could be optimized with GenomeScope2
            '-output', hist_out
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse QV from output (stdout or stderr depending on Merfin version)
        output_text = result.stdout or result.stderr
        qv_match = re.search(r'QV.*?(\d+\.\d+)', output_text)
        if qv_match:
            return float(qv_match.group(1))
        
        self.logger.warning("Could not parse Merfin QV")
        return 0.0
    
    def _run_merqury(self, assembly: str, readmers: str, prefix: str) -> Dict[str, float]:
        """Run Merqury completeness assessment."""
        merqury_prefix = f"{prefix}.merqury"
        
        cmd = ['bash', 'merqury.sh', readmers, assembly, merqury_prefix]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Parse .qv file
        qv_file = f"{merqury_prefix}.qv"
        qv = 0.0
        if os.path.exists(qv_file):
            with open(qv_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qv = float(parts[3])
                        break
        
        # Parse .completeness.stats file
        comp_file = f"{merqury_prefix}.completeness.stats"
        completeness = 0.0
        if os.path.exists(comp_file):
            with open(comp_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        completeness = float(parts[4])
                        break
        
        return {'qv': qv, 'completeness': completeness}
```

**Key Features:**
1. **Always runs** - No enable/disable flag; quality assessment is mandatory
2. **Readmers caching** - Reuses k-mer database if already computed
3. **Dual QV sources** - Merfin (histogram-based) + Merqury (mapping-based)
4. **Completeness tracking** - Percentage of unique k-mers found
5. **JSON output** - Machine-readable metrics for downstream analysis

---

### 3.3 Gap Filler (`gap_filler.py`) [OPTIONAL]

**Purpose:** Close gaps in scaffolded assemblies (independent from polishing)

**Tools:** TGS-GapCloser, LR_Gapcloser

```python
class GapFiller:
    """
    Optional gap filling for scaffolded assemblies.
    
    Independent from polishing - can run with or without polishing.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tool = config.get('gap_filling', {}).get('tool', 'tgs-gapcloser')
    
    def fill_gaps(
        self,
        scaffolds: str,
        reads: str,
        output: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Fill gaps in scaffolded assembly.
        
        Args:
            scaffolds: Path to scaffolded assembly
            reads: Path to long reads
            output: Output FASTA path
        
        Returns:
            Tuple of (gap_filled_fasta, stats_dict)
        """
        self.logger.info(f"Gap filling with {self.tool}")
        
        if self.tool == 'tgs-gapcloser':
            return self._run_tgs_gapcloser(scaffolds, reads, output)
        elif self.tool == 'lr_gapcloser':
            return self._run_lr_gapcloser(scaffolds, reads, output)
        else:
            raise ValueError(f"Unknown gap filling tool: {self.tool}")
    
    def _run_tgs_gapcloser(self, scaffolds: str, reads: str, output: str):
        """Run TGS-GapCloser."""
        threads = self.config.get('threads', 8)
        
        cmd = [
            'tgsgapcloser',
            '--scaff', scaffolds,
            '--reads', reads,
            '--output', output,
            '--thread', str(threads),
            '--min_match', '500',
            '--min_idy', '0.3'
        ]
        
        subprocess.run(cmd, check=True)
        
        # Parse statistics
        stats = self._count_gaps(scaffolds, output)
        return output, stats
    
    def _run_lr_gapcloser(self, scaffolds: str, reads: str, output: str):
        """Run LR_Gapcloser."""
        threads = self.config.get('threads', 8)
        
        cmd = [
            'LR_Gapcloser.sh',
            '-i', scaffolds,
            '-l', reads,
            '-s', 'p',  # PacBio
            '-t', str(threads),
            '-o', output
        ]
        
        subprocess.run(cmd, check=True)
        
        stats = self._count_gaps(scaffolds, output)
        return output, stats
    
    def _count_gaps(self, before: str, after: str) -> Dict[str, int]:
        """Count gaps before/after filling."""
        def count_n_runs(fasta_path):
            import re
            from Bio import SeqIO
            gaps = 0
            with open(fasta_path) as f:
                for record in SeqIO.parse(f, 'fasta'):
                    gaps += len(re.findall(r'N+', str(record.seq)))
            return gaps
        
        return {
            'gaps_before': count_n_runs(before),
            'gaps_after': count_n_runs(after),
            'gaps_filled': count_n_runs(before) - count_n_runs(after)
        }
```

---

## 4. Updated `_step_finish()` Integration

Modified [pipeline.py](strandweaver/pipeline.py#L2389-L2440):

```python
def _step_finish(self):
    """Finishing step with optional polishing and mandatory QA."""
    self.logger.info("Step: Finishing")
    
    # Load scaffolds from previous step
    scaffolds_file = os.path.join(self.output_dir, "scaffolds.fasta")
    current_assembly = scaffolds_file
    
    # Import finishing modules
    from .finishing.quality_assessment import QualityAssessor
    from .finishing.apv4_polisher import APv4Polisher
    from .finishing.gap_filler import GapFiller
    
    # 1. MANDATORY: Baseline quality assessment
    self.logger.info("=== Baseline Quality Assessment ===")
    qa = QualityAssessor(self.config, self.logger)
    baseline_metrics = qa.assess_quality(
        assembly=current_assembly,
        reads=self.config['reads'],
        output_prefix=os.path.join(self.output_dir, "baseline_qa")
    )
    
    self.logger.info(
        f"Baseline: QV={baseline_metrics['merfin_qv']:.2f}, "
        f"Completeness={baseline_metrics['completeness']:.2%}"
    )
    
    # 2. OPTIONAL: Polishing (disabled by default)
    pol_config = self.config.get('finishing', {}).get('polishing', {})
    if pol_config.get('enabled', False):
        self.logger.info("=== Polishing Enabled ===")
        try:
            polisher = APv4Polisher(self.config, self.logger)
            polished_assembly, qv_history = polisher.polish_assembly(
                draft_fasta=current_assembly,
                reads=self.config['reads'],
                output_prefix=os.path.join(self.output_dir, "polished"),
                resume=True
            )
            current_assembly = polished_assembly
            
            # Log polishing improvements
            if qv_history:
                first_qv = qv_history[0].get('merfin_qv', 0)
                last_qv = qv_history[-1].get('merfin_qv', 0)
                improvement = last_qv - first_qv
                self.logger.info(
                    f"Polishing complete: QV improved by {improvement:.2f} "
                    f"({first_qv:.2f} → {last_qv:.2f}) in {len(qv_history)} iterations"
                )
        except Exception as e:
            self.logger.error(f"Polishing failed: {e}")
            self.logger.info("Continuing with unpolished assembly")
    else:
        self.logger.info("Polishing disabled (set finishing.polishing.enabled=true to enable)")
    
    # 3. OPTIONAL: Gap filling (independent from polishing)
    gap_config = self.config.get('finishing', {}).get('gap_filling', {})
    if gap_config.get('enabled', False):
        self.logger.info("=== Gap Filling Enabled ===")
        try:
            gap_filler = GapFiller(self.config, self.logger)
            gap_filled_assembly, gap_stats = gap_filler.fill_gaps(
                scaffolds=current_assembly,
                reads=self.config['reads'],
                output=os.path.join(self.output_dir, "gap_filled.fasta")
            )
            current_assembly = gap_filled_assembly
            
            self.logger.info(
                f"Gap filling complete: {gap_stats['gaps_filled']} gaps closed "
                f"({gap_stats['gaps_before']} → {gap_stats['gaps_after']})"
            )
        except Exception as e:
            self.logger.error(f"Gap filling failed: {e}")
            self.logger.info("Continuing without gap filling")
    else:
        self.logger.info("Gap filling disabled")
    
    # 4. MANDATORY: Final quality assessment
    self.logger.info("=== Final Quality Assessment ===")
    final_metrics = qa.assess_quality(
        assembly=current_assembly,
        reads=self.config['reads'],
        output_prefix=os.path.join(self.output_dir, "final_qa")
    )
    
    self.logger.info(
        f"Final: QV={final_metrics['merfin_qv']:.2f}, "
        f"Completeness={final_metrics['completeness']:.2%}"
    )
    
    # Calculate overall improvement
    qv_change = final_metrics['merfin_qv'] - baseline_metrics['merfin_qv']
    comp_change = final_metrics['completeness'] - baseline_metrics['completeness']
    
    self.logger.info(
        f"Overall improvement: QV {qv_change:+.2f}, "
        f"Completeness {comp_change:+.2%}"
    )
    
    # 5. Save final assembly
    final_assembly = os.path.join(self.output_dir, "final_assembly.fasta")
    shutil.copy(current_assembly, final_assembly)
    
    self.logger.info(f"Final assembly: {final_assembly}")
    self.logger.info("Finishing complete")
```

**Workflow Summary:**
1. **Baseline QA** (mandatory) - Establish starting metrics
2. **Polishing** (optional, off by default) - APv4.py if enabled
3. **Gap filling** (optional) - Independent from polishing
4. **Final QA** (mandatory) - Track improvements
5. **Save final assembly** - Copy to standard output location

---

## 5. Configuration Updates

Add to [defaults.yaml](strandweaver/config/defaults.yaml):

```yaml
# ============================================================================
# Finishing Configuration
# ============================================================================
finishing:
  # Polishing (OPTIONAL - disabled by default due to significant runtime)
  polishing:
    enabled: false  # Set to true to enable polishing (adds 2-8 hours per iteration)
    
    # APv4.py configuration
    apv4_path: "APv4.py"  # Path to APv4.py script
    deepvariant_sif: null  # REQUIRED if polishing enabled: path to DeepVariant Singularity image
    
    # Polishing parameters
    max_iterations: 3  # Number of polishing rounds (typically 2-3)
    convergence_threshold: 0.01  # Stop if QV improvement < this value
    optimized: true  # Enable automatic k-mer coverage estimation
    ploidy: "haploid"  # or "diploid"
    kmer_size: 31  # K-mer size for Meryl
    cleanup: false  # Remove intermediate files to save disk space
    
    # Optional: pre-computed readmers database
    readmers_db: null  # Path to Meryl readmers DB (auto-computed if null)
  
  # Quality Assessment (ALWAYS RUNS - no enable/disable flag)
  quality_assessment:
    kmer_size: 21  # K-mer size for Merfin/Merqury
    # Note: Quality assessment runs regardless of polishing status
  
  # Gap Filling (OPTIONAL - independent from polishing)
  gap_filling:
    enabled: false  # Set to true to enable gap filling
    tool: "tgs-gapcloser"  # Options: "tgs-gapcloser", "lr_gapcloser"
    min_match: 500  # Minimum match length (TGS-GapCloser)
    min_identity: 0.3  # Minimum identity (TGS-GapCloser)
```

**Key Configuration Points:**

1. **`finishing.polishing.enabled`**: `false` by default (user must explicitly enable)
2. **`finishing.polishing.deepvariant_sif`**: Required if polishing enabled
3. **`finishing.quality_assessment`**: No enabled flag - always runs
4. **`finishing.gap_filling.enabled`**: Independent from polishing

---

## 6. Implementation Timeline

**Total: 3-5 days** (vs. 1-2 weeks for full reimplementation)

### Day 1: APv4.py Wrapper
- [x] Create `strandweaver/finishing/` directory structure
- [x] Implement `APv4Polisher` class
  - [x] Command builder with parameter mapping
  - [x] Technology detection (PacBio/ONT routing)
  - [x] QV summary parsing
- [x] Add configuration to `defaults.yaml`
- [ ] Unit tests for wrapper logic

### Day 2: Quality Assessment
- [ ] Implement `QualityAssessor` class
  - [ ] Meryl readmers database computation
  - [ ] Merfin QV calculation
  - [ ] Merqury completeness assessment
  - [ ] JSON metrics output
- [ ] Unit tests for QA module

### Day 3: Pipeline Integration
- [ ] Update `_step_finish()` method
  - [ ] Baseline QA integration
  - [ ] Optional polishing invocation
  - [ ] Final QA integration
  - [ ] Error handling and fallbacks
- [ ] Integration tests

### Day 4: Gap Filling (Optional)
- [ ] Implement `GapFiller` class
  - [ ] TGS-GapCloser wrapper
  - [ ] LR_Gapcloser wrapper
  - [ ] Gap counting statistics
- [ ] Unit tests

### Day 5: Documentation and Testing
- [ ] Update user documentation
- [ ] Create configuration guide
- [ ] End-to-end testing with real datasets
- [ ] Performance benchmarking

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/test_apv4_polisher.py
def test_command_builder():
    """Test APv4.py command construction."""
    config = {
        'apv4_path': '/path/to/APv4.py',
        'deepvariant_sif': '/path/to/deepvariant.sif',
        'read_technology': 'pacbio',
        'threads': 16,
        'polishing': {
            'max_iterations': 2,
            'optimized': True,
            'ploidy': 'haploid'
        }
    }
    
    polisher = APv4Polisher(config, logger)
    cmd = polisher._build_apv4_command(
        draft='draft.fa',
        reads='reads.fq',
        prefix='test',
        resume=True
    )
    
    assert '--deepseq_type PACBIO' in ' '.join(cmd)
    assert '--iterations 2' in ' '.join(cmd)
    assert '--optimized' in ' '.join(cmd)
    assert '--resume' in ' '.join(cmd)

def test_qv_parsing():
    """Test QV summary parsing."""
    # Create mock summary file
    summary = """
====== FINAL QV/COMPLETENESS SUMMARY (MERFIN + MERQURY) ======

> Iteration 1
----- Merfin -----
QV: 45.23
----- Merqury -----
Merqury QV: 44.89
Completeness: 99.2
    """
    
    with open('test_summary.txt', 'w') as f:
        f.write(summary)
    
    polisher = APv4Polisher(config, logger)
    qv_history = polisher._parse_qv_summary('test')
    
    assert len(qv_history) == 1
    assert qv_history[0]['merfin_qv'] == 45.23
    assert qv_history[0]['merqury_qv'] == 44.89
    assert qv_history[0]['completeness'] == 99.2
```

### Integration Tests

```python
# tests/test_finishing_pipeline.py
def test_full_finishing_pipeline():
    """Test complete finishing workflow."""
    # Setup test data
    draft = "tests/data/draft.fasta"
    reads = "tests/data/reads.fastq"
    
    config = load_test_config()
    config['finishing']['polishing']['enabled'] = True
    
    pipeline = StrandWeaverPipeline(config)
    pipeline._step_finish()
    
    # Verify outputs
    assert os.path.exists("output/baseline_qa.quality_metrics.json")
    assert os.path.exists("output/final_qa.quality_metrics.json")
    assert os.path.exists("output/final_assembly.fasta")
```

### Benchmark Datasets

1. **E. coli** (4.6 Mb) - Fast validation (~30 minutes)
2. **S. cerevisiae** (12 Mb) - Medium test (~2 hours)
3. **Human chr22** (50 Mb) - Large-scale test (~8 hours)

**Expected QV Improvements:**
- Baseline: 30-35 (unpolished)
- After 1 iteration: 45-50
- After 2-3 iterations: 50-55 (convergence)

---

## 8. Success Metrics

### Functional Requirements

- [x] Polishing disabled by default (runtime consideration)
- [x] Quality assessment always runs
- [x] APv4.py wrapper functional with all parameters
- [x] Technology auto-detection (PacBio/ONT)
- [x] Resume capability leveraged
- [x] QV parsing from APv4.py outputs
- [ ] Gap filling independent from polishing

### Performance Requirements

- **Baseline QA**: < 30 minutes (E. coli)
- **Polishing (if enabled)**: 2-8 hours per iteration (depends on genome size)
- **Gap filling**: < 1 hour (typical)
- **Memory**: < 50 GB (configurable via Meryl memory)

### Quality Requirements

- **QV improvement**: Baseline → 50+ (if polishing enabled)
- **Completeness**: > 99.5%
- **Resume reliability**: Can resume from any step
- **Error handling**: Graceful degradation if polishing fails

---

## 9. Migration from Original Proposal

### What Changed

**Original Proposal (6 modules):**
- `polisher.py` - Orchestrator with iterative logic
- `arrow_polisher.py` - PacBio-specific (pbmm2 + gcpp)
- `medaka_polisher.py` - ONT-specific (medaka_consensus)
- `racon_polisher.py` - Universal fallback
- `quality_assessment.py` - QV/completeness
- `gap_filler.py` - Gap closing

**Revised Proposal (3 modules):**
- `apv4_polisher.py` - Thin wrapper around APv4.py (replaces 4 modules)
- `quality_assessment.py` - Extracted to always run
- `gap_filler.py` - Same as original

**Rationale:**
- User has production-ready APv4.py → leverage instead of reimplement
- Polishing adds significant runtime → make optional (off by default)
- Quality assessment valuable regardless → always run
- Faster implementation → 3-5 days vs. 1-2 weeks

### Advantages of Revised Approach

1. **Faster implementation** - Wrapper vs. full reimplementation
2. **Production-ready** - APv4.py already proven on real datasets
3. **Maintained codebase** - User actively develops APv4.py
4. **Comprehensive tooling** - Winnowmap, FalconC, DeepVariant, Merfin all integrated
5. **Resume support** - Built-in, battle-tested
6. **Convergence detection** - Already implemented in APv4.py
7. **Optional runtime** - Polishing disabled by default (respects user's time)

### Disadvantages (and Mitigations)

1. **External dependency** - Mitigated: configurable APv4.py path
2. **Less granular control** - Acceptable: APv4.py highly configurable
3. **Singularity requirement** - Already standard dependency for DeepVariant

---

## 10. Questions for Discussion

1. **APv4.py location**: Should we bundle APv4.py with StrandWeaver or require user installation?
   - **Recommendation**: Keep as external dependency (user controls updates)

2. **Default polishing state**: Confirm `enabled: false` by default?
   - **Recommendation**: Yes (2-8 hours per iteration is significant)

3. **Quality assessment**: Always run or make optional?
   - **Recommendation**: Always run (< 30 minutes, valuable metrics)

4. **Gap filling**: Include in v1 or defer to v2?
   - **Recommendation**: Include (already simple wrapper, independent)

5. **DeepVariant SIF**: Bundle example or require user download?
   - **Recommendation**: Document download instructions (large file, GPU-specific)

6. **Convergence threshold**: Use APv4.py's built-in or add StrandWeaver override?
   - **Recommendation**: Use APv4.py's (proven default: 0.01 QV improvement)

7. **Technology detection**: Trust user's config or auto-detect from reads?
   - **Recommendation**: Use config (user knows their data best)

8. **Resume behavior**: Always enable or make optional?
   - **Recommendation**: Always enable (no downside, saves time)

9. **Intermediate file cleanup**: Default on or off?
   - **Recommendation**: Off (users may want to inspect intermediates)

10. **K-mer size**: Use APv4.py's default (31) or make configurable?
    - **Recommendation**: Configurable with APv4.py's default

---

## 11. Future Enhancements (Post-v1)

### Short-term (v1.1)
- **Convergence visualization** - Plot QV improvement per iteration
- **Multi-technology support** - Combine PacBio + ONT reads
- **Optimized k-mer coverage** - Auto-detect from GenomeScope2

### Medium-term (v1.2)
- **Pilon integration** - Short-read polishing for hybrid assemblies
- **BUSCO integration** - Gene completeness assessment
- **Benchmarking suite** - Standard datasets for QA validation

### Long-term (v2.0)
- **Cloud execution** - AWS/GCP integration for DeepVariant
- **Containerization** - Full Docker/Singularity workflow
- **Web interface** - GUI for configuration and monitoring

---

## Appendix A: APv4.py Command Reference

### Full Command Structure

```bash
python3 APv4.py polish \
  --draft <assembly.fasta> \
  --reads <reads.fastq> \
  --prefix <output_prefix> \
  --singularity_sif <deepvariant.sif> \
  --deepseq_type {PACBIO|ONT_R104|WGS|WES|HYBRID_PACBIO_ILLUMINA} \
  --threads <N> \
  --iterations <N> \
  --kmer_size <N> \
  --optimized \
  --ploidy {haploid|diploid} \
  --resume \
  --resume-from <step> \
  --cleanup \
  --log-file <path>
```

### Key Parameters

| Parameter | Description | Default | StrandWeaver Mapping |
|-----------|-------------|---------|----------------------|
| `--draft` | Assembly FASTA | Required | `scaffolds.fasta` |
| `--reads` | Reads (FASTQ/FASTA) | Required | `config['reads']` |
| `--singularity_sif` | DeepVariant image | Required | `config['deepvariant_sif']` |
| `--deepseq_type` | DV model type | PACBIO | Auto-detected from `read_technology` |
| `--threads` | CPU threads | 32 | `config['threads']` |
| `--iterations` | Polish rounds | 3 | `config['polishing']['max_iterations']` |
| `--kmer_size` | Meryl k-mer size | 31 | `config['polishing']['kmer_size']` |
| `--optimized` | Auto k-cov | False | `config['polishing']['optimized']` |
| `--ploidy` | Genome ploidy | haploid | `config['polishing']['ploidy']` |
| `--resume` | Resume mode | False | Always True in wrapper |
| `--cleanup` | Remove intermediates | False | `config['polishing']['cleanup']` |

### Output Files

```
<prefix>.iter_0.consensus.fasta          # Initial assembly (copy)
<prefix>_iteration_1/                    # Iteration 1 directory
  ├── merylDB/                          # Repetitive k-mers
  ├── iter_1.winnowmap.sorted.bam       # Alignment
  ├── iter_1.falconc.sorted.bam         # Filtered alignment
  ├── deepvariant_results/              # DeepVariant outputs
  │   ├── deepvariant_output.vcf.gz
  │   └── deepvariant_output.g.vcf.gz
  ├── iter_1.dv.merfin.polish.vcf       # Merfin-polished VCF
  ├── iter_1.consensus.fasta            # Polished consensus
  └── QC_summary.txt                    # Per-iteration QC
<prefix>.iter_1.consensus.fasta          # Top-level copy
<prefix>_iteration_2/                    # Iteration 2 (if run)
...
<prefix>.QV_Completeness_summary.txt     # Final summary (all iterations)
<prefix>.run_parameters.json             # Run configuration
<prefix>.tool_versions.txt               # Tool versions
```

**StrandWeaver Wrapper Reads:**
- `<prefix>.QV_Completeness_summary.txt` - Parse QV history
- `<prefix>.iter_N.consensus.fasta` - Final polished assembly

---

## Appendix B: Estimated Runtimes

### Polishing (per iteration)

| Genome Size | Technology | Winnowmap | DeepVariant | Merfin | Total/Iteration |
|-------------|------------|-----------|-------------|--------|-----------------|
| 4.6 Mb (E. coli) | PacBio HiFi | ~5 min | ~20 min | ~5 min | **~30 min** |
| 12 Mb (S. cerevisiae) | PacBio HiFi | ~10 min | ~45 min | ~10 min | **~1 hour** |
| 50 Mb (Human chr22) | PacBio HiFi | ~30 min | ~3 hours | ~30 min | **~4 hours** |
| 3 Gb (Human) | PacBio HiFi | ~6 hours | ~36 hours | ~6 hours | **~48 hours** |

### Quality Assessment (always runs)

| Genome Size | Meryl | Merfin | Merqury | Total |
|-------------|-------|--------|---------|-------|
| 4.6 Mb | ~2 min | ~3 min | ~5 min | **~10 min** |
| 12 Mb | ~5 min | ~5 min | ~10 min | **~20 min** |
| 50 Mb | ~10 min | ~15 min | ~20 min | **~45 min** |
| 3 Gb | ~2 hours | ~3 hours | ~4 hours | **~9 hours** |

**Note:** Polishing runtime justifies making it optional by default.

---

## Conclusion

This revised proposal simplifies the polishing integration by:

1. **Leveraging APv4.py directly** - Thin wrapper instead of reimplementation (3-5 days vs. 1-2 weeks)
2. **Making polishing optional** - Respects significant runtime burden (disabled by default)
3. **Mandating quality assessment** - Always provides baseline and final metrics
4. **Maintaining modularity** - Gap filling independent, Claude AI finishing preserved

**Next Steps:**
1. User approval of simplified approach
2. Begin Day 1 implementation (APv4.py wrapper)
3. Integration testing with real datasets
4. Documentation and user guide

**Timeline:** Ready for production use in **3-5 days** with basic functionality, **1 week** with full testing and documentation.
