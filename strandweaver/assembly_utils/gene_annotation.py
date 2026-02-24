#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Gene annotation — BUSCO and gene model integration for QC.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


# ============================================================================
#                         DATA STRUCTURES
# ============================================================================

@dataclass
class BlastHit:
    """BLAST alignment hit."""
    query_id: str
    subject_id: str
    identity: float
    alignment_length: int
    evalue: float
    bitscore: float
    query_start: int
    query_end: int
    subject_start: int
    subject_end: int


@dataclass
class Gene:
    """Predicted gene."""
    gene_id: str
    start: int
    end: int
    strand: str
    score: float
    gene_type: str = 'CDS'  # 'CDS', 'tRNA', 'rRNA'


@dataclass
class BUSCOResult:
    """BUSCO completeness assessment."""
    complete: int
    complete_single: int
    complete_duplicated: int
    fragmented: int
    missing: int
    total: int
    
    @property
    def completeness_percent(self) -> float:
        return (self.complete / self.total * 100) if self.total > 0 else 0.0


# ============================================================================
#                         BLAST WRAPPER
# ============================================================================

class BlastAnnotator:
    """
    BLAST-based homology search for gene detection.
    
    Uses NCBI BLAST (blastn or blastx) against protein/nucleotide databases
    to identify coding sequences via homology.
    """
    
    def __init__(
        self,
        blast_type: str = 'blastx',
        database: str = 'nr',
        num_threads: int = 4,
        evalue_threshold: float = 1e-5,
        max_target_seqs: int = 5
    ):
        """
        Initialize BLAST annotator.
        
        Args:
            blast_type: 'blastn' (nucleotide) or 'blastx' (translated)
            database: Path to BLAST database or name (e.g., 'nr', 'swissprot')
            num_threads: Number of threads for BLAST
            evalue_threshold: Maximum e-value for hits
            max_target_seqs: Maximum number of hits per query
        """
        self.blast_type = blast_type
        self.database = database
        self.num_threads = num_threads
        self.evalue_threshold = evalue_threshold
        self.max_target_seqs = max_target_seqs
        self.logger = logging.getLogger(f"{__name__}.BlastAnnotator")
    
    def search(self, sequence: str, scaffold_id: str = "query") -> List[BlastHit]:
        """
        Run BLAST search on sequence.
        
        Args:
            sequence: DNA sequence to search
            scaffold_id: Identifier for sequence
        
        Returns:
            List of BlastHit objects
        """
        # Check if BLAST is available
        if not self._check_blast_available():
            self.logger.warning(f"{self.blast_type} not found in PATH")
            return []
        
        # Write sequence to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">{scaffold_id}\n{sequence}\n")
            query_file = f.name
        
        try:
            # Run BLAST
            self.logger.debug(f"Running {self.blast_type} on {scaffold_id} ({len(sequence):,} bp)")
            
            output_file = query_file + '.blast.out'
            
            cmd = [
                self.blast_type,
                '-query', query_file,
                '-db', self.database,
                '-outfmt', '6',  # Tabular format
                '-evalue', str(self.evalue_threshold),
                '-max_target_seqs', str(self.max_target_seqs),
                '-num_threads', str(self.num_threads),
                '-out', output_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"BLAST failed: {result.stderr}")
                return []
            
            # Parse results
            hits = self._parse_blast_output(output_file)
            
            self.logger.debug(f"Found {len(hits)} BLAST hits")
            
            return hits
        
        except subprocess.TimeoutExpired:
            self.logger.warning("BLAST search timed out")
            return []
        
        except Exception as e:
            self.logger.error(f"BLAST error: {e}")
            return []
        
        finally:
            # Cleanup
            for file in [query_file, output_file]:
                if Path(file).exists():
                    Path(file).unlink()
    
    def _check_blast_available(self) -> bool:
        """Check if BLAST is installed."""
        try:
            result = subprocess.run(
                [self.blast_type, '-version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _parse_blast_output(self, output_file: str) -> List[BlastHit]:
        """
        Parse BLAST tabular output.
        
        Format: qseqid sseqid pident length evalue bitscore qstart qend sstart send
        """
        hits = []
        
        if not Path(output_file).exists():
            return hits
        
        with open(output_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                
                try:
                    hit = BlastHit(
                        query_id=fields[0],
                        subject_id=fields[1],
                        identity=float(fields[2]),
                        alignment_length=int(fields[3]),
                        evalue=float(fields[4]),
                        bitscore=float(fields[5]),
                        query_start=int(fields[6]),
                        query_end=int(fields[7]),
                        subject_start=int(fields[8]),
                        subject_end=int(fields[9])
                    )
                    hits.append(hit)
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Skipping malformed BLAST line: {e}")
                    continue
        
        return hits


# ============================================================================
#                         AUGUSTUS WRAPPER
# ============================================================================

class AugustusPredictor:
    """
    Augustus ab initio gene predictor wrapper.
    
    Predicts genes based on sequence content alone (no homology).
    """
    
    def __init__(
        self,
        species: str = 'human',
        min_gene_length: int = 300
    ):
        """
        Initialize Augustus predictor.
        
        Args:
            species: Species model to use (e.g., 'human', 'fly', 'arabidopsis')
            min_gene_length: Minimum gene length to report
        """
        self.species = species
        self.min_gene_length = min_gene_length
        self.logger = logging.getLogger(f"{__name__}.AugustusPredictor")
    
    def predict_genes(self, sequence: str, scaffold_id: str = "query") -> List[Gene]:
        """
        Predict genes in sequence.
        
        Args:
            sequence: DNA sequence
            scaffold_id: Identifier
        
        Returns:
            List of Gene objects
        """
        if not self._check_augustus_available():
            self.logger.warning("Augustus not found in PATH")
            return []
        
        # Write sequence to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">{scaffold_id}\n{sequence}\n")
            query_file = f.name
        
        try:
            self.logger.debug(f"Running Augustus on {scaffold_id}")
            
            cmd = [
                'augustus',
                '--species=' + self.species,
                '--gff3=off',
                '--stopCodonExcludedFromCDS=false',
                query_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Augustus failed: {result.stderr}")
                return []
            
            # Parse output
            genes = self._parse_augustus_output(result.stdout)
            
            # Filter by length
            genes = [g for g in genes if (g.end - g.start) >= self.min_gene_length]
            
            self.logger.debug(f"Predicted {len(genes)} genes")
            
            return genes
        
        except subprocess.TimeoutExpired:
            self.logger.warning("Augustus timed out")
            return []
        
        except Exception as e:
            self.logger.error(f"Augustus error: {e}")
            return []
        
        finally:
            if Path(query_file).exists():
                Path(query_file).unlink()
    
    def _check_augustus_available(self) -> bool:
        """Check if Augustus is installed."""
        try:
            result = subprocess.run(
                ['augustus', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _parse_augustus_output(self, output: str) -> List[Gene]:
        """Parse Augustus output."""
        genes = []
        gene_counter = 1
        
        for line in output.split('\n'):
            if not line.strip() or line.startswith('#'):
                continue
            
            # Look for gene lines
            if '\tgene\t' in line or '\tCDS\t' in line:
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                try:
                    gene = Gene(
                        gene_id=f"gene_{gene_counter}",
                        start=int(fields[3]),
                        end=int(fields[4]),
                        strand=fields[6],
                        score=float(fields[5]) if fields[5] != '.' else 0.0,
                        gene_type='CDS'
                    )
                    genes.append(gene)
                    gene_counter += 1
                except (ValueError, IndexError):
                    continue
        
        return genes


# ============================================================================
#                         BUSCO WRAPPER
# ============================================================================

class BUSCOAnalyzer:
    """
    BUSCO (Benchmarking Universal Single-Copy Orthologs) wrapper.
    
    Assesses genome completeness by searching for conserved genes.
    """
    
    def __init__(
        self,
        lineage: str = 'auto',
        mode: str = 'genome',
        num_threads: int = 4
    ):
        """
        Initialize BUSCO analyzer.
        
        Args:
            lineage: BUSCO lineage dataset (e.g., 'vertebrata_odb10', 'auto')
            mode: BUSCO mode ('genome', 'transcriptome', 'proteins')
            num_threads: Number of threads
        """
        self.lineage = lineage
        self.mode = mode
        self.num_threads = num_threads
        self.logger = logging.getLogger(f"{__name__}.BUSCOAnalyzer")
    
    def analyze(self, sequence: str, scaffold_id: str = "query") -> Optional[BUSCOResult]:
        """
        Run BUSCO analysis on sequence.
        
        Args:
            sequence: DNA sequence
            scaffold_id: Identifier
        
        Returns:
            BUSCOResult or None if failed
        """
        if not self._check_busco_available():
            self.logger.warning("BUSCO not found in PATH")
            return None
        
        # Write sequence to file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            query_file = tmpdir / f"{scaffold_id}.fasta"
            
            with open(query_file, 'w') as f:
                f.write(f">{scaffold_id}\n{sequence}\n")
            
            try:
                self.logger.debug(f"Running BUSCO on {scaffold_id}")
                
                cmd = [
                    'busco',
                    '-i', str(query_file),
                    '-o', scaffold_id,
                    '-m', self.mode,
                    '-c', str(self.num_threads),
                    '--out_path', str(tmpdir)
                ]
                
                if self.lineage != 'auto':
                    cmd.extend(['-l', self.lineage])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout
                    cwd=str(tmpdir)
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"BUSCO failed: {result.stderr}")
                    return None
                
                # Parse summary
                summary_file = tmpdir / scaffold_id / 'short_summary*.txt'
                summary_files = list(tmpdir.glob(f"{scaffold_id}/short_summary*.txt"))
                
                if not summary_files:
                    self.logger.warning("BUSCO summary file not found")
                    return None
                
                busco_result = self._parse_busco_summary(summary_files[0])
                
                return busco_result
            
            except subprocess.TimeoutExpired:
                self.logger.warning("BUSCO timed out")
                return None
            
            except Exception as e:
                self.logger.error(f"BUSCO error: {e}")
                return None
    
    def _check_busco_available(self) -> bool:
        """Check if BUSCO is installed."""
        try:
            result = subprocess.run(
                ['busco', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _parse_busco_summary(self, summary_file: Path) -> Optional[BUSCOResult]:
        """Parse BUSCO summary file."""
        with open(summary_file) as f:
            content = f.read()
        
        # Extract numbers using regex
        complete_match = re.search(r'C:(\d+\.\d+)%', content)
        single_match = re.search(r'S:(\d+\.\d+)%', content)
        duplicated_match = re.search(r'D:(\d+\.\d+)%', content)
        fragmented_match = re.search(r'F:(\d+\.\d+)%', content)
        missing_match = re.search(r'M:(\d+\.\d+)%', content)
        total_match = re.search(r'Total BUSCO groups searched:\s*(\d+)', content)
        
        if not all([complete_match, fragmented_match, missing_match, total_match]):
            return None
        
        total = int(total_match.group(1))
        complete_pct = float(complete_match.group(1))
        single_pct = float(single_match.group(1)) if single_match else 0
        duplicated_pct = float(duplicated_match.group(1)) if duplicated_match else 0
        fragmented_pct = float(fragmented_match.group(1))
        missing_pct = float(missing_match.group(1))
        
        return BUSCOResult(
            complete=int(total * complete_pct / 100),
            complete_single=int(total * single_pct / 100),
            complete_duplicated=int(total * duplicated_pct / 100),
            fragmented=int(total * fragmented_pct / 100),
            missing=int(total * missing_pct / 100),
            total=total
        )


# ============================================================================
#                         SIMPLE ORF FINDER
# ============================================================================

def find_orfs(sequence: str, min_length: int = 300) -> List[Gene]:
    """
    Simple ORF finder (no external tools required).
    
    Finds open reading frames between start and stop codons.
    
    Args:
        sequence: DNA sequence
        min_length: Minimum ORF length (bp)
    
    Returns:
        List of Gene objects representing ORFs
    """
    start_codons = ['ATG']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    orfs = []
    orf_id = 1
    
    # Check all three forward frames
    for frame in range(3):
        i = frame
        while i < len(sequence) - 2:
            codon = sequence[i:i+3].upper()
            
            if codon in start_codons:
                # Found start, look for stop
                start = i
                j = i + 3
                
                while j < len(sequence) - 2:
                    stop_codon = sequence[j:j+3].upper()
                    
                    if stop_codon in stop_codons:
                        # Found stop
                        end = j + 3
                        orf_length = end - start
                        
                        if orf_length >= min_length:
                            orf = Gene(
                                gene_id=f"ORF_{orf_id}",
                                start=start,
                                end=end,
                                strand='+',
                                score=float(orf_length),
                                gene_type='ORF'
                            )
                            orfs.append(orf)
                            orf_id += 1
                        
                        i = j + 3
                        break
                    
                    j += 3
                else:
                    # No stop found
                    break
            
            i += 3
    
    # Check reverse complement
    rev_comp = reverse_complement(sequence)
    for frame in range(3):
        i = frame
        while i < len(rev_comp) - 2:
            codon = rev_comp[i:i+3].upper()
            
            if codon in start_codons:
                start = i
                j = i + 3
                
                while j < len(rev_comp) - 2:
                    stop_codon = rev_comp[j:j+3].upper()
                    
                    if stop_codon in stop_codons:
                        end = j + 3
                        orf_length = end - start
                        
                        if orf_length >= min_length:
                            # Convert coordinates back to forward strand
                            orf = Gene(
                                gene_id=f"ORF_{orf_id}",
                                start=len(sequence) - end,
                                end=len(sequence) - start,
                                strand='-',
                                score=float(orf_length),
                                gene_type='ORF'
                            )
                            orfs.append(orf)
                            orf_id += 1
                        
                        i = j + 3
                        break
                    
                    j += 3
                else:
                    break
            
            i += 3
    
    return orfs


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base.upper(), 'N') for base in reversed(sequence))


__all__ = [
    'BlastAnnotator',
    'AugustusPredictor',
    'BUSCOAnalyzer',
    'BlastHit',
    'Gene',
    'BUSCOResult',
    'find_orfs',
    'reverse_complement',
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
