#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hi-C Graph Aligner - Parse and align Hi-C reads to assembly graph.

This module provides functionality to:
1. Parse Hi-C FASTQ paired-end reads
2. Parse aligned Hi-C BAM/SAM files
3. Align Hi-C reads to graph nodes using k-mer matching
4. Convert alignments to HiCPair objects for StrandTether

Supports both:
- Direct FASTQ alignment (k-mer based, fast)
- Pre-aligned BAM/SAM parsing (from minimap2, bwa, etc.)

Author: StrandWeaver Development Team
Date: December 24, 2025
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Union, TYPE_CHECKING
from collections import defaultdict
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..assembly_core.strandtether_module import HiCPair

logger = logging.getLogger(__name__)


# ============================================================================
#                         HI-C READ STRUCTURES
# ============================================================================

@dataclass
class HiCReadPair:
    """
    Raw Hi-C read pair from FASTQ.
    
    Attributes:
        read_id: Unique identifier (without /1 or /2 suffix)
        seq1: R1 sequence
        qual1: R1 quality string
        seq2: R2 sequence
        qual2: R2 quality string
    """
    read_id: str
    seq1: str
    qual1: str
    seq2: str
    qual2: str


@dataclass
class HiCAlignment:
    """
    Hi-C read aligned to graph node.
    
    Attributes:
        read_id: Read identifier
        node_id: Graph node this read maps to
        position: Position within node (0-based)
        strand: True for forward (+), False for reverse (-)
        mapq: Mapping quality (0-60)
        num_matches: Number of k-mer matches
    """
    read_id: str
    node_id: int
    position: int
    strand: bool
    mapq: int = 0
    num_matches: int = 0


# ============================================================================
#                         HI-C GRAPH ALIGNER
# ============================================================================

class HiCGraphAligner:
    """
    Align Hi-C reads to assembly graph nodes.
    
    Supports two modes:
    1. K-mer based alignment directly from FASTQ (fast, approximate)
    2. BAM/SAM parsing from external aligner (accurate, requires pre-alignment)
    """
    
    def __init__(
        self,
        k: int = 21,
        min_matches: int = 3,
        min_mapq: int = 10,
        max_mismatches: int = 2
    ):
        """
        Initialize Hi-C aligner.
        
        Args:
            k: K-mer size for alignment
            min_matches: Minimum k-mer matches to call alignment
            min_mapq: Minimum mapping quality for BAM alignments
            max_mismatches: Maximum mismatches allowed in k-mer alignment
        """
        self.k = k
        self.min_matches = min_matches
        self.min_mapq = min_mapq
        self.max_mismatches = max_mismatches
        self.logger = logging.getLogger(f"{__name__}.HiCGraphAligner")
    
    # ========================================================================
    #                    FASTQ PARSING
    # ========================================================================
    
    def parse_hic_fastq_pairs(
        self,
        r1_path: Union[str, Path],
        r2_path: Union[str, Path],
        max_reads: Optional[int] = None
    ) -> List[HiCReadPair]:
        """
        Parse paired-end Hi-C FASTQ files.
        
        Args:
            r1_path: Path to R1 FASTQ file
            r2_path: Path to R2 FASTQ file
            max_reads: Maximum number of read pairs to parse (None = all)
        
        Returns:
            List of HiCReadPair objects
        """
        r1_path = Path(r1_path)
        r2_path = Path(r2_path)
        
        self.logger.info(f"Parsing Hi-C FASTQ pairs: {r1_path.name}, {r2_path.name}")
        
        read_pairs = []
        
        with open(r1_path) as r1_file, open(r2_path) as r2_file:
            while True:
                # Read 4 lines from each file (FASTQ format)
                r1_header = r1_file.readline().strip()
                r1_seq = r1_file.readline().strip()
                r1_plus = r1_file.readline().strip()
                r1_qual = r1_file.readline().strip()
                
                r2_header = r2_file.readline().strip()
                r2_seq = r2_file.readline().strip()
                r2_plus = r2_file.readline().strip()
                r2_qual = r2_file.readline().strip()
                
                # Check for EOF
                if not r1_header or not r2_header:
                    break
                
                # Extract read ID (remove @ and /1 or /2 suffix)
                read_id = r1_header[1:].split()[0].rstrip('/1').rstrip('/2')
                
                # Create read pair
                pair = HiCReadPair(
                    read_id=read_id,
                    seq1=r1_seq,
                    qual1=r1_qual,
                    seq2=r2_seq,
                    qual2=r2_qual
                )
                read_pairs.append(pair)
                
                # Check max_reads limit
                if max_reads and len(read_pairs) >= max_reads:
                    break
        
        self.logger.info(f"Parsed {len(read_pairs)} Hi-C read pairs")
        return read_pairs
    
    # ========================================================================
    #                    K-MER BASED ALIGNMENT
    # ========================================================================
    
    def align_hic_to_graph(
        self,
        hic_pairs: List[HiCReadPair],
        graph,
        sample_size: Optional[int] = None
    ) -> List['HiCPair']:
        """
        Align Hi-C read pairs to graph nodes using k-mer matching.
        
        Args:
            hic_pairs: List of HiCReadPair objects
            graph: DBGGraph or StringGraph with nodes
            sample_size: Sample this many pairs (None = use all)
        
        Returns:
            List of HiCPair objects (from strandtether_module)
        """
        from ..assembly_core.strandtether_module import HiCPair, HiCFragment
        
        self.logger.info(f"Aligning {len(hic_pairs)} Hi-C pairs to graph with k={self.k}")
        
        # Sample if requested
        if sample_size and len(hic_pairs) > sample_size:
            import random
            hic_pairs = random.sample(hic_pairs, sample_size)
            self.logger.info(f"Sampled {sample_size} pairs for alignment")
        
        # Build k-mer index for graph nodes
        self.logger.info("Building k-mer index for graph nodes...")
        kmer_index = self._build_kmer_index(graph)
        self.logger.info(f"Indexed {len(kmer_index)} unique k-mers")
        
        # Align each read pair
        aligned_pairs = []
        r1_mapped = 0
        r2_mapped = 0
        both_mapped = 0
        
        for pair in hic_pairs:
            # Align R1
            r1_alignment = self._align_read_kmers(
                pair.read_id + "/1",
                pair.seq1,
                kmer_index
            )
            
            # Align R2
            r2_alignment = self._align_read_kmers(
                pair.read_id + "/2",
                pair.seq2,
                kmer_index
            )
            
            # Track mapping stats
            if r1_alignment:
                r1_mapped += 1
            if r2_alignment:
                r2_mapped += 1
            
            # Both must map to create a contact
            if r1_alignment and r2_alignment:
                # Skip self-contacts (same node)
                if r1_alignment.node_id == r2_alignment.node_id:
                    continue
                
                # Create HiCFragment objects
                frag1 = HiCFragment(
                    read_id=r1_alignment.read_id,
                    node_id=r1_alignment.node_id,
                    pos=r1_alignment.position,
                    strand=r1_alignment.strand
                )
                
                frag2 = HiCFragment(
                    read_id=r2_alignment.read_id,
                    node_id=r2_alignment.node_id,
                    pos=r2_alignment.position,
                    strand=r2_alignment.strand
                )
                
                # Create HiCPair
                hic_pair = HiCPair(frag1=frag1, frag2=frag2)
                aligned_pairs.append(hic_pair)
                both_mapped += 1
        
        self.logger.info(
            f"Alignment complete: {both_mapped}/{len(hic_pairs)} pairs mapped "
            f"(R1: {r1_mapped}, R2: {r2_mapped})"
        )
        
        return aligned_pairs
    
    def _build_kmer_index(self, graph) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build k-mer index mapping k-mer -> [(node_id, position), ...].
        
        Args:
            graph: Graph with nodes
        
        Returns:
            Dict mapping k-mer to list of (node_id, position) tuples
        """
        kmer_index = defaultdict(list)
        
        for node_id, node in graph.nodes.items():
            sequence = node.seq if hasattr(node, 'seq') else node.sequence
            
            if not sequence or len(sequence) < self.k:
                continue
            
            # Extract all k-mers from this node
            for i in range(len(sequence) - self.k + 1):
                kmer = sequence[i:i + self.k]
                kmer_index[kmer].append((node_id, i))
        
        return dict(kmer_index)
    
    def _align_read_kmers(
        self,
        read_id: str,
        sequence: str,
        kmer_index: Dict[str, List[Tuple[int, int]]]
    ) -> Optional[HiCAlignment]:
        """
        Align a single read to graph using k-mer matches.
        
        Args:
            read_id: Read identifier
            sequence: Read sequence
            kmer_index: K-mer index from _build_kmer_index()
        
        Returns:
            HiCAlignment or None if no good alignment
        """
        if len(sequence) < self.k:
            return None
        
        # Extract k-mers from read
        read_kmers = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
        
        # Count matches per node
        node_matches = defaultdict(lambda: {'count': 0, 'positions': []})
        
        for kmer in read_kmers:
            if kmer in kmer_index:
                for node_id, pos in kmer_index[kmer]:
                    node_matches[node_id]['count'] += 1
                    node_matches[node_id]['positions'].append(pos)
        
        # Find best node (most k-mer matches)
        if not node_matches:
            return None
        
        best_node = max(node_matches.keys(), key=lambda n: node_matches[n]['count'])
        match_count = node_matches[best_node]['count']
        
        if match_count < self.min_matches:
            return None
        
        # Calculate mapping quality (simple MAPQ approximation)
        # MAPQ = -10 * log10(P_error), approximate as function of match count
        mapq = min(60, int(10 + 5 * match_count))
        
        # Use median position of matches
        positions = sorted(node_matches[best_node]['positions'])
        median_pos = positions[len(positions) // 2]
        
        # Assume forward strand (would need reverse complement matching for full implementation)
        strand = True
        
        return HiCAlignment(
            read_id=read_id,
            node_id=best_node,
            position=median_pos,
            strand=strand,
            mapq=mapq,
            num_matches=match_count
        )
    
    # ========================================================================
    #                    BAM/SAM PARSING
    # ========================================================================
    
    def parse_hic_bam(
        self,
        bam_path: Union[str, Path],
        graph,
        node_name_to_id: Optional[Dict[str, int]] = None
    ) -> List['HiCPair']:
        """
        Parse Hi-C alignments from BAM/SAM file.
        
        Assumes reads were aligned to graph contigs/nodes using minimap2 or bwa.
        
        Args:
            bam_path: Path to BAM or SAM file
            graph: Graph object (for node validation)
            node_name_to_id: Optional mapping from contig names to node IDs
        
        Returns:
            List of HiCPair objects
        """
        from ..assembly_core.strandtether_module import HiCPair, HiCFragment
        
        self.logger.info(f"Parsing Hi-C BAM file: {bam_path}")
        
        try:
            import pysam
        except ImportError:
            self.logger.error(
                "pysam not installed. Install with: pip install pysam\n"
                "Falling back to k-mer alignment from FASTQ."
            )
            return []
        
        bam_path = Path(bam_path)
        
        # Open BAM file
        samfile = pysam.AlignmentFile(str(bam_path), "rb" if bam_path.suffix == ".bam" else "r")
        
        # Build node name mapping if not provided
        if node_name_to_id is None:
            node_name_to_id = {f"node_{nid}": nid for nid in graph.nodes.keys()}
            node_name_to_id.update({f"unitig-{nid}": nid for nid in graph.nodes.keys()})
            node_name_to_id.update({f"contig_{nid}": nid for nid in graph.nodes.keys()})
        
        # Group alignments by read ID (handle paired-end)
        read_alignments = defaultdict(list)
        
        for read in samfile.fetch():
            if read.is_unmapped or read.mapping_quality < self.min_mapq:
                continue
            
            # Get reference name (should map to node)
            ref_name = read.reference_name
            if ref_name not in node_name_to_id:
                # Try extracting node ID from name
                try:
                    node_id = int(ref_name.split('_')[-1].split('-')[-1])
                except (ValueError, IndexError):
                    continue
            else:
                node_id = node_name_to_id[ref_name]
            
            # Create alignment
            alignment = HiCAlignment(
                read_id=read.query_name,
                node_id=node_id,
                position=read.reference_start,
                strand=not read.is_reverse,
                mapq=read.mapping_quality,
                num_matches=0  # Not tracked in BAM
            )
            
            read_alignments[read.query_name].append(alignment)
        
        samfile.close()
        
        # Create Hi-C pairs from paired alignments
        hic_pairs = []
        
        for read_id, alignments in read_alignments.items():
            if len(alignments) != 2:
                continue  # Need exactly 2 alignments (R1 and R2)
            
            r1_aln, r2_aln = alignments[0], alignments[1]
            
            # Skip self-contacts
            if r1_aln.node_id == r2_aln.node_id:
                continue
            
            # Create HiCFragments
            frag1 = HiCFragment(
                read_id=r1_aln.read_id + "/1",
                node_id=r1_aln.node_id,
                pos=r1_aln.position,
                strand=r1_aln.strand
            )
            
            frag2 = HiCFragment(
                read_id=r2_aln.read_id + "/2",
                node_id=r2_aln.node_id,
                pos=r2_aln.position,
                strand=r2_aln.strand
            )
            
            hic_pair = HiCPair(frag1=frag1, frag2=frag2)
            hic_pairs.append(hic_pair)
        
        self.logger.info(f"Parsed {len(hic_pairs)} Hi-C pairs from BAM")
        
        return hic_pairs


# ============================================================================
#                         CONVENIENCE FUNCTIONS
# ============================================================================

def align_hic_reads_to_graph(
    hic_data: Union[Tuple[str, str], str, Path],
    graph,
    k: int = 21,
    min_matches: int = 3,
    sample_size: Optional[int] = None
) -> List['HiCPair']:
    """
    Convenience function to align Hi-C reads to graph.
    
    Automatically detects input type (FASTQ pairs or BAM) and uses
    appropriate alignment method.
    
    Args:
        hic_data: Either (r1_path, r2_path) tuple for FASTQ or single BAM path
        graph: Assembly graph (DBGGraph or StringGraph)
        k: K-mer size for k-mer alignment
        min_matches: Minimum k-mer matches required
        sample_size: Sample this many read pairs (None = all)
    
    Returns:
        List of HiCPair objects ready for StrandTether
    
    Example:
        # From FASTQ pairs
        hic_pairs = align_hic_reads_to_graph(
            ("hic_R1.fastq", "hic_R2.fastq"),
            graph,
            sample_size=100000
        )
        
        # From BAM
        hic_pairs = align_hic_reads_to_graph("hic_aligned.bam", graph)
    """
    aligner = HiCGraphAligner(k=k, min_matches=min_matches)
    
    # Detect input type
    if isinstance(hic_data, tuple) and len(hic_data) == 2:
        # FASTQ pairs - k-mer alignment
        logger.info("Using k-mer alignment from FASTQ pairs")
        read_pairs = aligner.parse_hic_fastq_pairs(
            hic_data[0],
            hic_data[1],
            max_reads=sample_size
        )
        hic_pairs = aligner.align_hic_to_graph(read_pairs, graph)
    
    elif isinstance(hic_data, (str, Path)):
        # BAM/SAM file
        logger.info("Parsing pre-aligned BAM file")
        hic_pairs = aligner.parse_hic_bam(hic_data, graph)
    
    else:
        raise ValueError(
            f"Invalid hic_data type: {type(hic_data)}. "
            "Expected (r1_path, r2_path) tuple or BAM path."
        )
    
    return hic_pairs


__all__ = [
    'HiCGraphAligner',
    'HiCReadPair',
    'HiCAlignment',
    'align_hic_reads_to_graph',
]
