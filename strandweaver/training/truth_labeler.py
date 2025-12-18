"""
Ground-Truth Labeler for ML Training

Extracts ground-truth labels from synthetic assemblies by comparing assembled
graphs against known-correct reference genomes and variant annotations.

This module bridges the gap between:
- INPUT: Simulated reads + reference genomes + SV truth tables
- OUTPUT: Labeled training examples for ML models

Key Operations:
1. Align reads to reference genomes (establish ground truth coordinates)
2. Build assembly graph from simulated reads
3. Align graph nodes back to reference
4. Label edges (true/repeat/chimeric/allelic/sv_break)
5. Label node haplotypes (A/B/both)
6. Label GNN paths (correct traversals)
7. Label UL routes (correct paths for ultralong reads)
8. Label SV features (which graph patterns = real SVs)

Author: StrandWeaver Training Infrastructure
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from pathlib import Path
import re

logger = logging.getLogger(__name__)


# ============================================================================
#                           LABEL TYPES
# ============================================================================

class EdgeLabel(Enum):
    """Ground-truth classification for graph edges."""
    TRUE = "true"                    # Legitimate overlap in reference
    REPEAT = "repeat"                # Both reads from same repeat region
    CHIMERIC = "chimeric"            # Reads from different loci/chromosomes
    ALLELIC = "allelic"              # Reads from homologous haplotype positions
    SV_BREAK = "sv_break"            # Edge crosses SV breakpoint
    UNKNOWN = "unknown"              # Cannot determine (shouldn't happen with synthetic)


class NodeHaplotype(Enum):
    """Haplotype assignment for graph nodes."""
    HAP_A = "A"                      # Node from haplotype A only
    HAP_B = "B"                      # Node from haplotype B only
    BOTH = "both"                    # Node shared by both haplotypes
    REPEAT = "repeat"                # Node from repeat region (ambiguous)
    UNKNOWN = "unknown"              # Cannot determine


class SVType(Enum):
    """Structural variant types (matches genome_simulator.py)."""
    DELETION = "deletion"
    INSERTION = "insertion"
    INVERSION = "inversion"
    DUPLICATION = "duplication"
    TRANSLOCATION = "translocation"


# ============================================================================
#                           DATA STRUCTURES
# ============================================================================

@dataclass
class ReadAlignment:
    """
    Alignment of a simulated read to reference genome.
    
    Attributes:
        read_id: Read identifier
        ref_chrom: Reference chromosome
        ref_start: Start position on reference
        ref_end: End position on reference
        haplotype: Source haplotype ('A' or 'B')
        strand: Alignment strand ('+' or '-')
        identity: Alignment identity (0-1)
        is_repeat: Whether read originates from repeat region
    """
    read_id: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    haplotype: str
    strand: str = '+'
    identity: float = 1.0
    is_repeat: bool = False


@dataclass
class EdgeGroundTruth:
    """
    Ground-truth label for a graph edge.
    
    Attributes:
        source_node: Source node ID
        target_node: Target node ID
        label: Edge classification
        explanation: Human-readable explanation
        read1_pos: Position of read 1 on reference
        read2_pos: Position of read 2 on reference
        overlap_distance: True distance between reads on reference
        crosses_sv: Whether edge crosses an SV breakpoint
        sv_type: Type of SV if crosses_sv is True
    """
    source_node: str
    target_node: str
    label: EdgeLabel
    explanation: str
    read1_pos: Optional[Tuple[str, int, int]] = None  # (chrom, start, end)
    read2_pos: Optional[Tuple[str, int, int]] = None
    overlap_distance: Optional[int] = None
    crosses_sv: bool = False
    sv_type: Optional[SVType] = None


@dataclass
class NodeGroundTruth:
    """
    Ground-truth label for a graph node.
    
    Attributes:
        node_id: Node identifier
        haplotype: Haplotype assignment
        ref_positions: List of reference positions this node maps to
        is_repeat: Whether node comes from repeat region
        spanning_reads: List of read IDs that contain this node
        sv_association: SV this node is associated with (if any)
    """
    node_id: str
    haplotype: NodeHaplotype
    ref_positions: List[Tuple[str, int, int]] = field(default_factory=list)
    is_repeat: bool = False
    spanning_reads: List[str] = field(default_factory=list)
    sv_association: Optional[Dict] = None


@dataclass
class PathGroundTruth:
    """
    Ground-truth path through the graph (for GNN training).
    
    Attributes:
        path_id: Path identifier
        node_sequence: Ordered list of node IDs
        haplotype: Which haplotype this path represents
        ref_chrom: Reference chromosome
        ref_start: Start position on reference
        ref_end: End position on reference
        is_correct: Whether this is a correct path
        confidence: Confidence score (0-1)
    """
    path_id: str
    node_sequence: List[str]
    haplotype: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    is_correct: bool = True
    confidence: float = 1.0


@dataclass
class ULRouteGroundTruth:
    """
    Ground-truth routing for ultralong reads.
    
    Attributes:
        read_id: UL read identifier
        correct_path: Correct node sequence for this read
        alternative_paths: Incorrect alternative paths
        ref_positions: Reference positions spanned by this read
        num_nodes: Number of nodes in correct path
        path_length: Total path length in bp
    """
    read_id: str
    correct_path: List[str]
    alternative_paths: List[List[str]] = field(default_factory=list)
    ref_positions: Tuple[str, int, int] = ("chr1", 0, 0)
    num_nodes: int = 0
    path_length: int = 0


@dataclass
class SVGroundTruth:
    """
    Ground-truth structural variant annotation.
    
    Attributes:
        sv_id: SV identifier
        sv_type: Type of structural variant
        haplotype: Which haplotype contains this SV
        ref_chrom: Reference chromosome
        ref_start: Start position
        ref_end: End position
        size: SV size in bp
        graph_signature: Graph features associated with this SV
    """
    sv_id: str
    sv_type: SVType
    haplotype: str
    ref_chrom: str
    ref_start: int
    ref_end: int
    size: int
    graph_signature: Dict = field(default_factory=dict)


# ============================================================================
#                    READ ALIGNMENT TO REFERENCE
# ============================================================================

def align_simulated_reads_to_reference(
    reads: List,  # List[SimulatedRead]
    reference_hapA: str,
    reference_hapB: str,
    min_identity: float = 0.90
) -> Dict[str, ReadAlignment]:
    """
    Align simulated reads back to their source reference genomes.
    
    Since reads are synthetic, we already know their true positions from
    SimulatedRead.start_pos and SimulatedRead.end_pos. This function just
    creates ReadAlignment objects with that information.
    
    Args:
        reads: List of SimulatedRead objects
        reference_hapA: Haplotype A reference sequence
        reference_hapB: Haplotype B reference sequence
        min_identity: Minimum alignment identity (not used for synthetic)
    
    Returns:
        Dictionary mapping read_id -> ReadAlignment
    """
    logger.info(f"Aligning {len(reads)} simulated reads to reference genomes...")
    
    alignments = {}
    
    for read in reads:
        # Simulated reads already have ground truth positions
        alignment = ReadAlignment(
            read_id=read.read_id,
            ref_chrom=read.chrom,
            ref_start=read.start_pos,
            ref_end=read.end_pos,
            haplotype=read.haplotype if hasattr(read, 'haplotype') else 'A',
            strand=read.strand if hasattr(read, 'strand') else '+',
            identity=1.0,  # Perfect alignment for synthetic reads
            is_repeat=False  # Will be determined later
        )
        
        alignments[read.read_id] = alignment
    
    logger.info(f"Created {len(alignments)} read alignments")
    return alignments


# ============================================================================
#                    EDGE LABELING
# ============================================================================

def label_graph_edge(
    source_node_id: str,
    target_node_id: str,
    alignments: Dict[str, ReadAlignment],
    sv_truth_table: List,
    node_to_read_ids: Optional[Dict[str, List[str]]] = None,
    max_true_distance: int = 10000,
    repeat_threshold: int = 3
) -> EdgeGroundTruth:
    """
    Determine ground-truth label for a graph edge.
    
    Classification logic:
    1. TRUE: Both reads from same haplotype, overlapping/adjacent positions
    2. ALLELIC: Reads from different haplotypes at homologous positions
    3. REPEAT: Both reads map to known repeat regions
    4. SV_BREAK: Edge crosses an SV breakpoint
    5. CHIMERIC: Reads from distant loci (shouldn't happen in good assembly)
    
    Args:
        source_read_id: Source read identifier
        target_read_id: Target read identifier
        alignments: Dictionary of read alignments
        sv_truth_table: List of StructuralVariant objects
        max_true_distance: Max distance for "true" overlap (bp)
        repeat_threshold: Number of alignments to consider "repeat"
    
    Returns:
        EdgeGroundTruth object with classification
    """
    # Get alignments - try exact match first, then fuzzy match
    # Try direct read-id lookup, but nodes may be unitigs. Prefer node_to_read_ids mapping.
    aln1 = None
    aln2 = None
    
    # Fuzzy matching: node IDs might have suffixes like "_L", "_R", etc.
    # Resolve via mapping first
    if node_to_read_ids:
        for rid in node_to_read_ids.get(source_node_id, []):
            if rid in alignments:
                aln1 = alignments[rid]
                break
    
    if node_to_read_ids:
        for rid in node_to_read_ids.get(target_node_id, []):
            if rid in alignments:
                aln2 = alignments[rid]
                break

    # If still missing, try resolving via node_to_read_ids mapping
    # Fallback fuzzy match on alignment keys
    if not aln1:
        for read_id in alignments.keys():
            if read_id in source_node_id or source_node_id in read_id:
                aln1 = alignments[read_id]
                break
    if not aln2:
        for read_id in alignments.keys():
            if read_id in target_node_id or target_node_id in read_id:
                aln2 = alignments[read_id]
                break
    
    if not aln1 or not aln2:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.UNKNOWN,
            explanation="Missing alignment data for node-derived IDs"
        )
    
    # Same chromosome?
    if aln1.ref_chrom != aln2.ref_chrom:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.CHIMERIC,
            explanation=f"Reads from different chromosomes: {aln1.ref_chrom} vs {aln2.ref_chrom}",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end)
        )
    
    # Calculate distance between reads
    if aln1.ref_start <= aln2.ref_start:
        distance = aln2.ref_start - aln1.ref_end
    else:
        distance = aln1.ref_start - aln2.ref_end
    
    # Check if edge crosses SV breakpoint
    crosses_sv = False
    sv_type = None
    for sv in sv_truth_table:
        if sv.chrom != aln1.ref_chrom:
            continue
        
        # Check if edge spans SV breakpoint
        sv_start = sv.pos
        sv_end = sv.end
        
        read1_span = (aln1.ref_start, aln1.ref_end)
        read2_span = (aln2.ref_start, aln2.ref_end)
        
        # Does edge cross this SV?
        if (read1_span[0] < sv_start < read2_span[1] or
            read2_span[0] < sv_start < read1_span[1] or
            read1_span[0] < sv_end < read2_span[1] or
            read2_span[0] < sv_end < read1_span[1]):
            crosses_sv = True
            sv_type = SVType(sv.sv_type.value)
            break
    
    if crosses_sv:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.SV_BREAK,
            explanation=f"Edge crosses {sv_type.value} SV breakpoint",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance,
            crosses_sv=True,
            sv_type=sv_type
        )
    
    # Check if both reads are repeats
    if aln1.is_repeat and aln2.is_repeat:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.REPEAT,
            explanation="Both reads from repeat regions",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance
        )
    
    # Different haplotypes?
    if aln1.haplotype != aln2.haplotype:
        # Check if reads are at homologous positions (similar coordinates)
        if abs(aln1.ref_start - aln2.ref_start) < max_true_distance:
            return EdgeGroundTruth(
                source_node=source_node_id,
                target_node=target_node_id,
                label=EdgeLabel.ALLELIC,
                explanation=f"Reads from different haplotypes ({aln1.haplotype} vs {aln2.haplotype}) at homologous positions",
                read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
                read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
                overlap_distance=distance
            )
        else:
            return EdgeGroundTruth(
                source_node=source_node_id,
                target_node=target_node_id,
                label=EdgeLabel.CHIMERIC,
                explanation=f"Reads from different haplotypes at distant positions",
                read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
                read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
                overlap_distance=distance
            )
    
    # Same haplotype, close distance = TRUE overlap
    if abs(distance) <= max_true_distance:
        return EdgeGroundTruth(
            source_node=source_node_id,
            target_node=target_node_id,
            label=EdgeLabel.TRUE,
            explanation=f"Legitimate overlap (distance={distance}bp, same haplotype)",
            read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
            read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
            overlap_distance=distance
        )
    
    # Same haplotype but distant = chimeric (shouldn't happen often)
    return EdgeGroundTruth(
        source_node=source_node_id,
        target_node=target_node_id,
        label=EdgeLabel.CHIMERIC,
        explanation=f"Same haplotype but distant loci (distance={distance}bp)",
        read1_pos=(aln1.ref_chrom, aln1.ref_start, aln1.ref_end),
        read2_pos=(aln2.ref_chrom, aln2.ref_start, aln2.ref_end),
        overlap_distance=distance
    )


def label_all_graph_edges(
    edges: List[Tuple[str, str]],
    alignments: Dict[str, ReadAlignment],
    sv_truth_table: List,
    node_to_read_ids: Optional[Dict[str, List[str]]] = None
) -> Dict[Tuple[str, str], EdgeGroundTruth]:
    """
    Label all edges in assembly graph.
    
    Args:
        edges: List of (source_id, target_id) tuples
        alignments: Dictionary of read alignments
        sv_truth_table: List of structural variants
    
    Returns:
        Dictionary mapping (source, target) -> EdgeGroundTruth
    """
    logger.info(f"Labeling {len(edges)} graph edges...")
    logger.info(f"Available alignments: {len(alignments)} reads")
    
    edge_labels = {}
    label_counts = {label: 0 for label in EdgeLabel}
    missing_count = 0
    
    # Debug: Check first few edges
    if edges:
        sample_edges = edges[:min(3, len(edges))]
        sample_alns = list(alignments.keys())[:min(3, len(alignments))]
        logger.info(f"Sample edge IDs: {sample_edges}")
        logger.info(f"Sample alignment IDs: {sample_alns}")
    
    for source, target in edges:
        label = label_graph_edge(source, target, alignments, sv_truth_table, node_to_read_ids)
        edge_labels[(source, target)] = label
        label_counts[label.label] += 1
        
        if label.label == EdgeLabel.UNKNOWN:
            missing_count += 1
    
    logger.info("Edge labeling complete:")
    for label_type, count in label_counts.items():
        if count > 0:
            logger.info(f"  {label_type.value}: {count} edges")
    
    if missing_count > 0:
        logger.warning(f"WARNING: {missing_count}/{len(edges)} edges labeled as UNKNOWN (missing alignment data)")
        logger.warning("This indicates edge node IDs don't match read IDs in alignment dictionary")
    
    return edge_labels


# ============================================================================
#                    NODE HAPLOTYPE LABELING
# ============================================================================

def label_node_haplotype(
    node_id: str,
    node_sequence: str,
    reference_hapA: str,
    reference_hapB: str,
    alignments: Dict[str, ReadAlignment],
    min_identity: float = 0.95
) -> NodeGroundTruth:
    """
    Determine haplotype assignment for a graph node.
    
    Strategy:
    1. Find all reads that span this node
    2. Check haplotype assignments of those reads
    3. If all reads from haplotype A → HAP_A
    4. If all reads from haplotype B → HAP_B
    5. If reads from both haplotypes → BOTH (shared sequence)
    6. If reads map to multiple locations → REPEAT
    
    Args:
        node_id: Node identifier
        node_sequence: Node sequence
        reference_hapA: Haplotype A reference
        reference_hapB: Haplotype B reference
        alignments: Read alignments
        min_identity: Minimum identity for alignment
    
    Returns:
        NodeGroundTruth object
    """
    # Find reads that contain this node
    # In assembly graphs, nodes are often derived from reads, so node_id matches/contains read_id
    spanning_reads = []
    haplotypes_seen = set()
    
    # Check which reads contributed to this node
    for read_id, aln in alignments.items():
        # Node ID typically contains the read ID (e.g., "read_123" or "read_123_L" or "read_123_R")
        if read_id in node_id or node_id in read_id or read_id == node_id:
            spanning_reads.append(read_id)
            haplotypes_seen.add(aln.haplotype)
        # Also check if node sequence matches read sequence (if node_sequence provided)
        elif node_sequence and len(node_sequence) > 50:
            # For longer sequences, check substring match
            # This catches k-mer based nodes that don't have read IDs
            for other_read_id, other_aln in alignments.items():
                if node_sequence in other_aln.read_sequence if hasattr(other_aln, 'read_sequence') else False:
                    spanning_reads.append(other_read_id)
                    haplotypes_seen.add(other_aln.haplotype)
                    break
    
    # Determine haplotype
    if len(haplotypes_seen) == 0:
        haplotype = NodeHaplotype.UNKNOWN
    elif len(haplotypes_seen) == 1:
        hap = list(haplotypes_seen)[0]
        haplotype = NodeHaplotype.HAP_A if hap == 'A' else NodeHaplotype.HAP_B
    else:
        # Reads from both haplotypes
        haplotype = NodeHaplotype.BOTH
    
    return NodeGroundTruth(
        node_id=node_id,
        haplotype=haplotype,
        spanning_reads=spanning_reads,
        is_repeat=False  # Would check against repeat annotations
    )


# ============================================================================
#                    PATH LABELING (GNN)
# ============================================================================

def extract_correct_paths(
    alignments: Dict[str, ReadAlignment],
    reference_hapA: str,
    reference_hapB: str
) -> List[PathGroundTruth]:
    """
    Extract correct paths through the graph for GNN training.
    
    Strategy:
    1. Sort reads by reference position (per haplotype)
    2. Reads in order = correct path
    3. Create path for each contig/chromosome
    
    Args:
        alignments: Read alignments
        reference_hapA: Haplotype A reference
        reference_hapB: Haplotype B reference
    
    Returns:
        List of PathGroundTruth objects
    """
    logger.info("Extracting correct paths for GNN training...")
    
    paths = []
    
    # Group reads by haplotype and chromosome
    by_haplotype = {}
    for read_id, aln in alignments.items():
        key = (aln.haplotype, aln.ref_chrom)
        if key not in by_haplotype:
            by_haplotype[key] = []
        by_haplotype[key].append((read_id, aln))
    
    # Create paths for each haplotype/chromosome
    for (haplotype, chrom), reads in by_haplotype.items():
        # Sort by start position
        reads.sort(key=lambda x: x[1].ref_start)
        
        # Create path
        node_sequence = [read_id for read_id, _ in reads]
        
        if len(node_sequence) > 0:
            first_aln = reads[0][1]
            last_aln = reads[-1][1]
            
            path = PathGroundTruth(
                path_id=f"path_{haplotype}_{chrom}",
                node_sequence=node_sequence,
                haplotype=haplotype,
                ref_chrom=chrom,
                ref_start=first_aln.ref_start,
                ref_end=last_aln.ref_end,
                is_correct=True,
                confidence=1.0
            )
            paths.append(path)
    
    logger.info(f"Extracted {len(paths)} correct paths")
    return paths


# ============================================================================
#                    UL ROUTING LABELING
# ============================================================================

def label_ul_read_routes(
    ul_reads: List,  # List[SimulatedRead] for UL reads
    alignments: Dict[str, ReadAlignment],
    graph_nodes: List[str]
) -> List[ULRouteGroundTruth]:
    """
    Determine correct routing for ultralong reads through graph.
    
    UL reads span multiple graph nodes. This function determines the
    correct node sequence each UL read should traverse.
    
    Args:
        ul_reads: List of ultralong SimulatedRead objects
        alignments: Read alignments
        graph_nodes: List of node IDs in graph
    
    Returns:
        List of ULRouteGroundTruth objects
    """
    logger.info(f"Labeling routes for {len(ul_reads)} ultralong reads...")
    
    routes = []
    
    for ul_read in ul_reads:
        if ul_read.read_id not in alignments:
            continue
        
        aln = alignments[ul_read.read_id]
        
        # Find all nodes that overlap with this UL read's span
        # Strategy: Find all reads whose reference positions overlap with this UL read
        # then find nodes derived from those reads
        correct_path = []
        
        ul_start = aln.ref_start
        ul_end = aln.ref_end
        ul_chrom = aln.ref_chrom
        ul_hap = aln.haplotype
        
        # Find all reads that overlap this UL read's span (same haplotype)
        overlapping_reads = []
        for other_read_id, other_aln in alignments.items():
            if other_aln.haplotype != ul_hap or other_aln.ref_chrom != ul_chrom:
                continue
            
            # Check for overlap
            if not (other_aln.ref_end < ul_start or other_aln.ref_start > ul_end):
                overlapping_reads.append((other_read_id, other_aln.ref_start))
        
        # Sort by reference position to get correct path order
        overlapping_reads.sort(key=lambda x: x[1])
        
        # Find nodes derived from these reads
        for read_id, _ in overlapping_reads:
            # Look for nodes that match this read
            for node_id in graph_nodes:
                if read_id in node_id or node_id in read_id or read_id == node_id:
                    if node_id not in correct_path:  # Avoid duplicates
                        correct_path.append(node_id)
        
        # Fallback: if no overlapping nodes found, use the UL read itself as single node
        if not correct_path:
            correct_path = [ul_read.read_id]
        
        route = ULRouteGroundTruth(
            read_id=ul_read.read_id,
            correct_path=correct_path,
            ref_positions=(aln.ref_chrom, aln.ref_start, aln.ref_end),
            num_nodes=len(correct_path),
            path_length=aln.ref_end - aln.ref_start
        )
        routes.append(route)
    
    logger.info(f"Labeled {len(routes)} UL routes")
    return routes


# ============================================================================
#                    SV LABELING
# ============================================================================

def label_sv_graph_signatures(
    sv_truth_table: List,
    alignments: Dict[str, ReadAlignment],
    graph_edges: List[Tuple[str, str]]
) -> List[SVGroundTruth]:
    """
    Associate graph features with true structural variants.
    
    For each SV, identify the graph signature:
    - Deletions: Coverage drops, missing edges
    - Insertions: Coverage spikes, graph bubbles
    - Inversions: Complex branching, reversed edges
    - Duplications: High coverage regions
    - Translocations: Long-distance edges
    
    Args:
        sv_truth_table: List of StructuralVariant objects
        alignments: Read alignments
        graph_edges: List of graph edges
    
    Returns:
        List of SVGroundTruth objects
    """
    logger.info(f"Labeling graph signatures for {len(sv_truth_table)} SVs...")
    
    sv_labels = []
    
    for i, sv in enumerate(sv_truth_table):
        # Find reads that span this SV
        spanning_reads = []
        for read_id, aln in alignments.items():
            if (aln.ref_chrom == sv.chrom and
                aln.haplotype == sv.haplotype and
                aln.ref_start <= sv.pos <= aln.ref_end):
                spanning_reads.append(read_id)
        
        # Determine graph signature
        signature = {
            'spanning_reads': spanning_reads,
            'num_spanning': len(spanning_reads),
            'sv_size': sv.size
        }
        
        # Type-specific signatures
        if sv.sv_type.value == 'deletion':
            signature['expected_pattern'] = 'coverage_drop'
        elif sv.sv_type.value == 'insertion':
            signature['expected_pattern'] = 'bubble_or_spike'
        elif sv.sv_type.value == 'inversion':
            signature['expected_pattern'] = 'complex_branching'
        elif sv.sv_type.value == 'duplication':
            signature['expected_pattern'] = 'high_coverage'
        elif sv.sv_type.value == 'translocation':
            signature['expected_pattern'] = 'long_distance_edge'
        
        sv_label = SVGroundTruth(
            sv_id=f"sv_{i}",
            sv_type=SVType(sv.sv_type.value),
            haplotype=sv.haplotype,
            ref_chrom=sv.chrom,
            ref_start=sv.pos,
            ref_end=sv.end,
            size=sv.size,
            graph_signature=signature
        )
        sv_labels.append(sv_label)
    
    logger.info(f"Labeled {len(sv_labels)} SV graph signatures")
    return sv_labels


# ============================================================================
#                    MAIN LABELING PIPELINE
# ============================================================================

@dataclass
class GroundTruthLabels:
    """Complete set of ground-truth labels for ML training."""
    edge_labels: Dict[Tuple[str, str], EdgeGroundTruth]
    node_labels: Dict[str, NodeGroundTruth]
    path_labels: List[PathGroundTruth]
    ul_route_labels: List[ULRouteGroundTruth]
    sv_labels: List[SVGroundTruth]


def generate_ground_truth_labels(
    simulated_reads: List,  # List[SimulatedRead]
    ul_reads: List,  # List[SimulatedRead] for UL
    reference_hapA: str,
    reference_hapB: str,
    sv_truth_table: List,  # List[StructuralVariant]
    graph_edges: List[Tuple[str, str]],
    graph_nodes: List[str],
    node_to_read_ids: Optional[Dict[str, List[str]]] = None
) -> GroundTruthLabels:
    """
    Generate complete ground-truth labels for ML training.
    
    This is the main entry point that orchestrates all labeling operations.
    
    Args:
        simulated_reads: All simulated reads
        ul_reads: Ultralong reads subset
        reference_hapA: Haplotype A reference sequence
        reference_hapB: Haplotype B reference sequence
        sv_truth_table: Structural variant truth table
        graph_edges: List of graph edges (source, target)
        graph_nodes: List of graph node IDs
        node_to_read_ids: Optional mapping of node_id -> contributing read_ids
    
    Returns:
        GroundTruthLabels object with all labels
    """
    logger.info("=" * 80)
    logger.info("Generating ground-truth labels for ML training")
    logger.info("=" * 80)
    
    # Step 1: Align reads to reference
    alignments = align_simulated_reads_to_reference(
        simulated_reads, reference_hapA, reference_hapB
    )
    
    # Use provided node_to_read_ids if available; otherwise build from node IDs
    if not node_to_read_ids:
        node_to_read_ids = {}
        for node_id in graph_nodes:
            node_to_read_ids[node_id] = []
            for read_id in alignments.keys():
                if read_id in node_id or node_id in read_id or read_id == node_id:
                    node_to_read_ids[node_id].append(read_id)
    
    # Step 2: Label graph edges (use mapping for robust resolution)
    edge_labels = label_all_graph_edges(graph_edges, alignments, sv_truth_table, node_to_read_ids)    # Step 3: Label node haplotypes
    logger.info(f"Labeling {len(graph_nodes)} graph nodes...")
    
    # Build efficient lookup: node_id -> reads that contain it
    # Most nodes are named after reads (e.g., "read_123" node from "read_123" read)
    node_labels = {}
    
    for node_id in graph_nodes:
        spanning_reads = []
        haplotypes_seen = set()
        
        # Check if this node ID matches any read ID (common in assembly graphs)
        # Node IDs are typically: read_id, read_id_suffix, or kmer-based
        for read_id, aln in alignments.items():
            # Node contains this read, or vice versa
            if read_id in node_id or node_id in read_id or read_id == node_id:
                spanning_reads.append(read_id)
                haplotypes_seen.add(aln.haplotype)
        
        # Determine haplotype from spanning reads
        if len(haplotypes_seen) == 0:
            haplotype = NodeHaplotype.UNKNOWN
        elif len(haplotypes_seen) == 1:
            hap = list(haplotypes_seen)[0]
            haplotype = NodeHaplotype.HAP_A if hap == 'A' else NodeHaplotype.HAP_B
        else:
            # Reads from both haplotypes = shared node
            haplotype = NodeHaplotype.BOTH
        
        # Check if node is from repeat region
        is_repeat = any(alignments[rid].is_repeat for rid in spanning_reads if rid in alignments)
        if is_repeat:
            haplotype = NodeHaplotype.REPEAT
        
        node_labels[node_id] = NodeGroundTruth(
            node_id=node_id,
            haplotype=haplotype,
            spanning_reads=spanning_reads,
            is_repeat=is_repeat
        )
    
    logger.info(f"Labeled {len(node_labels)} nodes:")
    hap_counts = {'A': 0, 'B': 0, 'both': 0, 'repeat': 0, 'unknown': 0}
    for nl in node_labels.values():
        if nl.haplotype == NodeHaplotype.HAP_A:
            hap_counts['A'] += 1
        elif nl.haplotype == NodeHaplotype.HAP_B:
            hap_counts['B'] += 1
        else:
            hap_counts[nl.haplotype.value] += 1
    logger.info(f"  Hap A: {hap_counts.get('A', 0)}, Hap B: {hap_counts.get('B', 0)}, Both: {hap_counts['both']}, Repeat: {hap_counts['repeat']}, Unknown: {hap_counts['unknown']}")
    
    # Step 4: Extract correct paths
    path_labels = extract_correct_paths(alignments, reference_hapA, reference_hapB)
    
    # Step 5: Label UL routes
    ul_route_labels = label_ul_read_routes(ul_reads, alignments, graph_nodes)
    
    # Step 6: Label SV graph signatures
    sv_labels = label_sv_graph_signatures(sv_truth_table, alignments, graph_edges)
    
    logger.info("=" * 80)
    logger.info("Ground-truth labeling complete")
    logger.info("=" * 80)
    logger.info(f"Edge labels: {len(edge_labels)}")
    logger.info(f"Node labels: {len(node_labels)}")
    logger.info(f"Path labels: {len(path_labels)}")
    logger.info(f"UL route labels: {len(ul_route_labels)}")
    logger.info(f"SV labels: {len(sv_labels)}")
    
    return GroundTruthLabels(
        edge_labels=edge_labels,
        node_labels=node_labels,
        path_labels=path_labels,
        ul_route_labels=ul_route_labels,
        sv_labels=sv_labels
    )


# ============================================================================
#                    EXPORT FUNCTIONS
# ============================================================================

def export_labels_to_tsv(labels: GroundTruthLabels, output_dir: Path) -> None:
    """
    Export ground-truth labels to TSV files for inspection.
    
    Args:
        labels: GroundTruthLabels object
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export edge labels
    edge_file = output_dir / "edge_labels.tsv"
    with open(edge_file, 'w') as f:
        f.write("source\ttarget\tlabel\texplanation\tdistance\tcrosses_sv\n")
        for (source, target), label in labels.edge_labels.items():
            f.write(f"{source}\t{target}\t{label.label.value}\t{label.explanation}\t"
                   f"{label.overlap_distance}\t{label.crosses_sv}\n")
    
    # Export node labels
    node_file = output_dir / "node_labels.tsv"
    with open(node_file, 'w') as f:
        f.write("node_id\thaplotype\tis_repeat\tnum_spanning_reads\n")
        for node_id, label in labels.node_labels.items():
            f.write(f"{node_id}\t{label.haplotype.value}\t{label.is_repeat}\t"
                   f"{len(label.spanning_reads)}\n")
    
    # Export SV labels
    sv_file = output_dir / "sv_labels.tsv"
    with open(sv_file, 'w') as f:
        f.write("sv_id\tsv_type\thaplotype\tchrom\tstart\tend\tsize\tnum_spanning_reads\n")
        for label in labels.sv_labels:
            f.write(f"{label.sv_id}\t{label.sv_type.value}\t{label.haplotype}\t"
                   f"{label.ref_chrom}\t{label.ref_start}\t{label.ref_end}\t"
                   f"{label.size}\t{label.graph_signature.get('num_spanning', 0)}\n")
    
    logger.info(f"Exported labels to {output_dir}")
    logger.info(f"  {edge_file}")
    logger.info(f"  {node_file}")
    logger.info(f"  {sv_file}")
