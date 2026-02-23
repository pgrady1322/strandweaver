#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Assembly Export — GFA graph export, coverage CSV, assembly FASTA,
statistics JSON, and scaffold path TSV.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Protocol
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
#                           GRAPH PROTOCOL
# ============================================================================

class GraphLike(Protocol):
    """
    Protocol defining the minimum interface for graphs that can be exported.
    
    Both DBGGraph and StringGraph should satisfy this protocol.
    """
    
    @property
    def nodes(self) -> dict[int, Any]:
        """Return mapping of node_id -> node object."""
        ...
    
    @property
    def edges(self) -> dict[int, Any]:
        """Return mapping of edge_id -> edge object."""
        ...
    
    def get_node_sequence(self, node_id: int) -> str:
        """Return sequence for a node."""
        ...
    
    def get_node_length(self, node_id: int) -> int:
        """Return length of node sequence."""
        ...
    
    def get_edge_endpoints(self, edge_id: int) -> tuple[int, int]:
        """Return (from_node_id, to_node_id) for an edge."""
        ...
    
    def get_node_orientation(self, node_id: int) -> str:
        """Return '+' or '-' for node orientation (default '+')."""
        ...


# ============================================================================
#                       GFA EXPORT FUNCTIONS
# ============================================================================

@dataclass
class GFASegment:
    """Represents a GFA S-line (segment)."""
    node_id: int
    name: str  # External name (e.g., 'unitig-1')
    sequence: str
    length: int
    
    def to_gfa_line(self) -> str:
        """
        Convert to GFA S-line format.
        
        Format: S <name> <sequence> LN:i:<length>
        
        For very long sequences, we can optionally use '*' and rely on LN tag.
        """
        # Use actual sequence if reasonable length, else use '*' placeholder
        seq_str = self.sequence if len(self.sequence) < 100000 else '*'
        return f"S\t{self.name}\t{seq_str}\tLN:i:{self.length}"


@dataclass
class GFALink:
    """Represents a GFA L-line (link/edge)."""
    from_name: str
    from_orient: str  # '+' or '-'
    to_name: str
    to_orient: str    # '+' or '-'
    overlap: str      # Overlap string (e.g., '0M' for exact adjacency)
    
    def to_gfa_line(self) -> str:
        """
        Convert to GFA L-line format.
        
        Format: L <from> <from_orient> <to> <to_orient> <overlap>
        """
        return f"L\t{self.from_name}\t{self.from_orient}\t{self.to_name}\t{self.to_orient}\t{self.overlap}"


def generate_unitig_name(node_id: int) -> str:
    """
    Generate a standard unitig name from internal node ID.
    
    Args:
        node_id: Internal numeric node identifier
    
    Returns:
        External name string (e.g., 'unitig-1')
    """
    return f"unitig-{node_id}"


def export_graph_to_gfa(
    graph: GraphLike,
    output_path: str | Path,
    include_sequence: bool = True
) -> None:
    """
    Export assembly graph to GFA format for BandageNG visualization.
    
    Generates a GFA v1-compatible file with:
    - S lines for segments (nodes/unitigs)
    - L lines for links (edges/overlaps)
    
    Args:
        graph: Graph object (DBGGraph or StringGraph) implementing GraphLike protocol
        output_path: Path to output GFA file
        include_sequence: If True, include full sequences in S lines;
                         if False, use '*' placeholder and rely on LN tag
    
    The GFA format allows BandageNG to:
    - Display graph topology
    - Show node lengths and sequences
    - Navigate through connected components
    - Allow user path editing and annotation
    """
    output_path = Path(output_path)
    logger.info(f"Exporting graph to GFA: {output_path}")
    
    segments: list[GFASegment] = []
    links: list[GFALink] = []
    
    # Create mapping from internal node IDs to external names
    node_name_map: dict[int, str] = {}
    
    # Step 1: Export all nodes as GFA segments
    logger.info(f"Processing {len(graph.nodes)} nodes...")
    for node_id, node in graph.nodes.items():
        unitig_name = generate_unitig_name(node_id)
        node_name_map[node_id] = unitig_name
        
        sequence = node.seq if include_sequence else ""
        length = node.length
        
        segment = GFASegment(
            node_id=node_id,
            name=unitig_name,
            sequence=sequence,
            length=length
        )
        segments.append(segment)
    
    # Step 2: Export all edges as GFA links
    logger.info(f"Processing {len(graph.edges)} edges...")
    for edge_id, edge in graph.edges.items():
        from_id = edge.from_id
        to_id = edge.to_id
        
        # Get node names
        from_name = node_name_map.get(from_id)
        to_name = node_name_map.get(to_id)
        
        if from_name is None or to_name is None:
            logger.warning(f"Edge {edge_id} references unknown nodes: {from_id} -> {to_id}")
            continue
        
        # Default to '+' orientation for both nodes
        from_orient = '+'
        to_orient = '+'
        
        # For simplicity, use '0M' (zero-length match) as overlap
        # In a real implementation, this could encode actual overlap length
        overlap = '0M'
        
        link = GFALink(
            from_name=from_name,
            from_orient=from_orient,
            to_name=to_name,
            to_orient=to_orient,
            overlap=overlap
        )
        links.append(link)
    
    # Step 3: Write GFA file
    logger.info(f"Writing GFA with {len(segments)} segments and {len(links)} links...")
    with open(output_path, 'w') as f:
        # Header
        f.write("H\tVN:Z:1.0\n")
        
        # Segments
        for seg in segments:
            f.write(seg.to_gfa_line() + "\n")
        
        # Links
        for link in links:
            f.write(link.to_gfa_line() + "\n")
    
    logger.info(f"GFA export complete: {output_path}")
    logger.info(f"  Segments: {len(segments)}")
    logger.info(f"  Links: {len(links)}")


# ============================================================================
#                    COVERAGE CSV EXPORT FUNCTIONS
# ============================================================================

def export_coverage_csv(
    graph: GraphLike,
    long_read_coverage: dict[int, float],
    ul_read_coverage: dict[int, float] | None,
    hic_support: dict[int, float] | None,
    output_prefix: str | Path,
    edge_quality_scores: dict[tuple[int, int], float] | None = None
) -> None:
    """
    Export per-node coverage/support data as CSV files for BandageNG overlay.
    
    Creates separate CSV files for each data type:
    - {output_prefix}_long.csv: Long read coverage
    - {output_prefix}_ul.csv: Ultralong read coverage (if provided)
    - {output_prefix}_hic.csv: Hi-C contact support (if provided)
    - {output_prefix}_edge_scores.csv: Edge quality scores 0-1 (if provided)
    
    Each CSV has columns:
    - node_name: External unitig name (e.g., 'unitig-1')
    - coverage: Numeric coverage/support value
    
    For edge scores CSV:
    - from_node: Source node name
    - to_node: Target node name  
    - quality_score: Edge quality score 0.0-1.0
    
    BandageNG can load these CSVs to color nodes by coverage level,
    helping users identify:
    - Low-coverage regions (potential errors)
    - High-coverage regions (repeats or duplications)
    - Differential support across data types
    - Low-quality edges (potential misassemblies)
    
    Args:
        graph: Graph object implementing GraphLike protocol
        long_read_coverage: Dict mapping node_id -> long read coverage
        ul_read_coverage: Optional dict for ultralong read coverage
        hic_support: Optional dict for Hi-C contact count/weight
        output_prefix: Prefix for output CSV files
        edge_quality_scores: Optional dict mapping (source_id, target_id) -> quality score (0-1)
    """
    output_prefix = Path(output_prefix)
    logger.info(f"Exporting coverage CSVs with prefix: {output_prefix}")
    
    # Create node name mapping
    node_name_map: dict[int, str] = {
        node_id: generate_unitig_name(node_id)
        for node_id in graph.nodes
    }
    
    # Export long read coverage (always required)
    long_path = f"{output_prefix}_long.csv"
    logger.info(f"Writing long read coverage: {long_path}")
    with open(long_path, 'w') as f:
        f.write("node_name,coverage\n")
        for node_id in sorted(graph.nodes.keys()):
            node_name = node_name_map[node_id]
            cov = long_read_coverage.get(node_id, 0.0)
            f.write(f"{node_name},{cov:.2f}\n")
    
    # Export UL coverage if provided
    if ul_read_coverage is not None:
        ul_path = f"{output_prefix}_ul.csv"
        logger.info(f"Writing ultralong read coverage: {ul_path}")
        with open(ul_path, 'w') as f:
            f.write("node_name,coverage\n")
            for node_id in sorted(graph.nodes.keys()):
                node_name = node_name_map[node_id]
                cov = ul_read_coverage.get(node_id, 0.0)
                f.write(f"{node_name},{cov:.2f}\n")
    
    # Export Hi-C support if provided
    if hic_support is not None:
        hic_path = f"{output_prefix}_hic.csv"
        logger.info(f"Writing Hi-C support: {hic_path}")
        with open(hic_path, 'w') as f:
            f.write("node_name,support\n")
            for node_id in sorted(graph.nodes.keys()):
                node_name = node_name_map[node_id]
                support = hic_support.get(node_id, 0.0)
                f.write(f"{node_name},{support:.2f}\n")
    
    # Export edge quality scores if provided
    if edge_quality_scores is not None:
        edge_path = f"{output_prefix}_edge_scores.csv"
        logger.info(f"Writing edge quality scores: {edge_path}")
        with open(edge_path, 'w') as f:
            f.write("from_node,to_node,quality_score\n")
            for (source_id, target_id), score in sorted(edge_quality_scores.items()):
                from_name = node_name_map.get(source_id, f"unitig-{source_id}")
                to_name = node_name_map.get(target_id, f"unitig-{target_id}")
                f.write(f"{from_name},{to_name},{score:.4f}\n")
    
    logger.info("Coverage CSV export complete")


def export_for_bandageng(
    graph: GraphLike,
    output_prefix: str | Path,
    long_read_coverage: dict[int, float],
    ul_read_coverage: dict[int, float] | None = None,
    hic_support: dict[int, float] | None = None,
    edge_quality_scores: dict[tuple[int, int], float] | None = None,
    include_sequence: bool = True
) -> dict[str, Path]:
    """
    Complete export pipeline for BandageNG visualization.
    
    Convenience function that exports both GFA and coverage CSVs in one call.
    
    Args:
        graph: Graph object to export
        output_prefix: Prefix for all output files
        long_read_coverage: Long read coverage per node (required)
        ul_read_coverage: Ultralong coverage per node (optional)
        hic_support: Hi-C support per node (optional)
        edge_quality_scores: Edge quality scores 0-1 per edge (optional)
        include_sequence: Whether to include sequences in GFA
    
    Returns:
        Dict mapping file type to output path:
        - 'gfa': Path to GFA file
        - 'long_cov': Path to long read coverage CSV
        - 'ul_cov': Path to UL coverage CSV (if provided)
        - 'hic_cov': Path to Hi-C support CSV (if provided)
        - 'edge_scores': Path to edge quality scores CSV (if provided)
    
    Example:
        >>> files = export_for_bandageng(
        ...     graph, "my_assembly",
        ...     long_cov_dict, ul_cov_dict, hic_dict, edge_scores_dict
        ... )
        >>> print(f"Load {files['gfa']} in BandageNG")
        >>> print(f"Overlay coverage from {files['long_cov']}")
        >>> print(f"Overlay edge quality from {files['edge_scores']}")
    """
    output_prefix = Path(output_prefix)
    output_files: dict[str, Path] = {}
    
    logger.info(f"Starting complete BandageNG export: {output_prefix}")
    
    # Export GFA
    gfa_path = output_prefix.with_suffix('.gfa')
    export_graph_to_gfa(graph, gfa_path, include_sequence=include_sequence)
    output_files['gfa'] = gfa_path
    
    # Export coverage CSVs
    export_coverage_csv(
        graph, long_read_coverage, ul_read_coverage, hic_support, output_prefix, edge_quality_scores
    )
    output_files['long_cov'] = Path(f"{output_prefix}_long.csv")
    
    if ul_read_coverage is not None:
        output_files['ul_cov'] = Path(f"{output_prefix}_ul.csv")
    
    if hic_support is not None:
        output_files['hic_cov'] = Path(f"{output_prefix}_hic.csv")
    
    if edge_quality_scores is not None:
        output_files['edge_scores'] = Path(f"{output_prefix}_edge_scores.csv")
    
    logger.info("Complete BandageNG export finished")
    logger.info(f"Generated {len(output_files)} output files")
    
    return output_files


# ============================================================================
#                           GFA READER (Graph Import)
# ============================================================================

def load_graph_from_gfa(gfa_path: str | Path) -> Any:
    """
    Load an assembly graph from a GFA v1 file.

    Reconstructs a KmerGraph (DBGGraph) with nodes and edges from S-lines
    and L-lines.  The graph object satisfies the GraphLike protocol and can
    be passed to any StrandWeaver module that accepts a graph.

    Args:
        gfa_path: Path to a GFA v1 file (as produced by export_graph_to_gfa)

    Returns:
        A KmerGraph instance populated with nodes and edges.

    Raises:
        FileNotFoundError: If gfa_path does not exist.
        ValueError: On malformed GFA lines.
    """
    from ..assembly_core.dbg_engine_module import KmerGraph, KmerNode, KmerEdge

    gfa_path = Path(gfa_path)
    if not gfa_path.exists():
        raise FileNotFoundError(f"GFA file not found: {gfa_path}")

    graph = KmerGraph()
    edge_counter = 0

    logger.info(f"Loading graph from GFA: {gfa_path}")

    with open(gfa_path, 'r') as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.rstrip('\n')
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            record_type = parts[0]

            if record_type == 'H':
                # Header line — skip
                continue

            elif record_type == 'S':
                # Segment: S <name> <sequence> [LN:i:<length>] ...
                if len(parts) < 3:
                    logger.warning(f"GFA line {line_no}: malformed S-line, skipping")
                    continue
                name = parts[1]
                sequence = parts[2] if parts[2] != '*' else ''
                # Extract LN tag if present
                length = len(sequence)
                for tag in parts[3:]:
                    if tag.startswith('LN:i:'):
                        length = int(tag.split(':')[2])
                        break

                node_id = parse_unitig_name(name) if name.startswith('unitig-') else line_no
                node = KmerNode(
                    id=node_id,
                    seq=sequence,
                    coverage=1.0,
                    length=length,
                )
                graph.add_node(node)

            elif record_type == 'L':
                # Link: L <from> <from_orient> <to> <to_orient> <overlap>
                if len(parts) < 6:
                    logger.warning(f"GFA line {line_no}: malformed L-line, skipping")
                    continue
                from_name = parts[1]
                to_name = parts[3]

                from_id = parse_unitig_name(from_name) if from_name.startswith('unitig-') else int(from_name)
                to_id = parse_unitig_name(to_name) if to_name.startswith('unitig-') else int(to_name)

                edge = KmerEdge(
                    id=edge_counter,
                    from_id=from_id,
                    to_id=to_id,
                    coverage=1.0,
                    overlap_len=0,
                )
                graph.add_edge(edge)
                edge_counter += 1

            # P, W, C lines ignored for now

    logger.info(
        f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )
    return graph


# ============================================================================
#                           UTILITY FUNCTIONS
# ============================================================================

def parse_unitig_name(name: str) -> int:
    """
    Parse a unitig name to extract internal node ID.
    
    Args:
        name: External unitig name (e.g., 'unitig-1', 'unitig-42')
    
    Returns:
        Internal node ID as integer
    
    Raises:
        ValueError: If name format is invalid
    
    Example:
        >>> parse_unitig_name('unitig-1')
        1
        >>> parse_unitig_name('unitig-42')
        42
    """
    if not name.startswith('unitig-'):
        raise ValueError(f"Invalid unitig name format: {name} (expected 'unitig-N')")
    
    try:
        node_id = int(name.split('-')[1])
        return node_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse unitig name: {name}") from e


def validate_gfa_file(gfa_path: str | Path) -> dict[str, int]:
    """
    Validate a GFA file and return basic statistics.
    
    Args:
        gfa_path: Path to GFA file
    
    Returns:
        Dict with keys: 'segments', 'links', 'version'
    
    Useful for checking export results or validating user-provided GFA.
    """
    gfa_path = Path(gfa_path)
    stats = {
        'segments': 0,
        'links': 0,
        'version': None
    }
    
    with open(gfa_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('H'):
                # Header line
                if 'VN:Z:' in line:
                    stats['version'] = line.split('VN:Z:')[1].split()[0]
            elif line.startswith('S'):
                stats['segments'] += 1
            elif line.startswith('L'):
                stats['links'] += 1
    
    return stats


def get_node_name_mapping(graph: GraphLike) -> dict[int, str]:
    """
    Generate complete mapping from internal node IDs to external unitig names.
    
    Args:
        graph: Graph object
    
    Returns:
        Dict mapping node_id -> unitig_name
    
    Useful for coordinate conversion in downstream analysis.
    """
    return {
        node_id: generate_unitig_name(node_id)
        for node_id in graph.nodes
    }


def get_reverse_node_mapping(graph: GraphLike) -> dict[str, int]:
    """
    Generate mapping from external unitig names to internal node IDs.
    
    Args:
        graph: Graph object
    
    Returns:
        Dict mapping unitig_name -> node_id
    
    Used when importing user corrections to convert names back to IDs.
    """
    return {
        generate_unitig_name(node_id): node_id
        for node_id in graph.nodes
    }


# ============================================================================
#                    ASSEMBLY SEQUENCE EXPORT (NEW)
# ============================================================================

def write_contigs_fasta(
    contigs: list[tuple[str, str]],
    output_path: str | Path,
    line_width: int = 80
) -> None:
    """
    Export assembled contigs to FASTA format.
    
    Args:
        contigs: List of (contig_id, sequence) tuples
        output_path: Path to output FASTA file
        line_width: Number of bases per line (0 = no wrapping)
    
    Example:
        >>> contigs = [
        ...     ('contig_1', 'ACGTACGT...'),
        ...     ('contig_2', 'TGCATGCA...')
        ... ]
        >>> write_contigs_fasta(contigs, 'assembly.fasta')
    """
    output_path = Path(output_path)
    logger.info(f"Writing {len(contigs)} contigs to {output_path}")
    
    with open(output_path, 'w') as f:
        for contig_id, sequence in contigs:
            f.write(f">{contig_id}\n")
            
            # Write sequence with line wrapping if specified
            if line_width > 0:
                for i in range(0, len(sequence), line_width):
                    f.write(sequence[i:i+line_width] + "\n")
            else:
                f.write(sequence + "\n")
    
    logger.info(f"Exported {len(contigs)} contigs ({sum(len(s) for _, s in contigs):,} bp)")


def write_scaffolds_fasta(
    scaffolds: list[tuple[str, list[str], list[str]]],
    output_path: str | Path,
    gap_char: str = 'N',
    gap_size: int = 100,
    line_width: int = 80
) -> None:
    """
    Export scaffolds to FASTA format with gaps between contigs.
    
    Args:
        scaffolds: List of (scaffold_id, contig_sequences, orientations) tuples
        output_path: Path to output FASTA file
        gap_char: Character to use for gaps (typically 'N')
        gap_size: Number of gap characters between contigs
        line_width: Number of bases per line (0 = no wrapping)
    
    Example:
        >>> scaffolds = [
        ...     ('chr1', ['ACGT...', 'TGCA...'], ['+', '+']),
        ...     ('chr2', ['GGCC...', 'AATT...'], ['+', '-'])
        ... ]
        >>> write_scaffolds_fasta(scaffolds, 'scaffolds.fasta')
    """
    output_path = Path(output_path)
    logger.info(f"Writing {len(scaffolds)} scaffolds to {output_path}")
    
    gap_sequence = gap_char * gap_size
    
    with open(output_path, 'w') as f:
        for scaffold_id, sequences, orientations in scaffolds:
            f.write(f">{scaffold_id}\n")
            
            # Build scaffold sequence
            scaffold_seq_parts = []
            for seq, orient in zip(sequences, orientations):
                if orient == '-':
                    # Reverse complement
                    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
                    seq = ''.join(complement.get(base, base) for base in reversed(seq))
                scaffold_seq_parts.append(seq)
            
            # Join with gaps
            scaffold_sequence = gap_sequence.join(scaffold_seq_parts)
            
            # Write with line wrapping
            if line_width > 0:
                for i in range(0, len(scaffold_sequence), line_width):
                    f.write(scaffold_sequence[i:i+line_width] + "\n")
            else:
                f.write(scaffold_sequence + "\n")
    
    logger.info(f"Exported {len(scaffolds)} scaffolds")


def export_assembly_stats(
    graph: GraphLike,
    output_path: str | Path,
    contigs: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    """
    Calculate and export assembly statistics to JSON.
    
    Computes standard assembly metrics:
    - Total length, number of sequences
    - N50, L50, N90, L90
    - GC content
    - Longest/shortest sequences
    
    Args:
        graph: Assembly graph
        output_path: Path to output JSON file
        contigs: Optional list of (id, sequence) tuples for sequence-based stats
    
    Returns:
        Dictionary of statistics
    
    Example:
        >>> stats = export_assembly_stats(graph, 'assembly_stats.json', contigs)
        >>> print(f"N50: {stats['n50']:,} bp")
    """
    output_path = Path(output_path)
    logger.info(f"Calculating assembly statistics...")
    
    stats: dict[str, Any] = {}
    
    # Graph-based stats
    stats['num_nodes'] = len(graph.nodes)
    stats['num_edges'] = len(graph.edges)
    
    # Get node lengths
    node_lengths = [node.length for node in graph.nodes.values()]
    node_lengths_sorted = sorted(node_lengths, reverse=True)
    
    stats['total_length'] = sum(node_lengths)
    stats['mean_node_length'] = sum(node_lengths) / len(node_lengths) if node_lengths else 0
    stats['median_node_length'] = node_lengths_sorted[len(node_lengths_sorted)//2] if node_lengths else 0
    stats['max_node_length'] = max(node_lengths) if node_lengths else 0
    stats['min_node_length'] = min(node_lengths) if node_lengths else 0
    
    # Calculate N50/L50
    cumsum = 0
    half_total = stats['total_length'] / 2
    for i, length in enumerate(node_lengths_sorted):
        cumsum += length
        if cumsum >= half_total:
            stats['n50'] = length
            stats['l50'] = i + 1
            break
    
    # Calculate N90/L90
    cumsum = 0
    ninety_pct = stats['total_length'] * 0.9
    for i, length in enumerate(node_lengths_sorted):
        cumsum += length
        if cumsum >= ninety_pct:
            stats['n90'] = length
            stats['l90'] = i + 1
            break
    
    # Contig-based stats (if provided)
    if contigs:
        contig_lengths = [len(seq) for _, seq in contigs]
        stats['num_contigs'] = len(contigs)
        stats['total_contig_length'] = sum(contig_lengths)
        
        # Calculate GC content
        total_gc = 0
        total_bases = 0
        for _, seq in contigs:
            total_gc += seq.upper().count('G') + seq.upper().count('C')
            total_bases += len(seq)
        stats['gc_content'] = (total_gc / total_bases * 100) if total_bases > 0 else 0
    
    # Write to JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Assembly statistics exported to {output_path}")
    logger.info(f"  Total length: {stats['total_length']:,} bp")
    logger.info(f"  N50: {stats.get('n50', 0):,} bp")
    logger.info(f"  L50: {stats.get('l50', 0):,}")
    
    return stats


def export_paths_tsv(
    paths: list[tuple[str, list[int], list[str]]],
    output_path: str | Path,
    node_name_func: callable = generate_unitig_name
) -> None:
    """
    Export scaffold paths to TSV format.
    
    Format matches BandageNG TSV for round-trip compatibility:
    chrom\tpath\tnotes
    
    Args:
        paths: List of (scaffold_id, node_ids, orientations) tuples
        output_path: Path to output TSV file
        node_name_func: Function to convert node_id to external name
    
    Example:
        >>> paths = [
        ...     ('chr1', [1, 2, 3], ['+', '+', '-']),
        ...     ('chr2', [4, 5], ['+', '+'])
        ... ]
        >>> export_paths_tsv(paths, 'scaffold_paths.tsv')
    """
    output_path = Path(output_path)
    logger.info(f"Exporting {len(paths)} paths to {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# Scaffold paths\n")
        f.write("# Format: scaffold_id\tpath\tnotes\n")
        f.write("scaffold_id\tpath\tnotes\n")
        
        for scaffold_id, node_ids, orientations in paths:
            # Convert to unitig names with orientations
            path_str = ','.join(
                f"{node_name_func(nid)}{orient}"
                for nid, orient in zip(node_ids, orientations)
            )
            f.write(f"{scaffold_id}\t{path_str}\t\n")
    
    logger.info(f"Exported {len(paths)} scaffold paths")

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.

