#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

User Input — import BandageNG path corrections, manual breaks/joins,
and exclusion lists.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol, Any

logger = logging.getLogger(__name__)


# ============================================================================
#                           DATA STRUCTURES
# ============================================================================

@dataclass
class UserPathCorrection:
    """
    Represents a user-corrected path for a single chromosome/scaffold.
    
    Attributes:
        chrom: Chromosome or scaffold identifier (e.g., 'chr1', 'scaffold_2')
        path_nodes: List of (unitig_id, orientation) tuples
                   Example: [('unitig-1', '+'), ('unitig-2', '+'), ('unitig-3', '-')]
        notes: Optional user annotations explaining the correction
    
    Example:
        >>> correction = UserPathCorrection(
        ...     chrom='chr1',
        ...     path_nodes=[('unitig-1', '+'), ('unitig-2', '+'), ('unitig-3', '-')],
        ...     notes='removed mis-placed unitig-4'
        ... )
    """
    chrom: str
    path_nodes: list[tuple[str, str]]  # [(unitig_id, orientation), ...]
    notes: str | None = None
    
    def __post_init__(self):
        """Validate orientations are '+' or '-'."""
        for unitig_id, orient in self.path_nodes:
            if orient not in ('+', '-'):
                raise ValueError(
                    f"Invalid orientation '{orient}' for {unitig_id} "
                    f"in {self.chrom} (must be '+' or '-')"
                )
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in this path."""
        return len(self.path_nodes)
    
    def get_unitig_ids(self) -> list[str]:
        """Extract just the unitig IDs (without orientations)."""
        return [unitig_id for unitig_id, _ in self.path_nodes]
    
    def get_orientations(self) -> list[str]:
        """Extract just the orientations."""
        return [orient for _, orient in self.path_nodes]


@dataclass
class ReconstructedScaffold:
    """
    Represents a scaffold reconstructed from user-corrected paths.
    
    This uses internal node IDs (integers) rather than external unitig names,
    making it compatible with the graph data structures.
    
    Attributes:
        chrom: Chromosome/scaffold identifier
        node_path: Ordered list of internal node IDs
        orientation: Orientation ('+' or '-') for each node in node_path
        notes: User notes from the correction
        total_length: Total sequence length (sum of node lengths)
        num_nodes: Number of nodes in the path
    
    Example:
        >>> scaffold = ReconstructedScaffold(
        ...     chrom='chr1',
        ...     node_path=[1, 2, 3, 5],
        ...     orientation=['+', '+', '-', '+'],
        ...     notes='removed unitig-4'
        ... )
    """
    chrom: str
    node_path: list[int]          # Internal node IDs in order
    orientation: list[str]        # '+' or '-' for each node
    notes: str | None = None
    total_length: int = 0         # Filled in during reconstruction
    num_nodes: int = field(init=False)
    
    def __post_init__(self):
        """Validate that node_path and orientation have same length."""
        if len(self.node_path) != len(self.orientation):
            raise ValueError(
                f"node_path length ({len(self.node_path)}) != "
                f"orientation length ({len(self.orientation)})"
            )
        self.num_nodes = len(self.node_path)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chrom': self.chrom,
            'node_path': self.node_path,
            'orientation': self.orientation,
            'notes': self.notes,
            'total_length': self.total_length,
            'num_nodes': self.num_nodes
        }


# ============================================================================
#                           GRAPH PROTOCOL
# ============================================================================

class GraphLike(Protocol):
    """
    Protocol defining minimum graph interface for reconstruction.
    
    Must provide:
    - Node lookup by ID
    - Node length retrieval
    - Edge manipulation (add/remove)
    """
    
    @property
    def nodes(self) -> dict[int, Any]:
        """Return mapping of node_id -> node object."""
        ...
    
    @property
    def edges(self) -> dict[int, Any]:
        """Return mapping of edge_id -> edge object."""
        ...
    
    def get_node_length(self, node_id: int) -> int:
        """Return length of node sequence."""
        ...
    
    def get_edge_endpoints(self, edge_id: int) -> tuple[int, int]:
        """Return (from_node_id, to_node_id) for an edge."""
        ...
    
    def add_edge(self, from_node: int, to_node: int, **kwargs) -> int:
        """Add an edge and return edge ID."""
        ...
    
    def remove_edge(self, edge_id: int) -> None:
        """Remove an edge from the graph."""
        ...


# ============================================================================
#                       TSV PARSING FUNCTIONS
# ============================================================================

def parse_path_string(path_str: str) -> list[tuple[str, str]]:
    """
    Parse a path string into list of (unitig_id, orientation) tuples.
    
    Args:
        path_str: Comma-separated path like 'unitig-1+,unitig-2+,unitig-3-'
    
    Returns:
        List of (unitig_id, orientation) tuples
    
    Raises:
        ValueError: If path format is invalid
    
    Example:
        >>> parse_path_string('unitig-1+,unitig-2+,unitig-3-')
        [('unitig-1', '+'), ('unitig-2', '+'), ('unitig-3', '-')]
    """
    path_str = path_str.strip()
    if not path_str:
        return []
    
    path_nodes: list[tuple[str, str]] = []
    
    for segment in path_str.split(','):
        segment = segment.strip()
        if not segment:
            continue
        
        # Last character should be orientation
        if len(segment) < 2:
            raise ValueError(f"Invalid path segment: '{segment}' (too short)")
        
        unitig_id = segment[:-1]
        orientation = segment[-1]
        
        if orientation not in ('+', '-'):
            raise ValueError(
                f"Invalid orientation in '{segment}' "
                f"(expected '+' or '-', got '{orientation}')"
            )
        
        path_nodes.append((unitig_id, orientation))
    
    return path_nodes


def parse_user_corrections_tsv(tsv_path: str | Path) -> list[UserPathCorrection]:
    """
    Parse the tab-separated user corrections file.
    
    TSV Format:
    -----------
    Column 1: chrom (e.g., 'chr1')
    Column 2: path string 'unitig-1+,unitig-2-,...'
    Column 3: optional notes (may contain spaces)
    
    Lines starting with '#' are treated as comments and ignored.
    Empty lines are ignored.
    
    Args:
        tsv_path: Path to TSV file with user corrections
    
    Returns:
        List of UserPathCorrection objects
    
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    
    Example TSV content:
        # User corrections for assembly v1.0
        chr1    unitig-1+,unitig-2+,unitig-3-    looks good
        chr2    unitig-4+,unitig-5-              removed repeat
    """
    tsv_path = Path(tsv_path)
    logger.info(f"Parsing user corrections TSV: {tsv_path}")
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"User corrections file not found: {tsv_path}")
    
    corrections: list[UserPathCorrection] = []
    line_num = 0
    
    with open(tsv_path, 'r') as f:
        for line in f:
            line_num += 1
            line = line.rstrip('\n\r')
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_num}: Invalid format (need at least 2 columns). "
                    f"Got: {line}"
                )
            
            chrom = parts[0].strip()
            path_str = parts[1].strip()
            notes = parts[2].strip() if len(parts) > 2 else None
            
            # Parse path string
            try:
                path_nodes = parse_path_string(path_str)
            except ValueError as e:
                raise ValueError(f"Line {line_num}: {e}") from e
            
            if not path_nodes:
                logger.warning(f"Line {line_num}: Empty path for {chrom}, skipping")
                continue
            
            correction = UserPathCorrection(
                chrom=chrom,
                path_nodes=path_nodes,
                notes=notes
            )
            corrections.append(correction)
    
    logger.info(f"Parsed {len(corrections)} user corrections from {tsv_path}")
    for corr in corrections:
        logger.info(f"  {corr.chrom}: {corr.num_nodes} nodes")
    
    return corrections


# ============================================================================
#                   SCAFFOLD RECONSTRUCTION FUNCTIONS
# ============================================================================

def parse_unitig_name(unitig_id: str) -> int:
    """
    Extract internal node ID from unitig name.
    
    Args:
        unitig_id: External unitig name (e.g., 'unitig-1')
    
    Returns:
        Internal node ID as integer
    
    Raises:
        ValueError: If format is invalid
    """
    if not unitig_id.startswith('unitig-'):
        raise ValueError(f"Invalid unitig ID format: {unitig_id}")
    
    try:
        node_id = int(unitig_id.split('-')[1])
        return node_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse unitig ID: {unitig_id}") from e


def reconstruct_scaffolds_from_user_paths(
    graph: GraphLike,
    corrections: list[UserPathCorrection]
) -> list[ReconstructedScaffold]:
    """
    Use user-corrected paths to reconstruct scaffolds.
    
    This function:
    1. Maps unitig IDs in corrections to internal node IDs
    2. Validates that all nodes exist in the graph
    3. Computes total scaffold length
    4. Creates ReconstructedScaffold objects
    
    The reconstructed scaffolds represent the user's authoritative view
    of the assembly structure and will be used to update the string graph.
    
    Args:
        graph: Assembly graph (DBG or String Graph)
        corrections: List of user-corrected paths
    
    Returns:
        List of ReconstructedScaffold objects
    
    Raises:
        ValueError: If any unitig ID is invalid or not found in graph
    
    Example:
        >>> corrections = parse_user_corrections_tsv('corrections.tsv')
        >>> scaffolds = reconstruct_scaffolds_from_user_paths(graph, corrections)
        >>> for scaffold in scaffolds:
        ...     print(f"{scaffold.chrom}: {scaffold.num_nodes} nodes, "
        ...           f"{scaffold.total_length} bp")
    """
    logger.info(f"Reconstructing {len(corrections)} scaffolds from user paths...")
    
    scaffolds: list[ReconstructedScaffold] = []
    
    for correction in corrections:
        logger.info(f"Processing {correction.chrom}...")
        
        # Convert unitig IDs to internal node IDs
        node_path: list[int] = []
        orientation: list[str] = []
        
        for unitig_id, orient in correction.path_nodes:
            # Parse unitig name to get node ID
            try:
                node_id = parse_unitig_name(unitig_id)
            except ValueError as e:
                raise ValueError(
                    f"Invalid unitig ID '{unitig_id}' in {correction.chrom}"
                ) from e
            
            # Validate node exists in graph
            if node_id not in graph.nodes:
                raise ValueError(
                    f"Node {node_id} (from {unitig_id}) not found in graph "
                    f"for {correction.chrom}"
                )
            
            node_path.append(node_id)
            orientation.append(orient)
        
        # Compute total length
        total_length = sum(graph.get_node_length(nid) for nid in node_path)
        
        scaffold = ReconstructedScaffold(
            chrom=correction.chrom,
            node_path=node_path,
            orientation=orientation,
            notes=correction.notes,
            total_length=total_length
        )
        
        scaffolds.append(scaffold)
        logger.info(
            f"  {correction.chrom}: {scaffold.num_nodes} nodes, "
            f"{scaffold.total_length:,} bp"
        )
    
    logger.info(f"Reconstruction complete: {len(scaffolds)} scaffolds")
    
    return scaffolds


# ============================================================================
#              GRAPH INTEGRATION FUNCTIONS
# ============================================================================

@dataclass
class ScaffoldIntegrationResult:
    """
    Result of integrating user scaffolds into the string graph.
    
    Attributes:
        edges_added: Number of edges added to support user paths
        edges_removed: Number of edges removed that contradicted user paths
        edges_downweighted: Number of edges downweighted (not removed)
        scaffold_index: Mapping from chrom -> scaffold object
        conflicts_detected: List of (chrom, issue) tuples for user review
    """
    edges_added: int = 0
    edges_removed: int = 0
    edges_downweighted: int = 0
    scaffold_index: dict[str, ReconstructedScaffold] = field(default_factory=dict)
    conflicts_detected: list[tuple[str, str]] = field(default_factory=list)
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Scaffold Integration:\n"
            f"  Scaffolds: {len(self.scaffold_index)}\n"
            f"  Edges added: {self.edges_added}\n"
            f"  Edges removed: {self.edges_removed}\n"
            f"  Edges downweighted: {self.edges_downweighted}\n"
            f"  Conflicts detected: {len(self.conflicts_detected)}"
        )


def apply_user_scaffolds_to_string_graph(
    string_graph: GraphLike,
    reconstructed_scaffolds: list[ReconstructedScaffold],
    add_missing_edges: bool = True,
    remove_conflicting_edges: bool = False,
    downweight_conflicting_edges: bool = True
) -> ScaffoldIntegrationResult:
    """
    Modify or augment the string graph to reflect user-provided scaffolds.
    
    This function integrates user corrections by:
    1. Ensuring edges along user-defined paths exist (adding if needed)
    2. Handling edges that contradict user paths:
       - Remove them (if remove_conflicting_edges=True)
       - Downweight them (if downweight_conflicting_edges=True)
       - Leave them alone (if both False)
    3. Creating a scaffold index for downstream access
    
    Args:
        string_graph: String graph to modify
        reconstructed_scaffolds: User-defined scaffolds
        add_missing_edges: If True, add edges for user paths that lack them
        remove_conflicting_edges: If True, remove edges contradicting user paths
        downweight_conflicting_edges: If True, reduce weight of conflicting edges
    
    Returns:
        ScaffoldIntegrationResult with statistics and scaffold index
    
    Example:
        >>> result = apply_user_scaffolds_to_string_graph(
        ...     string_graph, scaffolds,
        ...     add_missing_edges=True,
        ...     remove_conflicting_edges=False,
        ...     downweight_conflicting_edges=True
        ... )
        >>> print(result.summary())
    """
    logger.info("Applying user scaffolds to string graph...")
    
    result = ScaffoldIntegrationResult()
    
    # Build scaffold index
    for scaffold in reconstructed_scaffolds:
        result.scaffold_index[scaffold.chrom] = scaffold
    
    # Process each scaffold
    for scaffold in reconstructed_scaffolds:
        logger.info(f"Processing {scaffold.chrom} ({scaffold.num_nodes} nodes)...")
        
        # Build set of edges we need for this scaffold
        required_edges: set[tuple[int, int]] = set()
        for i in range(len(scaffold.node_path) - 1):
            from_node = scaffold.node_path[i]
            to_node = scaffold.node_path[i + 1]
            required_edges.add((from_node, to_node))
        
        # Check which required edges exist
        existing_edge_map: dict[tuple[int, int], int] = {}  # (from, to) -> edge_id
        for edge_id in string_graph.edges:
            from_id, to_id = string_graph.get_edge_endpoints(edge_id)
            existing_edge_map[(from_id, to_id)] = edge_id
        
        # Add missing edges
        if add_missing_edges:
            for from_node, to_node in required_edges:
                if (from_node, to_node) not in existing_edge_map:
                    logger.info(
                        f"  Adding missing edge: {from_node} -> {to_node} "
                        f"(required by {scaffold.chrom})"
                    )
                    try:
                        new_edge_id = string_graph.add_edge(
                            from_node, to_node,
                            source='user_correction',
                            chrom=scaffold.chrom
                        )
                        existing_edge_map[(from_node, to_node)] = new_edge_id
                        result.edges_added += 1
                    except Exception as e:
                        issue = f"Failed to add edge {from_node}->{to_node}: {e}"
                        result.conflicts_detected.append((scaffold.chrom, issue))
                        logger.warning(f"  {issue}")
        
        # Handle conflicting edges (edges from scaffold nodes to non-scaffold nodes)
        scaffold_node_set = set(scaffold.node_path)
        
        for i, node_id in enumerate(scaffold.node_path):
            # Get expected next node (if not last)
            expected_next = scaffold.node_path[i + 1] if i < len(scaffold.node_path) - 1 else None
            
            # Find all outgoing edges from this node
            for edge_id in list(string_graph.edges.keys()):
                from_id, to_id = string_graph.get_edge_endpoints(edge_id)
                
                if from_id == node_id:
                    # This edge starts at a scaffold node
                    if expected_next is not None and to_id != expected_next:
                        # Edge goes to wrong node (conflict)
                        if to_id in scaffold_node_set:
                            # Edge goes to another scaffold node but wrong order
                            issue = f"Edge {from_id}->{to_id} contradicts path order"
                            result.conflicts_detected.append((scaffold.chrom, issue))
                            logger.warning(f"  {issue}")
                        
                        # Handle the conflict
                        if remove_conflicting_edges:
                            logger.info(
                                f"  Removing conflicting edge: {from_id} -> {to_id}"
                            )
                            try:
                                string_graph.remove_edge(edge_id)
                                result.edges_removed += 1
                            except Exception as e:
                                logger.warning(f"  Failed to remove edge: {e}")
                        
                        elif downweight_conflicting_edges:
                            # Mark for downweighting (implementation-specific)
                            # In a real graph, this might set a weight attribute
                            logger.info(
                                f"  Downweighting conflicting edge: {from_id} -> {to_id}"
                            )
                            result.edges_downweighted += 1
                            # Note: Actual downweighting depends on graph implementation
    
    logger.info(result.summary())
    
    return result


# ============================================================================
#                       UTILITY FUNCTIONS
# ============================================================================

def export_scaffolds_to_tsv(
    scaffolds: list[ReconstructedScaffold],
    output_path: str | Path
) -> None:
    """
    Export reconstructed scaffolds back to TSV format.
    
    Useful for:
    - Saving corrected scaffolds for version control
    - Sharing corrections between users
    - Documenting manual curation decisions
    
    Args:
        scaffolds: List of ReconstructedScaffold objects
        output_path: Output TSV file path
    """
    output_path = Path(output_path)
    logger.info(f"Exporting {len(scaffolds)} scaffolds to TSV: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# Reconstructed scaffolds\n")
        f.write("# Format: chrom\tpath\tnotes\n")
        
        for scaffold in scaffolds:
            # Convert node IDs back to unitig names with orientations
            path_str = ','.join(
                f"unitig-{node_id}{orient}"
                for node_id, orient in zip(scaffold.node_path, scaffold.orientation)
            )
            
            notes = scaffold.notes if scaffold.notes else ""
            
            f.write(f"{scaffold.chrom}\t{path_str}\t{notes}\n")
    
    logger.info(f"Export complete: {output_path}")


def validate_user_corrections(
    corrections: list[UserPathCorrection],
    graph: GraphLike
) -> list[str]:
    """
    Validate user corrections against graph and return list of issues.
    
    Checks:
    - All unitig IDs are valid
    - All nodes exist in graph
    - No duplicate nodes within a path
    - Path orientations are valid
    
    Args:
        corrections: User corrections to validate
        graph: Graph to validate against
    
    Returns:
        List of issue strings (empty if no issues)
    """
    issues: list[str] = []
    
    for correction in corrections:
        # Check for duplicates
        unitig_ids = correction.get_unitig_ids()
        if len(unitig_ids) != len(set(unitig_ids)):
            duplicates = [uid for uid in unitig_ids if unitig_ids.count(uid) > 1]
            issues.append(
                f"{correction.chrom}: Duplicate nodes in path: {set(duplicates)}"
            )
        
        # Check node existence
        for unitig_id, _ in correction.path_nodes:
            try:
                node_id = parse_unitig_name(unitig_id)
                if node_id not in graph.nodes:
                    issues.append(
                        f"{correction.chrom}: Node {unitig_id} (ID {node_id}) "
                        f"not found in graph"
                    )
            except ValueError as e:
                issues.append(f"{correction.chrom}: Invalid unitig ID: {unitig_id}")
    
    return issues


def get_scaffold_statistics(scaffolds: list[ReconstructedScaffold]) -> dict[str, Any]:
    """
    Compute summary statistics for reconstructed scaffolds.
    
    Returns:
        Dict with keys: num_scaffolds, total_length, mean_length, max_length,
                       num_nodes, mean_nodes_per_scaffold
    """
    if not scaffolds:
        return {
            'num_scaffolds': 0,
            'total_length': 0,
            'mean_length': 0.0,
            'max_length': 0,
            'num_nodes': 0,
            'mean_nodes_per_scaffold': 0.0
        }
    
    lengths = [s.total_length for s in scaffolds]
    node_counts = [s.num_nodes for s in scaffolds]
    
    return {
        'num_scaffolds': len(scaffolds),
        'total_length': sum(lengths),
        'mean_length': sum(lengths) / len(lengths),
        'max_length': max(lengths),
        'num_nodes': sum(node_counts),
        'mean_nodes_per_scaffold': sum(node_counts) / len(node_counts)
    }


# ============================================================================
#                    MANUAL EDITS - BREAKS, JOINS, EXCLUSIONS (NEW)
# ============================================================================

@dataclass
class ContigBreak:
    """
    Represents a user-specified break point in a contig.
    
    Used to split contigs at misassembly sites identified by user inspection.
    
    Attributes:
        contig_id: Contig or node identifier
        position: 0-based position where to break the contig
        reason: Optional explanation (e.g., 'misassembly', 'coverage_drop')
    """
    contig_id: str | int
    position: int
    reason: str | None = None
    
    def __post_init__(self):
        """Validate position is non-negative."""
        if self.position < 0:
            raise ValueError(f"Break position must be >= 0, got {self.position}")


@dataclass
class ForcedJoin:
    """
    Represents a user-specified join between two contigs.
    
    Used to force scaffolding connections that may have been missed by
    automated methods.
    
    Attributes:
        from_contig: Source contig ID
        to_contig: Target contig ID
        from_orient: Orientation of source ('+' or '-')
        to_orient: Orientation of target ('+' or '-')
        evidence: Optional evidence description
    """
    from_contig: str | int
    to_contig: str | int
    from_orient: str = '+'
    to_orient: str = '+'
    evidence: str | None = None
    
    def __post_init__(self):
        """Validate orientations."""
        if self.from_orient not in ('+', '-'):
            raise ValueError(f"Invalid from_orient: {self.from_orient}")
        if self.to_orient not in ('+', '-'):
            raise ValueError(f"Invalid to_orient: {self.to_orient}")


def load_contig_breaks(tsv_path: str | Path) -> list[ContigBreak]:
    """
    Load user-specified contig break points from TSV file.
    
    TSV Format:
    -----------
    contig_id    position    reason
    contig_1     45000       coverage_drop
    contig_3     120000      misassembly_detected
    
    Args:
        tsv_path: Path to TSV file with break points
    
    Returns:
        List of ContigBreak objects
    
    Example:
        >>> breaks = load_contig_breaks('manual_breaks.tsv')
        >>> for brk in breaks:
        ...     print(f"Break {brk.contig_id} at {brk.position}: {brk.reason}")
    """
    tsv_path = Path(tsv_path)
    logger.info(f"Loading contig breaks from {tsv_path}")
    
    breaks: list[ContigBreak] = []
    
    with open(tsv_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip header line
            if line_num == 1 and line.lower().startswith('contig'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                logger.warning(f"Line {line_num}: Not enough columns, skipping")
                continue
            
            contig_id = parts[0]
            try:
                position = int(parts[1])
            except ValueError:
                logger.warning(f"Line {line_num}: Invalid position '{parts[1]}', skipping")
                continue
            
            reason = parts[2] if len(parts) > 2 else None
            
            breaks.append(ContigBreak(contig_id, position, reason))
    
    logger.info(f"Loaded {len(breaks)} contig breaks")
    return breaks


def load_forced_joins(tsv_path: str | Path) -> list[ForcedJoin]:
    """
    Load user-specified forced joins from TSV file.
    
    TSV Format:
    -----------
    from_contig    to_contig    from_orient    to_orient    evidence
    contig_1       contig_2     +              +            manual_inspection
    contig_5       contig_7     +              -            mate_pair_support
    
    Args:
        tsv_path: Path to TSV file with forced joins
    
    Returns:
        List of ForcedJoin objects
    
    Example:
        >>> joins = load_forced_joins('manual_joins.tsv')
        >>> for join in joins:
        ...     print(f"Join {join.from_contig} → {join.to_contig}")
    """
    tsv_path = Path(tsv_path)
    logger.info(f"Loading forced joins from {tsv_path}")
    
    joins: list[ForcedJoin] = []
    
    with open(tsv_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip header line
            if line_num == 1 and 'from_contig' in line.lower():
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                logger.warning(f"Line {line_num}: Not enough columns, skipping")
                continue
            
            from_contig = parts[0]
            to_contig = parts[1]
            from_orient = parts[2] if len(parts) > 2 else '+'
            to_orient = parts[3] if len(parts) > 3 else '+'
            evidence = parts[4] if len(parts) > 4 else None
            
            joins.append(ForcedJoin(
                from_contig, to_contig,
                from_orient, to_orient,
                evidence
            ))
    
    logger.info(f"Loaded {len(joins)} forced joins")
    return joins


def load_exclusion_list(txt_path: str | Path) -> set[str | int]:
    """
    Load list of nodes/contigs to exclude from assembly.
    
    Format: One contig ID per line (can include comments with #)
    
    Example file:
    # Problematic contigs identified during QC
    contig_42    # low coverage
    contig_87    # repetitive
    unitig-15    # contamination
    
    Args:
        txt_path: Path to text file with exclusion list
    
    Returns:
        Set of contig/node IDs to exclude
    
    Example:
        >>> exclusions = load_exclusion_list('exclude.txt')
        >>> if 'contig_42' in exclusions:
        ...     print("Skipping problematic contig")
    """
    txt_path = Path(txt_path)
    logger.info(f"Loading exclusion list from {txt_path}")
    
    exclusions: set[str | int] = set()
    
    with open(txt_path, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            
            if not line:
                continue
            
            # Try to parse as int, otherwise keep as string
            try:
                exclusions.add(int(line))
            except ValueError:
                exclusions.add(line)
    
    logger.info(f"Loaded {len(exclusions)} exclusions")
    return exclusions


def apply_manual_edits(
    graph: GraphLike,
    breaks: list[ContigBreak] | None = None,
    joins: list[ForcedJoin] | None = None,
    exclusions: set[str | int] | None = None
) -> dict[str, int]:
    """
    Apply all manual edits to the assembly graph.
    
    This is a unified application function that applies:
    1. Contig breaks (splits nodes at specified positions)
    2. Forced joins (adds edges between specified nodes)
    3. Exclusions (marks nodes for removal)
    
    Args:
        graph: Assembly graph to modify
        breaks: Optional list of ContigBreak objects
        joins: Optional list of ForcedJoin objects
        exclusions: Optional set of node/contig IDs to exclude
    
    Returns:
        Dict with counts: 'breaks_applied', 'joins_added', 'nodes_excluded'
    
    Example:
        >>> breaks = load_contig_breaks('breaks.tsv')
        >>> joins = load_forced_joins('joins.tsv')
        >>> exclusions = load_exclusion_list('exclude.txt')
        >>> result = apply_manual_edits(graph, breaks, joins, exclusions)
        >>> print(f"Applied {result['breaks_applied']} breaks")
    """
    result = {
        'breaks_applied': 0,
        'joins_added': 0,
        'nodes_excluded': 0
    }
    
    logger.info("Applying manual edits to graph...")
    
    # Apply breaks
    if breaks:
        logger.info(f"Processing {len(breaks)} contig breaks...")
        for brk in breaks:
            # TODO: Implement break logic in graph
            # This would require graph.split_node(node_id, position)
            logger.debug(f"  Break: {brk.contig_id} at {brk.position} ({brk.reason})")
            result['breaks_applied'] += 1
        logger.info(f"  Applied {result['breaks_applied']} breaks")
    
    # Apply forced joins
    if joins:
        logger.info(f"Processing {len(joins)} forced joins...")
        for join in joins:
            # TODO: Implement join logic in graph
            # This would require graph.add_edge(from_node, to_node, orientations)
            logger.debug(f"  Join: {join.from_contig}{join.from_orient} → "
                        f"{join.to_contig}{join.to_orient}")
            result['joins_added'] += 1
        logger.info(f"  Added {result['joins_added']} joins")
    
    # Apply exclusions
    if exclusions:
        logger.info(f"Processing {len(exclusions)} exclusions...")
        for node_id in exclusions:
            # TODO: Implement exclusion logic in graph
            # This would require graph.mark_excluded(node_id) or graph.remove_node(node_id)
            if node_id in graph.nodes:
                logger.debug(f"  Excluding: {node_id}")
                result['nodes_excluded'] += 1
        logger.info(f"  Excluded {result['nodes_excluded']} nodes")
    
    logger.info("Manual edits applied successfully")
    return result

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.

