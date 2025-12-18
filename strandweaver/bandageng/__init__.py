"""
BandageNG Integration Module

This module provides complete integration with BandageNG for user-guided
assembly refinement. It enables users to:

1. Export assembly graphs to GFA format with coverage overlays
2. Manually edit paths and scaffolds in BandageNG
3. Import user corrections back into the pipeline
4. Reconstruct scaffolds based on user edits
5. Update the assembly graph to reflect corrections

Workflow:
---------
1. Export: Generate GFA + coverage CSVs for visualization
2. Edit: User manually refines paths in BandageNG
3. Import: Parse user-corrected paths from TSV
4. Reconstruct: Build scaffolds from user paths
5. Integrate: Update graph with user corrections

BandageNG (https://github.com/rrwick/BandageNG) is a powerful graph
visualization tool that allows inspection and manual curation of assembly graphs.

Author: StrandWeaver Development Team
Date: December 2025
Phase: 5.2 - User-Guided Assembly Refinement
"""

# Export functions
from strandweaver.bandageng.bandageng_export import (
    # Main export functions
    export_graph_to_gfa,
    export_coverage_csv,
    export_for_bandageng,
    
    # Data structures
    GFASegment,
    GFALink,
    
    # Utilities
    generate_unitig_name,
    parse_unitig_name,
    validate_gfa_file,
    get_node_name_mapping,
    get_reverse_node_mapping,
)

# Import and reconstruction functions
from strandweaver.bandageng.user_corrections import (
    # Data structures
    UserPathCorrection,
    ReconstructedScaffold,
    ScaffoldIntegrationResult,
    
    # TSV parsing
    parse_path_string,
    parse_user_corrections_tsv,
    
    # Scaffold reconstruction
    reconstruct_scaffolds_from_user_paths,
    apply_user_scaffolds_to_string_graph,
    
    # Utilities
    export_scaffolds_to_tsv,
    validate_user_corrections,
    get_scaffold_statistics,
)

__all__ = [
    # Export functions
    'export_graph_to_gfa',
    'export_coverage_csv',
    'export_for_bandageng',
    'GFASegment',
    'GFALink',
    'generate_unitig_name',
    'parse_unitig_name',
    'validate_gfa_file',
    'get_node_name_mapping',
    'get_reverse_node_mapping',
    
    # Import and reconstruction
    'UserPathCorrection',
    'ReconstructedScaffold',
    'ScaffoldIntegrationResult',
    'parse_path_string',
    'parse_user_corrections_tsv',
    'reconstruct_scaffolds_from_user_paths',
    'apply_user_scaffolds_to_string_graph',
    'export_scaffolds_to_tsv',
    'validate_user_corrections',
    'get_scaffold_statistics',
]

__version__ = '0.1.0'
