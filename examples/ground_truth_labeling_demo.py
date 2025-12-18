"""
Complete Ground-Truth Labeling Demo

Demonstrates the full pipeline:
1. Generate synthetic diploid genome
2. Simulate reads (HiFi + UL)
3. Build mock assembly graph
4. Extract ground-truth labels
5. Export labels for ML training

This shows how synthetic data flows into labeled training examples.

Usage:
    python examples/ground_truth_labeling_demo.py

Author: StrandWeaver Training Infrastructure
Date: December 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.genome_simulator import (
    GenomeConfig,
    generate_diploid_genome
)
from strandweaver.training.read_simulator import (
    HiFiConfig,
    ULConfig,
    simulate_long_reads
)
from strandweaver.training.truth_labeler import (
    generate_ground_truth_labels,
    export_labels_to_tsv,
    EdgeLabel
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_assembly_graph(reads):
    """
    Create a mock assembly graph from reads.
    
    In a real pipeline, this would:
    1. Build de Bruijn graph
    2. Simplify to string graph
    3. Extract nodes and edges
    
    For demo, we create a simplified graph.
    """
    logger.info("Creating mock assembly graph...")
    
    # Use read IDs as node IDs
    nodes = [read.read_id for read in reads[:50]]  # First 50 reads
    
    # Create edges between consecutive reads (by position)
    edges = []
    sorted_reads = sorted(reads[:50], key=lambda r: r.start_pos)
    
    for i in range(len(sorted_reads) - 1):
        r1 = sorted_reads[i]
        r2 = sorted_reads[i + 1]
        
        # Only add edge if reads overlap or are close
        if r2.start_pos - r1.end_pos < 5000:
            edges.append((r1.read_id, r2.read_id))
    
    logger.info(f"Mock graph: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


def main():
    print("=" * 80)
    print("GROUND-TRUTH LABELING PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Generate Synthetic Diploid Genome
    # ========================================================================
    
    print("STEP 1: Generating synthetic diploid genome...")
    print("-" * 80)
    
    genome_config = GenomeConfig(
        length=100_000,  # Small 100kb genome for fast demo
        gc_content=0.42,
        repeat_density=0.20,  # Lower repeat density
        snp_rate=0.001,
        sv_density=0.0001,  # Higher SV density for demo
        sv_min_size=100,   # Smaller SVs for small genome
        sv_max_size=5000,  # Smaller max size for small genome
        random_seed=42
    )
    
    diploid = generate_diploid_genome(genome_config)
    
    print(f"✅ Generated diploid genome:")
    print(f"   Haplotype A: {len(diploid.hapA):,} bp")
    print(f"   Haplotype B: {len(diploid.hapB):,} bp")
    print(f"   SVs: {len(diploid.sv_truth_table)}")
    print()
    
    # ========================================================================
    # STEP 2: Simulate Reads
    # ========================================================================
    
    print("STEP 2: Simulating HiFi and ultralong reads...")
    print("-" * 80)
    
    # Simulate HiFi reads
    hifi_config = HiFiConfig(coverage=20.0, random_seed=42)
    hifi_A = simulate_long_reads(diploid.hapA, hifi_config, haplotype='A', read_type='hifi')
    hifi_B = simulate_long_reads(diploid.hapB, hifi_config, haplotype='B', read_type='hifi')
    
    # Simulate UL reads
    ul_config = ULConfig(coverage=5.0, random_seed=42)
    ul_A = simulate_long_reads(diploid.hapA, ul_config, haplotype='A', read_type='ul')
    ul_B = simulate_long_reads(diploid.hapB, ul_config, haplotype='B', read_type='ul')
    
    all_reads = hifi_A + hifi_B
    ul_reads = ul_A + ul_B
    
    print(f"✅ Simulated reads:")
    print(f"   HiFi: {len(hifi_A)} (hap A) + {len(hifi_B)} (hap B) = {len(all_reads)} total")
    print(f"   UL: {len(ul_A)} (hap A) + {len(ul_B)} (hap B) = {len(ul_reads)} total")
    print()
    
    # ========================================================================
    # STEP 3: Build Mock Assembly Graph
    # ========================================================================
    
    print("STEP 3: Building mock assembly graph...")
    print("-" * 80)
    
    graph_nodes, graph_edges = create_mock_assembly_graph(all_reads)
    
    print(f"✅ Assembly graph:")
    print(f"   Nodes: {len(graph_nodes)}")
    print(f"   Edges: {len(graph_edges)}")
    print()
    
    # ========================================================================
    # STEP 4: Generate Ground-Truth Labels
    # ========================================================================
    
    print("STEP 4: Generating ground-truth labels...")
    print("-" * 80)
    
    labels = generate_ground_truth_labels(
        simulated_reads=all_reads,
        ul_reads=ul_reads,
        reference_hapA=diploid.hapA,
        reference_hapB=diploid.hapB,
        sv_truth_table=diploid.sv_truth_table,
        graph_edges=graph_edges,
        graph_nodes=graph_nodes
    )
    
    print()
    print(f"✅ Generated ground-truth labels:")
    print(f"   Edge labels: {len(labels.edge_labels)}")
    print(f"   Node labels: {len(labels.node_labels)}")
    print(f"   Path labels: {len(labels.path_labels)}")
    print(f"   UL route labels: {len(labels.ul_route_labels)}")
    print(f"   SV labels: {len(labels.sv_labels)}")
    print()
    
    # ========================================================================
    # STEP 5: Analyze Edge Labels
    # ========================================================================
    
    print("STEP 5: Analyzing edge label distribution...")
    print("-" * 80)
    
    edge_label_counts = {}
    for label_obj in labels.edge_labels.values():
        label_type = label_obj.label
        edge_label_counts[label_type] = edge_label_counts.get(label_type, 0) + 1
    
    print("Edge label distribution:")
    for label_type, count in sorted(edge_label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(labels.edge_labels)
        print(f"   {label_type.value:12s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Show example edge labels
    print("Example edge labels (first 5):")
    for i, ((source, target), label) in enumerate(list(labels.edge_labels.items())[:5]):
        print(f"   {source} → {target}")
        print(f"      Label: {label.label.value}")
        print(f"      Explanation: {label.explanation}")
        if label.overlap_distance is not None:
            print(f"      Distance: {label.overlap_distance:,} bp")
        print()
    
    # ========================================================================
    # STEP 6: Analyze Node Haplotypes
    # ========================================================================
    
    print("STEP 6: Analyzing node haplotype assignments...")
    print("-" * 80)
    
    haplotype_counts = {}
    for node_label in labels.node_labels.values():
        hap = node_label.haplotype
        haplotype_counts[hap] = haplotype_counts.get(hap, 0) + 1
    
    print("Node haplotype distribution:")
    for hap_type, count in sorted(haplotype_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(labels.node_labels)
        print(f"   {hap_type.value:12s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # ========================================================================
    # STEP 7: Analyze SV Graph Signatures
    # ========================================================================
    
    print("STEP 7: Analyzing SV graph signatures...")
    print("-" * 80)
    
    if labels.sv_labels:
        print(f"Found {len(labels.sv_labels)} structural variants:")
        for sv_label in labels.sv_labels[:5]:  # Show first 5
            print(f"   {sv_label.sv_id}:")
            print(f"      Type: {sv_label.sv_type.value}")
            print(f"      Haplotype: {sv_label.haplotype}")
            print(f"      Position: {sv_label.ref_chrom}:{sv_label.ref_start:,}-{sv_label.ref_end:,}")
            print(f"      Size: {sv_label.size:,} bp")
            print(f"      Spanning reads: {sv_label.graph_signature.get('num_spanning', 0)}")
            print(f"      Expected pattern: {sv_label.graph_signature.get('expected_pattern', 'unknown')}")
            print()
    else:
        print("   No SVs in this genome (try increasing sv_density)")
        print()
    
    # ========================================================================
    # STEP 8: Export Labels
    # ========================================================================
    
    print("STEP 8: Exporting labels to TSV files...")
    print("-" * 80)
    
    output_dir = Path("ground_truth_labels")
    export_labels_to_tsv(labels, output_dir)
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("=" * 80)
    print("GROUND-TRUTH LABELING COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Input: {len(all_reads):,} reads, {len(ul_reads)} UL reads")
    print(f"  Graph: {len(graph_nodes)} nodes, {len(graph_edges)} edges")
    print(f"  Labels: {len(labels.edge_labels)} edges, {len(labels.node_labels)} nodes")
    print(f"  SVs: {len(labels.sv_labels)} structural variants")
    print()
    print(f"  Output directory: {output_dir.absolute()}")
    print()
    print("Generated files:")
    print(f"  - edge_labels.tsv (edge classifications)")
    print(f"  - node_labels.tsv (haplotype assignments)")
    print(f"  - sv_labels.tsv (SV graph signatures)")
    print()
    print("Next steps:")
    print("  1. Build training feature extractor")
    print("  2. Convert labels → ML-ready tensors")
    print("  3. Train models on labeled data")
    print()


if __name__ == "__main__":
    main()
