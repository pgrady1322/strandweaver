"""
Complete Training Pipeline Demo

Demonstrates the full end-to-end pipeline:
1. Generate synthetic diploid genome
2. Simulate reads
3. Build assembly graph
4. Extract ground-truth labels
5. Build training features
6. Export training dataset

This shows the complete flow from synthetic data to ML-ready training examples.

Usage:
    python examples/complete_training_pipeline.py

Author: StrandWeaver Training Infrastructure
Date: December 2025
"""

import sys
from pathlib import Path
import json

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
    generate_ground_truth_labels
)
from strandweaver.training.feature_builder import (
    build_training_dataset
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_assembly_graph(reads):
    """Create a mock assembly graph from reads."""
    logger.info("Creating mock assembly graph...")
    
    # Use read IDs as node IDs
    nodes = [read.read_id for read in reads[:50]]
    
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
    print("COMPLETE TRAINING PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Generate Synthetic Genome
    # ========================================================================
    
    print("STEP 1: Generating synthetic diploid genome...")
    print("-" * 80)
    
    genome_config = GenomeConfig(
        length=100_000,
        gc_content=0.42,
        repeat_density=0.20,
        snp_rate=0.001,
        sv_density=0.0001,
        sv_min_size=100,
        sv_max_size=5000,
        random_seed=42
    )
    
    diploid = generate_diploid_genome(genome_config)
    
    print(f"✅ Genome: {len(diploid.hapA):,} bp (hap A), {len(diploid.hapB):,} bp (hap B)")
    print(f"   SVs: {len(diploid.sv_truth_table)}")
    print()
    
    # ========================================================================
    # STEP 2: Simulate Reads
    # ========================================================================
    
    print("STEP 2: Simulating reads...")
    print("-" * 80)
    
    hifi_config = HiFiConfig(coverage=20.0, random_seed=42)
    hifi_A = simulate_long_reads(diploid.hapA, hifi_config, haplotype='A', read_type='hifi')
    hifi_B = simulate_long_reads(diploid.hapB, hifi_config, haplotype='B', read_type='hifi')
    
    ul_config = ULConfig(coverage=5.0, random_seed=42)
    ul_A = simulate_long_reads(diploid.hapA, ul_config, haplotype='A', read_type='ul')
    ul_B = simulate_long_reads(diploid.hapB, ul_config, haplotype='B', read_type='ul')
    
    all_reads = hifi_A + hifi_B
    ul_reads = ul_A + ul_B
    
    print(f"✅ Reads: {len(all_reads)} HiFi, {len(ul_reads)} UL")
    print()
    
    # ========================================================================
    # STEP 3: Build Assembly Graph
    # ========================================================================
    
    print("STEP 3: Building assembly graph...")
    print("-" * 80)
    
    graph_nodes, graph_edges = create_mock_assembly_graph(all_reads)
    
    print(f"✅ Graph: {len(graph_nodes)} nodes, {len(graph_edges)} edges")
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
    print(f"✅ Labels: {len(labels.edge_labels)} edges, {len(labels.node_labels)} nodes, "
          f"{len(labels.sv_labels)} SVs")
    print()
    
    # ========================================================================
    # STEP 5: Build Training Features
    # ========================================================================
    
    print("STEP 5: Building training features...")
    print("-" * 80)
    
    dataset = build_training_dataset(
        simulated_reads=all_reads,
        ul_reads=ul_reads,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        ground_truth_labels=labels
    )
    
    print()
    print(f"✅ Training Dataset:")
    print(f"   Overlap classifier: {len(dataset.overlap_features)} examples")
    print(f"   GNN tensors: {dataset.gnn_tensors.num_nodes} nodes, {dataset.gnn_tensors.num_edges} edges")
    print(f"   Diploid features: {len(dataset.diploid_features)} examples")
    print(f"   UL routing: {len(dataset.ul_routing_features)} examples")
    print(f"   SV detection: {len(dataset.sv_features)} examples")
    print()
    
    # ========================================================================
    # STEP 6: Analyze Features
    # ========================================================================
    
    print("STEP 6: Analyzing extracted features...")
    print("-" * 80)
    
    # Overlap classifier feature dimensions
    if dataset.overlap_features:
        sample_overlap = dataset.overlap_features[0]
        overlap_vec = sample_overlap.to_feature_vector()
        print(f"Overlap feature vector: {len(overlap_vec)} dimensions")
        print(f"  Example: {overlap_vec[:5]}... (first 5 values)")
        print(f"  Label: {sample_overlap.label}")
        print()
    
    # GNN tensors shape
    if dataset.gnn_tensors:
        print(f"GNN node features: shape {dataset.gnn_tensors.node_features.shape}")
        print(f"GNN edge index: shape {dataset.gnn_tensors.edge_index.shape}")
        print(f"GNN node labels: shape {dataset.gnn_tensors.node_labels.shape}")
        print(f"GNN paths: {len(dataset.gnn_tensors.path_sequences)} sequences")
        print()
    
    # Diploid features
    if dataset.diploid_features:
        sample_diploid = dataset.diploid_features[0]
        diploid_vec = sample_diploid.to_feature_vector()
        print(f"Diploid feature vector: {len(diploid_vec)} dimensions")
        print(f"  Example: {diploid_vec[:5]}... (first 5 values)")
        print(f"  True haplotype: {sample_diploid.true_haplotype}")
        print()
    
    # UL routing features
    if dataset.ul_routing_features:
        sample_ul = dataset.ul_routing_features[0]
        ul_vec = sample_ul.to_feature_vector()
        print(f"UL routing feature vector: {len(ul_vec)} dimensions (12D as specified)")
        print(f"  Values: {ul_vec}")
        print(f"  Is correct path: {sample_ul.is_correct_path}")
        print()
    
    # SV detection features
    if dataset.sv_features:
        sample_sv = dataset.sv_features[0]
        sv_vec = sample_sv.to_feature_vector()
        print(f"SV detection feature vector: {len(sv_vec)} dimensions")
        print(f"  Values: {sv_vec}")
        print(f"  True SV type: {sample_sv.true_sv_type}")
        print(f"  True SV size: {sample_sv.true_sv_size:,} bp")
        print()
    
    # ========================================================================
    # STEP 7: Export Sample Dataset
    # ========================================================================
    
    print("STEP 7: Exporting sample dataset...")
    print("-" * 80)
    
    output_dir = Path("training_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Export overlap classifier examples
    overlap_file = output_dir / "overlap_examples.jsonl"
    with open(overlap_file, 'w') as f:
        for features in dataset.overlap_features[:100]:  # First 100
            f.write(json.dumps(features.to_dict()) + '\n')
    print(f"✅ Wrote {min(100, len(dataset.overlap_features))} overlap examples to {overlap_file}")
    
    # Export diploid examples
    diploid_file = output_dir / "diploid_examples.jsonl"
    with open(diploid_file, 'w') as f:
        for features in dataset.diploid_features[:100]:
            example = {
                'node_id': features.node_id,
                'features': features.to_feature_vector().tolist(),
                'label': features.true_haplotype
            }
            f.write(json.dumps(example) + '\n')
    print(f"✅ Wrote {min(100, len(dataset.diploid_features))} diploid examples to {diploid_file}")
    
    # Export SV examples
    sv_file = output_dir / "sv_examples.jsonl"
    with open(sv_file, 'w') as f:
        for features in dataset.sv_features:
            example = {
                'region_id': features.region_id,
                'position': f"{features.chrom}:{features.start_pos}-{features.end_pos}",
                'features': features.to_feature_vector().tolist(),
                'sv_type': features.true_sv_type,
                'sv_size': features.true_sv_size
            }
            f.write(json.dumps(example) + '\n')
    print(f"✅ Wrote {len(dataset.sv_features)} SV examples to {sv_file}")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("=" * 80)
    print("COMPLETE TRAINING PIPELINE SUCCESS")
    print("=" * 80)
    print()
    print("Pipeline Summary:")
    print(f"  1. Generated {len(diploid.hapA):,} bp diploid genome")
    print(f"  2. Simulated {len(all_reads)} reads")
    print(f"  3. Built graph with {len(graph_nodes)} nodes")
    print(f"  4. Extracted {len(labels.edge_labels)} ground-truth labels")
    print(f"  5. Built {len(dataset.overlap_features)} training examples")
    print()
    print("Output directory: training_dataset/")
    print("  - overlap_examples.jsonl (edge classification)")
    print("  - diploid_examples.jsonl (haplotype assignment)")
    print("  - sv_examples.jsonl (SV detection)")
    print()
    print("Next steps:")
    print("  1. Implement dataset writer with sharding")
    print("  2. Train ML models on extracted features")
    print("  3. Evaluate on held-out test sets")
    print("  4. Deploy trained models to production pipeline")
    print()


if __name__ == "__main__":
    main()
