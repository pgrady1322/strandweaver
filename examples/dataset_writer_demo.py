"""
Dataset writer demonstration.

This script shows how to package training features into production-ready
datasets with automatic sharding, multiple format support, and train/val/test
splitting.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.genome_simulator import (
    generate_diploid_genome,
    GenomeConfig
)
from strandweaver.training.read_simulator import (
    simulate_long_reads,
    HiFiConfig,
    ULConfig
)
from strandweaver.training.truth_labeler import generate_ground_truth_labels
from strandweaver.training.feature_builder import build_training_dataset
from strandweaver.training.dataset_writer import (
    write_sharded_dataset,
    create_train_val_test_split,
    merge_dataset_shards,
    DatasetMetadata
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_mock_assembly_graph(reads, num_nodes=50):
    """Create a mock assembly graph for demonstration."""
    import random
    
    # Create nodes from reads (just use read IDs as node IDs)
    nodes = [read.read_id for read in reads[:num_nodes]]
    
    # Create edges
    edges = []
    for i in range(len(nodes) - 1):
        edges.append((nodes[i], nodes[i + 1]))
    
    return nodes, edges


def main():
    """Run dataset writer demonstration."""
    
    print("=" * 80)
    print("DATASET WRITER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic data
    print("STEP 1: Generating training data...")
    print("-" * 80)
    
    genome_config = GenomeConfig(
        length=200_000,  # 200kb genome for larger dataset
        gc_content=0.42,
        repeat_density=0.20,
        sv_density=0.0001,
        sv_min_size=50,
        sv_max_size=5000,  # Smaller SVs for 200kb genome
        snp_rate=0.001,
        indel_rate=0.0001
    )
    
    diploid = generate_diploid_genome(genome_config)
    
    # Simulate reads
    hifi_config = HiFiConfig(coverage=30.0)  # Higher coverage for more examples
    hifi_A = simulate_long_reads(diploid.hapA, hifi_config, read_type='hifi', haplotype='A')
    hifi_B = simulate_long_reads(diploid.hapB, hifi_config, read_type='hifi', haplotype='B')
    all_reads = hifi_A + hifi_B
    
    ul_config = ULConfig(coverage=5.0)
    ul_A = simulate_long_reads(diploid.hapA, ul_config, read_type='ul', haplotype='A')
    ul_B = simulate_long_reads(diploid.hapB, ul_config, read_type='ul', haplotype='B')
    ul_reads = ul_A + ul_B
    
    print(f"✅ Generated {len(all_reads)} HiFi reads, {len(ul_reads)} UL reads")
    print()
    
    # Step 2: Build graph and extract labels
    print("STEP 2: Building graph and extracting labels...")
    print("-" * 80)
    
    graph_nodes, graph_edges = create_mock_assembly_graph(all_reads, num_nodes=100)
    
    labels = generate_ground_truth_labels(
        simulated_reads=all_reads,
        ul_reads=ul_reads,
        reference_hapA=diploid.hapA,
        reference_hapB=diploid.hapB,
        sv_truth_table=diploid.sv_truth_table,
        graph_edges=graph_edges,
        graph_nodes=graph_nodes
    )
    
    print(f"✅ Labeled {len(labels.edge_labels)} edges, {len(labels.node_labels)} nodes")
    print()
    
    # Step 3: Build training features
    print("STEP 3: Building training features...")
    print("-" * 80)
    
    dataset = build_training_dataset(
        simulated_reads=all_reads,
        ul_reads=ul_reads,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        ground_truth_labels=labels
    )
    
    print(f"✅ Built dataset:")
    print(f"   Overlap features: {len(dataset.overlap_features)}")
    print(f"   Diploid features: {len(dataset.diploid_features)}")
    print(f"   SV features: {len(dataset.sv_features)}")
    print()
    
    # Step 4: Write dataset with sharding (JSONL format)
    print("STEP 4: Writing sharded dataset (JSONL format)...")
    print("-" * 80)
    
    output_dir = Path("training_datasets/demo_v1_jsonl")
    metadata = write_sharded_dataset(
        dataset=dataset,
        output_dir=output_dir,
        format="jsonl",
        shard_size=25,  # Small shards for demo
        version="v1_demo",
        description="Demo dataset showing JSONL sharding with 25 examples per shard"
    )
    
    print(f"✅ Wrote {metadata.num_shards} shard(s) to {output_dir}")
    print()
    
    # Step 5: Write dataset with sharding (NPZ format)
    print("STEP 5: Writing sharded dataset (NPZ format)...")
    print("-" * 80)
    
    output_dir_npz = Path("training_datasets/demo_v1_npz")
    metadata_npz = write_sharded_dataset(
        dataset=dataset,
        output_dir=output_dir_npz,
        format="npz",
        shard_size=25,
        version="v1_demo",
        description="Demo dataset showing NumPy NPZ sharding for efficient loading"
    )
    
    print(f"✅ Wrote {metadata_npz.num_shards} shard(s) to {output_dir_npz}")
    print()
    
    # Step 6: Write all formats
    print("STEP 6: Writing dataset in all formats...")
    print("-" * 80)
    
    output_dir_all = Path("training_datasets/demo_v1_all")
    metadata_all = write_sharded_dataset(
        dataset=dataset,
        output_dir=output_dir_all,
        format="all",
        shard_size=25,
        version="v1_demo",
        description="Demo dataset with both JSONL and NPZ formats"
    )
    
    print(f"✅ Wrote dataset in all formats to {output_dir_all}")
    print()
    
    # Step 7: Create train/val/test split
    print("STEP 7: Creating train/val/test split...")
    print("-" * 80)
    
    split_dir = Path("training_datasets/demo_split")
    create_train_val_test_split(
        dataset=dataset,
        output_dir=split_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        format="jsonl",
        version="v1_demo"
    )
    
    print(f"✅ Created train/val/test split in {split_dir}")
    print()
    
    # Step 8: Merge shards (demonstration)
    print("STEP 8: Merging shards (example)...")
    print("-" * 80)
    
    overlap_dir = output_dir / "overlap_classifier"
    merged_file = Path("training_datasets/merged_overlap.jsonl")
    merge_dataset_shards(
        input_dir=overlap_dir,
        output_file=merged_file,
        format="jsonl"
    )
    
    print(f"✅ Merged shards to {merged_file}")
    print()
    
    # Step 9: Show dataset statistics
    print("STEP 9: Dataset statistics...")
    print("-" * 80)
    
    # Load and display metadata
    metadata_loaded = DatasetMetadata.load(output_dir / "metadata.json")
    
    print(f"Version: {metadata_loaded.version}")
    print(f"Total examples: {metadata_loaded.num_examples}")
    print(f"Number of shards: {metadata_loaded.num_shards}")
    print(f"Shard size: {metadata_loaded.shard_size}")
    print(f"Format: {metadata_loaded.format}")
    print(f"Created: {metadata_loaded.created_at}")
    print()
    print("Feature dimensions:")
    for model, dims in metadata_loaded.feature_dimensions.items():
        print(f"  {model}: {dims}D")
    print()
    
    # Step 10: Show directory structure
    print("STEP 10: Dataset directory structure...")
    print("-" * 80)
    
    import subprocess
    result = subprocess.run(
        ["find", str(output_dir), "-type", "f", "-name", "*"],
        capture_output=True,
        text=True
    )
    
    files = sorted(result.stdout.strip().split('\n'))
    print(f"Total files: {len(files)}")
    print()
    print("Sample files:")
    for f in files[:15]:  # Show first 15 files
        rel_path = Path(f).relative_to(output_dir)
        print(f"  {rel_path}")
    
    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more files")
    print()
    
    print("=" * 80)
    print("DATASET WRITER DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✅ Created sharded datasets in JSONL, NPZ, and combined formats")
    print(f"  ✅ Generated train/val/test splits (70%/15%/15%)")
    print(f"  ✅ Demonstrated shard merging")
    print(f"  ✅ Exported metadata and README files")
    print()
    print("Output directories:")
    print(f"  - {output_dir} (JSONL format)")
    print(f"  - {output_dir_npz} (NPZ format)")
    print(f"  - {output_dir_all} (All formats)")
    print(f"  - {split_dir} (Train/val/test split)")
    print()
    print("Next steps:")
    print("  1. Implement ML model interfaces")
    print("  2. Create training corpus orchestrator")
    print("  3. Train models on generated datasets")
    print("  4. Evaluate on held-out test sets")


if __name__ == "__main__":
    main()
