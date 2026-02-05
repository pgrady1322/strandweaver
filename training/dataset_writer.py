"""
Dataset writer with sharding for ML training.

This module packages training features into production-ready datasets with:
- Multiple format support (JSONL, Parquet, NumPy/NPZ)
- Automatic sharding for large datasets
- Dataset versioning and metadata
- Efficient storage and loading
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
import numpy as np

from .feature_builder import (
    TrainingDataset,
    OverlapFeatures,
    GNNGraphTensors,
    DiploidNodeFeatures,
    ULPathFeatures,
    SVDetectionFeatures,
)

logger = logging.getLogger(__name__)

# Dataset format types
DatasetFormat = Literal["jsonl", "parquet", "npz", "all"]
DatasetVersion = Literal["v1_basic", "v2_repeat_enriched", "v3_sv_heavy", "v4_diploid_focus"]


@dataclass
class DatasetMetadata:
    """Metadata for a training dataset."""
    
    version: str
    num_examples: int
    num_shards: int
    shard_size: int
    format: str
    feature_dimensions: Dict[str, Any]
    created_at: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, output_path: Path):
        """Save metadata to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, metadata_path: Path) -> 'DatasetMetadata':
        """Load metadata from JSON file."""
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ShardInfo:
    """Information about a dataset shard."""
    
    shard_id: int
    num_examples: int
    file_path: str
    start_idx: int
    end_idx: int


def write_overlap_features_jsonl(
    features: List[OverlapFeatures],
    output_path: Path
):
    """
    Write overlap classifier features to JSONL format.
    
    Args:
        features: List of overlap features
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(json.dumps(feature.to_dict()) + '\n')


def write_diploid_features_jsonl(
    features: List[DiploidNodeFeatures],
    output_path: Path
):
    """
    Write diploid disentanglement features to JSONL format.
    
    Args:
        features: List of diploid node features
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(json.dumps(feature.to_dict()) + '\n')


def write_ul_routing_features_jsonl(
    features: List[ULPathFeatures],
    output_path: Path
):
    """
    Write UL routing features to JSONL format.
    
    Args:
        features: List of UL path features
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(json.dumps(feature.to_dict()) + '\n')


def write_sv_features_jsonl(
    features: List[SVDetectionFeatures],
    output_path: Path
):
    """
    Write SV detection features to JSONL format.
    
    Args:
        features: List of SV features
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(json.dumps(feature.to_dict()) + '\n')


def write_gnn_tensors_npz(
    tensors: GNNGraphTensors,
    output_path: Path
):
    """
    Write GNN graph tensors to NumPy NPZ format.
    
    Args:
        tensors: GNN graph tensors
        output_path: Output file path
    """
    # Convert to NumPy arrays
    node_features_np = tensors.node_features
    edge_index_np = tensors.edge_index
    edge_features_np = tensors.edge_features
    node_labels_np = np.array([label.value if hasattr(label, 'value') else str(label) for label in tensors.node_labels])
    
    # Save as compressed NPZ
    np.savez_compressed(
        output_path,
        node_features=node_features_np,
        edge_index=edge_index_np,
        edge_features=edge_features_np,
        node_labels=node_labels_np,
        num_nodes=len(tensors.node_features),
        num_edges=len(tensors.edge_index[0]),
        num_paths=len(tensors.path_sequences)
    )
    
    # Save path sequences separately (as text)
    paths_file = output_path.parent / f"{output_path.stem}_paths.txt"
    with open(paths_file, 'w') as f:
        for path in tensors.path_sequences:
            f.write(','.join(str(node) for node in path) + '\n')


def write_overlap_features_npz(
    features: List[OverlapFeatures],
    output_path: Path
):
    """
    Write overlap features to NumPy NPZ format.
    
    Args:
        features: List of overlap features
        output_path: Output file path
    """
    # Convert to arrays
    feature_vectors = np.array([f.to_feature_vector() for f in features])
    labels = np.array([f.label.value if hasattr(f.label, 'value') else f.label for f in features])
    
    # Save
    np.savez_compressed(
        output_path,
        features=feature_vectors,
        labels=labels,
        num_examples=len(features)
    )


def write_diploid_features_npz(
    features: List[DiploidNodeFeatures],
    output_path: Path
):
    """
    Write diploid features to NumPy NPZ format.
    
    Args:
        features: List of diploid node features
        output_path: Output file path
    """
    # Convert to arrays
    feature_vectors = np.array([f.to_feature_vector() for f in features])
    labels = np.array([f.label.value if hasattr(f.label, 'value') else f.true_haplotype for f in features])
    node_ids = [f.node_id for f in features]
    
    # Save
    np.savez_compressed(
        output_path,
        features=feature_vectors,
        labels=labels,
        num_examples=len(features)
    )
    
    # Save node IDs separately
    ids_file = output_path.parent / f"{output_path.stem}_ids.txt"
    with open(ids_file, 'w') as f:
        for node_id in node_ids:
            f.write(node_id + '\n')


def write_ul_routing_features_npz(
    features: List[ULPathFeatures],
    output_path: Path
):
    """
    Write UL routing features to NumPy NPZ format.
    
    Args:
        features: List of UL path features
        output_path: Output file path
    """
    if not features:
        logger.warning("No UL routing features to write")
        return
    
    # Convert to arrays
    feature_vectors = np.array([f.to_feature_vector() for f in features])
    scores = np.array([f.score for f in features])
    path_ids = [f.path_id for f in features]
    
    # Save
    np.savez_compressed(
        output_path,
        features=feature_vectors,
        scores=scores,
        num_examples=len(features)
    )
    
    # Save path IDs separately
    ids_file = output_path.parent / f"{output_path.stem}_ids.txt"
    with open(ids_file, 'w') as f:
        for path_id in path_ids:
            f.write(path_id + '\n')


def write_sv_features_npz(
    features: List[SVDetectionFeatures],
    output_path: Path
):
    """
    Write SV features to NumPy NPZ format.
    
    Args:
        features: List of SV features
        output_path: Output file path
    """
    # Convert to arrays
    feature_vectors = np.array([f.to_feature_vector() for f in features])
    sv_types = np.array([f.sv_type for f in features])
    sv_sizes = np.array([f.sv_size for f in features])
    region_ids = [f.region_id for f in features]
    positions = [f.position for f in features]
    
    # Save
    np.savez_compressed(
        output_path,
        features=feature_vectors,
        sv_types=sv_types,
        sv_sizes=sv_sizes,
        num_examples=len(features)
    )
    
    # Save metadata separately
    meta_file = output_path.parent / f"{output_path.stem}_meta.txt"
    with open(meta_file, 'w') as f:
        f.write("region_id\tposition\n")
        for region_id, position in zip(region_ids, positions):
            f.write(f"{region_id}\t{position}\n")


def write_sharded_dataset(
    dataset: TrainingDataset,
    output_dir: Path,
    format: DatasetFormat = "jsonl",
    shard_size: int = 5000,
    version: str = "v1_basic",
    description: str = ""
) -> DatasetMetadata:
    """
    Write training dataset with automatic sharding.
    
    This function splits large datasets into manageable shards for efficient
    loading and processing during training.
    
    Args:
        dataset: Training dataset to write
        output_dir: Output directory for dataset
        format: Output format (jsonl, parquet, npz, or all)
        shard_size: Number of examples per shard
        version: Dataset version identifier
        description: Human-readable description
    
    Returns:
        Dataset metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Writing sharded training dataset")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Format: {format}")
    logger.info(f"Shard size: {shard_size}")
    logger.info(f"Version: {version}")
    
    # Create subdirectories for each model type
    overlap_dir = output_dir / "overlap_classifier"
    gnn_dir = output_dir / "gnn_path_predictor"
    haplotype_dir = output_dir / "haplotype_detangler"
    ul_dir = output_dir / "ul_routing"
    sv_dir = output_dir / "sv_detection"
    
    for subdir in [overlap_dir, gnn_dir, diploid_dir, ul_dir, sv_dir]:
        subdir.mkdir(parents=True, exist_ok=True)
    
    # Write overlap classifier features
    if dataset.overlap_features:
        logger.info(f"Writing {len(dataset.overlap_features)} overlap examples...")
        num_shards = (len(dataset.overlap_features) + shard_size - 1) // shard_size
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, len(dataset.overlap_features))
            shard_features = dataset.overlap_features[start_idx:end_idx]
            
            if format in ["jsonl", "all"]:
                shard_file = overlap_dir / f"shard_{shard_id:04d}.jsonl"
                write_overlap_features_jsonl(shard_features, shard_file)
            
            if format in ["npz", "all"]:
                shard_file = overlap_dir / f"shard_{shard_id:04d}.npz"
                write_overlap_features_npz(shard_features, shard_file)
        
        logger.info(f"  Wrote {num_shards} shard(s)")
    
    # Write GNN graph tensors
    if dataset.gnn_tensors:
        logger.info("Writing GNN graph tensors...")
        
        if format in ["npz", "all"]:
            tensor_file = gnn_dir / "graph_tensors.npz"
            write_gnn_tensors_npz(dataset.gnn_tensors, tensor_file)
            logger.info("  Wrote graph tensors")
    
    # Write diploid features
    if dataset.diploid_features:
        logger.info(f"Writing {len(dataset.diploid_features)} diploid examples...")
        num_shards = (len(dataset.diploid_features) + shard_size - 1) // shard_size
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, len(dataset.diploid_features))
            shard_features = dataset.diploid_features[start_idx:end_idx]
            
            if format in ["jsonl", "all"]:
                shard_file = diploid_dir / f"shard_{shard_id:04d}.jsonl"
                write_diploid_features_jsonl(shard_features, shard_file)
            
            if format in ["npz", "all"]:
                shard_file = diploid_dir / f"shard_{shard_id:04d}.npz"
                write_diploid_features_npz(shard_features, shard_file)
        
        logger.info(f"  Wrote {num_shards} shard(s)")
    
    # Write UL routing features
    if dataset.ul_routing_features:
        logger.info(f"Writing {len(dataset.ul_routing_features)} UL routing examples...")
        num_shards = (len(dataset.ul_routing_features) + shard_size - 1) // shard_size
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, len(dataset.ul_routing_features))
            shard_features = dataset.ul_routing_features[start_idx:end_idx]
            
            if format in ["jsonl", "all"]:
                shard_file = ul_dir / f"shard_{shard_id:04d}.jsonl"
                write_ul_routing_features_jsonl(shard_features, shard_file)
            
            if format in ["npz", "all"]:
                shard_file = ul_dir / f"shard_{shard_id:04d}.npz"
                write_ul_routing_features_npz(shard_features, shard_file)
        
        logger.info(f"  Wrote {num_shards} shard(s)")
    else:
        logger.info("No UL routing features to write")
    
    # Write SV detection features
    if dataset.sv_features:
        logger.info(f"Writing {len(dataset.sv_features)} SV detection examples...")
        num_shards = (len(dataset.sv_features) + shard_size - 1) // shard_size
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, len(dataset.sv_features))
            shard_features = dataset.sv_features[start_idx:end_idx]
            
            if format in ["jsonl", "all"]:
                shard_file = sv_dir / f"shard_{shard_id:04d}.jsonl"
                write_sv_features_jsonl(shard_features, shard_file)
            
            if format in ["npz", "all"]:
                shard_file = sv_dir / f"shard_{shard_id:04d}.npz"
                write_sv_features_npz(shard_features, shard_file)
        
        logger.info(f"  Wrote {num_shards} shard(s)")
    
    # Create metadata
    from datetime import datetime
    
    feature_dims = {
        "overlap_classifier": 17,
        "gnn_node_features": 32,
        "gnn_edge_features": 16,
        "diploid_features": 42,
        "ul_routing": 12,
        "sv_detection": 14,
    }
    
    metadata = DatasetMetadata(
        version=version,
        num_examples=len(dataset.overlap_features) if dataset.overlap_features else 0,
        num_shards=(len(dataset.overlap_features) + shard_size - 1) // shard_size if dataset.overlap_features else 0,
        shard_size=shard_size,
        format=format,
        feature_dimensions=feature_dims,
        created_at=datetime.now().isoformat(),
        description=description or f"StrandWeaver training dataset {version}"
    )
    
    # Save metadata
    metadata.save(output_dir / "metadata.json")
    logger.info(f"Saved dataset metadata")
    
    # Create README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# StrandWeaver Training Dataset - {version}\n\n")
        f.write(f"{description}\n\n")
        f.write(f"## Dataset Statistics\n\n")
        f.write(f"- **Total examples**: {metadata.num_examples}\n")
        f.write(f"- **Number of shards**: {metadata.num_shards}\n")
        f.write(f"- **Examples per shard**: {metadata.shard_size}\n")
        f.write(f"- **Format**: {metadata.format}\n")
        f.write(f"- **Created**: {metadata.created_at}\n\n")
        f.write(f"## Directory Structure\n\n")
        f.write(f"```\n")
        f.write(f"{output_dir.name}/\n")
        f.write(f"├── metadata.json\n")
        f.write(f"├── README.md\n")
        f.write(f"├── overlap_classifier/\n")
        f.write(f"│   └── shard_*.{format}\n")
        f.write(f"├── gnn_path_predictor/\n")
        f.write(f"│   └── graph_tensors.npz\n")
        f.write(f"├── haplotype_detangler/\n")
        f.write(f"│   └── shard_*.{format}\n")
        f.write(f"├── ul_routing/\n")
        f.write(f"│   └── shard_*.{format}\n")
        f.write(f"└── sv_detection/\n")
        f.write(f"    └── shard_*.{format}\n")
        f.write(f"```\n\n")
        f.write(f"## Feature Dimensions\n\n")
        for model, dims in feature_dims.items():
            f.write(f"- **{model}**: {dims}D\n")
    
    logger.info("=" * 80)
    logger.info("Dataset writing complete")
    logger.info("=" * 80)
    
    return metadata


def load_jsonl_shard(shard_path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL shard file.
    
    Args:
        shard_path: Path to shard file
    
    Returns:
        List of feature dictionaries
    """
    examples = []
    with open(shard_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_npz_shard(shard_path: Path) -> Dict[str, np.ndarray]:
    """
    Load a NumPy NPZ shard file.
    
    Args:
        shard_path: Path to shard file
    
    Returns:
        Dictionary of arrays
    """
    return dict(np.load(shard_path))


def merge_dataset_shards(
    input_dir: Path,
    output_file: Path,
    format: Literal["jsonl", "npz"] = "jsonl"
):
    """
    Merge multiple dataset shards into a single file.
    
    This is useful for creating validation/test sets or when you want
    a single file for easier distribution.
    
    Args:
        input_dir: Directory containing shard files
        output_file: Output file path
        format: Output format (jsonl or npz)
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    logger.info(f"Merging shards from {input_dir}...")
    
    # Get all shard files
    shard_files = sorted(input_dir.glob(f"shard_*.{format}"))
    
    if not shard_files:
        logger.warning(f"No shard files found in {input_dir}")
        return
    
    logger.info(f"Found {len(shard_files)} shard(s)")
    
    if format == "jsonl":
        # Merge JSONL files
        with open(output_file, 'w') as out_f:
            for shard_file in shard_files:
                with open(shard_file, 'r') as in_f:
                    for line in in_f:
                        out_f.write(line)
        
        logger.info(f"Merged to {output_file}")
    
    elif format == "npz":
        # Merge NPZ files
        all_features = []
        all_labels = []
        
        for shard_file in shard_files:
            data = np.load(shard_file)
            all_features.append(data['features'])
            all_labels.append(data['labels'])
        
        # Concatenate
        merged_features = np.vstack(all_features)
        merged_labels = np.concatenate(all_labels)
        
        # Save
        np.savez_compressed(
            output_file,
            features=merged_features,
            labels=merged_labels,
            num_examples=len(merged_features)
        )
        
        logger.info(f"Merged {len(merged_features)} examples to {output_file}")


def create_train_val_test_split(
    dataset: TrainingDataset,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    format: DatasetFormat = "jsonl",
    version: str = "v1_basic"
):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Training dataset
        output_dir: Output directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        format: Output format
        version: Dataset version
    """
    output_dir = Path(output_dir)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    logger.info("=" * 80)
    logger.info("Creating train/val/test split")
    logger.info("=" * 80)
    logger.info(f"Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    # Shuffle indices - use the largest available feature set
    num_examples = max(
        len(dataset.overlap_features) if dataset.overlap_features else 0,
        len(dataset.diploid_features) if dataset.diploid_features else 0,
        len(dataset.ul_routing_features) if dataset.ul_routing_features else 0,
        len(dataset.sv_features) if dataset.sv_features else 0
    )
    
    if num_examples == 0:
        logger.warning("No examples to split - all feature sets are empty")
        return
    
    indices = np.random.permutation(num_examples)
    
    # Calculate split points
    train_end = int(num_examples * train_ratio)
    val_end = train_end + int(num_examples * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"Train: {len(train_indices)} examples")
    logger.info(f"Val: {len(val_indices)} examples")
    logger.info(f"Test: {len(test_indices)} examples")
    
    # Create split datasets
    def extract_subset(features, indices):
        if not features:
            return []
        # Only use indices that are valid for this feature list
        valid_indices = [i for i in indices if i < len(features)]
        return [features[i] for i in valid_indices]
    
    train_dataset = TrainingDataset(
        overlap_features=extract_subset(dataset.overlap_features, train_indices),
        gnn_tensors=dataset.gnn_tensors,  # Keep full graph for now
        diploid_features=extract_subset(dataset.diploid_features, train_indices),
        ul_routing_features=extract_subset(dataset.ul_routing_features, train_indices),
        sv_features=extract_subset(dataset.sv_features, train_indices)
    )
    
    val_dataset = TrainingDataset(
        overlap_features=extract_subset(dataset.overlap_features, val_indices),
        gnn_tensors=dataset.gnn_tensors,
        diploid_features=extract_subset(dataset.diploid_features, val_indices),
        ul_routing_features=extract_subset(dataset.ul_routing_features, val_indices),
        sv_features=extract_subset(dataset.sv_features, val_indices)
    )
    
    test_dataset = TrainingDataset(
        overlap_features=extract_subset(dataset.overlap_features, test_indices),
        gnn_tensors=dataset.gnn_tensors,
        diploid_features=extract_subset(dataset.diploid_features, test_indices),
        ul_routing_features=extract_subset(dataset.ul_routing_features, test_indices),
        sv_features=extract_subset(dataset.sv_features, test_indices)
    )
    
    # Write splits
    write_sharded_dataset(
        train_dataset,
        output_dir / "train",
        format=format,
        version=version,
        description=f"Training set ({train_ratio:.1%} of data)"
    )
    
    write_sharded_dataset(
        val_dataset,
        output_dir / "val",
        format=format,
        version=version,
        description=f"Validation set ({val_ratio:.1%} of data)"
    )
    
    write_sharded_dataset(
        test_dataset,
        output_dir / "test",
        format=format,
        version=version,
        description=f"Test set ({test_ratio:.1%} of data)"
    )
    
    logger.info("=" * 80)
    logger.info("Train/val/test split complete")
    logger.info("=" * 80)
