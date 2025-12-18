"""
Training Data Generation Infrastructure

This module provides tools for generating synthetic training data for ML models:
- Synthetic genome generation (diploid with SVs)
- Read simulation (Illumina, HiFi, ONT, UL, Hi-C)
- Ground-truth labeling
- Feature extraction
- Dataset writing

Author: StrandWeaver Training Infrastructure
Version: 0.1.0
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations

# Genome simulation
from .genome_simulator import (
    GenomeConfig,
    SVType,
    StructuralVariant,
    DiploidGenome,
    generate_random_sequence,
    generate_haploid_genome,
    generate_diploid_genome,
)

# Read simulation
from .read_simulator import (
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    AncientDNAConfig,
    SimulatedRead,
    SimulatedReadPair,
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_fastq,
    write_paired_fastq,
)

# Ground-truth labeling
from .truth_labeler import (
    EdgeLabel,
    NodeHaplotype,
    ReadAlignment,
    EdgeGroundTruth,
    NodeGroundTruth,
    PathGroundTruth,
    ULRouteGroundTruth,
    SVGroundTruth,
    GroundTruthLabels,
    generate_ground_truth_labels,
    export_labels_to_tsv,
)

# Training feature extraction
from .feature_builder import (
    OverlapFeatures,
    GNNGraphTensors,
    DiploidNodeFeatures,
    ULPathFeatures,
    SVDetectionFeatures,
    TrainingDataset,
    build_training_dataset,
    extract_overlap_features,
    build_gnn_graph_tensors,
    extract_diploid_features,
    extract_ul_routing_features,
    extract_sv_features,
)

# Dataset writing and packaging
from .dataset_writer import (
    DatasetMetadata,
    ShardInfo,
    DatasetFormat,
    DatasetVersion,
    write_sharded_dataset,
    create_train_val_test_split,
    merge_dataset_shards,
    load_jsonl_shard,
    load_npz_shard,
)

# ML model interfaces
from .ml_interfaces import (
    BaseMLModel,
    EdgeAIModel,
    PathGNNModel,
    DiploidAIModel,
    ULRoutingAIModel,
    SVAIModel,
    EdgePrediction,
    PathPrediction,
    HaplotypePrediction,
    RoutePrediction,
    SVPrediction,
    GraphTensors,
    ModelRegistry,
    create_model,
    TrainingConfig,
    TrainingMetrics,
    ModelTrainer,
    ModelEvaluator,
)

# Training corpus orchestration - Complete pipeline with graph assembly
from .full_corpus_generator import (
    generate_training_corpus,
    ScenarioConfig,
    SCENARIOS,
    list_scenarios,
    get_scenario_info,
    estimate_generation_time,
)

__version__ = "0.1.0"

__all__ = [
    # Genome simulation
    "GenomeConfig",
    "SVType",
    "StructuralVariant",
    "DiploidGenome",
    "generate_random_sequence",
    "generate_haploid_genome",
    "generate_diploid_genome",
    # Read simulation
    "IlluminaConfig",
    "HiFiConfig",
    "ONTConfig",
    "ULConfig",
    "HiCConfig",
    "AncientDNAConfig",
    "SimulatedRead",
    "SimulatedReadPair",
    "simulate_illumina_reads",
    "simulate_long_reads",
    "simulate_hic_reads",
    "write_fastq",
    "write_paired_fastq",
    # Ground-truth labeling
    "EdgeLabel",
    "NodeHaplotype",
    "ReadAlignment",
    "EdgeGroundTruth",
    "NodeGroundTruth",
    "PathGroundTruth",
    "ULRouteGroundTruth",
    "SVGroundTruth",
    "GroundTruthLabels",
    "generate_ground_truth_labels",
    "export_labels_to_tsv",
    # ML interfaces
    "BaseMLModel",
    "EdgeAIModel",
    "PathGNNModel",
    "DiploidAIModel",
    "ULRoutingAIModel",
    "SVAIModel",
    "EdgePrediction",
    "PathPrediction",
    "HaplotypePrediction",
    "RoutePrediction",
    "SVPrediction",
    "GraphTensors",
    "ModelRegistry",
    "create_model",
    "TrainingConfig",
    "TrainingMetrics",
    "ModelTrainer",
    "ModelEvaluator",
    # Training corpus orchestration
    "generate_training_corpus",
    "ScenarioConfig",
    "SCENARIOS",
    "list_scenarios",
    "get_scenario_info",
    "estimate_generation_time",
]
