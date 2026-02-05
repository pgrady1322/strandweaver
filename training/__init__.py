"""
Training Data Generation Infrastructure

This module provides tools for generating synthetic training data for ML models:
- Synthetic data generation (genomes + reads + ground truth) - CONSOLIDATED
- Feature extraction
- Dataset writing

Author: StrandWeaver Training Infrastructure
Version: 0.3.0
Date: December 2025
Phase: 5.3 - ML Training Data Generation
"""

from __future__ import annotations

# PHASE 1 CONSOLIDATION: Synthetic Data Generation (genome + reads + ground truth)
from .synthetic_data_generator import (
    # Genome simulation
    GenomeConfig,
    SVType,
    StructuralVariant,
    DiploidGenome,
    generate_random_sequence,
    generate_haploid_genome,
    generate_diploid_genome,
    # Read simulation
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
    # Ground-truth labeling
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
from .main_training_workflow import (
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

# PHASE 3 CONSOLIDATION: ML Training System (interfaces + models + training)
from .ml_training_system import (
    # Model Interfaces
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
    # Training Infrastructure
    TrainingConfig,
    TrainingMetrics,
    ModelTrainer,
    ModelEvaluator,
)

# Training corpus orchestration - Complete pipeline with graph assembly
from .main_training_workflow import (
    generate_training_corpus,
    ScenarioConfig,
    SCENARIOS,
    list_scenarios,
    get_scenario_info,
    estimate_generation_time,
)

__version__ = "0.3.0"

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
