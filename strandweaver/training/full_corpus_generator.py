"""
Complete Training Corpus Generator with Full ML Pipeline

This module implements the complete end-to-end training data generation pipeline,
integrating all Phase 5.3 components:

1. Genome Simulation → diploid genomes with SVs and repeats
2. Read Simulation → Illumina, HiFi, ONT, UL-ONT reads with realistic errors
3. Graph Assembly → de Bruijn graph construction from simulated reads  
4. Ground-Truth Labeling → edge, node, path, UL route, SV labels
5. Feature Extraction → ML features for all 5 models
6. Dataset Writing → sharded datasets in JSONL/NPZ/Parquet formats

Scenarios:
- simple: Quick test (10 genomes × 100kb, ~2 min)
- balanced: Production training (100 genomes × 1Mb, ~20 min)
- repeat_heavy: Repeat-rich (50 genomes × 2Mb, 60% repeats, ~30 min)
- sv_dense: SV-focused (50 genomes × 1Mb, 10× SV density, ~15 min)
- diploid_focus: High heterozygosity (100 genomes × 1Mb, 2% het, ~25 min)
- ultra_long_focus: UL-optimized (30 genomes × 5Mb, 50× UL, ~40 min)

Author: StrandWeaver Development Team  
Date: December 2025
Phase: 5.3 Complete ML Training Infrastructure
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import time
from datetime import datetime
from collections import defaultdict

# Phase 5.3 Module Imports - Genome & Read Simulation
from strandweaver.training.genome_simulator import (
    generate_diploid_genome,
    GenomeConfig,
    StructuralVariant,
    DiploidGenome
)

from strandweaver.training.read_simulator import (
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_paired_fastq,
    write_fastq,
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    SimulatedRead,
    SimulatedReadPair
)

# Phase 5.3 Module Imports - Graph Assembly
from strandweaver.assembly_core.data_structures import (
    DeBruijnGraphBuilder,
    KmerGraphConfig,
    KmerGraph,
    KmerNode,
    KmerEdge
)

# Phase 5.3 Module Imports - Labeling & Features
from strandweaver.training.truth_labeler import (
    generate_ground_truth_labels,
    GroundTruthLabels,
    export_labels_to_tsv,
    ReadAlignment
)

from strandweaver.training.feature_builder import (
    build_training_dataset,
    TrainingDataset,
    extract_overlap_features,
    build_gnn_graph_tensors
)

from strandweaver.training.dataset_writer import (
    write_sharded_dataset,
    DatasetMetadata,
    create_train_val_test_split
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a training scenario."""
    name: str
    genome_size: int
    num_genomes: int
    
    # Coverage parameters
    illumina_coverage: float = 30.0
    hifi_coverage: float = 30.0
    ont_coverage: float = 30.0
    ul_coverage: float = 10.0
    hic_coverage: float = 20.0
    
    # Genome complexity
    gc_content: float = 0.42
    repeat_density: float = 0.45
    snp_rate: float = 0.001
    sv_density: float = 0.00005
    sv_max_size: int = 100_000
    
    # Assembly parameters
    kmer_size: int = 31
    min_kmer_count: int = 2
    
    # Dataset parameters
    dataset_format: str = "jsonl"  # jsonl, npz, parquet, or all
    shard_size: int = 5000
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


# ============================================================================
#                    PREDEFINED TRAINING SCENARIOS
# ============================================================================

SCENARIOS = {
    'simple': ScenarioConfig(
        name='simple',
        genome_size=100_000,
        num_genomes=10,
        illumina_coverage=30.0,
        hifi_coverage=30.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=20.0,
        sv_max_size=10_000,
        kmer_size=21,  # Smaller k for small genomes
        dataset_format="jsonl"
    ),
    
    'fast_balanced': ScenarioConfig(
        name='fast_balanced',
        genome_size=500_000,  # Smaller for speed
        num_genomes=20,  # Full dataset
        illumina_coverage=20.0,  # Lower coverage
        hifi_coverage=20.0,
        ont_coverage=20.0,
        ul_coverage=5.0,
        hic_coverage=10.0,
        sv_max_size=50_000,
        kmer_size=27,  # Optimized k
        dataset_format="jsonl"
    ),
    
    'balanced': ScenarioConfig(
        name='balanced',
        genome_size=1_000_000,
        num_genomes=30,  # Reduced from 100 for faster generation
        illumina_coverage=30.0,
        hifi_coverage=30.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=20.0,
        dataset_format="all"  # All formats for production
    ),
    
    'repeat_heavy': ScenarioConfig(
        name='repeat_heavy',
        genome_size=2_000_000,
        num_genomes=50,
        repeat_density=0.60,  # 60% repeats (high)
        illumina_coverage=40.0,
        hifi_coverage=40.0,
        ont_coverage=30.0,
        ul_coverage=15.0,
        hic_coverage=30.0,
        kmer_size=51,  # Larger k for repeats
        dataset_format="all"
    ),
    
    'sv_dense': ScenarioConfig(
        name='sv_dense',
        genome_size=1_000_000,
        num_genomes=50,
        sv_density=0.0005,  # 10× normal SV density
        sv_max_size=500_000,  # Larger SVs
        illumina_coverage=40.0,
        hifi_coverage=40.0,
        ont_coverage=40.0,
        ul_coverage=20.0,  # Higher UL for SV spanning
        hic_coverage=30.0,
        dataset_format="all"
    ),
    
    'diploid_focus': ScenarioConfig(
        name='diploid_focus',
        genome_size=1_000_000,
        num_genomes=100,
        snp_rate=0.02,  # 2% heterozygosity (high)
        illumina_coverage=30.0,
        hifi_coverage=40.0,
        ont_coverage=30.0,
        ul_coverage=10.0,
        hic_coverage=40.0,  # Higher Hi-C for phasing
        dataset_format="all"
    ),
    
    'ultra_long_focus': ScenarioConfig(
        name='ultra_long_focus',
        genome_size=5_000_000,
        num_genomes=30,
        illumina_coverage=20.0,
        hifi_coverage=30.0,
        ont_coverage=20.0,
        ul_coverage=50.0,  # Very high UL coverage
        hic_coverage=20.0,
        kmer_size=31,
        dataset_format="all"
    ),
}


# ============================================================================
#                    GRAPH ASSEMBLY FROM SIMULATED READS
# ============================================================================

def build_assembly_graph_from_reads(
    illumina_reads: List[SimulatedReadPair],
    hifi_reads: List[SimulatedRead],
    kmer_config: KmerGraphConfig,
    genome_id: str
) -> Tuple[KmerGraph, List[Tuple[str, str]]]:
    """
    Build de Bruijn graph from simulated reads.
    
    Uses the StrandWeaver DeBruijnGraphBuilder to construct a graph
    from accurate reads (Illumina and HiFi).
    
    Args:
        illumina_reads: Simulated Illumina paired-end reads
        hifi_reads: Simulated PacBio HiFi reads
        kmer_config: K-mer graph configuration
        genome_id: Genome identifier for logging
    
    Returns:
        Tuple of (assembled graph, list of (read_id, sequence) tuples)
    """
    logger.info(f"[Genome {genome_id}] Building assembly graph (k={kmer_config.k})...")
    
    # Convert simulated reads to (read_id, sequence) format
    accurate_reads = []
    
    # Add Illumina reads
    for pair in illumina_reads:
        accurate_reads.append((pair.read1.read_id, pair.read1.sequence))
        accurate_reads.append((pair.read2.read_id, pair.read2.sequence))
    
    # Add HiFi reads
    for read in hifi_reads:
        accurate_reads.append((read.read_id, read.sequence))
    
    logger.info(f"[Genome {genome_id}]   Total accurate reads: {len(accurate_reads)}")
    
    # Build de Bruijn graph with GPU acceleration enabled
    builder = DeBruijnGraphBuilder(kmer_config, use_gpu=True)
    
    # Step 1: Build raw k-mer graph with read provenance
    raw_graph, node_to_reads = builder.build_raw_kmer_graph(accurate_reads, kmer_config)
    logger.info(f"[Genome {genome_id}]   Raw graph: {len(raw_graph.nodes)} nodes, {len(raw_graph.edges)} edges")
    
    # Step 2: Compact graph into unitigs, propagating read provenance
    compacted_graph, compacted_node_to_reads = builder.compact_graph(raw_graph, node_to_reads)
    logger.info(f"[Genome {genome_id}]   Compacted: {len(compacted_graph.nodes)} nodes, {len(compacted_graph.edges)} edges")
    
    return compacted_graph, accurate_reads, compacted_node_to_reads


def extract_graph_structure(graph: KmerGraph) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Extract node IDs and edge list from KmerGraph.
    
    Args:
        graph: Assembled k-mer graph
    
    Returns:
        Tuple of (node_ids, edge_list)
    """
    node_ids = [str(nid) for nid in graph.nodes.keys()]
    edge_list = [(str(e.from_id), str(e.to_id)) for e in graph.edges.values()]
    
    return node_ids, edge_list


# ============================================================================
#                    MAIN CORPUS GENERATION FUNCTION
# ============================================================================

def generate_training_corpus(
    scenario: str = 'balanced',
    output_dir: str = 'training_data',
    num_processes: int = 4,
    **override_params
) -> Dict[str, Any]:
    """
    Generate complete training corpus with full ML pipeline.
    
    This is the main entry point that orchestrates:
    1. Genome simulation (diploid with SVs)
    2. Read simulation (all technologies)
    3. Graph assembly (de Bruijn graph)
    4. Ground-truth labeling (all label types)
    5. Feature extraction (all ML models)
    6. Dataset writing (sharded output)
    
    Args:
        scenario: Scenario name ('simple', 'balanced', etc.)
        output_dir: Output directory for training data
        num_processes: Number of parallel processes
        **override_params: Override any ScenarioConfig parameters
    
    Returns:
        Metadata dictionary with generation statistics
    """
    start_time = time.time()
    output_path = Path(output_dir) / scenario
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get scenario configuration
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(SCENARIOS.keys())}")
    
    config = SCENARIOS[scenario]
    
    # Apply parameter overrides
    for key, value in override_params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info("=" * 80)
    logger.info(f"Generating Training Corpus: {scenario}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Genomes: {config.num_genomes} × {config.genome_size:,} bp")
    logger.info(f"  Coverage: Illumina={config.illumina_coverage}×, HiFi={config.hifi_coverage}×, "
                f"ONT={config.ont_coverage}×, UL={config.ul_coverage}×, Hi-C={config.hic_coverage}×")
    logger.info(f"  Complexity: GC={config.gc_content:.1%}, Repeats={config.repeat_density:.1%}, "
                f"SNPs={config.snp_rate:.3%}, SVs={config.sv_density:.5f}")
    logger.info(f"  K-mer: k={config.kmer_size}, min_count={config.min_kmer_count}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 80)
    
    # Initialize aggregate statistics
    total_stats = {
        'num_genomes': config.num_genomes,
        'scenario': scenario,
        'start_time': datetime.now().isoformat(),
        'genomes': [],
        'total_reads': {},
        'total_graph_nodes': 0,
        'total_graph_edges': 0,
        'total_labels': 0,
        'total_features': 0,
    }
    
    # Create K-mer graph configuration
    kmer_config = KmerGraphConfig(
        k=config.kmer_size,
        min_kmer_count=config.min_kmer_count,
        canonical=True
    )
    
    # ========================================================================
    # PHASE 1: Generate genomes, reads, and graphs
    # ========================================================================
    
    genome_data = []  # List of (genome, reads, graph, labels) tuples
    
    for i in range(config.num_genomes):
        genome_id = f"genome_{i:04d}"
        genome_output_dir = output_path / 'per_genome' / genome_id
        
        # Check if this genome has already been processed
        features_file = genome_output_dir / 'features.pkl'
        if features_file.exists():
            logger.info(f"\n{'='*80}")
            logger.info(f"SKIPPING {genome_id} ({i+1}/{config.num_genomes}) - Already processed")
            logger.info(f"{'='*80}")
            logger.info(f"Found existing features at: {features_file}")
            
            # Add to genome_data so it gets included in aggregation
            genome_data.append({
                'genome_id': genome_id,
                'features_file': features_file,
                'skipped': True
            })
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {genome_id} ({i+1}/{config.num_genomes})")
        logger.info(f"{'='*80}")
        
        # Step 1: Generate diploid genome
        logger.info(f"[{genome_id}] Step 1/6: Generating diploid genome...")
        genome_config = GenomeConfig(
            length=config.genome_size,
            gc_content=config.gc_content,
            repeat_density=config.repeat_density,
            snp_rate=config.snp_rate,
            sv_density=config.sv_density,
            sv_max_size=config.sv_max_size,
            random_seed=42 + i
        )
        
        diploid = generate_diploid_genome(genome_config)
        logger.info(f"[{genome_id}]   Generated: {len(diploid.hapA)} bp (A), {len(diploid.hapB)} bp (B)")
        logger.info(f"[{genome_id}]   Variants: {len(diploid.sv_truth_table)} SVs, "
                    f"{len([v for v in diploid.sv_truth_table if v.sv_type == 'deletion'])} deletions, "
                    f"{len([v for v in diploid.sv_truth_table if v.sv_type == 'insertion'])} insertions")
        
        # Step 2: Simulate reads
        logger.info(f"[{genome_id}] Step 2/6: Simulating sequencing reads...")
        
        # Illumina reads
        ill_config = IlluminaConfig(
            coverage=config.illumina_coverage,
            read_length=150,
            insert_size_mean=350,
            insert_size_std=50,
            error_rate=0.001,
            random_seed=42 + i
        )
        ill_reads = simulate_illumina_reads(diploid.hapA, ill_config)
        ill_reads_b = simulate_illumina_reads(diploid.hapB, ill_config)
        all_ill_reads = ill_reads + ill_reads_b
        
        # HiFi reads
        hifi_config = HiFiConfig(
            coverage=config.hifi_coverage,
            read_length_mean=15000,
            read_length_std=5000,
            error_rate=0.001,
            random_seed=42 + i
        )
        hifi_reads = simulate_long_reads(diploid.hapA, hifi_config, 'hifi')
        hifi_reads_b = simulate_long_reads(diploid.hapB, hifi_config, 'hifi')
        all_hifi_reads = hifi_reads + hifi_reads_b
        
        # ONT reads
        ont_config = ONTConfig(
            coverage=config.ont_coverage,
            read_length_mean=10000,
            read_length_std=5000,
            error_rate=0.05,
            indel_fraction=0.7,
            random_seed=42 + i
        )
        ont_reads = simulate_long_reads(diploid.hapA, ont_config, 'ont')
        ont_reads_b = simulate_long_reads(diploid.hapB, ont_config, 'ont')
        all_ont_reads = ont_reads + ont_reads_b
        
        # Ultra-long ONT reads
        ul_config = ULConfig(
            coverage=config.ul_coverage,
            read_length_mean=100000,
            read_length_std=25000,
            error_rate=0.08,
            indel_fraction=0.75,
            random_seed=42 + i
        )
        ul_reads = simulate_long_reads(diploid.hapA, ul_config, 'ultra_long')
        ul_reads_b = simulate_long_reads(diploid.hapB, ul_config, 'ultra_long')
        all_ul_reads = ul_reads + ul_reads_b
        
        # Hi-C reads
        hic_config = HiCConfig(
            num_pairs=int(config.genome_size * config.hic_coverage / 300),  # Estimate pairs from coverage
            read_length=150,
            cis_fraction=0.90,
            distance_decay_rate=1.0,
            random_seed=42 + i
        )
        hic_reads = simulate_hic_reads(diploid.hapA, diploid.hapB, hic_config)
        
        logger.info(f"[{genome_id}]   Illumina: {len(all_ill_reads)} pairs")
        logger.info(f"[{genome_id}]   HiFi: {len(all_hifi_reads)} reads")
        logger.info(f"[{genome_id}]   ONT: {len(all_ont_reads)} reads")
        logger.info(f"[{genome_id}]   UL-ONT: {len(all_ul_reads)} reads")
        logger.info(f"[{genome_id}]   Hi-C: {len(hic_reads)} pairs")
        
        # Step 3: Build assembly graph
        logger.info(f"[{genome_id}] Step 3/6: Building assembly graph...")
        graph, accurate_reads, node_to_reads = build_assembly_graph_from_reads(
            all_ill_reads,
            all_hifi_reads,
            kmer_config,
            genome_id
        )
        
        node_ids, edge_list = extract_graph_structure(graph)
        logger.info(f"[{genome_id}]   Graph: {len(node_ids)} nodes, {len(edge_list)} edges")
        
        # Step 4: Generate ground-truth labels
        logger.info(f"[{genome_id}] Step 4/6: Generating ground-truth labels...")
        
        # Convert SimulatedRead objects to list for labeler
        all_simulated_reads = (
            [pair.read1 for pair in all_ill_reads] + 
            [pair.read2 for pair in all_ill_reads] +
            all_hifi_reads + 
            all_ont_reads
        )
        
        # Convert node_to_reads keys to string for labeler
        node_to_reads_str = {str(k): v for k, v in node_to_reads.items()}
        
        labels = generate_ground_truth_labels(
            simulated_reads=all_simulated_reads,
            ul_reads=all_ul_reads,
            reference_hapA=diploid.hapA,
            reference_hapB=diploid.hapB,
            sv_truth_table=diploid.sv_truth_table,
            graph_edges=edge_list,
            graph_nodes=node_ids,
            node_to_read_ids=node_to_reads_str
        )
        
        logger.info(f"[{genome_id}]   Labels: {len(labels.edge_labels)} edges, "
                    f"{len(labels.node_labels)} nodes, {len(labels.path_labels)} paths, "
                    f"{len(labels.ul_route_labels)} UL routes, {len(labels.sv_labels)} SVs")
        
        # Step 5: Extract ML features
        logger.info(f"[{genome_id}] Step 5/6: Extracting ML features...")
        
        dataset = build_training_dataset(
            simulated_reads=all_simulated_reads,
            ul_reads=all_ul_reads,
            graph_nodes=node_ids,
            graph_edges=edge_list,
            ground_truth_labels=labels
        )
        
        logger.info(f"[{genome_id}]   Features: {len(dataset.overlap_features)} overlap, "
                    f"{len(dataset.diploid_features)} diploid, "
                    f"{len(dataset.ul_routing_features)} UL routing, "
                    f"{len(dataset.sv_features)} SV")
        
        # Update totals BEFORE freeing memory
        total_stats['total_graph_nodes'] += len(node_ids)
        total_stats['total_graph_edges'] += len(edge_list)
        total_stats['total_labels'] += (
            len(labels.edge_labels) + 
            len(labels.node_labels) + 
            len(labels.path_labels) +
            len(labels.ul_route_labels) +
            len(labels.sv_labels)
        )
        total_stats['total_features'] += (
            len(dataset.overlap_features) +
            len(dataset.diploid_features) +
            len(dataset.ul_routing_features) +
            len(dataset.sv_features)
        )
        
        # Save per-genome stats
        total_stats['genomes'].append({
            'id': genome_id,
            'size': config.genome_size,
            'svs': len(diploid.sv_truth_table),
            'reads': {
                'illumina': len(all_ill_reads),
                'hifi': len(all_hifi_reads),
                'ont': len(all_ont_reads),
                'ul': len(all_ul_reads),
                'hic': len(hic_reads)
            },
            'graph': {'nodes': len(node_ids), 'edges': len(edge_list)},
            'labels': {
                'edges': len(labels.edge_labels),
                'nodes': len(labels.node_labels),
                'paths': len(labels.path_labels),
                'ul_routes': len(labels.ul_route_labels),
                'svs': len(labels.sv_labels)
            },
            'features': {
                'overlap': len(dataset.overlap_features),
                'diploid': len(dataset.diploid_features),
                'ul_routing': len(dataset.ul_routing_features),
                'sv': len(dataset.sv_features)
            }
        })
        
        # Step 6 (per-genome): Write genome data immediately to disk
        logger.info(f"[{genome_id}] Step 6/6: Writing genome data to disk...")
        genome_output_dir = output_path / 'per_genome' / genome_id
        genome_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write features as pickle files (lightweight, temporary storage)
        import pickle
        features_file = genome_output_dir / 'features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                'overlap_features': dataset.overlap_features,
                'diploid_features': dataset.diploid_features,
                'ul_routing_features': dataset.ul_routing_features,
                'sv_features': dataset.sv_features,
                'gnn_tensors': dataset.gnn_tensors
            }, f)
        
        # Save minimal metadata (don't keep large objects in memory)
        genome_data.append({
            'genome_id': genome_id,
            'features_file': str(features_file)
        })
        
        logger.info(f"[{genome_id}]   Wrote features to {features_file.name}")
        
        # Free memory by deleting large objects
        del diploid, all_ill_reads, all_hifi_reads, all_ont_reads, all_ul_reads, hic_reads
        del graph, labels, dataset, node_ids, edge_list
        import gc
        gc.collect()
    
    # ========================================================================
    # PHASE 2: Stream features from disk and write sharded datasets incrementally
    # ========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Final aggregation: Streaming {len(genome_data)} genomes to sharded output...")
    logger.info(f"Memory-optimized: Processing one genome at a time")
    logger.info(f"{'='*80}")
    
    # Initialize output directories for each split
    for split_name in ['train', 'val', 'test']:
        for model_type in ['overlap_classifier', 'gnn_path_predictor', 'diploid_disentangler', 'ul_routing', 'sv_detection']:
            split_dir = output_path / split_name / model_type
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine train/val/test split for each genome
    import random
    random.seed(42)
    genome_indices = list(range(len(genome_data)))
    random.shuffle(genome_indices)
    
    num_train = int(len(genome_data) * config.train_split)
    num_val = int(len(genome_data) * config.val_split)
    
    train_indices = set(genome_indices[:num_train])
    val_indices = set(genome_indices[num_train:num_train + num_val])
    test_indices = set(genome_indices[num_train + num_val:])
    
    # Counters for tracking
    feature_counts = {
        'overlap': 0,
        'diploid': 0,
        'ul_routing': 0,
        'sv': 0
    }
    
    shard_counters = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    shard_buffers = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    def write_shard_buffer(split_name, model_type, buffer, shard_idx):
        """Write a buffer to a shard file."""
        if not buffer:
            return
        
        split_dir = output_path / split_name / model_type
        shard_file = split_dir / f'shard_{shard_idx:04d}.jsonl'
        
        with open(shard_file, 'w') as f:
            for example in buffer:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"  Wrote {len(buffer)} examples to {split_name}/{model_type}/shard_{shard_idx:04d}.jsonl")
    
    # Process each genome and write to shards incrementally
    import pickle
    
    for idx, gd in enumerate(genome_data):
        logger.info(f"Processing features from {gd['genome_id']} ({idx+1}/{len(genome_data)})...")
        
        # Determine split
        if idx in train_indices:
            split_name = 'train'
        elif idx in val_indices:
            split_name = 'val'
        else:
            split_name = 'test'
        
        # Load features from disk
        with open(gd['features_file'], 'rb') as f:
            features = pickle.load(f)
        
        # Process overlap features
        for feat in features['overlap_features']:
            shard_buffers[split_name]['overlap_classifier'].append(feat.to_dict())
            feature_counts['overlap'] += 1
            
            if len(shard_buffers[split_name]['overlap_classifier']) >= config.shard_size:
                write_shard_buffer(split_name, 'overlap_classifier', 
                                 shard_buffers[split_name]['overlap_classifier'],
                                 shard_counters[split_name]['overlap_classifier'])
                shard_counters[split_name]['overlap_classifier'] += 1
                shard_buffers[split_name]['overlap_classifier'] = []
        
        # Process diploid features
        for feat in features['diploid_features']:
            shard_buffers[split_name]['diploid_disentangler'].append(feat.to_dict())
            feature_counts['diploid'] += 1
            
            if len(shard_buffers[split_name]['diploid_disentangler']) >= config.shard_size:
                write_shard_buffer(split_name, 'diploid_disentangler',
                                 shard_buffers[split_name]['diploid_disentangler'],
                                 shard_counters[split_name]['diploid_disentangler'])
                shard_counters[split_name]['diploid_disentangler'] += 1
                shard_buffers[split_name]['diploid_disentangler'] = []
        
        # Process UL routing features
        for feat in features['ul_routing_features']:
            shard_buffers[split_name]['ul_routing'].append(feat.to_dict())
            feature_counts['ul_routing'] += 1
            
            if len(shard_buffers[split_name]['ul_routing']) >= config.shard_size:
                write_shard_buffer(split_name, 'ul_routing',
                                 shard_buffers[split_name]['ul_routing'],
                                 shard_counters[split_name]['ul_routing'])
                shard_counters[split_name]['ul_routing'] += 1
                shard_buffers[split_name]['ul_routing'] = []
        
        # Process SV features
        for feat in features['sv_features']:
            shard_buffers[split_name]['sv_detection'].append(feat.to_dict())
            feature_counts['sv'] += 1
            
            if len(shard_buffers[split_name]['sv_detection']) >= config.shard_size:
                write_shard_buffer(split_name, 'sv_detection',
                                 shard_buffers[split_name]['sv_detection'],
                                 shard_counters[split_name]['sv_detection'])
                shard_counters[split_name]['sv_detection'] += 1
                shard_buffers[split_name]['sv_detection'] = []
        
        # Free memory immediately after processing this genome
        del features
        import gc
        gc.collect()
    
    # Write remaining buffered data
    logger.info("Writing remaining buffered data...")
    for split_name in ['train', 'val', 'test']:
        for model_type in ['overlap_classifier', 'diploid_disentangler', 'ul_routing', 'sv_detection']:
            if shard_buffers[split_name][model_type]:
                write_shard_buffer(split_name, model_type,
                                 shard_buffers[split_name][model_type],
                                 shard_counters[split_name][model_type])
    
    logger.info(f"Streamed dataset:")
    logger.info(f"  Overlap features: {feature_counts['overlap']:,}")
    logger.info(f"  Diploid features: {feature_counts['diploid']:,}")
    logger.info(f"  UL routing features: {feature_counts['ul_routing']:,}")
    logger.info(f"  SV features: {feature_counts['sv']:,}")
    
    # ========================================================================
    # PHASE 3: Save metadata and summary
    # ========================================================================
    
    elapsed = time.time() - start_time
    total_stats['end_time'] = datetime.now().isoformat()
    total_stats['elapsed_seconds'] = elapsed
    
    # Add dataset statistics
    total_stats['dataset_stats'] = {
        'overlap_features': feature_counts['overlap'],
        'diploid_features': feature_counts['diploid'],
        'ul_routing_features': feature_counts['ul_routing'],
        'sv_features': feature_counts['sv']
    }
    
    # Add split distribution
    total_stats['splits'] = {
        'train': {'genome_count': len(train_indices)},
        'val': {'genome_count': len(val_indices)},
        'test': {'genome_count': len(test_indices)}
    }
    
    # Save metadata JSON
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING CORPUS GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Genomes: {config.num_genomes}")
    logger.info(f"Total nodes: {total_stats['total_graph_nodes']:,}")
    logger.info(f"Total edges: {total_stats['total_graph_edges']:,}")
    logger.info(f"Total labels: {total_stats['total_labels']:,}")
    logger.info(f"Total features: {total_stats['total_features']:,}")
    logger.info(f"Dataset features:")
    logger.info(f"  Overlap: {feature_counts['overlap']:,}")
    logger.info(f"  Diploid: {feature_counts['diploid']:,}")
    logger.info(f"  UL Routing: {feature_counts['ul_routing']:,}")
    logger.info(f"  SV Detection: {feature_counts['sv']:,}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Metadata: {metadata_file}")
    logger.info("=" * 80)
    
    return total_stats


# ============================================================================
#                    CONVENIENCE FUNCTIONS
# ============================================================================

def list_scenarios() -> List[str]:
    """List available training scenarios."""
    return list(SCENARIOS.keys())


def get_scenario_info(scenario: str) -> Dict[str, Any]:
    """Get configuration info for a scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return asdict(SCENARIOS[scenario])


def estimate_generation_time(scenario: str) -> Dict[str, float]:
    """
    Estimate generation time for a scenario.
    
    Returns:
        Dictionary with 'minutes', 'seconds', 'disk_mb' estimates
    """
    config = SCENARIOS[scenario]
    
    # Rough time estimates (seconds per genome)
    genome_time = config.genome_size / 100_000  # 1 sec per 100kb
    read_time = (
        config.illumina_coverage / 10 +  # ~3 sec for 30×
        config.hifi_coverage / 10 +
        config.ont_coverage / 10 +
        config.ul_coverage / 5
    )
    graph_time = config.genome_size / 50_000  # 2 sec per 100kb
    label_time = 2  # ~2 sec per genome
    feature_time = 3  # ~3 sec per genome
    
    per_genome = genome_time + read_time + graph_time + label_time + feature_time
    total_seconds = per_genome * config.num_genomes
    
    # Rough disk estimates (MB per genome)
    reads_mb = config.genome_size / 1000 * (
        config.illumina_coverage +
        config.hifi_coverage +
        config.ont_coverage +
        config.ul_coverage
    ) / 100
    features_mb = config.genome_size / 10_000  # ~0.1 MB per 1Mb genome
    total_mb = (reads_mb + features_mb) * config.num_genomes
    
    return {
        'seconds': total_seconds,
        'minutes': total_seconds / 60,
        'hours': total_seconds / 3600,
        'disk_mb': total_mb,
        'disk_gb': total_mb / 1024
    }
