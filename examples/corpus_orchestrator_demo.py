"""
Training Corpus Orchestrator Demo

Demonstrates complete end-to-end training data generation:
1. Generate synthetic diploid genomes
2. Simulate sequencing reads
3. Build assembly graphs
4. Extract ground-truth labels
5. Build ML-ready features
6. Package into sharded datasets
7. Generate comprehensive metadata

This demo shows all available scenarios and usage patterns.
"""

from pathlib import Path
import shutil
from strandweaver.training import (
    generate_training_corpus,
    TrainingCorpusOrchestrator,
    SCENARIOS,
    CorpusMetadata
)


def demo_scenario_overview():
    """Show all available training scenarios."""
    print("=" * 80)
    print("AVAILABLE TRAINING SCENARIOS")
    print("=" * 80)
    
    for name, config in SCENARIOS.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"Description: {config.description}")
        print(f"Genome: {config.genome_length:,} bp × {config.num_chromosomes} chromosomes")
        print(f"Variants: SNP={config.snp_rate:.6f}, Indel={config.indel_rate:.6f}, "
              f"SV={config.sv_rate:.6f}")
        print(f"Repeats: {config.repeat_density:.1%}")
        print(f"Coverage: HiFi={config.hifi_coverage}x, ONT={config.ont_coverage}x, "
              f"UL={config.ul_coverage}x, Hi-C={config.hic_coverage}x")


def demo_simple_generation():
    """Generate a small training corpus with the simple scenario."""
    print("\n" + "=" * 80)
    print("DEMO 1: Simple Training Corpus Generation")
    print("=" * 80)
    
    # Clean up previous run
    output_dir = Path('demo_corpus_simple')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("\nGenerating small corpus (5 genomes, simple scenario)...")
    print("This will take a few minutes...\n")
    
    # Generate corpus with convenience function
    metadata = generate_training_corpus(
        scenario='simple',
        num_genomes=5,
        output_dir=output_dir,
        version='demo_v1',
        num_workers=2,
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total genomes: {metadata.statistics.num_genomes}")
    print(f"Total reads: {metadata.statistics.total_reads:,}")
    print(f"Total graph nodes: {metadata.statistics.total_nodes:,}")
    print(f"Total graph edges: {metadata.statistics.total_edges:,}")
    print(f"Dataset size: {metadata.statistics.total_dataset_size_mb:.1f} MB")
    print(f"Output directory: {metadata.output_directory}")
    
    # Show directory structure
    print("\nDirectory structure:")
    for subdir in ['genomes', 'reads', 'labels', 'features', 'datasets']:
        dir_path = output_dir / subdir
        if dir_path.exists():
            num_files = len(list(dir_path.rglob('*')))
            print(f"  {subdir}/: {num_files} files")
    
    return metadata


def demo_scenario_comparison():
    """Generate small corpora for different scenarios to compare."""
    print("\n" + "=" * 80)
    print("DEMO 2: Scenario Comparison")
    print("=" * 80)
    
    scenarios_to_test = ['simple', 'repeat_heavy', 'sv_dense']
    results = {}
    
    for scenario in scenarios_to_test:
        print(f"\n--- Generating {scenario} corpus (2 genomes) ---")
        
        output_dir = Path(f'demo_corpus_{scenario}')
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        metadata = generate_training_corpus(
            scenario=scenario,
            num_genomes=2,
            output_dir=output_dir,
            version='comparison_v1',
            num_workers=1,
            seed=42
        )
        
        results[scenario] = metadata.statistics
        
        print(f"  SNPs: {metadata.statistics.total_snps:,}")
        print(f"  Indels: {metadata.statistics.total_indels:,}")
        print(f"  SVs: {metadata.statistics.total_svs}")
        print(f"  Reads: {metadata.statistics.total_reads:,}")
        print(f"  Nodes: {metadata.statistics.total_nodes:,}")
    
    # Compare results
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Simple':>15} {'Repeat Heavy':>15} {'SV Dense':>15}")
    print("-" * 80)
    
    metrics = [
        ('Total SNPs', 'total_snps'),
        ('Total Indels', 'total_indels'),
        ('Total SVs', 'total_svs'),
        ('Total Reads', 'total_reads'),
        ('Graph Nodes', 'total_nodes'),
        ('Graph Edges', 'total_edges')
    ]
    
    for label, attr in metrics:
        values = [getattr(results[s], attr) for s in scenarios_to_test]
        print(f"{label:<20} {values[0]:>15,} {values[1]:>15,} {values[2]:>15,}")


def demo_custom_orchestrator():
    """Use the orchestrator class directly for more control."""
    print("\n" + "=" * 80)
    print("DEMO 3: Custom Orchestrator Usage")
    print("=" * 80)
    
    output_dir = Path('demo_corpus_custom')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("\nCreating custom orchestrator...")
    orchestrator = TrainingCorpusOrchestrator(
        scenario='balanced',
        num_genomes=3,
        output_dir=output_dir,
        version='custom_v1',
        num_workers=2,
        seed=12345
    )
    
    print(f"Scenario: {orchestrator.config.name}")
    print(f"Description: {orchestrator.config.description}")
    print(f"Genome length: {orchestrator.config.genome_length:,} bp")
    print(f"Coverage: HiFi={orchestrator.config.hifi_coverage}x")
    
    print("\nGenerating corpus with custom parameters...")
    metadata = orchestrator.generate_corpus(
        shard_size=500,  # Smaller shards
        train_split=0.8,  # 80/10/10 split
        val_split=0.1,
        test_split=0.1
    )
    
    print("\nGeneration complete!")
    print(f"Total shards: {metadata.statistics.total_shards}")
    print(f"Edge features: {metadata.statistics.total_edge_features:,}")
    print(f"Node features: {metadata.statistics.total_node_features:,}")
    print(f"SV features: {metadata.statistics.total_sv_features:,}")
    
    # Show label distribution
    print("\nEdge label distribution:")
    for label, count in sorted(metadata.statistics.edge_labels.items()):
        print(f"  {label}: {count:,}")
    
    print("\nNode haplotype distribution:")
    for hap, count in sorted(metadata.statistics.node_haplotype_labels.items()):
        print(f"  {hap}: {count:,}")
    
    return metadata


def demo_metadata_usage():
    """Show how to load and use corpus metadata."""
    print("\n" + "=" * 80)
    print("DEMO 4: Working with Corpus Metadata")
    print("=" * 80)
    
    # Load metadata from previous demo
    metadata_path = Path('demo_corpus_simple/metadata/corpus_metadata.json')
    
    if not metadata_path.exists():
        print("Run demo_simple_generation() first to create metadata")
        return
    
    print(f"\nLoading metadata from {metadata_path}...")
    metadata = CorpusMetadata.load(metadata_path)
    
    print(f"Version: {metadata.version}")
    print(f"Scenario: {metadata.scenario}")
    print(f"Generated: {metadata.generation_date}")
    print(f"Number of genomes: {metadata.num_genomes}")
    
    print("\nSV type distribution:")
    for sv_type, count in sorted(metadata.statistics.sv_types.items()):
        print(f"  {sv_type}: {count}")
    
    print("\nRead type distribution:")
    for read_type, count in sorted(metadata.statistics.reads_by_type.items()):
        print(f"  {read_type}: {count:,}")
    
    print("\nDataset file paths:")
    for dataset_type, paths in metadata.file_paths.items():
        print(f"  {dataset_type}: {len(paths)} files")


def demo_production_corpus():
    """
    Example of generating a production-scale training corpus.
    
    WARNING: This would take significant time and disk space.
    Commented out by default.
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Production Corpus Example (NOT EXECUTED)")
    print("=" * 80)
    
    print("\nExample production corpus generation:")
    print("""
    # Generate large balanced corpus
    metadata = generate_training_corpus(
        scenario='balanced',
        num_genomes=1000,
        output_dir=Path('training_data/balanced_v1'),
        version='v1.0',
        num_workers=8,
        seed=42
    )
    
    # Expected output:
    # - ~1000 genomes (1GB each)
    # - ~10M reads
    # - ~100K graph nodes
    # - ~500K edge labels
    # - ~10GB total dataset size
    # - Train/val/test splits
    # - Comprehensive metadata
    
    # Then generate specialized corpora
    for scenario in ['repeat_heavy', 'sv_dense', 'diploid_focus']:
        generate_training_corpus(
            scenario=scenario,
            num_genomes=500,
            output_dir=Path(f'training_data/{scenario}_v1'),
            version='v1.0',
            num_workers=8
        )
    """)
    
    print("\nProduction workflow:")
    print("1. Generate balanced corpus (1000 genomes) - general training")
    print("2. Generate specialized corpora (500 each) - focused training")
    print("3. Train models on balanced data first")
    print("4. Fine-tune on specialized data for hard cases")
    print("5. Validate on held-out test sets")


def main():
    """Run all demos."""
    print("=" * 80)
    print("TRAINING CORPUS ORCHESTRATOR - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Demo 1: Show available scenarios
    demo_scenario_overview()
    
    # Demo 2: Generate simple corpus
    metadata = demo_simple_generation()
    
    # Demo 3: Compare scenarios
    demo_scenario_comparison()
    
    # Demo 4: Custom orchestrator
    custom_metadata = demo_custom_orchestrator()
    
    # Demo 5: Metadata usage
    demo_metadata_usage()
    
    # Demo 6: Production example
    demo_production_corpus()
    
    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\nGenerated corpus directories:")
    print("  - demo_corpus_simple/")
    print("  - demo_corpus_simple/")
    print("  - demo_corpus_repeat_heavy/")
    print("  - demo_corpus_sv_dense/")
    print("  - demo_corpus_custom/")
    print("\nEach contains:")
    print("  - genomes/ - Synthetic FASTA files")
    print("  - reads/ - Simulated FASTQ files")
    print("  - labels/ - Ground-truth TSV files")
    print("  - features/ - ML-ready JSON/NPZ files")
    print("  - datasets/ - Packaged training datasets")
    print("  - metadata/ - Corpus statistics and metadata")
    print("  - README.md - Comprehensive documentation")
    print("\nNext steps:")
    print("  1. Examine the generated datasets")
    print("  2. Load datasets for model training")
    print("  3. Train ML models using the interfaces")
    print("  4. Evaluate on test sets")
    print("=" * 80)


if __name__ == '__main__':
    main()
