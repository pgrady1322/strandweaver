"""
Simple Training Data Generator

Quick and functional training data generation for Phase 5.4 ML model training.
Bypasses the complex orchestrator to get you started quickly.

Usage:
    python examples/simple_training_data_generator.py

Generates:
    - 10 synthetic diploid genomes
    - Simulated reads (Illumina, HiFi, ONT)
    - Training-ready datasets
"""

import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.genome_simulator import generate_diploid_genome, GenomeConfig
from strandweaver.training.read_simulator import (
    simulate_illumina_reads, simulate_long_reads, simulate_hic_reads,
    IlluminaConfig, HiFiConfig, ONTConfig, HiCConfig,
    write_fastq, write_paired_fastq
)


def generate_simple_test_dataset(
    num_genomes: int = 10,
    output_dir: str = 'training_data/simple_test',
    genome_size: int = 100_000
):
    """
    Generate a simple test dataset for ML training.
    
    Args:
        num_genomes: Number of genomes to generate
        output_dir: Output directory
        genome_size: Size of each genome
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" SIMPLE TRAINING DATA GENERATOR")
    print("="*70)
    print(f"Configuration:")
    print(f"  • Genomes: {num_genomes}")
    print(f"  • Genome size: {genome_size:,} bp each")
    print(f"  • Output: {output_dir}/")
    print()
    
    start_time = time.time()
    
    all_genomes = []
    total_reads = {"illumina": 0, "hifi": 0, "ont": 0}
    
    for i in range(num_genomes):
        print(f"[{i+1}/{num_genomes}] Generating genome {i+1}...")
        
        # Generate diploid genome
        config = GenomeConfig(
            length=genome_size,
            gc_content=0.42,
            repeat_density=0.45,
            snp_rate=0.001,
            sv_density=0.00005,
            sv_max_size=min(10_000, genome_size // 10),  # Keep SVs reasonable
            random_seed=42 + i  # Different seed per genome
        )
        
        diploid = generate_diploid_genome(config)
        
        # Simulate reads (low coverage for quick testing)
        ill_config = IlluminaConfig(coverage=10.0)
        ill_reads = simulate_illumina_reads(diploid.hapA, ill_config, haplotype='A')
        
        hifi_config = HiFiConfig(coverage=10.0)
        hifi_reads = simulate_long_reads(diploid.hapA, hifi_config, read_type='hifi')
        
        ont_config = ONTConfig(coverage=10.0)
        ont_reads = simulate_long_reads(diploid.hapA, ont_config, read_type='ont')
        
        # Save genome info
        genome_dir = output_path / f"genome_{i:03d}"
        genome_dir.mkdir(exist_ok=True)
        
        # Save reference sequences
        with open(genome_dir / "reference_hapA.fasta", 'w') as f:
            f.write(f">hapA\n{diploid.hapA}\n")
        with open(genome_dir / "reference_hapB.fasta", 'w') as f:
            f.write(f">hapB\n{diploid.hapB}\n")
        
        # Save SV truth table
        sv_truth = []
        for sv in diploid.sv_truth_table:
            sv_truth.append({
                'type': sv.sv_type.value,
                'pos': sv.pos,
                'end': sv.end,
                'size': sv.size,
                'haplotype': sv.haplotype
            })
        
        with open(genome_dir / "sv_truth.json", 'w') as f:
            json.dump(sv_truth, f, indent=2)
        
        # Save reads
        write_paired_fastq(
            ill_reads,
            str(genome_dir / "illumina_R1.fastq"),
            str(genome_dir / "illumina_R2.fastq")
        )
        write_fastq(hifi_reads, str(genome_dir / "hifi.fastq"))
        write_fastq(ont_reads, str(genome_dir / "ont.fastq"))
        
        # Track stats
        total_reads["illumina"] += len(ill_reads)
        total_reads["hifi"] += len(hifi_reads)
        total_reads["ont"] += len(ont_reads)
        
        all_genomes.append({
            'id': i,
            'length_A': len(diploid.hapA),
            'length_B': len(diploid.hapB),
            'num_snps': len(diploid.snp_positions),
            'num_indels': len(diploid.indel_positions),
            'num_svs': len(diploid.sv_truth_table),
            'num_illumina_pairs': len(ill_reads),
            'num_hifi': len(hifi_reads),
            'num_ont': len(ont_reads)
        })
        
        print(f"       ✅ Generated: {len(diploid.hapA):,} bp, {len(diploid.sv_truth_table)} SVs")
        print(f"       ✅ Reads: {len(ill_reads)} Illumina pairs, {len(hifi_reads)} HiFi, {len(ont_reads)} ONT")
    
    elapsed = time.time() - start_time
    
    # Save metadata
    metadata = {
        'num_genomes': num_genomes,
        'genome_size': genome_size,
        'total_reads': total_reads,
        'generation_time_seconds': elapsed,
        'genomes': all_genomes
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("="*70)
    print(" GENERATION COMPLETE!")
    print("="*70)
    print(f"✅ Generated {num_genomes} genomes in {elapsed/60:.1f} minutes")
    print(f"✅ Total reads:")
    print(f"    • Illumina: {total_reads['illumina']:,} pairs")
    print(f"    • HiFi: {total_reads['hifi']:,} reads")
    print(f"    • ONT: {total_reads['ont']:,} reads")
    print(f"✅ Output directory: {output_dir}/")
    print()
    print("Next steps:")
    print("  1. Assemble reads into graphs")
    print("  2. Extract ground-truth labels")
    print("  3. Build ML features")
    print("  4. Train models!")
    print("="*70)
    
    return metadata


if __name__ == "__main__":
    # Generate small test dataset
    metadata = generate_simple_test_dataset(
        num_genomes=10,
        output_dir='training_data/simple_test_v1',
        genome_size=100_000  # 100kb genomes for quick testing
    )
    
    print("\n✅ Simple test dataset ready for Phase 5.4 ML training!")
