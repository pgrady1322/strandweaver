"""
Training Data Generation Demo

Demonstrates the complete training data pipeline:
1. Generate synthetic diploid genome with SVs
2. Simulate reads (Illumina, HiFi, ONT, UL, Hi-C)
3. Export simulated data for downstream assembly

This example shows how to create realistic training data for ML models.

Usage:
    python examples/training_data_demo.py

Author: StrandWeaver Training Infrastructure
Date: December 2025
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.training.genome_simulator import (
    GenomeConfig,
    generate_diploid_genome,
    SVType
)
from strandweaver.training.read_simulator import (
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_fastq,
    write_paired_fastq
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("TRAINING DATA GENERATION PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Generate Synthetic Diploid Genome
    # ========================================================================
    
    print("STEP 1: Generating synthetic diploid genome...")
    print("-" * 80)
    
    genome_config = GenomeConfig(
        length=500_000,  # 500kb genome (faster for demo)
        gc_content=0.42,
        repeat_density=0.30,  # Reduced for demo
        snp_rate=0.001,
        indel_rate=0.0001,
        sv_density=0.00005,  # ~2-3 SVs expected
        num_centromeres=1,
        centromere_length=10_000,
        random_seed=12345  # Reproducible
    )
    
    diploid = generate_diploid_genome(genome_config)
    
    print(f"✅ Generated diploid genome:")
    print(f"   Haplotype A: {len(diploid.hapA):,} bp")
    print(f"   Haplotype B: {len(diploid.hapB):,} bp")
    print(f"   SNPs: {len(diploid.snp_positions):,}")
    print(f"   Indels: {len(diploid.indel_positions):,}")
    print(f"   SVs: {len(diploid.sv_truth_table)}")
    print()
    
    # Show SV details
    if diploid.sv_truth_table:
        print("Structural Variants:")
        for sv in diploid.sv_truth_table:
            print(f"   - {sv.sv_type.value:12s} | Hap{sv.haplotype} | "
                  f"{sv.pos:,}-{sv.end:,} ({sv.size:,} bp) | {sv.description}")
        print()
    
    # ========================================================================
    # STEP 2: Simulate Illumina Reads
    # ========================================================================
    
    print("STEP 2: Simulating Illumina paired-end reads...")
    print("-" * 80)
    
    illumina_config = IlluminaConfig(
        coverage=10.0,  # Reduced for demo
        read_length=150,
        insert_size_mean=350,
        random_seed=12345
    )
    
    illumina_A = simulate_illumina_reads(diploid.hapA, illumina_config, haplotype='A')
    illumina_B = simulate_illumina_reads(diploid.hapB, illumina_config, haplotype='B')
    
    print(f"✅ Generated Illumina reads:")
    print(f"   Haplotype A: {len(illumina_A):,} pairs")
    print(f"   Haplotype B: {len(illumina_B):,} pairs")
    print(f"   Total reads: {(len(illumina_A) + len(illumina_B)) * 2:,}")
    print()
    
    # ========================================================================
    # STEP 3: Simulate HiFi Long Reads
    # ========================================================================
    
    print("STEP 3: Simulating PacBio HiFi long reads...")
    print("-" * 80)
    
    hifi_config = HiFiConfig(
        coverage=10.0,
        read_length_mean=15_000,
        read_length_std=5_000,
        error_rate=0.001,
        random_seed=12345
    )
    
    hifi_A = simulate_long_reads(diploid.hapA, hifi_config, haplotype='A', read_type='hifi')
    hifi_B = simulate_long_reads(diploid.hapB, hifi_config, haplotype='B', read_type='hifi')
    
    print(f"✅ Generated HiFi reads:")
    print(f"   Haplotype A: {len(hifi_A):,} reads")
    print(f"   Haplotype B: {len(hifi_B):,} reads")
    
    # Calculate stats
    hifi_all = hifi_A + hifi_B
    avg_len = sum(len(r.sequence) for r in hifi_all) / len(hifi_all)
    print(f"   Average length: {avg_len:,.0f} bp")
    print()
    
    # ========================================================================
    # STEP 4: Simulate ONT Reads
    # ========================================================================
    
    print("STEP 4: Simulating Oxford Nanopore reads...")
    print("-" * 80)
    
    ont_config = ONTConfig(
        coverage=10.0,
        read_length_mean=20_000,
        read_length_std=10_000,
        error_rate=0.05,
        random_seed=12345
    )
    
    ont_A = simulate_long_reads(diploid.hapA, ont_config, haplotype='A', read_type='ont')
    ont_B = simulate_long_reads(diploid.hapB, ont_config, haplotype='B', read_type='ont')
    
    print(f"✅ Generated ONT reads:")
    print(f"   Haplotype A: {len(ont_A):,} reads")
    print(f"   Haplotype B: {len(ont_B):,} reads")
    
    ont_all = ont_A + ont_B
    avg_len = sum(len(r.sequence) for r in ont_all) / len(ont_all)
    print(f"   Average length: {avg_len:,.0f} bp")
    print()
    
    # ========================================================================
    # STEP 5: Simulate Ultralong Reads
    # ========================================================================
    
    print("STEP 5: Simulating ultralong ONT reads...")
    print("-" * 80)
    
    ul_config = ULConfig(
        coverage=3.0,  # Lower coverage for UL
        read_length_mean=100_000,
        read_length_std=50_000,
        error_rate=0.08,
        random_seed=12345
    )
    
    ul_A = simulate_long_reads(diploid.hapA, ul_config, haplotype='A', read_type='ul')
    ul_B = simulate_long_reads(diploid.hapB, ul_config, haplotype='B', read_type='ul')
    
    print(f"✅ Generated UL reads:")
    print(f"   Haplotype A: {len(ul_A):,} reads")
    print(f"   Haplotype B: {len(ul_B):,} reads")
    
    ul_all = ul_A + ul_B
    if ul_all:
        avg_len = sum(len(r.sequence) for r in ul_all) / len(ul_all)
        max_len = max(len(r.sequence) for r in ul_all)
        print(f"   Average length: {avg_len:,.0f} bp")
        print(f"   Maximum length: {max_len:,} bp")
    print()
    
    # ========================================================================
    # STEP 6: Simulate Hi-C Reads
    # ========================================================================
    
    print("STEP 6: Simulating Hi-C proximity ligation reads...")
    print("-" * 80)
    
    hic_config = HiCConfig(
        num_pairs=50_000,  # Reduced for demo
        read_length=150,
        cis_fraction=0.90,
        distance_decay_rate=1.0,
        random_seed=12345
    )
    
    hic_pairs = simulate_hic_reads(diploid.hapA, diploid.hapB, hic_config)
    
    print(f"✅ Generated Hi-C reads:")
    print(f"   Read pairs: {len(hic_pairs):,}")
    
    # Calculate cis/trans stats
    cis_count = sum(1 for p in hic_pairs if p.read1.haplotype == p.read2.haplotype)
    trans_count = len(hic_pairs) - cis_count
    print(f"   Cis contacts: {cis_count:,} ({100*cis_count/len(hic_pairs):.1f}%)")
    print(f"   Trans contacts: {trans_count:,} ({100*trans_count/len(hic_pairs):.1f}%)")
    print()
    
    # ========================================================================
    # STEP 7: Write Output Files
    # ========================================================================
    
    print("STEP 7: Writing output files...")
    print("-" * 80)
    
    output_dir = Path("training_data_output")
    output_dir.mkdir(exist_ok=True)
    
    # Write reference genomes
    with open(output_dir / "reference_hapA.fasta", 'w') as f:
        f.write(">chr1_hapA\n")
        # Write in 80-character lines
        for i in range(0, len(diploid.hapA), 80):
            f.write(diploid.hapA[i:i+80] + "\n")
    
    with open(output_dir / "reference_hapB.fasta", 'w') as f:
        f.write(">chr1_hapB\n")
        for i in range(0, len(diploid.hapB), 80):
            f.write(diploid.hapB[i:i+80] + "\n")
    
    print(f"✅ Wrote reference genomes:")
    print(f"   {output_dir}/reference_hapA.fasta")
    print(f"   {output_dir}/reference_hapB.fasta")
    
    # Write ground truth SVs
    with open(output_dir / "sv_truth.tsv", 'w') as f:
        f.write("chrom\thaplotype\tsv_type\tstart\tend\tsize\tdescription\n")
        for sv in diploid.sv_truth_table:
            f.write(f"{sv.chrom}\t{sv.haplotype}\t{sv.sv_type.value}\t"
                   f"{sv.pos}\t{sv.end}\t{sv.size}\t{sv.description}\n")
    print(f"   {output_dir}/sv_truth.tsv")
    
    # Write Illumina reads
    write_paired_fastq(
        illumina_A + illumina_B,
        str(output_dir / "illumina_R1.fastq"),
        str(output_dir / "illumina_R2.fastq")
    )
    print(f"   {output_dir}/illumina_R1.fastq")
    print(f"   {output_dir}/illumina_R2.fastq")
    
    # Write HiFi reads
    write_fastq(hifi_A + hifi_B, str(output_dir / "hifi.fastq"))
    print(f"   {output_dir}/hifi.fastq")
    
    # Write ONT reads
    write_fastq(ont_A + ont_B, str(output_dir / "ont.fastq"))
    print(f"   {output_dir}/ont.fastq")
    
    # Write UL reads
    write_fastq(ul_A + ul_B, str(output_dir / "ul.fastq"))
    print(f"   {output_dir}/ul.fastq")
    
    # Write Hi-C reads
    write_paired_fastq(
        hic_pairs,
        str(output_dir / "hic_R1.fastq"),
        str(output_dir / "hic_R2.fastq")
    )
    print(f"   {output_dir}/hic_R1.fastq")
    print(f"   {output_dir}/hic_R2.fastq")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("=" * 80)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Genome: {len(diploid.hapA):,} bp (diploid)")
    print(f"  SNPs: {len(diploid.snp_positions):,}")
    print(f"  Indels: {len(diploid.indel_positions):,}")
    print(f"  SVs: {len(diploid.sv_truth_table)}")
    print()
    print(f"  Illumina: {(len(illumina_A) + len(illumina_B)) * 2:,} reads")
    print(f"  HiFi: {len(hifi_A) + len(hifi_B):,} reads")
    print(f"  ONT: {len(ont_A) + len(ont_B):,} reads")
    print(f"  UL: {len(ul_A) + len(ul_B):,} reads")
    print(f"  Hi-C: {len(hic_pairs) * 2:,} reads")
    print()
    print(f"  Output directory: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Assemble reads using StrandWeaver pipeline")
    print("  2. Extract ground-truth labels by aligning to reference")
    print("  3. Build training features from assembly graph")
    print("  4. Train ML models on labeled data")
    print()


if __name__ == "__main__":
    main()
