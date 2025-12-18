#!/usr/bin/env python3
"""
Generate real training data from downloaded sequencing datasets.

This script uses the CORRECT assembly architecture:
- Illumina: OLC assembly (ContigBuilder) → Artificial long reads → DBG
- HiFi: DBG assembly directly (skip OLC)
- ONT: DBG assembly directly (skip OLC)

The script:
1. Reads real FASTQ files from yeast and drosophila datasets
2. Routes to appropriate assembly engine based on technology
3. Runs assemblies with different k-mer values
4. Measures N50 for each k value
5. Labels the optimal k as the one with highest N50
6. Generates training CSV with features + labels
"""

import os
import sys
import gzip
from pathlib import Path
from collections import defaultdict
import csv
import argparse

# Add strandweaver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from strandweaver.assembly_core.illumina_olc_contig_module import ContigBuilder
from strandweaver.assembly_core.dbg_engine_module import build_dbg_from_long_reads
from strandweaver.io.fastq import read_fastq
from strandweaver.io.read import SeqRead
from strandweaver.read_correction.feature_extraction import FeatureExtractor


def read_fastq_file(filepath, limit=None):
    """Read FASTQ file (supports .gz compression)."""
    reads = []
    for read in read_fastq(filepath, sample_size=limit):
        reads.append(read)
    return reads


def calculate_n50(contigs):
    """Calculate N50 from list of contig SeqRead objects."""
    if not contigs:
        return 0
    
    lengths = sorted([len(c.sequence) for c in contigs], reverse=True)
    total_length = sum(lengths)
    
    if total_length == 0:
        return 0
    
    cumsum = 0
    for length in lengths:
        cumsum += length
        if cumsum >= total_length / 2:
            return length
    
    return lengths[-1] if lengths else 0


def assemble_with_k(reads, k_value, technology='illumina', min_overlap=None):
    """
    Assemble reads with specified k-mer size using technology-appropriate engine.
    
    - Illumina: OLC assembly (ContigBuilder)
    - HiFi/ONT: DBG assembly (build_dbg_from_long_reads)
    """
    
    try:
        if technology == 'illumina':
            # Use OLC assembly for short reads
            if min_overlap is None:
                min_overlap = max(20, k_value)
            
            builder = ContigBuilder(
                k_size=k_value,
                min_overlap=min_overlap,
                min_contig_length=100,
                use_paired_end=True,
                use_gpu=False,
                use_adaptive_k=False
            )
            
            contigs = builder.build_contigs(reads, verbose=False)
        
        elif technology in ['hifi', 'ont']:
            # Use DBG assembly for long reads
            min_coverage = 2 if technology == 'hifi' else 3  # Higher for noisy ONT
            
            dbg = build_dbg_from_long_reads(
                long_reads=reads,
                base_k=k_value,
                min_coverage=min_coverage,
                ml_k_model=None  # No ML during training data generation
            )
            
            # Convert DBG nodes (unitigs) to contigs
            contigs = []
            for node_id, node in dbg.nodes.items():
                contig = SeqRead(
                    id=f"contig_{node_id}",
                    sequence=node.seq,
                    quality="I" * len(node.seq)  # Placeholder quality
                )
                contigs.append(contig)
        
        else:
            raise ValueError(f"Unknown technology: {technology}")
        
        return {
            'contigs': contigs,
            'n50': calculate_n50(contigs),
            'num_contigs': len(contigs),
            'total_length': sum(len(c.sequence) for c in contigs),
            'longest_contig': max(len(c.sequence) for c in contigs) if contigs else 0
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'contigs': [],
            'n50': 0,
            'num_contigs': 0,
            'total_length': 0,
            'longest_contig': 0
        }


def generate_training_example(dataset_name, reads_files, technology, k_values, read_limit=None):
    """Generate one training example by testing multiple k values."""
    print(f"\nProcessing {dataset_name} ({technology})...")
    
    # Extract features from the first file
    extractor = FeatureExtractor(subsample=read_limit)
    features = extractor.extract_from_file(reads_files[0])
    
    print(f"  Reads: {features.num_reads}")
    print(f"  Mean length: {features.mean_read_length:.1f} bp")
    print(f"  Coverage: {features.estimated_coverage:.1f}x" if features.estimated_coverage else "  Coverage: Unknown")
    
    # Read actual sequences for assembly
    print(f"  Loading sequences...")
    all_reads = []
    for filepath in reads_files:
        reads = read_fastq_file(filepath, limit=read_limit)
        all_reads.extend(reads)
    
    print(f"  Total reads loaded: {len(all_reads)}")
    
    # Test each k value
    results = {}
    for k in k_values:
        print(f"  Testing k={k}...", end=' ', flush=True)
        result = assemble_with_k(all_reads, k, technology=technology)
        results[k] = result
        print(f"N50={result['n50']:,} bp, {result['num_contigs']} contigs")
    
    # Find optimal k (highest N50)
    optimal_k = max(results.items(), key=lambda x: x[1]['n50'])
    
    print(f"  ✓ Optimal k={optimal_k[0]} (N50={optimal_k[1]['n50']:,} bp)")
    
    return {
        'dataset': dataset_name,
        'technology': technology,
        'features': features,
        'optimal_k': optimal_k[0],
        'k_results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Generate real training data from sequencing datasets')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of reads per dataset (for testing)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[21, 31, 41, 51, 61, 71],
                       help='K-mer values to test (default: 21 31 41 51 61 71)')
    parser.add_argument('--output', type=str, default='real_training_data.csv',
                       help='Output CSV file (default: real_training_data.csv)')
    args = parser.parse_args()
    
    data_dir = Path('/Users/patrickgrady/Assembler_Test_Reads')
    
    # Define datasets
    datasets = [
        {
            'name': 'yeast_illumina',
            'technology': 'illumina',
            'files': [
                data_dir / 'yeast' / 'SRR36206016_illumina_1.fastq',
                data_dir / 'yeast' / 'SRR36206016_illumina_2.fastq'
            ]
        },
        {
            'name': 'yeast_hifi',
            'technology': 'hifi',
            'files': [data_dir / 'yeast' / 'SRR31637145_hifi.fastq.gz']
        },
        {
            'name': 'yeast_ont',
            'technology': 'ont',
            'files': [data_dir / 'yeast' / 'ERR15849533_ont.fastq.gz']
        },
        {
            'name': 'drosophila_illumina',
            'technology': 'illumina',
            'files': [
                data_dir / 'drosophila' / 'SRR36087049_illumina_1.fastq.gz',
                data_dir / 'drosophila' / 'SRR36087049_illumina_2.fastq.gz'
            ]
        },
        {
            'name': 'drosophila_hifi',
            'technology': 'hifi',
            'files': [data_dir / 'drosophila' / 'SRR33554835_hifi.fastq.gz']
        },
        {
            'name': 'drosophila_ont',
            'technology': 'ont',
            'files': [data_dir / 'drosophila' / 'SRR36086010_ont.fastq']
        }
    ]
    
    print("=" * 70)
    print("Real Training Data Generation")
    print("=" * 70)
    print(f"\nK-mer values to test: {args.k_values}")
    if args.limit:
        print(f"Read limit per dataset: {args.limit:,}")
    print()
    
    training_examples = []
    
    for dataset in datasets:
        # Check if files exist
        missing = [f for f in dataset['files'] if not f.exists()]
        if missing:
            print(f"\n⚠️  Skipping {dataset['name']}: Missing files")
            for f in missing:
                print(f"    - {f}")
            continue
        
        # Generate training example
        example = generate_training_example(
            dataset['name'],
            dataset['files'],  # Pass list of file paths
            dataset['technology'],
            args.k_values,
            read_limit=args.limit
        )
        training_examples.append(example)
    
    # Save to CSV
    print(f"\n{'=' * 70}")
    print(f"Saving training data to {args.output}...")
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['dataset', 'technology', 'num_reads', 'mean_read_length', 
                 'read_length_std', 'mean_base_quality', 'gc_content', 'estimated_coverage',
                 'optimal_dbg_k', 'optimal_ul_overlap_k', 'optimal_extension_k', 'optimal_polish_k']
        writer.writerow(header)
        
        # Data rows
        for example in training_examples:
            features = example['features']
            optimal_k = example['optimal_k']
            
            row = [
                example['dataset'],
                example['technology'],
                features.num_reads,
                features.mean_read_length,
                features.read_length_std,
                features.mean_base_quality,
                features.gc_content,
                features.estimated_coverage if features.estimated_coverage else 0,
                optimal_k,  # DBG k
                optimal_k * 3 if example['technology'] == 'ont' else optimal_k,  # UL overlap k
                optimal_k + 10,  # Extension k
                optimal_k  # Polish k
            ]
            writer.writerow(row)
    
    print(f"✓ Saved {len(training_examples)} training examples")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for example in training_examples:
        print(f"\n{example['dataset']} ({example['technology']}):")
        print(f"  Optimal k: {example['optimal_k']}")
        print(f"  Results by k:")
        for k, result in sorted(example['k_results'].items()):
            marker = " ← BEST" if k == example['optimal_k'] else ""
            print(f"    k={k:2d}: N50={result['n50']:8,} bp, "
                  f"{result['num_contigs']:4d} contigs{marker}")
    
    print(f"\n{'=' * 70}")
    print(f"✓ Training data generation complete!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
