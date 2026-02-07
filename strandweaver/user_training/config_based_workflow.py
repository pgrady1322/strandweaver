"""
StrandWeaver v0.1.0

Config-Based Training Workflow

User-configurable training data generation that allows fine-grained control
over genome and read parameters without predefined scenarios.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md

"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import asdict

from .training_config import (
    UserTrainingConfig,
    UserGenomeConfig,
    UserReadConfig,
    ReadType,
    Ploidy
)

# Import from existing training infrastructure
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.synthetic_data_generator import (
    GenomeConfig,
    DiploidGenome,
    generate_diploid_genome,
    IlluminaConfig,
    HiFiConfig,
    ONTConfig,
    ULConfig,
    HiCConfig,
    AncientDNAConfig,
    simulate_illumina_reads,
    simulate_long_reads,
    simulate_hic_reads,
    write_fastq,
    write_paired_fastq
)

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Orchestrates user-configured training data generation.
    """
    
    def __init__(self, config: UserTrainingConfig):
        """
        Initialize generator with user configuration.
        
        Args:
            config: User training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved training configuration to {config_path}")
    
    def _convert_to_internal_genome_config(self, genome_idx: int) -> GenomeConfig:
        """
        Convert user genome config to internal GenomeConfig format.
        
        Args:
            genome_idx: Genome index for seed offset
            
        Returns:
            GenomeConfig for synthetic_data_generator
        """
        uc = self.config.genome_config
        
        # Calculate SV type fractions
        num_sv_types = len(uc.sv_types)
        sv_fraction = 1.0 / num_sv_types if num_sv_types > 0 else 0.0
        
        return GenomeConfig(
            length=uc.genome_size,
            gc_content=uc.gc_content,
            repeat_density=uc.repeat_density,
            tandem_repeat_fraction=0.6,  # Reasonable default
            num_centromeres=uc.centromere_count,
            gene_dense_fraction=uc.gene_dense_fraction,
            snp_rate=uc.snp_rate,
            indel_rate=uc.indel_rate,
            sv_density=uc.sv_density,
            sv_deletion_fraction=sv_fraction if 'deletion' in uc.sv_types else 0.0,
            sv_insertion_fraction=sv_fraction if 'insertion' in uc.sv_types else 0.0,
            sv_inversion_fraction=sv_fraction if 'inversion' in uc.sv_types else 0.0,
            sv_duplication_fraction=sv_fraction if 'duplication' in uc.sv_types else 0.0,
            sv_translocation_fraction=sv_fraction if 'translocation' in uc.sv_types else 0.0,
            random_seed=uc.random_seed + genome_idx if uc.random_seed else None
        )
    
    def _simulate_reads_for_genome(
        self,
        diploid_genome: DiploidGenome,
        genome_idx: int,
        output_subdir: Path
    ) -> Dict[str, List[Path]]:
        """
        Simulate all configured read types for a diploid genome.
        
        Args:
            diploid_genome: Generated diploid genome
            genome_idx: Genome index
            output_subdir: Output subdirectory for this genome
            
        Returns:
            Dictionary mapping read type to list of output files
        """
        output_files = {}
        
        for read_config in self.config.read_configs:
            read_type = read_config.read_type
            logger.info(f"  Simulating {read_type.value} reads (coverage={read_config.coverage}x)...")
            
            if read_type == ReadType.ILLUMINA:
                # Illumina paired-end
                config = IlluminaConfig(
                    coverage=read_config.coverage,
                    read_length=read_config.read_length_mean,
                    insert_size=read_config.insert_size_mean or 500,
                    insert_std=read_config.insert_size_std or 100,
                    error_rate=read_config.error_rate,
                    random_seed=genome_idx
                )
                
                # Simulate from both haplotypes
                reads_A = simulate_illumina_reads(diploid_genome.hapA, config, 'A')
                reads_B = simulate_illumina_reads(diploid_genome.hapB, config, 'B')
                all_reads = reads_A + reads_B
                
                # Write paired FASTQ
                r1_path = output_subdir / f'illumina_R1.fastq'
                r2_path = output_subdir / f'illumina_R2.fastq'
                write_paired_fastq(all_reads, str(r1_path), str(r2_path))
                output_files[read_type.value] = [r1_path, r2_path]
                
            elif read_type == ReadType.HIFI:
                # PacBio HiFi
                config = HiFiConfig(
                    coverage=read_config.coverage,
                    read_length_mean=read_config.read_length_mean,
                    read_length_std=read_config.read_length_std,
                    error_rate=read_config.error_rate,
                    random_seed=genome_idx
                )
                
                reads_A = simulate_long_reads(diploid_genome.hapA, config, 'A', 'hifi')
                reads_B = simulate_long_reads(diploid_genome.hapB, config, 'B', 'hifi')
                all_reads = reads_A + reads_B
                
                fastq_path = output_subdir / f'hifi.fastq'
                write_fastq(all_reads, str(fastq_path))
                output_files[read_type.value] = [fastq_path]
                
            elif read_type == ReadType.ONT:
                # Oxford Nanopore
                config = ONTConfig(
                    coverage=read_config.coverage,
                    read_length_mean=read_config.read_length_mean,
                    read_length_std=read_config.read_length_std,
                    error_rate=read_config.error_rate,
                    random_seed=genome_idx
                )
                
                reads_A = simulate_long_reads(diploid_genome.hapA, config, 'A', 'ont')
                reads_B = simulate_long_reads(diploid_genome.hapB, config, 'B', 'ont')
                all_reads = reads_A + reads_B
                
                fastq_path = output_subdir / f'ont.fastq'
                write_fastq(all_reads, str(fastq_path))
                output_files[read_type.value] = [fastq_path]
                
            elif read_type == ReadType.ULTRA_LONG:
                # Ultra-long ONT
                config = ULConfig(
                    coverage=read_config.coverage,
                    read_length_mean=read_config.read_length_mean,
                    read_length_std=read_config.read_length_std,
                    error_rate=read_config.error_rate,
                    random_seed=genome_idx
                )
                
                reads_A = simulate_long_reads(diploid_genome.hapA, config, 'A', 'ul')
                reads_B = simulate_long_reads(diploid_genome.hapB, config, 'B', 'ul')
                all_reads = reads_A + reads_B
                
                fastq_path = output_subdir / f'ultralong.fastq'
                write_fastq(all_reads, str(fastq_path))
                output_files[read_type.value] = [fastq_path]
                
            elif read_type == ReadType.HIC:
                # Hi-C proximity ligation
                num_pairs = int((diploid_genome.length_A * read_config.coverage) / (2 * read_config.read_length_mean))
                config = HiCConfig(
                    num_pairs=num_pairs,
                    read_length=read_config.read_length_mean,
                    error_rate=read_config.error_rate,
                    random_seed=genome_idx
                )
                
                hic_pairs = simulate_hic_reads(diploid_genome.hapA, diploid_genome.hapB, config)
                
                r1_path = output_subdir / f'hic_R1.fastq'
                r2_path = output_subdir / f'hic_R2.fastq'
                write_paired_fastq(hic_pairs, str(r1_path), str(r2_path))
                output_files[read_type.value] = [r1_path, r2_path]
                
            elif read_type == ReadType.ANCIENT_DNA:
                # Ancient DNA fragments
                config = AncientDNAConfig(
                    coverage=read_config.coverage,
                    read_length_mean=read_config.read_length_mean,
                    read_length_std=read_config.read_length_std,
                    error_rate=read_config.error_rate,
                    damage_rate=0.3,  # C->T damage rate
                    random_seed=genome_idx
                )
                
                # Simulate as single-end reads with damage
                reads_A = simulate_long_reads(diploid_genome.hapA, config, 'A', 'ont')
                reads_B = simulate_long_reads(diploid_genome.hapB, config, 'B', 'ont')
                all_reads = reads_A + reads_B
                
                fastq_path = output_subdir / f'ancient_dna.fastq'
                write_fastq(all_reads, str(fastq_path))
                output_files[read_type.value] = [fastq_path]
        
        return output_files
    
    def generate_single_genome(self, genome_idx: int) -> Dict[str, Any]:
        """
        Generate a single genome with all configured read types.
        
        Args:
            genome_idx: Index of genome to generate
            
        Returns:
            Dictionary with genome metadata and output paths
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Generating genome {genome_idx + 1}/{self.config.genome_config.num_genomes}")
        logger.info(f"{'='*80}")
        
        # Create output subdirectory
        genome_dir = self.output_dir / f'genome_{genome_idx:04d}'
        genome_dir.mkdir(exist_ok=True)
        
        # Generate diploid genome
        genome_config = self._convert_to_internal_genome_config(genome_idx)
        
        if self.config.genome_config.ploidy == Ploidy.HAPLOID:
            # Generate haploid (only hapA)
            logger.info("Generating haploid genome...")
            # TODO: Add haploid support to synthetic_data_generator
            raise NotImplementedError("Haploid genome generation not yet implemented")
        else:
            # Generate diploid
            diploid_genome = generate_diploid_genome(genome_config)
        
        # Save genome sequences
        with open(genome_dir / 'haplotype_A.fasta', 'w') as f:
            f.write(f'>hapA\n{diploid_genome.hapA}\n')
        
        with open(genome_dir / 'haplotype_B.fasta', 'w') as f:
            f.write(f'>hapB\n{diploid_genome.hapB}\n')
        
        # Save SV truth table
        sv_data = [
            {
                'sv_type': sv.sv_type.value,
                'haplotype': sv.haplotype,
                'chrom': sv.chrom,
                'pos': sv.pos,
                'end': sv.end,
                'size': sv.size,
                'description': sv.description
            }
            for sv in diploid_genome.sv_truth_table
        ]
        with open(genome_dir / 'sv_truth.json', 'w') as f:
            json.dump(sv_data, f, indent=2)
        
        # Simulate reads
        output_files = self._simulate_reads_for_genome(diploid_genome, genome_idx, genome_dir)
        
        # Create metadata
        metadata = {
            'genome_idx': genome_idx,
            'genome_size': self.config.genome_config.genome_size,
            'haplotype_A_length': len(diploid_genome.hapA),
            'haplotype_B_length': len(diploid_genome.hapB),
            'num_snps': len(diploid_genome.snp_positions),
            'num_indels': len(diploid_genome.indel_positions),
            'num_svs': len(diploid_genome.sv_truth_table),
            'output_files': {k: [str(p) for p in v] for k, v in output_files.items()}
        }
        
        with open(genome_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Genome {genome_idx} complete: {len(output_files)} read types generated")
        
        return metadata
    
    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all configured genomes and reads.
        
        Returns:
            Summary statistics and metadata
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"USER-CONFIGURED TRAINING DATA GENERATION")
        logger.info(f"{'='*80}")
        logger.info(f"Genome size: {self.config.genome_config.genome_size:,} bp")
        logger.info(f"Number of genomes: {self.config.genome_config.num_genomes}")
        logger.info(f"Ploidy: {self.config.genome_config.ploidy.name}")
        logger.info(f"GC content: {self.config.genome_config.gc_content:.2%}")
        logger.info(f"Repeat density: {self.config.genome_config.repeat_density:.2%}")
        logger.info(f"Read types: {', '.join(rc.read_type.value for rc in self.config.read_configs)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        # Generate all genomes
        all_metadata = []
        for i in range(self.config.genome_config.num_genomes):
            metadata = self.generate_single_genome(i)
            all_metadata.append(metadata)
        
        # Create summary
        elapsed = time.time() - start_time
        summary = {
            'config': self.config.to_dict(),
            'generation_time_seconds': elapsed,
            'generation_time_human': f"{elapsed/60:.1f} minutes",
            'num_genomes_generated': len(all_metadata),
            'total_bases_generated': sum(m['haplotype_A_length'] + m['haplotype_B_length'] for m in all_metadata),
            'genomes': all_metadata
        }
        
        # Save summary
        summary_path = self.output_dir / 'generation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING DATA GENERATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Generated: {len(all_metadata)} genomes")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"{'='*80}\n")
        
        return summary


def generate_custom_training_data(config: UserTrainingConfig) -> Dict[str, Any]:
    """
    Main entry point for user-configured training data generation.
    
    Args:
        config: User training configuration
        
    Returns:
        Summary dictionary with generation statistics
        
    Example:
        >>> from strandweaver.user_training import *
        >>> 
        >>> # Configure genome
        >>> genome_config = UserGenomeConfig(
        ...     genome_size=1_000_000,
        ...     num_genomes=10,
        ...     gc_content=0.42,
        ...     repeat_density=0.30,
        ...     ploidy=Ploidy.DIPLOID
        ... )
        >>> 
        >>> # Configure reads
        >>> read_configs = [
        ...     UserReadConfig(ReadType.HIFI, coverage=30),
        ...     UserReadConfig(ReadType.ULTRA_LONG, coverage=10),
        ...     UserReadConfig(ReadType.HIC, coverage=20)
        ... ]
        >>> 
        >>> # Create full config
        >>> config = UserTrainingConfig(
        ...     genome_config=genome_config,
        ...     read_configs=read_configs,
        ...     output_dir='training_data/custom'
        ... )
        >>> 
        >>> # Generate data
        >>> summary = generate_custom_training_data(config)
    """
    generator = TrainingDataGenerator(config)
    return generator.generate_all()
