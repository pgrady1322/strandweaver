"""
StrandWeaver v0.1.0

User Training Configuration

Dataclasses and configuration objects for user-configurable training data generation.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from pathlib import Path


class ReadType(Enum):
    """Supported sequencing read types."""
    ILLUMINA = "illumina"
    HIFI = "hifi"
    ONT = "ont"
    ULTRA_LONG = "ultra_long"
    HIC = "hic"
    ANCIENT_DNA = "ancient_dna"


class Ploidy(Enum):
    """Ploidy levels."""
    HAPLOID = 1
    DIPLOID = 2
    TRIPLOID = 3
    TETRAPLOID = 4


@dataclass
class UserGenomeConfig:
    """
    User-specified genome generation parameters.
    
    Attributes:
        genome_size: Genome size in base pairs (e.g., 1_000_000 for 1Mb)
        num_genomes: Number of independent genomes to generate
        gc_content: Base GC content (0.0-1.0)
        repeat_density: Fraction of genome that is repetitive (0.0-1.0)
        ploidy: Ploidy level (haploid, diploid, triploid, tetraploid)
        snp_rate: SNP rate between haplotypes (per bp, e.g., 0.001 = 0.1%)
        indel_rate: Indel rate between haplotypes (per bp)
        sv_density: Structural variant density (per bp)
        sv_types: SV types to include (deletion, insertion, inversion, duplication, translocation)
        centromere_count: Number of centromeric regions per genome
        gene_dense_fraction: Fraction of genome that is gene-dense (higher GC)
        random_seed: Random seed for reproducibility (None = random)
    """
    genome_size: int = 1_000_000
    num_genomes: int = 10
    gc_content: float = 0.42
    repeat_density: float = 0.30
    ploidy: Ploidy = Ploidy.DIPLOID
    
    # Variation parameters
    snp_rate: float = 0.001
    indel_rate: float = 0.0001
    sv_density: float = 0.00001
    sv_types: List[str] = field(default_factory=lambda: ['deletion', 'insertion', 'inversion', 'duplication'])
    
    # Genome structure
    centromere_count: int = 1
    gene_dense_fraction: float = 0.30
    
    # Reproducibility
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 100 <= self.genome_size <= 1_000_000_000:
            raise ValueError(f"Genome size must be between 100 bp and 1 Gb, got {self.genome_size}")
        if not 1 <= self.num_genomes <= 10000:
            raise ValueError(f"Number of genomes must be between 1 and 10000, got {self.num_genomes}")
        if not 0.0 <= self.gc_content <= 1.0:
            raise ValueError(f"GC content must be between 0 and 1, got {self.gc_content}")
        if not 0.0 <= self.repeat_density <= 1.0:
            raise ValueError(f"Repeat density must be between 0 and 1, got {self.repeat_density}")
        if not 0.0 <= self.snp_rate <= 0.1:
            raise ValueError(f"SNP rate must be between 0 and 0.1, got {self.snp_rate}")
        if not 0.0 <= self.indel_rate <= 0.01:
            raise ValueError(f"Indel rate must be between 0 and 0.01, got {self.indel_rate}")
        if not 0.0 <= self.sv_density <= 0.001:
            raise ValueError(f"SV density must be between 0 and 0.001, got {self.sv_density}")


@dataclass
class UserReadConfig:
    """
    User-specified read generation parameters for a single technology.
    
    Attributes:
        read_type: Type of sequencing reads
        coverage: Sequencing coverage depth (e.g., 30 for 30x)
        read_length_mean: Mean read length in bp (for Illumina/HiFi/ONT)
        read_length_std: Standard deviation of read length
        error_rate: Base error rate (substitutions + indels)
        insert_size_mean: Insert size for paired-end (Illumina, Hi-C)
        insert_size_std: Insert size standard deviation
    """
    read_type: ReadType
    coverage: float = 30.0
    
    # Read characteristics
    read_length_mean: int = 150
    read_length_std: int = 20
    error_rate: float = 0.01
    
    # Paired-end parameters (Illumina, Hi-C)
    insert_size_mean: Optional[int] = 500
    insert_size_std: Optional[int] = 100
    
    def __post_init__(self):
        """Validate and set defaults based on read type."""
        # Set technology-specific defaults
        if self.read_type == ReadType.ILLUMINA:
            if self.read_length_mean == 150:  # If still default
                self.read_length_mean = 150
            if self.error_rate == 0.01:
                self.error_rate = 0.001
                
        elif self.read_type == ReadType.HIFI:
            if self.read_length_mean == 150:
                self.read_length_mean = 15000
            if self.error_rate == 0.01:
                self.error_rate = 0.001
                
        elif self.read_type == ReadType.ONT:
            if self.read_length_mean == 150:
                self.read_length_mean = 20000
            if self.error_rate == 0.01:
                self.error_rate = 0.05
                
        elif self.read_type == ReadType.ULTRA_LONG:
            if self.read_length_mean == 150:
                self.read_length_mean = 100000
            if self.error_rate == 0.01:
                self.error_rate = 0.05
                
        elif self.read_type == ReadType.HIC:
            if self.read_length_mean == 150:
                self.read_length_mean = 150
            if self.error_rate == 0.01:
                self.error_rate = 0.001
                
        elif self.read_type == ReadType.ANCIENT_DNA:
            if self.read_length_mean == 150:
                self.read_length_mean = 50
            if self.error_rate == 0.01:
                self.error_rate = 0.02
        
        # Validation
        if not 0.1 <= self.coverage <= 1000.0:
            raise ValueError(f"Coverage must be between 0.1x and 1000x, got {self.coverage}")
        if not 10 <= self.read_length_mean <= 1_000_000:
            raise ValueError(f"Read length must be between 10 bp and 1 Mb, got {self.read_length_mean}")
        if not 0.0 <= self.error_rate <= 0.5:
            raise ValueError(f"Error rate must be between 0 and 0.5, got {self.error_rate}")


@dataclass
class UserTrainingConfig:
    """
    Complete user training configuration.
    
    Attributes:
        genome_config: Genome generation parameters
        read_configs: List of read generation configs (one per technology)
        output_dir: Output directory for training data
        num_workers: Number of parallel workers for data generation
        generate_labels: Whether to generate ground-truth labels
        shard_size: Number of examples per shard file
        compress_output: Whether to compress output files
    """
    genome_config: UserGenomeConfig
    read_configs: List[UserReadConfig]
    output_dir: Path
    
    # Processing parameters
    num_workers: int = 4
    generate_labels: bool = True
    shard_size: int = 10000
    compress_output: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.read_configs:
            raise ValueError("At least one read configuration must be provided")
        if not 1 <= self.num_workers <= 64:
            raise ValueError(f"Number of workers must be between 1 and 64, got {self.num_workers}")
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'genome_config': {
                'genome_size': self.genome_config.genome_size,
                'num_genomes': self.genome_config.num_genomes,
                'gc_content': self.genome_config.gc_content,
                'repeat_density': self.genome_config.repeat_density,
                'ploidy': self.genome_config.ploidy.name,
                'snp_rate': self.genome_config.snp_rate,
                'indel_rate': self.genome_config.indel_rate,
                'sv_density': self.genome_config.sv_density,
                'sv_types': self.genome_config.sv_types,
                'centromere_count': self.genome_config.centromere_count,
                'gene_dense_fraction': self.genome_config.gene_dense_fraction,
                'random_seed': self.genome_config.random_seed
            },
            'read_configs': [
                {
                    'read_type': rc.read_type.value,
                    'coverage': rc.coverage,
                    'read_length_mean': rc.read_length_mean,
                    'read_length_std': rc.read_length_std,
                    'error_rate': rc.error_rate,
                    'insert_size_mean': rc.insert_size_mean,
                    'insert_size_std': rc.insert_size_std
                }
                for rc in self.read_configs
            ],
            'output_dir': str(self.output_dir),
            'num_workers': self.num_workers,
            'generate_labels': self.generate_labels,
            'shard_size': self.shard_size,
            'compress_output': self.compress_output
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary."""
        genome_config = UserGenomeConfig(
            genome_size=config_dict['genome_config']['genome_size'],
            num_genomes=config_dict['genome_config']['num_genomes'],
            gc_content=config_dict['genome_config']['gc_content'],
            repeat_density=config_dict['genome_config']['repeat_density'],
            ploidy=Ploidy[config_dict['genome_config']['ploidy']],
            snp_rate=config_dict['genome_config']['snp_rate'],
            indel_rate=config_dict['genome_config']['indel_rate'],
            sv_density=config_dict['genome_config']['sv_density'],
            sv_types=config_dict['genome_config']['sv_types'],
            centromere_count=config_dict['genome_config']['centromere_count'],
            gene_dense_fraction=config_dict['genome_config']['gene_dense_fraction'],
            random_seed=config_dict['genome_config'].get('random_seed')
        )
        
        read_configs = [
            UserReadConfig(
                read_type=ReadType(rc['read_type']),
                coverage=rc['coverage'],
                read_length_mean=rc['read_length_mean'],
                read_length_std=rc['read_length_std'],
                error_rate=rc['error_rate'],
                insert_size_mean=rc.get('insert_size_mean'),
                insert_size_std=rc.get('insert_size_std')
            )
            for rc in config_dict['read_configs']
        ]
        
        return cls(
            genome_config=genome_config,
            read_configs=read_configs,
            output_dir=Path(config_dict['output_dir']),
            num_workers=config_dict.get('num_workers', 4),
            generate_labels=config_dict.get('generate_labels', True),
            shard_size=config_dict.get('shard_size', 10000),
            compress_output=config_dict.get('compress_output', True)
        )
