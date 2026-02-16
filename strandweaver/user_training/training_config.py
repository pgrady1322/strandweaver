#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

User Training Configuration — dataclasses for training data generation parameters.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
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


    # Technology-specific default parameters
    _TECH_DEFAULTS: dict = field(default=None, init=False, repr=False)  # type: ignore[assignment]

# Lookup table: ReadType → (read_length_mean, read_length_std, error_rate,
#                            insert_size_mean, insert_size_std)
_READ_TYPE_DEFAULTS = {
    ReadType.ILLUMINA:   (150,     10,    0.001, 350,  50),
    ReadType.HIFI:       (15_000,  5_000, 0.001, None, None),
    ReadType.ONT:        (20_000,  8_000, 0.05,  None, None),
    ReadType.ULTRA_LONG: (100_000, 30_000,0.05,  None, None),
    ReadType.HIC:        (150,     10,    0.001, 500,  100),
    ReadType.ANCIENT_DNA:(50,      15,    0.02,  None, None),
}


@dataclass
class UserReadConfig:
    """
    User-specified read generation parameters for a single technology.

    Fields left as ``None`` are filled automatically with sensible
    technology-specific defaults (see ``_READ_TYPE_DEFAULTS``).

    Attributes:
        read_type: Type of sequencing reads
        coverage: Sequencing coverage depth (e.g., 30 for 30x)
        read_length_mean: Mean read length in bp (None → tech default)
        read_length_std: Standard deviation of read length (None → tech default)
        error_rate: Base error rate (None → tech default)
        insert_size_mean: Insert size for paired-end (None → tech default)
        insert_size_std: Insert size standard deviation (None → tech default)
    """
    read_type: ReadType
    coverage: float = 30.0

    # Read characteristics — None means "use technology default"
    read_length_mean: Optional[int] = None
    read_length_std: Optional[int] = None
    error_rate: Optional[float] = None

    # Paired-end parameters (Illumina, Hi-C)
    insert_size_mean: Optional[int] = None
    insert_size_std: Optional[int] = None

    def __post_init__(self):
        """Fill None fields with technology defaults, then validate."""
        defaults = _READ_TYPE_DEFAULTS.get(self.read_type)
        if defaults is None:
            raise ValueError(f"Unsupported read type: {self.read_type}")

        d_len, d_std, d_err, d_ins_mean, d_ins_std = defaults

        if self.read_length_mean is None:
            self.read_length_mean = d_len
        if self.read_length_std is None:
            self.read_length_std = d_std
        if self.error_rate is None:
            self.error_rate = d_err
        if self.insert_size_mean is None:
            self.insert_size_mean = d_ins_mean
        if self.insert_size_std is None:
            self.insert_size_std = d_ins_std

        # Validation
        if not 0.1 <= self.coverage <= 1000.0:
            raise ValueError(f"Coverage must be between 0.1x and 1000x, got {self.coverage}")
        if not 10 <= self.read_length_mean <= 1_000_000:
            raise ValueError(f"Read length must be between 10 bp and 1 Mb, got {self.read_length_mean}")
        if not 0.0 <= self.error_rate <= 0.5:
            raise ValueError(f"Error rate must be between 0 and 0.5, got {self.error_rate}")


@dataclass
class GraphTrainingConfig:
    """
    Configuration for synthetic assembly-graph training data generation.

    Controls how overlap graphs are constructed from simulated reads and
    which ground-truth label sets to produce.

    Attributes:
        enabled: Whether to generate graph training data at all.
        min_overlap_bp: Minimum overlap length (bp) between reads to create
            an edge in the graph.
        min_overlap_identity: Minimum sequence identity for an overlap to
            be accepted (0.0–1.0).
        max_overhang_fraction: Maximum fraction of a read that may extend
            beyond the overlap without being penalised (filters chimeric
            overlaps).
        add_noise_edges: Inject a fraction of false edges to train the
            classifier on negatives.  The value is a fraction of the total
            true edge count (e.g. 0.10 = 10 % extra noise edges).
        noise_edge_fraction: (see above)
        label_edges: Produce edge-level labels (TRUE / ALLELIC / REPEAT /
            SV_BREAK / CHIMERIC) for EdgeAI training.
        label_nodes: Produce node-level haplotype labels (HAP_A / HAP_B /
            BOTH / REPEAT) for DiploidAI training.
        label_paths: Produce correct-path labels for PathGNN training.
        label_ul_routes: Produce UL read routing labels.
        label_svs: Produce SV graph-signature labels.
        compute_features: Compute the full ML feature vectors alongside
            the labels (required for direct model training).
        export_gfa: Write the synthetic graph in GFA format for
            visualisation / debugging.
        max_coverage_for_graph: Subsample reads to this coverage before
            overlap detection (keeps RAM manageable for large genomes).
            ``None`` uses all reads.
    """
    enabled: bool = False
    min_overlap_bp: int = 500
    min_overlap_identity: float = 0.90
    max_overhang_fraction: float = 0.30
    add_noise_edges: bool = True
    noise_edge_fraction: float = 0.10
    label_edges: bool = True
    label_nodes: bool = True
    label_paths: bool = True
    label_ul_routes: bool = True
    label_svs: bool = True
    compute_features: bool = True
    export_gfa: bool = True
    max_coverage_for_graph: Optional[float] = None

    def __post_init__(self):
        if not 50 <= self.min_overlap_bp <= 100_000:
            raise ValueError(
                f"min_overlap_bp must be 50–100 000, got {self.min_overlap_bp}")
        if not 0.5 <= self.min_overlap_identity <= 1.0:
            raise ValueError(
                f"min_overlap_identity must be 0.5–1.0, got {self.min_overlap_identity}")
        if not 0.0 <= self.noise_edge_fraction <= 1.0:
            raise ValueError(
                f"noise_edge_fraction must be 0–1, got {self.noise_edge_fraction}")


@dataclass
class UserTrainingConfig:
    """
    Complete user training configuration.
    
    Attributes:
        genome_config: Genome generation parameters
        read_configs: List of read generation configs (one per technology)
        output_dir: Output directory for training data
        graph_config: Optional graph training data generation settings
        num_workers: Number of parallel workers for data generation
        generate_labels: Whether to generate ground-truth labels
        shard_size: Number of examples per shard file
        compress_output: Whether to compress output files
    """
    genome_config: UserGenomeConfig
    read_configs: List[UserReadConfig]
    output_dir: Path
    graph_config: Optional[GraphTrainingConfig] = None
    
    # ── Fast graph-only mode ─────────────────────────────────────
    # When True, reads are simulated in memory for realistic coverage
    # and overlap features, but FASTQ/FASTA files are NOT written to
    # disk.  Only the graph training CSVs are produced.  This is
    # ~10× faster and uses ~95% less disk.
    graph_only: bool = False
    
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
            'graph_only': self.graph_only,
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
            'graph_config': {
                'enabled': self.graph_config.enabled,
                'min_overlap_bp': self.graph_config.min_overlap_bp,
                'min_overlap_identity': self.graph_config.min_overlap_identity,
                'max_overhang_fraction': self.graph_config.max_overhang_fraction,
                'add_noise_edges': self.graph_config.add_noise_edges,
                'noise_edge_fraction': self.graph_config.noise_edge_fraction,
                'label_edges': self.graph_config.label_edges,
                'label_nodes': self.graph_config.label_nodes,
                'label_paths': self.graph_config.label_paths,
                'label_ul_routes': self.graph_config.label_ul_routes,
                'label_svs': self.graph_config.label_svs,
                'compute_features': self.graph_config.compute_features,
                'export_gfa': self.graph_config.export_gfa,
                'max_coverage_for_graph': self.graph_config.max_coverage_for_graph,
            } if self.graph_config else None,
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
        
        gc_raw = config_dict.get('graph_config')
        graph_config = GraphTrainingConfig(**gc_raw) if gc_raw else None

        return cls(
            genome_config=genome_config,
            read_configs=read_configs,
            output_dir=Path(config_dict['output_dir']),
            graph_config=graph_config,
            num_workers=config_dict.get('num_workers', 4),
            generate_labels=config_dict.get('generate_labels', True),
            shard_size=config_dict.get('shard_size', 10000),
            compress_output=config_dict.get('compress_output', True)
        )

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
