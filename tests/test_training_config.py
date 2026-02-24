#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for training configuration.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import copy
from pathlib import Path

import pytest

from strandweaver.user_training.training_config import (
    GraphTrainingConfig,
    Ploidy,
    ReadType,
    UserGenomeConfig,
    UserReadConfig,
    UserTrainingConfig,
    _READ_TYPE_DEFAULTS,
)


# ═══════════════════════════════════════════════════════════════════════
#  ReadType / Ploidy enums
# ═══════════════════════════════════════════════════════════════════════

class TestEnums:
    """Enum completeness and value checks."""

    def test_read_type_values(self):
        expected = {"illumina", "hifi", "ont", "ultra_long", "hic", "ancient_dna"}
        assert {rt.value for rt in ReadType} == expected

    def test_ploidy_values(self):
        assert Ploidy.HAPLOID.value == 1
        assert Ploidy.DIPLOID.value == 2
        assert Ploidy.TRIPLOID.value == 3
        assert Ploidy.TETRAPLOID.value == 4

    def test_read_type_lookup(self):
        assert ReadType("hifi") is ReadType.HIFI

    def test_ploidy_by_name(self):
        assert Ploidy["DIPLOID"] is Ploidy.DIPLOID

    def test_all_read_types_have_defaults(self):
        """Every ReadType must have an entry in _READ_TYPE_DEFAULTS."""
        for rt in ReadType:
            assert rt in _READ_TYPE_DEFAULTS, f"Missing defaults for {rt}"


# ═══════════════════════════════════════════════════════════════════════
#  UserGenomeConfig
# ═══════════════════════════════════════════════════════════════════════

class TestUserGenomeConfig:
    """Validation boundaries for genome generation parameters."""

    def test_defaults_are_valid(self):
        cfg = UserGenomeConfig()
        assert cfg.genome_size == 1_000_000
        assert cfg.num_genomes == 10
        assert cfg.gc_content == 0.42
        assert cfg.repeat_density == 0.30
        assert cfg.ploidy is Ploidy.DIPLOID
        assert cfg.random_seed is None

    def test_genome_size_lower_bound(self):
        UserGenomeConfig(genome_size=100)  # should pass
        with pytest.raises(ValueError, match="Genome size"):
            UserGenomeConfig(genome_size=99)

    def test_genome_size_upper_bound(self):
        UserGenomeConfig(genome_size=1_000_000_000)  # should pass
        with pytest.raises(ValueError, match="Genome size"):
            UserGenomeConfig(genome_size=1_000_000_001)

    def test_num_genomes_bounds(self):
        UserGenomeConfig(num_genomes=1)
        UserGenomeConfig(num_genomes=10_000)
        with pytest.raises(ValueError, match="Number of genomes"):
            UserGenomeConfig(num_genomes=0)
        with pytest.raises(ValueError, match="Number of genomes"):
            UserGenomeConfig(num_genomes=10_001)

    def test_gc_content_bounds(self):
        UserGenomeConfig(gc_content=0.0)
        UserGenomeConfig(gc_content=1.0)
        with pytest.raises(ValueError, match="GC content"):
            UserGenomeConfig(gc_content=-0.01)
        with pytest.raises(ValueError, match="GC content"):
            UserGenomeConfig(gc_content=1.01)

    def test_repeat_density_bounds(self):
        UserGenomeConfig(repeat_density=0.0)
        UserGenomeConfig(repeat_density=1.0)
        with pytest.raises(ValueError, match="Repeat density"):
            UserGenomeConfig(repeat_density=-0.1)
        with pytest.raises(ValueError, match="Repeat density"):
            UserGenomeConfig(repeat_density=1.1)

    def test_snp_rate_bounds(self):
        UserGenomeConfig(snp_rate=0.0)
        UserGenomeConfig(snp_rate=0.1)
        with pytest.raises(ValueError, match="SNP rate"):
            UserGenomeConfig(snp_rate=-0.001)
        with pytest.raises(ValueError, match="SNP rate"):
            UserGenomeConfig(snp_rate=0.11)

    def test_indel_rate_bounds(self):
        UserGenomeConfig(indel_rate=0.0)
        UserGenomeConfig(indel_rate=0.01)
        with pytest.raises(ValueError, match="Indel rate"):
            UserGenomeConfig(indel_rate=-0.0001)
        with pytest.raises(ValueError, match="Indel rate"):
            UserGenomeConfig(indel_rate=0.011)

    def test_sv_density_bounds(self):
        UserGenomeConfig(sv_density=0.0)
        UserGenomeConfig(sv_density=0.001)
        with pytest.raises(ValueError, match="SV density"):
            UserGenomeConfig(sv_density=-0.00001)
        with pytest.raises(ValueError, match="SV density"):
            UserGenomeConfig(sv_density=0.0011)

    def test_sv_types_default(self):
        cfg = UserGenomeConfig()
        assert set(cfg.sv_types) == {"deletion", "insertion", "inversion", "duplication"}

    def test_sv_types_independent_per_instance(self):
        """Default mutable fields must not be shared between instances."""
        a = UserGenomeConfig()
        b = UserGenomeConfig()
        a.sv_types.append("translocation")
        assert "translocation" not in b.sv_types


# ═══════════════════════════════════════════════════════════════════════
#  UserReadConfig
# ═══════════════════════════════════════════════════════════════════════

class TestUserReadConfig:
    """Auto-fill from tech defaults and validation."""

    def test_hifi_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.HIFI)
        assert cfg.read_length_mean == 15_000
        assert cfg.read_length_std == 5_000
        assert cfg.error_rate == 0.001
        assert cfg.insert_size_mean is None
        assert cfg.insert_size_std is None

    def test_illumina_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.ILLUMINA)
        assert cfg.read_length_mean == 150
        assert cfg.insert_size_mean == 350

    def test_ont_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.ONT)
        assert cfg.read_length_mean == 20_000
        assert cfg.error_rate == 0.05

    def test_hic_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.HIC)
        assert cfg.read_length_mean == 150
        assert cfg.insert_size_mean == 500

    def test_ultra_long_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.ULTRA_LONG)
        assert cfg.read_length_mean == 100_000
        assert cfg.error_rate == 0.05

    def test_ancient_dna_defaults(self):
        cfg = UserReadConfig(read_type=ReadType.ANCIENT_DNA)
        assert cfg.read_length_mean == 50
        assert cfg.error_rate == 0.02

    def test_user_override_preserved(self):
        """User-supplied values should NOT be overwritten by defaults."""
        cfg = UserReadConfig(read_type=ReadType.HIFI, read_length_mean=20_000)
        assert cfg.read_length_mean == 20_000  # kept user value

    def test_coverage_bounds(self):
        UserReadConfig(read_type=ReadType.HIFI, coverage=0.1)
        UserReadConfig(read_type=ReadType.HIFI, coverage=1000.0)
        with pytest.raises(ValueError, match="Coverage"):
            UserReadConfig(read_type=ReadType.HIFI, coverage=0.09)
        with pytest.raises(ValueError, match="Coverage"):
            UserReadConfig(read_type=ReadType.HIFI, coverage=1001.0)

    def test_read_length_bounds(self):
        UserReadConfig(read_type=ReadType.HIFI, read_length_mean=10)
        UserReadConfig(read_type=ReadType.HIFI, read_length_mean=1_000_000)
        with pytest.raises(ValueError, match="Read length"):
            UserReadConfig(read_type=ReadType.HIFI, read_length_mean=9)
        with pytest.raises(ValueError, match="Read length"):
            UserReadConfig(read_type=ReadType.HIFI, read_length_mean=1_000_001)

    def test_error_rate_bounds(self):
        UserReadConfig(read_type=ReadType.HIFI, error_rate=0.0)
        UserReadConfig(read_type=ReadType.HIFI, error_rate=0.5)
        with pytest.raises(ValueError, match="Error rate"):
            UserReadConfig(read_type=ReadType.HIFI, error_rate=-0.01)
        with pytest.raises(ValueError, match="Error rate"):
            UserReadConfig(read_type=ReadType.HIFI, error_rate=0.51)


# ═══════════════════════════════════════════════════════════════════════
#  GraphTrainingConfig
# ═══════════════════════════════════════════════════════════════════════

class TestGraphTrainingConfig:
    """Graph config defaults and validation."""

    def test_defaults(self):
        cfg = GraphTrainingConfig()
        assert cfg.enabled is False
        assert cfg.min_overlap_bp == 500
        assert cfg.min_overlap_identity == 0.90
        assert cfg.label_edges is True
        assert cfg.export_gfa is True

    def test_min_overlap_bp_bounds(self):
        GraphTrainingConfig(min_overlap_bp=50)
        GraphTrainingConfig(min_overlap_bp=100_000)
        with pytest.raises(ValueError, match="min_overlap_bp"):
            GraphTrainingConfig(min_overlap_bp=49)
        with pytest.raises(ValueError, match="min_overlap_bp"):
            GraphTrainingConfig(min_overlap_bp=100_001)

    def test_min_overlap_identity_bounds(self):
        GraphTrainingConfig(min_overlap_identity=0.5)
        GraphTrainingConfig(min_overlap_identity=1.0)
        with pytest.raises(ValueError, match="min_overlap_identity"):
            GraphTrainingConfig(min_overlap_identity=0.49)
        with pytest.raises(ValueError, match="min_overlap_identity"):
            GraphTrainingConfig(min_overlap_identity=1.01)

    def test_noise_edge_fraction_bounds(self):
        GraphTrainingConfig(noise_edge_fraction=0.0)
        GraphTrainingConfig(noise_edge_fraction=1.0)
        with pytest.raises(ValueError, match="noise_edge_fraction"):
            GraphTrainingConfig(noise_edge_fraction=-0.01)
        with pytest.raises(ValueError, match="noise_edge_fraction"):
            GraphTrainingConfig(noise_edge_fraction=1.01)


# ═══════════════════════════════════════════════════════════════════════
#  UserTrainingConfig (top-level orchestrator)
# ═══════════════════════════════════════════════════════════════════════

def _make_minimal_config(**overrides):
    """Helper: build a valid UserTrainingConfig with sensible minimums."""
    defaults = dict(
        genome_config=UserGenomeConfig(),
        read_configs=[UserReadConfig(read_type=ReadType.HIFI)],
        output_dir=Path("/tmp/sw_test"),
    )
    defaults.update(overrides)
    return UserTrainingConfig(**defaults)


class TestUserTrainingConfig:
    """Top-level config validation and serialisation."""

    def test_defaults(self):
        cfg = _make_minimal_config()
        assert cfg.num_workers == 4
        assert cfg.generate_labels is True
        assert cfg.shard_size == 10_000
        assert cfg.compress_output is True
        assert cfg.graph_only is False

    def test_empty_read_configs_rejected(self):
        with pytest.raises(ValueError, match="At least one read"):
            UserTrainingConfig(
                genome_config=UserGenomeConfig(),
                read_configs=[],
                output_dir=Path("/tmp"),
            )

    def test_num_workers_bounds(self):
        _make_minimal_config(num_workers=1)
        _make_minimal_config(num_workers=64)
        with pytest.raises(ValueError, match="Number of workers"):
            _make_minimal_config(num_workers=0)
        with pytest.raises(ValueError, match="Number of workers"):
            _make_minimal_config(num_workers=65)

    def test_output_dir_coerced_to_path(self):
        cfg = _make_minimal_config(output_dir="/tmp/test_str")
        assert isinstance(cfg.output_dir, Path)

    def test_graph_config_optional(self):
        cfg = _make_minimal_config()
        assert cfg.graph_config is None

    def test_graph_config_attached(self):
        gc = GraphTrainingConfig(enabled=True, min_overlap_bp=300)
        cfg = _make_minimal_config(graph_config=gc)
        assert cfg.graph_config is not None
        assert cfg.graph_config.min_overlap_bp == 300

    # ── Serialisation roundtrip ────────────────────────────────────────

    def test_to_dict_basic(self):
        cfg = _make_minimal_config()
        d = cfg.to_dict()
        assert d["genome_config"]["genome_size"] == 1_000_000
        assert d["genome_config"]["ploidy"] == "DIPLOID"
        assert len(d["read_configs"]) == 1
        assert d["read_configs"][0]["read_type"] == "hifi"
        assert d["output_dir"] == "/tmp/sw_test"
        assert d["graph_config"] is None

    def test_to_dict_with_graph_config(self):
        gc = GraphTrainingConfig(enabled=True)
        cfg = _make_minimal_config(graph_config=gc)
        d = cfg.to_dict()
        assert d["graph_config"]["enabled"] is True
        assert d["graph_config"]["min_overlap_bp"] == 500

    def test_from_dict_roundtrip(self):
        gc = GraphTrainingConfig(enabled=True, min_overlap_bp=800)
        original = _make_minimal_config(
            graph_config=gc,
            num_workers=8,
            graph_only=False,
        )
        d = original.to_dict()
        restored = UserTrainingConfig.from_dict(d)

        assert restored.genome_config.genome_size == original.genome_config.genome_size
        assert restored.genome_config.ploidy == original.genome_config.ploidy
        assert restored.genome_config.gc_content == original.genome_config.gc_content
        assert len(restored.read_configs) == 1
        assert restored.read_configs[0].read_type == ReadType.HIFI
        assert restored.graph_config is not None
        assert restored.graph_config.min_overlap_bp == 800
        assert restored.num_workers == 8
        assert restored.output_dir == original.output_dir

    def test_from_dict_multiple_read_configs(self):
        cfg = _make_minimal_config(
            read_configs=[
                UserReadConfig(read_type=ReadType.HIFI, coverage=30.0),
                UserReadConfig(read_type=ReadType.ILLUMINA, coverage=50.0),
                UserReadConfig(read_type=ReadType.HIC, coverage=10.0),
            ],
        )
        d = cfg.to_dict()
        restored = UserTrainingConfig.from_dict(d)
        assert len(restored.read_configs) == 3
        types = [rc.read_type for rc in restored.read_configs]
        assert types == [ReadType.HIFI, ReadType.ILLUMINA, ReadType.HIC]

    def test_from_dict_no_graph_config(self):
        cfg = _make_minimal_config()
        d = cfg.to_dict()
        restored = UserTrainingConfig.from_dict(d)
        assert restored.graph_config is None

    def test_roundtrip_preserves_all_genome_fields(self):
        gc = UserGenomeConfig(
            genome_size=5_000_000,
            num_genomes=50,
            gc_content=0.38,
            repeat_density=0.50,
            ploidy=Ploidy.TETRAPLOID,
            snp_rate=0.005,
            indel_rate=0.0005,
            sv_density=0.00005,
            sv_types=["deletion", "inversion"],
            centromere_count=3,
            gene_dense_fraction=0.40,
            random_seed=12345,
        )
        cfg = _make_minimal_config(genome_config=gc)
        restored = UserTrainingConfig.from_dict(cfg.to_dict())
        rg = restored.genome_config
        assert rg.genome_size == 5_000_000
        assert rg.num_genomes == 50
        assert rg.ploidy is Ploidy.TETRAPLOID
        assert rg.snp_rate == 0.005
        assert rg.sv_types == ["deletion", "inversion"]
        assert rg.centromere_count == 3
        assert rg.random_seed == 12345

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
