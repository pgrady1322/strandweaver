#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for CLI command interface.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from click.testing import CliRunner
from strandweaver.cli import main


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_cli_help(self):
        """Test that --help runs without error."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'StrandWeaver' in result.output
    
    def test_cli_version(self):
        """Test that --version displays version."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert 'version' in result.output.lower() or '0.1' in result.output
    
    def test_config_init_command(self):
        """Test config init command."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['config', 'init', '--output', 'test_config.yaml'])
            
            assert result.exit_code == 0
    
    def test_core_assemble_help(self):
        """Test core-assemble command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['core-assemble', '--help'])
        
        assert result.exit_code == 0
        assert 'assemble' in result.output.lower()
    
    def test_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        runner = CliRunner()
        result = runner.invoke(main, ['nonexistent_command'])
        
        # Should fail but not crash
        assert result.exit_code != 0


class TestCoreAssembleCLI:
    """Test core-assemble command specifically."""
    
    def test_core_assemble_missing_input(self):
        """Test core-assemble fails gracefully without input."""
        runner = CliRunner()
        result = runner.invoke(main, ['core-assemble', '--output', 'test_out'])
        
        # Should fail due to missing input reads
        assert result.exit_code != 0
    
    def test_core_assemble_with_nonexistent_file(self):
        """Test core-assemble handles nonexistent input files."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'core-assemble',
            '--hifi-long-reads', 'nonexistent.fastq',
            '--output', 'test_out'
        ])
        
        # Should fail but not crash
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Helper: write a small FASTA file
# ---------------------------------------------------------------------------
import os

TINY_ASSEMBLY = ">contig1\nACGTACGTACGTACGTACGT\n>contig2\nTTTTAAAAAACCCCGGGG\n"
TINY_READS = ">read1\nACGTACGTACGT\n>read2\nTTTTAAAAAACCCC\n>read3\nGGGGCCCCTTTT\n"
GAPPED_ASSEMBLY = ">ctg1\nACGTACGTACGT" + "N" * 50 + "TTTTAAAAAACC\n"


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# QV command tests
# ---------------------------------------------------------------------------
class TestQVCommand:
    """Tests for the qv CLI command."""

    def test_qv_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['qv', '--help'])
        assert result.exit_code == 0
        assert 'quality value' in result.output.lower() or 'QV' in result.output

    def test_qv_missing_assembly(self):
        """qv should fail when --assembly is missing."""
        runner = CliRunner()
        result = runner.invoke(main, ['qv'])
        assert result.exit_code != 0

    def test_qv_nonexistent_assembly(self):
        runner = CliRunner()
        result = runner.invoke(main, ['qv', '-a', 'missing.fa'])
        assert result.exit_code != 0

    def test_qv_assembly_only(self):
        """QV with assembly only (no reads) should succeed."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            result = runner.invoke(main, ['qv', '-a', 'asm.fa'])
            assert result.exit_code == 0
            assert 'QV' in result.output or 'qv' in result.output.lower()

    def test_qv_with_reads(self):
        """QV with reads and output file."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'qv', '-a', 'asm.fa', '-r', 'reads.fa', '-o', 'qv.json',
            ])
            assert result.exit_code == 0
            assert os.path.exists('qv.json')

    def test_qv_custom_k(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            result = runner.invoke(main, ['qv', '-a', 'asm.fa', '-k', '15'])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Polish command tests
# ---------------------------------------------------------------------------
class TestPolishCommand:
    """Tests for the polish CLI command."""

    def test_polish_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['polish', '--help'])
        assert result.exit_code == 0
        assert 'polish' in result.output.lower()

    def test_polish_missing_args(self):
        runner = CliRunner()
        result = runner.invoke(main, ['polish'])
        assert result.exit_code != 0

    def test_polish_nonexistent_reads(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            result = runner.invoke(main, [
                'polish', '-a', 'asm.fa', '-r', 'nope.fq', '-o', 'out.fa',
            ])
            assert result.exit_code != 0

    def test_polish_basic(self):
        """Run polish on tiny data end-to-end."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'polish', '-a', 'asm.fa', '-r', 'reads.fa', '-o', 'pol.fa',
            ])
            assert result.exit_code == 0
            assert os.path.exists('pol.fa')
            assert 'complete' in result.output.lower()

    def test_polish_custom_options(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('asm.fa', TINY_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'polish', '-a', 'asm.fa', '-r', 'reads.fa', '-o', 'pol.fa',
                '--rounds', '1', '--k', '15', '--min-coverage', '2',
            ])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Gap-fill command tests
# ---------------------------------------------------------------------------
class TestGapFillCommand:
    """Tests for the gap-fill CLI command."""

    def test_gap_fill_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['gap-fill', '--help'])
        assert result.exit_code == 0
        assert 'gap' in result.output.lower()

    def test_gap_fill_missing_args(self):
        runner = CliRunner()
        result = runner.invoke(main, ['gap-fill'])
        assert result.exit_code != 0

    def test_gap_fill_basic(self):
        """Run gap-fill on a small gapped assembly."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('gapped.fa', GAPPED_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'gap-fill', '-a', 'gapped.fa', '-r', 'reads.fa', '-o', 'filled.fa',
            ])
            assert result.exit_code == 0
            assert os.path.exists('filled.fa')
            assert 'complete' in result.output.lower()

    def test_gap_fill_custom_options(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('gapped.fa', GAPPED_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'gap-fill', '-a', 'gapped.fa', '-r', 'reads.fa', '-o', 'f.fa',
                '--max-gap-size', '5000', '--k', '11', '--min-spanning', '2',
            ])
            assert result.exit_code == 0

    def test_gap_fill_no_gaps(self):
        """Assembly without gaps should still succeed."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('clean.fa', TINY_ASSEMBLY)
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'gap-fill', '-a', 'clean.fa', '-r', 'reads.fa', '-o', 'out.fa',
            ])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# nf-merge removal and subsample flags
# ---------------------------------------------------------------------------
class TestNfMergeRemoved:
    """Verify nf-merge command no longer exists."""

    def test_nf_merge_removed(self):
        runner = CliRunner()
        result = runner.invoke(main, ['nf-merge', '--help'])
        assert result.exit_code != 0


class TestSubsampleFlags:
    """Tests for --subsample-* flags on the pipeline command."""

    def test_subsample_flags_in_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['pipeline', '--help'])
        assert result.exit_code == 0
        assert '--subsample-hifi' in result.output
        assert '--subsample-ont' in result.output
        assert '--subsample-ont-ul' in result.output
        assert '--subsample-illumina' in result.output
        assert '--subsample-ancient' in result.output

    def test_subsample_invalid_fraction(self):
        """Subsample > 1.0 should fail."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--hifi-long-reads', 'reads.fa',
                '-o', 'out',
                '--subsample-hifi', '1.5',
                '--dry-run',
            ])
            assert result.exit_code != 0
            assert 'must be between' in result.output or result.exit_code != 0

    def test_subsample_valid_dryrun(self):
        """Valid subsample fraction with --dry-run should succeed."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--hifi-long-reads', 'reads.fa',
                '-o', 'out',
                '--subsample-hifi', '0.5',
                '--dry-run',
            ])
            assert result.exit_code == 0
            assert 'DRY RUN' in result.output

    def test_subsample_zero_invalid(self):
        """Subsample 0.0 should fail (no reads would remain)."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-long-reads', 'reads.fa',
                '-o', 'out',
                '--subsample-ont', '0.0',
                '--dry-run',
            ])
            assert result.exit_code != 0


class TestChemistryFlags:
    """Tests for ErrorSmith chemistry designation flags."""

    def test_chemistry_flags_in_pipeline_help(self):
        """Chemistry flags should appear in pipeline --help."""
        runner = CliRunner()
        result = runner.invoke(main, ['pipeline', '--help'])
        assert result.exit_code == 0
        assert '--hifi-chemistry' in result.output
        assert '--ont-chemistry' in result.output
        assert '--ont-ul-chemistry' in result.output
        assert '--illumina-chemistry' in result.output

    def test_chemistry_flags_in_correct_help(self):
        """Chemistry flags should appear in correct --help."""
        runner = CliRunner()
        result = runner.invoke(main, ['correct', '--help'])
        assert result.exit_code == 0
        assert '--hifi-chemistry' in result.output
        assert '--ont-chemistry' in result.output
        assert '--ont-ul-chemistry' in result.output
        assert '--illumina-chemistry' in result.output

    def test_invalid_ont_chemistry(self):
        """Invalid ONT chemistry should be rejected."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-long-reads', 'reads.fa',
                '-o', 'out',
                '--ont-chemistry', 'ont_fake_chemistry',
                '--dry-run',
            ])
            assert result.exit_code != 0

    def test_valid_ont_chemistry_dryrun(self):
        """Valid ONT chemistry with --dry-run should succeed."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-long-reads', 'reads.fa',
                '-o', 'out',
                '--ont-chemistry', 'ont_lsk114_r1041',
                '--dry-run',
            ])
            assert result.exit_code == 0
            assert 'ont_lsk114_r1041' in result.output

    def test_valid_hifi_chemistry_dryrun(self):
        """Valid HiFi chemistry with --dry-run should succeed."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--hifi-long-reads', 'reads.fa',
                '-o', 'out',
                '--hifi-chemistry', 'pacbio_hifi_sequel2',
                '--dry-run',
            ])
            assert result.exit_code == 0
            assert 'pacbio_hifi_sequel2' in result.output

    def test_ont_ul_chemistry_separate_flag(self):
        """ONT-UL chemistry flag should be separate from ONT chemistry."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-ul', 'reads.fa',
                '--ont-ul-chemistry', 'ont_ulk114_r1041',
                '-o', 'out',
                '--dry-run',
            ])
            assert result.exit_code == 0
            assert 'ONT-UL=ont_ulk114_r1041' in result.output

    def test_ont_and_ul_separate_chemistries(self):
        """ONT and ONT-UL can have different chemistries in same run."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('ont.fa', TINY_READS)
            _write('ul.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-long-reads', 'ont.fa',
                '--ont-ul', 'ul.fa',
                '--ont-chemistry', 'ont_lsk114_r1041',
                '--ont-ul-chemistry', 'ont_ulk114_r1041',
                '-o', 'out',
                '--dry-run',
            ])
            assert result.exit_code == 0
            assert 'ONT=ont_lsk114_r1041' in result.output
            assert 'ONT-UL=ont_ulk114_r1041' in result.output

    def test_ligation_kit_rejected_for_ul(self):
        """Ligation kit chemistry should be rejected for --ont-ul-chemistry."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write('reads.fa', TINY_READS)
            result = runner.invoke(main, [
                'pipeline',
                '--ont-ul', 'reads.fa',
                '--ont-ul-chemistry', 'ont_lsk114_r1041',
                '-o', 'out',
                '--dry-run',
            ])
            assert result.exit_code != 0


class TestChemistryModule:
    """Tests for ErrorSmith chemistry resolution in errorsmith_module."""

    def test_chemistry_codes_defined(self):
        """CHEMISTRY_CODES should have 13 entries."""
        from strandweaver.preprocessing.errorsmith_module import CHEMISTRY_CODES
        assert len(CHEMISTRY_CODES) == 13
        assert 'pacbio_hifi_sequel2' in CHEMISTRY_CODES
        assert 'pacbio_hifi_revio' in CHEMISTRY_CODES
        assert 'ont_lsk110_r941' in CHEMISTRY_CODES
        assert 'ont_ulk001_r941' in CHEMISTRY_CODES
        assert 'ont_lsk114_r1041' in CHEMISTRY_CODES
        assert 'ont_r1041_duplex' in CHEMISTRY_CODES
        assert 'ont_ulk114_r1041' in CHEMISTRY_CODES
        assert 'ont_ulk114_r1041_hiacc' in CHEMISTRY_CODES
        assert 'ont_ulk114_r1041_dorado' in CHEMISTRY_CODES
        assert 'illumina_hiseq2500' in CHEMISTRY_CODES
        assert 'pacbio_onso' in CHEMISTRY_CODES
        assert 'element_aviti' in CHEMISTRY_CODES
        assert 'element_ultraq' in CHEMISTRY_CODES

    def test_chemistry_names_reverse_map(self):
        """CHEMISTRY_NAMES should be correct reverse map."""
        from strandweaver.preprocessing.errorsmith_module import (
            CHEMISTRY_CODES, CHEMISTRY_NAMES
        )
        for name, code in CHEMISTRY_CODES.items():
            assert CHEMISTRY_NAMES[code] == name

    def test_resolve_chemistry_explicit(self):
        """Explicit chemistry should resolve correctly."""
        from strandweaver.preprocessing.errorsmith_module import resolve_chemistry
        name, code = resolve_chemistry('ont', chemistry='ont_lsk114_r1041')
        assert name == 'ont_lsk114_r1041'
        assert code == 3

    def test_resolve_chemistry_default(self):
        """Default chemistry should be chosen for known technology."""
        from strandweaver.preprocessing.errorsmith_module import resolve_chemistry
        name, code = resolve_chemistry('hifi')
        assert name == 'pacbio_hifi_sequel2'
        assert code == 0

    def test_resolve_chemistry_ont_default_is_ligation(self):
        """Default ONT chemistry should be a ligation kit, not ultra-long."""
        from strandweaver.preprocessing.errorsmith_module import resolve_chemistry
        name, code = resolve_chemistry('ont')
        assert name == 'ont_lsk114_r1041'
        assert code == 3

    def test_resolve_chemistry_ont_ul_default_is_ultralong(self):
        """Default ONT ultra-long chemistry should be an ultra-long kit."""
        from strandweaver.preprocessing.errorsmith_module import resolve_chemistry
        name, code = resolve_chemistry('ont_ultralong')
        assert name == 'ont_ulk001_r941'
        assert code == 2

    def test_resolve_chemistry_invalid(self):
        """Invalid chemistry should raise ValueError."""
        from strandweaver.preprocessing.errorsmith_module import resolve_chemistry
        with pytest.raises(ValueError, match='Unknown chemistry'):
            resolve_chemistry('ont', chemistry='ont_fake_v99')

    def test_get_corrector_stamps_chemistry(self):
        """get_corrector() should stamp chemistry and chemistry_code."""
        from strandweaver.preprocessing.errorsmith_module import get_corrector
        c = get_corrector('ont', chemistry='ont_lsk114_r1041')
        assert c.chemistry == 'ont_lsk114_r1041'
        assert c.chemistry_code == 3

    def test_get_corrector_default_chemistry(self):
        """get_corrector() without explicit chemistry should use default."""
        from strandweaver.preprocessing.errorsmith_module import get_corrector
        c = get_corrector('illumina')
        assert c.chemistry == 'illumina_hiseq2500'
        assert c.chemistry_code == 5

    def test_get_corrector_all_technologies(self):
        """get_corrector() should work for all 11 technology families."""
        from strandweaver.preprocessing.errorsmith_module import get_corrector
        for tech in ['ont', 'pacbio', 'illumina', 'ancient_dna',
                     'onso', 'element', 'ultraq', 'revio',
                     'duplex', 'dorado', 'ont_hiacc']:
            c = get_corrector(tech)
            assert c.chemistry is not None
            assert c.chemistry_code is not None

    # ── Chemistry binary feature decomposition tests ────────────────

    def test_chemistry_features_defined_for_all_codes(self):
        """CHEMISTRY_FEATURES should have an entry for every chemistry code."""
        from strandweaver.preprocessing.errorsmith_module import (
            CHEMISTRY_CODES, CHEMISTRY_FEATURES, CHEMISTRY_FEATURE_NAMES
        )
        for name, code in CHEMISTRY_CODES.items():
            assert code in CHEMISTRY_FEATURES, f"Missing features for {name} (code {code})"
            assert len(CHEMISTRY_FEATURES[code]) == len(CHEMISTRY_FEATURE_NAMES)

    def test_chemistry_features_are_binary(self):
        """All chemistry feature values should be 0 or 1."""
        from strandweaver.preprocessing.errorsmith_module import CHEMISTRY_FEATURES
        for code, vec in CHEMISTRY_FEATURES.items():
            for val in vec:
                assert val in (0, 1), f"Non-binary value {val} in code {code}"

    def test_get_chemistry_features_returns_dict(self):
        """get_chemistry_features() should return named dict."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_FEATURE_NAMES
        )
        feats = get_chemistry_features(0)
        assert isinstance(feats, dict)
        assert set(feats.keys()) == set(CHEMISTRY_FEATURE_NAMES)

    def test_get_chemistry_features_unknown_code(self):
        """Unknown chemistry code should return all zeros."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        feats = get_chemistry_features(999)
        assert all(v == 0 for v in feats.values())

    def test_ont_codes_share_is_ont(self):
        """All ONT chemistry codes should have is_ont=1."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_CODES
        )
        ont_keys = [k for k in CHEMISTRY_CODES if k.startswith('ont_')]
        assert len(ont_keys) >= 7  # 7 ONT codes
        for key in ont_keys:
            feats = get_chemistry_features(CHEMISTRY_CODES[key])
            assert feats['is_ont'] == 1, f"{key} should have is_ont=1"
            assert feats['is_long_read'] == 1, f"{key} should have is_long_read=1"

    def test_pacbio_hifi_codes_share_is_pacbio_hifi(self):
        """PacBio HiFi codes (Sequel II + Revio) should share is_pacbio_hifi=1."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        for code in [0, 9]:  # pacbio_hifi_sequel2, pacbio_hifi_revio
            feats = get_chemistry_features(code)
            assert feats['is_pacbio_hifi'] == 1
            assert feats['is_long_read'] == 1
            assert feats['is_ont'] == 0

    def test_onso_is_unique(self):
        """PacBio Onso should be is_pacbio_onso=1, short-read, NOT is_pacbio_hifi."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        feats = get_chemistry_features(6)  # pacbio_onso
        assert feats['is_pacbio_onso'] == 1
        assert feats['is_short_read'] == 1
        assert feats['is_pacbio_hifi'] == 0
        assert feats['is_long_read'] == 0
        assert feats['is_ont'] == 0

    def test_hiacc_flag(self):
        """is_hiacc should only be set for ont_ulk114_r1041_hiacc (code 11)."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_FEATURES
        )
        for code in CHEMISTRY_FEATURES:
            feats = get_chemistry_features(code)
            if code == 11:
                assert feats['is_hiacc'] == 1
                assert feats['is_ont'] == 1      # still ONT family
                assert feats['is_ultralong'] == 1 # still ultra-long
            else:
                assert feats['is_hiacc'] == 0, f"code {code} should not be hiacc"

    def test_illumina_is_short_read(self):
        """Illumina should be is_illumina=1, is_short_read=1."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        feats = get_chemistry_features(5)  # illumina_hiseq2500
        assert feats['is_illumina'] == 1
        assert feats['is_short_read'] == 1
        assert feats['is_long_read'] == 0

    def test_element_is_element_and_short_read(self):
        """Element codes should be is_element=1, is_short_read=1."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        for code in [7, 8]:  # element_aviti, element_ultraq
            feats = get_chemistry_features(code)
            assert feats['is_element'] == 1
            assert feats['is_short_read'] == 1
            assert feats['is_illumina'] == 0

    def test_r10_and_ultralong_axes(self):
        """R10 and ultra-long flags should be set correctly."""
        from strandweaver.preprocessing.errorsmith_module import get_chemistry_features
        # ont_ulk114_r1041 (code 4): R10 + UL
        feats = get_chemistry_features(4)
        assert feats['is_r10'] == 1
        assert feats['is_ultralong'] == 1
        # ont_lsk114_r1041 (code 3): R10 but NOT UL
        feats = get_chemistry_features(3)
        assert feats['is_r10'] == 1
        assert feats['is_ultralong'] == 0
        # ont_lsk110_r941 (code 1): R9 — no R10, no UL
        feats = get_chemistry_features(1)
        assert feats['is_r10'] == 0
        assert feats['is_ultralong'] == 0

    def test_duplex_flag(self):
        """Duplex flag should only be set for ont_r1041_duplex (code 10)."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_FEATURES
        )
        for code in CHEMISTRY_FEATURES:
            feats = get_chemistry_features(code)
            if code == 10:
                assert feats['is_duplex'] == 1
            else:
                assert feats['is_duplex'] == 0, f"code {code} should not be duplex"

    def test_mutual_exclusivity_company(self):
        """Each chemistry should belong to exactly one company family."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_FEATURES
        )
        company_keys = ['is_ont', 'is_pacbio_hifi', 'is_pacbio_onso',
                        'is_illumina', 'is_element']
        for code in CHEMISTRY_FEATURES:
            feats = get_chemistry_features(code)
            count = sum(feats[k] for k in company_keys)
            assert count == 1, (
                f"Code {code} belongs to {count} company families, expected 1"
            )

    def test_long_short_read_exclusive(self):
        """Each chemistry should be exactly one of long-read or short-read."""
        from strandweaver.preprocessing.errorsmith_module import (
            get_chemistry_features, CHEMISTRY_FEATURES
        )
        for code in CHEMISTRY_FEATURES:
            feats = get_chemistry_features(code)
            assert feats['is_long_read'] + feats['is_short_read'] == 1, (
                f"Code {code}: long={feats['is_long_read']}, short={feats['is_short_read']}"
            )


# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
