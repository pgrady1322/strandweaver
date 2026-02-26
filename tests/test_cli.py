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
        """CHEMISTRY_CODES should have 6 entries."""
        from strandweaver.preprocessing.errorsmith_module import CHEMISTRY_CODES
        assert len(CHEMISTRY_CODES) == 6
        assert 'pacbio_hifi_sequel2' in CHEMISTRY_CODES
        assert 'ont_lsk110_r941' in CHEMISTRY_CODES
        assert 'ont_ulk001_r941' in CHEMISTRY_CODES
        assert 'ont_lsk114_r1041' in CHEMISTRY_CODES
        assert 'ont_ulk114_r1041' in CHEMISTRY_CODES
        assert 'illumina_hiseq2500' in CHEMISTRY_CODES

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
        assert name == 'ont_lsk110_r941'
        assert code == 1

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
        """get_corrector() should work for all 4 technology families."""
        from strandweaver.preprocessing.errorsmith_module import get_corrector
        for tech in ['ont', 'pacbio', 'illumina', 'ancient_dna']:
            c = get_corrector(tech)
            assert c.chemistry is not None
            assert c.chemistry_code is not None


# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
