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

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
