#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

CLI smoke tests.

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
    
    def test_assemble_help(self):
        """Test assemble command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['assemble', '--help'])
        
        assert result.exit_code == 0
        assert 'assemble' in result.output.lower()
    
    def test_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        runner = CliRunner()
        result = runner.invoke(main, ['nonexistent_command'])
        
        # Should fail but not crash
        assert result.exit_code != 0


class TestAssembleCLI:
    """Test assemble command specifically."""
    
    def test_assemble_missing_input(self):
        """Test assemble fails gracefully without input."""
        runner = CliRunner()
        result = runner.invoke(main, ['assemble', '--output', 'test_out'])
        
        # Should fail due to missing input reads
        assert result.exit_code != 0
    
    def test_assemble_with_nonexistent_file(self):
        """Test assemble handles nonexistent input files."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'assemble',
            '--hifi', 'nonexistent.fastq',
            '--output', 'test_out'
        ])
        
        # Should fail but not crash
        assert result.exit_code != 0


# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
