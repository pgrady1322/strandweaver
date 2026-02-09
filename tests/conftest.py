#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Pytest configuration and shared fixtures.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="strandweaver_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def simple_fasta():
    """Generate simple FASTA sequence for testing."""
    return ">test_sequence\nATCGATCGATCGATCGATCGATCGATCGATCG\n"


@pytest.fixture
def simple_fastq():
    """Generate simple FASTQ reads for testing."""
    return """@read1
ATCGATCGATCG
+
IIIIIIIIIIII
@read2
GCTAGCTAGCTA
+
IIIIIIIIIIII
"""


@pytest.fixture
def diploid_sequences():
    """Generate diploid sequences with SNPs for haplotype testing."""
    hap_a = "ATCGATCGATCGATCG"
    hap_b = "ATCGATCGTTCGATCG"  # SNP at position 9 (A→T)
    return {"haplotype_a": hap_a, "haplotype_b": hap_b}

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
