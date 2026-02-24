#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for sequence manipulation utilities.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from strandweaver.utils.sequence_utils import (
    extract_kmers,
    calculate_gc_content,
    reverse_complement,
    count_homopolymers
)


class TestKmerExtraction:
    """Test k-mer extraction functions."""
    
    def test_basic_kmer_extraction(self):
        """Test extraction of k-mers from sequence."""
        sequence = "ATCGATCG"
        k = 3
        
        kmers = extract_kmers(sequence, k)
        
        expected = ["ATC", "TCG", "CGA", "GAT", "ATC", "TCG"]
        assert kmers == expected
    
    def test_kmer_count_correct(self):
        """Test that number of k-mers is correct."""
        sequence = "ATCGATCG"  # Length 8
        k = 3
        
        kmers = extract_kmers(sequence, k)
        
        # Should have (length - k + 1) k-mers
        expected_count = len(sequence) - k + 1
        assert len(kmers) == expected_count
    
    def test_kmer_larger_than_sequence(self):
        """Test handling when k > sequence length."""
        sequence = "ATG"
        k = 5
        
        kmers = extract_kmers(sequence, k)
        
        assert len(kmers) == 0  # No k-mers possible


class TestGCContent:
    """Test GC content calculation."""
    
    def test_gc_content_all_gc(self):
        """Test GC content of 100% GC sequence."""
        sequence = "GCGCGCGC"
        gc = calculate_gc_content(sequence)
        
        assert gc == 1.0
    
    def test_gc_content_all_at(self):
        """Test GC content of 0% GC sequence."""
        sequence = "ATATATAT"
        gc = calculate_gc_content(sequence)
        
        assert gc == 0.0
    
    def test_gc_content_50_percent(self):
        """Test GC content of 50% GC sequence."""
        sequence = "ATGC"
        gc = calculate_gc_content(sequence)
        
        assert gc == 0.5
    
    def test_gc_content_case_insensitive(self):
        """Test that GC calculation is case-insensitive."""
        upper = "ATGC"
        lower = "atgc"
        
        assert calculate_gc_content(upper) == calculate_gc_content(lower)


class TestReverseComplement:
    """Test reverse complement functions."""
    
    def test_reverse_complement_basic(self):
        """Test basic reverse complement."""
        sequence = "ATCG"
        rc = reverse_complement(sequence)
        
        assert rc == "CGAT"
    
    def test_reverse_complement_palindrome(self):
        """Test reverse complement of palindromic sequence."""
        sequence = "GAATTC"  # EcoRI site
        rc = reverse_complement(sequence)
        
        assert rc == sequence  # Should equal itself
    
    def test_double_reverse_complement(self):
        """Test that reverse complement twice gives original."""
        sequence = "ATCGATCG"
        rc1 = reverse_complement(sequence)
        rc2 = reverse_complement(rc1)
        
        assert rc2 == sequence


class TestHomopolymers:
    """Test homopolymer detection."""
    
    def test_homopolymer_detection(self):
        """Test detection of homopolymer runs."""
        sequence = "ATCGAAAAAGTC"  # 5 A's
        
        homopolymers = count_homopolymers(sequence, min_length=4)
        
        assert len(homopolymers) > 0
        assert any(hp['base'] == 'A' and hp['length'] >= 5 for hp in homopolymers)
    
    def test_no_homopolymers(self):
        """Test sequence with no long homopolymers."""
        sequence = "ATCGATCGATCG"
        
        homopolymers = count_homopolymers(sequence, min_length=4)
        
        assert len(homopolymers) == 0

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
