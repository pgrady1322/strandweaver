#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for read classification utility.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from strandweaver.preprocessing.read_classification_utility import (
    classify_read_technology,
    detect_technology_from_header
)


class TestReadClassification:
    """Test read technology classification."""
    
    def test_ont_detection_from_header(self):
        """Test ONT read detection from FASTQ header."""
        ont_headers = [
            "@read1 runid=abc123 ch=1 start_time=2024-01-01",
            "@m54321_210101_012345/1/ccs runid=xyz",
            "@read_ont ch=100 read=500"
        ]
        
        for header in ont_headers:
            tech = detect_technology_from_header(header)
            assert tech == "ont" or tech == "ONT", f"Failed to detect ONT from: {header}"
    
    def test_hifi_detection_from_header(self):
        """Test PacBio HiFi read detection from FASTQ header."""
        hifi_headers = [
            "@m64011_190830_220126/4194483/ccs",
            "@m54321_210101_012345/1/ccs",
            "@SRR123456.1 m64011e_190901_095311/1/ccs"
        ]
        
        for header in hifi_headers:
            tech = detect_technology_from_header(header)
            assert tech in ["hifi", "pacbio", "HiFi", "PacBio"], f"Failed to detect HiFi from: {header}"
    
    def test_illumina_detection_from_header(self):
        """Test Illumina read detection from FASTQ header."""
        illumina_headers = [
            "@SRR123456.1 1 length=150",
            "@HISEQ:123:H123ABC:1:1101:1234:5678 1:N:0:ATCACG",
            "@M00123:456:000000000-A1234:1:1101:12345:1234 1:N:0:1"
        ]
        
        for header in illumina_headers:
            tech = detect_technology_from_header(header)
            assert tech in ["illumina", "Illumina"], f"Failed to detect Illumina from: {header}"
    
    def test_unknown_technology_handling(self):
        """Test handling of unrecognized read headers."""
        unknown_header = "@read1 some_random_header"
        
        tech = detect_technology_from_header(unknown_header)
        
        # Should either return 'unknown' or a default
        assert tech is not None


class TestSequenceQuality:
    """Test sequence quality metrics."""
    
    def test_quality_score_parsing(self):
        """Test parsing of Phred quality scores."""
        from strandweaver.preprocessing.read_classification_utility import parse_quality_scores
        
        # Phred+33 encoding
        quality_string = "IIIII"  # All high quality (Q=40)
        scores = parse_quality_scores(quality_string)
        
        assert all(score >= 30 for score in scores), "High quality scores not parsed correctly"
    
    def test_low_quality_detection(self):
        """Test detection of low-quality bases."""
        from strandweaver.preprocessing.read_classification_utility import has_low_quality_bases
        
        high_quality = "IIIIIIIIII"  # All Q=40
        low_quality = "##########"   # All Q=2
        
        assert not has_low_quality_bases(high_quality, threshold=20)
        assert has_low_quality_bases(low_quality, threshold=20)

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
