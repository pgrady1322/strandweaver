#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

EdgeWarden unit tests.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
import numpy as np
from strandweaver.assembly_core.edgewarden_module import (
    EdgeWarden,
    EdgeFeatures,
    EdgeQuality
)


class TestEdgeWardenBasics:
    """Test basic EdgeWarden functionality."""
    
    def test_edgewarden_initialization(self):
        """Test that EdgeWarden initializes correctly."""
        warden = EdgeWarden(technology="hifi")
        assert warden is not None
        assert warden.technology == "hifi"
    
    def test_edge_score_range(self):
        """Test that edge scores are normalized to [0, 1]."""
        warden = EdgeWarden(technology="hifi")
        
        # Create minimal edge data
        edge_data = {
            "coverage": 30,
            "length": 1000,
            "quality_mean": 25
        }
        
        score = warden.score_edge(edge_data)
        
        assert 0.0 <= score <= 1.0, f"Score {score} out of valid range"
    
    def test_high_quality_edge_scores_high(self):
        """Test that high-quality edges get high scores."""
        warden = EdgeWarden(technology="hifi")
        
        high_quality_edge = {
            "coverage": 50,  # High coverage
            "length": 5000,  # Long overlap
            "quality_mean": 35,  # High quality
            "identity": 0.999  # Very similar sequences
        }
        
        score = warden.score_edge(high_quality_edge)
        
        assert score > 0.7, f"High quality edge scored too low: {score}"
    
    def test_low_quality_edge_scores_low(self):
        """Test that low-quality edges get low scores."""
        warden = EdgeWarden(technology="ont")
        
        low_quality_edge = {
            "coverage": 3,  # Low coverage
            "length": 100,  # Short overlap
            "quality_mean": 8,  # Low quality
            "identity": 0.85  # Lower identity
        }
        
        score = warden.score_edge(low_quality_edge)
        
        assert score < 0.5, f"Low quality edge scored too high: {score}"
    
    def test_technology_specific_scoring(self):
        """Test that different technologies have different scoring thresholds."""
        edge_data = {
            "coverage": 20,
            "length": 1000,
            "quality_mean": 20
        }
        
        hifi_warden = EdgeWarden(technology="hifi")
        ont_warden = EdgeWarden(technology="ont")
        
        hifi_score = hifi_warden.score_edge(edge_data)
        ont_score = ont_warden.score_edge(edge_data)
        
        # Scores should differ based on technology expectations
        # (HiFi has higher quality expectations)
        assert hifi_score != ont_score

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
