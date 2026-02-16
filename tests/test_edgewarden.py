#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0-dev

EdgeWarden unit tests.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
import numpy as np
from strandweaver.assembly_core.edgewarden_module import (
    EdgeWarden,
    EdgeScore,
    EdgeWardenScoreManager,
    TechnologyType,
)


class TestEdgeWardenBasics:
    """Test basic EdgeWarden functionality."""

    def test_edgewarden_initialization(self):
        """Test that EdgeWarden initializes correctly."""
        warden = EdgeWarden(technology="pacbio_hifi")
        assert warden is not None
        assert warden.technology.value == "pacbio_hifi"

    def test_edgewarden_all_technologies(self):
        """Test that EdgeWarden initializes for every TechnologyType."""
        for tech in TechnologyType:
            warden = EdgeWarden(technology=tech.value)
            assert warden.technology == tech

    def test_edgewarden_invalid_technology(self):
        """Test that invalid technology raises ValueError."""
        with pytest.raises(ValueError):
            EdgeWarden(technology="invalid_tech")


class TestEdgeWardenScoreManager:
    """Test EdgeWardenScoreManager scoring logic."""

    def test_score_manager_initialization(self):
        """Test that ScoreManager initializes for each technology."""
        for tech in TechnologyType:
            mgr = EdgeWardenScoreManager(technology=tech)
            assert mgr.technology == tech
            assert mgr.tech_weights[tech] is not None

    def test_register_edge_score(self):
        """Test registering an edge score and getting a weighted result."""
        mgr = EdgeWardenScoreManager(technology=TechnologyType.PACBIO_HIFI)
        score = mgr.register_edge_score(
            edge_id="e1", source_node=0, target_node=1,
            edge_confidence=0.9, coverage_consistency=0.85,
            repeat_score=0.1, quality_score=0.95,
            error_pattern_score=0.8, support_count=5,
            coverage_ratio=1.0,
        )
        assert isinstance(score, EdgeScore)
        assert 0.0 <= score.weighted_score <= 1.0
        assert score.edge_id == "e1"

    def test_high_quality_scores_higher(self):
        """Test that high-quality edges score higher than low-quality edges."""
        mgr = EdgeWardenScoreManager(technology=TechnologyType.PACBIO_HIFI)

        high = mgr.register_edge_score(
            edge_id="high", source_node=0, target_node=1,
            edge_confidence=0.95, coverage_consistency=0.90,
            repeat_score=0.05, quality_score=0.95,
            error_pattern_score=0.90,
        )
        low = mgr.register_edge_score(
            edge_id="low", source_node=2, target_node=3,
            edge_confidence=0.2, coverage_consistency=0.3,
            repeat_score=0.9, quality_score=0.15,
            error_pattern_score=0.1,
        )
        assert high.weighted_score > low.weighted_score

    def test_scores_clamped_to_unit_interval(self):
        """Test that scores are clamped to [0, 1] even with extreme inputs."""
        mgr = EdgeWardenScoreManager(technology=TechnologyType.NANOPORE_R10)
        score = mgr.register_edge_score(
            edge_id="extreme", source_node=0, target_node=1,
            edge_confidence=5.0, coverage_consistency=5.0,
            repeat_score=-2.0, quality_score=5.0,
            error_pattern_score=5.0,
        )
        assert 0.0 <= score.weighted_score <= 1.0
        assert 0.0 <= score.edge_confidence <= 1.0

    def test_technology_weights_differ(self):
        """Test that different technologies produce different weighted scores."""
        # Use asymmetric values so different weight distributions yield different results
        kwargs = dict(
            edge_id="e1", source_node=0, target_node=1,
            edge_confidence=0.9, coverage_consistency=0.3,
            repeat_score=0.2, quality_score=0.7,
            error_pattern_score=0.4,
        )
        hifi_mgr = EdgeWardenScoreManager(technology=TechnologyType.PACBIO_HIFI)
        ancient_mgr = EdgeWardenScoreManager(technology=TechnologyType.ANCIENT_DNA)

        hifi_score = hifi_mgr.register_edge_score(**kwargs)
        ancient_score = ancient_mgr.register_edge_score(**kwargs)

        # Weights differ, so weighted scores should differ
        assert hifi_score.weighted_score != ancient_score.weighted_score

# StrandWeaver v0.3.0-dev
# Any usage is subject to this software's license.
