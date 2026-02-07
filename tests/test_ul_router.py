#!/usr/bin/env python3
"""Tests for ULRouter/ThreadCompass (UL read routing) functionality.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestULRouterBasics:
    """Test basic ULRouter/ThreadCompass functionality."""
    
    def test_ulrouter_is_threadcompass(self):
        """Test that ULRouter is an alias for ThreadCompass."""
        from strandweaver.assembly_core import ULRouter, ThreadCompass
        
        assert ULRouter is ThreadCompass
        assert ULRouter.__name__ == 'ThreadCompass'
    
    def test_threadcompass_imports(self):
        """Test that ThreadCompass can be imported."""
        from strandweaver.assembly_core import ThreadCompass
        from strandweaver.assembly_core.threadcompass_module import ThreadCompass as TC
        
        assert ThreadCompass is TC
    
    def test_threadcompass_initialization(self):
        """Test ThreadCompass can be initialized with basic parameters."""
        from strandweaver.assembly_core import ThreadCompass
        
        # Create a mock graph
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(
            graph=mock_graph,
            k_mer_size=31
        )
        
        assert compass is not None
        assert compass.k_mer_size == 31
        assert compass.graph == mock_graph
    
    def test_threadcompass_with_custom_params(self):
        """Test ThreadCompass initialization with custom parameters."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(
            graph=mock_graph,
            k_mer_size=51,
            min_anchor_count=3,
            min_mapq=30,
            mapq_weight=0.3
        )
        
        assert compass.k_mer_size == 51
        assert compass.min_anchor_count == 3
        assert compass.min_mapq == 30
        assert compass.mapq_weight == 0.3
    
    def test_threadcompass_has_expected_methods(self):
        """Test ThreadCompass has all expected public methods."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(graph=mock_graph)
        
        # Check all public API methods exist
        assert hasattr(compass, 'route_ul_reads')
        assert hasattr(compass, 'detect_new_joins')
        assert hasattr(compass, 'score_join')
        assert hasattr(compass, 'score_path')
        assert hasattr(compass, 'get_routes')
        assert hasattr(compass, 'get_mapping_stats')
        assert hasattr(compass, 'register_ul_mappings')
        
        # Check they're callable
        assert callable(compass.route_ul_reads)
        assert callable(compass.detect_new_joins)
        assert callable(compass.score_join)
        assert callable(compass.score_path)


class TestULRouterDataStructures:
    """Test ULRouter/ThreadCompass data structures."""
    
    def test_ulmapping_creation(self):
        """Test ULMapping dataclass can be created."""
        from strandweaver.assembly_core.threadcompass_module import ULMapping
        
        mapping = ULMapping(
            read_id="read_001",
            primary_node=42,
            secondary_nodes=[43, 44],
            mapping_quality=60
        )
        
        assert mapping.read_id == "read_001"
        assert mapping.primary_node == 42
        assert mapping.secondary_nodes == [43, 44]
        assert mapping.mapping_quality == 60
        assert mapping.is_multimapping is False
    
    def test_uljoinscore_creation(self):
        """Test ULJoinScore dataclass can be created."""
        from strandweaver.assembly_core.threadcompass_module import ULJoinScore
        
        score = ULJoinScore(
            join_id="42_43",
            from_node=42,
            to_node=43,
            ul_confidence=0.85
        )
        
        assert score.join_id == "42_43"
        assert score.from_node == 42
        assert score.to_node == 43
        assert score.ul_confidence == 0.85
    
    def test_ulpath_creation(self):
        """Test ULPath dataclass can be created."""
        from strandweaver.assembly_core.threadcompass_module import ULPath
        
        path = ULPath(
            nodes=[1, 2, 3, 4],
            anchors=[],
            total_aligned=15000,
            gaps=[],
            strand_consistent=True
        )
        
        assert path.nodes == [1, 2, 3, 4]
        assert path.total_aligned == 15000
        assert path.strand_consistent is True


class TestThreadCompassIntegration:
    """Test ThreadCompass integration with mock graphs."""
    
    def test_detect_new_joins_empty(self):
        """Test detect_new_joins with no mappings returns empty list."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(graph=mock_graph)
        joins = compass.detect_new_joins()
        
        assert joins == []
    
    def test_score_path_single_node(self):
        """Test score_path with single-node path returns neutral score."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {1: Mock()}
        mock_graph.edges = {}
        
        compass = ThreadCompass(graph=mock_graph)
        score = compass.score_path([1])
        
        # Single-node path should get neutral score
        assert score == 0.5
    
    def test_get_mapping_stats_empty(self):
        """Test get_mapping_stats with no mappings."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(graph=mock_graph)
        stats = compass.get_mapping_stats()
        
        assert 'error' in stats or 'total_reads' in stats
    
    def test_get_routes_empty(self):
        """Test get_routes with no routing performed."""
        from strandweaver.assembly_core import ThreadCompass
        
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.edges = {}
        
        compass = ThreadCompass(graph=mock_graph)
        routes = compass.get_routes()
        
        assert routes == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

