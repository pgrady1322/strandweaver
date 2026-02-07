"""
Unit tests for K-Weaver module (k-mer size prediction).

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from strandweaver.preprocessing.kweaver_module import (
    KWeaverPredictor,
    KmerPrediction,
    extract_read_features
)


class TestKWeaverBasics:
    """Test basic K-Weaver functionality."""
    
    def test_predictor_initialization(self):
        """Test that KWeaverPredictor initializes correctly."""
        predictor = KWeaverPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'predict')
    
    def test_kmer_prediction_structure(self):
        """Test KmerPrediction dataclass has required fields."""
        prediction = KmerPrediction(
            dbg_k=31,
            ul_overlap_k=1001,
            extension_k=55,
            polish_k=77,
            dbg_confidence=0.9,
            ul_confidence=0.85,
            extension_confidence=0.88,
            polish_confidence=0.92
        )
        
        assert prediction.dbg_k == 31
        assert prediction.ul_overlap_k == 1001
        assert prediction.extension_k == 55
        assert prediction.polish_k == 77
        assert 0 <= prediction.dbg_confidence <= 1.0
    
    def test_kmer_values_are_odd(self):
        """Test that k-mer values are odd (required for De Bruijn graphs)."""
        prediction = KmerPrediction(
            dbg_k=31,
            ul_overlap_k=1001,
            extension_k=55,
            polish_k=77
        )
        
        assert prediction.dbg_k % 2 == 1
        assert prediction.ul_overlap_k % 2 == 1
        assert prediction.extension_k % 2 == 1
        assert prediction.polish_k % 2 == 1
    
    def test_default_predictions(self):
        """Test that predictor returns reasonable defaults when no data."""
        predictor = KWeaverPredictor()
        
        # Test with minimal/default inputs
        prediction = predictor.get_default_prediction(technology="hifi")
        
        assert prediction.dbg_k > 0
        assert prediction.dbg_k < 200  # Reasonable upper bound
        assert prediction.ul_overlap_k > prediction.dbg_k  # UL k-mers should be larger
