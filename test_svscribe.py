#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick test for SVScribe implementation.
Run this to verify basic functionality without full pipeline integration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from strandweaver.assembly_core.svscribe_module import (
    SVScribe, SVCall, SVEvidence, SVSignature,
    detect_structural_variants, svs_to_dict_list
)


def test_data_structures():
    """Test data structure creation."""
    print("Testing data structures...")
    
    # Test SVEvidence
    evidence = SVEvidence(
        has_sequence_support=True,
        has_ul_support=True,
        sequence_confidence=0.8,
        ul_confidence=0.6,
        supporting_reads=5,
        edge_types={'sequence', 'ul'}
    )
    assert evidence.has_sequence_support
    assert evidence.has_ul_support
    assert not evidence.has_hic_support
    print("✓ SVEvidence works")
    
    # Test SVSignature
    signature = SVSignature(
        pattern_type='ul_spanning_gap',
        involved_nodes=[1, 2],
        involved_edges=[10],
        coverage_pattern='gap',
        topology_score=0.7
    )
    assert signature.pattern_type == 'ul_spanning_gap'
    assert len(signature.involved_nodes) == 2
    print("✓ SVSignature works")
    
    # Test SVCall
    sv_call = SVCall(
        sv_id='SV000001',
        sv_type='DEL',
        nodes=[1, 2, 3],
        size=5000,
        confidence=0.75,
        evidence={'test': 'data'},
        haplotype=0,
        breakpoints=[(1, 3)]
    )
    assert sv_call.sv_id == 'SV000001'
    assert sv_call.sv_type == 'DEL'
    assert sv_call.haplotype == 0
    print("✓ SVCall works")
    
    print("All data structures: PASS\n")


def test_svscribe_initialization():
    """Test SVScribe initialization."""
    print("Testing SVScribe initialization...")
    
    # Without AI
    sv_scribe = SVScribe(
        use_ai=False,
        min_confidence=0.5,
        min_size=50
    )
    assert sv_scribe.min_confidence == 0.5
    assert sv_scribe.min_size == 50
    assert not sv_scribe.use_ai
    assert sv_scribe.ml_model is None
    print("✓ Basic initialization works")
    
    # Check detectors initialized
    assert sv_scribe.deletion_detector is not None
    assert sv_scribe.insertion_detector is not None
    assert sv_scribe.inversion_detector is not None
    assert sv_scribe.duplication_detector is not None
    assert sv_scribe.translocation_detector is not None
    print("✓ All detectors initialized")
    
    # Check SV counts initialized
    counts = sv_scribe.get_sv_type_counts()
    assert counts == {'DEL': 0, 'INS': 0, 'INV': 0, 'DUP': 0, 'TRA': 0}
    print("✓ SV type counts initialized")
    
    print("SVScribe initialization: PASS\n")


def test_convenience_functions():
    """Test convenience functions."""
    print("Testing convenience functions...")
    
    # Test svs_to_dict_list
    sv_calls = [
        SVCall(
            sv_id='SV000001',
            sv_type='DEL',
            nodes=[1, 2],
            size=1000,
            confidence=0.8,
            evidence={'source': 'test'},
            haplotype=0,
            breakpoints=[(1, 2)],
            metadata={'test': 'value'}
        ),
        SVCall(
            sv_id='SV000002',
            sv_type='INS',
            nodes=[3, 4, 5],
            size=500,
            confidence=0.6,
            evidence={},
            haplotype=1,
            breakpoints=[(3, 5)],
            metadata={}
        )
    ]
    
    sv_dicts = svs_to_dict_list(sv_calls)
    assert len(sv_dicts) == 2
    assert sv_dicts[0]['sv_id'] == 'SV000001'
    assert sv_dicts[0]['sv_type'] == 'DEL'
    assert sv_dicts[1]['sv_id'] == 'SV000002'
    assert sv_dicts[1]['sv_type'] == 'INS'
    print("✓ svs_to_dict_list works")
    
    print("Convenience functions: PASS\n")


def test_edge_categorization():
    """Test edge categorization logic."""
    print("Testing edge categorization...")
    
    # Mock graph
    class MockEdge:
        def __init__(self, edge_type='sequence'):
            self.edge_type = edge_type
    
    class MockGraph:
        def __init__(self):
            self.edges = {
                1: MockEdge('sequence'),
                2: MockEdge('sequence'),
                3: MockEdge('ul'),
                4: MockEdge('ul'),
                5: MockEdge('hic'),
            }
            self.nodes = {}
            self.out_edges = {}
            self.in_edges = {}
    
    sv_scribe = SVScribe()
    graph = MockGraph()
    
    # Test with distinguish_edge_types=True
    seq, ul, hic = sv_scribe._categorize_edges(graph, True)
    assert len(seq) == 2
    assert len(ul) == 2
    assert len(hic) == 1
    assert 1 in seq and 2 in seq
    assert 3 in ul and 4 in ul
    assert 5 in hic
    print("✓ Edge categorization with types works")
    
    # Test with distinguish_edge_types=False
    seq, ul, hic = sv_scribe._categorize_edges(graph, False)
    assert len(seq) == 5
    assert len(ul) == 0
    assert len(hic) == 0
    print("✓ Edge categorization without types works")
    
    print("Edge categorization: PASS\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SVScribe Implementation Test")
    print("=" * 60)
    print()
    
    try:
        test_data_structures()
        test_svscribe_initialization()
        test_convenience_functions()
        test_edge_categorization()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        print("SVScribe implementation is ready for integration!")
        print()
        print("Next steps:")
        print("1. Add SVScribe call to pipeline.py:_step_assemble()")
        print("2. Pass ul_routes from ThreadCompass.get_routes()")
        print("3. Pass phasing_info from HaplotypeDetangler.phase_graph()")
        print("4. Export SVs to JSON file")
        print("5. Test on real assembly graphs")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
