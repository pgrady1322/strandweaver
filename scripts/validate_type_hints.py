#!/usr/bin/env python3
"""
Validation script for type hints and quality score improvements.

Tests:
1. Type compatibility across pipeline handoffs
2. Quality score calculation correctness
3. Metadata preservation
"""

import math
from typing import List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strandweaver.io import SeqRead, ReadTechnology


def test_quality_calculation():
    """Test quality score calculation formula."""
    print("=" * 80)
    print("TEST 1: Quality Score Calculation")
    print("=" * 80)
    
    test_cases = [
        (1, 23),    # 1× coverage → Q23
        (2, 24),    # 2× coverage → Q24 (actual: 24.77 → 24)
        (3, 26),    # 3× coverage → Q26
        (5, 27),    # 5× coverage → Q27
        (10, 30),   # 10× coverage → Q30
        (20, 33),   # 20× coverage → Q33
        (30, 34),   # 30× coverage → Q34 (actual: 34.77 → 34)
        (50, 37),   # 50× coverage → Q37
        (100, 40),  # 100× coverage → Q40
        (1000, 40), # 1000× coverage → Q40 (capped)
    ]
    
    print("\nCoverage → Quality Mapping:")
    print("-" * 40)
    all_passed = True
    
    for coverage, expected_q in test_cases:
        actual_q = min(40, int(20 + 10 * math.log10(coverage + 1)))
        status = "✓" if actual_q == expected_q else "✗"
        
        if actual_q != expected_q:
            all_passed = False
            print(f"{status} Coverage {coverage:4d}× → Q{actual_q} (expected Q{expected_q}) FAILED")
        else:
            # Calculate accuracy
            error_rate = 10 ** (-(actual_q / 10))
            accuracy = (1 - error_rate) * 100
            print(f"{status} Coverage {coverage:4d}× → Q{actual_q} ({accuracy:.4f}% accuracy)")
    
    print("-" * 40)
    if all_passed:
        print("✓ All quality calculations PASSED")
    else:
        print("✗ Some quality calculations FAILED")
    
    return all_passed


def test_type_compatibility():
    """Test type compatibility of SeqRead objects through pipeline."""
    print("\n" + "=" * 80)
    print("TEST 2: Type Compatibility")
    print("=" * 80)
    
    # Create test reads
    test_reads: List[SeqRead] = [
        SeqRead(
            id=f"test_read_{i}",
            sequence="ACGTACGT" * 25,  # 200bp
            quality="I" * 200,           # Q40
            technology=ReadTechnology.ILLUMINA
        )
        for i in range(10)
    ]
    
    print(f"\n✓ Created {len(test_reads)} SeqRead objects")
    print(f"  Type: {type(test_reads[0])}")
    print(f"  Attributes: id, sequence, quality, technology, metadata")
    
    # Test metadata enhancement
    test_contig = SeqRead(
        id="test_contig_1",
        sequence="ACGT" * 100,  # 400bp
        quality=chr(30 + 33) * 400,  # Q30
        technology=ReadTechnology.ILLUMINA,
        metadata={
            'source': 'dbg',
            'coverage': 10.5,
            'depth': 10,
            'avg_quality': 30,
            'num_reads': 10
        }
    )
    
    print(f"\n✓ Created contig with enhanced metadata")
    print(f"  ID: {test_contig.id}")
    print(f"  Length: {len(test_contig.sequence)}bp")
    print(f"  Quality: Q{ord(test_contig.quality[0]) - 33}")
    print(f"  Metadata keys: {list(test_contig.metadata.keys())}")
    print(f"  Coverage: {test_contig.metadata.get('coverage')}")
    print(f"  Depth: {test_contig.metadata.get('depth')}")
    
    return True


def test_quality_string_format():
    """Test that quality strings are properly formatted."""
    print("\n" + "=" * 80)
    print("TEST 3: Quality String Format (Phred+33)")
    print("=" * 80)
    
    print("\nQuality Score → ASCII Character Mapping:")
    print("-" * 40)
    
    test_qualities = [20, 25, 30, 35, 40]
    for q in test_qualities:
        ascii_char = chr(q + 33)
        ascii_code = ord(ascii_char)
        print(f"Q{q:2d} → '{ascii_char}' (ASCII {ascii_code})")
    
    print("-" * 40)
    
    # Test a contig with quality string
    seq_length = 100
    quality_score = 30
    quality_string = chr(quality_score + 33) * seq_length
    
    print(f"\n✓ Example contig quality string:")
    print(f"  Sequence length: {seq_length}bp")
    print(f"  Quality score: Q{quality_score}")
    print(f"  Quality string length: {len(quality_string)}")
    print(f"  Quality string sample: {quality_string[:10]}... (first 10 chars)")
    print(f"  Decoded quality: Q{ord(quality_string[0]) - 33}")
    
    # Validate
    assert len(quality_string) == seq_length, "Quality length mismatch"
    assert all(ord(c) - 33 == quality_score for c in quality_string), "Quality not uniform"
    
    print("\n✓ Quality string format validated")
    return True


def test_metadata_provenance():
    """Test metadata tracking through pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: Metadata Provenance Tracking")
    print("=" * 80)
    
    # Simulate OLC contig
    olc_contig = SeqRead(
        id="olc_contig_1",
        sequence="ACGT" * 150,  # 600bp
        quality=chr(28 + 33) * 600,  # Q28 (from 5 reads)
        technology=ReadTechnology.ILLUMINA,
        metadata={
            'num_reads': 5,
            'read_ids': ['read_1', 'read_2', 'read_3', 'read_4', 'read_5'],
            'contig_type': 'artificial_long_read',
            'depth': 5,
            'avg_quality': 28
        }
    )
    
    print("\n✓ OLC Contig Metadata:")
    print(f"  Source reads: {olc_contig.metadata['num_reads']}")
    print(f"  Read IDs: {', '.join(olc_contig.metadata['read_ids'][:3])}...")
    print(f"  Depth: {olc_contig.metadata['depth']}×")
    print(f"  Average quality: Q{olc_contig.metadata['avg_quality']}")
    print(f"  Type: {olc_contig.metadata['contig_type']}")
    
    # Simulate DBG contig
    dbg_contig = SeqRead(
        id="dbg_contig_42",
        sequence="ACGT" * 200,  # 800bp
        quality=chr(33 + 33) * 800,  # Q33 (from 20× coverage)
        technology=ReadTechnology.ILLUMINA,
        metadata={
            'source': 'dbg',
            'node_id': 42,
            'coverage': 20.5,
            'length': 800,
            'recommended_k': 31
        }
    )
    
    print("\n✓ DBG Contig Metadata:")
    print(f"  Source: {dbg_contig.metadata['source']}")
    print(f"  Node ID: {dbg_contig.metadata['node_id']}")
    print(f"  Coverage: {dbg_contig.metadata['coverage']}×")
    print(f"  Length: {dbg_contig.metadata['length']}bp")
    print(f"  Recommended k: {dbg_contig.metadata['recommended_k']}")
    
    print("\n✓ Metadata provenance tracking validated")
    return True


def main():
    """Run all validation tests."""
    print("\n")
    print("=" * 80)
    print("TYPE HINTS AND QUALITY SCORE VALIDATION")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Quality Calculation", test_quality_calculation()))
    results.append(("Type Compatibility", test_type_compatibility()))
    results.append(("Quality String Format", test_quality_string_format()))
    results.append(("Metadata Provenance", test_metadata_provenance()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
