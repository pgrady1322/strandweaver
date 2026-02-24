#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for QV estimation.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from strandweaver.assembly_utils.qv_estimator import (
    QVEstimator,
    ContigQV,
    AssemblyQV,
    estimate_assembly_qv,
    _qv_from_error_rate,
    _qv_from_completeness,
    _gc_content,
    _compute_n50,
    _extract_kmers,
)


# ---------------------------------------------------------------------------
# Minimal mock read/contig — only needs .id and .sequence
# ---------------------------------------------------------------------------

@dataclass
class MockSeqRead:
    id: str
    sequence: str
    quality: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper math tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_qv_from_error_rate_typical(self):
        """Q30 ≈ error rate 0.001."""
        qv = _qv_from_error_rate(0.001)
        assert abs(qv - 30.0) < 0.01

    def test_qv_from_error_rate_zero(self):
        """Zero error rate → max QV (70)."""
        assert _qv_from_error_rate(0.0) == 70.0

    def test_qv_from_error_rate_high(self):
        """10% error → Q10."""
        qv = _qv_from_error_rate(0.1)
        assert abs(qv - 10.0) < 0.01

    def test_qv_from_completeness_perfect(self):
        """100% completeness → max QV."""
        assert _qv_from_completeness(1.0) == 70.0

    def test_qv_from_completeness_typical(self):
        """99.9% completeness → ~Q30."""
        qv = _qv_from_completeness(0.999)
        assert abs(qv - 30.0) < 0.01

    def test_qv_from_completeness_zero(self):
        """0% completeness → Q0."""
        qv = _qv_from_completeness(0.0)
        assert abs(qv - 0.0) < 0.01

    def test_gc_content_balanced(self):
        assert abs(_gc_content("ATCG") - 0.5) < 0.01

    def test_gc_content_all_gc(self):
        assert abs(_gc_content("GGCC") - 1.0) < 0.01

    def test_gc_content_no_gc(self):
        assert abs(_gc_content("AATT") - 0.0) < 0.01

    def test_gc_content_empty(self):
        assert _gc_content("") == 0.0

    def test_n50_simple(self):
        """Three contigs of length 3, 4, 5 → total 12, half=6, N50 = 4."""
        assert _compute_n50([3, 4, 5]) == 4

    def test_n50_single(self):
        assert _compute_n50([100]) == 100

    def test_n50_empty(self):
        assert _compute_n50([]) == 0

    def test_extract_kmers_canonical(self):
        """Canonical k-mers: forward and RC collapsed."""
        kmers = _extract_kmers("ATCG", 3)
        # ATC → canonical min(ATC, GAT) = ATC
        # TCG → canonical min(TCG, CGA) = CGA
        assert len(kmers) == 2
        assert "ATC" in kmers or "GAT" in kmers
        assert "TCG" in kmers or "CGA" in kmers

    def test_extract_kmers_skips_N(self):
        """K-mers containing N are skipped."""
        kmers = _extract_kmers("ANCG", 3)
        # ANC and NCG both contain N → 0 k-mers
        assert len(kmers) == 0


# ---------------------------------------------------------------------------
# QVEstimator construction tests
# ---------------------------------------------------------------------------

class TestQVEstimatorInit:
    """Test QVEstimator constructor validation."""

    def test_default_construction(self):
        est = QVEstimator()
        assert est.k == 21

    def test_custom_k(self):
        est = QVEstimator(k=31)
        assert est.k == 31

    def test_invalid_k_low(self):
        with pytest.raises(ValueError, match="k must be between"):
            QVEstimator(k=5)

    def test_invalid_k_high(self):
        with pytest.raises(ValueError, match="k must be between"):
            QVEstimator(k=200)

    def test_weight_mismatch(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            QVEstimator(kmer_weight=0.5, error_rate_weight=0.3)


# ---------------------------------------------------------------------------
# Core estimation tests
# ---------------------------------------------------------------------------

class TestEstimate:
    """Test QVEstimator.estimate() with synthetic data."""

    @staticmethod
    def _make_contig(cid: str, seq: str) -> MockSeqRead:
        return MockSeqRead(id=cid, sequence=seq)

    @staticmethod
    def _make_read(rid: str, seq: str) -> MockSeqRead:
        return MockSeqRead(id=rid, sequence=seq)

    def test_no_contigs(self):
        """Empty contig list → empty report."""
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate([])
        assert result.num_contigs == 0
        assert result.total_bases == 0

    def test_contigs_below_min_length(self):
        """All contigs shorter than min_contig_length → empty report."""
        contigs = [self._make_contig("c1", "ATCG")]
        est = QVEstimator(k=11, min_contig_length=1000)
        result = est.estimate(contigs)
        assert result.num_contigs == 0

    def test_error_rate_only(self):
        """No reads provided — QV comes from error_rates only."""
        seq = "ATCGATCGATCGATCGATCGATCG"  # 24 bp
        contigs = [self._make_contig("c1", seq)]
        error_rates = {"c1": 0.001}  # Q30
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs, error_rates=error_rates)

        assert result.num_contigs == 1
        assert result.total_bases == 24
        # Without reads, combined_qv == error_rate_qv
        assert abs(result.global_combined_qv - 30.0) < 0.1
        assert result.global_kmer_qv == 0.0  # No reads

    def test_perfect_kmer_completeness(self):
        """Reads contain all contig k-mers → completeness ≈ 1.0."""
        seq = "ATCGATCGATCGATCGATCGATCGATCG"  # 28 bp
        contigs = [self._make_contig("c1", seq)]
        reads = [self._make_read("r1", seq)]  # Same sequence
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs, reads=reads)

        c = result.per_contig[0]
        assert c.kmer_completeness == 1.0
        assert c.kmer_qv == 70.0  # Max QV

    def test_partial_kmer_completeness(self):
        """Reads cover only part of contig → completeness < 1."""
        # Use a non-periodic sequence so trimming removes unique k-mers
        contig_seq = "AACCGGTTAAGGCCTTAACCGGTTAAGGCCTT"  # 32 bp, non-periodic
        read_seq = contig_seq[:16]  # First half only — missing unique 11-mers from the tail
        contigs = [self._make_contig("c1", contig_seq)]
        reads = [self._make_read("r1", read_seq)]
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs, reads=reads)

        c = result.per_contig[0]
        assert 0.0 < c.kmer_completeness < 1.0
        assert c.kmer_qv < 70.0

    def test_multiple_contigs(self):
        """Multiple contigs get individual and global QV."""
        seq1 = "AAACCCGGGTTTAAACCC"  # 18 bp
        seq2 = "TTTGGGCCCAAATTTGGG"  # 18 bp
        contigs = [
            self._make_contig("c1", seq1),
            self._make_contig("c2", seq2),
        ]
        error_rates = {"c1": 0.001, "c2": 0.01}
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs, error_rates=error_rates)

        assert result.num_contigs == 2
        assert len(result.per_contig) == 2
        # c1 should have higher QV than c2
        qv1 = next(c for c in result.per_contig if c.contig_id == "c1")
        qv2 = next(c for c in result.per_contig if c.contig_id == "c2")
        assert qv1.error_rate_qv > qv2.error_rate_qv

    def test_default_error_rate_used(self):
        """When error_rates dict is None, default_error_rate is used."""
        seq = "ATCGATCGATCGATCGATCGATCG"
        contigs = [self._make_contig("c1", seq)]
        est = QVEstimator(k=11, min_contig_length=0, default_error_rate=0.0001)
        result = est.estimate(contigs)

        # default 0.0001 → Q40
        assert abs(result.global_error_rate_qv - 40.0) < 0.1

    def test_n50_computed(self):
        """N50 in report matches expected value."""
        contigs = [
            self._make_contig("c1", "A" * 5000),
            self._make_contig("c2", "A" * 3000),
            self._make_contig("c3", "A" * 2000),
        ]
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs)
        assert result.n50 == 5000


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test JSON output."""

    def test_to_dict_roundtrip(self):
        """AssemblyQV.to_dict() produces valid JSON."""
        seq = "ATCGATCGATCGATCGATCGATCG"
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs)
        d = result.to_dict()
        # Should be JSON-serialisable
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["num_contigs"] == 1

    def test_save_report_file(self, tmp_path):
        """save_report() writes a valid JSON file."""
        seq = "ATCGATCGATCGATCGATCGATCG"
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        est = QVEstimator(k=11, min_contig_length=0)
        result = est.estimate(contigs)
        out = tmp_path / "qv_report.json"
        est.save_report(result, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "global_combined_qv" in data


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    """Test estimate_assembly_qv() top-level function."""

    def test_basic_call(self):
        # Sequence must be >= default min_contig_length (1000 bp)
        seq = "ATCGATCG" * 200  # 1600 bp
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        result = estimate_assembly_qv(contigs, k=11)
        assert result.num_contigs == 1

    def test_with_output(self, tmp_path):
        seq = "ATCGATCG" * 200  # 1600 bp
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        out = tmp_path / "report.json"
        result = estimate_assembly_qv(contigs, k=11, output_path=out)
        assert out.exists()
        assert result.num_contigs == 1

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
