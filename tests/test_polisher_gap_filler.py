#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Tests for iterative polisher and gap filler.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from strandweaver.assembly_utils.iterative_polisher import (
    IterativePolisher,
    PolishingStats,
    PolishingSummary,
    _build_kmer_index,
    _find_best_anchor,
    _build_pileup,
    _call_consensus,
)
from strandweaver.assembly_utils.gap_filler import (
    GapFiller,
    GapRegion,
    GapFillingSummary,
    _find_gaps,
    _build_kmer_set,
    _consensus_from_bridges,
)


# ---------------------------------------------------------------------------
# Minimal mock
# ---------------------------------------------------------------------------

@dataclass
class MockSeqRead:
    id: str
    sequence: str
    quality: Optional[str] = None


# ============================================================================
# POLISHER TESTS
# ============================================================================


class TestPolisherHelpers:
    """Test polisher internal helpers."""

    def test_build_kmer_index(self):
        idx = _build_kmer_index("ACGTACGT", 4)
        assert "ACGT" in idx
        assert len(idx["ACGT"]) == 2  # Positions 0 and 4

    def test_build_kmer_index_skips_N(self):
        idx = _build_kmer_index("ACNTACGT", 4)
        # ACNT and CNTA contain N → skipped
        assert "ACNT" not in idx

    def test_find_best_anchor_exact(self):
        """Read identical to contig → offset 0."""
        seq = "AACCGGTTAACCGGTT"
        idx = _build_kmer_index(seq, 5)
        offset = _find_best_anchor(seq, idx, 5, len(seq))
        assert offset == 0

    def test_find_best_anchor_shifted(self):
        """Read is a substring starting at position 4."""
        contig = "AACCGGTTAACCGGTTAACC"
        read = contig[4:14]  # "GGTTAACCGG"
        idx = _build_kmer_index(contig, 5)
        offset = _find_best_anchor(read, idx, 5, len(contig))
        assert offset == 4

    def test_find_best_anchor_no_match(self):
        contig = "AAAAAAAAAAAA"
        read = "CCCCCCCCCCCC"
        idx = _build_kmer_index(contig, 5)
        offset = _find_best_anchor(read, idx, 5, len(contig))
        assert offset is None

    def test_build_pileup_basic(self):
        contig = "AACCGGTTAACCGG"
        reads = [MockSeqRead(id="r1", sequence=contig)]
        pileup = _build_pileup(contig, reads, k=5)
        assert len(pileup) == len(contig)
        # Every position should have at least 1 entry from the identical read
        for col in pileup:
            assert len(col) >= 1

    def test_call_consensus_no_change(self):
        """Consensus of identical sequences → no changes."""
        seq = "AACCGGTT"
        pileup = [[(b, 30)] * 5 for b in seq]  # 5x coverage, Q30
        new_seq, changed = _call_consensus(pileup, seq, min_coverage=3)
        assert new_seq == seq
        assert changed == 0

    def test_call_consensus_corrects_error(self):
        """Majority of reads disagree with original → correction."""
        seq = "AACCGGTT"
        # Position 0: original 'A', but 4 reads say 'T', 1 says 'A'
        pileup = []
        pileup.append([('T', 30)] * 4 + [('A', 30)])  # pos 0 → T
        for b in seq[1:]:
            pileup.append([(b, 30)] * 5)
        new_seq, changed = _call_consensus(pileup, seq, min_coverage=3)
        assert new_seq[0] == 'T'
        assert changed == 1

    def test_call_consensus_low_coverage_skipped(self):
        """Positions below min_coverage keep original base."""
        seq = "AACCGGTT"
        pileup = [[('T', 30)] * 2 for _ in seq]  # Only 2x coverage
        new_seq, changed = _call_consensus(pileup, seq, min_coverage=3)
        assert new_seq == seq
        assert changed == 0


class TestIterativePolisher:
    """Test IterativePolisher.polish()."""

    def test_empty_contigs(self):
        polisher = IterativePolisher(max_rounds=1, k=5)
        contigs, summary = polisher.polish([], [])
        assert summary.total_rounds == 0
        assert summary.converged is True

    def test_polish_with_identical_reads(self):
        """Polishing with reads identical to contigs → 0 corrections."""
        seq = "AACCGGTTAACCGGTTAACCGGTTAACCGGTT"  # 32 bp, enough for k=11
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        reads = [MockSeqRead(id="r1", sequence=seq)]
        polisher = IterativePolisher(max_rounds=2, k=11, min_coverage=1)
        polished, summary = polisher.polish(contigs, reads)
        assert summary.total_bases_corrected == 0
        assert summary.converged is True
        assert polished[0].sequence == seq

    def test_polish_corrects_error(self):
        """
        Contig has a single error at position 15.
        Multiple reads cover the correct base → polisher should fix it.
        """
        correct_seq = "AACCGGTTAACCGGTTAACCGGTTAACCGGTT"  # 32 bp
        # Introduce error at position 15: change 'T' to 'G'
        error_seq = correct_seq[:15] + 'G' + correct_seq[16:]
        assert error_seq != correct_seq

        contigs = [MockSeqRead(id="c1", sequence=error_seq)]
        # 5 reads with the correct sequence → strong consensus
        reads = [MockSeqRead(id=f"r{i}", sequence=correct_seq) for i in range(5)]

        polisher = IterativePolisher(max_rounds=2, k=11, min_coverage=3)
        polished, summary = polisher.polish(contigs, reads)

        assert summary.total_bases_corrected >= 1
        assert polished[0].sequence[15] == correct_seq[15]  # Error corrected

    def test_max_rounds_respected(self):
        """Polisher does not exceed max_rounds."""
        seq = "AACCGGTTAACCGGTTAACCGGTTAACCGGTT"  # 32 bp
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        reads = [MockSeqRead(id="r1", sequence=seq)]
        polisher = IterativePolisher(max_rounds=3, k=11, min_coverage=1)
        _, summary = polisher.polish(contigs, reads)
        assert summary.total_rounds <= 3

    def test_convergence_detection(self):
        """With identical reads, polisher converges quickly."""
        seq = "AACCGGTTAACCGGTTAACCGGTTAACCGGTT"  # 32 bp
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        reads = [MockSeqRead(id="r1", sequence=seq)]
        polisher = IterativePolisher(
            max_rounds=10, k=11, min_coverage=1,
            use_qv_convergence=True, convergence_delta=0.1,
        )
        _, summary = polisher.polish(contigs, reads)
        assert summary.converged is True
        assert summary.total_rounds <= 2  # Should stop after round 1


class TestPolisherInit:
    """Test IterativePolisher construction."""

    def test_defaults(self):
        p = IterativePolisher()
        assert p.max_rounds == 2
        assert p.k == 21

    def test_custom_params(self):
        p = IterativePolisher(max_rounds=5, k=31, min_coverage=5)
        assert p.max_rounds == 5
        assert p.k == 31
        assert p.min_coverage == 5

    def test_min_rounds_clamped(self):
        p = IterativePolisher(max_rounds=0)
        assert p.max_rounds == 1  # Clamped to minimum 1


# ============================================================================
# GAP FILLER TESTS
# ============================================================================


class TestGapFinderHelpers:
    """Test gap filler internal helpers."""

    def test_find_gaps_basic(self):
        seq = "ACGTNNNNACGT"
        gaps = _find_gaps(seq)
        assert len(gaps) == 1
        assert gaps[0] == (4, 8)

    def test_find_gaps_multiple(self):
        seq = "ACGTNNNACGTNNNNNNACGT"
        gaps = _find_gaps(seq)
        assert len(gaps) == 2

    def test_find_gaps_none(self):
        seq = "ACGTACGTACGT"
        gaps = _find_gaps(seq)
        assert len(gaps) == 0

    def test_find_gaps_min_size(self):
        seq = "ACGTNACGTNNNNACGT"
        gaps = _find_gaps(seq, min_gap_size=3)
        assert len(gaps) == 1
        assert gaps[0] == (9, 13)

    def test_build_kmer_set(self):
        s = _build_kmer_set("ACGTACGT", 4)
        assert "ACGT" in s
        assert len(s) > 0

    def test_build_kmer_set_skips_N(self):
        s = _build_kmer_set("ACNTNNN", 3)
        # All 3-mers contain N → empty
        assert len(s) == 0

    def test_consensus_from_bridges_identical(self):
        """Identical bridges → perfect consensus."""
        bridge = "ACGTACGT"
        bridges = [(bridge, [30] * len(bridge))] * 5
        cons, conf = _consensus_from_bridges(bridges, gap_size=8)
        assert cons == bridge
        assert conf > 0.9

    def test_consensus_from_bridges_empty(self):
        cons, conf = _consensus_from_bridges([], gap_size=10)
        assert cons is None
        assert conf == 0.0

    def test_consensus_majority_vote(self):
        """Majority base at position 0 should win."""
        bridges = [
            ("TACGT", [30, 30, 30, 30, 30]),
            ("TACGT", [30, 30, 30, 30, 30]),
            ("TACGT", [30, 30, 30, 30, 30]),
            ("AACGT", [30, 30, 30, 30, 30]),  # Disagrees at pos 0
        ]
        cons, conf = _consensus_from_bridges(bridges, gap_size=5)
        assert cons is not None
        assert cons[0] == 'T'  # 3 vs 1


class TestGapFiller:
    """Test GapFiller.fill()."""

    def test_no_gaps(self):
        """Contigs without gaps → nothing changes."""
        seq = "ACGTACGTACGTACGTACGT"  # No Ns
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        reads = [MockSeqRead(id="r1", sequence=seq)]
        filler = GapFiller(k=5)
        filled, summary = filler.fill(contigs, reads)
        assert summary.total_gaps == 0
        assert summary.gaps_filled == 0
        assert filled[0].sequence == seq

    def test_empty_contigs(self):
        filler = GapFiller(k=5)
        filled, summary = filler.fill([], [])
        assert summary.total_gaps == 0

    def test_gap_too_large_skipped(self):
        """Gaps exceeding max_gap_size are skipped."""
        gap = "N" * 200
        seq = "ACGTACGTACGT" + gap + "ACGTACGTACGT"
        contigs = [MockSeqRead(id="c1", sequence=seq)]
        reads = [MockSeqRead(id="r1", sequence="ACGTACGTACGT")]
        filler = GapFiller(k=5, max_gap_size=100)
        filled, summary = filler.fill(contigs, reads)
        assert summary.gaps_unfilled == 1
        assert summary.gaps_filled == 0

    def test_gap_filled_with_spanning_reads(self):
        """
        A small gap (5 Ns) with reads that span both flanks and cover
        the gap region → gap should be filled.
        """
        left = "AACCGGTTAACC"   # 12 bp
        right = "TTGGCCAATTGG"  # 12 bp
        bridge = "ACGTG"        # 5 bp that should fill the gap
        gap_seq = left + "NNNNN" + right   # 29 bp with 5-bp gap
        full_seq = left + bridge + right   # 29 bp, correct sequence

        contigs = [MockSeqRead(id="c1", sequence=gap_seq)]
        # 5 reads that span the full region
        reads = [MockSeqRead(id=f"r{i}", sequence=full_seq) for i in range(5)]

        filler = GapFiller(k=5, min_spanning_reads=3, min_fill_confidence=0.3)
        filled, summary = filler.fill(contigs, reads)

        # The gap should be detected
        assert summary.total_gaps >= 1
        # Check that the N-run is gone (or at least reduced)
        n_count = filled[0].sequence.count('N')
        assert n_count < 5  # Some or all Ns should be filled

    def test_insufficient_spanning_reads(self):
        """Gap with only 1 spanning read (need 3) → not filled."""
        gap_seq = "AACCGGTTAACC" + "NNNNN" + "TTGGCCAATTGG"
        full_seq = "AACCGGTTAACCACGTGTTGGCCAATTGG"

        contigs = [MockSeqRead(id="c1", sequence=gap_seq)]
        reads = [MockSeqRead(id="r1", sequence=full_seq)]

        filler = GapFiller(k=5, min_spanning_reads=3)
        filled, summary = filler.fill(contigs, reads)
        assert summary.gaps_filled == 0

    def test_extract_gap_fill_shorter(self):
        """If consensus shorter than gap, use full consensus."""
        assert GapFiller._extract_gap_fill("ACG", 10) == "ACG"

    def test_extract_gap_fill_longer(self):
        """If consensus longer than gap, centre-crop."""
        result = GapFiller._extract_gap_fill("AACCGGTT", 4)
        assert len(result) == 4
        # Centre crop of "AACCGGTT" (len 8) for gap 4:
        # excess=4, trim_left=2 → "CCGG"
        assert result == "CCGG"

    def test_extract_gap_fill_exact(self):
        """Exact length → no trimming."""
        assert GapFiller._extract_gap_fill("ACGT", 4) == "ACGT"


class TestGapFillerInit:
    """Test GapFiller construction."""

    def test_defaults(self):
        f = GapFiller()
        assert f.max_gap_size == 10000
        assert f.k == 21

    def test_custom(self):
        f = GapFiller(max_gap_size=5000, k=31, min_spanning_reads=5)
        assert f.max_gap_size == 5000
        assert f.min_spanning_reads == 5

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
