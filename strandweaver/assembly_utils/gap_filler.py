#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Gap Filler — detects N-gap runs in contigs and attempts to fill them
using spanning reads via local consensus.

Algorithm:
  1. Scan each contig for runs of N characters (gaps).
  2. For each gap, collect reads whose k-mer anchors span both flanks.
  3. Extract the read sub-sequences that bridge the gap region.
  4. Build a quality-weighted consensus of the bridging segments.
  5. Replace N-runs with consensus if coverage and confidence thresholds
     are met; otherwise leave the gap unchanged.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GapRegion:
    """A contiguous run of Ns in a contig."""
    contig_id: str
    start: int           # 0-based inclusive start of N-run
    end: int             # 0-based exclusive end of N-run
    size: int            # end - start
    filled: bool = False
    fill_sequence: Optional[str] = None
    fill_confidence: float = 0.0
    spanning_reads: int = 0


@dataclass
class GapFillingSummary:
    """Statistics from gap-filling."""
    total_gaps: int
    gaps_filled: int
    gaps_unfilled: int
    bases_filled: int            # Total N bases replaced
    bases_inserted: int          # Total bases in fill sequences (may differ from gap size)
    mean_fill_confidence: float
    per_gap: List[GapRegion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAP_PATTERN = re.compile(r'[Nn]+')

FLANK_SIZE = 500   # bp of flanking sequence used for read anchoring


def _find_gaps(sequence: str, min_gap_size: int = 1) -> List[Tuple[int, int]]:
    """
    Find all N-runs (gaps) in *sequence*.

    Returns:
        List of (start, end) tuples (0-based, end exclusive).
    """
    gaps = []
    for m in _GAP_PATTERN.finditer(sequence):
        if m.end() - m.start() >= min_gap_size:
            gaps.append((m.start(), m.end()))
    return gaps


def _build_kmer_set(sequence: str, k: int) -> set:
    """Return a set of k-mers in *sequence* (uppercase, skip Ns)."""
    s = set()
    seq_upper = sequence.upper()
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if 'N' not in kmer:
            s.add(kmer)
    return s


def _find_spanning_reads(
    contig_seq: str,
    gap_start: int,
    gap_end: int,
    reads: List[Any],
    k: int,
    flank_size: int = FLANK_SIZE,
) -> List[Tuple[str, List[int]]]:
    """
    Identify reads that anchor on both flanks of a gap.

    Returns:
        List of (bridging_subsequence, quality_scores) for the region
        that covers the gap.
    """
    contig_len = len(contig_seq)

    # Build k-mer sets for the left and right flanks
    left_start = max(0, gap_start - flank_size)
    left_flank = contig_seq[left_start:gap_start].upper()
    right_end = min(contig_len, gap_end + flank_size)
    right_flank = contig_seq[gap_end:right_end].upper()

    left_kmers = _build_kmer_set(left_flank, k)
    right_kmers = _build_kmer_set(right_flank, k)

    if not left_kmers or not right_kmers:
        return []

    spanning: List[Tuple[str, List[int]]] = []

    for read in reads:
        read_upper = read.sequence.upper()
        read_len = len(read_upper)
        if read_len < k:
            continue

        # Count anchor hits on each flank
        left_hits = 0
        right_hits = 0
        # Track positions of first left-hit and last right-hit within the read
        first_left_pos: Optional[int] = None
        last_right_pos: Optional[int] = None

        for i in range(read_len - k + 1):
            kmer = read_upper[i:i + k]
            if 'N' in kmer:
                continue
            if kmer in left_kmers:
                left_hits += 1
                if first_left_pos is None:
                    first_left_pos = i
            if kmer in right_kmers:
                right_hits += 1
                last_right_pos = i + k  # end of the right anchor

        # Need at least 2 hits on each flank
        if left_hits < 2 or right_hits < 2:
            continue
        if first_left_pos is None or last_right_pos is None:
            continue
        if first_left_pos >= last_right_pos:
            continue

        # Extract the bridging sub-sequence (between left anchor start
        # and right anchor end)
        bridge_seq = read.sequence[first_left_pos:last_right_pos]
        if read.quality and len(read.quality) == len(read.sequence):
            bridge_qual = [
                ord(q) - 33
                for q in read.quality[first_left_pos:last_right_pos]
            ]
        else:
            bridge_qual = [20] * len(bridge_seq)

        spanning.append((bridge_seq, bridge_qual))

    return spanning


def _consensus_from_bridges(
    bridges: List[Tuple[str, List[int]]],
    gap_size: int,
    consensus_threshold: float = 0.6,
) -> Tuple[Optional[str], float]:
    """
    Build a consensus sequence from bridging read segments.

    The bridges may differ in length; we use a simple column-wise majority
    vote aligned from the left. The final consensus is trimmed to remove
    the flanking anchors and return only the gap-filling portion.

    Args:
        bridges: List of (sequence, quality_scores).
        gap_size: Size of the original N-run (used as target length hint).
        consensus_threshold: Minimum weighted fraction for consensus call.

    Returns:
        (consensus_sequence, mean_confidence) or (None, 0.0) on failure.
    """
    if not bridges:
        return None, 0.0

    # Use the median bridge length as the target
    lengths = [len(seq) for seq, _ in bridges]
    target_len = int(np.median(lengths))

    if target_len == 0:
        return None, 0.0

    consensus_chars: List[str] = []
    confidences: List[float] = []

    for pos in range(target_len):
        base_scores: Dict[str, float] = defaultdict(float)
        for seq, quals in bridges:
            if pos < len(seq):
                base = seq[pos].upper()
                if base in 'ACGT':
                    score = 10.0 ** (quals[pos] / 10.0) if pos < len(quals) else 1.0
                    base_scores[base] += score

        total = sum(base_scores.values())
        if total == 0:
            consensus_chars.append('N')
            confidences.append(0.0)
            continue

        best_base = max(base_scores, key=base_scores.get)  # type: ignore[arg-type]
        confidence = base_scores[best_base] / total

        if confidence >= consensus_threshold:
            consensus_chars.append(best_base)
        else:
            consensus_chars.append('N')
        confidences.append(confidence)

    consensus = ''.join(consensus_chars)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return consensus, mean_conf


# ---------------------------------------------------------------------------
# Main gap filler
# ---------------------------------------------------------------------------

class GapFiller:
    """
    Assembly gap filler using spanning-read consensus.

    Scans contigs for N-runs, finds reads that span each gap, builds
    consensus, and replaces the Ns.

    Usage
    -----
    >>> filler = GapFiller(max_gap_size=10000)
    >>> contigs, summary = filler.fill(contigs, reads)
    """

    def __init__(
        self,
        max_gap_size: int = 10000,
        min_overlap_to_gap: int = 100,
        k: int = 21,
        *,
        min_spanning_reads: int = 3,
        consensus_threshold: float = 0.6,
        min_fill_confidence: float = 0.5,
    ):
        """
        Args:
            max_gap_size: Ignore gaps larger than this (bp).
            min_overlap_to_gap: Minimum read overlap on each flank.
            k: K-mer size for flank anchoring.
            min_spanning_reads: Minimum number of spanning reads required.
            consensus_threshold: Minimum fraction for column consensus.
            min_fill_confidence: Overall confidence threshold to accept fill.
        """
        self.max_gap_size = max_gap_size
        self.min_overlap_to_gap = min_overlap_to_gap
        self.k = k
        self.min_spanning_reads = min_spanning_reads
        self.consensus_threshold = consensus_threshold
        self.min_fill_confidence = min_fill_confidence

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fill(
        self,
        contigs: List[Any],
        reads: List[Any],
    ) -> Tuple[List[Any], GapFillingSummary]:
        """
        Fill gaps in *contigs* using *reads*.

        Args:
            contigs: List of SeqRead-like objects (.id, .sequence).
            reads: List of SeqRead-like objects.

        Returns:
            (gap_filled_contigs, GapFillingSummary)
        """
        if not contigs:
            return contigs, GapFillingSummary(
                total_gaps=0, gaps_filled=0, gaps_unfilled=0,
                bases_filled=0, bases_inserted=0,
                mean_fill_confidence=0.0,
            )

        logger.info(
            "Starting gap filling: max_gap=%d  k=%d  "
            "min_spanning=%d  threshold=%.2f",
            self.max_gap_size, self.k,
            self.min_spanning_reads, self.consensus_threshold,
        )

        all_gaps: List[GapRegion] = []
        total_filled = 0
        total_unfilled = 0
        total_bases_filled = 0
        total_bases_inserted = 0
        fill_confidences: List[float] = []

        for contig in contigs:
            gaps = _find_gaps(contig.sequence, min_gap_size=1)
            if not gaps:
                continue

            logger.debug(
                "  Contig %s: %d gaps detected", contig.id, len(gaps)
            )

            # Process gaps in reverse order so earlier indices remain valid
            # after sequence replacement
            seq_list = list(contig.sequence)
            for gap_start, gap_end in reversed(gaps):
                gap_size = gap_end - gap_start
                gap_region = GapRegion(
                    contig_id=contig.id,
                    start=gap_start,
                    end=gap_end,
                    size=gap_size,
                )

                if gap_size > self.max_gap_size:
                    logger.debug(
                        "    Gap %d-%d (%d bp) exceeds max_gap_size — skipping",
                        gap_start, gap_end, gap_size,
                    )
                    all_gaps.append(gap_region)
                    total_unfilled += 1
                    continue

                # Find spanning reads
                spanning = _find_spanning_reads(
                    contig.sequence, gap_start, gap_end,
                    reads, self.k,
                    flank_size=max(self.min_overlap_to_gap, FLANK_SIZE),
                )
                gap_region.spanning_reads = len(spanning)

                if len(spanning) < self.min_spanning_reads:
                    logger.debug(
                        "    Gap %d-%d: only %d spanning reads (need %d) — skipping",
                        gap_start, gap_end, len(spanning), self.min_spanning_reads,
                    )
                    all_gaps.append(gap_region)
                    total_unfilled += 1
                    continue

                # Build consensus
                consensus, confidence = _consensus_from_bridges(
                    spanning, gap_size,
                    consensus_threshold=self.consensus_threshold,
                )

                if consensus is None or confidence < self.min_fill_confidence:
                    logger.debug(
                        "    Gap %d-%d: low confidence (%.3f) — skipping",
                        gap_start, gap_end, confidence,
                    )
                    all_gaps.append(gap_region)
                    total_unfilled += 1
                    continue

                # Accept the fill — replace N-run in seq_list
                # The consensus covers the bridging region (flanks + gap).
                # We only want the portion that replaces the Ns.
                # Since _find_spanning_reads returns sub-sequences from
                # left-anchor start to right-anchor end, the consensus
                # length may exceed gap_size. We take the central portion.
                fill_seq = self._extract_gap_fill(consensus, gap_size)

                seq_list[gap_start:gap_end] = list(fill_seq)

                gap_region.filled = True
                gap_region.fill_sequence = fill_seq
                gap_region.fill_confidence = confidence
                all_gaps.append(gap_region)

                total_filled += 1
                total_bases_filled += gap_size
                total_bases_inserted += len(fill_seq)
                fill_confidences.append(confidence)

                logger.debug(
                    "    Gap %d-%d filled: %d N → %d bp (conf=%.3f, %d reads)",
                    gap_start, gap_end, gap_size, len(fill_seq),
                    confidence, len(spanning),
                )

            # Reconstruct contig sequence
            contig.sequence = ''.join(seq_list)

        mean_conf = float(np.mean(fill_confidences)) if fill_confidences else 0.0

        summary = GapFillingSummary(
            total_gaps=total_filled + total_unfilled,
            gaps_filled=total_filled,
            gaps_unfilled=total_unfilled,
            bases_filled=total_bases_filled,
            bases_inserted=total_bases_inserted,
            mean_fill_confidence=mean_conf,
            per_gap=all_gaps,
        )

        logger.info(
            "Gap filling complete: %d/%d gaps filled  "
            "%d N-bases → %d bp  mean_conf=%.3f",
            summary.gaps_filled, summary.total_gaps,
            summary.bases_filled, summary.bases_inserted,
            summary.mean_fill_confidence,
        )

        return contigs, summary

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_gap_fill(consensus: str, gap_size: int) -> str:
        """
        Extract the portion of the consensus that replaces the N-run.

        If the consensus is longer than the gap (because it includes
        flanking anchors), take the central *gap_size* characters.
        If shorter, use the full consensus.
        """
        if len(consensus) <= gap_size:
            return consensus

        # Centre-crop
        excess = len(consensus) - gap_size
        trim_left = excess // 2
        return consensus[trim_left:trim_left + gap_size]


__all__ = [
    "GapFiller",
    "GapRegion",
    "GapFillingSummary",
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
