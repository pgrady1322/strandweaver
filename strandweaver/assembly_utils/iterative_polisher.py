#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Iterative polisher — multi-round consensus sequence correction.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolishingStats:
    """Statistics from one polishing round."""
    round_number: int
    bases_corrected: int
    total_bases: int
    correction_rate: float        # bases_corrected / total_bases
    mean_coverage: float          # Average read depth across all contigs
    qv_before: float
    qv_after: float
    qv_delta: float               # qv_after - qv_before
    contigs_improved: int
    contigs_unchanged: int


@dataclass
class PolishingSummary:
    """Summary across all polishing rounds."""
    total_rounds: int
    total_bases_corrected: int
    final_qv: float
    initial_qv: float
    qv_improvement: float
    converged: bool               # True if stopped early due to convergence
    per_round: List[PolishingStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# K-mer anchor index
# ---------------------------------------------------------------------------

def _build_kmer_index(sequence: str, k: int) -> Dict[str, List[int]]:
    """Build a dict mapping each k-mer in *sequence* to its start positions."""
    index: Dict[str, List[int]] = defaultdict(list)
    seq_upper = sequence.upper()
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if 'N' not in kmer:
            index[kmer].append(i)
    return index


def _find_best_anchor(
    read_seq: str,
    contig_index: Dict[str, List[int]],
    k: int,
    contig_len: int,
) -> Optional[int]:
    """
    Find the best alignment offset for *read_seq* against the contig
    represented by *contig_index*.

    Returns the estimated offset of the read start relative to the contig,
    or None if no anchor is found.
    """
    votes: Counter = Counter()
    read_upper = read_seq.upper()
    for i in range(0, len(read_upper) - k + 1, max(1, k // 2)):
        kmer = read_upper[i:i + k]
        if kmer in contig_index:
            for pos in contig_index[kmer]:
                offset = pos - i
                votes[offset] += 1

    if not votes:
        return None

    best_offset, count = votes.most_common(1)[0]
    # Require at least 2 anchor hits for a credible mapping
    if count < 2:
        return None
    # Offset must place at least part of the read on the contig
    if best_offset + len(read_upper) <= 0 or best_offset >= contig_len:
        return None
    return best_offset


# ---------------------------------------------------------------------------
# Pileup & consensus
# ---------------------------------------------------------------------------

def _build_pileup(
    contig_seq: str,
    reads: List[Any],
    k: int,
) -> List[List[Tuple[str, int]]]:
    """
    Build a per-position pileup of (base, quality) tuples.

    Returns a list of length len(contig_seq) where each element is a list
    of (base, phred_score) tuples from aligned reads.
    """
    contig_len = len(contig_seq)
    pileup: List[List[Tuple[str, int]]] = [[] for _ in range(contig_len)]
    contig_index = _build_kmer_index(contig_seq, k)

    for read in reads:
        offset = _find_best_anchor(read.sequence, contig_index, k, contig_len)
        if offset is None:
            continue

        read_upper = read.sequence.upper()
        # Determine quality scores
        if read.quality and len(read.quality) == len(read.sequence):
            quals = [ord(q) - 33 for q in read.quality]
        else:
            # Default Q20 when quality absent
            quals = [20] * len(read.sequence)

        for ri in range(len(read_upper)):
            ci = offset + ri
            if 0 <= ci < contig_len:
                pileup[ci].append((read_upper[ri], quals[ri]))

    return pileup


def _call_consensus(
    pileup: List[List[Tuple[str, int]]],
    original_seq: str,
    min_coverage: int = 3,
    consensus_threshold: float = 0.6,
) -> Tuple[str, int]:
    """
    Call consensus from a pileup, returning (new_sequence, bases_changed).

    Positions with coverage < *min_coverage* keep the original base.
    """
    result = list(original_seq.upper())
    bases_changed = 0

    for i, column in enumerate(pileup):
        if len(column) < min_coverage:
            continue

        # Quality-weighted vote
        base_scores: Dict[str, float] = defaultdict(float)
        for base, qual in column:
            base_scores[base] += 10.0 ** (qual / 10.0)

        total = sum(base_scores.values())
        if total == 0:
            continue

        best_base = max(base_scores, key=base_scores.get)  # type: ignore[arg-type]
        confidence = base_scores[best_base] / total

        if confidence >= consensus_threshold and best_base != result[i]:
            result[i] = best_base
            bases_changed += 1

    return ''.join(result), bases_changed


# ---------------------------------------------------------------------------
# Main polisher
# ---------------------------------------------------------------------------

class IterativePolisher:
    """
    Consensus-based iterative assembly polisher.

    Performs multiple rounds of read-to-contig pileup consensus correction,
    using the QVEstimator to track quality improvement and stop when
    converged.

    Usage
    -----
    >>> polisher = IterativePolisher(max_rounds=3, k=21)
    >>> contigs, summary = polisher.polish(contigs, reads)
    >>> print(summary.final_qv)
    """

    def __init__(
        self,
        max_rounds: int = 2,
        k: int = 21,
        *,
        min_coverage: int = 3,
        consensus_threshold: float = 0.6,
        convergence_delta: float = 0.1,
        use_qv_convergence: bool = True,
    ):
        """
        Args:
            max_rounds: Maximum polishing rounds.
            k: K-mer size for read-to-contig anchoring.
            min_coverage: Minimum pileup depth to call consensus.
            consensus_threshold: Fraction of weighted evidence required.
            convergence_delta: Stop if QV improves by less than this.
            use_qv_convergence: Enable early stopping on QV plateau.
        """
        self.max_rounds = max(1, max_rounds)
        self.k = k
        self.min_coverage = min_coverage
        self.consensus_threshold = consensus_threshold
        self.convergence_delta = convergence_delta
        self.use_qv_convergence = use_qv_convergence

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def polish(
        self,
        contigs: List[Any],
        reads: List[Any],
        *,
        qv_estimator: Optional[Any] = None,
    ) -> Tuple[List[Any], PolishingSummary]:
        """
        Iteratively polish *contigs* using *reads*.

        Args:
            contigs: List of SeqRead-like objects (.id, .sequence, writable).
            reads: List of SeqRead-like objects used for pileup.
            qv_estimator: Optional QVEstimator instance. If None, a default
                one is created internally.

        Returns:
            (polished_contigs, PolishingSummary)
        """
        if not contigs:
            return contigs, PolishingSummary(
                total_rounds=0, total_bases_corrected=0,
                final_qv=0.0, initial_qv=0.0, qv_improvement=0.0,
                converged=True,
            )

        # Lazy-import QVEstimator to avoid circular imports
        if qv_estimator is None:
            from .qv_estimator import QVEstimator
            qv_estimator = QVEstimator(k=self.k, min_contig_length=0)

        logger.info(
            "Starting iterative polishing: max_rounds=%d  k=%d  "
            "min_cov=%d  threshold=%.2f",
            self.max_rounds, self.k,
            self.min_coverage, self.consensus_threshold,
        )

        # Initial QV
        initial_qv_result = qv_estimator.estimate(contigs, reads=reads)
        prev_qv = initial_qv_result.global_combined_qv
        initial_qv = prev_qv

        round_stats: List[PolishingStats] = []
        total_corrected = 0
        converged = False

        for rnd in range(1, self.max_rounds + 1):
            logger.info("── Polishing round %d/%d ──", rnd, self.max_rounds)

            round_corrected = 0
            contigs_improved = 0
            contigs_unchanged = 0
            coverage_values: List[float] = []

            for ci, contig in enumerate(contigs):
                pileup = _build_pileup(contig.sequence, reads, self.k)

                # Mean coverage for this contig
                covs = [len(col) for col in pileup]
                mean_cov = float(np.mean(covs)) if covs else 0.0
                coverage_values.append(mean_cov)

                new_seq, changed = _call_consensus(
                    pileup, contig.sequence,
                    min_coverage=self.min_coverage,
                    consensus_threshold=self.consensus_threshold,
                )

                if changed > 0:
                    contig.sequence = new_seq
                    round_corrected += changed
                    contigs_improved += 1
                else:
                    contigs_unchanged += 1

            total_bases = sum(len(c.sequence) for c in contigs)
            total_corrected += round_corrected

            # Measure QV after this round
            post_qv_result = qv_estimator.estimate(contigs, reads=reads)
            post_qv = post_qv_result.global_combined_qv
            qv_delta = post_qv - prev_qv

            stats = PolishingStats(
                round_number=rnd,
                bases_corrected=round_corrected,
                total_bases=total_bases,
                correction_rate=round_corrected / total_bases if total_bases else 0.0,
                mean_coverage=float(np.mean(coverage_values)) if coverage_values else 0.0,
                qv_before=prev_qv,
                qv_after=post_qv,
                qv_delta=qv_delta,
                contigs_improved=contigs_improved,
                contigs_unchanged=contigs_unchanged,
            )
            round_stats.append(stats)

            logger.info(
                "  Round %d: %d bases corrected (%.4f%%)  "
                "QV %.1f → %.1f (Δ%.2f)  cov=%.1fx",
                rnd, round_corrected,
                stats.correction_rate * 100,
                prev_qv, post_qv, qv_delta,
                stats.mean_coverage,
            )

            prev_qv = post_qv

            # Convergence check
            if self.use_qv_convergence and rnd > 1:
                if abs(qv_delta) < self.convergence_delta:
                    logger.info(
                        "  Converged after %d rounds (Δ%.3f < %.3f)",
                        rnd, abs(qv_delta), self.convergence_delta,
                    )
                    converged = True
                    break

            # Also stop if zero corrections
            if round_corrected == 0:
                logger.info("  No bases corrected — stopping early")
                converged = True
                break

        summary = PolishingSummary(
            total_rounds=len(round_stats),
            total_bases_corrected=total_corrected,
            final_qv=prev_qv,
            initial_qv=initial_qv,
            qv_improvement=prev_qv - initial_qv,
            converged=converged,
            per_round=round_stats,
        )

        logger.info(
            "Polishing complete: %d rounds, %d bases corrected, "
            "QV %.1f → %.1f (+%.2f)%s",
            summary.total_rounds,
            summary.total_bases_corrected,
            summary.initial_qv,
            summary.final_qv,
            summary.qv_improvement,
            " [converged]" if converged else "",
        )

        return contigs, summary


__all__ = [
    "IterativePolisher",
    "PolishingStats",
    "PolishingSummary",
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
