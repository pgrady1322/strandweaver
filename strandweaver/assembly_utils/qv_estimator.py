#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

QV Estimation — Merqury-style k-mer completeness quality value (QV) scoring,
per-contig error-rate QV, and assembly quality reporting.

Provides three estimation modes:
  1. K-mer completeness QV: Counts k-mers in reads vs assembly to estimate
     consensus accuracy (Merqury approach: Rhie et al., 2020).
  2. Per-contig QV from ErrorSmith error predictions: Aggregates per-base
     error probabilities into per-contig and whole-assembly QV scores.
  3. Combined report: JSON output with global QV, per-contig QV, and
     k-mer completeness metrics.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContigQV:
    """Quality metrics for a single contig."""
    contig_id: str
    length: int
    kmer_completeness: float      # Fraction of read k-mers found (0-1)
    kmer_qv: float                # -10 log10(1 - completeness), capped at 70
    error_rate_qv: float          # -10 log10(error_rate), from ErrorSmith
    combined_qv: float            # Weighted mean of kmer_qv and error_rate_qv
    num_errors_estimated: int     # Estimated erroneous bases
    gc_content: float             # GC fraction of this contig


@dataclass
class AssemblyQV:
    """Whole-assembly quality summary."""
    global_kmer_completeness: float
    global_kmer_qv: float
    global_error_rate_qv: float
    global_combined_qv: float
    total_bases: int
    num_contigs: int
    n50: int
    per_contig: List[ContigQV] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "global_kmer_completeness": round(self.global_kmer_completeness, 6),
            "global_kmer_qv": round(self.global_kmer_qv, 2),
            "global_error_rate_qv": round(self.global_error_rate_qv, 2),
            "global_combined_qv": round(self.global_combined_qv, 2),
            "total_bases": self.total_bases,
            "num_contigs": self.num_contigs,
            "n50": self.n50,
            "per_contig": [
                {
                    "contig_id": c.contig_id,
                    "length": c.length,
                    "kmer_completeness": round(c.kmer_completeness, 6),
                    "kmer_qv": round(c.kmer_qv, 2),
                    "error_rate_qv": round(c.error_rate_qv, 2),
                    "combined_qv": round(c.combined_qv, 2),
                    "num_errors_estimated": c.num_errors_estimated,
                    "gc_content": round(c.gc_content, 4),
                }
                for c in self.per_contig
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_QV = 70.0  # Cap QV at Q70 (1 error per 10 million bases)


def _qv_from_error_rate(error_rate: float) -> float:
    """Convert an error rate to a Phred-scale QV, capped at _MAX_QV."""
    if error_rate <= 0:
        return _MAX_QV
    qv = -10.0 * math.log10(error_rate)
    return min(qv, _MAX_QV)


def _qv_from_completeness(completeness: float) -> float:
    """Convert k-mer completeness to QV: -10 log10(1 - completeness)."""
    missing = 1.0 - completeness
    if missing <= 0:
        return _MAX_QV
    qv = -10.0 * math.log10(missing)
    return min(qv, _MAX_QV)


def _gc_content(seq: str) -> float:
    """Return GC fraction of a DNA sequence."""
    if not seq:
        return 0.0
    gc = seq.count('G') + seq.count('C') + seq.count('g') + seq.count('c')
    return gc / len(seq)


def _compute_n50(lengths: List[int]) -> int:
    """Compute N50 from a list of contig lengths."""
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    cumulative = 0
    for length in sorted_lengths:
        cumulative += length
        if cumulative >= total / 2:
            return length
    return sorted_lengths[-1]


def _extract_kmers(sequence: str, k: int) -> Counter:
    """Extract canonical k-mers from a sequence (forward + reverse complement)."""
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    counts: Counter = Counter()
    seq_upper = sequence.upper()
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if 'N' in kmer:
            continue
        rc = kmer.translate(complement)[::-1]
        canonical = min(kmer, rc)
        counts[canonical] += 1
    return counts


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class QVEstimator:
    """
    Assembly quality-value estimator.

    Supports three estimation strategies that can be used independently
    or combined:

      * **K-mer completeness** (Merqury-style): requires raw reads.
      * **Error-rate QV**: requires per-base or per-contig error rates
        (typically from ErrorSmith predictions).
      * **Combined QV**: weighted harmonic mean of both signals.

    Usage
    -----
    >>> estimator = QVEstimator(k=21)
    >>> report = estimator.estimate(contigs, reads=reads)
    >>> report.global_combined_qv
    42.5
    """

    def __init__(
        self,
        k: int = 21,
        *,
        kmer_weight: float = 0.6,
        error_rate_weight: float = 0.4,
        min_contig_length: int = 1000,
        default_error_rate: float = 0.001,
    ):
        """
        Args:
            k: K-mer size for completeness estimation.
            kmer_weight: Weight for k-mer QV in the combined score (0-1).
            error_rate_weight: Weight for error-rate QV (0-1).
            min_contig_length: Ignore contigs shorter than this for QV.
            default_error_rate: Assumed error rate when no ErrorSmith
                predictions are available.
        """
        if k < 11 or k > 101:
            raise ValueError(f"k must be between 11 and 101, got {k}")
        if not math.isclose(kmer_weight + error_rate_weight, 1.0, abs_tol=1e-6):
            raise ValueError("kmer_weight + error_rate_weight must equal 1.0")

        self.k = k
        self.kmer_weight = kmer_weight
        self.error_rate_weight = error_rate_weight
        self.min_contig_length = min_contig_length
        self.default_error_rate = default_error_rate

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def estimate(
        self,
        contigs: List[Any],
        *,
        reads: Optional[List[Any]] = None,
        error_rates: Optional[Dict[str, float]] = None,
    ) -> AssemblyQV:
        """
        Estimate assembly QV.

        Args:
            contigs: List of SeqRead (or any object with .id and .sequence).
            reads: Optional list of raw reads (SeqRead) for k-mer
                completeness. If None, k-mer QV defaults to 0.
            error_rates: Optional dict mapping contig_id → per-base error
                rate (float, e.g. 0.0001). If None, *default_error_rate*
                is used for every contig.

        Returns:
            AssemblyQV with global and per-contig quality metrics.
        """
        # Filter short contigs
        filtered = [c for c in contigs if len(c.sequence) >= self.min_contig_length]
        if not filtered:
            logger.warning("No contigs >= %d bp — returning empty QV report",
                           self.min_contig_length)
            return AssemblyQV(
                global_kmer_completeness=0.0,
                global_kmer_qv=0.0,
                global_error_rate_qv=0.0,
                global_combined_qv=0.0,
                total_bases=0,
                num_contigs=0,
                n50=0,
            )

        logger.info("Estimating QV for %d contigs (k=%d)", len(filtered), self.k)

        # --- K-mer completeness (per contig) ---
        per_contig_completeness: Dict[str, float] = {}
        if reads is not None:
            per_contig_completeness = self._kmer_completeness(filtered, reads)
        else:
            logger.info("No reads provided — k-mer QV will use default (0)")

        # --- Per-contig QV ---
        contig_qvs: List[ContigQV] = []
        for contig in filtered:
            cid = contig.id
            seq = contig.sequence
            length = len(seq)

            # K-mer completeness
            completeness = per_contig_completeness.get(cid, 0.0)
            kqv = _qv_from_completeness(completeness) if reads is not None else 0.0

            # Error-rate QV
            err = (error_rates or {}).get(cid, self.default_error_rate)
            eqv = _qv_from_error_rate(err)

            # Combined (weighted mean; skip kmer term if no reads)
            if reads is not None:
                combined = self.kmer_weight * kqv + self.error_rate_weight * eqv
            else:
                combined = eqv  # Fall back to error-rate QV only

            num_errors = max(0, round(err * length))

            contig_qvs.append(ContigQV(
                contig_id=cid,
                length=length,
                kmer_completeness=completeness,
                kmer_qv=kqv,
                error_rate_qv=eqv,
                combined_qv=combined,
                num_errors_estimated=num_errors,
                gc_content=_gc_content(seq),
            ))

        # --- Global metrics (length-weighted) ---
        total_bases = sum(c.length for c in contig_qvs)
        lengths = [c.length for c in contig_qvs]

        if total_bases > 0:
            global_kmer_comp = sum(
                c.kmer_completeness * c.length for c in contig_qvs
            ) / total_bases
            global_kmer_qv = sum(
                c.kmer_qv * c.length for c in contig_qvs
            ) / total_bases
            global_err_qv = sum(
                c.error_rate_qv * c.length for c in contig_qvs
            ) / total_bases
            global_combined = sum(
                c.combined_qv * c.length for c in contig_qvs
            ) / total_bases
        else:
            global_kmer_comp = global_kmer_qv = global_err_qv = global_combined = 0.0

        result = AssemblyQV(
            global_kmer_completeness=global_kmer_comp,
            global_kmer_qv=global_kmer_qv,
            global_error_rate_qv=global_err_qv,
            global_combined_qv=global_combined,
            total_bases=total_bases,
            num_contigs=len(contig_qvs),
            n50=_compute_n50(lengths),
            per_contig=contig_qvs,
        )

        logger.info(
            "QV estimate complete: global_combined=Q%.1f  kmer=Q%.1f  "
            "error_rate=Q%.1f  contigs=%d  bases=%s",
            result.global_combined_qv,
            result.global_kmer_qv,
            result.global_error_rate_qv,
            result.num_contigs,
            f"{result.total_bases:,}",
        )
        return result

    def save_report(self, qv: AssemblyQV, path: Path) -> None:
        """Write QV report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(qv.to_dict(), fh, indent=2)
        logger.info("QV report saved to %s", path)

    # -----------------------------------------------------------------
    # K-mer completeness (internal)
    # -----------------------------------------------------------------

    def _kmer_completeness(
        self,
        contigs: List[Any],
        reads: List[Any],
    ) -> Dict[str, float]:
        """
        Compute per-contig k-mer completeness (Merqury-style).

        For each contig, measure what fraction of its distinct k-mers
        also appear in the read set (true k-mers). K-mers that appear
        in the assembly but not in reads are presumed erroneous.

        Returns:
            Dict mapping contig_id → completeness (0-1).
        """
        k = self.k
        logger.info("Building read k-mer set (k=%d) from %d reads …", k, len(reads))

        # Build global set of "true" k-mers from reads
        # Use a set (not Counter) — we only need presence, not count
        read_kmers: set = set()
        total_read_bases = 0
        for read in reads:
            total_read_bases += len(read.sequence)
            seq_upper = read.sequence.upper()
            complement = str.maketrans('ACGTacgt', 'TGCAtgca')
            for i in range(len(seq_upper) - k + 1):
                kmer = seq_upper[i:i + k]
                if 'N' in kmer:
                    continue
                rc = kmer.translate(complement)[::-1]
                read_kmers.add(min(kmer, rc))

        logger.info(
            "Read k-mer set: %s distinct canonical k-mers from %s bases",
            f"{len(read_kmers):,}",
            f"{total_read_bases:,}",
        )

        # Evaluate each contig
        results: Dict[str, float] = {}
        for contig in contigs:
            asm_kmers = _extract_kmers(contig.sequence, k)
            if not asm_kmers:
                results[contig.id] = 0.0
                continue

            total_distinct = len(asm_kmers)
            found = sum(1 for km in asm_kmers if km in read_kmers)
            completeness = found / total_distinct
            results[contig.id] = completeness

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def estimate_assembly_qv(
    contigs: List[Any],
    *,
    reads: Optional[List[Any]] = None,
    error_rates: Optional[Dict[str, float]] = None,
    k: int = 21,
    output_path: Optional[Path] = None,
) -> AssemblyQV:
    """
    One-call convenience wrapper around QVEstimator.

    Args:
        contigs: List of SeqRead-like objects (.id, .sequence).
        reads: Optional reads for k-mer completeness.
        error_rates: Optional per-contig error rates.
        k: K-mer size (default 21).
        output_path: If given, writes JSON report to this path.

    Returns:
        AssemblyQV result.
    """
    estimator = QVEstimator(k=k)
    result = estimator.estimate(contigs, reads=reads, error_rates=error_rates)
    if output_path is not None:
        estimator.save_report(result, output_path)
    return result


__all__ = [
    "QVEstimator",
    "ContigQV",
    "AssemblyQV",
    "estimate_assembly_qv",
]

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
