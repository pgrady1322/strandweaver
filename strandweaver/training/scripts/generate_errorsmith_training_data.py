#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver — ErrorSmith Training Data Generator (CHM13-based)

Generates training CSVs for the ErrorSmith XGBoost error classifier by
processing real sequencing data aligned to the T2T-CHM13 reference:

  1. Downloads (or locates) public HiFi, ONT, and Illumina BAMs aligned
     to CHM13v2.0
  2. Parses CIGAR strings to identify per-base error positions and types
  3. Extracts local features per base: k-mer context, quality, GC content,
     homopolymer length, position in read, technology
  4. Labels each base: correct / substitution / insertion / deletion /
     homopolymer_error
  5. Exports balanced training CSV (subsample correct class to avoid >99%
     majority)

Designed for Colab with GPU instances.
Run time: ~1-3 h depending on subsample size.

Data sources (public):
  - HiFi:     SRR19325710 (T2T CHM13 35× PacBio HiFi)
  - ONT R10:  SRR23571061 (T2T CHM13 ONT R10.4.1)
  - Illumina: SRR3189741  (CHM13 Illumina PCR-free 2×150 bp)

Usage:
    # Full run with automatic download
    python scripts/generate_errorsmith_training_data.py \\
        --output training_output/errorsmith \\
        --reference /path/to/chm13v2.0.fa \\
        --download \\
        --subsample 5000000

    # Local BAMs already downloaded
    python scripts/generate_errorsmith_training_data.py \\
        --output training_output/errorsmith \\
        --reference /path/to/chm13v2.0.fa \\
        --hifi-bam chm13_hifi.sorted.bam \\
        --ont-bam  chm13_ont.sorted.bam \\
        --illumina-bam chm13_illumina.sorted.bam

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

# ── StrandWeaver imports ──────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("strandweaver.errorsmith_training")

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# CHM13 public accessions (NCBI SRA) — these are the most widely-used
# benchmark datasets for the T2T consortium
CHM13_ACCESSIONS = {
    'hifi':     'SRR19325710',  # PacBio HiFi, CHM13, ~35×
    'ont':      'SRR23571061',  # ONT R10.4.1, CHM13, ~50×
    'illumina': 'SRR3189741',  # Illumina PCR-free, CHM13, ~40×
}

# CHM13v2.0 reference (T2T consortium)
CHM13_REF_URL = (
    'https://s3-us-west-2.amazonaws.com/human-pangenomics/'
    'T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz'
)

# Feature names for the output CSV
ERROR_FEATURES = [
    # Per-base features (extracted from alignment)
    'base_quality',              # Phred quality at error position
    'mean_quality_window_5',     # Mean quality in ±5 bp window
    'mean_quality_window_20',    # Mean quality in ±20 bp window
    'position_in_read',          # Normalised position (0.0 = start, 1.0 = end)
    'read_length',               # Total read length
    'gc_content_local',          # GC content in ±25 bp window
    'gc_content_read',           # GC content of full read

    # Homopolymer context
    'homopolymer_length',        # Length of homopolymer run at this position (0 = not in HP)
    'homopolymer_base',          # Encoded base of HP (A=0, C=1, G=2, T=3, NA=-1)
    'distance_to_hp',            # Distance to nearest homopolymer (0 = in HP)

    # K-mer context
    'trinucleotide_context',     # Encoded trinucleotide (64 categories → integer)
    'pentanucleotide_context',   # Encoded pentanucleotide (1024 categories → integer)

    # Sequencing technology
    'technology_encoded',        # 0=HiFi, 1=ONT, 2=Illumina

    # Reference context
    'ref_gc_window_50',          # GC content of ±50 bp reference window
    'ref_repeat_flag',           # 1 if reference base is lowercase (masked repeat)
    'ref_homopolymer_length',    # HP length at this position in reference
]

LABEL_COLUMN = 'error_type'

# Error type encoding
ERROR_TYPES = {
    'correct': 0,
    'substitution': 1,
    'insertion': 2,
    'deletion': 3,
    'homopolymer_error': 4,
}
ERROR_TYPE_NAMES = {v: k for k, v in ERROR_TYPES.items()}

# Metadata columns (not features)
META_COLUMNS = [
    'chrom', 'ref_pos', 'read_name', 'technology',
]

# Base encoding
BASE_ENCODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}

# Trinucleotide and pentanucleotide encoding
_TRINUC_MAP = {}
_PENTANUC_MAP = {}
_idx = 0
for b1 in 'ACGT':
    for b2 in 'ACGT':
        for b3 in 'ACGT':
            _TRINUC_MAP[b1 + b2 + b3] = _idx
            _idx += 1
_idx = 0
for b1 in 'ACGT':
    for b2 in 'ACGT':
        for b3 in 'ACGT':
            for b4 in 'ACGT':
                for b5 in 'ACGT':
                    _PENTANUC_MAP[b1 + b2 + b3 + b4 + b5] = _idx
                    _idx += 1


# ═══════════════════════════════════════════════════════════════════════
#  CIGAR PARSING — extract per-base errors from aligned reads
# ═══════════════════════════════════════════════════════════════════════

# CIGAR operation codes (from SAM spec)
_CIGAR_OPS = {
    0: 'M',   # alignment match (or mismatch)
    1: 'I',   # insertion to reference
    2: 'D',   # deletion from reference
    3: 'N',   # skipped region
    4: 'S',   # soft clipping
    5: 'H',   # hard clipping
    6: 'P',   # padding
    7: '=',   # sequence match
    8: 'X',   # sequence mismatch
}

_CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')


@dataclass
class AlignedBase:
    """A single base from an alignment with error annotation."""
    read_pos: int           # Position in read sequence
    ref_pos: int            # Position in reference
    read_base: str          # Base in read
    ref_base: str           # Base in reference
    quality: int            # Phred quality score
    error_type: str         # 'correct', 'substitution', 'insertion', 'deletion'


def parse_cigar_string(cigar_str: str) -> List[Tuple[int, str]]:
    """Parse CIGAR string into list of (length, operation) tuples."""
    return [(int(length), op) for length, op in _CIGAR_RE.findall(cigar_str)]


def extract_errors_from_alignment(
    read_seq: str,
    read_qual: str,
    ref_seq: str,
    cigar: List[Tuple[int, str]],
    ref_start: int,
) -> Generator[AlignedBase, None, None]:
    """
    Walk through a CIGAR alignment and yield AlignedBase for each position.

    Yields both correct and erroneous bases for balanced training data.
    """
    read_pos = 0
    ref_pos = ref_start

    for length, op in cigar:
        if op in ('M', '=', 'X'):
            # Match/mismatch — consume both read and ref
            for i in range(length):
                if read_pos >= len(read_seq) or (ref_pos - ref_start) >= len(ref_seq):
                    break
                r_base = read_seq[read_pos]
                q_base = ref_seq[ref_pos - ref_start] if (ref_pos - ref_start) < len(ref_seq) else 'N'
                qual = ord(read_qual[read_pos]) - 33 if read_pos < len(read_qual) else 30

                if op == '=' or r_base.upper() == q_base.upper():
                    error = 'correct'
                else:
                    error = 'substitution'

                yield AlignedBase(
                    read_pos=read_pos,
                    ref_pos=ref_pos,
                    read_base=r_base,
                    ref_base=q_base,
                    quality=qual,
                    error_type=error,
                )
                read_pos += 1
                ref_pos += 1

        elif op == 'I':
            # Insertion — consumes read but not ref
            for i in range(length):
                if read_pos >= len(read_seq):
                    break
                qual = ord(read_qual[read_pos]) - 33 if read_pos < len(read_qual) else 30
                yield AlignedBase(
                    read_pos=read_pos,
                    ref_pos=ref_pos,  # ref_pos stays the same
                    read_base=read_seq[read_pos],
                    ref_base='-',
                    quality=qual,
                    error_type='insertion',
                )
                read_pos += 1

        elif op == 'D':
            # Deletion — consumes ref but not read
            for i in range(length):
                if (ref_pos - ref_start) >= len(ref_seq):
                    break
                yield AlignedBase(
                    read_pos=read_pos,
                    ref_pos=ref_pos,
                    read_base='-',
                    ref_base=ref_seq[ref_pos - ref_start],
                    quality=0,
                    error_type='deletion',
                )
                ref_pos += 1

        elif op == 'S':
            # Soft clip — consume read only, skip
            read_pos += length

        elif op == 'H':
            # Hard clip — nothing to consume
            pass

        elif op == 'N':
            # Skipped region — consume ref only
            ref_pos += length


# ═══════════════════════════════════════════════════════════════════════
#  REFERENCE LOADING AND CONTEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def load_reference_fasta(ref_path: str) -> Dict[str, str]:
    """
    Load reference FASTA (possibly gzipped) into a dict {chrom: sequence}.

    Preserves case so that lowercase = soft-masked repeats can be detected.
    """
    logger.info(f"Loading reference from {ref_path}...")
    ref = {}
    current_chrom = None
    current_seq = []

    open_fn = gzip.open if str(ref_path).endswith('.gz') else open
    mode = 'rt' if str(ref_path).endswith('.gz') else 'r'

    with open_fn(ref_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_chrom is not None:
                    ref[current_chrom] = ''.join(current_seq)
                current_chrom = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_chrom is not None:
        ref[current_chrom] = ''.join(current_seq)

    logger.info(f"  Loaded {len(ref)} chromosomes, total {sum(len(s) for s in ref.values()):,} bp")
    return ref


def _find_homopolymers(seq: str, min_len: int = 3) -> Dict[int, Tuple[str, int]]:
    """
    Index homopolymer runs in a sequence.

    Returns:
        {position: (base, hp_length)} for every position within a homopolymer.
    """
    hp_index = {}
    i = 0
    while i < len(seq):
        base = seq[i].upper()
        if base not in 'ACGT':
            i += 1
            continue
        j = i + 1
        while j < len(seq) and seq[j].upper() == base:
            j += 1
        hp_len = j - i
        if hp_len >= min_len:
            for pos in range(i, j):
                hp_index[pos] = (base, hp_len)
        i = j
    return hp_index


def _gc_content_window(seq: str, centre: int, window: int) -> float:
    """GC content in a window around `centre`."""
    start = max(0, centre - window)
    end = min(len(seq), centre + window + 1)
    sub = seq[start:end].upper()
    if not sub:
        return 0.5
    gc = sub.count('G') + sub.count('C')
    return gc / len(sub)


def _trinuc_code(seq: str, pos: int) -> int:
    """Encode the trinucleotide context centred at `pos`."""
    if pos < 1 or pos >= len(seq) - 1:
        return -1
    tri = seq[pos - 1:pos + 2].upper()
    return _TRINUC_MAP.get(tri, -1)


def _pentanuc_code(seq: str, pos: int) -> int:
    """Encode the pentanucleotide context centred at `pos`."""
    if pos < 2 or pos >= len(seq) - 2:
        return -1
    penta = seq[pos - 2:pos + 3].upper()
    return _PENTANUC_MAP.get(penta, -1)


def _distance_to_hp(hp_index: Dict[int, Tuple[str, int]], pos: int, max_dist: int = 50) -> int:
    """Distance from `pos` to the nearest homopolymer run."""
    if pos in hp_index:
        return 0
    for d in range(1, max_dist + 1):
        if (pos - d) in hp_index or (pos + d) in hp_index:
            return d
    return max_dist


# ═══════════════════════════════════════════════════════════════════════
#  BAM PROCESSING — requires pysam
# ═══════════════════════════════════════════════════════════════════════

def _check_pysam():
    """Check that pysam is available."""
    try:
        import pysam  # noqa: F401
        return True
    except ImportError:
        logger.error(
            "pysam is required for BAM processing. "
            "Install with: pip install pysam"
        )
        return False


def process_bam(
    bam_path: str,
    reference: Dict[str, str],
    technology: str,
    subsample: int = 5_000_000,
    chroms: Optional[List[str]] = None,
    seed: int = 42,
) -> Generator[Dict[str, Any], None, None]:
    """
    Process a BAM file and yield per-base feature dictionaries.

    Args:
        bam_path: Path to sorted, indexed BAM
        reference: {chrom: sequence} dict
        technology: 'hifi', 'ont', or 'illumina'
        subsample: Max bases to sample (across all chroms)
        chroms: Subset of chromosomes to process (None = all)
        seed: Random seed for subsampling

    Yields:
        Dict with feature columns + label + metadata for each base
    """
    import pysam

    rng = random.Random(seed)
    tech_code = {'hifi': 0, 'ont': 1, 'illumina': 2}.get(technology, 3)

    bam = pysam.AlignmentFile(bam_path, 'rb')
    target_chroms = chroms or [c for c in bam.references if c in reference]

    bases_emitted = 0
    reads_processed = 0

    for chrom in target_chroms:
        if chrom not in reference:
            logger.warning(f"  Skipping {chrom}: not in reference")
            continue

        ref_seq = reference[chrom]
        ref_hp_index = _find_homopolymers(ref_seq, min_len=3)

        logger.info(f"  Processing {chrom} ({len(ref_seq):,} bp)...")

        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < 20:
                continue

            # Subsample reads to hit target
            if bases_emitted >= subsample:
                break

            read_seq = read.query_sequence
            read_qual = read.qual
            if not read_seq or not read_qual:
                continue

            # Get aligned reference segment
            ref_start = read.reference_start
            ref_end = read.reference_end or (ref_start + len(read_seq))
            if ref_end > len(ref_seq):
                ref_end = len(ref_seq)
            ref_segment = ref_seq[ref_start:ref_end]

            # Parse CIGAR
            cigar_tuples = read.cigartuples
            if not cigar_tuples:
                continue

            cigar_list = [(length, _CIGAR_OPS.get(op, '?'))
                          for op, length in cigar_tuples]

            # Build read homopolymer index
            read_hp = _find_homopolymers(read_seq, min_len=3)

            read_len = len(read_seq)
            gc_read = (read_seq.count('G') + read_seq.count('C')) / read_len if read_len > 0 else 0.5

            for ab in extract_errors_from_alignment(
                read_seq, read_qual, ref_segment, cigar_list, ref_start
            ):
                # Reclassify as homopolymer_error if in a HP context
                error_type = ab.error_type
                if error_type in ('insertion', 'deletion'):
                    ref_in_hp = ab.ref_pos in ref_hp_index
                    read_in_hp = ab.read_pos in read_hp
                    if ref_in_hp or read_in_hp:
                        error_type = 'homopolymer_error'

                # Subsample correct bases (they dominate)
                if error_type == 'correct':
                    # Keep ~2% of correct bases for balance
                    if rng.random() > 0.02:
                        continue

                # ── Extract features ───────────────────────────
                # Quality features
                qual = ab.quality
                q_window5 = _mean_quality_window(read_qual, ab.read_pos, 5)
                q_window20 = _mean_quality_window(read_qual, ab.read_pos, 20)

                # Position
                pos_norm = ab.read_pos / read_len if read_len > 0 else 0.5

                # GC features
                gc_local = _gc_content_of_read_window(read_seq, ab.read_pos, 25)

                # Homopolymer features (read-side)
                if ab.read_pos in read_hp:
                    hp_base_enc, hp_len = BASE_ENCODE.get(read_hp[ab.read_pos][0], -1), read_hp[ab.read_pos][1]
                else:
                    hp_base_enc, hp_len = -1, 0
                dist_hp = _distance_to_hp(read_hp, ab.read_pos)

                # K-mer context
                trinuc = _trinuc_code(read_seq, ab.read_pos)
                pentanuc = _pentanuc_code(read_seq, ab.read_pos)

                # Reference context
                ref_pos_clamped = max(0, min(ab.ref_pos, len(ref_seq) - 1))
                ref_gc_50 = _gc_content_window(ref_seq, ref_pos_clamped, 50)
                ref_is_repeat = 1 if (ref_pos_clamped < len(ref_seq) and ref_seq[ref_pos_clamped].islower()) else 0
                ref_hp_len = ref_hp_index.get(ref_pos_clamped, ('N', 0))[1]

                row = {
                    # Features
                    'base_quality': qual,
                    'mean_quality_window_5': q_window5,
                    'mean_quality_window_20': q_window20,
                    'position_in_read': round(pos_norm, 5),
                    'read_length': read_len,
                    'gc_content_local': round(gc_local, 4),
                    'gc_content_read': round(gc_read, 4),
                    'homopolymer_length': hp_len,
                    'homopolymer_base': hp_base_enc,
                    'distance_to_hp': dist_hp,
                    'trinucleotide_context': trinuc,
                    'pentanucleotide_context': pentanuc,
                    'technology_encoded': tech_code,
                    'ref_gc_window_50': round(ref_gc_50, 4),
                    'ref_repeat_flag': ref_is_repeat,
                    'ref_homopolymer_length': ref_hp_len,

                    # Label
                    'error_type': ERROR_TYPES[error_type],

                    # Metadata
                    'chrom': chrom,
                    'ref_pos': ab.ref_pos,
                    'read_name': read.query_name[:50],
                    'technology': technology,
                }
                yield row
                bases_emitted += 1

            reads_processed += 1
            if reads_processed % 10000 == 0:
                logger.info(
                    f"    {reads_processed:,} reads → {bases_emitted:,} bases emitted"
                )

        if bases_emitted >= subsample:
            break

    bam.close()
    logger.info(f"  Done: {reads_processed:,} reads, {bases_emitted:,} bases emitted")


def _mean_quality_window(qual_str: str, centre: int, window: int) -> float:
    """Mean Phred quality in a window around `centre`."""
    start = max(0, centre - window)
    end = min(len(qual_str), centre + window + 1)
    if start >= end:
        return 30.0
    scores = [ord(qual_str[i]) - 33 for i in range(start, end)]
    return round(float(np.mean(scores)), 2)


def _gc_content_of_read_window(seq: str, centre: int, window: int) -> float:
    """GC content of a window around `centre` in the read."""
    start = max(0, centre - window)
    end = min(len(seq), centre + window + 1)
    sub = seq[start:end].upper()
    if not sub:
        return 0.5
    return (sub.count('G') + sub.count('C')) / len(sub)


# ═══════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════

# ENA mirror URLs — direct HTTPS download, no SRA toolkit needed.
# Pattern: https://ftp.sra.ebi.ac.uk/vol1/fastq/SRRxxx/0pp/SRRxxxxxxxxx/
#   where xxx = first 6 digits after SRR, 0pp = zero-padded last 2 digits
# (only for accessions with 10+ digit numbers).
ENA_FASTQ_URLS = {
    'SRR19325710': [  # HiFi — single-end
        'https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR193/010/SRR19325710/SRR19325710.fastq.gz',
    ],
    'SRR23571061': [  # ONT — single-end
        'https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR235/061/SRR23571061/SRR23571061.fastq.gz',
    ],
    'SRR3189741': [   # Illumina — paired-end
        'https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR318/001/SRR3189741/SRR3189741_1.fastq.gz',
        'https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR318/001/SRR3189741/SRR3189741_2.fastq.gz',
    ],
}


def _configure_sra_toolkit() -> None:
    """Configure SRA toolkit for headless/Colab environments."""
    ncbi_dir = Path.home() / '.ncbi'
    ncbi_dir.mkdir(exist_ok=True)
    config = ncbi_dir / 'user-settings.mkfg'
    if not config.exists():
        config.write_text(
            '/LIBS/GUID = "strandweaver-colab"\n'
            '/libs/cloud/accept_aws_charges = "false"\n'
            '/libs/cloud/accept_gcp_charges = "true"\n'
            '/libs/cloud/report_instance_identity = "true"\n'
        )
    # Also try the CLI flags (some versions need this)
    for cmd in [
        ['vdb-config', '--accept-gcp-charges', 'yes'],
        ['vdb-config', '--set', '/repository/user/cache-disabled=true'],
    ]:
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


def _download_from_ena(accession: str, fastq_dir: Path) -> List[Path]:
    """Download FASTQ files directly from ENA mirror (no SRA toolkit)."""
    urls = ENA_FASTQ_URLS.get(accession)
    if not urls:
        raise ValueError(
            f"No ENA mirror URL configured for {accession}. "
            f"Known accessions: {list(ENA_FASTQ_URLS.keys())}"
        )

    downloaded = []
    for url in urls:
        fname = url.rsplit('/', 1)[-1]
        gz_path = fastq_dir / fname
        fq_path = fastq_dir / fname.replace('.fastq.gz', '.fastq')

        # Already decompressed?
        if fq_path.exists():
            downloaded.append(fq_path)
            continue

        if not gz_path.exists():
            logger.info(f"  Downloading {fname} from ENA...")
            subprocess.run(
                ['wget', '-q', '--show-progress', '-O', str(gz_path), url],
                check=True,
            )

        # Decompress
        logger.info(f"  Decompressing {fname}...")
        subprocess.run(['gunzip', '-f', str(gz_path)], check=True)
        downloaded.append(fq_path)

    return downloaded


def download_sra_and_align(
    accession: str,
    technology: str,
    reference_path: str,
    output_dir: Path,
    threads: int = 8,
) -> Path:
    """
    Download SRA data, convert to FASTQ, align to reference, sort + index.

    Download strategy (in order of preference):
      1. ``fasterq-dump`` directly from accession (no prefetch)
      2. ``prefetch`` + ``fasterq-dump`` from local cache
      3. ``fastq-dump --split-3`` (slower, more tolerant)
      4. Direct HTTPS download from ENA mirror (no SRA toolkit at all)

    Returns:
        Path to sorted, indexed BAM file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bam_path = output_dir / f'{accession}_{technology}.sorted.bam'
    if bam_path.exists() and (bam_path.with_suffix('.bam.bai')).exists():
        logger.info(f"  BAM already exists: {bam_path}")
        return bam_path

    logger.info(f"  Downloading {accession} ({technology})...")

    # Configure SRA toolkit (idempotent)
    _configure_sra_toolkit()

    fastq_dir = output_dir / 'fastq'
    fastq_dir.mkdir(exist_ok=True)
    temp_dir = output_dir / 'fasterq_tmp'
    temp_dir.mkdir(exist_ok=True)

    fastq_obtained = False

    # ── Strategy 1: fasterq-dump directly from accession ────────────
    #    (downloads on the fly, no prefetch needed)
    if not fastq_obtained:
        try:
            logger.info("  Strategy 1: fasterq-dump from accession...")
            subprocess.run(
                ['fasterq-dump',
                 '--outdir', str(fastq_dir),
                 '--temp', str(temp_dir),
                 '--threads', str(threads),
                 '--split-3',
                 accession],
                check=True, capture_output=True, text=True,
            )
            fastq_obtained = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning(f"  Strategy 1 failed: {exc}")

    # ── Strategy 2: prefetch + fasterq-dump ─────────────────────────
    if not fastq_obtained:
        sra_dir = output_dir / 'sra_cache'
        sra_dir.mkdir(exist_ok=True)
        try:
            logger.info("  Strategy 2: prefetch + fasterq-dump...")
            subprocess.run(
                ['prefetch', '--max-size', '50G',
                 '--output-directory', str(sra_dir), accession],
                check=True, capture_output=True, text=True,
            )
            # Find the prefetch output
            sra_file = sra_dir / accession
            if not sra_file.exists():
                sra_file = sra_dir / accession / f'{accession}.sra'

            subprocess.run(
                ['fasterq-dump',
                 '--outdir', str(fastq_dir),
                 '--temp', str(temp_dir),
                 '--threads', str(threads),
                 '--split-3',
                 str(sra_file)],
                check=True, capture_output=True, text=True,
            )
            fastq_obtained = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning(f"  Strategy 2 failed: {exc}")

    # ── Strategy 3: fastq-dump (slower, more tolerant) ──────────────
    if not fastq_obtained:
        try:
            logger.info("  Strategy 3: fastq-dump --split-3...")
            subprocess.run(
                ['fastq-dump', '--split-3',
                 '--outdir', str(fastq_dir),
                 accession],
                check=True, capture_output=True, text=True,
            )
            fastq_obtained = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning(f"  Strategy 3 failed: {exc}")

    # ── Strategy 4: Direct download from ENA ────────────────────────
    if not fastq_obtained:
        logger.info("  Strategy 4: Direct download from ENA mirror...")
        _download_from_ena(accession, fastq_dir)
        fastq_obtained = True

    # Step 3: Align with minimap2
    # Preset selection
    if technology == 'hifi':
        preset = 'map-hifi'
    elif technology == 'ont':
        preset = 'map-ont'
    elif technology == 'illumina':
        preset = 'sr'
    else:
        preset = 'map-ont'

    # Find FASTQ files
    fq_files = sorted(fastq_dir.glob(f'{accession}*.fastq'))
    if not fq_files:
        fq_files = sorted(fastq_dir.glob(f'{accession}*.fq'))
    if not fq_files:
        raise FileNotFoundError(f"No FASTQ files found for {accession}")

    unsorted_bam = output_dir / f'{accession}_{technology}.unsorted.bam'
    logger.info(f"  Aligning with minimap2 -x {preset}...")

    # minimap2 → samtools sort
    minimap_cmd = [
        'minimap2', '-ax', preset,
        '-t', str(threads),
        str(reference_path),
    ] + [str(f) for f in fq_files]

    sort_cmd = [
        'samtools', 'sort',
        '-@', str(threads),
        '-o', str(bam_path),
    ]

    with subprocess.Popen(minimap_cmd, stdout=subprocess.PIPE) as mm_proc:
        subprocess.run(sort_cmd, stdin=mm_proc.stdout, check=True)

    # Step 4: Index
    subprocess.run(['samtools', 'index', '-@', str(threads), str(bam_path)], check=True)

    # Clean up intermediate files
    if unsorted_bam.exists():
        unsorted_bam.unlink()

    logger.info(f"  BAM ready: {bam_path}")
    return bam_path


def download_reference(output_dir: Path) -> Path:
    """Download CHM13v2.0 reference FASTA if not present."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_gz = output_dir / 'chm13v2.0.fa.gz'
    ref_fa = output_dir / 'chm13v2.0.fa'

    if ref_fa.exists():
        return ref_fa
    if ref_gz.exists():
        logger.info("  Decompressing reference...")
        subprocess.run(['gunzip', '-k', str(ref_gz)], check=True)
        return ref_fa

    logger.info(f"  Downloading CHM13v2.0 reference...")
    subprocess.run(['wget', '-O', str(ref_gz), CHM13_REF_URL], check=True)
    subprocess.run(['gunzip', '-k', str(ref_gz)], check=True)
    return ref_fa


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def generate_errorsmith_training_data(
    output_dir: Path,
    reference_path: str,
    hifi_bam: Optional[str] = None,
    ont_bam: Optional[str] = None,
    illumina_bam: Optional[str] = None,
    download: bool = False,
    subsample: int = 5_000_000,
    chroms: Optional[List[str]] = None,
    threads: int = 8,
    seed: int = 42,
) -> Path:
    """
    Generate ErrorSmith training CSV from real CHM13 alignments.

    Args:
        output_dir: Output directory
        reference_path: Path to CHM13v2.0 FASTA
        hifi_bam: Path to HiFi BAM (optional)
        ont_bam: Path to ONT BAM (optional)
        illumina_bam: Path to Illumina BAM (optional)
        download: Auto-download SRA data if BAMs not provided
        subsample: Max bases per technology
        chroms: Chromosomes to process (default: chr1-chr5 for speed)
        threads: Threads for alignment
        seed: Random seed

    Returns:
        Path to output CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _check_pysam():
        raise ImportError("pysam is required. Install with: pip install pysam")

    # Default to chr1–chr5 for manageable runtime
    if chroms is None:
        chroms = [f'chr{i}' for i in range(1, 6)]

    # Resolve BAMs
    bam_map: Dict[str, str] = {}
    if hifi_bam:
        bam_map['hifi'] = hifi_bam
    if ont_bam:
        bam_map['ont'] = ont_bam
    if illumina_bam:
        bam_map['illumina'] = illumina_bam

    if download:
        download_dir = output_dir / 'downloads'
        for tech, accession in CHM13_ACCESSIONS.items():
            if tech not in bam_map:
                bam_path = download_sra_and_align(
                    accession, tech, reference_path, download_dir, threads,
                )
                bam_map[tech] = str(bam_path)

    if not bam_map:
        raise ValueError(
            "No BAM files provided and --download not set. "
            "Provide at least one BAM or use --download."
        )

    # Load reference
    reference = load_reference_fasta(reference_path)

    # Process each technology
    csv_path = output_dir / 'errorsmith_training.csv'
    all_columns = ERROR_FEATURES + [LABEL_COLUMN] + META_COLUMNS

    total_rows = 0
    class_counts = Counter()
    t_start = time.time()

    logger.info(f"\nErrorSmith training data generation")
    logger.info(f"  Technologies: {list(bam_map.keys())}")
    logger.info(f"  Chromosomes:  {chroms}")
    logger.info(f"  Subsample:    {subsample:,} bases/technology")
    logger.info(f"  Output:       {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for tech, bam_path in bam_map.items():
            logger.info(f"\n{'─'*60}")
            logger.info(f"Processing {tech}: {bam_path}")
            logger.info(f"{'─'*60}")

            tech_rows = 0
            for row in process_bam(
                bam_path, reference, tech,
                subsample=subsample, chroms=chroms, seed=seed,
            ):
                writer.writerow(row)
                total_rows += 1
                tech_rows += 1
                class_counts[ERROR_TYPE_NAMES[row['error_type']]] += 1

                if tech_rows % 100_000 == 0:
                    f.flush()

            logger.info(f"  {tech}: {tech_rows:,} rows")

    elapsed = time.time() - t_start

    logger.info(f"\n{'='*60}")
    logger.info(f"ErrorSmith training data generation complete!")
    logger.info(f"  Total rows:   {total_rows:,}")
    logger.info(f"  Class distribution:")
    for cls, count in sorted(class_counts.items()):
        pct = count / total_rows * 100 if total_rows > 0 else 0
        logger.info(f"    {cls:22s}: {count:>10,} ({pct:5.1f}%)")
    logger.info(f"  Output CSV:   {csv_path}")
    logger.info(f"  Elapsed:      {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"{'='*60}")

    # Save config
    config_path = output_dir / 'errorsmith_generation_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'technologies': list(bam_map.keys()),
            'bam_files': bam_map,
            'reference': reference_path,
            'chroms': chroms,
            'subsample_per_tech': subsample,
            'seed': seed,
            'total_rows': total_rows,
            'class_distribution': dict(class_counts),
            'elapsed_seconds': elapsed,
        }, f, indent=2)

    return csv_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='generate-errorsmith-training-data',
        description='Generate ErrorSmith training data from real CHM13 alignments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Auto-download CHM13 reads, align, and extract errors
              python scripts/generate_errorsmith_training_data.py \\
                  --output training_output/errorsmith \\
                  --reference chm13v2.0.fa \\
                  --download --subsample 5000000

              # Use pre-aligned BAMs
              python scripts/generate_errorsmith_training_data.py \\
                  --output training_output/errorsmith \\
                  --reference chm13v2.0.fa \\
                  --hifi-bam chm13_hifi.sorted.bam \\
                  --ont-bam  chm13_ont.sorted.bam \\
                  --illumina-bam chm13_illumina.sorted.bam

              # Quick test on chr1 only
              python scripts/generate_errorsmith_training_data.py \\
                  --output training_output/errorsmith_test \\
                  --reference chm13v2.0.fa \\
                  --hifi-bam chm13_hifi.sorted.bam \\
                  --chroms chr1 --subsample 500000

              # After generation, train ErrorSmith:
              python -m strandweaver.user_training.train_models \\
                  --data-dir training_output/errorsmith \\
                  --output-dir trained_models/ \\
                  --models errorsmith
        """),
    )

    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory for training CSV and config',
    )
    parser.add_argument(
        '--reference', '-r', required=True,
        help='Path to CHM13v2.0 reference FASTA (or .fa.gz)',
    )
    parser.add_argument(
        '--hifi-bam', default=None,
        help='Path to HiFi BAM aligned to CHM13',
    )
    parser.add_argument(
        '--ont-bam', default=None,
        help='Path to ONT BAM aligned to CHM13',
    )
    parser.add_argument(
        '--illumina-bam', default=None,
        help='Path to Illumina BAM aligned to CHM13',
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Auto-download SRA data and align (requires sra-toolkit, minimap2, samtools)',
    )
    parser.add_argument(
        '--download-ref', action='store_true',
        help='Download CHM13v2.0 reference if not found at --reference path',
    )
    parser.add_argument(
        '--subsample', type=int, default=5_000_000,
        help='Max bases per technology (default: 5000000)',
    )
    parser.add_argument(
        '--chroms', nargs='+', default=None,
        help='Chromosomes to process (default: chr1-chr5)',
    )
    parser.add_argument(
        '--threads', '-t', type=int, default=8,
        help='Threads for alignment (default: 8)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose logging',
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Handle reference download
    ref_path = args.reference
    if args.download_ref and not os.path.exists(ref_path):
        ref_dir = Path(args.output) / 'reference'
        ref_path = str(download_reference(ref_dir))

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     StrandWeaver · ErrorSmith Training Data Generator      ║")
    print("║                   (CHM13 Real Data)                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Reference       : {Path(ref_path).name}")
    print(f"║  Technologies    : ", end='')
    techs = []
    if args.hifi_bam:
        techs.append('HiFi')
    if args.ont_bam:
        techs.append('ONT')
    if args.illumina_bam:
        techs.append('Illumina')
    if args.download:
        techs = ['HiFi', 'ONT', 'Illumina']
    print(', '.join(techs) or 'TBD (from BAMs)')
    print(f"║  Subsample       : {args.subsample:,} bases/tech")
    print(f"║  Chromosomes     : {args.chroms or 'chr1-chr5'}")
    print(f"║  Output          : {args.output}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    csv_path = generate_errorsmith_training_data(
        output_dir=Path(args.output),
        reference_path=ref_path,
        hifi_bam=args.hifi_bam,
        ont_bam=args.ont_bam,
        illumina_bam=args.illumina_bam,
        download=args.download,
        subsample=args.subsample,
        chroms=args.chroms,
        threads=args.threads,
        seed=args.seed,
    )

    print(f"\n✅ Training CSV ready: {csv_path}")
    print(f"\nNext step — train ErrorSmith classifier:")
    print(f"  python -m strandweaver.user_training.train_models \\")
    print(f"      --data-dir {args.output} \\")
    print(f"      --output-dir trained_models/ \\")
    print(f"      --models errorsmith")


if __name__ == '__main__':
    main()

# StrandWeaver v0.3.0-dev
# Any usage is subject to this software's license.
