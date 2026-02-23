#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Misassembly Detection — 12-method multi-signal detector with confidence scoring,
breakpoint characterization, and correction strategy recommendations.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class MisassemblyType(str, Enum):
    """Types of misassemblies."""
    RELOCATION = "relocation"  # Wrong order of contigs
    INVERSION = "inversion"  # Inverted sequence
    TRANSLOCATION = "translocation"  # Wrong chromosome/scaffold
    REPEAT_COLLAPSE = "repeat_collapse"  # Collapsed repeat
    REPEAT_EXPANSION = "repeat_expansion"  # Expanded repeat
    CHIMERA = "chimera"  # Two unrelated sequences joined
    INSERTION = "insertion"  # Spurious insertion
    DELETION = "deletion"  # Missing sequence
    DUPLICATION = "duplication"  # Duplicated region
    INTERSPECIES = "interspecies"  # Contamination
    HAPLOTYPE_SWITCH = "haplotype_switch"  # Switch between haplotypes
    UNKNOWN = "unknown"  # Uncharacterized


class ConfidenceLevel(str, Enum):
    """Confidence in misassembly call."""
    HIGH = "HIGH"  # Strong evidence, very likely misassembly
    MEDIUM = "MEDIUM"  # Moderate evidence, probable misassembly
    LOW = "LOW"  # Weak evidence, possible misassembly
    UNCERTAIN = "UNCERTAIN"  # Conflicting signals


class CorrectionStrategy(str, Enum):
    """Recommended correction approach."""
    BREAK_AND_REJOIN = "break_and_rejoin"  # Break at junction, rejoin correctly
    LOCAL_REASSEMBLY = "local_reassembly"  # Reassemble the region
    REMOVE_SEGMENT = "remove_segment"  # Delete problematic segment
    MANUAL_REVIEW = "manual_review"  # Requires manual inspection
    UL_VALIDATION = "ul_validation"  # Use ultra-long reads to validate
    HIC_SCAFFOLDING = "hic_scaffolding"  # Use Hi-C to correct
    LEAVE_AS_IS = "leave_as_is"  # Evidence insufficient for correction


@dataclass
class MisassemblySignal:
    """Single detection signal for a putative misassembly."""
    signal_type: str  # "coverage_drop", "edgewarden_low", etc.
    position: int  # Base position in contig
    score: float  # Signal strength (0-1)
    description: str  # Human-readable description
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MisassemblyFlag:
    """
    Comprehensive misassembly flag with location and evidence.
    
    This is passed to downstream modules for correction.
    """
    # Location
    contig_id: str
    start_pos: int  # Base position
    end_pos: int  # Base position
    breakpoint_pos: Optional[int] = None  # Precise breakpoint if known
    
    # Classification
    misassembly_type: MisassemblyType = MisassemblyType.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.UNCERTAIN
    
    # Evidence
    detection_signals: List[MisassemblySignal] = field(default_factory=list)
    num_signals: int = 0
    composite_score: float = 0.0  # Aggregate evidence score (0-100)
    
    # Context
    left_edge_id: Optional[str] = None  # Edge before breakpoint
    right_edge_id: Optional[str] = None  # Edge after breakpoint
    left_node_id: Optional[int] = None
    right_node_id: Optional[int] = None
    
    # EdgeWarden scores
    edgewarden_left: float = 0.5
    edgewarden_right: float = 0.5
    edgewarden_min: float = 0.5
    
    # Coverage
    coverage_left: float = 0.0
    coverage_right: float = 0.0
    coverage_ratio: float = 1.0  # right / left
    
    # Supporting evidence counts
    ul_reads_supporting: int = 0  # Ultra-long reads supporting misassembly
    ul_reads_conflicting: int = 0  # Ultra-long reads conflicting
    hic_links_supporting: int = 0  # Hi-C links supporting
    hic_links_conflicting: int = 0
    
    # Recommendations
    correction_strategy: CorrectionStrategy = CorrectionStrategy.MANUAL_REVIEW
    priority: int = 5  # 1-10, 1=highest priority
    notes: str = ""
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.num_signals = len(self.detection_signals)
        if self.detection_signals:
            self.composite_score = sum(s.score for s in self.detection_signals) / len(self.detection_signals) * 100


class MisassemblyDetector:
    """
    Multi-signal misassembly detection engine.
    
    Analyzes assembly paths to identify putative misassemblies using
    12 different detection methods. Produces confidence-scored flags
    for downstream correction.
    """
    
    def __init__(
        self,
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        min_composite_score: float = 40.0,
        edgewarden_threshold: float = 0.4,
        coverage_ratio_threshold: float = 2.0,
        enable_all_detectors: bool = True,
    ):
        """
        Initialize misassembly detector.
        
        Args:
            min_confidence: Minimum confidence to flag
            min_composite_score: Minimum composite score to flag
            edgewarden_threshold: EdgeWarden score below which to flag
            coverage_ratio_threshold: Coverage ratio above which to flag
            enable_all_detectors: Enable all 12 detection methods
        """
        self.min_confidence = min_confidence
        self.min_composite_score = min_composite_score
        self.edgewarden_threshold = edgewarden_threshold
        self.coverage_ratio_threshold = coverage_ratio_threshold
        self.enable_all_detectors = enable_all_detectors
        
        self.detected_flags: Dict[str, List[MisassemblyFlag]] = {}
        self.logger = logging.getLogger(f"{__name__}.MisassemblyDetector")
    
    def detect_in_path(
        self,
        contig_id: str,
        node_ids: List[int],
        edge_ids: List[str],
        edgewarden_scores: Optional[Dict[str, float]] = None,
        coverage_data: Optional[Dict[int, float]] = None,
        ul_alignments: Optional[List[Dict]] = None,
        hic_links: Optional[List[Dict]] = None,
        kmer_spectrum: Optional[Dict[int, Dict[str, float]]] = None,
        strand_data: Optional[Dict[int, Dict[str, int]]] = None,
        insert_sizes: Optional[Dict[str, Dict[str, float]]] = None,
        gc_content: Optional[Dict[int, float]] = None,
        het_markers: Optional[Dict[int, Dict[str, Any]]] = None,
        alignment_data: Optional[Dict[int, Dict[str, Any]]] = None,
        graph_topology: Optional[Dict[int, Dict[str, int]]] = None,
        **kwargs
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies in a single assembled path.
        
        Args:
            contig_id: Contig identifier
            node_ids: Sequence of node IDs in path
            edge_ids: Sequence of edge IDs connecting nodes
            edgewarden_scores: Edge confidence scores from EdgeWarden
            coverage_data: Node coverage data
            ul_alignments: Ultra-long read alignments
            hic_links: Hi-C contact links
            kmer_spectrum: Per-node k-mer multiplicity stats. Dict mapping
                node_id -> {'unique_ratio': float, 'mean_freq': float,
                            'max_freq': float, 'freq_stdev': float}
            strand_data: Per-node read strand orientation counts. Dict mapping
                node_id -> {'fwd': int, 'rev': int}
            insert_sizes: Per-edge insert size distributions. Dict mapping
                edge_id -> {'mean': float, 'stdev': float, 'n': int}
            gc_content: Per-node GC content (0.0-1.0). Dict mapping
                node_id -> float
            het_markers: Per-node heterozygous marker info. Dict mapping
                node_id -> {'het_count': int, 'hom_count': int,
                            'het_ratio': float}
            alignment_data: Per-node alignment breakpoint info. Dict mapping
                node_id -> {'clipped_reads': int, 'total_reads': int,
                            'split_reads': int}
            graph_topology: Per-node graph topology info. Dict mapping
                node_id -> {'in_degree': int, 'out_degree': int}
            **kwargs: Additional detection data
        
        Returns:
            List of MisassemblyFlag objects
        """
        flags = []
        
        # Validate inputs
        if len(edge_ids) != len(node_ids) - 1:
            self.logger.warning(f"Contig {contig_id}: edge count mismatch")
            return flags
        
        # === Core detectors (1-2) ===
        if edgewarden_scores:
            flags.extend(self._detect_low_confidence_edges(
                contig_id, node_ids, edge_ids, edgewarden_scores
            ))
        
        if coverage_data:
            flags.extend(self._detect_coverage_discontinuities(
                contig_id, node_ids, coverage_data
            ))
        
        # === Insert size detector (3) ===
        if insert_sizes:
            flags.extend(self._detect_insert_size_anomalies(
                contig_id, node_ids, edge_ids, insert_sizes
            ))
        
        # === Strand orientation detector (4) ===
        if strand_data:
            flags.extend(self._detect_strand_orientation_inconsistencies(
                contig_id, node_ids, strand_data
            ))
        
        # === Repeat boundary detector (5) ===
        if coverage_data:
            flags.extend(self._detect_repeat_boundary_violations(
                contig_id, node_ids, coverage_data, gc_content
            ))
        
        # === GC content anomaly detector (6) ===
        if gc_content:
            flags.extend(self._detect_gc_content_anomalies(
                contig_id, node_ids, gc_content
            ))
        
        # === K-mer spectrum detector (7) ===
        if kmer_spectrum:
            flags.extend(self._detect_kmer_spectrum_disruptions(
                contig_id, node_ids, kmer_spectrum, coverage_data
            ))
        
        # === UL conflict detector (8) ===
        if ul_alignments:
            flags.extend(self._detect_ul_conflicts(
                contig_id, node_ids, ul_alignments
            ))
        
        # === Hi-C violation detector (9) ===
        if hic_links:
            flags.extend(self._detect_hic_violations(
                contig_id, node_ids, hic_links
            ))
        
        # === Heterozygous marker detector (10) ===
        if het_markers:
            flags.extend(self._detect_het_marker_inconsistencies(
                contig_id, node_ids, het_markers
            ))
        
        # === Alignment breakpoint detector (11) ===
        if alignment_data:
            flags.extend(self._detect_alignment_breakpoints(
                contig_id, node_ids, alignment_data
            ))
        
        # === Graph topology detector (12) ===
        if graph_topology:
            flags.extend(self._detect_graph_topology_inconsistencies(
                contig_id, node_ids, graph_topology
            ))
        
        # Merge overlapping flags
        flags = self._merge_overlapping_flags(flags)
        
        # Filter by confidence and score
        flags = self._filter_flags(flags)
        
        # Assign correction strategies
        flags = self._assign_correction_strategies(flags)
        
        # Store results
        self.detected_flags[contig_id] = flags
        
        self.logger.info(
            f"Contig {contig_id}: Detected {len(flags)} putative misassemblies"
        )
        
        return flags
    
    def _detect_low_confidence_edges(
        self,
        contig_id: str,
        node_ids: List[int],
        edge_ids: List[str],
        edgewarden_scores: Dict[str, float],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies based on low EdgeWarden confidence.
        
        CRITICAL: EdgeWarden scores below threshold indicate unreliable edges.
        """
        flags = []
        
        for i, edge_id in enumerate(edge_ids):
            score = edgewarden_scores.get(edge_id, 0.5)
            
            if score < self.edgewarden_threshold:
                # Low confidence edge detected
                left_node = node_ids[i]
                right_node = node_ids[i + 1]
                
                signal = MisassemblySignal(
                    signal_type="edgewarden_low_confidence",
                    position=i,  # Edge index
                    score=1.0 - score,  # Invert: low confidence = high signal
                    description=f"EdgeWarden confidence {score:.3f} below threshold {self.edgewarden_threshold}",
                    supporting_data={"edge_id": edge_id, "confidence": score}
                )
                
                # Determine misassembly type based on score
                if score < 0.2:
                    mistype = MisassemblyType.CHIMERA  # Very low confidence
                    confidence = ConfidenceLevel.HIGH
                elif score < 0.3:
                    mistype = MisassemblyType.UNKNOWN
                    confidence = ConfidenceLevel.MEDIUM
                else:
                    mistype = MisassemblyType.UNKNOWN
                    confidence = ConfidenceLevel.LOW
                
                flag = MisassemblyFlag(
                    contig_id=contig_id,
                    start_pos=i * 1000,  # Approximate position
                    end_pos=(i + 1) * 1000,
                    breakpoint_pos=i * 1000 + 500,
                    misassembly_type=mistype,
                    confidence=confidence,
                    detection_signals=[signal],
                    left_edge_id=edge_id if i > 0 else None,
                    right_edge_id=edge_id,
                    left_node_id=left_node,
                    right_node_id=right_node,
                    edgewarden_left=edgewarden_scores.get(edge_ids[i-1], 0.5) if i > 0 else 0.5,
                    edgewarden_right=score,
                    edgewarden_min=score,
                )
                
                flags.append(flag)
        
        return flags
    
    def _detect_coverage_discontinuities(
        self,
        contig_id: str,
        node_ids: List[int],
        coverage_data: Dict[int, float],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies based on coverage discontinuities.
        
        Sudden coverage changes indicate potential misjoins.
        """
        flags = []
        
        for i in range(len(node_ids) - 1):
            left_node = node_ids[i]
            right_node = node_ids[i + 1]
            
            left_cov = coverage_data.get(left_node, 0.0)
            right_cov = coverage_data.get(right_node, 0.0)
            
            if left_cov == 0 or right_cov == 0:
                continue
            
            ratio = max(left_cov, right_cov) / min(left_cov, right_cov)
            
            if ratio > self.coverage_ratio_threshold:
                # Coverage discontinuity detected
                signal = MisassemblySignal(
                    signal_type="coverage_discontinuity",
                    position=i,
                    score=min(1.0, (ratio - 1.0) / 3.0),  # Normalize to 0-1
                    description=f"Coverage ratio {ratio:.2f}× (left={left_cov:.1f}, right={right_cov:.1f})",
                    supporting_data={"left_cov": left_cov, "right_cov": right_cov, "ratio": ratio}
                )
                
                # High ratio = high confidence
                if ratio > 5.0:
                    confidence = ConfidenceLevel.HIGH
                    mistype = MisassemblyType.REPEAT_COLLAPSE
                elif ratio > 3.0:
                    confidence = ConfidenceLevel.MEDIUM
                    mistype = MisassemblyType.REPEAT_COLLAPSE
                else:
                    confidence = ConfidenceLevel.LOW
                    mistype = MisassemblyType.UNKNOWN
                
                flag = MisassemblyFlag(
                    contig_id=contig_id,
                    start_pos=i * 1000,
                    end_pos=(i + 1) * 1000,
                    breakpoint_pos=i * 1000 + 500,
                    misassembly_type=mistype,
                    confidence=confidence,
                    detection_signals=[signal],
                    left_node_id=left_node,
                    right_node_id=right_node,
                    coverage_left=left_cov,
                    coverage_right=right_cov,
                    coverage_ratio=ratio,
                )
                
                flags.append(flag)
        
        return flags
    
    def _detect_ul_conflicts(
        self,
        contig_id: str,
        node_ids: List[int],
        ul_alignments: List[Dict],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies based on ultra-long read conflicts.
        
        UL reads that don't support the path indicate misassemblies.
        """
        flags = []
        
        # Group alignments by position
        conflicts_by_edge = defaultdict(list)
        
        for aln in ul_alignments:
            if not aln.get('conflicts', False):
                continue
            
            edge_idx = aln.get('edge_index', -1)
            if edge_idx >= 0 and edge_idx < len(node_ids) - 1:
                conflicts_by_edge[edge_idx].append(aln)
        
        # Create flags for edges with conflicts
        for edge_idx, conflicts in conflicts_by_edge.items():
            num_conflicts = len(conflicts)
            
            if num_conflicts >= 2:  # At least 2 conflicting UL reads
                signal = MisassemblySignal(
                    signal_type="ul_read_conflict",
                    position=edge_idx,
                    score=min(1.0, num_conflicts / 5.0),  # Normalize
                    description=f"{num_conflicts} ultra-long reads conflict with path",
                    supporting_data={"num_conflicts": num_conflicts, "reads": conflicts}
                )
                
                # More conflicts = higher confidence
                if num_conflicts >= 5:
                    confidence = ConfidenceLevel.HIGH
                elif num_conflicts >= 3:
                    confidence = ConfidenceLevel.MEDIUM
                else:
                    confidence = ConfidenceLevel.LOW
                
                flag = MisassemblyFlag(
                    contig_id=contig_id,
                    start_pos=edge_idx * 1000,
                    end_pos=(edge_idx + 1) * 1000,
                    breakpoint_pos=edge_idx * 1000 + 500,
                    misassembly_type=MisassemblyType.CHIMERA,
                    confidence=confidence,
                    detection_signals=[signal],
                    left_node_id=node_ids[edge_idx],
                    right_node_id=node_ids[edge_idx + 1],
                    ul_reads_conflicting=num_conflicts,
                )
                
                flags.append(flag)
        
        return flags
    
    def _detect_hic_violations(
        self,
        contig_id: str,
        node_ids: List[int],
        hic_links: List[Dict],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies based on Hi-C link violations.
        
        Missing Hi-C links where expected indicate misassemblies.
        """
        flags = []
        
        # Expected Hi-C links between adjacent nodes
        expected_links = defaultdict(int)
        actual_links = defaultdict(int)
        
        for i in range(len(node_ids) - 1):
            left_node = node_ids[i]
            right_node = node_ids[i + 1]
            expected_links[i] = 1  # We expect links here
        
        # Count actual Hi-C links
        for link in hic_links:
            node1 = link.get('node1')
            node2 = link.get('node2')
            
            # Find edge index
            for i in range(len(node_ids) - 1):
                if (node_ids[i] == node1 and node_ids[i+1] == node2) or \
                   (node_ids[i] == node2 and node_ids[i+1] == node1):
                    actual_links[i] += 1
        
        # Flag edges with missing Hi-C support
        for edge_idx in expected_links:
            if actual_links[edge_idx] == 0:
                # No Hi-C support
                signal = MisassemblySignal(
                    signal_type="hic_link_violation",
                    position=edge_idx,
                    score=0.7,  # Moderate signal
                    description="Missing Hi-C links between adjacent nodes",
                    supporting_data={"expected": 1, "actual": 0}
                )
                
                flag = MisassemblyFlag(
                    contig_id=contig_id,
                    start_pos=edge_idx * 1000,
                    end_pos=(edge_idx + 1) * 1000,
                    breakpoint_pos=edge_idx * 1000 + 500,
                    misassembly_type=MisassemblyType.TRANSLOCATION,
                    confidence=ConfidenceLevel.MEDIUM,
                    detection_signals=[signal],
                    left_node_id=node_ids[edge_idx],
                    right_node_id=node_ids[edge_idx + 1],
                    hic_links_supporting=0,
                    hic_links_conflicting=0,
                )
                
                flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 3: Insert Size Anomalies
    # ====================================================================
    def _detect_insert_size_anomalies(
        self,
        contig_id: str,
        node_ids: List[int],
        edge_ids: List[str],
        insert_sizes: Dict[str, Dict[str, float]],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies from abnormal insert size distributions.
        
        For paired-end / mate-pair libraries, the insert size across a
        junction should be consistent with the library distribution.
        Significant deviations indicate structural misassemblies:
          - Mean shift > 2 stdevs from expected → relocation / insertion
          - Very high stdev relative to mean → chimeric junction
          - Extremely small n (few spanning pairs) → missing data (warn)
        
        insert_sizes: edge_id -> {'mean': float, 'stdev': float, 'n': int}
        """
        flags: List[MisassemblyFlag] = []
        
        if not insert_sizes:
            return flags
        
        # Compute global expected insert size from all edges
        all_means = [v['mean'] for v in insert_sizes.values() if v.get('n', 0) >= 5]
        if not all_means:
            return flags
        
        global_mean = float(np.median(all_means))
        global_stdev = float(np.std(all_means)) if len(all_means) > 1 else global_mean * 0.1
        if global_stdev < 1.0:
            global_stdev = 1.0
        
        for i, edge_id in enumerate(edge_ids):
            isize = insert_sizes.get(edge_id)
            if isize is None:
                continue
            
            edge_mean = isize.get('mean', global_mean)
            edge_stdev = isize.get('stdev', 0.0)
            edge_n = isize.get('n', 0)
            
            if edge_n < 3:
                continue  # Too few spanning pairs to judge
            
            signals = []
            
            # Signal 1: Mean shift from global median
            z_score = abs(edge_mean - global_mean) / global_stdev
            if z_score > 2.0:
                score = min(1.0, z_score / 5.0)
                signals.append(MisassemblySignal(
                    signal_type="insert_size_shift",
                    position=i,
                    score=score,
                    description=(
                        f"Insert size mean {edge_mean:.0f} deviates "
                        f"{z_score:.1f}σ from expected {global_mean:.0f}"
                    ),
                    supporting_data={
                        'edge_mean': edge_mean,
                        'global_mean': global_mean,
                        'z_score': z_score,
                        'n': edge_n,
                    }
                ))
            
            # Signal 2: High local stdev (bimodal / noisy)
            cv = edge_stdev / max(edge_mean, 1.0)  # coefficient of variation
            if cv > 0.5 and edge_n >= 5:
                score = min(1.0, cv / 1.0)
                signals.append(MisassemblySignal(
                    signal_type="insert_size_high_variance",
                    position=i,
                    score=score,
                    description=(
                        f"Insert size CV={cv:.2f} (stdev={edge_stdev:.0f}, "
                        f"mean={edge_mean:.0f}) — possible chimeric junction"
                    ),
                    supporting_data={
                        'cv': cv,
                        'edge_stdev': edge_stdev,
                        'edge_mean': edge_mean,
                    }
                ))
            
            if not signals:
                continue
            
            has_shift = any(s.signal_type == 'insert_size_shift' for s in signals)
            has_variance = any(s.signal_type == 'insert_size_high_variance' for s in signals)
            
            if has_shift and has_variance:
                confidence = ConfidenceLevel.HIGH
                mistype = MisassemblyType.CHIMERA
            elif has_shift and z_score > 4.0:
                confidence = ConfidenceLevel.HIGH
                mistype = MisassemblyType.RELOCATION
            elif has_shift:
                confidence = ConfidenceLevel.MEDIUM
                mistype = MisassemblyType.RELOCATION
            else:
                confidence = ConfidenceLevel.LOW
                mistype = MisassemblyType.UNKNOWN
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=mistype,
                confidence=confidence,
                detection_signals=signals,
                left_node_id=node_ids[i],
                right_node_id=node_ids[i + 1],
            )
            flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 5: Repeat Boundary Violations
    # ====================================================================
    def _detect_repeat_boundary_violations(
        self,
        contig_id: str,
        node_ids: List[int],
        coverage_data: Dict[int, float],
        gc_content: Optional[Dict[int, float]] = None,
    ) -> List[MisassemblyFlag]:
        """
        Detect repeat boundary violations.
        
        A repeat region is characterised by elevated coverage. A
        *boundary* violation occurs when the path transitions from a
        repeat node (high coverage) to a unique node (normal coverage)
        without a corresponding change on the other end — i.e. only
        one side of the repeat is properly anchored.
        
        We look for junctions where:
          1. One side has coverage > 2× median (repeat)
          2. The other side has coverage < median (unique)
          3. The GC content also shifts sharply (optional corroboration)
        """
        flags: List[MisassemblyFlag] = []
        
        if len(node_ids) < 3:
            return flags
        
        # Compute median coverage
        cov_values = [coverage_data.get(nid, 0.0) for nid in node_ids if coverage_data.get(nid, 0.0) > 0]
        if not cov_values:
            return flags
        median_cov = float(np.median(cov_values))
        if median_cov <= 0:
            return flags
        
        repeat_threshold = 2.0 * median_cov  # Coverage above this = repeat
        unique_ceiling = 1.2 * median_cov    # Coverage below this = unique
        
        for i in range(len(node_ids) - 1):
            left_id = node_ids[i]
            right_id = node_ids[i + 1]
            left_cov = coverage_data.get(left_id, 0.0)
            right_cov = coverage_data.get(right_id, 0.0)
            
            if left_cov <= 0 or right_cov <= 0:
                continue
            
            # Check repeat↔unique transitions
            left_is_repeat = left_cov > repeat_threshold
            right_is_unique = right_cov < unique_ceiling
            right_is_repeat = right_cov > repeat_threshold
            left_is_unique = left_cov < unique_ceiling
            
            is_boundary = (left_is_repeat and right_is_unique) or \
                          (right_is_repeat and left_is_unique)
            
            if not is_boundary:
                continue
            
            # Check if the repeat is properly anchored:
            # Look at the node *beyond* the unique side — if it's also repeat,
            # this is likely a collapsed tandem. If it's unique, the boundary
            # is suspicious.
            cov_ratio = max(left_cov, right_cov) / min(left_cov, right_cov)
            
            score = min(1.0, (cov_ratio - 1.0) / 4.0)
            
            desc = (
                f"Repeat boundary: cov_left={left_cov:.1f} ({'repeat' if left_is_repeat else 'unique'}), "
                f"cov_right={right_cov:.1f} ({'repeat' if right_is_repeat else 'unique'}), "
                f"median={median_cov:.1f}"
            )
            
            signal = MisassemblySignal(
                signal_type="repeat_boundary_violation",
                position=i,
                score=score,
                description=desc,
                supporting_data={
                    'left_cov': left_cov,
                    'right_cov': right_cov,
                    'median_cov': median_cov,
                    'cov_ratio': cov_ratio,
                }
            )
            
            signals = [signal]
            confidence = ConfidenceLevel.LOW
            
            # GC corroboration boosts confidence
            if gc_content:
                left_gc = gc_content.get(left_id)
                right_gc = gc_content.get(right_id)
                if left_gc is not None and right_gc is not None:
                    gc_diff = abs(left_gc - right_gc)
                    if gc_diff > 0.15:
                        signals.append(MisassemblySignal(
                            signal_type="repeat_boundary_gc_shift",
                            position=i,
                            score=min(1.0, gc_diff / 0.3),
                            description=f"GC shift {gc_diff:.2f} at repeat boundary",
                            supporting_data={'left_gc': left_gc, 'right_gc': right_gc}
                        ))
                        confidence = ConfidenceLevel.MEDIUM
            
            # High coverage ratio also boosts
            if cov_ratio > 4.0:
                confidence = ConfidenceLevel.MEDIUM
            if cov_ratio > 4.0 and len(signals) > 1:
                confidence = ConfidenceLevel.HIGH
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=MisassemblyType.REPEAT_COLLAPSE,
                confidence=confidence,
                detection_signals=signals,
                left_node_id=left_id,
                right_node_id=right_id,
                coverage_left=left_cov,
                coverage_right=right_cov,
                coverage_ratio=cov_ratio,
            )
            flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 6: GC Content Anomalies
    # ====================================================================
    def _detect_gc_content_anomalies(
        self,
        contig_id: str,
        node_ids: List[int],
        gc_content: Dict[int, float],
    ) -> List[MisassemblyFlag]:
        """
        Detect chimeric junctions via GC content analysis.
        
        GC content is relatively stable within a genome region. Abrupt
        shifts between adjacent nodes suggest that two unrelated genomic
        regions have been erroneously joined (chimera), or that
        contamination is present.
        
        gc_content: node_id -> float (0.0–1.0)
        """
        flags: List[MisassemblyFlag] = []
        
        if len(node_ids) < 2:
            return flags
        
        gc_shift_threshold = 0.12   # 12-percentage-point shift
        gc_shift_strong = 0.20      # 20-point shift = strong signal
        
        # Compute local GC median for context (sliding window of 5 nodes)
        gc_values = [gc_content.get(nid) for nid in node_ids if gc_content.get(nid) is not None]
        if len(gc_values) < 2:
            return flags
        global_gc_median = float(np.median(gc_values))
        
        for i in range(len(node_ids) - 1):
            left_id = node_ids[i]
            right_id = node_ids[i + 1]
            left_gc = gc_content.get(left_id)
            right_gc = gc_content.get(right_id)
            
            if left_gc is None or right_gc is None:
                continue
            
            gc_diff = abs(left_gc - right_gc)
            
            if gc_diff < gc_shift_threshold:
                continue
            
            score = min(1.0, gc_diff / 0.35)
            
            signal = MisassemblySignal(
                signal_type="gc_content_anomaly",
                position=i,
                score=score,
                description=(
                    f"GC content shift {gc_diff:.3f} "
                    f"(left={left_gc:.3f}, right={right_gc:.3f}, "
                    f"genome_median={global_gc_median:.3f})"
                ),
                supporting_data={
                    'left_gc': left_gc,
                    'right_gc': right_gc,
                    'gc_diff': gc_diff,
                    'global_gc_median': global_gc_median,
                }
            )
            
            if gc_diff > gc_shift_strong:
                confidence = ConfidenceLevel.MEDIUM
                mistype = MisassemblyType.CHIMERA
            else:
                confidence = ConfidenceLevel.LOW
                mistype = MisassemblyType.UNKNOWN
            
            # Very large shift may indicate contamination
            if gc_diff > 0.30:
                confidence = ConfidenceLevel.HIGH
                mistype = MisassemblyType.INTERSPECIES
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=mistype,
                confidence=confidence,
                detection_signals=[signal],
                left_node_id=left_id,
                right_node_id=right_id,
            )
            flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 10: Heterozygous Marker Inconsistencies
    # ====================================================================
    def _detect_het_marker_inconsistencies(
        self,
        contig_id: str,
        node_ids: List[int],
        het_markers: Dict[int, Dict[str, Any]],
    ) -> List[MisassemblyFlag]:
        """
        Detect haplotype switch errors via heterozygous marker analysis.
        
        In a diploid assembly, heterozygous markers should be phased
        consistently along each haplotype. A sudden change in het/hom
        ratio between adjacent nodes suggests a haplotype switch —
        the assembler jumped from one parental chromosome to the other.
        
        het_markers: node_id -> {
            'het_count': int,   # heterozygous markers in node
            'hom_count': int,   # homozygous markers in node
            'het_ratio': float, # het_count / (het_count + hom_count)
        }
        """
        flags: List[MisassemblyFlag] = []
        
        if len(node_ids) < 2:
            return flags
        
        het_ratio_flip_threshold = 0.30  # 30-point swing in het ratio
        min_markers = 3  # Need at least 3 markers to trust ratio
        
        for i in range(len(node_ids) - 1):
            left_id = node_ids[i]
            right_id = node_ids[i + 1]
            left_het = het_markers.get(left_id)
            right_het = het_markers.get(right_id)
            
            if left_het is None or right_het is None:
                continue
            
            left_total = left_het.get('het_count', 0) + left_het.get('hom_count', 0)
            right_total = right_het.get('het_count', 0) + right_het.get('hom_count', 0)
            
            if left_total < min_markers or right_total < min_markers:
                continue
            
            left_ratio = left_het.get('het_ratio', 0.0)
            right_ratio = right_het.get('het_ratio', 0.0)
            ratio_change = abs(left_ratio - right_ratio)
            
            if ratio_change < het_ratio_flip_threshold:
                continue
            
            score = min(1.0, ratio_change / 0.6)
            
            signal = MisassemblySignal(
                signal_type="het_marker_inconsistency",
                position=i,
                score=score,
                description=(
                    f"Het ratio flip: left={left_ratio:.2f} → right={right_ratio:.2f} "
                    f"(Δ={ratio_change:.2f}, left_markers={left_total}, "
                    f"right_markers={right_total})"
                ),
                supporting_data={
                    'left_ratio': left_ratio,
                    'right_ratio': right_ratio,
                    'ratio_change': ratio_change,
                    'left_total_markers': left_total,
                    'right_total_markers': right_total,
                }
            )
            
            if ratio_change > 0.50 and min(left_total, right_total) >= 10:
                confidence = ConfidenceLevel.HIGH
            elif ratio_change > 0.40:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=MisassemblyType.HAPLOTYPE_SWITCH,
                confidence=confidence,
                detection_signals=[signal],
                left_node_id=left_id,
                right_node_id=right_id,
            )
            flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 11: Alignment Breakpoints
    # ====================================================================
    def _detect_alignment_breakpoints(
        self,
        contig_id: str,
        node_ids: List[int],
        alignment_data: Dict[int, Dict[str, Any]],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies from split/clipped read alignment patterns.
        
        A structural misassembly causes reads spanning the breakpoint to
        be soft-clipped or split-mapped. A high proportion of clipped /
        split reads at a node boundary is a strong indicator.
        
        alignment_data: node_id -> {
            'clipped_reads': int,  # soft/hard-clipped reads
            'split_reads': int,    # split-mapped reads
            'total_reads': int,    # total aligned reads
        }
        """
        flags: List[MisassemblyFlag] = []
        
        if len(node_ids) < 2:
            return flags
        
        clip_ratio_threshold = 0.15   # >15% clipped = suspicious
        split_ratio_threshold = 0.10  # >10% split = suspicious
        min_reads = 10
        
        for i in range(len(node_ids) - 1):
            # Check both nodes at the junction
            for side, nid in [('left', node_ids[i]), ('right', node_ids[i + 1])]:
                aln = alignment_data.get(nid)
                if aln is None:
                    continue
                
                total = aln.get('total_reads', 0)
                if total < min_reads:
                    continue
                
                clipped = aln.get('clipped_reads', 0)
                split = aln.get('split_reads', 0)
                clip_ratio = clipped / total
                split_ratio = split / total
                combined_ratio = (clipped + split) / total
                
                if clip_ratio < clip_ratio_threshold and split_ratio < split_ratio_threshold:
                    continue
                
                signals = []
                
                if clip_ratio >= clip_ratio_threshold:
                    signals.append(MisassemblySignal(
                        signal_type="alignment_clipped_reads",
                        position=i,
                        score=min(1.0, clip_ratio / 0.4),
                        description=(
                            f"{side} node {nid}: {clip_ratio:.1%} clipped reads "
                            f"({clipped}/{total})"
                        ),
                        supporting_data={
                            'node_id': nid,
                            'side': side,
                            'clip_ratio': clip_ratio,
                            'clipped': clipped,
                            'total': total,
                        }
                    ))
                
                if split_ratio >= split_ratio_threshold:
                    signals.append(MisassemblySignal(
                        signal_type="alignment_split_reads",
                        position=i,
                        score=min(1.0, split_ratio / 0.3),
                        description=(
                            f"{side} node {nid}: {split_ratio:.1%} split reads "
                            f"({split}/{total})"
                        ),
                        supporting_data={
                            'node_id': nid,
                            'side': side,
                            'split_ratio': split_ratio,
                            'split': split,
                            'total': total,
                        }
                    ))
                
                if not signals:
                    continue
                
                if combined_ratio > 0.30 and total >= 20:
                    confidence = ConfidenceLevel.HIGH
                    mistype = MisassemblyType.CHIMERA
                elif combined_ratio > 0.20:
                    confidence = ConfidenceLevel.MEDIUM
                    mistype = MisassemblyType.UNKNOWN
                else:
                    confidence = ConfidenceLevel.LOW
                    mistype = MisassemblyType.UNKNOWN
                
                flag = MisassemblyFlag(
                    contig_id=contig_id,
                    start_pos=i * 1000,
                    end_pos=(i + 1) * 1000,
                    breakpoint_pos=i * 1000 + 500,
                    misassembly_type=mistype,
                    confidence=confidence,
                    detection_signals=signals,
                    left_node_id=node_ids[i],
                    right_node_id=node_ids[i + 1],
                )
                flags.append(flag)
                break  # One flag per junction, not per side
        
        return flags
    
    # ====================================================================
    # Detector 12: Graph Topology Inconsistencies
    # ====================================================================
    def _detect_graph_topology_inconsistencies(
        self,
        contig_id: str,
        node_ids: List[int],
        graph_topology: Dict[int, Dict[str, int]],
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies from assembly graph topology.
        
        In a correctly assembled linear path, interior nodes should have
        in-degree=1 and out-degree=1. Deviations indicate structural
        complexity that the path resolver may have traversed incorrectly:
          - High out-degree at a node → path chose one of multiple
            outgoing branches (possible mis-routing)
          - High in-degree → multiple paths converge (repeat collapse)
          - Both high → complex repeat region
        
        graph_topology: node_id -> {'in_degree': int, 'out_degree': int}
        """
        flags: List[MisassemblyFlag] = []
        
        if len(node_ids) < 3:
            return flags
        
        # Interior nodes (skip first and last — they're expected to have
        # degree-1 on one side)
        for i in range(1, len(node_ids) - 1):
            nid = node_ids[i]
            topo = graph_topology.get(nid)
            if topo is None:
                continue
            
            in_deg = topo.get('in_degree', 1)
            out_deg = topo.get('out_degree', 1)
            
            # Linear interior node = in_deg=1, out_deg=1 → skip
            if in_deg <= 1 and out_deg <= 1:
                continue
            
            signals = []
            max_degree = max(in_deg, out_deg)
            
            if out_deg > 1:
                signals.append(MisassemblySignal(
                    signal_type="graph_high_out_degree",
                    position=i,
                    score=min(1.0, (out_deg - 1) / 4.0),
                    description=(
                        f"Node {nid}: out_degree={out_deg} (branching point, "
                        f"path chose one of {out_deg} branches)"
                    ),
                    supporting_data={'node_id': nid, 'out_degree': out_deg}
                ))
            
            if in_deg > 1:
                signals.append(MisassemblySignal(
                    signal_type="graph_high_in_degree",
                    position=i,
                    score=min(1.0, (in_deg - 1) / 4.0),
                    description=(
                        f"Node {nid}: in_degree={in_deg} (convergence point, "
                        f"possible repeat collapse)"
                    ),
                    supporting_data={'node_id': nid, 'in_degree': in_deg}
                ))
            
            if not signals:
                continue
            
            # Confidence based on degree complexity
            if in_deg > 2 and out_deg > 2:
                confidence = ConfidenceLevel.MEDIUM
                mistype = MisassemblyType.REPEAT_COLLAPSE
            elif max_degree > 3:
                confidence = ConfidenceLevel.MEDIUM
                mistype = MisassemblyType.UNKNOWN
            else:
                confidence = ConfidenceLevel.LOW
                mistype = MisassemblyType.UNKNOWN
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=mistype,
                confidence=confidence,
                detection_signals=signals,
                left_node_id=node_ids[i - 1],
                right_node_id=node_ids[i],
            )
            flags.append(flag)
        
        return flags
    
    # ====================================================================
    # Detector 7: K-mer Spectrum Disruptions
    # ====================================================================
    def _detect_kmer_spectrum_disruptions(
        self,
        contig_id: str,
        node_ids: List[int],
        kmer_spectrum: Dict[int, Dict[str, float]],
        coverage_data: Optional[Dict[int, float]] = None,
    ) -> List[MisassemblyFlag]:
        """
        Detect misassemblies via k-mer spectrum analysis.
        
        Repeat collapses compress multiple genomic copies into a single
        assembled region.  This shows up as:
          - Abrupt increase in mean k-mer frequency (collapsed copies)
          - Drop in unique k-mer ratio (repetitive region suddenly unique
            on one side, highly repetitive on the other)
          - Elevated frequency stdev (mixed-copy k-mers from a misjoin)
        
        Conversely, repeat *expansions* cause an abrupt decrease in mean
        k-mer frequency.
        
        kmer_spectrum maps node_id -> {
            'unique_ratio': float,  # fraction of k-mers seen exactly once
            'mean_freq': float,     # mean k-mer multiplicity
            'max_freq': float,      # maximum k-mer frequency in node
            'freq_stdev': float,    # stdev of k-mer frequencies
        }
        """
        flags = []
        
        # Need at least two consecutive nodes to compare
        if len(node_ids) < 2:
            return flags
        
        # Thresholds
        freq_ratio_threshold = 2.5   # Mean-freq jump > 2.5× between neighbours
        unique_drop_threshold = 0.35  # Unique-ratio drop > 35 percentage-points
        stdev_spike_threshold = 3.0   # Stdev jump > 3× between neighbours
        
        for i in range(len(node_ids) - 1):
            left_id = node_ids[i]
            right_id = node_ids[i + 1]
            
            left_spec = kmer_spectrum.get(left_id)
            right_spec = kmer_spectrum.get(right_id)
            
            if left_spec is None or right_spec is None:
                continue
            
            left_mean = left_spec.get('mean_freq', 1.0)
            right_mean = right_spec.get('mean_freq', 1.0)
            left_unique = left_spec.get('unique_ratio', 1.0)
            right_unique = right_spec.get('unique_ratio', 1.0)
            left_stdev = left_spec.get('freq_stdev', 0.0)
            right_stdev = right_spec.get('freq_stdev', 0.0)
            
            # Guard against division by zero
            if left_mean <= 0:
                left_mean = 0.01
            if right_mean <= 0:
                right_mean = 0.01
            
            signals = []
            
            # --- Signal 1: Mean k-mer frequency jump ---
            freq_ratio = max(left_mean, right_mean) / min(left_mean, right_mean)
            if freq_ratio > freq_ratio_threshold:
                score = min(1.0, (freq_ratio - 1.0) / 5.0)
                higher_side = 'right' if right_mean > left_mean else 'left'
                signals.append(MisassemblySignal(
                    signal_type="kmer_frequency_jump",
                    position=i,
                    score=score,
                    description=(
                        f"K-mer mean frequency jump {freq_ratio:.1f}× "
                        f"(left={left_mean:.1f}, right={right_mean:.1f}) — "
                        f"possible repeat {'collapse' if higher_side == 'right' else 'expansion'}"
                    ),
                    supporting_data={
                        'left_mean_freq': left_mean,
                        'right_mean_freq': right_mean,
                        'freq_ratio': freq_ratio,
                    }
                ))
            
            # --- Signal 2: Unique k-mer ratio drop ---
            unique_diff = abs(left_unique - right_unique)
            if unique_diff > unique_drop_threshold:
                score = min(1.0, unique_diff / 0.6)
                signals.append(MisassemblySignal(
                    signal_type="kmer_unique_ratio_drop",
                    position=i,
                    score=score,
                    description=(
                        f"Unique k-mer ratio shift {unique_diff:.2f} "
                        f"(left={left_unique:.2f}, right={right_unique:.2f})"
                    ),
                    supporting_data={
                        'left_unique_ratio': left_unique,
                        'right_unique_ratio': right_unique,
                        'unique_diff': unique_diff,
                    }
                ))
            
            # --- Signal 3: Frequency stdev spike ---
            max_stdev = max(left_stdev, right_stdev)
            min_stdev = max(min(left_stdev, right_stdev), 0.01)  # avoid div-by-zero
            stdev_ratio = max_stdev / min_stdev
            if stdev_ratio > stdev_spike_threshold and max_stdev > 2.0:
                score = min(1.0, (stdev_ratio - 1.0) / 5.0)
                signals.append(MisassemblySignal(
                    signal_type="kmer_stdev_spike",
                    position=i,
                    score=score,
                    description=(
                        f"K-mer frequency stdev spike {stdev_ratio:.1f}× "
                        f"(left={left_stdev:.1f}, right={right_stdev:.1f})"
                    ),
                    supporting_data={
                        'left_stdev': left_stdev,
                        'right_stdev': right_stdev,
                        'stdev_ratio': stdev_ratio,
                    }
                ))
            
            if not signals:
                continue
            
            # Determine misassembly type and confidence
            has_freq_jump = any(s.signal_type == 'kmer_frequency_jump' for s in signals)
            has_unique_drop = any(s.signal_type == 'kmer_unique_ratio_drop' for s in signals)
            
            if has_freq_jump and has_unique_drop:
                # Strong signal: both frequency jump AND unique ratio change
                confidence = ConfidenceLevel.HIGH
                mistype = MisassemblyType.REPEAT_COLLAPSE
            elif has_freq_jump:
                confidence = ConfidenceLevel.MEDIUM
                mistype = MisassemblyType.REPEAT_COLLAPSE
            else:
                confidence = ConfidenceLevel.LOW
                mistype = MisassemblyType.UNKNOWN
            
            # If coverage data also shows a jump at this position, boost confidence
            if coverage_data:
                left_cov = coverage_data.get(left_id, 0)
                right_cov = coverage_data.get(right_id, 0)
                if left_cov > 0 and right_cov > 0:
                    cov_ratio = max(left_cov, right_cov) / min(left_cov, right_cov)
                    if cov_ratio > self.coverage_ratio_threshold:
                        # Coverage corroborates k-mer signal — upgrade to HIGH
                        if confidence == ConfidenceLevel.MEDIUM:
                            confidence = ConfidenceLevel.HIGH
                        elif confidence == ConfidenceLevel.LOW:
                            confidence = ConfidenceLevel.MEDIUM
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=mistype,
                confidence=confidence,
                detection_signals=signals,
                left_node_id=left_id,
                right_node_id=right_id,
                coverage_left=coverage_data.get(left_id, 0.0) if coverage_data else 0.0,
                coverage_right=coverage_data.get(right_id, 0.0) if coverage_data else 0.0,
            )
            flags.append(flag)
        
        return flags
    
    def _detect_strand_orientation_inconsistencies(
        self,
        contig_id: str,
        node_ids: List[int],
        strand_data: Dict[int, Dict[str, int]],
    ) -> List[MisassemblyFlag]:
        """
        Detect inversions via read strand orientation analysis.
        
        In a correctly assembled region, reads aligning to a node should have
        a consistent strand bias (predominantly forward or predominantly
        reverse, depending on orientation convention).  An *inversion*
        misassembly manifests as a sudden strand-bias flip: the dominant
        strand switches from forward to reverse (or vice-versa) at the
        junction between two adjacent nodes.
        
        strand_data maps node_id -> {
            'fwd': int,   # reads aligning in forward orientation
            'rev': int,   # reads aligning in reverse orientation
        }
        
        Detection logic:
        1. Compute strand bias per node: bias = fwd / (fwd + rev)
           - bias ~1.0  →  all forward
           - bias ~0.0  →  all reverse
           - bias ~0.5  →  balanced (ambiguous or palindromic region)
        2. A strand flip occurs when bias swings from > 0.7 to < 0.3
           (or vice-versa) across a junction.
        3. If both nodes also have strong strand bias (not near 0.5),
           the inversion signal is HIGH confidence.
        """
        flags = []
        
        if len(node_ids) < 2:
            return flags
        
        # Thresholds
        strong_bias = 0.70    # Above this = strongly forward-biased
        weak_bias = 0.30      # Below this = strongly reverse-biased
        min_reads = 5         # Minimum reads to trust strand estimate
        flip_threshold = 0.35  # Minimum bias change to call a flip
        
        # Pre-compute per-node strand bias
        bias = {}  # node_id -> float (0-1)
        depth = {}  # node_id -> total reads
        for nid in node_ids:
            sd = strand_data.get(nid)
            if sd is None:
                continue
            fwd = sd.get('fwd', 0)
            rev = sd.get('rev', 0)
            total = fwd + rev
            depth[nid] = total
            if total >= min_reads:
                bias[nid] = fwd / total
            # If < min_reads, we don't compute a bias (leave it out of dict)
        
        for i in range(len(node_ids) - 1):
            left_id = node_ids[i]
            right_id = node_ids[i + 1]
            
            left_bias = bias.get(left_id)
            right_bias = bias.get(right_id)
            
            if left_bias is None or right_bias is None:
                continue
            
            bias_change = abs(left_bias - right_bias)
            
            if bias_change < flip_threshold:
                continue
            
            # Check if this is a true strand flip (one side forward-heavy,
            # the other reverse-heavy) vs. just noisy
            left_strong = left_bias > strong_bias or left_bias < weak_bias
            right_strong = right_bias > strong_bias or right_bias < weak_bias
            
            # A flip is when one side is forward-dominated and the other
            # is reverse-dominated
            is_flip = (
                (left_bias > strong_bias and right_bias < weak_bias) or
                (left_bias < weak_bias and right_bias > strong_bias)
            )
            
            if not is_flip:
                # Bias change is present but not a clear flip — could be
                # noise, a strand-switch error, or a short inversion
                # Only flag if the change is very large
                if bias_change < 0.50:
                    continue
            
            # Build signal
            score = min(1.0, bias_change / 0.7)  # Normalize: 0.7 flip → 1.0
            
            signal = MisassemblySignal(
                signal_type="strand_orientation_flip",
                position=i,
                score=score,
                description=(
                    f"Strand bias flip: left={left_bias:.2f} → right={right_bias:.2f} "
                    f"(Δ={bias_change:.2f}, left_depth={depth.get(left_id, 0)}, "
                    f"right_depth={depth.get(right_id, 0)})"
                ),
                supporting_data={
                    'left_bias': left_bias,
                    'right_bias': right_bias,
                    'bias_change': bias_change,
                    'left_depth': depth.get(left_id, 0),
                    'right_depth': depth.get(right_id, 0),
                    'is_clean_flip': is_flip,
                }
            )
            
            # Confidence based on signal strength and read depth
            left_depth = depth.get(left_id, 0)
            right_depth = depth.get(right_id, 0)
            min_depth = min(left_depth, right_depth)
            
            if is_flip and left_strong and right_strong and min_depth >= 20:
                confidence = ConfidenceLevel.HIGH
            elif is_flip and min_depth >= 10:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            flag = MisassemblyFlag(
                contig_id=contig_id,
                start_pos=i * 1000,
                end_pos=(i + 1) * 1000,
                breakpoint_pos=i * 1000 + 500,
                misassembly_type=MisassemblyType.INVERSION,
                confidence=confidence,
                detection_signals=[signal],
                left_node_id=left_id,
                right_node_id=right_id,
            )
            flags.append(flag)
        
        return flags
    
    def _merge_overlapping_flags(
        self,
        flags: List[MisassemblyFlag]
    ) -> List[MisassemblyFlag]:
        """
        Merge overlapping flags into single comprehensive flags.
        
        Multiple detection signals for the same location are combined.
        """
        if not flags:
            return flags
        
        # Sort by position
        flags = sorted(flags, key=lambda f: f.start_pos)
        
        merged = []
        current = flags[0]
        
        for next_flag in flags[1:]:
            # Check for overlap
            if next_flag.start_pos <= current.end_pos:
                # Merge signals
                current.detection_signals.extend(next_flag.detection_signals)
                current.end_pos = max(current.end_pos, next_flag.end_pos)
                
                # Update scores
                current.edgewarden_min = min(current.edgewarden_min, next_flag.edgewarden_min)
                current.coverage_ratio = max(current.coverage_ratio, next_flag.coverage_ratio)
                current.ul_reads_conflicting += next_flag.ul_reads_conflicting
                current.hic_links_conflicting += next_flag.hic_links_conflicting
                
                # Update confidence (take highest)
                conf_order = [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
                if conf_order.index(next_flag.confidence) > conf_order.index(current.confidence):
                    current.confidence = next_flag.confidence
                
                # Recalculate composite score
                current.num_signals = len(current.detection_signals)
                current.composite_score = sum(s.score for s in current.detection_signals) / current.num_signals * 100
            else:
                # No overlap, add current and start new
                merged.append(current)
                current = next_flag
        
        # Add final flag
        merged.append(current)
        
        return merged
    
    def _filter_flags(
        self,
        flags: List[MisassemblyFlag]
    ) -> List[MisassemblyFlag]:
        """Filter flags by confidence and composite score."""
        conf_order = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.UNCERTAIN: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.HIGH: 2,
        }
        
        min_conf_val = conf_order[self.min_confidence]
        
        filtered = [
            f for f in flags
            if conf_order[f.confidence] >= min_conf_val and
               f.composite_score >= self.min_composite_score
        ]
        
        return filtered
    
    def _assign_correction_strategies(
        self,
        flags: List[MisassemblyFlag]
    ) -> List[MisassemblyFlag]:
        """
        Assign correction strategies and priorities based on evidence.
        """
        for flag in flags:
            # Determine strategy based on evidence
            has_ul = flag.ul_reads_conflicting > 0
            has_hic = flag.hic_links_conflicting > 0
            low_edgewarden = flag.edgewarden_min < 0.3
            high_cov_ratio = flag.coverage_ratio > 3.0
            
            # UL reads provide strong correction signal
            if has_ul and flag.ul_reads_conflicting >= 3:
                flag.correction_strategy = CorrectionStrategy.UL_VALIDATION
                flag.priority = 1  # Highest
            
            # Hi-C can guide scaffolding corrections
            elif has_hic:
                flag.correction_strategy = CorrectionStrategy.HIC_SCAFFOLDING
                flag.priority = 2
            
            # Low EdgeWarden with high coverage suggests chimera
            elif low_edgewarden and high_cov_ratio:
                flag.correction_strategy = CorrectionStrategy.BREAK_AND_REJOIN
                flag.priority = 2
            
            # Low EdgeWarden alone
            elif low_edgewarden:
                flag.correction_strategy = CorrectionStrategy.LOCAL_REASSEMBLY
                flag.priority = 3
            
            # Coverage ratio alone (repeat collapse/expansion)
            elif high_cov_ratio:
                flag.correction_strategy = CorrectionStrategy.LOCAL_REASSEMBLY
                flag.priority = 3
            
            # Weak evidence
            else:
                flag.correction_strategy = CorrectionStrategy.MANUAL_REVIEW
                flag.priority = 5
            
            # Add notes
            signals = ", ".join(set(s.signal_type for s in flag.detection_signals))
            flag.notes = f"Detected by: {signals}. {flag.num_signals} signals."
        
        return flags
    
    def generate_report(
        self,
        output_format: str = "tsv"
    ) -> str:
        """
        Generate misassembly report for all contigs.
        
        Args:
            output_format: "tsv", "json", or "bed"
        
        Returns:
            Formatted report string
        """
        if output_format == "tsv":
            return self._generate_tsv_report()
        elif output_format == "json":
            return self._generate_json_report()
        elif output_format == "bed":
            return self._generate_bed_report()
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _generate_tsv_report(self) -> str:
        """Generate TSV format report."""
        lines = [
            "\t".join([
                "contig_id",
                "start_pos",
                "end_pos",
                "breakpoint_pos",
                "misassembly_type",
                "confidence",
                "composite_score",
                "num_signals",
                "edgewarden_min",
                "coverage_ratio",
                "ul_conflicts",
                "hic_conflicts",
                "correction_strategy",
                "priority",
                "notes"
            ])
        ]
        
        for contig_id, flags in sorted(self.detected_flags.items()):
            for flag in flags:
                lines.append("\t".join([
                    str(flag.contig_id),
                    str(flag.start_pos),
                    str(flag.end_pos),
                    str(flag.breakpoint_pos if flag.breakpoint_pos else ""),
                    flag.misassembly_type.value,
                    flag.confidence.value,
                    f"{flag.composite_score:.1f}",
                    str(flag.num_signals),
                    f"{flag.edgewarden_min:.3f}",
                    f"{flag.coverage_ratio:.2f}",
                    str(flag.ul_reads_conflicting),
                    str(flag.hic_links_conflicting),
                    flag.correction_strategy.value,
                    str(flag.priority),
                    flag.notes
                ]))
        
        return "\n".join(lines)
    
    def _generate_bed_report(self) -> str:
        """Generate BED format for genome browser visualization."""
        lines = []
        
        for contig_id, flags in sorted(self.detected_flags.items()):
            for flag in flags:
                # BED: chrom start end name score strand
                color_map = {
                    ConfidenceLevel.HIGH: "255,0,0",  # Red
                    ConfidenceLevel.MEDIUM: "255,165,0",  # Orange
                    ConfidenceLevel.LOW: "255,255,0",  # Yellow
                }
                color = color_map.get(flag.confidence, "128,128,128")
                
                lines.append("\t".join([
                    str(flag.contig_id),
                    str(flag.start_pos),
                    str(flag.end_pos),
                    f"{flag.misassembly_type.value}_{flag.confidence.value}",
                    str(int(flag.composite_score)),
                    ".",
                    str(flag.start_pos),
                    str(flag.end_pos),
                    color
                ]))
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        import json
        
        report = {}
        for contig_id, flags in self.detected_flags.items():
            report[contig_id] = [
                {
                    "start_pos": f.start_pos,
                    "end_pos": f.end_pos,
                    "breakpoint_pos": f.breakpoint_pos,
                    "type": f.misassembly_type.value,
                    "confidence": f.confidence.value,
                    "composite_score": f.composite_score,
                    "signals": [
                        {
                            "type": s.signal_type,
                            "position": s.position,
                            "score": s.score,
                            "description": s.description,
                        }
                        for s in f.detection_signals
                    ],
                    "correction_strategy": f.correction_strategy.value,
                    "priority": f.priority,
                    "notes": f.notes,
                }
                for f in flags
            ]
        
        return json.dumps(report, indent=2)
    
    def export_for_downstream(
        self,
        contig_id: str,
        module: str = "ul_integration"
    ) -> Dict[str, Any]:
        """
        Export misassembly flags in format for downstream modules.
        
        Args:
            contig_id: Contig to export
            module: Target module ("ul_integration", "scaffolder", etc.)
        
        Returns:
            Module-specific formatted data
        """
        flags = self.detected_flags.get(contig_id, [])
        
        if module == "ul_integration":
            # Ultra-long read integration module format
            return {
                "contig_id": contig_id,
                "misassembly_regions": [
                    {
                        "start": f.start_pos,
                        "end": f.end_pos,
                        "breakpoint": f.breakpoint_pos,
                        "type": f.misassembly_type.value,
                        "confidence": f.confidence.value,
                        "correction_needed": f.correction_strategy != CorrectionStrategy.LEAVE_AS_IS,
                        "priority": f.priority,
                        "ul_evidence": {
                            "conflicting_reads": f.ul_reads_conflicting,
                            "supporting_reads": f.ul_reads_supporting,
                        },
                    }
                    for f in flags
                ],
            }
        
        elif module == "scaffolder":
            # Scaffolding module format
            return {
                "contig_id": contig_id,
                "breakpoints": [
                    {
                        "position": f.breakpoint_pos,
                        "confidence": f.confidence.value,
                        "hic_supported": f.hic_links_supporting > 0,
                    }
                    for f in flags
                    if f.breakpoint_pos is not None
                ],
            }
        
        else:
            # Generic format
            return {
                "contig_id": contig_id,
                "flags": [
                    {
                        "start": f.start_pos,
                        "end": f.end_pos,
                        "type": f.misassembly_type.value,
                        "confidence": f.confidence.value,
                        "score": f.composite_score,
                    }
                    for f in flags
                ]
            }


if __name__ == "__main__":
    print("Misassembly Detection & Flagging System")
    print("=" * 60)
    
    # Quick demonstration
    detector = MisassemblyDetector(
        min_confidence=ConfidenceLevel.LOW,
        min_composite_score=30.0,
    )
    
    # Mock data
    node_ids = [0, 1, 2, 3, 4, 5]
    edge_ids = ["e01", "e12", "e23", "e34", "e45"]
    
    edgewarden_scores = {
        "e01": 0.95,
        "e12": 0.88,
        "e23": 0.25,  # Low confidence!
        "e34": 0.82,
        "e45": 0.90,
    }
    
    coverage_data = {
        0: 30.0,
        1: 28.0,
        2: 29.0,
        3: 85.0,  # High coverage jump!
        4: 82.0,
        5: 80.0,
    }
    
    # K-mer spectrum data  (node 3 = collapsed repeat)
    kmer_spectrum = {
        0: {'unique_ratio': 0.85, 'mean_freq': 1.2, 'max_freq': 3, 'freq_stdev': 0.5},
        1: {'unique_ratio': 0.82, 'mean_freq': 1.3, 'max_freq': 4, 'freq_stdev': 0.6},
        2: {'unique_ratio': 0.80, 'mean_freq': 1.1, 'max_freq': 3, 'freq_stdev': 0.4},
        3: {'unique_ratio': 0.30, 'mean_freq': 4.5, 'max_freq': 12, 'freq_stdev': 3.2},  # Repeat!
        4: {'unique_ratio': 0.78, 'mean_freq': 1.4, 'max_freq': 4, 'freq_stdev': 0.7},
        5: {'unique_ratio': 0.81, 'mean_freq': 1.2, 'max_freq': 3, 'freq_stdev': 0.5},
    }
    
    # Strand orientation data  (node 4-5 = inversion)
    strand_data = {
        0: {'fwd': 25, 'rev': 3},
        1: {'fwd': 22, 'rev': 4},
        2: {'fwd': 27, 'rev': 2},
        3: {'fwd': 24, 'rev': 5},
        4: {'fwd': 3,  'rev': 28},  # Strand flip!
        5: {'fwd': 2,  'rev': 26},
    }
    
    # GC content data  (node 3 = GC shift at chimeric junction)
    gc_content = {
        0: 0.42,
        1: 0.40,
        2: 0.43,
        3: 0.71,  # Big GC shift!
        4: 0.69,
        5: 0.70,
    }
    
    # Graph topology data  (node 3 = branching repeat)
    graph_topology = {
        0: {'in_degree': 0, 'out_degree': 1},
        1: {'in_degree': 1, 'out_degree': 1},
        2: {'in_degree': 1, 'out_degree': 1},
        3: {'in_degree': 3, 'out_degree': 2},  # Complex node!
        4: {'in_degree': 1, 'out_degree': 1},
        5: {'in_degree': 1, 'out_degree': 0},
    }
    
    # Detect
    flags = detector.detect_in_path(
        "contig_1",
        node_ids,
        edge_ids,
        edgewarden_scores=edgewarden_scores,
        coverage_data=coverage_data,
        kmer_spectrum=kmer_spectrum,
        strand_data=strand_data,
        gc_content=gc_content,
        graph_topology=graph_topology,
    )
    
    print(f"\nDetected {len(flags)} putative misassemblies:")
    for flag in flags:
        print(f"\n  Location: {flag.start_pos}-{flag.end_pos}")
        print(f"  Type: {flag.misassembly_type.value}")
        print(f"  Confidence: {flag.confidence.value}")
        print(f"  Score: {flag.composite_score:.1f}")
        print(f"  Signals: {flag.num_signals}")
        print(f"  Strategy: {flag.correction_strategy.value}")
        print(f"  Priority: {flag.priority}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("TSV Report:")
    print("=" * 60)
    print(detector.generate_report("tsv"))

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
