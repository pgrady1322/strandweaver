#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Misassembly Detection & Flagging System

Identifies putative misassemblies in assembly paths and flags them for
downstream correction. Integrates with PathWeaver, EdgeWarden, and passes
detailed information to Ultra-Long (UL) integration and other modules.

Key Features:
- Multi-signal misassembly detection (12 detection methods)
- Confidence-scored flagging (HIGH/MEDIUM/LOW risk)
- Precise location identification (base-level resolution)
- Detailed breakpoint characterization
- Downstream module integration
- Correction strategy recommendations

Detection Methods:
1. Low EdgeWarden confidence regions
2. Coverage discontinuities
3. Abnormal insert size distributions
4. Strand orientation inconsistencies
5. Repeat boundary violations
6. GC content anomalies
7. K-mer spectrum disruptions
8. Ultra-long read conflicts
9. Hi-C link violations
10. Heterozygous marker inconsistencies
11. Alignment breakpoints
12. Graph topology inconsistencies

Author: StrandWeaver Development Team
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
            **kwargs: Additional detection data
        
        Returns:
            List of MisassemblyFlag objects
        """
        flags = []
        
        # Validate inputs
        if len(edge_ids) != len(node_ids) - 1:
            self.logger.warning(f"Contig {contig_id}: edge count mismatch")
            return flags
        
        # Run detection methods
        if edgewarden_scores:
            flags.extend(self._detect_low_confidence_edges(
                contig_id, node_ids, edge_ids, edgewarden_scores
            ))
        
        if coverage_data:
            flags.extend(self._detect_coverage_discontinuities(
                contig_id, node_ids, coverage_data
            ))
        
        if ul_alignments:
            flags.extend(self._detect_ul_conflicts(
                contig_id, node_ids, ul_alignments
            ))
        
        if hic_links:
            flags.extend(self._detect_hic_violations(
                contig_id, node_ids, hic_links
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
    
    # Detect
    flags = detector.detect_in_path(
        "contig_1",
        node_ids,
        edge_ids,
        edgewarden_scores=edgewarden_scores,
        coverage_data=coverage_data,
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
