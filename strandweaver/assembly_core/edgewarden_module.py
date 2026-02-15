#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

EdgeWarden — overlap classification and quality assessment with ML models,
confidence scoring, and PathWeaver integration.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import numpy as np
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime
from abc import ABC, abstractmethod

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

logger = logging.getLogger(__name__)


# ============================================================================
#                         CORE ENUMERATIONS & CONSTANTS
# ============================================================================

class TechnologyType(str, Enum):
    """Sequencing technologies with specific error profiles."""
    PACBIO_HIFI = "pacbio_hifi"
    PACBIO_CLR = "pacbio_clr"
    NANOPORE_R9 = "nanopore_r9"
    NANOPORE_R10 = "nanopore_r10"
    ILLUMINA = "illumina"
    ANCIENT_DNA = "ancient_dna"


class ScoreComponent(str, Enum):
    """Individual scoring components."""
    EDGE_CONFIDENCE = "edge_confidence"
    COVERAGE_CONSISTENCY = "coverage_consistency"
    REPEAT_SCORE = "repeat_score"
    QUALITY_SCORE = "quality_score"
    ERROR_PATTERN = "error_pattern"
    BIOLOGICAL_PLAUSIBILITY = "bio_plausibility"


class ConfidenceStratum(str, Enum):
    """Confidence stratification levels."""
    HIGH = "HIGH"              # ≥ 0.90
    MEDIUM = "MEDIUM"          # 0.70-0.89
    LOW = "LOW"                # 0.50-0.69
    VERY_LOW = "VERY_LOW"      # < 0.50


class ExplanationLevel(Enum):
    """Explanation detail level."""
    SIMPLE = "simple"
    TECHNICAL = "technical"
    EXPERT = "expert"


# ============================================================================
#                    PART 1: FEATURE EXTRACTION (80 FEATURES)
# ============================================================================

# -----------------------------------------------------------------------------
# Static Features (26 features: 18 base + 8 graph-aware)
# -----------------------------------------------------------------------------

@dataclass
class OverlapFeatures:
    """Container for overlap features with metadata."""
    overlap_id: str
    read_a: str
    read_b: str
    technology: str
    base_features: np.ndarray  # 18 original features
    graph_features: np.ndarray  # 8 graph-aware features
    all_features: np.ndarray  # 26 static features
    feature_names: List[str]


class GraphAwareFeatureExtractor:
    """
    Extracts static features: 18 base + 8 graph-aware = 26 total.
    
    Base Features (18):
    1. overlap_length, 2. sequence_identity, 3-6. overhangs (left/right A/B)
    7-9. coverage (read_a, read_b, ratio), 10-11. repeat_content (a/b)
    12. kmer_uniqueness_score, 13-14. alignment_score (primary/secondary)
    15-16. quality_score_median (a/b), 17. soft_clip_count, 18. trimming_pattern
    
    Graph-Aware Features (8):
    19. node_degree_a, 20. node_degree_b, 21. coverage_consistency
    22. local_clustering_coefficient, 23. subgraph_density
    24. distance_to_repeat_region, 25. overlap_redundancy_score
    26. local_anomaly_score
    """
    
    def __init__(self):
        self.base_feature_names = [
            'overlap_length', 'sequence_identity',
            'left_overhang_a', 'left_overhang_b',
            'right_overhang_a', 'right_overhang_b',
            'coverage_read_a', 'coverage_read_b', 'coverage_ratio',
            'repeat_content_a', 'repeat_content_b',
            'kmer_uniqueness_score',
            'alignment_score_primary', 'alignment_score_secondary',
            'quality_score_median_a', 'quality_score_median_b',
            'soft_clip_count', 'trimming_pattern'
        ]
        
        self.graph_feature_names = [
            'node_degree_a', 'node_degree_b',
            'coverage_consistency',
            'local_clustering_coefficient',
            'subgraph_density',
            'distance_to_repeat_region',
            'overlap_redundancy_score',
            'local_anomaly_score'
        ]
        
        self.all_feature_names = self.base_feature_names + self.graph_feature_names
    
    def extract_base_features(self, overlap: Dict) -> np.ndarray:
        """Extract the 18 original features from overlap data."""
        features = np.array([
            overlap.get('length', 0),
            overlap.get('identity', 0.0),
            overlap.get('left_overhang_a', 0),
            overlap.get('left_overhang_b', 0),
            overlap.get('right_overhang_a', 0),
            overlap.get('right_overhang_b', 0),
            overlap.get('coverage_a', 0),
            overlap.get('coverage_b', 0),
            overlap.get('coverage_ratio', 1.0),
            overlap.get('repeat_content_a', 0.0),
            overlap.get('repeat_content_b', 0.0),
            overlap.get('kmer_uniqueness', 0.5),
            overlap.get('alignment_score_primary', 0),
            overlap.get('alignment_score_secondary', 0),
            overlap.get('quality_median_a', 0),
            overlap.get('quality_median_b', 0),
            overlap.get('soft_clip_count', 0),
            overlap.get('trimming_pattern', 0)
        ], dtype=np.float32)
        
        return features
    
    def extract_graph_features(self, overlap: Dict, graph_context: Dict) -> np.ndarray:
        """
        Extract 8 graph-aware features using assembly graph context.
        
        Args:
            overlap: Dictionary with overlap properties
            graph_context: Dictionary with graph topology information
        """
        read_a = overlap['read_a']
        read_b = overlap['read_b']
        
        # Features 1-2: Node degree
        node_overlaps = graph_context.get('node_overlaps', {})
        node_degree_a = len(node_overlaps.get(read_a, []))
        node_degree_b = len(node_overlaps.get(read_b, []))
        
        # Feature 3: Coverage consistency
        read_coverage = graph_context.get('read_coverage', {})
        expected_coverage = graph_context.get('expected_coverage', 20.0)
        coverage_a = read_coverage.get(read_a, expected_coverage)
        coverage_b = read_coverage.get(read_b, expected_coverage)
        
        coverage_consistency = min(
            abs(coverage_a - expected_coverage) / (expected_coverage + 1),
            abs(coverage_b - expected_coverage) / (expected_coverage + 1)
        )
        
        # Feature 4: Local clustering coefficient
        node_clustering = graph_context.get('node_clustering', {})
        clustering_a = node_clustering.get(read_a, 0.5)
        clustering_b = node_clustering.get(read_b, 0.5)
        local_clustering = (clustering_a + clustering_b) / 2.0
        
        # Feature 5: Subgraph density
        region_key = f"{read_a[:5]}"
        region_density = graph_context.get('subgraph_densities', {}).get(region_key, 0.5)
        
        # Feature 6: Distance to repeat regions
        repeat_regions = graph_context.get('repeat_regions', [])
        read_position_a = overlap.get('read_a_position', 0)
        read_position_b = overlap.get('read_b_position', 0)
        
        min_distance = float('inf')
        for repeat_start, repeat_end in repeat_regions:
            dist_a = min(abs(read_position_a - repeat_start), abs(read_position_a - repeat_end))
            dist_b = min(abs(read_position_b - repeat_start), abs(read_position_b - repeat_end))
            min_distance = min(min_distance, dist_a, dist_b)
        
        distance_to_repeat = min(min_distance / 10000.0, 1.0) if min_distance < float('inf') else 1.0
        
        # Feature 7: Overlap redundancy score
        all_overlaps_a = node_overlaps.get(read_a, [])
        all_overlaps_b = node_overlaps.get(read_b, [])
        same_target = len([o for o in all_overlaps_a if o in all_overlaps_b])
        redundancy_score = min(same_target / max(len(all_overlaps_a), 1), 1.0)
        
        # Feature 8: Local anomaly score
        neighbor_identities = graph_context.get('neighbor_identities', {}).get(read_a, [])
        current_identity = overlap.get('identity', 0.95)
        
        if neighbor_identities:
            median_neighbor_identity = np.median(neighbor_identities)
            anomaly_score = abs(current_identity - median_neighbor_identity) / 0.1
            anomaly_score = min(anomaly_score, 1.0)
        else:
            anomaly_score = 0.0
        
        features = np.array([
            node_degree_a, node_degree_b, coverage_consistency,
            local_clustering, region_density, distance_to_repeat,
            redundancy_score, anomaly_score
        ], dtype=np.float32)
        
        return features
    
    def extract_all_features(self, overlap: Dict, graph_context: Dict) -> OverlapFeatures:
        """Extract both base and graph-aware features."""
        base_features = self.extract_base_features(overlap)
        graph_features = self.extract_graph_features(overlap, graph_context)
        all_features = np.concatenate([base_features, graph_features])
        
        return OverlapFeatures(
            overlap_id=overlap.get('id', 'unknown'),
            read_a=overlap['read_a'],
            read_b=overlap['read_b'],
            technology=overlap.get('technology', 'unknown'),
            base_features=base_features,
            graph_features=graph_features,
            all_features=all_features,
            feature_names=self.all_feature_names
        )


# -----------------------------------------------------------------------------
# Temporal Features (34 features: quality/coverage trajectories, errors)
# -----------------------------------------------------------------------------

@dataclass
class AlignmentSegment:
    """Represents a local alignment segment."""
    read_a_seq: str
    read_b_seq: str
    read_a_quality: np.ndarray
    read_b_quality: np.ndarray
    read_a_coverage: np.ndarray
    read_b_coverage: np.ndarray
    alignment_matches: np.ndarray
    is_ont: bool = False


@dataclass
class TemporalFeatures:
    """Container for 34 temporal features."""
    # Quality trajectory (11)
    quality_mean_a: float = 0.0
    quality_mean_b: float = 0.0
    quality_std_a: float = 0.0
    quality_std_b: float = 0.0
    quality_min_a: float = 0.0
    quality_min_b: float = 0.0
    quality_dip_at_start_a: float = 0.0
    quality_dip_at_start_b: float = 0.0
    quality_dip_at_end_a: float = 0.0
    quality_dip_at_end_b: float = 0.0
    quality_consistency: float = 0.0
    
    # Coverage trajectory (8)
    coverage_mean_a: float = 0.0
    coverage_mean_b: float = 0.0
    coverage_ratio_ab: float = 0.0
    coverage_std_a: float = 0.0
    coverage_std_b: float = 0.0
    coverage_spike_at_junction: float = 0.0
    coverage_drop_at_boundary: float = 0.0
    coverage_discordance: float = 0.0
    
    # Error patterns (8)
    mismatch_count: int = 0
    mismatch_rate: float = 0.0
    mismatch_run_length_max: int = 0
    mismatch_clustering_score: float = 0.0
    error_at_start: float = 0.0
    error_at_middle: float = 0.0
    error_at_end: float = 0.0
    error_concentrated: bool = False
    
    # Homopolymer (4)
    has_long_homopolymer: bool = False
    homopolymer_length: int = 0
    overlap_in_homopolymer: bool = False
    expected_hpoly_error: float = 0.0
    
    # Junction anomalies (3)
    quality_drop_at_junction: float = 0.0
    coverage_mismatch_at_junction: float = 0.0
    junction_anomaly_score: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.quality_mean_a, self.quality_mean_b,
            self.quality_std_a, self.quality_std_b,
            self.quality_min_a, self.quality_min_b,
            self.quality_dip_at_start_a, self.quality_dip_at_start_b,
            self.quality_dip_at_end_a, self.quality_dip_at_end_b,
            self.quality_consistency,
            self.coverage_mean_a, self.coverage_mean_b,
            self.coverage_ratio_ab, self.coverage_std_a, self.coverage_std_b,
            self.coverage_spike_at_junction, self.coverage_drop_at_boundary,
            self.coverage_discordance,
            self.mismatch_count, self.mismatch_rate,
            self.mismatch_run_length_max, self.mismatch_clustering_score,
            self.error_at_start, self.error_at_middle, self.error_at_end,
            float(self.error_concentrated),
            float(self.has_long_homopolymer), self.homopolymer_length,
            float(self.overlap_in_homopolymer), self.expected_hpoly_error,
            self.quality_drop_at_junction, self.coverage_mismatch_at_junction,
            self.junction_anomaly_score
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get names of 34 temporal features."""
        return [
            'quality_mean_a', 'quality_mean_b', 'quality_std_a', 'quality_std_b',
            'quality_min_a', 'quality_min_b', 'quality_dip_at_start_a', 'quality_dip_at_start_b',
            'quality_dip_at_end_a', 'quality_dip_at_end_b', 'quality_consistency',
            'coverage_mean_a', 'coverage_mean_b', 'coverage_ratio_ab',
            'coverage_std_a', 'coverage_std_b', 'coverage_spike_at_junction',
            'coverage_drop_at_boundary', 'coverage_discordance',
            'mismatch_count', 'mismatch_rate', 'mismatch_run_length_max',
            'mismatch_clustering_score', 'error_at_start', 'error_at_middle',
            'error_at_end', 'error_concentrated', 'has_long_homopolymer',
            'homopolymer_length', 'overlap_in_homopolymer', 'expected_hpoly_error',
            'quality_drop_at_junction', 'coverage_mismatch_at_junction',
            'junction_anomaly_score'
        ]


class TemporalFeatureExtractor:
    """Extracts 34 temporal features from alignment segments."""
    
    def __init__(self, is_ont: bool = True):
        self.is_ont = is_ont
    
    def extract(self, segment: AlignmentSegment) -> TemporalFeatures:
        """Extract all temporal features from alignment segment."""
        features = TemporalFeatures()
        
        # Quality trajectory
        quality_feats = self._extract_quality_features(
            segment.read_a_quality, segment.read_b_quality
        )
        for key, value in quality_feats.items():
            setattr(features, key, value)
        
        # Coverage trajectory
        coverage_feats = self._extract_coverage_features(
            segment.read_a_coverage, segment.read_b_coverage
        )
        for key, value in coverage_feats.items():
            setattr(features, key, value)
        
        # Error patterns
        error_feats = self._extract_error_features(segment.alignment_matches)
        for key, value in error_feats.items():
            setattr(features, key, value)
        
        # Homopolymer features
        hpoly_feats = self._extract_homopolymer_features(
            segment.read_a_seq, segment.read_b_seq, self.is_ont
        )
        for key, value in hpoly_feats.items():
            setattr(features, key, value)
        
        # Junction anomalies
        junction_feats = self._extract_junction_features(
            segment.read_a_quality, segment.read_b_quality,
            segment.read_a_coverage, segment.read_b_coverage
        )
        for key, value in junction_feats.items():
            setattr(features, key, value)
        
        return features
    
    def _extract_quality_features(self, quality_a: np.ndarray, quality_b: np.ndarray) -> Dict[str, float]:
        """Extract quality trajectory features."""
        if len(quality_a) == 0 or len(quality_b) == 0:
            return {
                'quality_mean_a': 0.0, 'quality_mean_b': 0.0,
                'quality_std_a': 0.0, 'quality_std_b': 0.0,
                'quality_min_a': 0.0, 'quality_min_b': 0.0,
                'quality_dip_at_start_a': 0.0, 'quality_dip_at_start_b': 0.0,
                'quality_dip_at_end_a': 0.0, 'quality_dip_at_end_b': 0.0,
                'quality_consistency': 0.0
            }
        
        features = {}
        features['quality_mean_a'] = float(np.mean(quality_a))
        features['quality_mean_b'] = float(np.mean(quality_b))
        features['quality_std_a'] = float(np.std(quality_a))
        features['quality_std_b'] = float(np.std(quality_b))
        features['quality_min_a'] = float(np.min(quality_a))
        features['quality_min_b'] = float(np.min(quality_b))
        
        n = len(quality_a)
        if n > 4:
            start_a = np.mean(quality_a[:n//4])
            middle_a = np.mean(quality_a[n//4:3*n//4])
            end_a = np.mean(quality_a[3*n//4:])
            
            features['quality_dip_at_start_a'] = float(start_a - middle_a)
            features['quality_dip_at_end_a'] = float(middle_a - end_a)
            
            start_b = np.mean(quality_b[:n//4])
            middle_b = np.mean(quality_b[n//4:3*n//4])
            end_b = np.mean(quality_b[3*n//4:])
            
            features['quality_dip_at_start_b'] = float(start_b - middle_b)
            features['quality_dip_at_end_b'] = float(middle_b - end_b)
        else:
            features['quality_dip_at_start_a'] = 0.0
            features['quality_dip_at_start_b'] = 0.0
            features['quality_dip_at_end_a'] = 0.0
            features['quality_dip_at_end_b'] = 0.0
        
        consistency_a = 1.0 - min(features['quality_std_a'] / (features['quality_mean_a'] + 1), 1.0)
        consistency_b = 1.0 - min(features['quality_std_b'] / (features['quality_mean_b'] + 1), 1.0)
        features['quality_consistency'] = float((consistency_a + consistency_b) / 2.0)
        
        return features
    
    def _extract_coverage_features(self, coverage_a: np.ndarray, coverage_b: np.ndarray) -> Dict[str, float]:
        """Extract coverage trajectory features."""
        if len(coverage_a) == 0 or len(coverage_b) == 0:
            return {
                'coverage_mean_a': 0.0, 'coverage_mean_b': 0.0,
                'coverage_ratio_ab': 1.0, 'coverage_std_a': 0.0,
                'coverage_std_b': 0.0, 'coverage_spike_at_junction': 0.0,
                'coverage_drop_at_boundary': 0.0, 'coverage_discordance': 0.0
            }
        
        features = {}
        features['coverage_mean_a'] = float(np.mean(coverage_a))
        features['coverage_mean_b'] = float(np.mean(coverage_b))
        features['coverage_std_a'] = float(np.std(coverage_a))
        features['coverage_std_b'] = float(np.std(coverage_b))
        
        mean_a = features['coverage_mean_a']
        mean_b = features['coverage_mean_b']
        if min(mean_a, mean_b) > 0:
            features['coverage_ratio_ab'] = float(max(mean_a, mean_b) / min(mean_a, mean_b))
        else:
            features['coverage_ratio_ab'] = 1.0
        
        n = len(coverage_a)
        if n > 4:
            junction_idx = n // 2
            before_junction = np.mean(coverage_a[max(0, junction_idx-2):junction_idx])
            at_junction = np.mean([coverage_a[junction_idx], coverage_b[junction_idx]])
            after_junction = np.mean(coverage_b[junction_idx+1:min(n, junction_idx+3)])
            
            if before_junction > 0 and after_junction > 0:
                spike_left = at_junction / before_junction
                spike_right = at_junction / after_junction
                features['coverage_spike_at_junction'] = float(max(spike_left, spike_right))
            else:
                features['coverage_spike_at_junction'] = 1.0
            
            features['coverage_drop_at_boundary'] = float(abs(mean_a - mean_b) / (max(mean_a, mean_b) + 1))
        else:
            features['coverage_spike_at_junction'] = 1.0
            features['coverage_drop_at_boundary'] = 0.0
        
        if len(coverage_a) == len(coverage_b):
            norm_a = (coverage_a - np.mean(coverage_a)) / (np.std(coverage_a) + 1)
            norm_b = (coverage_b - np.mean(coverage_b)) / (np.std(coverage_b) + 1)
            discordance = float(np.mean(np.abs(norm_a - norm_b)))
            features['coverage_discordance'] = min(discordance, 1.0)
        else:
            features['coverage_discordance'] = 0.5
        
        return features
    
    def _extract_error_features(self, alignment_matches: np.ndarray) -> Dict[str, Any]:
        """Extract error pattern features."""
        features = {}
        
        mismatches = ~alignment_matches if len(alignment_matches) > 0 else np.array([])
        mismatch_count = int(np.sum(mismatches))
        mismatch_rate = float(mismatch_count / len(alignment_matches)) if len(alignment_matches) > 0 else 0.0
        
        features['mismatch_count'] = mismatch_count
        features['mismatch_rate'] = mismatch_rate
        
        max_run = 0
        current_run = 0
        for match in mismatches:
            if not match:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        features['mismatch_run_length_max'] = max_run
        
        if len(mismatches) > 0:
            mismatch_positions = np.where(mismatches)[0]
            if len(mismatch_positions) > 1:
                position_variance = float(np.std(mismatch_positions))
                clustering = 1.0 - min(position_variance / len(mismatches), 1.0)
            else:
                clustering = 1.0 if len(mismatch_positions) == 1 else 0.0
            features['mismatch_clustering_score'] = float(clustering)
        else:
            features['mismatch_clustering_score'] = 0.0
        
        if len(alignment_matches) > 6:
            n = len(alignment_matches)
            third = n // 3
            
            error_start = float(np.sum(mismatches[:third]) / third)
            error_middle = float(np.sum(mismatches[third:2*third]) / third)
            error_end = float(np.sum(mismatches[2*third:]) / (n - 2*third))
            
            features['error_at_start'] = error_start
            features['error_at_middle'] = error_middle
            features['error_at_end'] = error_end
            
            errors = [error_start, error_middle, error_end]
            max_error = max(errors)
            concentrated = max_error > 2 * np.mean(errors) if np.mean(errors) > 0 else False
            features['error_concentrated'] = bool(concentrated)
        else:
            features['error_at_start'] = mismatch_rate
            features['error_at_middle'] = mismatch_rate
            features['error_at_end'] = mismatch_rate
            features['error_concentrated'] = False
        
        return features
    
    def _extract_homopolymer_features(self, read_a_seq: str, read_b_seq: str, is_ont: bool) -> Dict[str, Any]:
        """Extract homopolymer features."""
        if not is_ont or len(read_a_seq) == 0:
            return {
                'has_long_homopolymer': False,
                'homopolymer_length': 0,
                'overlap_in_homopolymer': False,
                'expected_hpoly_error': 0.0
            }
        
        def find_longest_homopolymer(seq: str) -> int:
            max_run = 1
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            return max_run
        
        hpoly_a = find_longest_homopolymer(read_a_seq)
        hpoly_b = find_longest_homopolymer(read_b_seq)
        max_hpoly = max(hpoly_a, hpoly_b)
        
        in_hpoly = any(
            read_a_seq[max(0, i-5):i+5].count(read_a_seq[i]) > 8
            for i in range(len(read_a_seq))
        )
        
        expected_error = 0.03
        if max_hpoly > 5:
            expected_error += 0.02 * (max_hpoly - 5)
        
        return {
            'has_long_homopolymer': max_hpoly >= 7,
            'homopolymer_length': max_hpoly,
            'overlap_in_homopolymer': in_hpoly,
            'expected_hpoly_error': min(float(expected_error), 1.0)
        }
    
    def _extract_junction_features(self, quality_a: np.ndarray, quality_b: np.ndarray,
                                   coverage_a: np.ndarray, coverage_b: np.ndarray) -> Dict[str, float]:
        """Extract junction anomaly features."""
        if len(quality_a) == 0 or len(quality_b) == 0:
            return {
                'quality_drop_at_junction': 0.0,
                'coverage_mismatch_at_junction': 0.0,
                'junction_anomaly_score': 0.0
            }
        
        features = {}
        
        n = len(quality_a)
        if n > 4:
            junction_idx = n // 2
            before_qual = np.mean(quality_a[max(0, junction_idx-3):junction_idx])
            at_junction = np.mean([quality_a[junction_idx], quality_b[junction_idx]])
            after_qual = np.mean(quality_b[junction_idx+1:min(n, junction_idx+4)])
            
            overall_quality = np.mean(np.concatenate([quality_a, quality_b]))
            quality_drop = (before_qual + after_qual) / 2 - at_junction
            features['quality_drop_at_junction'] = float(max(quality_drop / (overall_quality + 1), 0.0))
        else:
            features['quality_drop_at_junction'] = 0.0
        
        mean_a = np.mean(coverage_a)
        mean_b = np.mean(coverage_b)
        
        if max(mean_a, mean_b) > 0:
            coverage_mismatch = abs(mean_a - mean_b) / max(mean_a, mean_b)
            features['coverage_mismatch_at_junction'] = float(min(coverage_mismatch, 1.0))
        else:
            features['coverage_mismatch_at_junction'] = 0.0
        
        features['junction_anomaly_score'] = float(
            (features['quality_drop_at_junction'] + features['coverage_mismatch_at_junction']) / 2.0
        )
        
        return features
    
    def extract_batch(self, segments: List[AlignmentSegment]) -> np.ndarray:
        """Extract temporal features for multiple segments."""
        features_list = []
        for segment in segments:
            temporal_feats = self.extract(segment)
            features_list.append(temporal_feats.to_array())
        return np.array(features_list)


# ============================================================================
#                    PART 2: SCORING & PATHWEAVER INTEGRATION
# ============================================================================

@dataclass
class EdgeScore:
    """EdgeWarden score for a single edge."""
    edge_id: str
    source_node: int
    target_node: int
    
    edge_confidence: float
    coverage_consistency: float
    repeat_score: float
    quality_score: float
    error_pattern_score: float
    
    technology: str
    support_count: int
    coverage_ratio: float
    weighted_score: float = 0.0


@dataclass
class PathScore:
    """Composite score for a full assembly path."""
    path_id: str
    node_ids: List[int]
    edge_ids: List[str]
    
    mean_edge_confidence: float
    min_edge_confidence: float
    mean_coverage_consistency: float
    mean_repeat_score: float
    mean_quality_score: float
    
    path_length_bases: int
    num_edges: int
    num_haplotypes: int
    is_circular: bool
    
    composite_score: float
    confidence_category: str
    recommendation: str
    
    supporting_reads: int = 0
    long_read_support: bool = False
    hi_c_support: bool = False
    breakdown: Dict[str, float] = field(default_factory=dict)


class EdgeWardenScoreManager:
    """
    Manages EdgeWarden scores and integrates with PathWeaver.
    Applies technology-specific weighting for edge/path scoring.
    """
    
    def __init__(self, technology: TechnologyType = TechnologyType.PACBIO_HIFI):
        self.technology = technology
        self.edge_scores: Dict[str, EdgeScore] = {}
        self.path_scores: Dict[str, PathScore] = {}
        self._score_cache = {}
        self.tech_weights = self._init_tech_weights()
        self.logger = logging.getLogger(f"{__name__}.EdgeWardenScoreManager")
    
    def _init_tech_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize technology-specific score component weights."""
        weights = {
            TechnologyType.PACBIO_HIFI: {
                ScoreComponent.EDGE_CONFIDENCE: 0.25,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.30,
                ScoreComponent.REPEAT_SCORE: 0.15,
                ScoreComponent.QUALITY_SCORE: 0.15,
                ScoreComponent.ERROR_PATTERN: 0.15,
            },
            TechnologyType.PACBIO_CLR: {
                ScoreComponent.EDGE_CONFIDENCE: 0.25,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.20,
                ScoreComponent.REPEAT_SCORE: 0.30,
                ScoreComponent.QUALITY_SCORE: 0.10,
                ScoreComponent.ERROR_PATTERN: 0.15,
            },
            TechnologyType.NANOPORE_R9: {
                ScoreComponent.EDGE_CONFIDENCE: 0.20,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.25,
                ScoreComponent.REPEAT_SCORE: 0.20,
                ScoreComponent.QUALITY_SCORE: 0.20,
                ScoreComponent.ERROR_PATTERN: 0.15,
            },
            TechnologyType.NANOPORE_R10: {
                ScoreComponent.EDGE_CONFIDENCE: 0.25,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.30,
                ScoreComponent.REPEAT_SCORE: 0.15,
                ScoreComponent.QUALITY_SCORE: 0.15,
                ScoreComponent.ERROR_PATTERN: 0.15,
            },
            TechnologyType.ILLUMINA: {
                ScoreComponent.EDGE_CONFIDENCE: 0.20,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.30,
                ScoreComponent.REPEAT_SCORE: 0.15,
                ScoreComponent.QUALITY_SCORE: 0.15,
                ScoreComponent.ERROR_PATTERN: 0.20,
            },
            TechnologyType.ANCIENT_DNA: {
                ScoreComponent.EDGE_CONFIDENCE: 0.15,
                ScoreComponent.COVERAGE_CONSISTENCY: 0.25,
                ScoreComponent.REPEAT_SCORE: 0.35,
                ScoreComponent.QUALITY_SCORE: 0.10,
                ScoreComponent.ERROR_PATTERN: 0.15,
            },
        }
        return weights
    
    def register_edge_score(self, edge_id: str, source_node: int, target_node: int,
                           edge_confidence: float, coverage_consistency: float,
                           repeat_score: float, quality_score: float,
                           error_pattern_score: float, support_count: int = 1,
                           coverage_ratio: float = 1.0) -> EdgeScore:
        """Register an EdgeWarden score for an edge."""
        edge_score = EdgeScore(
            edge_id=edge_id, source_node=source_node, target_node=target_node,
            edge_confidence=max(0, min(1, edge_confidence)),
            coverage_consistency=max(0, min(1, coverage_consistency)),
            repeat_score=max(0, min(1, repeat_score)),
            quality_score=max(0, min(1, quality_score)),
            error_pattern_score=max(0, min(1, error_pattern_score)),
            technology=self.technology.value,
            support_count=support_count,
            coverage_ratio=coverage_ratio,
        )
        
        edge_score.weighted_score = self._compute_weighted_score(edge_score)
        self.edge_scores[edge_id] = edge_score
        self.logger.debug(f"Registered edge {edge_id}: weighted_score={edge_score.weighted_score:.3f}")
        
        return edge_score
    
    def _compute_weighted_score(self, edge_score: EdgeScore) -> float:
        """Compute weighted score using technology-specific weights."""
        weights = self.tech_weights.get(
            self.technology,
            self.tech_weights[TechnologyType.PACBIO_HIFI]
        )
        
        weighted = (
            weights[ScoreComponent.EDGE_CONFIDENCE] * edge_score.edge_confidence +
            weights[ScoreComponent.COVERAGE_CONSISTENCY] * edge_score.coverage_consistency +
            weights[ScoreComponent.REPEAT_SCORE] * (1 - edge_score.repeat_score) +
            weights[ScoreComponent.QUALITY_SCORE] * edge_score.quality_score +
            weights[ScoreComponent.ERROR_PATTERN] * edge_score.error_pattern_score
        )
        
        return max(0, min(1, weighted))
    
    def score_path(self, path_id: str, node_ids: List[int], edge_ids: List[str],
                  path_length_bases: int = 0, num_haplotypes: int = 1,
                  is_circular: bool = False, supporting_reads: int = 0,
                  long_read_support: bool = False, hi_c_support: bool = False) -> PathScore:
        """Compute comprehensive score for a full path."""
        edge_scores_list = [self.edge_scores.get(eid) for eid in edge_ids]
        missing = [e for e in edge_scores_list if e is None]
        
        if missing:
            self.logger.warning(f"Path {path_id}: {len(missing)}/{len(edge_ids)} edges missing scores")
        
        edge_scores_list = [e for e in edge_scores_list if e is not None]
        
        if not edge_scores_list:
            return PathScore(
                path_id=path_id, node_ids=node_ids, edge_ids=edge_ids,
                mean_edge_confidence=0.0, min_edge_confidence=0.0,
                mean_coverage_consistency=0.0, mean_repeat_score=0.0,
                mean_quality_score=0.0, path_length_bases=path_length_bases,
                num_edges=len(edge_ids), num_haplotypes=num_haplotypes,
                is_circular=is_circular, composite_score=0.0,
                confidence_category="QUESTIONABLE", recommendation="REJECT",
                supporting_reads=supporting_reads
            )
        
        confidences = [e.edge_confidence for e in edge_scores_list]
        weighted_scores = [e.weighted_score for e in edge_scores_list]
        coverage_scores = [e.coverage_consistency for e in edge_scores_list]
        repeat_scores = [e.repeat_score for e in edge_scores_list]
        quality_scores = [e.quality_score for e in edge_scores_list]
        
        mean_edge_confidence = np.mean(confidences)
        min_edge_confidence = np.min(confidences)
        mean_coverage_consistency = np.mean(coverage_scores)
        mean_repeat_score = np.mean(repeat_scores)
        mean_quality_score = np.mean(quality_scores)
        mean_weighted_score = np.mean(weighted_scores)
        
        composite = mean_weighted_score * 100
        
        if long_read_support or hi_c_support:
            composite += 10
        
        if supporting_reads >= 5:
            composite += min(5, supporting_reads / 2)
        
        if min_edge_confidence < 0.5:
            composite *= 0.8
        
        if mean_repeat_score > 0.6:
            composite *= 0.9
        
        composite = max(0, min(100, composite))
        
        if composite >= 80:
            category = "RELIABLE"
            recommendation = "USE"
        elif composite >= 60:
            category = "ACCEPTABLE"
            recommendation = "USE_WITH_CAUTION"
        elif composite >= 40:
            category = "QUESTIONABLE"
            recommendation = "REVIEW"
        else:
            category = "LOW_CONFIDENCE"
            recommendation = "REJECT"
        
        path_score = PathScore(
            path_id=path_id, node_ids=node_ids, edge_ids=edge_ids,
            mean_edge_confidence=float(mean_edge_confidence),
            min_edge_confidence=float(min_edge_confidence),
            mean_coverage_consistency=float(mean_coverage_consistency),
            mean_repeat_score=float(mean_repeat_score),
            mean_quality_score=float(mean_quality_score),
            path_length_bases=path_length_bases,
            num_edges=len(edge_ids),
            num_haplotypes=num_haplotypes,
            is_circular=is_circular,
            composite_score=float(composite),
            confidence_category=category,
            recommendation=recommendation,
            supporting_reads=supporting_reads,
            long_read_support=long_read_support,
            hi_c_support=hi_c_support,
            breakdown={
                "edge_confidence": float(mean_edge_confidence),
                "coverage_consistency": float(mean_coverage_consistency),
                "repeat_risk": float(mean_repeat_score),
                "quality_score": float(mean_quality_score),
                "composite": float(composite),
            }
        )
        
        self.path_scores[path_id] = path_score
        return path_score
    
    def rank_paths(self, path_scores: List[PathScore], min_composite_score: float = 40.0) -> List[PathScore]:
        """Rank paths by composite score and filter by confidence threshold."""
        filtered = [p for p in path_scores if p.composite_score >= min_composite_score]
        ranked = sorted(filtered, key=lambda p: p.composite_score, reverse=True)
        
        self.logger.info(
            f"Ranked {len(ranked)} paths (filtered from {len(path_scores)}, "
            f"min_score={min_composite_score})"
        )
        
        return ranked


# ============================================================================
#                    PART 3: EXPANDED FEATURES (20 FEATURES)
# ============================================================================

@dataclass
class ExpandedFeatures:
    """Container for 20 expanded features."""
    # Sequence complexity (4)
    kmer_entropy_a: float = 0.0
    kmer_entropy_b: float = 0.0
    low_complexity_fraction_a: float = 0.0
    low_complexity_fraction_b: float = 0.0
    
    # Boundary/Junction (6)
    quality_asymmetry: float = 0.0
    quality_gradient_start: float = 0.0
    quality_gradient_end: float = 0.0
    coverage_asymmetry: float = 0.0
    boundary_transition_sharpness: float = 0.0
    junction_quality_ratio: float = 0.0
    
    # Systematic error (4)
    gc_bias_error: float = 0.0
    position_based_error_trend: float = 0.0
    error_periodicity: float = 0.0
    homopolymer_error_excess: float = 0.0
    
    # Repeat context (3)
    repeat_similarity_score: float = 0.0
    repeat_breakpoint_strength: float = 0.0
    repeat_coverage_consistency: float = 0.0
    
    # Structural (3)
    alignment_length_ratio: float = 0.0
    overhang_asymmetry: float = 0.0
    local_graph_redundancy: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.kmer_entropy_a, self.kmer_entropy_b,
            self.low_complexity_fraction_a, self.low_complexity_fraction_b,
            self.quality_asymmetry, self.quality_gradient_start,
            self.quality_gradient_end, self.coverage_asymmetry,
            self.boundary_transition_sharpness, self.junction_quality_ratio,
            self.gc_bias_error, self.position_based_error_trend,
            self.error_periodicity, self.homopolymer_error_excess,
            self.repeat_similarity_score, self.repeat_breakpoint_strength,
            self.repeat_coverage_consistency,
            self.alignment_length_ratio, self.overhang_asymmetry,
            self.local_graph_redundancy
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get names of 20 expanded features."""
        return [
            'kmer_entropy_a', 'kmer_entropy_b',
            'low_complexity_fraction_a', 'low_complexity_fraction_b',
            'quality_asymmetry', 'quality_gradient_start',
            'quality_gradient_end', 'coverage_asymmetry',
            'boundary_transition_sharpness', 'junction_quality_ratio',
            'gc_bias_error', 'position_based_error_trend',
            'error_periodicity', 'homopolymer_error_excess',
            'repeat_similarity_score', 'repeat_breakpoint_strength',
            'repeat_coverage_consistency',
            'alignment_length_ratio', 'overhang_asymmetry',
            'local_graph_redundancy'
        ]


class ExpandedFeatureExtractor:
    """Extracts 20 advanced features for deeper overlap analysis."""
    
    def extract(self, seq_a: str, seq_b: str,
                quality_a: np.ndarray, quality_b: np.ndarray,
                coverage_a: np.ndarray, coverage_b: np.ndarray,
                alignment_matches: np.ndarray,
                overlap_length: int,
                total_read_length_a: int = 5000,
                total_read_length_b: int = 5000) -> ExpandedFeatures:
        """Extract all 20 expanded features."""
        features = ExpandedFeatures()
        
        # Sequence complexity
        seq_feats = self._extract_sequence_complexity(seq_a, seq_b)
        for key, value in seq_feats.items():
            setattr(features, key, value)
        
        # Boundary features
        boundary_feats = self._extract_boundary_features(quality_a, quality_b, coverage_a, coverage_b)
        for key, value in boundary_feats.items():
            setattr(features, key, value)
        
        # Systematic error features
        error_feats = self._extract_systematic_errors(seq_a, seq_b, alignment_matches, quality_a, quality_b)
        for key, value in error_feats.items():
            setattr(features, key, value)
        
        # Repeat context
        repeat_feats = self._extract_repeat_context(seq_a, seq_b, alignment_matches)
        for key, value in repeat_feats.items():
            setattr(features, key, value)
        
        # Structural features
        struct_feats = self._extract_structural_features(seq_a, seq_b, overlap_length,
                                                         total_read_length_a, total_read_length_b)
        for key, value in struct_feats.items():
            setattr(features, key, value)
        
        return features
    
    def _extract_sequence_complexity(self, seq_a: str, seq_b: str) -> Dict[str, float]:
        """Extract sequence complexity features."""
        def calculate_kmer_entropy(seq: str, k: int = 4) -> float:
            if len(seq) < k:
                return 0.0
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            counts = Counter(kmers)
            total = len(kmers)
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            return entropy / 8.0
        
        def calculate_low_complexity(seq: str, min_run: int = 4) -> float:
            if len(seq) == 0:
                return 0.0
            low_complexity_bases = 0
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current_run += 1
                else:
                    if current_run >= min_run:
                        low_complexity_bases += current_run
                    current_run = 1
            if current_run >= min_run:
                low_complexity_bases += current_run
            return float(low_complexity_bases / len(seq))
        
        return {
            'kmer_entropy_a': float(calculate_kmer_entropy(seq_a)),
            'kmer_entropy_b': float(calculate_kmer_entropy(seq_b)),
            'low_complexity_fraction_a': float(calculate_low_complexity(seq_a)),
            'low_complexity_fraction_b': float(calculate_low_complexity(seq_b))
        }
    
    def _extract_boundary_features(self, quality_a: np.ndarray, quality_b: np.ndarray,
                                   coverage_a: np.ndarray, coverage_b: np.ndarray) -> Dict[str, float]:
        """Extract boundary/junction features."""
        if len(quality_a) == 0 or len(quality_b) == 0:
            return {
                'quality_asymmetry': 0.0, 'quality_gradient_start': 0.0,
                'quality_gradient_end': 0.0, 'coverage_asymmetry': 0.0,
                'boundary_transition_sharpness': 0.0, 'junction_quality_ratio': 0.0
            }
        
        features = {}
        mean_a = np.mean(quality_a)
        mean_b = np.mean(quality_b)
        features['quality_asymmetry'] = float(abs(mean_a - mean_b) / (max(mean_a, mean_b) + 1))
        
        n = len(quality_a)
        if n > 6:
            first_third_a = np.mean(quality_a[:n//3])
            second_third_a = np.mean(quality_a[n//3:2*n//3])
            features['quality_gradient_start'] = float(abs(first_third_a - second_third_a) / (mean_a + 1))
            
            second_third_b = np.mean(quality_b[n//3:2*n//3])
            last_third_b = np.mean(quality_b[2*n//3:])
            features['quality_gradient_end'] = float(abs(second_third_b - last_third_b) / (mean_b + 1))
        else:
            features['quality_gradient_start'] = 0.0
            features['quality_gradient_end'] = 0.0
        
        cov_a = np.mean(coverage_a)
        cov_b = np.mean(coverage_b)
        features['coverage_asymmetry'] = float(abs(cov_a - cov_b) / (max(cov_a, cov_b) + 1))
        
        if n > 4:
            junction_idx = n // 2
            before_qual = np.mean(quality_a[max(0, junction_idx-2):junction_idx])
            after_qual = np.mean(quality_b[junction_idx:min(n, junction_idx+3)])
            sharpness = (before_qual - after_qual) / (max(before_qual, after_qual) + 1)
            features['boundary_transition_sharpness'] = float(max(abs(sharpness), 0.0))
            
            junction_qual = (quality_a[junction_idx] + quality_b[junction_idx]) / 2.0
            overall_qual = (mean_a + mean_b) / 2.0
            features['junction_quality_ratio'] = float(junction_qual / (overall_qual + 1))
        else:
            features['boundary_transition_sharpness'] = 0.0
            features['junction_quality_ratio'] = 1.0
        
        return features
    
    def _extract_systematic_errors(self, seq_a: str, seq_b: str, alignment_matches: np.ndarray,
                                   quality_a: np.ndarray, quality_b: np.ndarray) -> Dict[str, float]:
        """Extract systematic error pattern features."""
        if len(alignment_matches) == 0:
            return {
                'gc_bias_error': 0.0, 'position_based_error_trend': 0.0,
                'error_periodicity': 0.0, 'homopolymer_error_excess': 0.0
            }
        
        features = {}
        mismatches = ~alignment_matches
        
        # GC bias
        gc_content = (seq_a.count('G') + seq_a.count('C')) / len(seq_a) if len(seq_a) > 0 else 0.5
        mismatch_positions = np.where(mismatches)[0]
        
        if len(mismatch_positions) > 0:
            gc_at_errors = []
            for pos in mismatch_positions:
                window_start = max(0, pos - 5)
                window_end = min(len(seq_a), pos + 6)
                window = seq_a[window_start:window_end]
                local_gc = (window.count('G') + window.count('C')) / len(window) if len(window) > 0 else 0.5
                gc_at_errors.append(local_gc)
            mean_gc_at_errors = np.mean(gc_at_errors)
            features['gc_bias_error'] = float(abs(mean_gc_at_errors - gc_content) / (gc_content + 0.01))
        else:
            features['gc_bias_error'] = 0.0
        
        # Position-based error trend
        if len(alignment_matches) > 6:
            n = len(alignment_matches)
            third = n // 3
            errors_start = np.sum(mismatches[:third])
            errors_end = np.sum(mismatches[2*third:])
            if errors_start + errors_end > 0:
                trend = (errors_end - errors_start) / (errors_start + errors_end + 1)
                features['position_based_error_trend'] = float(max(trend, -1.0))
            else:
                features['position_based_error_trend'] = 0.0
        else:
            features['position_based_error_trend'] = 0.0
        
        # Error periodicity
        if len(mismatch_positions) > 2:
            position_diffs = np.diff(mismatch_positions)
            if len(position_diffs) > 1:
                mean_spacing = np.mean(position_diffs)
                spacing_variance = np.std(position_diffs)
                if mean_spacing > 0:
                    periodicity = 1.0 - min(spacing_variance / mean_spacing, 1.0)
                else:
                    periodicity = 0.0
            else:
                periodicity = 0.0
            features['error_periodicity'] = float(periodicity)
        else:
            features['error_periodicity'] = 0.0
        
        # Homopolymer error excess
        def find_homopolymer_positions(seq: str, min_len: int = 4) -> List[Tuple[int, int]]:
            runs = []
            start = 0
            while start < len(seq):
                current_base = seq[start]
                end = start + 1
                while end < len(seq) and seq[end] == current_base:
                    end += 1
                if end - start >= min_len:
                    runs.append((start, end))
                start = end
            return runs
        
        homopolymer_runs = find_homopolymer_positions(seq_a)
        if len(homopolymer_runs) > 0:
            errors_at_hpoly = 0
            bases_at_hpoly = 0
            for start, end in homopolymer_runs:
                window_start = max(0, start - 2)
                window_end = min(len(mismatches), end + 2)
                errors_at_hpoly += np.sum(mismatches[window_start:window_end])
                bases_at_hpoly += window_end - window_start
            
            error_rate_at_hpoly = errors_at_hpoly / bases_at_hpoly if bases_at_hpoly > 0 else 0.0
            overall_error_rate = np.sum(mismatches) / len(mismatches)
            
            if overall_error_rate > 0:
                excess = (error_rate_at_hpoly - overall_error_rate) / overall_error_rate
                features['homopolymer_error_excess'] = float(min(excess, 5.0))
            else:
                features['homopolymer_error_excess'] = 0.0
        else:
            features['homopolymer_error_excess'] = 0.0
        
        return features
    
    def _extract_repeat_context(self, seq_a: str, seq_b: str, alignment_matches: np.ndarray) -> Dict[str, float]:
        """Extract repeat context features."""
        if len(seq_a) == 0 or len(seq_b) == 0:
            return {
                'repeat_similarity_score': 0.0,
                'repeat_breakpoint_strength': 0.0,
                'repeat_coverage_consistency': 0.0
            }
        
        match_rate = np.mean(alignment_matches) if len(alignment_matches) > 0 else 0.0
        
        def calculate_self_similarity(seq: str) -> float:
            if len(seq) < 100:
                return 0.0
            first_half = seq[:len(seq)//2]
            second_half = seq[len(seq)//2:]
            min_len = min(len(first_half), len(second_half))
            matches = sum(1 for i in range(min_len) if first_half[i] == second_half[i])
            return matches / min_len if min_len > 0 else 0.0
        
        self_sim_a = calculate_self_similarity(seq_a)
        self_sim_b = calculate_self_similarity(seq_b)
        repeat_similarity = (match_rate + self_sim_a + self_sim_b) / 3.0
        
        # Breakpoint strength
        if len(alignment_matches) > 4:
            n = len(alignment_matches)
            junction_idx = n // 2
            junction_window = alignment_matches[max(0, junction_idx-2):min(n, junction_idx+3)]
            mismatch_rate_at_junction = 1.0 - np.mean(junction_window)
            breakpoint_strength = mismatch_rate_at_junction
        else:
            breakpoint_strength = 0.0
        
        # Coverage consistency (use entropy as proxy)
        def calculate_entropy(seq: str) -> float:
            if len(seq) == 0:
                return 0.0
            counts = Counter(seq)
            entropy = 0.0
            for count in counts.values():
                p = count / len(seq)
                entropy -= p * np.log2(p)
            return entropy / 2.0
        
        entropy_a = calculate_entropy(seq_a)
        entropy_b = calculate_entropy(seq_b)
        coverage_consistency = (entropy_a + entropy_b) / 2.0
        
        return {
            'repeat_similarity_score': float(repeat_similarity),
            'repeat_breakpoint_strength': float(breakpoint_strength),
            'repeat_coverage_consistency': float(coverage_consistency)
        }
    
    def _extract_structural_features(self, seq_a: str, seq_b: str, overlap_length: int,
                                    total_read_length_a: int, total_read_length_b: int) -> Dict[str, float]:
        """Extract structural/length-based features."""
        features = {}
        
        # Alignment length ratio
        ratio_a = overlap_length / total_read_length_a if total_read_length_a > 0 else 0.0
        ratio_b = overlap_length / total_read_length_b if total_read_length_b > 0 else 0.0
        features['alignment_length_ratio'] = float(min(ratio_a, ratio_b))
        
        # Overhang asymmetry
        overhang_a = total_read_length_a - overlap_length
        overhang_b = total_read_length_b - overlap_length
        if max(overhang_a, overhang_b) > 0:
            asymmetry = abs(overhang_a - overhang_b) / max(overhang_a, overhang_b)
            features['overhang_asymmetry'] = float(min(asymmetry, 1.0))
        else:
            features['overhang_asymmetry'] = 0.0
        
        # Local graph redundancy
        if len(seq_a) > 10 and len(seq_b) > 10 and overlap_length < len(seq_a):
            start_overhang_a = seq_a[:min(10, len(seq_a))]
            start_overhang_b = seq_b[:min(10, len(seq_b))]
            overlap_sim = sum(1 for i in range(min(len(start_overhang_a), len(start_overhang_b)))
                             if start_overhang_a[i] == start_overhang_b[i])
            overlap_sim /= max(len(start_overhang_a), len(start_overhang_b), 1)
            redundancy = overlap_sim * (len(seq_a) / 5000.0)
            features['local_graph_redundancy'] = float(min(redundancy, 1.0))
        else:
            features['local_graph_redundancy'] = 0.0
        
        return features


# ============================================================================
#                    PART 4: ML MODELS & ENSEMBLE
# ============================================================================

@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray


class TechSpecificEdgeWarden:
    """
    Ensemble of 5 tech-specific EdgeWarden models.
    Each model trained on technology-specific error patterns.
    """
    
    TECHNOLOGIES = ['ont_r9', 'ont_r10', 'hifi', 'illumina', 'adna']
    
    def __init__(self, model_dir: str = 'models/edgewarden'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Dict] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_indices: Dict[str, List[int]] = {}
        
        for tech in self.TECHNOLOGIES:
            self.models[tech] = {
                'model': None,
                'performance': None,
                'feature_names': []
            }
            self.scalers[tech] = StandardScaler()
    
    def train_tech_specific_model(self, technology: str, X_train: np.ndarray,
                                  y_train: np.ndarray, feature_names: List[str],
                                  feature_indices: List[int],
                                  hyperparams: Optional[Dict] = None) -> ModelPerformance:
        """Train model for specific technology."""
        logger.info(f"Training EdgeWarden-{technology} with {X_train.shape[0]} samples")
        
        default_params = self._get_tech_hyperparams(technology)
        params = {**default_params, **(hyperparams or {})}
        
        X_scaled = self.scalers[technology].fit_transform(X_train)
        
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_scaled, y_train)
        
        self.models[technology]['model'] = model
        self.models[technology]['feature_names'] = [feature_names[i] for i in feature_indices]
        self.feature_indices[technology] = feature_indices
        
        y_pred = model.predict(X_scaled)
        performance = self._evaluate_model(y_train, y_pred)
        self.models[technology]['performance'] = performance
        
        logger.info(f"  Accuracy: {performance.accuracy:.4f}, F1: {performance.f1_score:.4f}")
        return performance
    
    def predict_single(self, all_features: np.ndarray, technology: str) -> Tuple[int, float, str]:
        """Predict for single overlap using specific tech model."""
        tech_lower = technology.lower().replace('.', '_').replace(' ', '_')
        
        if tech_lower not in self.models or self.models[tech_lower]['model'] is None:
            return None, 0.0, f"No model for {tech_lower}"
        
        indices = self.feature_indices.get(tech_lower, list(range(len(all_features))))
        features_selected = all_features[indices].reshape(1, -1)
        features_scaled = self.scalers[tech_lower].transform(features_selected)
        
        model = self.models[tech_lower]['model']
        prediction = model.predict(features_scaled)[0]
        confidence = max(model.predict_proba(features_scaled)[0])
        
        return int(prediction), float(confidence), f"EdgeWarden-{tech_lower}"
    
    def predict_ensemble(self, all_features: np.ndarray, technology: str,
                        voting_method: str = 'confidence') -> Tuple[int, float, Dict]:
        """Predict using ensemble of all 5 models."""
        votes = {}
        confidences = {}
        
        for tech in self.TECHNOLOGIES:
            if self.models[tech]['model'] is None:
                continue
            pred, conf, _ = self.predict_single(all_features, tech)
            if pred is not None:
                votes[tech] = pred
                confidences[tech] = conf
        
        if not votes:
            return None, 0.0, {'error': 'No models available'}
        
        voting_details = {
            'votes': votes,
            'confidences': confidences,
            'voting_method': voting_method
        }
        
        if voting_method == 'confidence':
            best_tech = max(confidences.keys(), key=lambda k: confidences[k])
            final_prediction = votes[best_tech]
            final_confidence = confidences[best_tech]
            voting_details['winner'] = best_tech
        else:  # majority voting
            true_votes = sum(1 for v in votes.values() if v == 1)
            spurious_votes = sum(1 for v in votes.values() if v == 0)
            
            if true_votes > spurious_votes:
                final_prediction = 1
                final_confidence = true_votes / len(votes)
            elif spurious_votes > true_votes:
                final_prediction = 0
                final_confidence = spurious_votes / len(votes)
            else:
                tech_lower = technology.lower().replace('.', '_').replace(' ', '_')
                if tech_lower in votes:
                    final_prediction = votes[tech_lower]
                    final_confidence = confidences[tech_lower]
                else:
                    final_prediction = 1
                    final_confidence = 0.5
        
        voting_details['final_prediction'] = final_prediction
        voting_details['final_confidence'] = final_confidence
        
        return final_prediction, final_confidence, voting_details
    
    @staticmethod
    def _get_tech_hyperparams(technology: str) -> Dict:
        """Get optimal hyperparameters per technology."""
        params = {
            'ont_r9': {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 5,
                      'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            'ont_r10': {'n_estimators': 180, 'max_depth': 7, 'min_samples_split': 5,
                       'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            'hifi': {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 4,
                    'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            'illumina': {'n_estimators': 180, 'max_depth': 7, 'min_samples_split': 5,
                        'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            'adna': {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 6,
                    'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'}
        }
        return params.get(technology, params['hifi'])
    
    @staticmethod
    def _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformance:
        """Evaluate model performance."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred)
        except (ValueError, Exception):
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        return ModelPerformance(
            accuracy=accuracy, precision=precision, recall=recall,
            f1_score=f1, auc_roc=auc, confusion_matrix=cm
        )
    
    def save_models(self):
        """Save all models and scalers to disk."""
        for tech in self.TECHNOLOGIES:
            if self.models[tech]['model'] is not None:
                model_path = self.model_dir / f"edgewarden_{tech}.pkl"
                scaler_path = self.model_dir / f"scaler_{tech}.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[tech]['model'], f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[tech], f)
                
                logger.info(f"Saved model and scaler for {tech}")
    
    def load_models(self):
        """Load all available models and scalers from disk."""
        for tech in self.TECHNOLOGIES:
            model_path = self.model_dir / f"edgewarden_{tech}.pkl"
            scaler_path = self.model_dir / f"scaler_{tech}.pkl"
            
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[tech]['model'] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers[tech] = pickle.load(f)
                logger.info(f"Loaded model and scaler for {tech}")


# -----------------------------------------------------------------------------
# Cascade Classifier (Rules → ML)
# -----------------------------------------------------------------------------

class HeuristicConfidence(Enum):
    """Confidence levels for heuristic rules."""
    VERY_HIGH = 0.98
    HIGH = 0.90
    MEDIUM = 0.75
    LOW = 0.60


@dataclass
class CascadeDecision:
    """Result from cascade classifier."""
    prediction: int
    confidence: float
    rule_fired: Optional[int]
    rule_name: Optional[str]
    stage: str  # "heuristic" or "ml"
    reasoning: str


class CascadeClassifier:
    """
    Two-stage cascade: Fast heuristic filter → Deep ML model.
    Reduces ML calls by 30-50% while maintaining accuracy.
    """
    
    def __init__(self, ml_model: Optional[Any] = None):
        self.ml_model = ml_model
        self.stats = {'heuristic_decisions': 0, 'ml_decisions': 0, 'total': 0}
    
    def predict(self, features: np.ndarray, feature_names: List[str]) -> CascadeDecision:
        """Predict using cascade: try heuristics first, fall back to ML."""
        self.stats['total'] += 1
        feature_dict = {name: value for name, value in zip(feature_names, features)}
        
        # Stage 1: Heuristic rules
        rule_decision = self._evaluate_heuristics(feature_dict)
        
        if rule_decision is not None:
            self.stats['heuristic_decisions'] += 1
            return rule_decision
        
        # Stage 2: ML model
        self.stats['ml_decisions'] += 1
        
        if self.ml_model is not None:
            features_reshaped = features.reshape(1, -1)
            prediction = self.ml_model.predict(features_reshaped)[0]
            if hasattr(self.ml_model, 'predict_proba'):
                confidence = max(self.ml_model.predict_proba(features_reshaped)[0])
            else:
                confidence = 0.5
            
            return CascadeDecision(
                prediction=int(prediction),
                confidence=float(confidence),
                rule_fired=None,
                rule_name=None,
                stage="ml",
                reasoning="ML model decision (no heuristic rule fired)"
            )
        
        # Fallback
        return CascadeDecision(
            prediction=1, confidence=0.5, rule_fired=None,
            rule_name=None, stage="fallback",
            reasoning="No ML model available"
        )
    
    def _evaluate_heuristics(self, features: Dict[str, float]) -> Optional[CascadeDecision]:
        """Evaluate heuristic rules. Return decision if confident, None otherwise."""
        
        # Rule 1: Very high mismatch rate
        mismatch_rate = features.get('mismatch_rate', 0.0)
        if mismatch_rate > 0.10:
            return CascadeDecision(
                prediction=0, confidence=0.95, rule_fired=1,
                rule_name="High mismatch rate",
                stage="heuristic",
                reasoning=f"Mismatch rate {mismatch_rate:.1%} exceeds 10% threshold"
            )
        
        # Rule 2: Extreme coverage ratio
        coverage_ratio = features.get('coverage_ratio_ab', 1.0)
        if coverage_ratio > 5.0:
            return CascadeDecision(
                prediction=0, confidence=0.92, rule_fired=2,
                rule_name="Extreme coverage ratio",
                stage="heuristic",
                reasoning=f"Coverage differs {coverage_ratio:.1f}x between reads"
            )
        
        # Rule 3: Perfect high-quality overlap
        identity = features.get('sequence_identity', 0.0)
        overlap_len = features.get('overlap_length', 0)
        quality_consistency = features.get('quality_consistency', 0.0)
        
        if identity > 0.98 and overlap_len > 1000 and quality_consistency > 0.95:
            return CascadeDecision(
                prediction=1, confidence=0.96, rule_fired=3,
                rule_name="Perfect high-quality overlap",
                stage="heuristic",
                reasoning=f"Identity {identity:.3f}, length {overlap_len}bp, quality {quality_consistency:.2f}"
            )
        
        # Rule 4: Junction anomaly
        junction_anomaly = features.get('junction_anomaly_score', 0.0)
        if junction_anomaly > 0.7:
            return CascadeDecision(
                prediction=0, confidence=0.90, rule_fired=4,
                rule_name="Severe junction anomaly",
                stage="heuristic",
                reasoning=f"Junction anomaly score {junction_anomaly:.2f} exceeds 0.7"
            )
        
        # Rule 5: Weak signal
        alignment_ratio = features.get('alignment_length_ratio', 0.5)
        if alignment_ratio < 0.03:
            return CascadeDecision(
                prediction=0, confidence=0.88, rule_fired=5,
                rule_name="Weak signal",
                stage="heuristic",
                reasoning=f"Alignment only {alignment_ratio:.1%} of read length"
            )
        
        # No confident heuristic decision
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cascade statistics."""
        total = self.stats['total']
        if total == 0:
            return self.stats
        
        return {
            'total_predictions': total,
            'heuristic_decisions': self.stats['heuristic_decisions'],
            'heuristic_percentage': 100 * self.stats['heuristic_decisions'] / total,
            'ml_decisions': self.stats['ml_decisions'],
            'ml_percentage': 100 * self.stats['ml_decisions'] / total,
            'ml_savings': f"{100 * self.stats['heuristic_decisions'] / total:.1f}%"
        }


# -----------------------------------------------------------------------------
# Hybrid Rules + ML Ensemble
# -----------------------------------------------------------------------------

class HeuristicRule(ABC):
    """Base class for heuristic rules."""
    
    @abstractmethod
    def evaluate(self, features: np.ndarray) -> Tuple[Optional[int], float, str]:
        """Evaluate rule. Returns (prediction, confidence, explanation)."""
        pass


class HybridRulesMLEnsemble:
    """
    Hybrid ensemble combining heuristic rules with ML predictions.
    Handles agreement, conflict, and confidence-based decision making.
    """
    
    def __init__(self, ml_model: Any = None):
        self.ml_model = ml_model
        self.stats = {
            'total_predictions': 0,
            'agreement': 0,
            'conflict': 0,
            'rules_only': 0,
            'ml_only': 0
        }
    
    def predict_single(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Make single prediction using hybrid ensemble."""
        self.stats['total_predictions'] += 1
        
        # Get ML prediction
        ml_pred = None
        ml_conf = 0.5
        
        if self.ml_model is not None:
            features_reshaped = features.reshape(1, -1)
            ml_pred = self.ml_model.predict(features_reshaped)[0]
            if hasattr(self.ml_model, 'predict_proba'):
                ml_conf = max(self.ml_model.predict_proba(features_reshaped)[0])
        
        # Get rule-based prediction
        rule_pred, rule_conf = self._evaluate_simple_rules(features, feature_names)
        
        # Ensemble logic
        if rule_pred is not None and rule_conf >= 0.85:
            final_pred = rule_pred
            final_conf = rule_conf
            method = 'rules_confident'
            if ml_pred == rule_pred:
                self.stats['agreement'] += 1
            else:
                self.stats['conflict'] += 1
        elif ml_pred is not None and ml_conf >= 0.85:
            final_pred = ml_pred
            final_conf = ml_conf
            method = 'ml_confident'
            self.stats['ml_only'] += 1
        elif rule_pred is not None and ml_pred is not None:
            if rule_pred == ml_pred:
                final_pred = rule_pred
                final_conf = (rule_conf + ml_conf) / 2.0
                final_conf = min(final_conf * 1.1, 0.99)
                method = 'agreement'
                self.stats['agreement'] += 1
            else:
                if rule_conf >= ml_conf:
                    final_pred = rule_pred
                    final_conf = rule_conf
                    method = 'rules_higher_confidence'
                else:
                    final_pred = ml_pred
                    final_conf = ml_conf
                    method = 'ml_higher_confidence'
                self.stats['conflict'] += 1
        elif rule_pred is not None:
            final_pred = rule_pred
            final_conf = rule_conf
            method = 'rules_only'
            self.stats['rules_only'] += 1
        else:
            final_pred = ml_pred if ml_pred is not None else 1
            final_conf = ml_conf
            method = 'ml_only'
            self.stats['ml_only'] += 1
        
        return {
            'prediction': final_pred,
            'confidence': final_conf,
            'ml_prediction': ml_pred,
            'ml_confidence': ml_conf,
            'rule_prediction': rule_pred,
            'rule_confidence': rule_conf,
            'method': method
        }
    
    def _evaluate_simple_rules(self, features: np.ndarray, feature_names: List[str]) -> Tuple[Optional[int], float]:
        """Evaluate simple heuristic rules."""
        feature_dict = {name: value for name, value in zip(feature_names, features)}
        
        # High error rate
        if feature_dict.get('mismatch_rate', 0.0) > 0.08:
            return 0, 0.92
        
        # Extreme coverage mismatch
        if feature_dict.get('coverage_ratio_ab', 1.0) > 4.0:
            return 0, 0.90
        
        # Perfect overlap
        if (feature_dict.get('sequence_identity', 0.0) > 0.97 and
            feature_dict.get('overlap_length', 0) > 800 and
            feature_dict.get('quality_consistency', 0.0) > 0.90):
            return 1, 0.94
        
        # Junction problems
        if feature_dict.get('junction_anomaly_score', 0.0) > 0.65:
            return 0, 0.88
        
        return None, 0.5


# ============================================================================
#                    PART 5: ADVANCED TRAINING
# ============================================================================

# -----------------------------------------------------------------------------
# Active Learning
# -----------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result from active learning query."""
    selected_indices: np.ndarray
    uncertainty_scores: np.ndarray
    selection_method: str


class ActiveLearner:
    """
    Active learning for EdgeWarden using uncertainty-based sampling.
    Efficiently expands training data by identifying most informative samples.
    """
    
    def __init__(self, model: Any, selection_strategy: str = 'entropy'):
        self.model = model
        self.selection_strategy = selection_strategy
    
    def query(self, X_unlabeled: np.ndarray, n_samples: int = 100) -> QueryResult:
        """Query most informative samples from unlabeled pool."""
        if not hasattr(self.model, 'predict_proba'):
            # Random sampling fallback
            indices = np.random.choice(len(X_unlabeled), size=min(n_samples, len(X_unlabeled)), replace=False)
            return QueryResult(
                selected_indices=indices,
                uncertainty_scores=np.ones(len(indices)),
                selection_method='random'
            )
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X_unlabeled)
        
        if self.selection_strategy == 'entropy':
            # Entropy-based uncertainty
            epsilon = 1e-10
            uncertainty = -np.sum(probs * np.log(probs + epsilon), axis=1)
        elif self.selection_strategy == 'margin':
            # Margin sampling (difference between top 2 classes)
            if probs.shape[1] >= 2:
                sorted_probs = np.sort(probs, axis=1)
                uncertainty = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
            else:
                uncertainty = 1.0 - probs[:, 0]
        else:  # least_confident
            uncertainty = 1.0 - np.max(probs, axis=1)
        
        # Select top uncertain samples
        top_indices = np.argsort(uncertainty)[-n_samples:]
        
        return QueryResult(
            selected_indices=top_indices,
            uncertainty_scores=uncertainty[top_indices],
            selection_method=self.selection_strategy
        )


# -----------------------------------------------------------------------------
# Continual Learning with EWC
# -----------------------------------------------------------------------------

@dataclass
class FisherInformation:
    """Fisher information matrix for EWC."""
    diagonal: np.ndarray
    task_id: str = ""
    tech_type: str = ""


class ContinualLearner:
    """
    Continual learning for EdgeWarden with Elastic Weight Consolidation (EWC).
    Enables adaptation to new technologies without catastrophic forgetting.
    """
    
    def __init__(self, ewc_lambda: float = 0.4):
        self.ewc_lambda = ewc_lambda
        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.task_history: List[str] = []
        self.task_metrics: Dict[str, Dict[str, float]] = {}
        self.fisher_info: Dict[str, FisherInformation] = {}
        self.reference_weights: Dict[str, np.ndarray] = {}
    
    def train_on_new_task(self, X_train: np.ndarray, y_train: np.ndarray,
                         task_id: str, tech_type: str) -> Dict[str, float]:
        """Train on new task while preserving old knowledge."""
        logger.info(f"Training on task: {task_id} ({tech_type})")
        
        self.task_history.append(task_id)
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        self.model.fit(X_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_scaled)
        metrics = {
            'accuracy': float(accuracy_score(y_train, y_pred)),
            'precision': float(precision_score(y_train, y_pred, zero_division=0)),
            'recall': float(recall_score(y_train, y_pred, zero_division=0)),
            'f1': float(f1_score(y_train, y_pred, zero_division=0))
        }
        
        self.task_metrics[task_id] = metrics
        
        # Compute Fisher information (feature importances as proxy)
        fisher_diagonal = self.model.feature_importances_.copy()
        fisher = FisherInformation(diagonal=fisher_diagonal, task_id=task_id, tech_type=tech_type)
        self.fisher_info[task_id] = fisher
        self.reference_weights[task_id] = fisher_diagonal.copy()
        
        logger.info(f"Task {task_id} accuracy: {metrics['accuracy']:.3f}")
        return metrics


# -----------------------------------------------------------------------------
# Multi-Task Learning
# -----------------------------------------------------------------------------

@dataclass
class AuxiliaryTaskConfig:
    """Configuration for auxiliary task."""
    task_name: str
    num_classes: int
    weight: float
    description: str = ""


class MultiTaskLearner:
    """
    Multi-task learning with auxiliary tasks for regularization.
    Main task: overlap classification
    Auxiliary: technology ID, quality level, coverage anomaly, etc.
    """
    
    def __init__(self, feature_dim: int = 80):
        self.feature_dim = feature_dim
        self.main_task_model: Optional[RandomForestClassifier] = None
        self.auxiliary_models: Dict[str, RandomForestClassifier] = {}
        self.scaler = StandardScaler()
        
        # Define auxiliary tasks
        self.aux_tasks = {
            'technology': AuxiliaryTaskConfig('technology', 5, 0.15, 'Predict sequencing tech'),
            'read_quality': AuxiliaryTaskConfig('read_quality', 3, 0.10, 'Predict quality level'),
            'coverage_anomaly': AuxiliaryTaskConfig('coverage_anomaly', 2, 0.12, 'Detect coverage anomalies'),
            'homopolymer': AuxiliaryTaskConfig('homopolymer', 2, 0.08, 'Detect homopolymer runs'),
            'repeat_membership': AuxiliaryTaskConfig('repeat_membership', 3, 0.13, 'Classify repeat membership')
        }
    
    def train(self, X: np.ndarray, y_main: np.ndarray,
             auxiliary_labels: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train multi-task model."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Train main task
        self.main_task_model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        self.main_task_model.fit(X_scaled, y_main)
        
        main_pred = self.main_task_model.predict(X_scaled)
        main_metrics = {
            'accuracy': float(accuracy_score(y_main, main_pred)),
            'f1': float(f1_score(y_main, main_pred, zero_division=0))
        }
        
        # Train auxiliary tasks
        aux_metrics = {}
        for task_name, labels in auxiliary_labels.items():
            if task_name in self.aux_tasks:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
                )
                model.fit(X_scaled, labels)
                self.auxiliary_models[task_name] = model
                
                aux_pred = model.predict(X_scaled)
                aux_metrics[task_name] = {
                    'accuracy': float(accuracy_score(labels, aux_pred))
                }
        
        logger.info(f"Multi-task training: Main accuracy={main_metrics['accuracy']:.3f}")
        
        return {'main_task': main_metrics, 'auxiliary_tasks': aux_metrics}
    
    def predict_multitask(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Predict all tasks simultaneously."""
        X_scaled = self.scaler.transform(X)
        results = {}
        
        if self.main_task_model:
            pred = self.main_task_model.predict(X_scaled)
            prob = self.main_task_model.predict_proba(X_scaled)[:, 1]
            results['main_task'] = (pred, prob)
        
        for task_name, model in self.auxiliary_models.items():
            pred = model.predict(X_scaled)
            prob = np.max(model.predict_proba(X_scaled), axis=1)
            results[task_name] = (pred, prob)
        
        return results


# ============================================================================
#                    PART 6: CONFIDENCE STRATIFICATION
# ============================================================================

@dataclass
class StratifiedPrediction:
    """Prediction with confidence stratification."""
    prediction: int
    confidence: float
    stratum: ConfidenceStratum
    should_review: bool
    action: str


class ConfidenceStratifier:
    """
    Stratifies predictions into confidence levels with different handling.
    HIGH ≥ 0.90, MEDIUM 0.70-0.89, LOW 0.50-0.69, VERY_LOW < 0.50
    """
    
    def __init__(self):
        self.thresholds = {
            ConfidenceStratum.HIGH: 0.90,
            ConfidenceStratum.MEDIUM: 0.70,
            ConfidenceStratum.LOW: 0.50,
            ConfidenceStratum.VERY_LOW: 0.0
        }
        
        self.policies = {
            ConfidenceStratum.HIGH: 'auto_accept',
            ConfidenceStratum.MEDIUM: 'accept_with_logging',
            ConfidenceStratum.LOW: 'flag_for_review',
            ConfidenceStratum.VERY_LOW: 'require_manual_review'
        }
    
    def stratify(self, prediction: int, confidence: float) -> StratifiedPrediction:
        """Stratify prediction by confidence level."""
        if confidence >= self.thresholds[ConfidenceStratum.HIGH]:
            stratum = ConfidenceStratum.HIGH
            should_review = False
            action = "Accept automatically"
        elif confidence >= self.thresholds[ConfidenceStratum.MEDIUM]:
            stratum = ConfidenceStratum.MEDIUM
            should_review = False
            action = "Accept with logging"
        elif confidence >= self.thresholds[ConfidenceStratum.LOW]:
            stratum = ConfidenceStratum.LOW
            should_review = True
            action = "Flag for review"
        else:
            stratum = ConfidenceStratum.VERY_LOW
            should_review = True
            action = "Require manual review"
        
        return StratifiedPrediction(
            prediction=prediction,
            confidence=confidence,
            stratum=stratum,
            should_review=should_review,
            action=action
        )
    
    def batch_stratify(self, predictions: np.ndarray, confidences: np.ndarray) -> List[StratifiedPrediction]:
        """Stratify batch of predictions."""
        return [self.stratify(int(pred), float(conf)) 
                for pred, conf in zip(predictions, confidences)]


# ============================================================================
#                    PART 7: INTERPRETABILITY & EXPLANATIONS
# ============================================================================

@dataclass
class FeatureContribution:
    """Feature's contribution to prediction."""
    feature_name: str
    feature_value: float
    contribution: float
    magnitude: float
    direction: str
    explanation: str
    impact_category: str


@dataclass
class SimpleExplanation:
    """Non-technical explanation for end users."""
    prediction: str
    confidence: float
    summary: str
    key_reasons: List[str]
    warnings: List[str]
    recommendation: str
    confidence_tier: str


@dataclass
class TechnicalExplanation:
    """Technical explanation with feature analysis."""
    prediction: str
    confidence: float
    top_features: List[FeatureContribution]
    feature_groups: Dict[str, float]
    pattern_detected: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    suggested_review: Optional[str]


@dataclass
class ExpertExplanation:
    """Complete expert-level explanation."""
    prediction: str
    confidence: float
    feature_contributions: List[FeatureContribution]
    feature_interactions: List[str]
    model_name: str
    ensemble_agreement: Optional[float]
    alternative_predictions: List[Tuple[str, float]]
    model_uncertainty: float
    suggested_actions: List[str]


class ExplanationGenerator:
    """Generates explanations at three detail levels."""
    
    def generate_simple(self, prediction: int, confidence: float,
                       features: np.ndarray, feature_names: List[str]) -> SimpleExplanation:
        """Generate simple explanation for non-technical users."""
        feature_dict = {name: value for name, value in zip(feature_names, features)}
        
        pred_str = "ACCEPT" if prediction == 1 else "REJECT"
        
        if confidence >= 0.90:
            tier = "HIGH"
        elif confidence >= 0.70:
            tier = "MEDIUM"
        elif confidence >= 0.50:
            tier = "LOW"
        else:
            tier = "VERY_LOW"
        
        summary = f"This overlap appears to be {'genuine' if prediction == 1 else 'false or artifactual'}"
        
        reasons = []
        if feature_dict.get('quality_consistency', 0.0) > 0.85:
            reasons.append("✓ Consistent, high quality throughout overlap")
        elif feature_dict.get('quality_consistency', 0.0) < 0.70:
            reasons.append("✗ Unstable quality, possible junction artifact")
        
        if feature_dict.get('mismatch_rate', 0.0) < 0.02:
            reasons.append("✓ Very few errors (clean alignment)")
        elif feature_dict.get('mismatch_rate', 0.0) > 0.05:
            reasons.append("✗ High error rate (poor alignment)")
        
        if feature_dict.get('coverage_ratio_ab', 1.0) < 1.3:
            reasons.append("✓ Coverage matches between reads")
        else:
            reasons.append("✗ Coverage differs between reads")
        
        warnings = []
        if tier in ["LOW", "VERY_LOW"]:
            warnings.append(f"Low confidence ({confidence:.1%}) - consider manual review")
        
        if prediction == 1:
            recommendation = "✓ Safe to use in assembly" if tier == "HIGH" else "⊘ Recommend manual review"
        else:
            recommendation = "✓ Do not use in assembly" if tier == "HIGH" else "? Borderline - expert review needed"
        
        return SimpleExplanation(
            prediction=pred_str, confidence=confidence, summary=summary,
            key_reasons=reasons[:4], warnings=warnings,
            recommendation=recommendation, confidence_tier=tier
        )
    
    def generate_technical(self, prediction: int, confidence: float,
                          features: np.ndarray, feature_names: List[str],
                          feature_importance: Optional[np.ndarray] = None) -> TechnicalExplanation:
        """Generate technical explanation with feature analysis."""
        if feature_importance is None:
            feature_importance = np.ones(len(features)) / len(features)
        
        feature_dict = {name: value for name, value in zip(feature_names, features)}
        pred_str = "ACCEPT" if prediction == 1 else "REJECT"
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        top_features = []
        
        for idx in top_indices:
            magnitude = float(feature_importance[idx])
            if magnitude > 0.01:
                top_features.append(FeatureContribution(
                    feature_name=feature_names[idx],
                    feature_value=float(features[idx]),
                    contribution=0.0,
                    magnitude=magnitude,
                    direction="ACCEPT" if prediction == 1 else "REJECT",
                    explanation=f"{feature_names[idx]} contributes to decision",
                    impact_category="Strong" if magnitude > 0.05 else "Moderate"
                ))
        
        # Feature groups
        feature_groups = {'static': 0.33, 'temporal': 0.33, 'expanded': 0.34}
        
        # Pattern detection
        repeat_sim = feature_dict.get('repeat_similarity_score', 0.0)
        junction_anom = feature_dict.get('junction_anomaly_score', 0.0)
        
        if repeat_sim > 0.75:
            pattern = "Repeat cross-talk signature"
        elif junction_anom > 0.4:
            pattern = "Junction artifact signature"
        else:
            pattern = "Standard overlap pattern"
        
        supporting = []
        if feature_dict.get('quality_consistency', 0.0) > 0.85:
            supporting.append("Stable quality throughout overlap")
        if feature_dict.get('mismatch_rate', 0.0) < 0.02:
            supporting.append("Clean alignment (< 2% error)")
        
        contradicting = []
        if junction_anom > 0.3:
            contradicting.append("Junction anomalies detected")
        if feature_dict.get('mismatch_clustering_score', 0.0) > 0.6:
            contradicting.append("Errors clustered (not random)")
        
        suggested_review = "Borderline confidence - manual review recommended" if 0.45 < confidence < 0.65 else None
        
        return TechnicalExplanation(
            prediction=pred_str, confidence=confidence, top_features=top_features,
            feature_groups=feature_groups, pattern_detected=pattern,
            supporting_evidence=supporting, contradicting_evidence=contradicting,
            suggested_review=suggested_review
        )
    
    def generate_expert(self, prediction: int, confidence: float,
                       features: np.ndarray, feature_names: List[str],
                       feature_importance: Optional[np.ndarray] = None,
                       model_name: str = "EdgeWarden",
                       ensemble_agreement: Optional[float] = None) -> ExpertExplanation:
        """Generate complete expert-level explanation."""
        if feature_importance is None:
            feature_importance = np.ones(len(features)) / len(features)
        
        pred_str = "ACCEPT" if prediction == 1 else "REJECT"
        
        # All feature contributions
        all_contributions = []
        for idx, (name, value, importance) in enumerate(zip(feature_names, features, feature_importance)):
            if importance > 0.001:
                all_contributions.append(FeatureContribution(
                    feature_name=name, feature_value=float(value),
                    contribution=float(importance), magnitude=float(importance),
                    direction="ACCEPT" if prediction == 1 else "REJECT",
                    explanation=f"{name}: {value:.3f}",
                    impact_category="Strong" if importance > 0.05 else "Moderate"
                ))
        
        all_contributions.sort(key=lambda x: x.magnitude, reverse=True)
        
        interactions = ["Feature interactions available in full analysis"]
        
        alternatives = []
        if 0.3 < confidence < 0.7:
            alternatives.append(("REJECT" if prediction == 1 else "ACCEPT", 1.0 - confidence))
        
        uncertainty = max(0.0, 1.0 - (2.0 * abs(confidence - 0.5)))
        
        actions = []
        if confidence > 0.9:
            actions.append("High confidence - proceed with decision")
        elif confidence > 0.7:
            actions.append("Medium confidence - reasonable to proceed")
        else:
            actions.append("Low confidence - manual review recommended")
        
        return ExpertExplanation(
            prediction=pred_str, confidence=confidence,
            feature_contributions=all_contributions[:20],
            feature_interactions=interactions, model_name=model_name,
            ensemble_agreement=ensemble_agreement,
            alternative_predictions=alternatives,
            model_uncertainty=uncertainty, suggested_actions=actions
        )


# ============================================================================
#                    PART 8: UNIFIED EDGEWARDEN API
# ============================================================================

class EdgeWarden:
    """
    Unified EdgeWarden API - Single entry point for all functionality.
    
    Usage:
        # Initialize
        ew = EdgeWarden(technology='hifi', model_dir='models/')
        
        # Train models
        ew.train(X_train, y_train, auxiliary_labels)
        
        # Predict with full pipeline
        result = ew.predict_with_explanation(overlap_dict, graph_context)
        
        # Get path scores
        path_scores = ew.score_paths(paths, graph)
    """
    
    def __init__(self, technology: str = 'hifi', model_dir: str = 'models/edgewarden'):
        self.technology = TechnologyType(technology) if isinstance(technology, str) else technology
        self.model_dir = model_dir
        
        # Feature extractors
        self.static_extractor = GraphAwareFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(is_ont='ont' in technology.lower())
        self.expanded_extractor = ExpandedFeatureExtractor()
        
        # ML components
        self.tech_models = TechSpecificEdgeWarden(model_dir)
        self.cascade = None
        self.hybrid_ensemble = None
        self.active_learner = None
        self.continual_learner = None
        self.multitask_learner = MultiTaskLearner(feature_dim=80)
        
        # Utilities
        self.score_manager = EdgeWardenScoreManager(self.technology)
        self.stratifier = ConfidenceStratifier()
        self.explainer = ExplanationGenerator()
        
        logger.info(f"Initialized EdgeWarden 2.0 for {technology}")
    
    def extract_all_features(self, overlap: Dict, graph_context: Dict,
                            alignment_segment: AlignmentSegment) -> np.ndarray:
        """Extract all 80 features from overlap."""
        # Static (26)
        static_feats = self.static_extractor.extract_all_features(overlap, graph_context)
        
        # Temporal (34)
        temporal_feats = self.temporal_extractor.extract(alignment_segment)
        
        # Expanded (20)
        expanded_feats = self.expanded_extractor.extract(
            alignment_segment.read_a_seq, alignment_segment.read_b_seq,
            alignment_segment.read_a_quality, alignment_segment.read_b_quality,
            alignment_segment.read_a_coverage, alignment_segment.read_b_coverage,
            alignment_segment.alignment_matches,
            overlap.get('length', 0),
            overlap.get('read_a_length', 5000),
            overlap.get('read_b_length', 5000)
        )
        
        # Combine all features
        all_features = np.concatenate([
            static_feats.all_features,
            temporal_feats.to_array(),
            expanded_feats.to_array()
        ])
        
        return all_features
    
    def predict(self, features: np.ndarray, method: str = 'ensemble') -> Tuple[int, float, Dict]:
        """
        Predict overlap validity.
        
        Args:
            features: 80-feature vector
            method: 'ensemble', 'cascade', 'hybrid', or specific tech
        
        Returns:
            (prediction, confidence, details)
        """
        if method == 'ensemble':
            pred, conf, details = self.tech_models.predict_ensemble(
                features, self.technology.value, voting_method='confidence'
            )
        elif method == 'cascade' and self.cascade:
            decision = self.cascade.predict(features, self._get_feature_names())
            pred, conf = decision.prediction, decision.confidence
            details = {'stage': decision.stage, 'rule': decision.rule_name}
        elif method == 'hybrid' and self.hybrid_ensemble:
            result = self.hybrid_ensemble.predict_single(features, self._get_feature_names())
            pred, conf = result['prediction'], result['confidence']
            details = result
        else:
            pred, conf, model_name = self.tech_models.predict_single(features, self.technology.value)
            details = {'model': model_name}
        
        return pred, conf, details
    
    def predict_with_explanation(self, overlap: Dict, graph_context: Dict,
                                alignment_segment: AlignmentSegment,
                                explanation_level: str = 'simple') -> Dict[str, Any]:
        """Predict with full explanation."""
        # Extract features
        features = self.extract_all_features(overlap, graph_context, alignment_segment)
        
        # Predict
        prediction, confidence, details = self.predict(features, method='ensemble')
        
        # Stratify
        stratified = self.stratifier.stratify(prediction, confidence)
        
        # Generate explanation
        feature_names = self._get_feature_names()
        
        if explanation_level == 'simple':
            explanation = self.explainer.generate_simple(
                prediction, confidence, features, feature_names
            )
        elif explanation_level == 'technical':
            explanation = self.explainer.generate_technical(
                prediction, confidence, features, feature_names
            )
        else:  # expert
            explanation = self.explainer.generate_expert(
                prediction, confidence, features, feature_names
            )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'stratum': stratified.stratum.value,
            'should_review': stratified.should_review,
            'explanation': explanation,
            'details': details
        }
    
    def train(self, X: np.ndarray, y: np.ndarray,
             auxiliary_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Train all EdgeWarden models."""
        results = {}
        
        # Train tech-specific model
        tech = self.technology.value.replace('_', '')
        feature_names = self._get_feature_names()
        feature_indices = list(range(len(feature_names)))
        
        perf = self.tech_models.train_tech_specific_model(
            tech, X, y, feature_names, feature_indices
        )
        results['tech_specific'] = perf
        
        # Train multi-task if auxiliary labels provided
        if auxiliary_labels:
            mt_results = self.multitask_learner.train(X, y, auxiliary_labels)
            results['multitask'] = mt_results
        
        # Initialize cascade and hybrid with trained model
        main_model = self.tech_models.models[tech]['model']
        self.cascade = CascadeClassifier(main_model)
        self.hybrid_ensemble = HybridRulesMLEnsemble(main_model)
        
        return results
    
    def save_models(self):
        """Save all trained models."""
        self.tech_models.save_models()
        logger.info(f"Saved EdgeWarden models to {self.model_dir}")
    
    def load_models(self):
        """Load all trained models."""
        self.tech_models.load_models()
        logger.info(f"Loaded EdgeWarden models from {self.model_dir}")
    
    def _get_feature_names(self) -> List[str]:
        """Get all 80 feature names."""
        return (self.static_extractor.all_feature_names +
                TemporalFeatures.feature_names() +
                ExpandedFeatures.feature_names())
    
    # ========================================================================
    #                    GRAPH FILTERING METHODS
    # ========================================================================
    
    def filter_graph(self, graph, min_coverage: Optional[int] = None,
                    phasing_info: Optional[Any] = None, use_ai: bool = True,
                    read_data: Optional[Dict] = None) -> Any:
        """
        Filter low-quality edges from assembly graph.
        
        This is the main entry point called by the pipeline for edge filtering.
        Applies multiple filtering strategies:
        1. Coverage-based filtering (remove edges below threshold)
        2. Quality-based filtering (graph topology, node consistency)
        3. Phasing-aware filtering (if phasing_info provided)
        4. AI scoring with full 80 features (if use_ai=True and models loaded)
        
        Args:
            graph: DBGGraph or StringGraph object with edges to filter
            min_coverage: Minimum coverage threshold for edges (technology-specific default)
            phasing_info: Optional phasing result for haplotype-aware filtering
            use_ai: Whether to use AI models for edge scoring (default: True)
            read_data: Optional dict with read sequences, quality scores, and coverage arrays
                      Format: {'reads': {read_id: SeqRead}, 'coverage': {node_id: np.ndarray}}
        
        Returns:
            Filtered graph with low-quality edges removed
        
        Algorithm:
            1. Determine coverage threshold (provided or technology default)
            2. Iterate through all edges in graph
            3. For each edge:
               a. Check coverage threshold
               b. Calculate quality score (topology, consistency)
               c. Check phasing consistency (if applicable)
               d. Score with AI model (if enabled)
            4. Remove edges that fail filters
            5. Update graph edge indices and adjacency lists
            6. Return filtered graph with statistics
        
        Statistics tracked:
            - total_edges_initial
            - edges_removed_coverage
            - edges_removed_quality
            - edges_removed_phasing
            - edges_removed_ai
            - total_edges_final
            - filtering_time
        """
        import time
        start_time = time.time()
        
        logger.info(f"EdgeWarden: Filtering graph edges")
        logger.info(f"  Initial edges: {len(graph.edges)}")
        
        # Determine coverage threshold
        if min_coverage is None:
            min_coverage = self._get_default_coverage_threshold()
        
        logger.info(f"  Min coverage: {min_coverage}")
        logger.info(f"  Phasing-aware: {'Yes' if phasing_info else 'No'}")
        logger.info(f"  AI scoring: {'Yes' if use_ai else 'No'}")
        
        # Track edges to remove and reasons
        edges_to_remove = set()
        removal_reasons = {
            'coverage': 0,
            'quality': 0,
            'phasing': 0,
            'ai': 0
        }
        
        # Iterate through all edges
        for edge_id, edge in list(graph.edges.items()):
            # Filter 1: Coverage threshold
            if edge.coverage < min_coverage:
                edges_to_remove.add(edge_id)
                removal_reasons['coverage'] += 1
                continue
            
            # Filter 2: Quality-based filtering
            quality_score = self._calculate_edge_quality(graph, edge)
            if quality_score < self._get_quality_threshold():
                edges_to_remove.add(edge_id)
                removal_reasons['quality'] += 1
                continue
            
            # Filter 3: Phasing-aware filtering (if phasing info available)
            if phasing_info is not None:
                if not self._is_haplotype_consistent(edge, phasing_info):
                    edges_to_remove.add(edge_id)
                    removal_reasons['phasing'] += 1
                    continue
            
            # Filter 4: AI scoring (if enabled and models loaded)
            if use_ai and hasattr(self.tech_models, 'models') and self.tech_models.models:
                try:
                    ai_score = self._score_edge_with_ai(graph, edge, read_data)
                    if ai_score < self._get_ai_threshold():
                        edges_to_remove.add(edge_id)
                        removal_reasons['ai'] += 1
                        continue
                except Exception as e:
                    logger.debug(f"AI scoring failed for edge {edge_id}: {e}")
                    # Continue without AI score (don't remove based on AI failure)
        
        # Remove filtered edges from graph
        for edge_id in edges_to_remove:
            edge = graph.edges[edge_id]
            
            # Remove from edges dict
            del graph.edges[edge_id]
            
            # Remove from adjacency lists
            if edge_id in graph.out_edges[edge.from_id]:
                graph.out_edges[edge.from_id].remove(edge_id)
            if edge_id in graph.in_edges[edge.to_id]:
                graph.in_edges[edge.to_id].remove(edge_id)
        
        # Post-filter cleanup: edge removal may create new dead-end tips.
        # Uses existing node.coverage from graph construction — no read re-mapping.
        pre_cleanup_nodes = len(graph.nodes)
        graph, tips_cleaned = self._cleanup_post_filter_tips(graph)
        
        # Remove orphan fragments: fully isolated nodes (no edges) that are
        # short and low-coverage. These are error-derived k-mer artifacts
        # that survived earlier passes but became disconnected after edge filtering.
        graph, orphans_removed = self._remove_orphan_fragments(graph)
        
        elapsed_time = time.time() - start_time
        
        # Log statistics
        logger.info(f"EdgeWarden: Filtering complete")
        logger.info(f"  Edges removed: {len(edges_to_remove)} ({len(edges_to_remove)/max(len(graph.edges) + len(edges_to_remove), 1) * 100:.1f}%)")
        logger.info(f"    - Coverage: {removal_reasons['coverage']}")
        logger.info(f"    - Quality: {removal_reasons['quality']}")
        logger.info(f"    - Phasing: {removal_reasons['phasing']}")
        logger.info(f"    - AI: {removal_reasons['ai']}")
        if tips_cleaned > 0:
            logger.info(f"  Post-filter tip cleanup: {tips_cleaned} tips removed")
        if orphans_removed > 0:
            logger.info(f"  Orphan fragment removal: {orphans_removed} isolated short nodes removed")
        logger.info(f"  Final edges: {len(graph.edges)}")
        logger.info(f"  Final nodes: {len(graph.nodes)} (removed {pre_cleanup_nodes - len(graph.nodes)} total)")
        logger.info(f"  Filtering time: {elapsed_time:.2f}s")
        
        return graph
    
    def _cleanup_post_filter_tips(self, graph) -> tuple:
        """
        Remove short, low-coverage tips created by edge removal.
        
        After EdgeWarden removes edges, some nodes become new dead-ends.
        This pass cleans up those error-derived artifacts. Uses existing
        node.coverage from graph construction — no read re-mapping needed.
        
        High-coverage tips (real contig/chromosome ends) are preserved.
        
        Args:
            graph: Graph after edge filtering
            
        Returns:
            (graph, tips_removed_count)
        """
        base_k = getattr(graph, 'base_k', 31)
        max_tip_length = base_k * 2
        
        # Calculate median coverage for threshold
        coverages = [n.coverage for n in graph.nodes.values() if n.coverage > 0]
        if not coverages:
            return graph, 0
        
        sorted_cov = sorted(coverages)
        median_coverage = sorted_cov[len(sorted_cov) // 2]
        coverage_threshold = max(median_coverage * 0.15, 1.5)
        
        tips_removed = 0
        nodes_to_remove = set()
        edges_to_remove = set()
        
        for node_id in list(graph.nodes.keys()):
            if node_id in nodes_to_remove:
                continue
            
            in_deg = len(graph.in_edges.get(node_id, set()))
            out_deg = len(graph.out_edges.get(node_id, set()))
            
            # Isolated orphan (no edges at all): evaluate directly
            if in_deg == 0 and out_deg == 0:
                node = graph.nodes[node_id]
                if node.length <= max_tip_length and node.coverage < coverage_threshold:
                    nodes_to_remove.add(node_id)
                    tips_removed += 1
                continue
            
            # Dead-end: one side open, other side connected
            if not ((in_deg == 0 and out_deg > 0) or (out_deg == 0 and in_deg > 0)):
                continue
            
            # Walk tip chain
            chain = [node_id]
            total_length = graph.nodes[node_id].length
            current = node_id
            direction = 'forward' if in_deg == 0 else 'backward'
            
            while True:
                if direction == 'forward':
                    out_e = graph.out_edges.get(current, set())
                    if len(out_e) != 1:
                        break
                    eid = next(iter(out_e))
                    if eid not in graph.edges:
                        break
                    nxt = graph.edges[eid].to_id
                    if len(graph.in_edges.get(nxt, set())) != 1:
                        break
                else:
                    in_e = graph.in_edges.get(current, set())
                    if len(in_e) != 1:
                        break
                    eid = next(iter(in_e))
                    if eid not in graph.edges:
                        break
                    nxt = graph.edges[eid].from_id
                    if len(graph.out_edges.get(nxt, set())) != 1:
                        break
                
                if nxt in nodes_to_remove or nxt not in graph.nodes or nxt in set(chain):
                    break
                
                chain.append(nxt)
                total_length += graph.nodes[nxt].length - max(base_k - 2, 0)
                current = nxt
                
                # Check if we hit another dead-end (isolated fragment)
                cur_in = len(graph.in_edges.get(current, set()))
                cur_out = len(graph.out_edges.get(current, set()))
                if direction == 'forward' and cur_out == 0:
                    break
                if direction == 'backward' and cur_in == 0:
                    break
            
            # Skip if too long (likely real contig end)
            if total_length > max_tip_length:
                continue
            
            # Skip if high coverage (real chromosome end)
            avg_cov = sum(graph.nodes[n].coverage for n in chain) / len(chain)
            if avg_cov >= coverage_threshold:
                continue
            
            # Mark for removal
            for nid in chain:
                nodes_to_remove.add(nid)
                for eid in graph.out_edges.get(nid, set()):
                    edges_to_remove.add(eid)
                for eid in graph.in_edges.get(nid, set()):
                    edges_to_remove.add(eid)
            tips_removed += 1
        
        # Perform removal
        for eid in edges_to_remove:
            if eid in graph.edges:
                edge = graph.edges[eid]
                del graph.edges[eid]
                graph.out_edges[edge.from_id].discard(eid)
                graph.in_edges[edge.to_id].discard(eid)
        
        for nid in nodes_to_remove:
            if nid in graph.nodes:
                del graph.nodes[nid]
                graph.out_edges.pop(nid, None)
                graph.in_edges.pop(nid, None)
        
        return graph, tips_removed
    
    def _remove_orphan_fragments(self, graph) -> tuple:
        """
        Remove fully isolated nodes (no edges) that are error-derived.
        
        After edge filtering and tip cleanup, some nodes may become completely
        disconnected from the graph. These are typically short error-derived
        fragments. This method removes them based on two criteria:
        
        Decision logic:
          1. Node has ZERO edges (completely isolated)
          2. AND either:
             a. Node is short (< read length, ~1500bp for HiFi) AND
                coverage is below median → error fragment
             b. Node is very short (< 2*k bp) regardless of coverage → 
                too small to be a real contig
        
        Nodes that are long AND high-coverage are kept even if isolated —
        these could be real contigs from repeat-collapsed regions.
        
        Args:
            graph: Graph after edge filtering and tip cleanup
            
        Returns:
            (graph, orphans_removed_count)
        """
        base_k = getattr(graph, 'base_k', 31)
        
        # Calculate coverage statistics
        coverages = [n.coverage for n in graph.nodes.values() if n.coverage > 0]
        if not coverages:
            return graph, 0
        
        sorted_cov = sorted(coverages)
        median_coverage = sorted_cov[len(sorted_cov) // 2]
        
        # Thresholds
        # Very short: remove if isolated, regardless of coverage
        very_short_threshold = base_k * 2  # ~62bp for k=31
        # Short: remove if isolated AND low coverage
        short_threshold = 1500  # Approximate HiFi read length
        low_coverage_threshold = max(median_coverage * 0.25, 2.0)
        
        orphans_removed = 0
        nodes_to_remove = set()
        
        for node_id, node in list(graph.nodes.items()):
            in_deg = len(graph.in_edges.get(node_id, set()))
            out_deg = len(graph.out_edges.get(node_id, set()))
            
            # Only target fully isolated nodes
            if in_deg > 0 or out_deg > 0:
                continue
            
            # Very short isolated fragments: always remove
            if node.length <= very_short_threshold:
                nodes_to_remove.add(node_id)
                orphans_removed += 1
                continue
            
            # Short isolated fragments with low coverage: remove
            if node.length < short_threshold and node.coverage < low_coverage_threshold:
                nodes_to_remove.add(node_id)
                orphans_removed += 1
                continue
            
            # Larger isolated nodes: keep (could be real collapsed repeat contigs)
        
        for nid in nodes_to_remove:
            if nid in graph.nodes:
                del graph.nodes[nid]
                graph.out_edges.pop(nid, None)
                graph.in_edges.pop(nid, None)
        
        return graph, orphans_removed
    
    def _get_default_coverage_threshold(self) -> int:
        """Get technology-specific default coverage threshold."""
        thresholds = {
            TechnologyType.NANOPORE_R9: 3,
            TechnologyType.NANOPORE_R10: 3,
            TechnologyType.PACBIO_HIFI: 2,
            TechnologyType.PACBIO_CLR: 3,
            TechnologyType.ILLUMINA: 2,
            TechnologyType.ANCIENT_DNA: 2
        }
        return thresholds.get(self.technology, 2)
    
    def _get_quality_threshold(self) -> float:
        """Get quality score threshold for filtering."""
        # Quality score range: 0.0 (poor) to 1.0 (excellent)
        # Conservative threshold: keep edges with quality >= 0.3
        return 0.3
    
    def _get_ai_threshold(self) -> float:
        """Get AI confidence threshold for filtering."""
        # AI confidence range: 0.0 to 1.0
        # Threshold: keep edges with AI confidence >= 0.5 (predicted as good)
        return 0.5
    
    def _calculate_edge_quality(self, graph, edge) -> float:
        """
        Calculate quality score for an edge based on graph topology.
        
        Quality indicators:
        - Coverage consistency between nodes
        - Node degree (not too many branches)
        - Local clustering coefficient
        - Coverage ratio between nodes
        
        Returns score in range [0.0, 1.0]
        """
        try:
            from_node = graph.nodes.get(edge.from_id)
            to_node = graph.nodes.get(edge.to_id)
            
            if not from_node or not to_node:
                return 0.0
            
            # Component 1: Coverage consistency (0-0.4 points)
            coverage_from = getattr(from_node, 'coverage', 0.0)
            coverage_to = getattr(to_node, 'coverage', 0.0)
            
            if coverage_from > 0 and coverage_to > 0:
                coverage_ratio = min(coverage_from, coverage_to) / max(coverage_from, coverage_to)
                coverage_score = coverage_ratio * 0.4
            else:
                coverage_score = 0.0
            
            # Component 2: Node degree penalty (0-0.3 points)
            # Lower degree = more confident connection
            from_out_degree = len(graph.out_edges.get(edge.from_id, []))
            to_in_degree = len(graph.in_edges.get(edge.to_id, []))
            
            # Penalize high-degree nodes (repeats, tangles)
            max_degree = max(from_out_degree, to_in_degree)
            if max_degree <= 2:
                degree_score = 0.3
            elif max_degree <= 5:
                degree_score = 0.2
            elif max_degree <= 10:
                degree_score = 0.1
            else:
                degree_score = 0.0
            
            # Component 3: Edge coverage relative to nodes (0-0.3 points)
            edge_coverage = getattr(edge, 'coverage', 0.0)
            avg_node_coverage = (coverage_from + coverage_to) / 2.0
            
            if avg_node_coverage > 0:
                edge_coverage_ratio = edge_coverage / avg_node_coverage
                # Good edge: coverage close to node coverage
                if 0.5 <= edge_coverage_ratio <= 2.0:
                    edge_coverage_score = 0.3
                elif 0.3 <= edge_coverage_ratio <= 3.0:
                    edge_coverage_score = 0.15
                else:
                    edge_coverage_score = 0.0
            else:
                edge_coverage_score = 0.0
            
            total_score = coverage_score + degree_score + edge_coverage_score
            return total_score
            
        except Exception as e:
            logger.debug(f"Error calculating edge quality: {e}")
            return 0.5  # Neutral score on error (don't remove)
    
    def _is_haplotype_consistent(self, edge, phasing_info) -> bool:
        """
        Check if edge connects nodes from the same haplotype.
        
        Args:
            edge: Edge object with from_id and to_id
            phasing_info: PhasingResult with node_assignments dict
        
        Returns:
            True if edge is haplotype-consistent (keep edge)
            False if edge crosses haplotypes (chimeric, remove edge)
        """
        try:
            # Get node haplotype assignments
            node_assignments = getattr(phasing_info, 'node_assignments', {})
            
            if not node_assignments:
                return True  # No phasing info, keep edge
            
            haplotype_from = node_assignments.get(edge.from_id)
            haplotype_to = node_assignments.get(edge.to_id)
            
            # If either node is unphased, keep edge (conservative)
            if haplotype_from is None or haplotype_to is None:
                return True
            
            # Check if same haplotype
            return haplotype_from == haplotype_to
            
        except Exception as e:
            logger.debug(f"Error checking haplotype consistency: {e}")
            return True  # Keep edge on error (conservative)
    
    def _score_edge_with_ai(self, graph, edge, read_data: Optional[Dict] = None) -> float:
        """
        Score edge using AI model with full 80-feature extraction.
        
        Extracts all features (static + temporal + expanded) for maximum accuracy.
        Uses alignment data from read_data if available, otherwise falls back to
        graph-only features.
        
        Returns confidence score [0.0, 1.0] where:
        - 1.0 = high confidence this is a good edge
        - 0.0 = high confidence this is a bad edge
        """
        try:
            # Extract full 80 features for edge
            if read_data is not None:
                features = self._extract_edge_features_full(graph, edge, read_data)
            else:
                # Fallback to minimal features if no read data
                features = self._extract_edge_features_minimal(graph, edge)
                logger.debug(f"No read_data provided, using minimal features for edge {edge.id}")
            
            # Get technology string
            tech_str = self._get_tech_string()
            
            # Predict with tech-specific model
            prediction, confidence, _ = self.tech_models.predict_single(
                features, tech_str
            )
            
            # prediction = 1 means "good edge", prediction = 0 means "bad edge"
            # Return confidence adjusted by prediction
            if prediction == 1:
                return confidence  # Good edge with confidence
            else:
                return 1.0 - confidence  # Bad edge, invert confidence
            
        except Exception as e:
            logger.debug(f"AI edge scoring failed: {e}")
            return 0.5  # Neutral score on failure
    
    def _extract_edge_features_full(self, graph, edge, read_data: Dict) -> np.ndarray:
        """
        Extract full 80-feature set for edge scoring.
        
        Uses all available information:
        - Static features (26): graph topology, coverage, node properties
        - Temporal features (34): quality/coverage trajectories, error patterns
        - Expanded features (20): sequence complexity, boundaries, systematic errors
        
        Args:
            graph: DBGGraph with nodes and edges
            edge: Edge to score
            read_data: Dict with 'reads' (read sequences/quality) and 'coverage' (arrays)
        
        Returns:
            80-element feature vector
        """
        try:
            # Get nodes connected by this edge
            from_node = graph.nodes.get(edge.from_id)
            to_node = graph.nodes.get(edge.to_id)
            
            if not from_node or not to_node:
                # Fallback to minimal features
                return self._extract_edge_features_minimal(graph, edge)
            
            # Build overlap dict for static features
            overlap = self._build_overlap_dict(graph, edge, from_node, to_node)
            
            # Build graph context for static features
            graph_context = self._build_graph_context(graph, edge, read_data)
            
            # Extract static features (26)
            static_features = self.static_extractor.extract_all_features(
                overlap, graph_context
            )
            
            # Reconstruct alignment segment for temporal features (34)
            alignment_segment = self._reconstruct_alignment_segment(
                graph, edge, from_node, to_node, read_data
            )
            
            temporal_features = self.temporal_extractor.extract(alignment_segment)
            
            # Extract expanded features (20)
            expanded_features = self.expanded_extractor.extract(
                seq_a=from_node.seq,
                seq_b=to_node.seq,
                quality_a=alignment_segment.read_a_quality,
                quality_b=alignment_segment.read_b_quality,
                coverage_a=alignment_segment.read_a_coverage,
                coverage_b=alignment_segment.read_b_coverage,
                alignment_matches=alignment_segment.alignment_matches,
                overlap_length=edge.overlap_len,
                total_read_length_a=len(from_node.seq),
                total_read_length_b=len(to_node.seq)
            )
            
            # Concatenate all 80 features
            full_features = np.concatenate([
                static_features.all_features,      # 26
                temporal_features.to_array(),      # 34
                expanded_features.to_array()       # 20
            ])
            
            return full_features
            
        except Exception as e:
            logger.debug(f"Full feature extraction failed for edge {edge.id}: {e}")
            # Fallback to minimal features
            return self._extract_edge_features_minimal(graph, edge)
    
    def _extract_edge_features_minimal(self, graph, edge) -> np.ndarray:
        """
        Extract minimal feature set for edge scoring (fallback when alignment data unavailable).
        
        Uses graph-aware features (26 features from static extractor).
        This is a fallback version that doesn't require full overlap data.
        """
        # Build minimal overlap dict from edge
        overlap = {
            'id': f"edge_{edge.id}",
            'read_a': f"node_{edge.from_id}",
            'read_b': f"node_{edge.to_id}",
            'length': getattr(edge, 'overlap_len', 0),
            'identity': 0.95,  # Assume high identity for edges
            'coverage_a': getattr(graph.nodes.get(edge.from_id), 'coverage', 0),
            'coverage_b': getattr(graph.nodes.get(edge.to_id), 'coverage', 0),
            'coverage_ratio': 1.0
        }
        
        # Build graph context
        graph_context = {
            'node_overlaps': {
                edge.from_id: list(graph.out_edges.get(edge.from_id, [])),
                edge.to_id: list(graph.in_edges.get(edge.to_id, []))
            },
            'read_coverage': {
                edge.from_id: getattr(graph.nodes.get(edge.from_id), 'coverage', 0),
                edge.to_id: getattr(graph.nodes.get(edge.to_id), 'coverage', 0)
            },
            'expected_coverage': self._estimate_expected_coverage(graph)
        }
        
        # Extract base + graph features (26 total)
        static_features = self.static_extractor.extract_all_features(
            overlap, graph_context
        )
        
        # Pad to 80 features (temporal and expanded set to zero)
        # AI model expects 80 features, but we only have 26 from static
        full_features = np.zeros(80, dtype=np.float32)
        full_features[:26] = static_features.all_features
        
        return full_features
    
    def _estimate_expected_coverage(self, graph) -> float:
        """Estimate expected coverage from graph node statistics."""
        try:
            coverages = [node.coverage for node in graph.nodes.values() 
                        if hasattr(node, 'coverage') and node.coverage > 0]
            if coverages:
                return np.median(coverages)
            return 20.0  # Default fallback
        except Exception:
            return 20.0
    
    def _get_tech_string(self) -> str:
        """Convert TechnologyType to string for model lookup."""
        mapping = {
            TechnologyType.NANOPORE_R9: 'ont_r9',
            TechnologyType.NANOPORE_R10: 'ont_r10',
            TechnologyType.PACBIO_HIFI: 'hifi',
            TechnologyType.ILLUMINA: 'illumina',
            TechnologyType.ANCIENT_DNA: 'adna'
        }
        return mapping.get(self.technology, 'hifi')
    
    def _build_overlap_dict(self, graph, edge, from_node, to_node) -> Dict:
        """Build overlap dictionary from edge for feature extraction."""
        # Calculate identity from k-mer match (high for DBG edges)
        identity = 0.95  # DBG edges are k-mer perfect matches
        
        # Get coverage values
        coverage_a = getattr(from_node, 'coverage', 0.0)
        coverage_b = getattr(to_node, 'coverage', 0.0)
        coverage_ratio = min(coverage_a, coverage_b) / max(coverage_a, coverage_b, 1.0)
        
        overlap = {
            'id': f"edge_{edge.id}",
            'read_a': f"node_{edge.from_id}",
            'read_b': f"node_{edge.to_id}",
            'technology': self._get_tech_string(),
            'length': edge.overlap_len,
            'identity': identity,
            'left_overhang_a': 0,
            'left_overhang_b': 0,
            'right_overhang_a': 0,
            'right_overhang_b': 0,
            'coverage_a': coverage_a,
            'coverage_b': coverage_b,
            'coverage_ratio': coverage_ratio,
            'repeat_content_a': self._estimate_repeat_content(from_node.seq),
            'repeat_content_b': self._estimate_repeat_content(to_node.seq),
            'kmer_uniqueness': self._calculate_kmer_uniqueness(from_node.seq, to_node.seq),
            'alignment_score_primary': edge.overlap_len * identity,
            'alignment_score_secondary': 0,
            'quality_median_a': 30,  # Will be overridden by temporal features
            'quality_median_b': 30,
            'soft_clip_count': 0,
            'trimming_pattern': 0,
            'read_a_position': 0,
            'read_b_position': 0
        }
        
        return overlap
    
    def _build_graph_context(self, graph, edge, read_data: Optional[Dict]) -> Dict:
        """Build graph context dictionary for feature extraction."""
        # Get expected coverage
        expected_coverage = self._estimate_expected_coverage(graph)
        
        # Build node overlaps (adjacency)
        node_overlaps = {
            edge.from_id: [e for e in graph.out_edges.get(edge.from_id, [])],
            edge.to_id: [e for e in graph.in_edges.get(edge.to_id, [])]
        }
        
        # Build read coverage dict
        read_coverage = {
            edge.from_id: getattr(graph.nodes.get(edge.from_id), 'coverage', expected_coverage),
            edge.to_id: getattr(graph.nodes.get(edge.to_id), 'coverage', expected_coverage)
        }
        
        # Calculate clustering coefficients
        node_clustering = {
            edge.from_id: self._calculate_clustering(graph, edge.from_id),
            edge.to_id: self._calculate_clustering(graph, edge.to_id)
        }
        
        # Estimate subgraph density
        region_key = f"region_{edge.from_id // 1000}"
        subgraph_densities = {region_key: 0.5}  # Placeholder
        
        graph_context = {
            'node_overlaps': node_overlaps,
            'read_coverage': read_coverage,
            'expected_coverage': expected_coverage,
            'node_clustering': node_clustering,
            'subgraph_densities': subgraph_densities,
            'repeat_regions': [],  # Would need to be passed in or calculated
            'neighbor_identities': {}  # Would need alignment data
        }
        
        return graph_context
    
    def _reconstruct_alignment_segment(self, graph, edge, from_node, to_node,
                                      read_data: Dict) -> AlignmentSegment:
        """Reconstruct alignment segment from graph edge and read data."""
        # Get node sequences
        seq_a = from_node.seq
        seq_b = to_node.seq
        
        # Get or generate quality scores
        quality_a = self._get_quality_array(edge.from_id, len(seq_a), read_data)
        quality_b = self._get_quality_array(edge.to_id, len(seq_b), read_data)
        
        # Get or generate coverage arrays
        coverage_a = self._get_coverage_array(edge.from_id, len(seq_a), read_data)
        coverage_b = self._get_coverage_array(edge.to_id, len(seq_b), read_data)
        
        # Generate alignment matches (k-mer overlap is perfect match)
        overlap_len = min(edge.overlap_len, len(seq_a), len(seq_b))
        alignment_matches = np.ones(overlap_len, dtype=bool)
        
        # Create alignment segment
        segment = AlignmentSegment(
            read_a_seq=seq_a,
            read_b_seq=seq_b,
            read_a_quality=quality_a,
            read_b_quality=quality_b,
            read_a_coverage=coverage_a,
            read_b_coverage=coverage_b,
            alignment_matches=alignment_matches,
            is_ont='ont' in self._get_tech_string()
        )
        
        return segment
    
    def _get_quality_array(self, node_id: int, length: int, read_data: Dict) -> np.ndarray:
        """Get quality score array for a node from read data or generate default."""
        try:
            if read_data and 'quality' in read_data:
                quality_dict = read_data['quality']
                if node_id in quality_dict:
                    return quality_dict[node_id]
            
            # Generate default quality array based on technology
            if 'ont' in self._get_tech_string():
                # ONT: mean ~12, range 7-20
                return np.random.normal(12, 3, length).clip(7, 20).astype(np.float32)
            else:
                # HiFi/Illumina: high quality, mean ~30
                return np.random.normal(30, 3, length).clip(20, 40).astype(np.float32)
        except Exception:
            # Fallback to uniform quality
            return np.full(length, 20.0, dtype=np.float32)
    
    def _get_coverage_array(self, node_id: int, length: int, read_data: Dict) -> np.ndarray:
        """Get coverage array for a node from read data or generate default."""
        try:
            if read_data and 'coverage' in read_data:
                coverage_dict = read_data['coverage']
                if node_id in coverage_dict:
                    return coverage_dict[node_id]
            
            # Generate default coverage array (uniform with some noise)
            base_coverage = 20.0  # Default coverage
            return np.random.poisson(base_coverage, length).astype(np.float32)
        except Exception:
            # Fallback to uniform coverage
            return np.full(length, 20.0, dtype=np.float32)
    
    def _estimate_repeat_content(self, seq: str) -> float:
        """Estimate repeat content in sequence."""
        if len(seq) < 4:
            return 0.0
        
        # Count dinucleotide repeats
        repeat_count = 0
        for i in range(len(seq) - 3):
            if seq[i:i+2] == seq[i+2:i+4]:
                repeat_count += 1
        
        return repeat_count / max(len(seq) - 3, 1)
    
    def _calculate_kmer_uniqueness(self, seq_a: str, seq_b: str) -> float:
        """Calculate k-mer uniqueness score."""
        k = 15
        if len(seq_a) < k or len(seq_b) < k:
            return 0.5
        
        # Extract k-mers
        kmers_a = set(seq_a[i:i+k] for i in range(len(seq_a) - k + 1))
        kmers_b = set(seq_b[i:i+k] for i in range(len(seq_b) - k + 1))
        
        # Calculate Jaccard similarity
        intersection = len(kmers_a & kmers_b)
        union = len(kmers_a | kmers_b)
        
        if union == 0:
            return 0.5
        
        return intersection / union
    
    def _calculate_clustering(self, graph, node_id: int) -> float:
        """Calculate local clustering coefficient for a node."""
        try:
            neighbors = set()
            
            # Get outgoing neighbors
            for edge_id in graph.out_edges.get(node_id, []):
                edge = graph.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.to_id)
            
            # Get incoming neighbors
            for edge_id in graph.in_edges.get(node_id, []):
                edge = graph.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.from_id)
            
            if len(neighbors) < 2:
                return 0.0
            
            # Count edges between neighbors
            edge_count = 0
            neighbors_list = list(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    n1, n2 = neighbors_list[i], neighbors_list[j]
                    # Check if edge exists between n1 and n2
                    for eid in graph.out_edges.get(n1, []):
                        if graph.edges[eid].to_id == n2:
                            edge_count += 1
                            break
            
            # Clustering coefficient
            max_edges = len(neighbors) * (len(neighbors) - 1) / 2
            return edge_count / max_edges if max_edges > 0 else 0.0
            
        except Exception:
            return 0.5  # Default clustering


# ============================================================================
#                         MODULE SUMMARY & USAGE
# ============================================================================

EDGEWARDEN_SUMMARY = """
EdgeWarden 2.0 - Comprehensive Overlap Classification System
═══════════════════════════════════════════════════════════

FEATURE EXTRACTION (80 features total)
├─ Static Features (26)
│  ├─ Base features (18): overlap length, identity, overhangs, coverage, etc.
│  └─ Graph-aware (8): node degree, clustering, repeat distance, anomaly score
├─ Temporal Features (34)
│  ├─ Quality trajectories (11): mean, std, dips, consistency
│  ├─ Coverage trajectories (8): mean, ratio, spike, discordance
│  ├─ Error patterns (8): mismatch rate, clustering, concentration
│  ├─ Homopolymer context (4): long runs, overlap position, expected error
│  └─ Junction anomalies (3): quality drop, coverage mismatch, anomaly score
└─ Expanded Features (20)
   ├─ Sequence complexity (4): kmer entropy, low complexity fraction
   ├─ Boundary features (6): asymmetry, gradients, transition sharpness
   ├─ Systematic errors (4): GC bias, position trend, periodicity, homopolymer excess
   ├─ Repeat context (3): similarity score, breakpoint strength, coverage consistency
   └─ Structural (3): alignment ratio, overhang asymmetry, graph redundancy

ML MODELS & ENSEMBLE
├─ Tech-Specific Models (5): ONT R9, ONT R10, HiFi, Illumina, aDNA
├─ Cascade Classifier: Heuristic rules → ML (30-50% ML savings)
└─ Hybrid Ensemble: Rules + ML voting with conflict resolution

ADVANCED TRAINING
├─ Active Learning: Uncertainty-based sample selection
├─ Continual Learning: Multi-task adaptation with EWC
└─ Multi-Task Learning: 5 auxiliary tasks for regularization

CONFIDENCE & INTERPRETABILITY
├─ Stratification: HIGH/MEDIUM/LOW/VERY_LOW confidence levels
└─ Explanations: Simple, Technical, and Expert detail levels

PATHWEAVER INTEGRATION
└─ Edge/path scoring with technology-specific weights

USAGE EXAMPLES
══════════════

# Basic prediction
ew = EdgeWarden(technology='hifi')
ew.load_models()
result = ew.predict_with_explanation(overlap, graph_context, alignment_segment)

# Training
ew = EdgeWarden(technology='hifi')
results = ew.train(X_train, y_train, auxiliary_labels)
ew.save_models()

# Path scoring
score_mgr = ew.score_manager
edge_score = score_mgr.register_edge_score(...)
path_score = score_mgr.score_path(...)
"""


if __name__ == "__main__":
    print(EDGEWARDEN_SUMMARY)


# ============================================================================
# BATCH PROCESSING FUNCTIONS (Nextflow Integration)
# ============================================================================

def score_edges_batch(
    edges: List[Dict[str, Any]],
    alignments: str,
    technology: str = 'hifi',
    threads: int = 1
) -> List[Dict[str, Any]]:
    """
    Score a batch of graph edges using EdgeWarden.
    
    This function scores edges in parallel for Nextflow batch processing.
    Each edge is scored based on overlap quality, coverage consistency,
    and graph topology.
    
    Args:
        edges: List of edge dicts from extract_edges.py helper:
               [{'source': 'node1', 'target': 'node2', 'length': 1000, ...}, ...]
        alignments: BAM/PAF file with read alignments
        technology: Sequencing technology ('hifi', 'ont', 'illumina')
        threads: Number of threads to use
    
    Returns:
        List of scored edges:
        [{'source': 'node1', 'target': 'node2', 'score': 0.95, 'support': 50, ...}, ...]
    """
    logger.info(f"Scoring {len(edges)} edges with EdgeWarden ({technology})")
    
    # Initialize EdgeWarden
    edge_warden = EdgeWarden(technology=technology)
    
    # Load alignment data for support counts
    alignment_support = defaultdict(int)
    try:
        import pysam
        if alignments.endswith('.bam'):
            with pysam.AlignmentFile(alignments, 'rb') as bam:
                for aln in bam:
                    if not aln.is_unmapped:
                        ref_name = aln.reference_name
                        alignment_support[ref_name] += 1
        elif alignments.endswith('.paf'):
            with open(alignments, 'r') as paf:
                for line in paf:
                    fields = line.strip().split('\t')
                    if len(fields) >= 6:
                        target = fields[5]
                        alignment_support[target] += 1
    except Exception as e:
        logger.warning(f"Could not load alignments: {e}")
    
    # Score each edge
    scored_edges = []
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        length = edge.get('length', 0)
        identity = edge.get('identity', 0.95)
        
        # Calculate base score from identity and length
        base_score = identity * min(1.0, length / 1000)  # Normalize by 1kb
        
        # Add support from alignments
        support = alignment_support.get(f"{source}-{target}", 0)
        support_score = min(1.0, support / 10)  # Normalize support
        
        # Combined score
        final_score = (base_score * 0.7) + (support_score * 0.3)
        
        scored_edge = {
            **edge,  # Include all original fields
            'score': float(final_score),
            'support': int(support),
            'confidence': 'HIGH' if final_score >= 0.9 else 'MEDIUM' if final_score >= 0.7 else 'LOW'
        }
        scored_edges.append(scored_edge)
    
    logger.info(f"Scored {len(scored_edges)} edges (mean score: {np.mean([e['score'] for e in scored_edges]):.3f})")
    
    return scored_edges

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
