#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-Based Overlap Filtering Module for StrandWeaver.

This module uses machine learning to classify edges in assembly graphs as:
- True adjacencies (correct assembly connections)
- Repeat-induced false connections
- Allelic alternatives (haplotype variants)
- Chimeric or error-induced artifacts

The module extracts comprehensive features from edges and applies
a pluggable ML model to score edge quality.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class EdgeAIAnnotation:
    """
    AI/ML-based annotation for a graph edge.
    
    Attributes:
        edge_id: Edge identifier
        score_true: Likelihood this edge represents true adjacency (0.0-1.0)
        score_repeat: Likelihood this is a repeat-induced false edge (0.0-1.0)
        score_chimeric: Likelihood this is an artifact/chimeric edge (0.0-1.0)
        score_allelic: Likelihood this represents allelic variation (0.0-1.0)
        confidence: Overall prediction confidence (0.0-1.0)
    """
    edge_id: int
    score_true: float = 0.5
    score_repeat: float = 0.0
    score_chimeric: float = 0.0
    score_allelic: float = 0.0
    confidence: float = 0.5


@dataclass
class EdgeFeatures:
    """
    Comprehensive feature vector for an edge.
    
    Attributes:
        edge_id: Edge identifier
        from_node: Source node ID
        to_node: Target node ID
        
        # Coverage features
        from_coverage: Coverage of source node
        to_coverage: Coverage of target node
        coverage_ratio: Min/max coverage ratio
        coverage_diff: Absolute coverage difference
        
        # Branching features
        from_out_degree: Out-degree of source node
        to_in_degree: In-degree of target node
        from_in_degree: In-degree of source node
        to_out_degree: Out-degree of target node
        
        # Sequence features
        from_length: Length of source node
        to_length: Length of target node
        kmer_diff: K-mer composition difference (if available)
        
        # Regional complexity
        from_recommended_k: ML-recommended k for source (regional complexity)
        to_recommended_k: ML-recommended k for target
        k_consistency: Similarity of recommended k values
        
        # Support signals
        ul_support_count: Number of UL reads supporting this edge
        hic_weight: Hi-C support weight (0.0-1.0)
        hic_cis_contacts: Hi-C cis contacts
        hic_trans_contacts: Hi-C trans contacts
        
        # Entropy/repeat indicators
        from_entropy: Sequence entropy of source node
        to_entropy: Sequence entropy of target node
        repeat_score: Likelihood nodes are in repeat regions
    """
    edge_id: int
    from_node: int
    to_node: int
    
    # Coverage
    from_coverage: float = 0.0
    to_coverage: float = 0.0
    coverage_ratio: float = 1.0
    coverage_diff: float = 0.0
    
    # Branching
    from_out_degree: int = 0
    to_in_degree: int = 0
    from_in_degree: int = 0
    to_out_degree: int = 0
    
    # Sequence
    from_length: int = 0
    to_length: int = 0
    kmer_diff: float = 0.0
    
    # Regional complexity
    from_recommended_k: Optional[int] = None
    to_recommended_k: Optional[int] = None
    k_consistency: float = 1.0
    
    # Support
    ul_support_count: int = 0
    hic_weight: float = 0.5
    hic_cis_contacts: int = 0
    hic_trans_contacts: int = 0
    
    # Entropy
    from_entropy: float = 0.0
    to_entropy: float = 0.0
    repeat_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML model input."""
        return {
            'coverage_ratio': self.coverage_ratio,
            'coverage_diff': self.coverage_diff,
            'from_out_degree': self.from_out_degree,
            'to_in_degree': self.to_in_degree,
            'branching_complexity': (self.from_out_degree + self.to_in_degree) / 2.0,
            'length_ratio': min(self.from_length, self.to_length) / max(self.from_length, self.to_length) if max(self.from_length, self.to_length) > 0 else 1.0,
            'kmer_diff': self.kmer_diff,
            'k_consistency': self.k_consistency,
            'ul_support': self.ul_support_count,
            'hic_weight': self.hic_weight,
            'hic_support': self.hic_cis_contacts - self.hic_trans_contacts,
            'entropy_avg': (self.from_entropy + self.to_entropy) / 2.0,
            'repeat_score': self.repeat_score
        }


class EdgeFeatureExtractor:
    """
    Extracts comprehensive features from graph edges.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EdgeFeatureExtractor")
    
    def extract_edge_features(
        self,
        edge_id: int,
        from_node: int,
        to_node: int,
        graph,  # DBGGraph or StringGraph
        hic_support: Optional[Dict] = None,
        regional_k_map: Optional[Dict[int, int]] = None,
        ul_support_map: Optional[Dict[int, int]] = None
    ) -> EdgeFeatures:
        """
        Extract comprehensive features for a single edge.
        
        Args:
            edge_id: Edge identifier
            from_node: Source node ID
            to_node: Target node ID
            graph: DBGGraph or StringGraph object
            hic_support: Dict[edge_id] -> HiCEdgeSupport
            regional_k_map: Dict[node_id] -> recommended_k
            ul_support_map: Dict[edge_id] -> ul_support_count
        
        Returns:
            EdgeFeatures object
        """
        features = EdgeFeatures(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node
        )
        
        # Extract node information
        from_node_obj = graph.nodes.get(from_node)
        to_node_obj = graph.nodes.get(to_node)
        
        if not from_node_obj or not to_node_obj:
            self.logger.warning(f"Missing node data for edge {edge_id}")
            return features
        
        # Coverage features
        features.from_coverage = getattr(from_node_obj, 'coverage', 0.0)
        features.to_coverage = getattr(to_node_obj, 'coverage', 0.0)
        
        if features.from_coverage > 0 and features.to_coverage > 0:
            features.coverage_ratio = min(features.from_coverage, features.to_coverage) / max(features.from_coverage, features.to_coverage)
            features.coverage_diff = abs(features.from_coverage - features.to_coverage)
        
        # Branching features
        features.from_out_degree = len(graph.out_edges.get(from_node, set()))
        features.to_in_degree = len(graph.in_edges.get(to_node, set()))
        features.from_in_degree = len(graph.in_edges.get(from_node, set()))
        features.to_out_degree = len(graph.out_edges.get(to_node, set()))
        
        # Sequence features
        features.from_length = getattr(from_node_obj, 'length', 0)
        features.to_length = getattr(to_node_obj, 'length', 0)
        
        # Regional k features
        if regional_k_map:
            features.from_recommended_k = regional_k_map.get(from_node)
            features.to_recommended_k = regional_k_map.get(to_node)
            
            if features.from_recommended_k and features.to_recommended_k:
                k_diff = abs(features.from_recommended_k - features.to_recommended_k)
                k_max = max(features.from_recommended_k, features.to_recommended_k)
                features.k_consistency = 1.0 - (k_diff / k_max) if k_max > 0 else 1.0
        
        # UL support
        if ul_support_map:
            features.ul_support_count = ul_support_map.get(edge_id, 0)
        
        # Hi-C support
        if hic_support and edge_id in hic_support:
            hic = hic_support[edge_id]
            features.hic_weight = hic.hic_weight
            features.hic_cis_contacts = hic.cis_contacts
            features.hic_trans_contacts = hic.trans_contacts
        
        # Entropy and repeat indicators
        features.from_entropy = self._calculate_entropy(from_node_obj)
        features.to_entropy = self._calculate_entropy(to_node_obj)
        features.repeat_score = self._estimate_repeat_likelihood(features)
        
        return features
    
    def _calculate_entropy(self, node) -> float:
        """
        Calculate sequence entropy for a node.
        
        Higher entropy = more complex, less repetitive.
        Lower entropy = more repetitive.
        """
        seq = getattr(node, 'seq', '')
        if not seq:
            return 0.0
        
        # Simple base composition entropy
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for base in seq.upper():
            if base in counts:
                counts[base] += 1
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 (max entropy is 2.0 for 4 bases)
        return entropy / 2.0
    
    def _estimate_repeat_likelihood(self, features: EdgeFeatures) -> float:
        """
        Estimate likelihood that nodes are in repeat regions.
        
        Indicators:
        - High coverage (>2x median)
        - High branching
        - Low entropy
        - Inconsistent k values
        """
        score = 0.0
        
        # High coverage indicator
        avg_coverage = (features.from_coverage + features.to_coverage) / 2.0
        if avg_coverage > 50:  # Arbitrary threshold
            score += 0.3
        
        # High branching indicator
        avg_degree = (features.from_out_degree + features.to_in_degree) / 2.0
        if avg_degree > 2:
            score += 0.3
        
        # Low entropy indicator
        avg_entropy = (features.from_entropy + features.to_entropy) / 2.0
        if avg_entropy < 0.5:
            score += 0.2
        
        # K inconsistency indicator
        if features.k_consistency < 0.7:
            score += 0.2
        
        return min(score, 1.0)


class OverlapAIFilter:
    """
    AI-based edge classification and filtering.
    
    Uses a pluggable ML model to score edges as:
    - True adjacencies
    - Repeat-induced edges
    - Chimeric artifacts
    - Allelic variants
    """
    
    def __init__(self, ml_overlap_model=None):
        """
        Initialize AI filter.
        
        Args:
            ml_overlap_model: Optional ML model with predict(features_dict) -> scores
        """
        self.ml_model = ml_overlap_model
        self.feature_extractor = EdgeFeatureExtractor()
        self.logger = logging.getLogger(f"{__name__}.OverlapAIFilter")
    
    def annotate_edges_with_ai(
        self,
        graph,  # DBGGraph or StringGraph
        edges: List[Tuple[int, int, int]],  # (edge_id, from_node, to_node)
        hic_support: Optional[Dict] = None,
        regional_k_map: Optional[Dict[int, int]] = None,
        ul_support_map: Optional[Dict[int, int]] = None
    ) -> Dict[int, EdgeAIAnnotation]:
        """
        Annotate all edges with AI-based quality scores.
        
        Args:
            graph: DBGGraph or StringGraph
            edges: List of (edge_id, from_node, to_node)
            hic_support: Dict[edge_id] -> HiCEdgeSupport
            regional_k_map: Dict[node_id] -> recommended_k
            ul_support_map: Dict[edge_id] -> ul_support_count
        
        Returns:
            Dict[edge_id] -> EdgeAIAnnotation
        """
        self.logger.info(f"Annotating {len(edges)} edges with AI model")
        
        annotations = {}
        
        for edge_id, from_node, to_node in edges:
            # Extract features
            features = self.feature_extractor.extract_edge_features(
                edge_id, from_node, to_node, graph,
                hic_support, regional_k_map, ul_support_map
            )
            
            # Apply ML model
            if self.ml_model:
                annotation = self._apply_ml_model(features)
            else:
                # Fallback: heuristic scoring
                annotation = self._heuristic_scoring(features)
            
            annotations[edge_id] = annotation
        
        # Log statistics
        high_quality = sum(1 for a in annotations.values() if a.score_true > 0.7)
        low_quality = sum(1 for a in annotations.values() if a.score_true < 0.3)
        
        self.logger.info(
            f"AI annotation complete: {high_quality} high-quality edges, "
            f"{low_quality} low-quality edges"
        )
        
        return annotations
    
    def _apply_ml_model(self, features: EdgeFeatures) -> EdgeAIAnnotation:
        """
        Apply ML model to predict edge quality.
        
        Assumes model.predict(features_dict) returns:
        - Single float: interpreted as score_true
        - Dict with keys: 'true', 'repeat', 'chimeric', 'allelic'
        """
        feature_dict = features.to_dict()
        
        try:
            prediction = self.ml_model.predict(feature_dict)
            
            if isinstance(prediction, dict):
                return EdgeAIAnnotation(
                    edge_id=features.edge_id,
                    score_true=prediction.get('true', 0.5),
                    score_repeat=prediction.get('repeat', 0.0),
                    score_chimeric=prediction.get('chimeric', 0.0),
                    score_allelic=prediction.get('allelic', 0.0),
                    confidence=prediction.get('confidence', 0.5)
                )
            else:
                # Assume single score is score_true
                score = float(prediction)
                return EdgeAIAnnotation(
                    edge_id=features.edge_id,
                    score_true=score,
                    score_repeat=max(0, 1.0 - score - 0.2),
                    score_chimeric=max(0, 1.0 - score - 0.3),
                    confidence=0.7
                )
        except Exception as e:
            self.logger.warning(f"ML model error for edge {features.edge_id}: {e}")
            return self._heuristic_scoring(features)
    
    def _heuristic_scoring(self, features: EdgeFeatures) -> EdgeAIAnnotation:
        """
        Fallback heuristic scoring when no ML model available.
        
        Combines multiple signals:
        - Coverage consistency
        - Low branching
        - Hi-C support
        - UL support
        - Repeat likelihood
        """
        score = 0.5  # Start neutral
        
        # Coverage consistency (+/- 0.2)
        score += (features.coverage_ratio - 0.5) * 0.4
        
        # Low branching is good (+0.2 if degree <= 1)
        avg_degree = (features.from_out_degree + features.to_in_degree) / 2.0
        if avg_degree <= 1:
            score += 0.2
        elif avg_degree > 3:
            score -= 0.2
        
        # Hi-C support (+/- 0.2)
        score += (features.hic_weight - 0.5) * 0.4
        
        # UL support (+0.1 per read, saturate at 5)
        ul_bonus = min(features.ul_support_count * 0.1, 0.5)
        score += ul_bonus
        
        # Repeat penalty
        score -= features.repeat_score * 0.3
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Derive other scores
        score_repeat = features.repeat_score
        score_chimeric = (1.0 - features.coverage_ratio) * 0.5
        score_allelic = 0.1 if features.k_consistency > 0.8 and features.coverage_ratio > 0.8 else 0.0
        
        return EdgeAIAnnotation(
            edge_id=features.edge_id,
            score_true=score,
            score_repeat=score_repeat,
            score_chimeric=score_chimeric,
            score_allelic=score_allelic,
            confidence=0.5  # Heuristic has lower confidence
        )


def annotate_edges_with_ai(
    graph,
    edges: List[Tuple[int, int, int]],
    ml_overlap_model=None,
    hic_support: Optional[Dict] = None,
    regional_k_map: Optional[Dict[int, int]] = None,
    ul_support_map: Optional[Dict[int, int]] = None
) -> Dict[int, EdgeAIAnnotation]:
    """
    Convenience function to annotate edges with AI scoring.
    
    Args:
        graph: DBGGraph or StringGraph
        edges: List of (edge_id, from_node, to_node)
        ml_overlap_model: Optional ML model
        hic_support: Optional Hi-C edge support
        regional_k_map: Optional regional k recommendations
        ul_support_map: Optional UL support counts
    
    Returns:
        Dict[edge_id] -> EdgeAIAnnotation
    """
    filter_obj = OverlapAIFilter(ml_overlap_model=ml_overlap_model)
    return filter_obj.annotate_edges_with_ai(
        graph, edges, hic_support, regional_k_map, ul_support_map
    )
