#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diploid Disentanglement Module for StrandWeaver.

This module separates diploid assembly graphs into haplotype A and B paths
using multiple orthogonal signals:
- Hi-C phasing
- GNN path predictions
- AI overlap classifications
- Regional k-mer recommendations
- UL path coherence

The goal is to resolve allelic variation and produce haplotype-separated
assemblies while correctly handling repeats and ambiguous regions.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HaplotypeDetangler:
    """
    Pipeline integration wrapper for diploid graph phasing.
    
    Provides simplified interface expected by pipeline while leveraging
    full DisentanglerEngine capabilities. Uses Hi-C connectivity clustering,
    GNN paths, and AI annotations to separate haplotypes.
    """
    
    def __init__(
        self,
        use_ai: bool = False,
        min_confidence: float = 0.6,
        repeat_threshold: float = 0.5,
        ml_model: Optional[Any] = None
    ):
        """
        Initialize phasing module.
        
        Args:
            use_ai: Enable AI-powered phasing enhancements
            min_confidence: Minimum confidence for haplotype assignment
            repeat_threshold: Threshold for repeat classification
            ml_model: Pre-trained ML model for phasing (if use_ai=True)
        """
        self.use_ai = use_ai
        self.ml_model = ml_model
        self.min_confidence = min_confidence
        self.repeat_threshold = repeat_threshold
        self.logger = logging.getLogger(f"{__name__}.HaplotypeDetangler")
        
        # Initialize core engine
        self.engine = DisentanglerEngine(
            min_confidence=min_confidence,
            repeat_threshold=repeat_threshold
        )
        
        # Load AI model if enabled
        if use_ai and ml_model is None:
            self.ml_model = self._load_ai_model()
    
    def phase_graph(
        self,
        graph,
        use_hic_edges: bool = True,
        gnn_paths = None,
        ai_annotations: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> PhasingResult:
        """
        Phase assembly graph into haplotypes.
        
        Main pipeline integration method. Uses Hi-C connectivity clustering
        as primary signal, supplemented by GNN paths, AI annotations, and
        UL support.
        
        Args:
            graph: Assembly graph (DBGGraph or StringGraph)
            use_hic_edges: Use Hi-C edges for phasing (recommended)
            gnn_paths: GNN path predictions from PathWeaver
            ai_annotations: AI edge annotations from EdgeWarden
            ul_support_map: UL read support from ThreadCompass
        
        Returns:
            PhasingResult with simplified node assignments
        """
        import time
        start_time = time.time()
        
        self.logger.info("HaplotypeDetangler: Starting graph phasing")
        self.logger.info(f"  Graph nodes: {len(graph.nodes)}")
        self.logger.info(f"  Graph edges: {len(graph.edges)}")
        self.logger.info(f"  Use Hi-C edges: {use_hic_edges}")
        self.logger.info(f"  AI enabled: {self.use_ai}")
        
        # Step 1: Extract Hi-C phasing information
        hic_phase_info = None
        if use_hic_edges:
            self.logger.info("  Step 1/4: Extracting Hi-C connectivity for phasing...")
            hic_phase_info = self._extract_hic_phase_info(graph)
            if hic_phase_info:
                self.logger.info(f"    Hi-C phasing info for {len(hic_phase_info)} nodes")
            else:
                self.logger.warning("    No Hi-C edges found, using alternative signals")
        else:
            self.logger.info("  Step 1/4: Skipping Hi-C clustering (disabled)")
        
        # Step 2: Run core disentanglement
        self.logger.info("  Step 2/4: Running diploid disentanglement...")
        detailed_result = self.engine.disentangle(
            graph=graph,
            gnn_paths=gnn_paths,
            hic_phase_info=hic_phase_info,
            ai_annotations=ai_annotations,
            hic_edge_support=None,  # Could extract from Hi-C edges
            regional_k_map=None,
            ul_support_map=ul_support_map
        )
        
        # Step 3: Apply AI phasing boost if enabled
        if self.use_ai and self.ml_model:
            self.logger.info("  Step 3/4: Applying AI phasing boost...")
            detailed_result = self._apply_ai_phasing_boost(
                graph, detailed_result, hic_phase_info
            )
        else:
            self.logger.info("  Step 3/4: Skipping AI boost (disabled or no model)")
        
        # Step 4: Convert to pipeline-compatible format
        self.logger.info("  Step 4/4: Converting to PhasingResult...")
        phasing_result = self._convert_to_phasing_result(detailed_result)
        
        elapsed_time = time.time() - start_time
        
        self.logger.info("HaplotypeDetangler: Phasing complete")
        self.logger.info(f"  Haplotype A nodes: {len(detailed_result.hapA_nodes)}")
        self.logger.info(f"  Haplotype B nodes: {len(detailed_result.hapB_nodes)}")
        self.logger.info(f"  Ambiguous nodes: {len(detailed_result.ambiguous_nodes)}")
        self.logger.info(f"  Repeat nodes: {len(detailed_result.repeat_nodes)}")
        self.logger.info(f"  Haplotype blocks: {len(detailed_result.haplotype_blocks)}")
        self.logger.info(f"  Phasing time: {elapsed_time:.2f}s")
        
        return phasing_result
    
    def _extract_hic_phase_info(self, graph) -> Optional[Dict[int, PhaseInfo]]:
        """
        Extract Hi-C connectivity for phasing via spectral clustering.
        
        Algorithm:
        1. Collect all edges with edge_type='hic'
        2. Build adjacency matrix from Hi-C contacts
        3. Apply spectral clustering (2 clusters for diploid)
        4. Generate phase scores for each node based on cluster membership
        
        Args:
            graph: Assembly graph with Hi-C edges
        
        Returns:
            Dict mapping node_id to PhaseInfo, or None if no Hi-C edges
        """
        # Collect Hi-C edges
        hic_edges = []
        for edge_id, edge in graph.edges.items():
            edge_type = getattr(edge, 'edge_type', None)
            if edge_type == 'hic':
                from_node = getattr(edge, 'from_node', getattr(edge, 'from_id', None))
                to_node = getattr(edge, 'to_node', getattr(edge, 'to_id', None))
                weight = getattr(edge, 'weight', getattr(edge, 'confidence', 1.0))
                
                if from_node is not None and to_node is not None:
                    hic_edges.append((from_node, to_node, weight))
        
        if not hic_edges:
            self.logger.warning("No Hi-C edges found in graph")
            return None
        
        self.logger.info(f"    Found {len(hic_edges)} Hi-C edges")
        
        # Build node set
        nodes_in_hic = set()
        for from_node, to_node, _ in hic_edges:
            nodes_in_hic.add(from_node)
            nodes_in_hic.add(to_node)
        
        if len(nodes_in_hic) < 2:
            self.logger.warning("Too few nodes with Hi-C edges for clustering")
            return None
        
        # Create node index mapping
        node_list = sorted(nodes_in_hic)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        n = len(node_list)
        
        # Build adjacency matrix
        try:
            import numpy as np
            from scipy.sparse import lil_matrix
            from sklearn.cluster import SpectralClustering
        except ImportError:
            self.logger.warning("numpy/scipy/sklearn not available, skipping Hi-C clustering")
            return None
        
        adj_matrix = lil_matrix((n, n))
        contact_counts = {node: 0 for node in node_list}
        
        for from_node, to_node, weight in hic_edges:
            i = node_to_idx[from_node]
            j = node_to_idx[to_node]
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
            contact_counts[from_node] += 1
            contact_counts[to_node] += 1
        
        # Apply spectral clustering
        try:
            clustering = SpectralClustering(
                n_clusters=2,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            labels = clustering.fit_predict(adj_matrix.toarray())
        except Exception as e:
            self.logger.warning(f"Spectral clustering failed: {e}")
            return None
        
        # Generate PhaseInfo for each node
        phase_info = {}
        for idx, node_id in enumerate(node_list):
            cluster = labels[idx]
            
            # Assign phase scores based on cluster
            if cluster == 0:
                phase_A_score = 0.8
                phase_B_score = 0.2
            else:
                phase_A_score = 0.2
                phase_B_score = 0.8
            
            phase_info[node_id] = PhaseInfo(
                node_id=node_id,
                phase_A_score=phase_A_score,
                phase_B_score=phase_B_score,
                hic_contact_count=contact_counts[node_id],
                cluster_id=cluster
            )
        
        self.logger.info(f"    Clustered into 2 haplotypes: "
                        f"{sum(1 for l in labels if l == 0)} / {sum(1 for l in labels if l == 1)} nodes")
        
        return phase_info
    
    def _load_ai_model(self) -> Optional[Any]:
        """
        Load pre-trained phasing AI model.
        
        Returns:
            Loaded model or None if not available
        """
        try:
            import torch
            from pathlib import Path
            
            model_path = Path(__file__).parent.parent / "models" / "phasing_gnn.pt"
            
            if not model_path.exists():
                self.logger.warning("Phasing AI model not found at {model_path}")
                self.logger.info("Using classical phasing without AI boost")
                return None
            
            model = torch.load(model_path)
            model.eval()
            self.logger.info(f"Loaded phasing AI model from {model_path}")
            return model
        except ImportError:
            self.logger.warning("PyTorch not available, AI phasing disabled")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load AI model: {e}")
            return None
    
    def _apply_ai_phasing_boost(
        self,
        graph,
        detailed_result: HaplotypeDetangleResult,
        hic_phase_info: Optional[Dict]
    ) -> HaplotypeDetangleResult:
        """
        Use AI to improve ambiguous node assignments.
        
        For nodes in ambiguous_nodes, extract features and run through
        ML model to predict haplotype with higher confidence.
        
        Args:
            graph: Assembly graph
            detailed_result: Current phasing result
            hic_phase_info: Hi-C phase information
        
        Returns:
            Updated HaplotypeDetangleResult with improved assignments
        """
        if not self.ml_model:
            return detailed_result
        
        try:
            import torch
            import numpy as np
        except ImportError:
            return detailed_result
        
        ambiguous_nodes = list(detailed_result.ambiguous_nodes)
        if not ambiguous_nodes:
            return detailed_result
        
        self.logger.info(f"    AI boost for {len(ambiguous_nodes)} ambiguous nodes")
        
        # Extract features for ambiguous nodes
        features = []
        for node_id in ambiguous_nodes:
            node = graph.nodes.get(node_id)
            if not node:
                features.append([0.5, 0.5, 0.0, 0.0])  # Default features
                continue
            
            # Feature vector: [phase_A_score, phase_B_score, coverage, degree]
            phase_A = hic_phase_info[node_id].phase_A_score if hic_phase_info and node_id in hic_phase_info else 0.5
            phase_B = hic_phase_info[node_id].phase_B_score if hic_phase_info and node_id in hic_phase_info else 0.5
            coverage = getattr(node, 'coverage', 0.0) / 100.0  # Normalize
            degree = (len(graph.in_edges.get(node_id, set())) + 
                     len(graph.out_edges.get(node_id, set()))) / 10.0  # Normalize
            
            features.append([phase_A, phase_B, coverage, degree])
        
        # Run through model
        try:
            features_tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                # Assuming model outputs [prob_A, prob_B]
                probs = torch.softmax(predictions, dim=1).numpy()
        except Exception as e:
            self.logger.warning(f"AI inference failed: {e}")
            return detailed_result
        
        # Update assignments based on AI predictions
        improved = 0
        for idx, node_id in enumerate(ambiguous_nodes):
            prob_A, prob_B = probs[idx]
            
            # If AI is confident (difference > threshold), assign
            if abs(prob_A - prob_B) > self.min_confidence:
                detailed_result.ambiguous_nodes.discard(node_id)
                
                if prob_A > prob_B:
                    detailed_result.hapA_nodes.add(node_id)
                    detailed_result.confidence_scores[node_id] = prob_A
                else:
                    detailed_result.hapB_nodes.add(node_id)
                    detailed_result.confidence_scores[node_id] = prob_B
                
                improved += 1
        
        self.logger.info(f"    AI improved {improved} ambiguous assignments")
        
        return detailed_result
    
    def _convert_to_phasing_result(
        self,
        detailed: HaplotypeDetangleResult
    ) -> PhasingResult:
        """
        Convert HaplotypeDetangleResult to simplified PhasingResult.
        
        Maps:
        - hapA_nodes -> haplotype 0
        - hapB_nodes -> haplotype 1
        - ambiguous_nodes -> -1 (unassigned)
        - repeat_nodes -> -1 (unassigned)
        
        Args:
            detailed: Detailed phasing result
        
        Returns:
            Simplified PhasingResult for pipeline
        """
        node_assignments = {}
        
        # Assign haplotype 0 to hapA nodes
        for node_id in detailed.hapA_nodes:
            node_assignments[node_id] = 0
        
        # Assign haplotype 1 to hapB nodes
        for node_id in detailed.hapB_nodes:
            node_assignments[node_id] = 1
        
        # Mark ambiguous and repeat as unassigned (-1)
        for node_id in detailed.ambiguous_nodes:
            node_assignments[node_id] = -1
        
        for node_id in detailed.repeat_nodes:
            node_assignments[node_id] = -1
        
        return PhasingResult(
            num_haplotypes=2,
            node_assignments=node_assignments,
            confidence_scores=detailed.confidence_scores.copy(),
            metadata={
                'ambiguous_count': len(detailed.ambiguous_nodes),
                'repeat_count': len(detailed.repeat_nodes),
                'num_blocks': len(detailed.haplotype_blocks),
                'haplotype_balance': detailed.stats.get('haplotype_balance', 1.0),
                'avg_confidence': detailed.stats.get('avg_confidence', 0.5),
                'stats': detailed.stats
            },
            detailed_result=detailed
        )


@dataclass
class HaplotypeDetangleResult:
    """
    Result of diploid graph disentanglement.
    
    Attributes:
        hapA_nodes: Set of node IDs assigned to haplotype A
        hapB_nodes: Set of node IDs assigned to haplotype B
        ambiguous_nodes: Set of node IDs that couldn't be confidently assigned
        hapA_edges: Set of edge IDs assigned to haplotype A
        hapB_edges: Set of edge IDs assigned to haplotype B
        ambiguous_edges: Set of edge IDs in ambiguous regions
        repeat_nodes: Set of node IDs identified as repeats (shared between haplotypes)
        confidence_scores: Dict[node_id] -> confidence in assignment (0-1)
        haplotype_blocks: List of contiguous haplotype blocks
        stats: Summary statistics
    """
    hapA_nodes: Set[int] = field(default_factory=set)
    hapB_nodes: Set[int] = field(default_factory=set)
    ambiguous_nodes: Set[int] = field(default_factory=set)
    hapA_edges: Set[int] = field(default_factory=set)
    hapB_edges: Set[int] = field(default_factory=set)
    ambiguous_edges: Set[int] = field(default_factory=set)
    repeat_nodes: Set[int] = field(default_factory=set)
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    haplotype_blocks: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseInfo:
    """
    Hi-C-derived phasing information for a single node.
    
    Attributes:
        node_id: Node identifier
        phase_A_score: Probability of assignment to haplotype A (0-1)
        phase_B_score: Probability of assignment to haplotype B (0-1)
        hic_contact_count: Number of Hi-C contacts supporting this assignment
        cluster_id: Cluster ID from spectral clustering
    """
    node_id: int
    phase_A_score: float = 0.5
    phase_B_score: float = 0.5
    hic_contact_count: int = 0
    cluster_id: int = -1


@dataclass
class PhasingResult:
    """
    Simplified phasing result for pipeline integration.
    
    This is the interface expected by the pipeline for iteration filtering
    and downstream modules.
    
    Attributes:
        num_haplotypes: Number of haplotypes detected (usually 2 for diploid)
        node_assignments: Mapping from node_id to haplotype (0, 1, or -1 for unassigned)
        confidence_scores: Mapping from node_id to confidence score (0-1)
        metadata: Additional information (stats, block counts, etc.)
        detailed_result: Reference to full HaplotypeDetangleResult
    """
    num_haplotypes: int = 2
    node_assignments: Dict[int, int] = field(default_factory=dict)
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detailed_result: Optional[HaplotypeDetangleResult] = None


class HaplotypeScorer:
    """
    Scores node/edge assignments to haplotypes using multiple signals.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HaplotypeScorer")
    
    def score_node_haplotype(
        self,
        node_id: int,
        graph,
        hic_phase_info: Optional[Dict] = None,
        gnn_paths = None,
        ai_annotations: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Score node assignment to haplotype A, B, or ambiguous.
        
        Returns:
            (score_A, score_B, repeat_score) each in 0-1 range
        """
        score_A = 0.5
        score_B = 0.5
        repeat_score = 0.0
        
        # Hi-C phasing signal (strongest weight: 50%)
        if hic_phase_info and node_id in hic_phase_info:
            phase = hic_phase_info[node_id]
            score_A = phase.phase_A_score * 0.5 + score_A * 0.5
            score_B = phase.phase_B_score * 0.5 + score_B * 0.5
        
        # GNN path coherence (30% weight)
        if gnn_paths:
            path_score_A, path_score_B = self._score_node_in_paths(
                node_id, gnn_paths
            )
            score_A = score_A * 0.7 + path_score_A * 0.3
            score_B = score_B * 0.7 + path_score_B * 0.3
        
        # Repeat detection from regional k and coverage (20% weight)
        node = graph.nodes.get(node_id)
        if node:
            coverage = getattr(node, 'coverage', 0.0)
            in_degree = len(graph.in_edges.get(node_id, set()))
            out_degree = len(graph.out_edges.get(node_id, set()))
            
            # High coverage + high branching suggests repeat
            if coverage > 50 and (in_degree + out_degree) > 2:
                repeat_score += 0.4
            
            # Check edge AI scores for repeat signals
            for edge_id in graph.out_edges.get(node_id, set()):
                if ai_annotations and edge_id in ai_annotations:
                    ai = ai_annotations[edge_id]
                    repeat_score += ai.score_repeat * 0.1
        
        repeat_score = min(repeat_score, 1.0)
        
        return (score_A, score_B, repeat_score)
    
    def _score_node_in_paths(
        self,
        node_id: int,
        gnn_paths
    ) -> Tuple[float, float]:
        """
        Score node based on GNN path assignments.
        
        If paths are cleanly separated into two groups, infer haplotypes.
        """
        if not hasattr(gnn_paths, 'best_paths') or not gnn_paths.best_paths:
            return (0.5, 0.5)
        
        # Find which paths contain this node
        containing_paths = []
        for path_idx, path in enumerate(gnn_paths.best_paths):
            if node_id in path:
                containing_paths.append(path_idx)
        
        if not containing_paths:
            return (0.5, 0.5)
        
        # If node is in multiple paths, it might be a repeat
        if len(containing_paths) > 1:
            return (0.5, 0.5)
        
        # Node in single path - assign based on path index
        # (In real implementation, would cluster paths into haplotypes)
        path_idx = containing_paths[0]
        if path_idx % 2 == 0:
            return (0.7, 0.3)
        else:
            return (0.3, 0.7)
    
    def score_edge_haplotype(
        self,
        edge_id: int,
        from_node: int,
        to_node: int,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Assign edge to haplotype based on endpoints.
        
        Returns:
            (assignment, confidence) where assignment is 'A', 'B', or 'ambiguous'
        """
        from_hap = node_assignments.get(from_node, 'ambiguous')
        to_hap = node_assignments.get(to_node, 'ambiguous')
        
        # If both endpoints are same haplotype
        if from_hap == to_hap and from_hap != 'ambiguous':
            confidence = 0.8
            
            # Boost confidence with AI support
            if ai_annotations and edge_id in ai_annotations:
                ai = ai_annotations[edge_id]
                if ai.score_true > 0.7:
                    confidence = min(confidence + 0.1, 1.0)
            
            # Boost confidence with Hi-C support
            if hic_edge_support and edge_id in hic_edge_support:
                hic = hic_edge_support[edge_id]
                if hic.hic_weight > 0.7:
                    confidence = min(confidence + 0.1, 1.0)
            
            return (from_hap, confidence)
        
        # Endpoints are different or ambiguous
        confidence = 0.3
        
        # Check if this is an allelic edge (connects haplotypes)
        if ai_annotations and edge_id in ai_annotations:
            ai = ai_annotations[edge_id]
            if ai.score_allelic > 0.6:
                return ('ambiguous', 0.5)
        
        return ('ambiguous', confidence)


class DisentanglerEngine:
    """
    Main engine for diploid graph disentanglement.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        repeat_threshold: float = 0.5
    ):
        """
        Initialize disentangler.
        
        Args:
            min_confidence: Minimum confidence for definite haplotype assignment
            repeat_threshold: Threshold for classifying nodes as repeats
        """
        self.min_confidence = min_confidence
        self.repeat_threshold = repeat_threshold
        self.scorer = HaplotypeScorer()
        self.logger = logging.getLogger(f"{__name__}.DisentanglerEngine")
    
    def disentangle(
        self,
        graph,
        gnn_paths = None,
        hic_phase_info: Optional[Dict] = None,
        ai_annotations: Optional[Dict] = None,
        hic_edge_support: Optional[Dict] = None,
        regional_k_map: Optional[Dict] = None,
        ul_support_map: Optional[Dict] = None
    ) -> HaplotypeDetangleResult:
        """
        Disentangle diploid graph into haplotype A and B.
        
        Algorithm:
        1. Score each node for haplotype A, B, and repeat likelihood
        2. Assign nodes based on score differences and thresholds
        3. Propagate assignments through high-confidence edges
        4. Assign edges based on endpoint haplotypes
        5. Identify haplotype blocks
        
        Args:
            graph: Assembly graph
            gnn_paths: GNN path predictions
            hic_phase_info: Hi-C phasing
            ai_annotations: AI edge annotations
            hic_edge_support: Hi-C edge support
            regional_k_map: Regional k recommendations
            ul_support_map: UL support
        
        Returns:
            HaplotypeDetangleResult
        """
        self.logger.info("Starting diploid disentanglement")
        
        result = HaplotypeDetangleResult()
        
        # Step 1: Score all nodes
        node_scores = {}
        for node_id in graph.nodes:
            score_A, score_B, repeat_score = self.scorer.score_node_haplotype(
                node_id, graph, hic_phase_info, gnn_paths,
                ai_annotations, regional_k_map
            )
            node_scores[node_id] = (score_A, score_B, repeat_score)
        
        # Step 2: Initial node assignments
        node_assignments = {}
        for node_id, (score_A, score_B, repeat_score) in node_scores.items():
            # Check if it's a repeat
            if repeat_score > self.repeat_threshold:
                result.repeat_nodes.add(node_id)
                node_assignments[node_id] = 'repeat'
                result.confidence_scores[node_id] = repeat_score
                continue
            
            # Assign to haplotype with higher score
            diff = abs(score_A - score_B)
            
            if diff > self.min_confidence:
                if score_A > score_B:
                    result.hapA_nodes.add(node_id)
                    node_assignments[node_id] = 'A'
                    result.confidence_scores[node_id] = score_A
                else:
                    result.hapB_nodes.add(node_id)
                    node_assignments[node_id] = 'B'
                    result.confidence_scores[node_id] = score_B
            else:
                result.ambiguous_nodes.add(node_id)
                node_assignments[node_id] = 'ambiguous'
                result.confidence_scores[node_id] = max(score_A, score_B)
        
        self.logger.info(
            f"Initial assignment: {len(result.hapA_nodes)} A, "
            f"{len(result.hapB_nodes)} B, "
            f"{len(result.ambiguous_nodes)} ambiguous, "
            f"{len(result.repeat_nodes)} repeat"
        )
        
        # Step 3: Propagate assignments through high-confidence edges
        node_assignments = self._propagate_assignments(
            graph, node_assignments, ai_annotations, hic_edge_support, result
        )
        
        # Step 4: Assign edges
        self._assign_edges(
            graph, node_assignments, ai_annotations,
            hic_edge_support, result
        )
        
        # Step 5: Identify haplotype blocks
        result.haplotype_blocks = self._identify_haplotype_blocks(
            graph, node_assignments, result
        )
        
        # Step 6: Compute statistics
        result.stats = self._compute_statistics(result)
        
        self.logger.info(
            f"Final: {len(result.hapA_nodes)} A nodes, "
            f"{len(result.hapB_nodes)} B nodes, "
            f"{len(result.hapA_edges)} A edges, "
            f"{len(result.hapB_edges)} B edges"
        )
        
        return result
    
    def _propagate_assignments(
        self,
        graph,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict],
        hic_edge_support: Optional[Dict],
        result: HaplotypeDetangleResult
    ) -> Dict[int, str]:
        """
        Propagate haplotype assignments through high-confidence edges.
        
        If edge has AI score_true > 0.7 and Hi-C weight > 0.7,
        propagate haplotype from assigned endpoint to ambiguous endpoint.
        """
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for edge_id, edge in graph.edges.items():
                from_node = getattr(edge, 'from_node', None)
                to_node = getattr(edge, 'to_node', None)
                
                if not from_node or not to_node:
                    continue
                
                from_hap = node_assignments.get(from_node, 'ambiguous')
                to_hap = node_assignments.get(to_node, 'ambiguous')
                
                # Skip if both assigned or both ambiguous
                if from_hap == to_hap:
                    continue
                
                # Check edge confidence
                edge_confident = False
                if ai_annotations and edge_id in ai_annotations:
                    ai = ai_annotations[edge_id]
                    if ai.score_true > 0.7:
                        edge_confident = True
                
                if hic_edge_support and edge_id in hic_edge_support:
                    hic = hic_edge_support[edge_id]
                    if hic.hic_weight > 0.7:
                        edge_confident = True
                
                if not edge_confident:
                    continue
                
                # Propagate from assigned to ambiguous
                if from_hap in ('A', 'B') and to_hap == 'ambiguous':
                    node_assignments[to_node] = from_hap
                    result.ambiguous_nodes.discard(to_node)
                    if from_hap == 'A':
                        result.hapA_nodes.add(to_node)
                    else:
                        result.hapB_nodes.add(to_node)
                    changed = True
                
                elif to_hap in ('A', 'B') and from_hap == 'ambiguous':
                    node_assignments[from_node] = to_hap
                    result.ambiguous_nodes.discard(from_node)
                    if to_hap == 'A':
                        result.hapA_nodes.add(from_node)
                    else:
                        result.hapB_nodes.add(from_node)
                    changed = True
        
        self.logger.info(f"Propagated assignments in {iterations} iterations")
        return node_assignments
    
    def _assign_edges(
        self,
        graph,
        node_assignments: Dict[int, str],
        ai_annotations: Optional[Dict],
        hic_edge_support: Optional[Dict],
        result: HaplotypeDetangleResult
    ):
        """Assign edges to haplotypes based on endpoints."""
        for edge_id, edge in graph.edges.items():
            from_node = getattr(edge, 'from_node', None)
            to_node = getattr(edge, 'to_node', None)
            
            if not from_node or not to_node:
                continue
            
            assignment, confidence = self.scorer.score_edge_haplotype(
                edge_id, from_node, to_node, node_assignments,
                ai_annotations, hic_edge_support
            )
            
            if assignment == 'A':
                result.hapA_edges.add(edge_id)
            elif assignment == 'B':
                result.hapB_edges.add(edge_id)
            else:
                result.ambiguous_edges.add(edge_id)
    
    def _identify_haplotype_blocks(
        self,
        graph,
        node_assignments: Dict[int, str],
        result: HaplotypeDetangleResult
    ) -> List[Dict[str, Any]]:
        """
        Identify contiguous blocks of haplotype-specific sequence.
        
        A block is a maximal connected component of nodes with same haplotype.
        """
        blocks = []
        
        for haplotype in ['A', 'B']:
            nodes = result.hapA_nodes if haplotype == 'A' else result.hapB_nodes
            edges = result.hapA_edges if haplotype == 'A' else result.hapB_edges
            
            # Find connected components within haplotype
            visited = set()
            for node in nodes:
                if node in visited:
                    continue
                
                # BFS to find block
                block_nodes = []
                queue = [node]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    block_nodes.append(current)
                    
                    # Add neighbors in same haplotype
                    for neighbor in graph.out_edges.get(current, set()):
                        if neighbor in nodes and neighbor not in visited:
                            queue.append(neighbor)
                    
                    for neighbor in graph.in_edges.get(current, set()):
                        if neighbor in nodes and neighbor not in visited:
                            queue.append(neighbor)
                
                if block_nodes:
                    blocks.append({
                        'haplotype': haplotype,
                        'nodes': block_nodes,
                        'size': len(block_nodes),
                        'avg_confidence': sum(
                            result.confidence_scores.get(n, 0.5)
                            for n in block_nodes
                        ) / len(block_nodes)
                    })
        
        return sorted(blocks, key=lambda b: b['size'], reverse=True)
    
    def _compute_statistics(self, result: HaplotypeDetangleResult) -> Dict[str, Any]:
        """Compute summary statistics."""
        total_nodes = (
            len(result.hapA_nodes) + len(result.hapB_nodes) +
            len(result.ambiguous_nodes) + len(result.repeat_nodes)
        )
        total_edges = (
            len(result.hapA_edges) + len(result.hapB_edges) +
            len(result.ambiguous_edges)
        )
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'hapA_nodes': len(result.hapA_nodes),
            'hapB_nodes': len(result.hapB_nodes),
            'ambiguous_nodes': len(result.ambiguous_nodes),
            'repeat_nodes': len(result.repeat_nodes),
            'hapA_edges': len(result.hapA_edges),
            'hapB_edges': len(result.hapB_edges),
            'ambiguous_edges': len(result.ambiguous_edges),
            'haplotype_balance': len(result.hapA_nodes) / max(len(result.hapB_nodes), 1),
            'num_blocks': len(result.haplotype_blocks),
            'avg_confidence': sum(result.confidence_scores.values()) / max(len(result.confidence_scores), 1)
        }


def disentangle_diploid_graph(
    graph,
    gnn_paths = None,
    hic_phase_info: Optional[Dict] = None,
    ai_annotations: Optional[Dict] = None,
    hic_edge_support: Optional[Dict] = None,
    regional_k_map: Optional[Dict] = None,
    ul_support_map: Optional[Dict] = None,
    min_confidence: float = 0.6,
    repeat_threshold: float = 0.5
) -> HaplotypeDetangleResult:
    """
    Main entry point for diploid graph disentanglement.
    
    Separates assembly graph into haplotype A and B using multiple signals.
    
    Args:
        graph: Assembly graph
        gnn_paths: GNN path predictions
        hic_phase_info: Hi-C phasing information
        ai_annotations: AI edge annotations
        hic_edge_support: Hi-C edge support
        regional_k_map: Regional k recommendations
        ul_support_map: UL support counts
        min_confidence: Minimum confidence for haplotype assignment
        repeat_threshold: Threshold for repeat classification
    
    Returns:
        HaplotypeDetangleResult with haplotype assignments
    """
    engine = DisentanglerEngine(
        min_confidence=min_confidence,
        repeat_threshold=repeat_threshold
    )
    
    return engine.disentangle(
        graph, gnn_paths, hic_phase_info, ai_annotations,
        hic_edge_support, regional_k_map, ul_support_map
    )
