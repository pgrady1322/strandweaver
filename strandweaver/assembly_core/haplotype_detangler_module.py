#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Haplotype Detangler — Hi-C and UL-read phasing-aware haplotype resolution.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations  # Enable forward references
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class Ploidy(IntEnum):
    """Assembly ploidy level. Controls whether phasing is applied."""
    HAPLOID = 1
    DIPLOID = 2
    TRIPLOID = 3
    TETRAPLOID = 4


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
        ml_model: Optional[Any] = None,
        ploidy: int = 2
    ):
        """
        Initialize phasing module.
        
        Args:
            use_ai: Enable AI-powered phasing enhancements
            min_confidence: Minimum confidence for haplotype assignment
            repeat_threshold: Threshold for repeat classification
            ml_model: Pre-trained ML model for phasing (if use_ai=True)
            ploidy: Assembly ploidy (1=haploid skips phasing, 2+=diploid+)
        """
        self.use_ai = use_ai
        self.ml_model = ml_model
        self.min_confidence = min_confidence
        self.repeat_threshold = repeat_threshold
        self.ploidy = max(1, int(ploidy))
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
        
        For diploid+ assemblies (ploidy >= 2), uses bubble-aware local
        phasing: heterozygous bubbles in the graph are detected, Hi-C
        contacts are binned per-bubble, and local phase decisions are
        chained along each connected component.  This replaces the
        previous global spectral clustering approach that incorrectly
        separated chromosomes rather than haplotypes (G2 fix).
        
        For haploid assemblies (ploidy == 1), phasing is skipped entirely
        and all nodes are marked as haplotype 0.
        
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
        self.logger.info(f"  Ploidy: {self.ploidy}")
        self.logger.info(f"  Use Hi-C edges: {use_hic_edges}")
        self.logger.info(f"  AI enabled: {self.use_ai}")
        
        # ── Haploid fast-path: skip phasing entirely ──
        if self.ploidy < 2:
            self.logger.info("  Ploidy=1 (haploid): skipping phasing")
            node_assignments = {nid: 0 for nid in graph.nodes}
            return PhasingResult(
                num_haplotypes=1,
                node_assignments=node_assignments,
                confidence_scores={nid: 1.0 for nid in graph.nodes},
                metadata={'haploid_skip': True, 'ploidy': 1},
            )
        
        # Step 1: Bubble-aware Hi-C phasing (G2 fix)
        # Instead of global spectral clustering (which separates chromosomes
        # not haplotypes), detect heterozygous bubbles in the graph and use
        # Hi-C contacts within each bubble to determine local phase.
        hic_phase_info = None
        if use_hic_edges:
            self.logger.info("  Step 1/4: Bubble-aware Hi-C phasing...")
            hic_phase_info = self._bubble_aware_hic_phasing(graph)
            if hic_phase_info:
                self.logger.info(f"    Bubble-aware phasing info for {len(hic_phase_info)} nodes")
            else:
                self.logger.warning("    No bubbles or Hi-C edges found, using alternative signals")
        else:
            self.logger.info("  Step 1/4: Skipping Hi-C clustering (disabled)")
        
        # Step 2: Run core disentanglement
        self.logger.info("  Step 2/4: Running diploid disentanglement...")
        detailed_result = self.engine.disentangle(
            graph=graph,
            gnn_paths=gnn_paths,
            hic_phase_info=hic_phase_info,
            ai_annotations=ai_annotations,
            hic_edge_support=None,
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
    
    def _bubble_aware_hic_phasing(self, graph) -> Optional[Dict[int, PhaseInfo]]:
        """
        Bubble-aware local Hi-C phasing (G2 fix).
        
        Instead of running global spectral clustering on the full Hi-C
        contact matrix (which separates chromosomes, not haplotypes),
        this method:
        
        1. Detects heterozygous bubbles (source → 2 arms → sink) in the
           assembly graph.
        2. Bins Hi-C contacts per bubble: reads mapping to opposite arms
           of the SAME bubble are trans-allelic; reads linking arms of
           DIFFERENT bubbles reveal which arms are co-phased.
        3. Chains local phase decisions across bubbles along each
           connected component using a greedy propagation.
        
        Based on the approach used by hifiasm (Cheng et al., 2021).
        
        Args:
            graph: Assembly graph with Hi-C edges
        
        Returns:
            Dict mapping node_id to PhaseInfo, or None if insufficient data
        """
        # ── Step 1: Collect Hi-C edges ──
        hic_edges = []
        for edge_id, edge in graph.edges.items():
            edge_type = getattr(edge, 'edge_type', None)
            if edge_type == 'hic':
                from_node = getattr(edge, 'from_node',
                            getattr(edge, 'from_id',
                            getattr(edge, 'source', None)))
                to_node = getattr(edge, 'to_node',
                          getattr(edge, 'to_id',
                          getattr(edge, 'target', None)))
                weight = getattr(edge, 'weight',
                         getattr(edge, 'confidence',
                         getattr(edge, 'contact_count', 1.0)))
                if from_node is not None and to_node is not None:
                    hic_edges.append((from_node, to_node, weight))
        
        if not hic_edges:
            self.logger.warning("No Hi-C edges found in graph")
            return None
        
        self.logger.info(f"    Found {len(hic_edges)} Hi-C edges")
        
        # Build a quick Hi-C contact lookup: (nodeA, nodeB) -> total weight
        hic_contacts: Dict[Tuple[int, int], float] = defaultdict(float)
        for n1, n2, w in hic_edges:
            pair = (min(n1, n2), max(n1, n2))
            hic_contacts[pair] += w
        
        # ── Step 2: Detect bubbles ──
        bubbles = self._detect_bubbles(graph)
        if not bubbles:
            self.logger.warning("    No bubbles detected — falling back to spectral clustering")
            return self._extract_hic_phase_info(graph)
        
        self.logger.info(f"    Detected {len(bubbles)} heterozygous bubbles")
        
        # Map each node to the bubble(s) and arm it belongs to.
        # bubble = (source, sink, arm_a_nodes, arm_b_nodes)
        node_to_bubble: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for bub_idx, (source, sink, arm_a, arm_b) in enumerate(bubbles):
            for nid in arm_a:
                node_to_bubble[nid].append((bub_idx, 'A'))
            for nid in arm_b:
                node_to_bubble[nid].append((bub_idx, 'B'))
        
        # ── Step 3: Score inter-bubble phasing via Hi-C contacts ──
        # For each pair of bubbles that share Hi-C contacts between their
        # arms, tally: do contacts link same-labelled arms (cis = co-phased)
        # or opposite-labelled arms (trans = anti-phased)?
        # Edge between bubble pair: positive weight → cis, negative → trans.
        bubble_graph: Dict[Tuple[int, int], float] = defaultdict(float)
        
        for (n1, n2), contact_w in hic_contacts.items():
            entries_1 = node_to_bubble.get(n1, [])
            entries_2 = node_to_bubble.get(n2, [])
            
            for bub_i, arm_i in entries_1:
                for bub_j, arm_j in entries_2:
                    if bub_i == bub_j:
                        # Intra-bubble trans contact (confirms they are
                        # on opposite haplotypes — already encoded by
                        # the bubble structure itself).
                        continue
                    
                    pair = (min(bub_i, bub_j), max(bub_i, bub_j))
                    # Same arm label → cis (want to keep same phase)
                    # Different arm label → trans (want to flip phase)
                    if arm_i == arm_j:
                        bubble_graph[pair] += contact_w   # cis evidence
                    else:
                        bubble_graph[pair] -= contact_w   # trans evidence
        
        # ── Step 4: Chain phases across bubbles (greedy BFS) ──
        num_bubbles = len(bubbles)
        bubble_phase: Dict[int, int] = {}   # bubble_idx → 0 or 1
        
        # Build adjacency list for the bubble graph
        bubble_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (bi, bj), w in bubble_graph.items():
            bubble_adj[bi].append((bj, w))
            bubble_adj[bj].append((bi, w))
        
        # BFS from highest-connectivity bubbles outward
        bubble_order = sorted(
            range(num_bubbles),
            key=lambda b: sum(abs(w) for _, w in bubble_adj.get(b, [])),
            reverse=True,
        )
        
        for start_bub in bubble_order:
            if start_bub in bubble_phase:
                continue
            # Seed this component
            bubble_phase[start_bub] = 0
            queue = [start_bub]
            while queue:
                current = queue.pop(0)
                for neighbor, weight in bubble_adj.get(current, []):
                    if neighbor in bubble_phase:
                        continue
                    # weight > 0 → cis (same phase), weight < 0 → trans (flip)
                    if weight >= 0:
                        bubble_phase[neighbor] = bubble_phase[current]
                    else:
                        bubble_phase[neighbor] = 1 - bubble_phase[current]
                    queue.append(neighbor)
        
        # Assign any isolated bubbles (no inter-bubble contacts)
        for bub_idx in range(num_bubbles):
            if bub_idx not in bubble_phase:
                bubble_phase[bub_idx] = 0
        
        # ── Step 5: Convert bubble phases to node PhaseInfo ──
        phase_info: Dict[int, PhaseInfo] = {}
        
        for bub_idx, (source, sink, arm_a, arm_b) in enumerate(bubbles):
            bp = bubble_phase[bub_idx]  # 0 or 1
            
            # If bubble phase = 0: arm_a → hapA, arm_b → hapB
            # If bubble phase = 1: arm_a → hapB, arm_b → hapA (flip)
            for nid in arm_a:
                if bp == 0:
                    pa, pb = 0.85, 0.15
                    cluster = 0
                else:
                    pa, pb = 0.15, 0.85
                    cluster = 1
                
                contact_count = sum(
                    1 for (n1, n2) in hic_contacts
                    if nid in (n1, n2)
                )
                phase_info[nid] = PhaseInfo(
                    node_id=nid,
                    phase_A_score=pa,
                    phase_B_score=pb,
                    hic_contact_count=contact_count,
                    cluster_id=cluster,
                )
            
            for nid in arm_b:
                if bp == 0:
                    pa, pb = 0.15, 0.85
                    cluster = 1
                else:
                    pa, pb = 0.85, 0.15
                    cluster = 0
                
                contact_count = sum(
                    1 for (n1, n2) in hic_contacts
                    if nid in (n1, n2)
                )
                phase_info[nid] = PhaseInfo(
                    node_id=nid,
                    phase_A_score=pa,
                    phase_B_score=pb,
                    hic_contact_count=contact_count,
                    cluster_id=cluster,
                )
            
            # Source and sink nodes are shared (homozygous flanks)
            for shared_nid in (source, sink):
                if shared_nid not in phase_info:
                    phase_info[shared_nid] = PhaseInfo(
                        node_id=shared_nid,
                        phase_A_score=0.5,
                        phase_B_score=0.5,
                        hic_contact_count=0,
                        cluster_id=-1,
                    )
        
        hap_a_count = sum(1 for p in phase_info.values() if p.phase_A_score > 0.6)
        hap_b_count = sum(1 for p in phase_info.values() if p.phase_B_score > 0.6)
        self.logger.info(
            f"    Bubble-aware phasing: {hap_a_count} hap-A nodes, "
            f"{hap_b_count} hap-B nodes from {len(bubbles)} bubbles"
        )
        
        return phase_info
    
    def _detect_bubbles(
        self,
        graph,
        max_arm_length: int = 50
    ) -> List[Tuple[int, int, List[int], List[int]]]:
        """
        Detect heterozygous bubble structures in the assembly graph.
        
        A bubble is defined as two diverging paths from a common source
        node that reconverge at a common sink node.  This is the
        structural signature of heterozygosity in diploid genomes.
        
        Algorithm mirrors DBGEngine._pop_error_bubbles() bubble detection
        but returns bubbles instead of collapsing them.
        
        Args:
            graph: Assembly graph
            max_arm_length: Maximum number of nodes in each bubble arm
        
        Returns:
            List of (source_id, sink_id, arm_a_nodes, arm_b_nodes)
        """
        bubbles = []
        seen_sources: Set[int] = set()
        
        for source_id in graph.nodes:
            if source_id in seen_sources:
                continue
            
            # Only consider sequence edges (not Hi-C proximity edges)
            out_edge_ids = [
                eid for eid in graph.out_edges.get(source_id, set())
                if eid in graph.edges and getattr(graph.edges[eid], 'edge_type', None) != 'hic'
            ]
            if len(out_edge_ids) < 2:
                continue
            
            # Check all pairs of outgoing edges for bubble structures
            for i in range(len(out_edge_ids)):
                for j in range(i + 1, len(out_edge_ids)):
                    if out_edge_ids[i] not in graph.edges or out_edge_ids[j] not in graph.edges:
                        continue
                    
                    arm_a_start = graph.edges[out_edge_ids[i]].to_id
                    arm_b_start = graph.edges[out_edge_ids[j]].to_id
                    
                    # Walk each arm to find convergence
                    path_a, sink_a = self._walk_arm(graph, arm_a_start, max_arm_length)
                    path_b, sink_b = self._walk_arm(graph, arm_b_start, max_arm_length)
                    
                    if sink_a is None or sink_b is None:
                        continue
                    if sink_a != sink_b:
                        continue
                    
                    # Valid bubble found
                    bubbles.append((source_id, sink_a, path_a, path_b))
                    seen_sources.add(source_id)
        
        return bubbles
    
    def _walk_arm(
        self,
        graph,
        start_node: int,
        max_length: int
    ) -> Tuple[List[int], Optional[int]]:
        """
        Walk a linear path from start_node until a convergence point.
        
        A convergence point is a node with in_degree > 1 (another path
        merges here).  Returns the path and the convergence node, or
        (path, None) if the arm branches or dead-ends.
        
        Only follows sequence-based edges (ignores Hi-C proximity edges).
        
        Args:
            graph: Assembly graph
            start_node: First node of the arm
            max_length: Maximum nodes to walk
        
        Returns:
            (path_node_ids, sink_node_or_None)
        """
        path = [start_node]
        current = start_node
        
        for _ in range(max_length):
            # Only count sequence edges (skip Hi-C edges)
            out_edges = [
                eid for eid in graph.out_edges.get(current, set())
                if eid in graph.edges and getattr(graph.edges[eid], 'edge_type', None) != 'hic'
            ]
            if len(out_edges) != 1:
                # Dead-end or branch — not a clean bubble arm
                return path, None
            
            edge = graph.edges.get(out_edges[0])
            if edge is None:
                return path, None
            
            next_node = edge.to_id
            
            # Check if next node is a convergence point (sequence edges only)
            in_edges_seq = [
                eid for eid in graph.in_edges.get(next_node, set())
                if eid in graph.edges and getattr(graph.edges[eid], 'edge_type', None) != 'hic'
            ]
            if len(in_edges_seq) > 1:
                # Convergence point — this is the potential sink
                return path, next_node
            
            path.append(next_node)
            current = next_node
        
        # Exceeded max length
        return path, None
    
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
                from_node = getattr(edge, 'from_node',
                            getattr(edge, 'from_id',
                            getattr(edge, 'source', None)))
                to_node = getattr(edge, 'to_node',
                          getattr(edge, 'to_id',
                          getattr(edge, 'target', None)))
                weight = getattr(edge, 'weight',
                         getattr(edge, 'confidence',
                         getattr(edge, 'contact_count', 1.0)))
                
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
            import numpy as np
        except ImportError:
            return detailed_result
        
        # torch is only needed for PyTorch models, not XGBoost
        torch = None
        if not hasattr(self.ml_model, 'predict_proba'):
            try:
                import torch as _torch
                torch = _torch
            except ImportError:
                self.logger.warning("PyTorch not available for AI phasing model")
                return detailed_result
        
        ambiguous_nodes = list(detailed_result.ambiguous_nodes)
        if not ambiguous_nodes:
            return detailed_result
        
        self.logger.info(f"    AI boost for {len(ambiguous_nodes)} ambiguous nodes")
        
        # Extract features for ambiguous nodes
        # XGBoost diploid model expects 26 NODE_SIGNAL_FEATURES;
        # PyTorch GNN uses simplified 4-feature vector
        is_xgb = hasattr(self.ml_model, 'predict_proba')
        
        features = []
        for node_id in ambiguous_nodes:
            node = graph.nodes.get(node_id)
            if not node:
                if is_xgb:
                    features.append([0.0] * 26)
                else:
                    features.append([0.5, 0.5, 0.0, 0.0])
                continue
            
            if is_xgb:
                # Full 26-feature vector matching NODE_SIGNAL_FEATURES schema:
                # coverage, gc_content, repeat_fraction, kmer_diversity,
                # branching_factor, hic_contact_density, allele_frequency,
                # heterozygosity, phase_consistency, mappability,
                # hic_intra_contacts, hic_inter_contacts,
                # hic_contact_ratio, hic_phase_signal,
                # clustering_coeff, component_size,
                # shannon_entropy, dinucleotide_bias,
                # homopolymer_max_run, homopolymer_density, low_complexity_fraction,
                # coverage_skewness, coverage_kurtosis, coverage_cv,
                # coverage_p10, coverage_p90
                seq = getattr(node, 'seq', '')
                cov = getattr(node, 'coverage', 0.0)
                seq_len = max(len(seq), 1)
                
                # GC content
                gc = (seq.count('G') + seq.count('C') + seq.count('g') + seq.count('c')) / seq_len if seq else 0.0
                
                # Repeat fraction (from metadata or estimate via k-mer compression)
                repeat_frac = node.metadata.get('repeat_fraction', 0.0) if hasattr(node, 'metadata') else 0.0
                
                # K-mer diversity (unique k-mers / total k-mers)
                k = 4
                if len(seq) >= k:
                    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
                    kmer_div = len(set(kmers)) / max(len(kmers), 1)
                else:
                    kmer_div = 1.0
                
                # Branching factor
                in_deg = len(graph.in_edges.get(node_id, set()))
                out_deg = len(graph.out_edges.get(node_id, set()))
                branch = (in_deg + out_deg) / 2.0
                
                # Hi-C features
                hic_info = hic_phase_info.get(node_id) if hic_phase_info else None
                hic_density = getattr(hic_info, 'contact_density', 0.0) if hic_info else 0.0
                allele_freq = getattr(hic_info, 'allele_frequency', 0.5) if hic_info else 0.5
                het = getattr(hic_info, 'heterozygosity', 0.0) if hic_info else 0.0
                phase_consist = getattr(hic_info, 'phase_consistency', 0.0) if hic_info else 0.0
                mappability = getattr(hic_info, 'mappability', 1.0) if hic_info else 1.0
                hic_intra = getattr(hic_info, 'intra_contacts', 0.0) if hic_info else 0.0
                hic_inter = getattr(hic_info, 'inter_contacts', 0.0) if hic_info else 0.0
                hic_ratio = hic_intra / max(hic_intra + hic_inter, 1.0)
                phase_A = getattr(hic_info, 'phase_A_score', 0.5) if hic_info else 0.5
                phase_B = getattr(hic_info, 'phase_B_score', 0.5) if hic_info else 0.5
                hic_phase_sig = phase_A - phase_B
                
                # Graph topology
                cluster_coeff = node.metadata.get('clustering_coeff', 0.0) if hasattr(node, 'metadata') else 0.0
                comp_size = node.metadata.get('component_size', 1) if hasattr(node, 'metadata') else 1
                
                # Sequence complexity
                from math import log2
                char_counts = [seq.upper().count(c) for c in 'ACGT']
                entropy = -sum((c/seq_len) * log2(c/seq_len + 1e-10) for c in char_counts if c > 0) if seq else 0.0
                
                # Dinucleotide bias
                if len(seq) >= 2:
                    dinucs = [seq[i:i+2] for i in range(len(seq)-1)]
                    dinuc_div = len(set(dinucs)) / max(len(dinucs), 1)
                    dinuc_bias = 1.0 - dinuc_div
                else:
                    dinuc_bias = 0.0
                
                # Homopolymer features
                max_run = 0
                cur_run = 1
                total_hp = 0
                for i in range(1, len(seq)):
                    if seq[i] == seq[i-1]:
                        cur_run += 1
                    else:
                        if cur_run >= 3:
                            total_hp += cur_run
                        max_run = max(max_run, cur_run)
                        cur_run = 1
                max_run = max(max_run, cur_run)
                if cur_run >= 3:
                    total_hp += cur_run
                hp_density = total_hp / seq_len
                
                # Low complexity fraction (runs of same base >= 3)
                low_complex = total_hp / seq_len
                
                # Coverage distribution features (not available per-node, use defaults)
                cov_skew = node.metadata.get('coverage_skewness', 0.0) if hasattr(node, 'metadata') else 0.0
                cov_kurt = node.metadata.get('coverage_kurtosis', 0.0) if hasattr(node, 'metadata') else 0.0
                cov_cv = node.metadata.get('coverage_cv', 0.0) if hasattr(node, 'metadata') else 0.0
                cov_p10 = node.metadata.get('coverage_p10', cov * 0.8) if hasattr(node, 'metadata') else cov * 0.8
                cov_p90 = node.metadata.get('coverage_p90', cov * 1.2) if hasattr(node, 'metadata') else cov * 1.2
                
                features.append([
                    cov, gc, repeat_frac, kmer_div,
                    branch, hic_density, allele_freq,
                    het, phase_consist, mappability,
                    hic_intra, hic_inter,
                    hic_ratio, hic_phase_sig,
                    cluster_coeff, comp_size,
                    entropy, dinuc_bias,
                    max_run, hp_density, low_complex,
                    cov_skew, cov_kurt, cov_cv,
                    cov_p10, cov_p90,
                ])
            else:
                # PyTorch GNN: simplified 4-feature vector
                phase_A = hic_phase_info[node_id].phase_A_score if hic_phase_info and node_id in hic_phase_info else 0.5
                phase_B = hic_phase_info[node_id].phase_B_score if hic_phase_info and node_id in hic_phase_info else 0.5
                coverage = getattr(node, 'coverage', 0.0) / 100.0
                degree = (in_deg + out_deg) / 10.0 if 'in_deg' in dir() else \
                    (len(graph.in_edges.get(node_id, set())) + len(graph.out_edges.get(node_id, set()))) / 10.0
                features.append([phase_A, phase_B, coverage, degree])
        
        # Run through model — supports both PyTorch (callable) and XGBoost (predict_proba)
        try:
            features_array = np.array(features, dtype=np.float32)
            if hasattr(self.ml_model, 'predict_proba'):
                # XGBoost / sklearn model
                raw_probs = self.ml_model.predict_proba(features_array)
                # Ensure we have exactly 2 columns (haplotype A/B)
                if raw_probs.shape[1] >= 2:
                    probs = raw_probs[:, :2]
                else:
                    probs = np.column_stack([raw_probs[:, 0], 1.0 - raw_probs[:, 0]])
            else:
                # PyTorch model
                features_tensor = torch.tensor(features_array)
                with torch.no_grad():
                    predictions = self.ml_model(features_tensor)
                    probs = torch.softmax(predictions, dim=1).numpy()[:, :2]
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

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
