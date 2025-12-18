#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assembly Orchestrator for StrandWeaver.

This module implements the central orchestration logic that wires together
the complete assembly pipeline based on input read type:

Pipeline flows:
- Illumina → OLC → DBG → String Graph (if UL) → Hi-C Scaffolding
- HiFi → DBG → String Graph (if UL) → Hi-C Scaffolding  
- ONT → DBG → String Graph (if UL) → Hi-C Scaffolding
- Ancient DNA → (preprocessing) → DBG → String Graph (if UL) → Hi-C Scaffolding

Key principle: String graph ALWAYS follows DBG when UL reads are available.
The ordering is deterministic and explicit for each read_type.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import logging

from strandweaver.io.read import SeqRead
from ..assembly_core.dbg_engine_module import build_dbg_from_long_reads, DBGGraph
from ..assembly_core.string_graph_engine_module import build_string_graph_from_dbg_and_ul, StringGraph, ULAnchor
from ..assembly_core.illumina_olc_contig_module import ContigBuilder

logger = logging.getLogger(__name__)


@dataclass
class AssemblyResult:
    """
    Result of assembly pipeline.
    
    Attributes:
        dbg: De Bruijn graph
        string_graph: String graph with UL overlay
        contigs: Final assembled contigs
        scaffolds: Hi-C scaffolds (if Hi-C data provided)
        stats: Assembly statistics
    """
    dbg: Optional[DBGGraph] = None
    string_graph: Optional[StringGraph] = None
    contigs: List = None
    scaffolds: List = None
    stats: Dict[str, Any] = None


class AssemblyOrchestrator:
    """
    Central orchestration component for the assembly pipeline.
    
    Determines the correct module ordering based on read type and
    coordinates execution across:
    - OLC assembly (for Illumina)
    - DBG construction
    - String graph overlay
    - Hi-C scaffolding
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.AssemblyOrchestrator")
    
    def run_assembly_pipeline(
        self,
        read_type: str,
        corrected_reads: List[SeqRead],
        olc_long_reads: Optional[List[SeqRead]] = None,
        long_reads: Optional[List[SeqRead]] = None,
        ul_reads: Optional[List[SeqRead]] = None,
        ul_anchors: Optional[List[ULAnchor]] = None,
        hic_data: Optional[Any] = None,
        ml_k_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AssemblyResult:
        """
        Orchestrate the full assembly flow based on read_type.
        
        Args:
            read_type: One of "illumina", "hifi", "ont", "ancient", "mixed"
            corrected_reads: Error-corrected reads (from preprocessing)
            olc_long_reads: Optional artificial long reads from OLC (for Illumina)
            long_reads: Optional true long reads (HiFi, ONT)
            ul_reads: Optional ultra-long ONT reads (for string graph overlay)
            ul_anchors: Optional pre-computed UL anchors to DBG nodes
            hic_data: Optional Hi-C data for scaffolding
            ml_k_model: Optional ML model for regional k-mer selection
            config: Optional config override
        
        Returns:
            AssemblyResult containing DBG, string graph, contigs, scaffolds
        
        Pipeline flows:
            Illumina:  OLC → DBG → String Graph → Hi-C
            HiFi:      DBG → String Graph → Hi-C
            ONT:       DBG → String Graph (+ UL overlay) → Hi-C
            Ancient:   DBG → String Graph → Hi-C
            Mixed:     DBG → String Graph → Hi-C
        """
        if config:
            self.config.update(config)
        
        self.logger.info(f"Starting assembly pipeline for read_type={read_type}")
        
        result = AssemblyResult(stats={})
        
        # Determine pipeline flow based on read_type
        if read_type.lower() == "illumina":
            result = self._run_illumina_pipeline(
                corrected_reads, olc_long_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type.lower() == "hifi":
            result = self._run_hifi_pipeline(
                long_reads or corrected_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type.lower() == "ont":
            result = self._run_ont_pipeline(
                long_reads or corrected_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        elif read_type.lower() == "ancient":
            result = self._run_ancient_pipeline(
                corrected_reads, ul_reads, ul_anchors, hic_data, ml_k_model
            )
        
        elif read_type.lower() == "mixed":
            result = self._run_mixed_pipeline(
                corrected_reads, long_reads, ul_reads, ul_anchors,
                hic_data, ml_k_model
            )
        
        else:
            raise ValueError(
                f"Unknown read_type: {read_type}. "
                f"Must be one of: illumina, hifi, ont, ancient, mixed"
            )
        
        self.logger.info("Assembly pipeline complete")
        return result
    
    def _run_illumina_pipeline(
        self,
        corrected_reads: List[SeqRead],
        olc_long_reads: Optional[List[SeqRead]],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Illumina pipeline: OLC → DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Use OLC to generate artificial long reads (if not provided)
        2. Build DBG from artificial long reads
        3. Build string graph IF UL reads available (always follows DBG)
        4. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running Illumina pipeline: OLC → DBG → String Graph → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'illumina'})
        
        # Step 1: OLC assembly to generate artificial long reads
        if olc_long_reads is None:
            self.logger.info("Running OLC to generate artificial long reads")
            olc_long_reads = self._run_olc(corrected_reads)
            result.stats['olc_contigs'] = len(olc_long_reads)
        else:
            self.logger.info(f"Using {len(olc_long_reads)} pre-computed OLC contigs")
        
        # Step 2: Build DBG from artificial long reads
        self.logger.info("Building DBG from OLC-derived long reads")
        base_k = self.config.get('dbg_k', 31)
        min_coverage = self.config.get('min_coverage', 2)
        
        result.dbg = build_dbg_from_long_reads(
            olc_long_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 3: Build string graph
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        else:
            self.logger.info("No UL reads; skipping string graph overlay")
        
        # Step 4: Extract contigs from graph
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 5: Hi-C scaffolding (if available)
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
        return result
    
    def _run_hifi_pipeline(
        self,
        hifi_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        HiFi pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Build DBG directly from HiFi reads (skip OLC)
        2. Build string graph IF UL reads available (always follows DBG)
        3. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running HiFi pipeline: DBG → String Graph → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'hifi'})
        
        # Step 1: Build DBG from HiFi reads
        self.logger.info(f"Building DBG from {len(hifi_reads)} HiFi reads")
        base_k = self.config.get('dbg_k', 51)  # Higher k for HiFi
        min_coverage = self.config.get('min_coverage', 2)
        
        result.dbg = build_dbg_from_long_reads(
            hifi_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 2: Build string graph
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        
        # Step 3: Extract contigs
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 4: Hi-C scaffolding
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
        return result
    
    def _run_ont_pipeline(
        self,
        ont_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        ONT pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Flow:
        1. Build DBG from ONT reads (skip OLC)
        2. Build string graph IF UL reads available (always follows DBG, essential for ONT)
        3. Scaffold with Hi-C (if available)
        """
        self.logger.info("Running ONT pipeline: DBG → String Graph + UL → Hi-C")
        
        result = AssemblyResult(stats={'read_type': 'ont'})
        
        # Step 1: Build DBG from ONT reads
        self.logger.info(f"Building DBG from {len(ont_reads)} ONT reads")
        base_k = self.config.get('dbg_k', 41)  # Medium k for ONT
        min_coverage = self.config.get('min_coverage', 3)  # Higher for noisy ONT
        
        result.dbg = build_dbg_from_long_reads(
            ont_reads,
            base_k=base_k,
            min_coverage=min_coverage,
            ml_k_model=ml_k_model
        )
        result.stats['dbg_nodes'] = len(result.dbg.nodes)
        result.stats['dbg_edges'] = len(result.dbg.edges)
        
        # Step 2: Build string graph (UL overlay is critical for ONT)
        if ul_reads or ul_anchors:
            self.logger.info("Building string graph with UL overlay (essential for ONT)")
            if ul_anchors is None:
                ul_anchors = self._generate_ul_anchors(result.dbg, ul_reads)
            
            result.string_graph = build_string_graph_from_dbg_and_ul(
                result.dbg,
                ul_anchors,
                min_support=self.config.get('min_ul_support', 2)
            )
            result.stats['string_edges'] = len(result.string_graph.edges)
        else:
            self.logger.warning("No UL reads provided for ONT assembly - may have low contiguity")
        
        # Step 3: Extract contigs
        result.contigs = self._extract_contigs_from_graph(
            result.string_graph or result.dbg
        )
        result.stats['num_contigs'] = len(result.contigs)
        
        # Step 4: Hi-C scaffolding
        if hic_data:
            self.logger.info("Running Hi-C scaffolding")
            result.scaffolds = self._run_hic_scaffolding(result.contigs, hic_data)
            result.stats['num_scaffolds'] = len(result.scaffolds)
        
        return result
    
    def _run_ancient_pipeline(
        self,
        corrected_reads: List[SeqRead],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Ancient DNA pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Assumes preprocessing has already merged/assembled ancient DNA
        fragments into longer "pseudo-long-reads".
        String graph built IF UL reads available (always follows DBG).
        """
        self.logger.info("Running Ancient DNA pipeline: DBG → String Graph → Hi-C")
        
        # Ancient DNA uses similar flow to HiFi
        return self._run_hifi_pipeline(
            corrected_reads, ul_reads, ul_anchors, hic_data, ml_k_model
        )
    
    def _run_mixed_pipeline(
        self,
        corrected_reads: List[SeqRead],
        long_reads: Optional[List[SeqRead]],
        ul_reads: Optional[List[SeqRead]],
        ul_anchors: Optional[List[ULAnchor]],
        hic_data: Optional[Any],
        ml_k_model: Optional[Any]
    ) -> AssemblyResult:
        """
        Mixed pipeline: DBG → [String Graph if UL] → Hi-C.
        
        Uses best available long reads (prioritize: HiFi > ONT > corrected).
        String graph built IF UL reads available (always follows DBG).
        """
        self.logger.info("Running Mixed pipeline: DBG → String Graph → Hi-C")
        
        # Use long_reads if available, otherwise corrected_reads
        reads_for_dbg = long_reads if long_reads else corrected_reads
        
        return self._run_hifi_pipeline(
            reads_for_dbg, ul_reads, ul_anchors, hic_data, ml_k_model
        )
    
    def _run_olc(self, corrected_reads: List[SeqRead]) -> List[SeqRead]:
        """
        Run OLC assembly to generate artificial long reads from Illumina.
        
        Args:
            corrected_reads: List of corrected Illumina reads (SeqRead objects)
        
        Returns:
            List of assembled contigs as SeqRead objects (artificial long reads)
        """
        self.logger.info("Running OLC assembly")
        
        # Use existing ContigBuilder for OLC
        builder = ContigBuilder(
            k_size=self.config.get('olc_k', 31),
            min_overlap=self.config.get('olc_min_overlap', 50),
            min_contig_length=self.config.get('olc_min_contig', 500),
            use_gpu=False,
            use_adaptive_k=False
        )
        
        olc_contigs = builder.build_contigs(corrected_reads, verbose=False)
        return olc_contigs
    
    def _generate_ul_anchors(self, dbg: DBGGraph, ul_reads: List[SeqRead]) -> List[ULAnchor]:
        """
        Generate UL anchor points by aligning UL reads to DBG nodes.
        
        Uses ULReadMapper with GraphAligner to create high-quality mappings,
        then converts to simplified ULAnchor format for string graph construction.
        
        Args:
            dbg: De Bruijn graph with nodes to anchor to
            ul_reads: List of ultra-long reads (SeqRead objects)
        
        Returns:
            List of ULAnchor objects mapping UL reads to DBG nodes
        
        Algorithm:
        1. Convert DBG to KmerGraph format (required by ULReadMapper)
        2. Use ULReadMapper to align UL reads to graph (k-mer anchors + GraphAligner)
        3. Extract anchor points from ULReadMapping results
        4. Convert to simplified ULAnchor format
        5. Filter by quality metrics
        """
        from strandweaver.assembly_core.data_structures import ULReadMapper, KmerGraph, KmerNode, KmerEdge
        
        self.logger.info(f"Generating UL anchors for {len(ul_reads)} reads...")
        
        # Convert DBG to KmerGraph format
        self.logger.debug("Converting DBG to KmerGraph format...")
        kmer_graph = self._dbg_to_kmer_graph(dbg)
        
        # Initialize ULReadMapper
        mapper = ULReadMapper(
            min_anchor_length=100,
            min_identity=0.7,
            anchor_k=15,
            min_anchors=3,
            use_mbg=True  # Enable GraphAligner
        )
        
        # Prepare read data as (read_id, sequence) tuples
        ul_read_tuples = [(read.id, read.sequence) for read in ul_reads]
        
        # Map reads to graph (uses batch processing automatically)
        self.logger.info("Mapping UL reads to DBG nodes...")
        mappings = mapper.build_ul_paths(kmer_graph, ul_read_tuples, min_coverage=0.3)
        
        self.logger.info(f"Successfully mapped {len(mappings)}/{len(ul_reads)} UL reads")
        
        # Convert ULReadMapping objects to ULAnchor objects
        anchors = []
        for mapping in mappings:
            # Each node in the path becomes an anchor point
            for i, node_id in enumerate(mapping.path):
                orientation = mapping.orientations[i] if i < len(mapping.orientations) else '+'
                
                # Extract anchor details from mapping.anchors if available
                node_anchors = [a for a in mapping.anchors if a.node_id == node_id]
                
                if node_anchors:
                    # Use actual anchor positions
                    for anchor_obj in node_anchors:
                        anchor = ULAnchor(
                            ul_read_id=mapping.read_id,
                            node_id=str(node_id),
                            read_start=anchor_obj.read_start,
                            read_end=anchor_obj.read_end,
                            node_start=anchor_obj.node_start,
                            strand=anchor_obj.orientation
                        )
                        anchors.append(anchor)
                else:
                    # Create synthetic anchor for this node
                    # (happens when GraphAligner fills gaps between exact anchors)
                    anchor = ULAnchor(
                        ul_read_id=mapping.read_id,
                        node_id=str(node_id),
                        read_start=0,  # Unknown exact position
                        read_end=0,
                        node_start=0,
                        strand=orientation
                    )
                    anchors.append(anchor)
        
        self.logger.info(f"Generated {len(anchors)} UL anchor points from {len(mappings)} read mappings")
        
        return anchors
    
    def _dbg_to_kmer_graph(self, dbg: DBGGraph) -> 'KmerGraph':
        """
        Convert DBGGraph to KmerGraph format for ULReadMapper.
        
        KmerGraph is the format used by assembly_core.py components.
        This conversion enables reuse of ULReadMapper functionality.
        
        Args:
            dbg: DBGGraph object from dbg_engine.py
        
        Returns:
            KmerGraph object compatible with ULReadMapper
        """
        from strandweaver.assembly_core.data_structures import KmerGraph, KmerNode, KmerEdge
        
        kmer_graph = KmerGraph()
        
        # Convert nodes
        for node_id, dbg_node in dbg.nodes.items():
            kmer_node = KmerNode(
                id=node_id,
                seq=dbg_node.seq,
                coverage=dbg_node.coverage,
                length=dbg_node.length
            )
            kmer_graph.nodes[node_id] = kmer_node
        
        # Convert edges
        for edge_id, dbg_edge in dbg.edges.items():
            kmer_edge = KmerEdge(
                id=edge_id,
                from_id=dbg_edge.from_id,
                to_id=dbg_edge.to_id,
                coverage=dbg_edge.coverage
            )
            kmer_graph.edges[edge_id] = kmer_edge
        
        return kmer_graph
    
    def _extract_contigs_from_graph(self, graph: Union[DBGGraph, StringGraph]) -> List[SeqRead]:
        """
        Extract linear contig sequences from DBG or string graph.
        
        Traverses graph to find linear paths and generate consensus sequences.
        
        Args:
            graph: DBGGraph or StringGraph object
        
        Returns:
            List of SeqRead objects representing assembled contigs
        """
        self.logger.info("Extracting contigs from graph")
        
        # Placeholder implementation
        if isinstance(graph, DBGGraph):
            # Convert DBG nodes to contigs
            import math
            contigs = []
            for node_id, node in graph.nodes.items():
                # Each unitig becomes a contig
                from strandweaver.io import SeqRead
                
                # Calculate quality from coverage (log scale)
                # Q = 20 + 10 * log10(coverage + 1), capped at Q40
                avg_qual = min(40, int(20 + 10 * math.log10(node.coverage + 1)))
                quality = chr(avg_qual + 33) * len(node.seq)
                
                contig = SeqRead(
                    id=f"contig_{node_id}",
                    sequence=node.seq,
                    quality=quality,
                    metadata={
                        'source': 'dbg',
                        'node_id': node_id,
                        'coverage': node.coverage,
                        'length': node.length,
                        'recommended_k': node.recommended_k
                    }
                )
                contigs.append(contig)
            return contigs
        
        elif isinstance(graph, StringGraph):
            # Traverse string graph to find paths
            # Placeholder: just return DBG contigs
            return self._extract_contigs_from_graph(graph.dbg)
        
        return []
    
    def _run_hic_scaffolding(self, contigs: List[SeqRead], hic_data: Optional[Any]) -> List[SeqRead]:
        """
        Run Hi-C scaffolding on contigs.
        
        TODO: This is a placeholder implementation.
        
        Args:
            contigs: List of assembled contigs (SeqRead objects)
            hic_data: Hi-C read data for proximity ligation
        
        Returns:
            List of scaffolded contigs (SeqRead objects)
        
        Real implementation would:
        1. Build contig contact matrix from Hi-C alignments
        2. Cluster contigs by contact frequency
        3. Order and orient contigs within scaffolds
        4. Join contigs with N-gaps
        5. Return scaffolded sequences
        """
        self.logger.warning(
            f"Hi-C scaffolding is not yet implemented. "
            f"Returning {len(contigs)} unscaffolded contigs."
        )
        
        # Placeholder: return contigs unchanged
        return contigs


def run_assembly_pipeline(
    read_type: str,
    corrected_reads: List[SeqRead],
    olc_long_reads: Optional[List[SeqRead]] = None,
    long_reads: Optional[List[SeqRead]] = None,
    ul_reads: Optional[List[SeqRead]] = None,
    ul_anchors: Optional[List[ULAnchor]] = None,
    hic_data: Optional[Any] = None,
    ml_k_model: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> AssemblyResult:
    """
    Convenience function to run assembly pipeline.
    
    Args:
        read_type: One of "illumina", "hifi", "ont", "ancient", "mixed"
        corrected_reads: Error-corrected reads
        olc_long_reads: Optional OLC-derived artificial long reads
        long_reads: Optional true long reads
        ul_reads: Optional ultra-long reads
        ul_anchors: Optional UL anchor points
        hic_data: Optional Hi-C data
        ml_k_model: Optional ML k-mer model
        config: Optional configuration
    
    Returns:
        AssemblyResult with DBG, string graph, contigs, scaffolds
    """
    orchestrator = AssemblyOrchestrator(config=config)
    return orchestrator.run_assembly_pipeline(
        read_type=read_type,
        corrected_reads=corrected_reads,
        olc_long_reads=olc_long_reads,
        long_reads=long_reads,
        ul_reads=ul_reads,
        ul_anchors=ul_anchors,
        hic_data=hic_data,
        ml_k_model=ml_k_model,
        config=config
    )
