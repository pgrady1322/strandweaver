#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Chromosome Classifier — three-tier system for identifying microchromosomes
and small chromosomal segments (pre-filter, gene content, advanced features).

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import json

from .gene_annotation import (
    BlastAnnotator,
    AugustusPredictor,
    BUSCOAnalyzer,
    find_orfs,
    BlastHit,
    Gene,
    BUSCOResult
)

logger = logging.getLogger(__name__)


# ============================================================================
#                         DATA STRUCTURES
# ============================================================================

@dataclass
class ChromosomeClassification:
    """Result of chromosome classification."""
    scaffold_id: str
    length: int
    classification: str  # 'HIGH_CONFIDENCE_CHROMOSOME', 'LIKELY_CHROMOSOME', 'POSSIBLE_CHROMOSOME', 'LIKELY_JUNK'
    probability: float  # 0-1
    scores: Dict[str, float]
    reasons: List[str]
    
    # Optional advanced features
    has_telomeres: Optional[bool] = None
    hic_pattern_score: Optional[float] = None
    synteny_chromosome: Optional[str] = None


# ============================================================================
#                         TIER 1: PRE-FILTERING
# ============================================================================

class ChromosomePrefilter:
    """
    Fast pre-filtering to identify candidate chromosomes.
    
    Uses simple metrics (length, coverage, GC, connectivity) to
    quickly eliminate obvious junk scaffolds.
    
    Includes telomere detection — a basic, fast check that strongly
    indicates real chromosomal sequence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pre-filter.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.min_length = config.get('min_length', 50_000)
        self.max_length = config.get('max_length', 20_000_000)
        self.min_coverage_ratio = config.get('min_coverage_ratio', 0.3)
        self.max_coverage_ratio = config.get('max_coverage_ratio', 1.5)
        
        # Telomere detection parameters (configurable via CLI)
        self.telomere_min_units = config.get('telomere_min_units', 10)
        self.telomere_search_depth = config.get('telomere_search_depth', 5000)
        self.telomere_sequence = config.get('telomere_sequence', 'TTAGGG')
        
        self.logger = logging.getLogger(f"{__name__}.ChromosomePrefilter")
    
    def filter_candidates(
        self,
        scaffolds: List[Any],
        graph: Any
    ) -> List[Tuple[Any, float]]:
        """
        Filter scaffolds to identify chromosome candidates.
        
        Args:
            scaffolds: List of scaffold objects
            graph: Assembly graph
        
        Returns:
            List of (scaffold, pre_filter_score) tuples
        """
        # Estimate genome-wide metrics
        genome_coverage = self._estimate_genome_coverage(scaffolds)
        genome_gc = self._estimate_genome_gc(scaffolds)
        
        self.logger.info(f"Genome metrics: coverage={genome_coverage:.1f}x, GC={genome_gc:.2%}")
        
        candidates = []
        
        for scaffold in scaffolds:
            score = 0.0
            
            # 1. Length check
            length = len(scaffold.sequence)
            if self.min_length < length < self.max_length:
                score += 2.0
            else:
                continue  # Skip if outside range
            
            # 2. Coverage check
            coverage = scaffold.metadata.get('coverage', 0)
            expected = genome_coverage
            
            if expected > 0:
                coverage_ratio = coverage / expected
                if self.min_coverage_ratio < coverage_ratio < self.max_coverage_ratio:
                    score += 3.0
            
            # 3. GC content check
            gc = self._calculate_gc(scaffold.sequence)
            if abs(gc - genome_gc) < 0.1:  # Within 10%
                score += 2.0
            
            # 4. Connectivity check (prefer slightly connected)
            connections = self._count_connections(scaffold, graph)
            if 0 < connections < 5:
                score += 2.0
            elif connections == 0:
                score += 1.0  # Totally isolated is OK for microchromosomes
            
            # 5. Telomere detection (strong chromosome indicator)
            has_telomeres = self._detect_telomeres(scaffold.sequence)
            if has_telomeres:
                score += 4.0  # Strong boost — telomeres are definitive
                self.logger.info(f"  {getattr(scaffold, 'id', '?')}: "
                                 f"Telomeric repeats detected (Tier 1)")
            
            # Store telomere result in scaffold metadata for downstream use
            if hasattr(scaffold, 'metadata') and isinstance(scaffold.metadata, dict):
                scaffold.metadata['has_telomeres'] = has_telomeres
            
            # Only keep candidates with reasonable scores
            if score >= 4.0:
                candidates.append((scaffold, score))
        
        self.logger.info(f"Pre-filter: {len(candidates)}/{len(scaffolds)} candidates (score >= 4.0)")
        
        return candidates
    
    def _estimate_genome_coverage(self, scaffolds: List[Any]) -> float:
        """Estimate median genome coverage from scaffolds."""
        coverages = [s.metadata.get('coverage', 0) for s in scaffolds]
        coverages = [c for c in coverages if c > 0]
        return float(np.median(coverages)) if coverages else 0.0
    
    def _estimate_genome_gc(self, scaffolds: List[Any]) -> float:
        """Estimate median genome GC content."""
        gcs = [self._calculate_gc(s.sequence) for s in scaffolds]
        return float(np.median(gcs)) if gcs else 0.4
    
    def _calculate_gc(self, sequence: str) -> float:
        """Calculate GC content."""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total = len(sequence)
        return gc_count / total if total > 0 else 0.0
    
    def _count_connections(self, scaffold: Any, graph: Any) -> int:
        """Count graph connections for this scaffold."""
        # Get node ID from scaffold metadata
        node_id = scaffold.metadata.get('node_id') or scaffold.metadata.get('seed_node')
        if node_id is None:
            return 0
        
        # Count edges involving this node
        connections = 0
        if hasattr(graph, 'edges'):
            for edge in graph.edges:
                if hasattr(edge, 'source') and hasattr(edge, 'target'):
                    if edge.source == node_id or edge.target == node_id:
                        connections += 1
        
        return connections

    def _detect_telomeres(self, sequence: str) -> bool:
        """
        Detect telomeric repeats at scaffold ends.

        Searches for the user-configured telomere motif (and its reverse
        complement) in the first and last *search_depth* bases of the
        scaffold.  Returns True if at least *min_units* tandem copies are
        found at either end.

        Parameters are set via CLI flags:
          --telomere-sequence   (default TTAGGG)
          --telomere-search-depth (default 5000 bp)
          --telomere-min-units  (default 10)
        """
        depth = min(self.telomere_search_depth, len(sequence))
        start_region = sequence[:depth].upper()
        end_region = sequence[-depth:].upper() if len(sequence) > depth else start_region

        motif = self.telomere_sequence.upper()

        # Also check the reverse complement (e.g. CCCTAA for TTAGGG)
        complement = str.maketrans('ACGT', 'TGCA')
        rc_motif = motif.translate(complement)[::-1]

        for m in (motif, rc_motif):
            if start_region.count(m) >= self.telomere_min_units:
                return True
            if end_region.count(m) >= self.telomere_min_units:
                return True

        return False


# ============================================================================
#                         TIER 2: GENE CONTENT ANALYSIS
# ============================================================================

class GeneContentClassifier:
    """
    Gene-based chromosome classification.
    
    Uses BLAST (default), Augustus, or BUSCO to detect coding content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize gene classifier.
        
        Args:
            config: Configuration dict
        """
        self.mode = config.get('mode', 'fast')
        self.gene_detection_method = config.get('gene_detection_method', 'blast')
        self.blast_database = config.get('blast_database', 'nr')
        self.min_gene_density = config.get('min_gene_density', 5)
        self.logger = logging.getLogger(f"{__name__}.GeneContentClassifier")
        
        # Initialize tools
        self.blast = BlastAnnotator(
            blast_type='blastx',
            database=self.blast_database,
            evalue_threshold=1e-5
        ) if self.gene_detection_method == 'blast' else None
        
        self.augustus = AugustusPredictor(
            species=config.get('augustus_species', 'human')
        ) if self.gene_detection_method == 'augustus' else None
        
        self.busco = BUSCOAnalyzer(
            lineage=config.get('busco_lineage', 'auto')
        ) if self.gene_detection_method == 'busco' else None
    
    def analyze_scaffold(self, scaffold: Any, scaffold_id: str) -> Dict[str, Any]:
        """
        Analyze gene content in scaffold.
        
        Args:
            scaffold: Scaffold object
            scaffold_id: Scaffold identifier
        
        Returns:
            Dict with scores and classification
        """
        results = {
            'scaffold_id': scaffold_id,
            'length': len(scaffold.sequence),
            'scores': {},
            'reasons': []
        }
        
        method_failed = False
        
        # Choose analysis method based on mode and config
        if self.mode == 'fast' or self.gene_detection_method == 'blast':
            # Fast mode: BLAST homology search
            results, method_failed = self._analyze_with_blast(scaffold, scaffold_id, results)
        
        elif self.gene_detection_method == 'augustus':
            # Accurate mode: Ab initio gene prediction
            results, method_failed = self._analyze_with_augustus(scaffold, scaffold_id, results)
        
        elif self.gene_detection_method == 'busco':
            # Comprehensive mode: BUSCO completeness
            results, method_failed = self._analyze_with_busco(scaffold, scaffold_id, results)
        
        elif self.gene_detection_method == 'orf':
            # Simple ORF finding
            results, method_failed = self._analyze_with_orfs(scaffold, scaffold_id, results)
        
        else:
            # Unknown method - fallback to ORF
            self.logger.warning(f"Unknown gene detection method: {self.gene_detection_method}, using ORF finder")
            results, method_failed = self._analyze_with_orfs(scaffold, scaffold_id, results)
        
        # Automatic fallback to ORF finder if primary method failed
        if method_failed:
            self.logger.warning(
                f"  Primary method ({self.gene_detection_method}) failed for {scaffold_id}, "
                "falling back to ORF finder"
            )
            results, _ = self._analyze_with_orfs(scaffold, scaffold_id, results)
            results['reasons'].append(f"Fallback: Used ORF finder (primary method unavailable)")
        
        # Calculate overall probability
        results['probability'] = self._calculate_probability(results['scores'])
        results['classification'] = self._classify(results['probability'])
        
        return results
    
    def _analyze_with_blast(self, scaffold: Any, scaffold_id: str, results: Dict) -> Tuple[Dict, bool]:
        """
        Analyze using BLAST.
        
        Returns:
            Tuple of (results dict, method_failed bool)
        """
        self.logger.info(f"  Running BLAST on {scaffold_id}...")
        
        # Check if BLAST is available
        if not self.blast._check_blast_available():
            self.logger.warning(f"  BLAST not available (not found in PATH)")
            results['scores']['blast_hits'] = 0
            results['scores']['gene_density'] = 0
            return results, True  # Method failed
        
        hits = self.blast.search(scaffold.sequence, scaffold_id)
        
        if hits:
            # Calculate metrics
            unique_proteins = len(set(h.subject_id for h in hits))
            avg_identity = sum(h.identity for h in hits) / len(hits)
            avg_evalue = sum(h.evalue for h in hits) / len(hits)
            
            results['scores']['blast_hits'] = len(hits)
            results['scores']['unique_proteins'] = unique_proteins
            results['scores']['avg_identity'] = avg_identity
            results['scores']['avg_evalue'] = avg_evalue
            
            # Gene density estimate (hits / Mb)
            gene_density = len(hits) / (len(scaffold.sequence) / 1_000_000)
            results['scores']['gene_density'] = gene_density
            
            results['reasons'].append(f"BLAST: {len(hits)} hits ({unique_proteins} proteins)")
            results['reasons'].append(f"Gene density: {gene_density:.1f} hits/Mb")
        else:
            results['scores']['blast_hits'] = 0
            results['scores']['gene_density'] = 0
            results['reasons'].append("BLAST: No hits found")
        
        return results, False  # Method succeeded
    
    def _analyze_with_augustus(self, scaffold: Any, scaffold_id: str, results: Dict) -> Tuple[Dict, bool]:
        """
        Analyze using Augustus.
        
        Returns:
            Tuple of (results dict, method_failed bool)
        """
        self.logger.info(f"  Running Augustus on {scaffold_id}...")
        
        # Check if Augustus is available
        if not self.augustus._check_augustus_available():
            self.logger.warning(f"  Augustus not available (not found in PATH)")
            results['scores']['gene_count'] = 0
            return results, True  # Method failed
        
        genes = self.augustus.predict_genes(scaffold.sequence, scaffold_id)
        
        if genes:
            gene_density = len(genes) / (len(scaffold.sequence) / 1_000_000)
            avg_length = sum(g.end - g.start for g in genes) / len(genes)
            
            results['scores']['gene_count'] = len(genes)
            results['scores']['gene_density'] = gene_density
            results['scores']['avg_gene_length'] = avg_length
            
            results['reasons'].append(f"Augustus: {len(genes)} genes")
            results['reasons'].append(f"Gene density: {gene_density:.1f} genes/Mb")
        else:
            results['scores']['gene_count'] = 0
            results['reasons'].append("Augustus: No genes found")
        
        return results, False  # Method succeeded
    
    def _analyze_with_busco(self, scaffold: Any, scaffold_id: str, results: Dict) -> Tuple[Dict, bool]:
        """
        Analyze using BUSCO.
        
        Returns:
            Tuple of (results dict, method_failed bool)
        """
        self.logger.info(f"  Running BUSCO on {scaffold_id}...")
        
        # Check if BUSCO is available
        if not self.busco._check_busco_available():
            self.logger.warning(f"  BUSCO not available (not found in PATH)")
            results['scores']['busco_complete'] = 0
            return results, True  # Method failed
        
        busco_result = self.busco.analyze(scaffold.sequence, scaffold_id)
        
        if busco_result:
            results['scores']['busco_complete'] = busco_result.completeness_percent
            results['scores']['busco_complete_count'] = busco_result.complete
            results['scores']['busco_fragmented'] = busco_result.fragmented
            results['scores']['busco_total'] = busco_result.total
            
            results['reasons'].append(f"BUSCO: {busco_result.completeness_percent:.1f}% complete")
            results['reasons'].append(f"BUSCO: {busco_result.complete}/{busco_result.total} genes")
        else:
            results['scores']['busco_complete'] = 0
            results['reasons'].append("BUSCO: Analysis failed or no results")
        
        return results, False  # Method succeeded
    
    def _analyze_with_orfs(self, scaffold: Any, scaffold_id: str, results: Dict) -> Tuple[Dict, bool]:
        """
        Fallback: Simple ORF finding.
        
        Returns:
            Tuple of (results dict, method_failed bool)
        """
        self.logger.info(f"  Finding ORFs in {scaffold_id}...")
        
        orfs = find_orfs(scaffold.sequence, min_length=300)
        
        if orfs:
            orf_density = len(orfs) / (len(scaffold.sequence) / 1_000_000)
            avg_length = sum(o.end - o.start for o in orfs) / len(orfs)
            
            results['scores']['orf_count'] = len(orfs)
            results['scores']['orf_density'] = orf_density
            results['scores']['avg_orf_length'] = avg_length
            
            results['reasons'].append(f"ORFs: {len(orfs)} found")
            results['reasons'].append(f"ORF density: {orf_density:.1f} ORFs/Mb")
        else:
            results['scores']['orf_count'] = 0
            results['reasons'].append("ORFs: None found")
        
        return results, False  # ORF finder never fails (always succeeds)
    
    def _calculate_probability(self, scores: Dict[str, float]) -> float:
        """
        Calculate chromosome probability from scores.
        
        Returns:
            Probability 0-1
        """
        prob = 0.0
        
        # BLAST-based scoring
        if 'gene_density' in scores:
            density = scores['gene_density']
            # Typical chromosomes: 10-30 genes/Mb
            if density > 5:
                prob += 0.4 * min(density / 20, 1.0)
        
        if 'blast_hits' in scores:
            hits = scores['blast_hits']
            prob += 0.3 * min(hits / 50, 1.0)
        
        # Augustus-based scoring
        if 'gene_count' in scores:
            count = scores['gene_count']
            prob += 0.4 * min(count / 20, 1.0)
        
        # BUSCO-based scoring (most reliable)
        if 'busco_complete' in scores:
            prob = 0.6 * (scores['busco_complete'] / 100)
        
        # ORF-based scoring (least reliable)
        if 'orf_density' in scores and prob == 0:
            density = scores['orf_density']
            prob += 0.3 * min(density / 50, 1.0)
        
        return min(prob, 1.0)
    
    def _classify(self, probability: float) -> str:
        """Classify based on probability."""
        if probability >= 0.7:
            return "HIGH_CONFIDENCE_CHROMOSOME"
        elif probability >= 0.4:
            return "LIKELY_CHROMOSOME"
        elif probability >= 0.2:
            return "POSSIBLE_CHROMOSOME"
        else:
            return "LIKELY_JUNK"


# ============================================================================
#                         TIER 3: ADVANCED FEATURES
# ============================================================================

class AdvancedChromosomeFeatures:
    """
    Advanced chromosome analysis (optional, slower).
    
    - Hi-C contact pattern analysis
    - Synteny with reference genome
    
    Note: Telomere detection was moved to Tier 1 (ChromosomePrefilter)
    because it is a fast, basic check that strongly indicates real
    chromosomal sequence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced features."""
        self.check_hic_patterns = config.get('check_hic_patterns', True)
        self.check_synteny = config.get('check_synteny', False)
        self.reference_genome = config.get('reference_genome', None)
        self.logger = logging.getLogger(f"{__name__}.AdvancedChromosomeFeatures")
    
    def analyze_scaffold(
        self,
        scaffold: Any,
        scaffold_id: str,
        contact_map: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run advanced analyses on scaffold.
        
        Args:
            scaffold: Scaffold object
            scaffold_id: Identifier
            contact_map: Optional Hi-C contact map
        
        Returns:
            Dict with advanced features
        """
        results = {}
        
        # Telomere results are already available from Tier 1 (ChromosomePrefilter)
        # Propagate from scaffold metadata so the classification dataclass is populated
        if hasattr(scaffold, 'metadata') and isinstance(scaffold.metadata, dict):
            if 'has_telomeres' in scaffold.metadata:
                results['has_telomeres'] = scaffold.metadata['has_telomeres']
        
        # Hi-C pattern analysis
        if self.check_hic_patterns and contact_map:
            hic_score = self._analyze_hic_pattern(scaffold, contact_map)
            results['hic_pattern_score'] = hic_score
            if hic_score > 5.0:
                self.logger.info(f"  {scaffold_id}: Strong Hi-C self-contact pattern")
        
        # Synteny analysis
        if self.check_synteny and self.reference_genome:
            synteny = self._analyze_synteny(scaffold, self.reference_genome)
            if synteny:
                results['synteny_chromosome'] = synteny.get('reference_chr')
                self.logger.info(f"  {scaffold_id}: Aligns to {synteny['reference_chr']}")
        
        return results
    
    def _analyze_hic_pattern(self, scaffold: Any, contact_map: Any) -> float:
        """
        Analyze Hi-C contact pattern for this scaffold.
        
        Chromosomes show high self-contacts, low external contacts.
        
        Returns:
            Self/external contact ratio
        """
        node_id = scaffold.metadata.get('node_id') or scaffold.metadata.get('seed_node')
        if node_id is None:
            return 0.0
        
        # Get self-contacts
        self_contacts = contact_map.get_contact(node_id, node_id) if hasattr(contact_map, 'get_contact') else 0
        
        # Get external contacts
        external_contacts = 0
        if hasattr(contact_map, 'contacts'):
            for (n1, n2), count in contact_map.contacts.items():
                if n1 == node_id and n2 != node_id:
                    external_contacts += count
                elif n2 == node_id and n1 != node_id:
                    external_contacts += count
        
        # Calculate ratio
        if external_contacts == 0:
            return 0.0
        
        ratio = self_contacts / external_contacts
        return ratio
    
    def _analyze_synteny(self, scaffold: Any, reference_path: str) -> Optional[Dict]:
        """
        Align scaffold to reference genome to check synteny.
        
        Returns alignment info if matches known chromosome.
        """
        # This would use minimap2 or similar
        # Placeholder for now
        self.logger.warning("Synteny analysis not yet implemented")
        return None


# ============================================================================
#                         MAIN CLASSIFIER
# ============================================================================

class ChromosomeClassifier:
    """
    Main chromosome classification pipeline.
    
    Combines all three tiers:
    1. Pre-filtering (fast)
    2. Gene content analysis (main classifier)
    3. Advanced features (optional)
    """
    
    def __init__(self, config: Dict[str, Any], advanced: bool = False):
        """
        Initialize chromosome classifier.
        
        Args:
            config: Configuration dict
            advanced: Enable advanced features (Tier 3)
        """
        self.config = config
        self.advanced = advanced
        self.logger = logging.getLogger(f"{__name__}.ChromosomeClassifier")
        
        # Initialize tiers
        self.prefilter = ChromosomePrefilter(config)
        self.gene_classifier = GeneContentClassifier(config)
        self.advanced_features = AdvancedChromosomeFeatures(config) if advanced else None
    
    def classify_scaffolds(
        self,
        scaffolds: List[Any],
        graph: Any,
        contact_map: Optional[Any] = None
    ) -> List[ChromosomeClassification]:
        """
        Classify all scaffolds.
        
        Args:
            scaffolds: List of scaffold objects
            graph: Assembly graph
            contact_map: Optional Hi-C contact map
        
        Returns:
            List of ChromosomeClassification results
        """
        self.logger.info(f"Classifying {len(scaffolds)} scaffolds...")
        
        # Tier 1: Pre-filter
        self.logger.info("Tier 1: Pre-filtering candidates...")
        candidates = self.prefilter.filter_candidates(scaffolds, graph)
        
        if not candidates:
            self.logger.warning("No candidates passed pre-filter")
            return []
        
        # Tier 2: Gene content analysis
        self.logger.info(f"Tier 2: Analyzing gene content in {len(candidates)} candidates...")
        results = []
        
        for scaffold, prefilter_score in candidates:
            scaffold_id = scaffold.id if hasattr(scaffold, 'id') else f"scaffold_{id(scaffold)}"
            
            # Analyze genes
            gene_results = self.gene_classifier.analyze_scaffold(scaffold, scaffold_id)
            
            # Tier 3: Advanced features (if enabled)
            advanced_results = {}
            if self.advanced and self.advanced_features:
                self.logger.info(f"Tier 3: Advanced analysis for {scaffold_id}...")
                advanced_results = self.advanced_features.analyze_scaffold(
                    scaffold, scaffold_id, contact_map
                )
            
            # Telomere info is always available from Tier 1 (stored in scaffold metadata)
            has_telomeres = advanced_results.get('has_telomeres')
            if has_telomeres is None and hasattr(scaffold, 'metadata') and isinstance(scaffold.metadata, dict):
                has_telomeres = scaffold.metadata.get('has_telomeres')
            
            # Combine results
            classification = ChromosomeClassification(
                scaffold_id=scaffold_id,
                length=gene_results['length'],
                classification=gene_results['classification'],
                probability=gene_results['probability'],
                scores=gene_results['scores'],
                reasons=gene_results['reasons'],
                has_telomeres=has_telomeres,
                hic_pattern_score=advanced_results.get('hic_pattern_score'),
                synteny_chromosome=advanced_results.get('synteny_chromosome')
            )
            
            results.append(classification)
        
        # Summary
        high_conf = sum(1 for r in results if r.classification == 'HIGH_CONFIDENCE_CHROMOSOME')
        likely = sum(1 for r in results if r.classification == 'LIKELY_CHROMOSOME')
        possible = sum(1 for r in results if r.classification == 'POSSIBLE_CHROMOSOME')
        
        self.logger.info(f"Classification complete:")
        self.logger.info(f"  High confidence: {high_conf}")
        self.logger.info(f"  Likely: {likely}")
        self.logger.info(f"  Possible: {possible}")
        self.logger.info(f"  Junk: {len(results) - high_conf - likely - possible}")
        
        return results
    
    def export_results(
        self,
        results: List[ChromosomeClassification],
        output_path: Path,
        format: str = 'json'
    ):
        """
        Export classification results.
        
        Args:
            results: Classification results
            output_path: Output file path
            format: 'json' or 'csv'
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            self.logger.info(f"Results exported to {output_path}")
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                if not results:
                    return
                
                fieldnames = ['scaffold_id', 'length', 'classification', 'probability']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in results:
                    writer.writerow({
                        'scaffold_id': r.scaffold_id,
                        'length': r.length,
                        'classification': r.classification,
                        'probability': r.probability
                    })
            self.logger.info(f"Results exported to {output_path}")


__all__ = [
    'ChromosomeClassifier',
    'ChromosomePrefilter',
    'GeneContentClassifier',
    'AdvancedChromosomeFeatures',
    'ChromosomeClassification',
]

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
