#!/usr/bin/env python3
"""
Technology-Aware Feature Builder for ErrorSmith

Generates 55-60D feature vectors with tech-specific engineering:
- Features only included if relevant to that technology
- Separate embeddings for each technology signature
- Dynamic feature masking based on technology

Tech-specific patterns:
- ONT: Homopolymer errors (3D bias, 2× deletions vs substitutions)
- HiFi: Rare but systematic errors at read boundaries
- Illumina: Cycle-dependent errors, GC bias
- aDNA: Deamination patterns (C->T/G->A at ends)
- CLR: Polymerase bias, high error rates generally
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechAwareFeatures:
    """Technology-aware feature vector with masking."""
    
    # Core sequence features (always present, 25D)
    sequence_features: np.ndarray  # (25,)
    
    # Tech-specific features (selective, 15-20D)
    tech_features: np.ndarray     # (20,)
    
    # Tech-conditional masks
    tech_feature_mask: np.ndarray # (20,) - boolean array of active features
    
    # Metadata
    technology: str
    position: int
    read_length: int
    
    @property
    def full_vector(self) -> np.ndarray:
        """Return concatenated feature vector (45-60D depending on tech)."""
        return np.concatenate([
            self.sequence_features,
            self.tech_features[self.tech_feature_mask]
        ])
    
    @property
    def vector_dim(self) -> int:
        """Actual dimensionality for this example."""
        return len(self.sequence_features) + np.sum(self.tech_feature_mask)


class TechAwareFeatureBuilder:
    """
    Build feature vectors with technology-aware engineering.
    
    Philosophy: Features should only be included if they're predictive
    for that specific technology. Use masking and conditional inclusion.
    """
    
    # Technology profiles
    TECH_PROFILES = {
        'ont_r9': {
            'name': 'Oxford Nanopore R9',
            'error_rate': 0.12,
            'error_type': 'deletion_bias',  # Deletions > substitutions
            'homopolymer_sensitive': True,
            'position_bias': False,
            'quality_reliable': False,
            'deamination_sensitive': False,
        },
        'ont_r10': {
            'name': 'Oxford Nanopore R10',
            'error_rate': 0.04,
            'error_type': 'deletion_slight',
            'homopolymer_sensitive': True,
            'position_bias': False,
            'quality_reliable': True,
            'deamination_sensitive': False,
        },
        'pacbio_hifi': {
            'name': 'PacBio HiFi',
            'error_rate': 0.001,
            'error_type': 'random',
            'homopolymer_sensitive': False,  # Mostly fixed
            'position_bias': True,            # Errors at ends
            'quality_reliable': True,
            'deamination_sensitive': False,
        },
        'pacbio_clr': {
            'name': 'PacBio CLR',
            'error_rate': 0.10,
            'error_type': 'substitution_bias',
            'homopolymer_sensitive': True,
            'position_bias': True,
            'quality_reliable': False,
            'deamination_sensitive': False,
        },
        'illumina': {
            'name': 'Illumina',
            'error_rate': 0.001,
            'error_type': 'cycle_dependent',
            'homopolymer_sensitive': False,
            'position_bias': False,
            'quality_reliable': True,  # But overoptimistic
            'deamination_sensitive': False,
        },
        'ancient_dna': {
            'name': 'Ancient DNA',
            'error_rate': 0.05,
            'error_type': 'deamination',
            'homopolymer_sensitive': False,
            'position_bias': True,      # Errors at read ends
            'quality_reliable': False,
            'deamination_sensitive': True,
        },
    }
    
    def __init__(self):
        self.tech_embeddings = {
            tech: self._create_tech_embedding(tech)
            for tech in self.TECH_PROFILES.keys()
        }
    
    def _create_tech_embedding(self, technology: str) -> np.ndarray:
        """Create 4D learned embedding for technology."""
        profile = self.TECH_PROFILES[technology]
        # Start with error rate and type characteristics
        embedding = np.array([
            profile['error_rate'],
            1.0 if profile['homopolymer_sensitive'] else 0.0,
            1.0 if profile['position_bias'] else 0.0,
            1.0 if profile['deamination_sensitive'] else 0.0,
        ], dtype=np.float32)
        return embedding
    
    def build_features(
        self,
        base: str,
        left_context: str,
        right_context: str,
        quality_score: Optional[int],
        kmer_coverage: Optional[int],
        position: int,
        read_length: int,
        technology: str,
    ) -> TechAwareFeatures:
        """
        Build feature vector with tech-aware engineering.
        
        Returns 45-60D vector depending on technology relevance.
        """
        
        # Part 1: Core Sequence Features (25D, always present)
        seq_features = self._build_sequence_features(
            base, left_context, right_context, technology
        )
        
        # Part 2: Tech-Specific Features (20D with selective masking)
        tech_features, mask = self._build_tech_features(
            base, left_context, right_context, quality_score,
            kmer_coverage, position, read_length, technology
        )
        
        return TechAwareFeatures(
            sequence_features=seq_features,
            tech_features=tech_features,
            tech_feature_mask=mask,
            technology=technology,
            position=position,
            read_length=read_length,
        )
    
    def _build_sequence_features(
        self,
        base: str,
        left_context: str,
        right_context: str,
        technology: str,
    ) -> np.ndarray:
        """
        Core sequence features (25D, always present, tech-independent).
        
        These are universal predictive features:
        - One-hot encoded sequence context
        - Local composition
        - Sequence complexity
        """
        
        # 1. One-hot encode context (21D)
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Pad to 10 left + 1 center + 10 right = 21
        left = (left_context[-10:] if left_context else '').ljust(10, 'N')
        right = (right_context[:10] if right_context else '').ljust(10, 'N')
        
        context_onehot = np.zeros(21, dtype=np.float32)
        for i, b in enumerate(left):
            if b in base_map:
                context_onehot[i] = base_map[b] + 1  # 1-4 for ACGT, 0 for N
        
        context_onehot[10] = base_map.get(base, 0) + 1  # Center base
        
        for i, b in enumerate(right):
            if b in base_map:
                context_onehot[11 + i] = base_map[b] + 1
        
        # 2. GC content in window (1D)
        window = left_context[-5:] + base + right_context[:5]
        gc_count = window.count('G') + window.count('C')
        gc_content = gc_count / max(len(window), 1)
        
        # 3. Sequence complexity - entropy (1D)
        complexity = self._calculate_entropy(window)
        
        # 4. Dinucleotide pattern (2D)
        left_di = 0.0
        if len(left_context) >= 1:
            di = left_context[-1] + base
            left_di = self._dinucleotide_score(di)
        
        right_di = 0.0
        if len(right_context) >= 1:
            di = base + right_context[0]
            right_di = self._dinucleotide_score(di)
        
        features = np.concatenate([
            context_onehot,
            [gc_content],
            [complexity],
            [left_di],
            [right_di],
        ])
        
        return features.astype(np.float32)  # (25,)
    
    def _build_tech_features(
        self,
        base: str,
        left_context: str,
        right_context: str,
        quality_score: Optional[int],
        kmer_coverage: Optional[int],
        position: int,
        read_length: int,
        technology: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Technology-specific features (20D with selective masking).
        
        Returns:
        - tech_features: (20,) array of feature values
        - mask: (20,) boolean array indicating which are active
        """
        
        profile = self.TECH_PROFILES.get(technology, self.TECH_PROFILES['ont_r9'])
        features = np.zeros(20, dtype=np.float32)
        mask = np.zeros(20, dtype=bool)
        
        idx = 0
        
        # ========== HOMOPOLYMER FEATURES (4D, indices 0-3) ==========
        # Only relevant for: ONT R9/R10, PacBio CLR
        if profile['homopolymer_sensitive']:
            mask[idx:idx+4] = True
            
            # Homopolymer run length (normalized 0-1)
            hp_length, hp_position = self._get_homopolymer_info(
                left_context, base, right_context
            )
            features[idx] = min(hp_length / 20.0, 1.0)  # Normalize by typical max
            idx += 1
            
            # Position within homopolymer run (0-1)
            features[idx] = min(hp_position / max(hp_length, 1), 1.0)
            idx += 1
            
            # Homopolymer type one-hot (2D: is_purine, is_AT)
            is_purine = 1.0 if base in 'AG' else 0.0
            is_at = 1.0 if base in 'AT' else 0.0
            features[idx] = is_purine
            idx += 1
            features[idx] = is_at
            idx += 1
        else:
            idx += 4
        
        # ========== QUALITY CONTEXT FEATURES (7D, indices 4-10) ==========
        # Only relevant for: Illumina, aDNA, PacBio HiFi, ONT R10
        if profile['quality_reliable'] and quality_score is not None:
            mask[idx:idx+7] = True
            
            # Enhanced quality context (4D) - without left/right arrays
            center_qual = min(quality_score / 60.0, 1.0)
            
            # Quality trend (inferred from position for some techs)
            if technology == 'illumina':
                # Illumina quality degrades at read end
                quality_trend = max(0, 30 - position) / 30.0
            else:
                quality_trend = 0.5  # Neutral for others
            
            # Quality variance (estimated based on quality score)
            if quality_score >= 30:
                quality_var = 0.2  # Low variance, high confidence
            elif quality_score >= 20:
                quality_var = 0.5  # Moderate variance
            else:
                quality_var = 0.8  # High variance, low confidence
            
            # Quality reliability score
            if technology in ['illumina', 'ont_r10']:
                quality_reliability = 0.8  # High trust
            elif technology == 'ancient_dna':
                quality_reliability = 0.3  # Low trust
            else:
                quality_reliability = 0.5
            
            features[idx] = center_qual
            idx += 1
            features[idx] = quality_trend
            idx += 1
            features[idx] = quality_var
            idx += 1
            features[idx] = quality_reliability
            idx += 1
            
            # Repeat/STR context (3D)
            context_start = max(0, position - 10)
            context_end = min(read_length, position + 11)
            context_len = context_end - context_start
            
            # Estimate repeat score based on sequence context
            # (would be enhanced with actual sequence in training)
            repeat_score = 0.3  # Default estimate
            
            features[idx] = repeat_score
            idx += 1
            
            # Repeat position bias (in repeat region?)
            is_in_repeat_region = 1.0 if repeat_score > 0.5 else 0.0
            features[idx] = is_in_repeat_region
            idx += 1
            
            # Repeat-quality interaction
            repeat_quality_interaction = repeat_score * (1.0 - quality_reliability)
            features[idx] = repeat_quality_interaction
            idx += 1
        else:
            idx += 7
        
        # ========== COVERAGE FEATURES (3D, indices 9-11) ==========
        # Always relevant where available
        if kmer_coverage is not None:
            mask[idx:idx+3] = True
            
            # Coverage (normalized)
            features[idx] = min(kmer_coverage / 100.0, 1.0)  # Typical max ~100x
            idx += 1
            
            # Coverage reliability
            if kmer_coverage < 3:
                features[idx] = 0.2  # Very uncertain
            elif kmer_coverage < 10:
                features[idx] = 0.5  # Moderately uncertain
            else:
                features[idx] = 0.9  # High confidence
            idx += 1
            
            # Coverage outlier detection
            features[idx] = 0.5  # Placeholder
            idx += 1
        else:
            idx += 3
        
        # ========== POSITION BIAS FEATURES (3D, indices 12-14) ==========
        # Only relevant for: aDNA, PacBio CLR/HiFi
        if profile['position_bias']:
            mask[idx:idx+3] = True
            
            # Distance from read start (normalized)
            dist_from_start = min(position / 100.0, 1.0)  # Errors in first 100bp
            features[idx] = dist_from_start
            idx += 1
            
            # Distance from read end
            dist_from_end = min((read_length - position) / 100.0, 1.0)
            features[idx] = dist_from_end
            idx += 1
            
            # Position risk score (combination)
            if position < 50 or (read_length - position) < 50:
                features[idx] = 0.7  # High risk at ends
            else:
                features[idx] = 0.2
            idx += 1
        else:
            idx += 3
        
        # ========== DEAMINATION FEATURES (3D, indices 15-17) ==========
        # Only relevant for: Ancient DNA
        if profile['deamination_sensitive']:
            mask[idx:idx+3] = True
            
            # C->T likelihood (deamination marker)
            is_c_to_t = 1.0 if base == 'T' and 'C' in left_context[-1:] else 0.0
            features[idx] = is_c_to_t
            idx += 1
            
            # G->A likelihood (complement deamination)
            is_g_to_a = 1.0 if base == 'A' and 'G' in left_context[-1:] else 0.0
            features[idx] = is_g_to_a
            idx += 1
            
            # Terminal deamination score (high at read ends)
            term_score = max(
                1.0 - (position / max(read_length, 1)),
                1.0 - ((read_length - position) / max(read_length, 1))
            )
            features[idx] = term_score
            idx += 1
        else:
            idx += 3
        
        # ========== TECHNOLOGY EMBEDDING (4D, indices 16-19) ==========
        # Always include tech signature
        tech_emb = self.tech_embeddings[technology]
        features[16:20] = tech_emb  # 4D tech embedding directly
        mask[16:20] = True
        
        return features, mask
    
    def _get_homopolymer_info(
        self,
        left_context: str,
        base: str,
        right_context: str,
    ) -> Tuple[int, int]:
        """
        Get homopolymer run length and position within run.
        
        Returns:
        - length: total run length
        - position: position within run (how many same bases before this one)
        """
        
        # Count backwards from current base
        position = 0
        i = len(left_context) - 1
        while i >= 0 and left_context[i] == base:
            position += 1
            i -= 1
        
        # Count forwards
        length = position + 1  # Include center base
        i = 0
        while i < len(right_context) and right_context[i] == base:
            length += 1
            i += 1
        
        return length, position
    
    def _calculate_entropy(self, seq: str) -> float:
        """Calculate sequence complexity (Shannon entropy)."""
        if not seq:
            return 0.0
        
        counts = {}
        for b in seq:
            if b in 'ACGT':
                counts[b] = counts.get(b, 0) + 1
        
        entropy = 0.0
        for count in counts.values():
            p = count / len(seq)
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy / 2.0  # Normalize to 0-1 (max entropy for DNA = 2)
    
    def _dinucleotide_score(self, dinucleotide: str) -> float:
        """Score dinucleotide for error-proneness."""
        # Some dinucleotides are more error-prone
        error_prone = {
            'AA': 0.8,  # Homopolymer
            'TT': 0.8,
            'GG': 0.8,
            'CC': 0.8,
            'CG': 0.6,  # CpG depletion zone
            'GC': 0.6,
        }
        return error_prone.get(dinucleotide, 0.3)
    
    def _detect_repeats(self, seq: str, min_length: int = 2, max_repeat: int = 5) -> float:
        """
        Detect tandem repeats (STRs) in sequence.
        Returns score 0-1 indicating repeat density.
        """
        if len(seq) < min_length * 2:
            return 0.0
        
        repeat_score = 0.0
        positions_with_repeats = set()
        
        # Check for repeating patterns of length 2-5
        for pattern_len in range(min_length, min(max_repeat + 1, len(seq) // 2 + 1)):
            for start in range(len(seq) - pattern_len * 2):
                pattern = seq[start:start + pattern_len]
                
                # Count consecutive repeats
                repeat_count = 1
                pos = start + pattern_len
                while pos + pattern_len <= len(seq) and seq[pos:pos + pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len
                
                # If pattern repeats at least twice, mark it
                if repeat_count >= 2:
                    for i in range(start, min(pos, len(seq))):
                        positions_with_repeats.add(i)
        
        # Return normalized repeat density
        return len(positions_with_repeats) / max(len(seq), 1)


# Test functionality
if __name__ == '__main__':
    builder = TechAwareFeatureBuilder()
    
    # Test example
    features = builder.build_features(
        base='A',
        left_context='ACGTACGTAC',
        right_context='TGCATGCATG',
        quality_score=20,
        kmer_coverage=15,
        position=100,
        read_length=1000,
        technology='ont_r9',
    )
    
    print(f"Feature vector dimension: {features.vector_dim}")
    print(f"Full vector shape: {features.full_vector.shape}")
    print(f"Technology: {features.technology}")
    print(f"Active features: {np.sum(features.tech_feature_mask)}/20")
    print(f"Sample values: {features.full_vector[:10]}")
