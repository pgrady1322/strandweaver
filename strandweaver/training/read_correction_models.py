"""
Read correction ML models for StrandWeaver.

This module implements AI-powered read correction models:
1. AdaptiveKmerAI - ML-guided k-mer selection
2. BaseErrorClassifierAI - Per-base error classification

These models enhance the existing heuristic correction methods with learned
approaches that adapt to sequence context and technology-specific error patterns.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import pickle
import logging
import json
from collections import Counter
import re

from strandweaver.training.ml_interfaces import (
    AdaptiveKmerAIModel,
    BaseErrorClassifierAIModel,
    KmerPrediction,
    ReadContext,
    BaseErrorPrediction,
    BaseContext
)

logger = logging.getLogger(__name__)


# ============================================================================
#                    ADAPTIVE KMER AI - XGBOOST IMPLEMENTATION
# ============================================================================

class XGBoostAdaptiveKmerAI(AdaptiveKmerAIModel):
    """
    XGBoost-based adaptive k-mer selection for read correction.
    
    Predicts optimal k-mer size based on sequence context:
    - Homopolymer density and run lengths
    - STR (short tandem repeat) patterns
    - Low-complexity regions
    - Quality score distributions
    - Technology-specific error profiles
    
    Features (32D):
    - Homopolymer stats (6D): mean_length, max_length, density, A/T/G/C ratios
    - STR stats (6D): di/tri/tetra repeat density, max_repeat_length
    - Complexity stats (8D): Shannon entropy, GC content, dinuc diversity, compression ratio
    - Quality stats (6D): mean_qual, min_qual, q25/q50/q75, qual_variance
    - Error context (6D): technology type (one-hot: ONT/PacBio/Illumina), region_length, gc_skew
    
    Output:
    - K-mer size: [15, 17, 19, 21, 25, 31, 41, 51]
    - Confidence score
    - Risk assessment (over/under-correction)
    """
    
    def __init__(self, version: str = "v1.0"):
        super().__init__(model_name="XGBoostAdaptiveKmerAI", version=version)
        self.model = None
        self.scaler = StandardScaler()
        self.supported_k_values = [15, 17, 19, 21, 25, 31, 41, 51]
        self.k_to_class = {k: i for i, k in enumerate(self.supported_k_values)}
        self.class_to_k = {i: k for i, k in enumerate(self.supported_k_values)}
        
        # Technology encoding (expanded for R9/R10 ONT and ancient DNA)
        self.tech_encoding = {
            'ont_r9': 0, 'ont_r10': 1, 'ont': 1,  # Default ONT to R10
            'pacbio': 2, 'pacbio_hifi': 2, 'pacbio_clr': 3,
            'illumina': 4,
            'ancient_dna': 5
        }
        self.num_technologies = 6
    
    def extract_context_features(self, context: ReadContext) -> Dict[str, float]:
        """
        Extract 32D feature vector from read context.
        
        Args:
            context: Read sequence and metadata
        
        Returns:
            Feature dictionary with 32 features
        """
        seq = context.sequence.upper()
        region_len = len(seq)
        
        # 1. Homopolymer features (6D)
        homopoly_features = self._extract_homopolymer_features(seq)
        
        # 2. STR features (6D)
        str_features = self._extract_str_features(seq)
        
        # 3. Complexity features (8D)
        complexity_features = self._extract_complexity_features(seq)
        
        # 4. Quality score features (6D)
        quality_features = self._extract_quality_features(context.quality_scores)
        
        # 5. Error context features (6D)
        error_context = self._extract_error_context(context, region_len)
        
        # Combine all features
        features = {
            **homopoly_features,
            **str_features,
            **complexity_features,
            **quality_features,
            **error_context
        }
        
        return features
    
    def _extract_homopolymer_features(self, seq: str) -> Dict[str, float]:
        """Extract homopolymer-related features."""
        # Find all homopolymer runs
        runs = []
        current_base = seq[0] if seq else ''
        current_len = 1
        
        for base in seq[1:]:
            if base == current_base:
                current_len += 1
            else:
                if current_len >= 2:  # Count as homopolymer if >= 2
                    runs.append((current_base, current_len))
                current_base = base
                current_len = 1
        
        if current_len >= 2:
            runs.append((current_base, current_len))
        
        if not runs:
            return {
                'homopoly_mean_length': 0.0,
                'homopoly_max_length': 0.0,
                'homopoly_density': 0.0,
                'homopoly_A_ratio': 0.0,
                'homopoly_T_ratio': 0.0,
                'homopoly_GC_ratio': 0.0
            }
        
        lengths = [r[1] for r in runs]
        bases = [r[0] for r in runs]
        base_counts = Counter(bases)
        
        return {
            'homopoly_mean_length': np.mean(lengths),
            'homopoly_max_length': max(lengths),
            'homopoly_density': len(runs) / len(seq) if seq else 0.0,
            'homopoly_A_ratio': base_counts.get('A', 0) / len(runs),
            'homopoly_T_ratio': base_counts.get('T', 0) / len(runs),
            'homopoly_GC_ratio': (base_counts.get('G', 0) + base_counts.get('C', 0)) / len(runs)
        }
    
    def _extract_str_features(self, seq: str) -> Dict[str, float]:
        """Extract short tandem repeat (STR) features."""
        # Di-nucleotide repeats
        di_repeats = self._find_tandem_repeats(seq, 2)
        
        # Tri-nucleotide repeats
        tri_repeats = self._find_tandem_repeats(seq, 3)
        
        # Tetra-nucleotide repeats
        tetra_repeats = self._find_tandem_repeats(seq, 4)
        
        return {
            'str_di_density': len(di_repeats) / len(seq) if seq else 0.0,
            'str_tri_density': len(tri_repeats) / len(seq) if seq else 0.0,
            'str_tetra_density': len(tetra_repeats) / len(seq) if seq else 0.0,
            'str_max_di_length': max([r[1] for r in di_repeats], default=0),
            'str_max_tri_length': max([r[1] for r in tri_repeats], default=0),
            'str_max_tetra_length': max([r[1] for r in tetra_repeats], default=0)
        }
    
    def _find_tandem_repeats(self, seq: str, unit_length: int) -> List[Tuple[str, int]]:
        """Find tandem repeats of given unit length."""
        repeats = []
        i = 0
        while i < len(seq) - unit_length:
            unit = seq[i:i+unit_length]
            repeat_count = 1
            j = i + unit_length
            
            while j + unit_length <= len(seq) and seq[j:j+unit_length] == unit:
                repeat_count += 1
                j += unit_length
            
            if repeat_count >= 3:  # Require at least 3 repeats
                repeats.append((unit, repeat_count * unit_length))
                i = j
            else:
                i += 1
        
        return repeats
    
    def _extract_complexity_features(self, seq: str) -> Dict[str, float]:
        """Extract sequence complexity features."""
        if not seq:
            return {
                'complexity_shannon_entropy': 0.0,
                'complexity_gc_content': 0.0,
                'complexity_dinuc_diversity': 0.0,
                'complexity_compression_ratio': 1.0,
                'complexity_at_richness': 0.0,
                'complexity_purine_ratio': 0.0,
                'complexity_kmer_diversity_15': 0.0,
                'complexity_kmer_diversity_21': 0.0
            }
        
        # Shannon entropy
        base_counts = Counter(seq)
        total = len(seq)
        entropy = -sum((count/total) * np.log2(count/total) for count in base_counts.values())
        
        # GC content
        gc_count = seq.count('G') + seq.count('C')
        gc_content = gc_count / total
        
        # Dinucleotide diversity
        dinucs = [seq[i:i+2] for i in range(len(seq)-1)]
        dinuc_diversity = len(set(dinucs)) / 16.0  # Max 16 possible dinucleotides
        
        # Compression ratio (simpler estimate)
        compression_ratio = len(set(seq)) / total
        
        # AT richness
        at_richness = (seq.count('A') + seq.count('T')) / total
        
        # Purine ratio (A, G)
        purine_ratio = (seq.count('A') + seq.count('G')) / total
        
        # K-mer diversity
        kmers_15 = [seq[i:i+15] for i in range(len(seq)-14)]
        kmer_div_15 = len(set(kmers_15)) / len(kmers_15) if kmers_15 else 0.0
        
        kmers_21 = [seq[i:i+21] for i in range(len(seq)-20)]
        kmer_div_21 = len(set(kmers_21)) / len(kmers_21) if kmers_21 else 0.0
        
        return {
            'complexity_shannon_entropy': entropy,
            'complexity_gc_content': gc_content,
            'complexity_dinuc_diversity': dinuc_diversity,
            'complexity_compression_ratio': compression_ratio,
            'complexity_at_richness': at_richness,
            'complexity_purine_ratio': purine_ratio,
            'complexity_kmer_diversity_15': kmer_div_15,
            'complexity_kmer_diversity_21': kmer_div_21
        }
    
    def _extract_quality_features(self, quality_scores: Optional[List[int]]) -> Dict[str, float]:
        """Extract quality score features."""
        if quality_scores is None or not quality_scores:
            return {
                'quality_mean': 30.0,  # Default ONT quality
                'quality_min': 10.0,
                'quality_q25': 25.0,
                'quality_q50': 30.0,
                'quality_q75': 35.0,
                'quality_variance': 5.0
            }
        
        quals = np.array(quality_scores)
        return {
            'quality_mean': float(np.mean(quals)),
            'quality_min': float(np.min(quals)),
            'quality_q25': float(np.percentile(quals, 25)),
            'quality_q50': float(np.percentile(quals, 50)),
            'quality_q75': float(np.percentile(quals, 75)),
            'quality_variance': float(np.var(quals))
        }
    
    def _extract_error_context(self, context: ReadContext, region_len: int) -> Dict[str, float]:
        """Extract error context features."""
        tech_idx = self.tech_encoding.get(context.technology.lower(), 0)
        
        # GC skew
        seq = context.sequence.upper()
        g_count = seq.count('G')
        c_count = seq.count('C')
        gc_skew = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0.0
        
        return {
            'error_tech_ont_r9': 1.0 if tech_idx == 0 else 0.0,
            'error_tech_ont_r10': 1.0 if tech_idx == 1 else 0.0,
            'error_tech_pacbio': 1.0 if tech_idx == 2 else 0.0,
            'error_tech_pacbio_clr': 1.0 if tech_idx == 3 else 0.0,
            'error_tech_illumina': 1.0 if tech_idx == 4 else 0.0,
            'error_tech_ancient': 1.0 if tech_idx == 5 else 0.0,
            'error_region_length': float(region_len),
            'error_gc_skew': gc_skew,
            'error_log_length': np.log10(region_len) if region_len > 0 else 0.0
        }
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array in consistent order."""
        # Expected 32 features in specific order
        feature_order = [
            # Homopolymer (6)
            'homopoly_mean_length', 'homopoly_max_length', 'homopoly_density',
            'homopoly_A_ratio', 'homopoly_T_ratio', 'homopoly_GC_ratio',
            # STR (6)
            'str_di_density', 'str_tri_density', 'str_tetra_density',
            'str_max_di_length', 'str_max_tri_length', 'str_max_tetra_length',
            # Complexity (8)
            'complexity_shannon_entropy', 'complexity_gc_content', 'complexity_dinuc_diversity',
            'complexity_compression_ratio', 'complexity_at_richness', 'complexity_purine_ratio',
            'complexity_kmer_diversity_15', 'complexity_kmer_diversity_21',
            # Quality (6)
            'quality_mean', 'quality_min', 'quality_q25', 'quality_q50', 'quality_q75', 'quality_variance',
            # Error context (9)
            'error_tech_ont_r9', 'error_tech_ont_r10', 'error_tech_pacbio',
            'error_tech_pacbio_clr', 'error_tech_illumina', 'error_tech_ancient',
            'error_region_length', 'error_gc_skew', 'error_log_length'
        ]
        
        # Handle both dict and list inputs
        if isinstance(features, (list, np.ndarray)):
            return np.array(features, dtype=float)
        
        return np.array([features.get(f, 0.0) for f in feature_order], dtype=float)
    
    def predict_optimal_k(self, context: ReadContext) -> KmerPrediction:
        """
        Predict optimal k-mer size for correction.
        
        Args:
            context: Read sequence and metadata
        
        Returns:
            KmerPrediction with optimal k and confidence
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_context_features(context)
        feature_array = self._dict_to_array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_array)
        
        # Predict
        k_class = self.model.predict(scaled_features)[0]
        optimal_k = self.class_to_k[k_class]
        
        # Get probability distribution for confidence
        probs = self.model.predict_proba(scaled_features)[0]
        confidence = float(probs[k_class])
        
        # Assess risk
        risk = self.assess_correction_risk(context, optimal_k)
        
        region_end = context.region_end if context.region_end is not None else len(context.sequence)
        
        return KmerPrediction(
            read_id=context.read_id,
            region_start=context.region_start,
            region_end=region_end,
            optimal_k=optimal_k,
            confidence=confidence,
            context_features=features,
            risk_assessment=risk
        )
    
    def assess_correction_risk(self, context: ReadContext, k: int) -> Dict[str, float]:
        """
        Assess over/under-correction risk.
        
        Heuristic-based risk assessment:
        - Over-correction risk: High for low-complexity, homopolymer-rich regions
        - Under-correction risk: High for high-error-rate, low-quality regions
        
        Args:
            context: Read sequence and metadata
            k: K-mer size to assess
        
        Returns:
            Risk scores
        """
        features = self.extract_context_features(context)
        
        # Over-correction risk factors
        homopoly_risk = features['homopoly_density'] * 0.5
        low_complexity_risk = (1.0 - features['complexity_shannon_entropy'] / 2.0) * 0.3
        str_risk = (features['str_di_density'] + features['str_tri_density']) * 0.2
        
        over_correction_risk = min(1.0, homopoly_risk + low_complexity_risk + str_risk)
        
        # Under-correction risk factors
        low_quality_risk = (40.0 - features['quality_mean']) / 40.0 * 0.6 if features['quality_mean'] < 40 else 0.0
        high_diversity_risk = features['complexity_kmer_diversity_21'] * 0.4
        
        under_correction_risk = min(1.0, low_quality_risk + high_diversity_risk)
        
        # Confidence based on k appropriateness
        # Smaller k for complex regions, larger k for simple regions
        k_appropriateness = 1.0 - abs(k - 21) / 36.0  # Center around k=21
        
        return {
            'over_correction_risk': over_correction_risk,
            'under_correction_risk': under_correction_risk,
            'confidence': k_appropriateness
        }
    
    def predict_k_distribution(self, context: ReadContext) -> Dict[int, float]:
        """Get probability distribution over k-mer sizes."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features = self.extract_context_features(context)
        feature_array = self._dict_to_array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(feature_array)
        
        probs = self.model.predict_proba(scaled_features)[0]
        
        return {self.class_to_k[i]: float(prob) for i, prob in enumerate(probs)}
    
    def train(
        self,
        train_data: List[Tuple[Any, int]],
        val_data: Optional[List[Tuple[Any, int]]] = None,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        random_seed: int = 42,
        verbose: int = 20
    ) -> Dict[str, Any]:
        """
        Train the k-mer selection model.
        
        Args:
            train_data: List of (features, k_value) tuples
            val_data: Optional validation data
            max_depth: Max tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            random_seed: Random seed
            verbose: Logging verbosity
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and labels
        X_train = []
        y_train = []
        
        for features, k_value in train_data:
            if isinstance(features, dict):
                feature_array = self._dict_to_array(features)
            else:
                feature_array = np.array(features, dtype=float)
            
            X_train.append(feature_array)
            
            # Convert k to class
            if k_value in self.k_to_class:
                y_train.append(self.k_to_class[k_value])
            else:
                # Find nearest k
                nearest_k = min(self.supported_k_values, key=lambda k: abs(k - k_value))
                y_train.append(self.k_to_class[nearest_k])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data
        eval_set = []
        if val_data:
            X_val = []
            y_val = []
            for features, k_value in val_data:
                if isinstance(features, dict):
                    feature_array = self._dict_to_array(features)
                else:
                    feature_array = np.array(features, dtype=float)
                X_val.append(feature_array)
                
                if k_value in self.k_to_class:
                    y_val.append(self.k_to_class[k_value])
                else:
                    nearest_k = min(self.supported_k_values, key=lambda k: abs(k - k_value))
                    y_val.append(self.k_to_class[nearest_k])
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train XGBoost classifier
        # Note: num_class is inferred from unique values in y_train
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='multi:softprob',
            random_state=random_seed,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Compute metrics
        train_acc = float(self.model.score(X_train_scaled, y_train))
        
        metrics = {
            'train_accuracy': train_acc,
            'num_train_samples': len(train_data),
            'num_k_classes': len(self.supported_k_values)
        }
        
        if val_data:
            val_acc = float(self.model.score(X_val_scaled, y_val))
            metrics['val_accuracy'] = val_acc
            metrics['num_val_samples'] = len(val_data)
        
        self.training_metadata = metrics
        logger.info(f"Training complete: {metrics}")
        
        return metrics
    
    def predict(self, contexts: List[ReadContext]) -> List[KmerPrediction]:
        """Predict optimal k for multiple contexts."""
        return [self.predict_optimal_k(ctx) for ctx in contexts]
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_file = path / "adaptive_kmer_model.json"
        self.model.save_model(str(model_file))
        
        # Save scaler and metadata
        metadata = {
            'version': self.version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'supported_k_values': self.supported_k_values,
            'k_to_class': self.k_to_class,
            'class_to_k': self.class_to_k,
            'tech_encoding': self.tech_encoding
        }
        
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'metadata': metadata
            }, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        model_file = path / "adaptive_kmer_model.json"
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))
        
        # Load scaler and metadata
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            metadata = data['metadata']
        
        self.version = metadata['version']
        self.is_trained = metadata['is_trained']
        self.training_metadata = metadata['training_metadata']
        self.supported_k_values = metadata['supported_k_values']
        self.k_to_class = metadata['k_to_class']
        self.class_to_k = {int(k): v for k, v in metadata['class_to_k'].items()}
        self.tech_encoding = metadata['tech_encoding']
        
        logger.info(f"Loaded {self.model_name} from {path}")


# ============================================================================
#             BASE ERROR CLASSIFIER AI - XGBOOST IMPLEMENTATION
# ============================================================================

class XGBoostBaseErrorClassifierAI(BaseErrorClassifierAIModel):
    """
    XGBoost-based per-base error classification.
    
    Classifies each base as:
    - 'correct': High confidence correct base
    - 'error': High confidence error (needs correction)
    - 'ambiguous': Uncertain
    
    Features (48D) per base:
    - Quality score (1D)
    - K-mer coverage (1D)
    - Homopolymer context (8D): in_homopoly, position_in_run, run_length, base_type (ATGC)
    - Dinucleotide context (4D): left_dinuc, right_dinuc (one-hot)
    - Window context (16D): GC content, entropy in [-5, +5] window
    - Error signatures (10D): typical ONT/PacBio/Illumina error patterns
    - Position features (8D): distance to read ends, relative position
    
    Benefits:
    - Selective correction (only high-confidence errors)
    - Faster polishing (skip correct regions)
    - Avoid ancient DNA miscorrection
    """
    
    def __init__(self, version: str = "v1.0"):
        super().__init__(model_name="XGBoostBaseErrorClassifierAI", version=version)
        self.model = None
        self.scaler = StandardScaler()
        self.error_classes = ['correct', 'error', 'ambiguous']
        self.class_to_idx = {c: i for i, c in enumerate(self.error_classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.error_classes)}
        
        # Base encoding
        self.base_encoding = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    
    def extract_base_features(self, context: BaseContext) -> Dict[str, float]:
        """
        Extract 48D feature vector for base classification.
        
        Args:
            context: Base and surrounding context
        
        Returns:
            Feature dictionary with 48 features
        """
        # 1. Quality score (1D)
        quality_features = {'quality_score': float(context.quality_score) if context.quality_score is not None else 30.0}
        
        # 2. K-mer coverage (1D)
        coverage_features = {'kmer_coverage': float(context.kmer_coverage) if context.kmer_coverage is not None else 30.0}
        
        # 3. Homopolymer context (8D)
        homopoly_features = self._extract_homopolymer_context(context)
        
        # 4. Dinucleotide context (4D)
        dinuc_features = self._extract_dinucleotide_context(context)
        
        # 5. Window context (16D)
        window_features = self._extract_window_context(context)
        
        # 6. Error signatures (10D)
        error_sig_features = self._extract_error_signatures(context)
        
        # 7. Position features (8D)
        position_features = self._extract_position_features(context)
        
        # Combine all features
        features = {
            **quality_features,
            **coverage_features,
            **homopoly_features,
            **dinuc_features,
            **window_features,
            **error_sig_features,
            **position_features
        }
        
        return features
    
    def _extract_homopolymer_context(self, context: BaseContext) -> Dict[str, float]:
        """Extract homopolymer context features."""
        combined = context.left_context + context.base + context.right_context
        pos_in_combined = len(context.left_context)
        
        # Find homopolymer run containing this position
        in_homopoly = False
        run_length = 1
        position_in_run = 0
        
        # Extend left
        left_ext = 0
        for i in range(pos_in_combined - 1, -1, -1):
            if combined[i] == context.base:
                left_ext += 1
            else:
                break
        
        # Extend right
        right_ext = 0
        for i in range(pos_in_combined + 1, len(combined)):
            if combined[i] == context.base:
                right_ext += 1
            else:
                break
        
        run_length = 1 + left_ext + right_ext
        position_in_run = left_ext
        in_homopoly = run_length >= 3
        
        # Base type encoding
        base_idx = self.base_encoding.get(context.base, 4)
        
        return {
            'homopoly_in_run': 1.0 if in_homopoly else 0.0,
            'homopoly_run_length': float(run_length),
            'homopoly_position': float(position_in_run),
            'homopoly_base_A': 1.0 if base_idx == 0 else 0.0,
            'homopoly_base_T': 1.0 if base_idx == 1 else 0.0,
            'homopoly_base_G': 1.0 if base_idx == 2 else 0.0,
            'homopoly_base_C': 1.0 if base_idx == 3 else 0.0,
            'homopoly_relative_pos': position_in_run / run_length if run_length > 0 else 0.0
        }
    
    def _extract_dinucleotide_context(self, context: BaseContext) -> Dict[str, float]:
        """Extract dinucleotide context features."""
        # Get left and right dinucleotides
        left_dinuc = (context.left_context[-1] + context.base) if context.left_context else "NN"
        right_dinuc = (context.base + context.right_context[0]) if context.right_context else "NN"
        
        # One-hot encode (16 possible dinucleotides)
        dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                  'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        
        return {
            'dinuc_left_idx': float(dinucs.index(left_dinuc)) if left_dinuc in dinucs else 16.0,
            'dinuc_right_idx': float(dinucs.index(right_dinuc)) if right_dinuc in dinucs else 16.0,
            'dinuc_at_rich_left': 1.0 if left_dinuc in ['AA', 'AT', 'TA', 'TT'] else 0.0,
            'dinuc_gc_rich_right': 1.0 if right_dinuc in ['GG', 'GC', 'CG', 'CC'] else 0.0
        }
    
    def _extract_window_context(self, context: BaseContext) -> Dict[str, float]:
        """Extract window context features."""
        # Use [-5, +5] window
        window_seq = context.left_context[-5:] + context.base + context.right_context[:5]
        
        if not window_seq:
            return {f'window_{i}': 0.0 for i in range(16)}
        
        # GC content in window
        gc_count = window_seq.count('G') + window_seq.count('C')
        gc_content = gc_count / len(window_seq)
        
        # Entropy in window
        base_counts = Counter(window_seq)
        total = len(window_seq)
        entropy = -sum((count/total) * np.log2(count/total) for count in base_counts.values() if count > 0)
        
        # Purine/pyrimidine ratio
        purines = window_seq.count('A') + window_seq.count('G')
        pyrimidines = window_seq.count('C') + window_seq.count('T')
        pu_py_ratio = purines / pyrimidines if pyrimidines > 0 else 1.0
        
        # Complexity
        complexity = len(set(window_seq)) / len(window_seq)
        
        return {
            'window_gc_content': gc_content,
            'window_entropy': entropy,
            'window_purine_pyrimidine_ratio': pu_py_ratio,
            'window_complexity': complexity,
            'window_length': float(len(window_seq)),
            'window_a_freq': window_seq.count('A') / len(window_seq),
            'window_t_freq': window_seq.count('T') / len(window_seq),
            'window_g_freq': window_seq.count('G') / len(window_seq),
            'window_c_freq': window_seq.count('C') / len(window_seq),
            'window_n_freq': window_seq.count('N') / len(window_seq),
            'window_gc_skew': (window_seq.count('G') - window_seq.count('C')) / (gc_count + 1),
            'window_at_skew': (window_seq.count('A') - window_seq.count('T')) / (len(window_seq) - gc_count + 1),
            'window_has_n': 1.0 if 'N' in window_seq else 0.0,
            'window_left_gc': (context.left_context[-5:].count('G') + context.left_context[-5:].count('C')) / max(len(context.left_context[-5:]), 1),
            'window_right_gc': (context.right_context[:5].count('G') + context.right_context[:5].count('C')) / max(len(context.right_context[:5]), 1),
            'window_asymmetry': abs(len(context.left_context[-5:]) - len(context.right_context[:5]))
        }
    
    def _extract_error_signatures(self, context: BaseContext) -> Dict[str, float]:
        """Extract technology-specific error signatures."""
        # Expanded technology encoding
        tech_map = {
            'ont_r9': 0, 'ont_r10': 1, 'ont': 1,
            'pacbio': 2, 'pacbio_hifi': 2, 'pacbio_clr': 3,
            'illumina': 4, 'ancient_dna': 5
        }
        tech_idx = tech_map.get(context.technology, 1)  # Default to ONT R10
        
        # ONT-specific: homopolymer indel signature (R9 has much higher error)
        ont_r9_sig = 0.0
        ont_r10_sig = 0.0
        if context.technology in ['ont', 'ont_r9', 'ont_r10']:
            homopoly_features = self._extract_homopolymer_context(context)
            base_sig = homopoly_features['homopoly_in_run'] * (homopoly_features['homopoly_run_length'] / 10.0)
            if context.technology == 'ont_r9':
                ont_r9_sig = base_sig * 2.0  # R9 has ~2x higher homopolymer error
            else:
                ont_r10_sig = base_sig
        
        # PacBio-specific: systematic error signature
        pacbio_hifi_sig = 0.0
        pacbio_clr_sig = 0.0
        if context.technology in ['pacbio', 'pacbio_hifi', 'pacbio_clr']:
            # PacBio has lower homopolymer error but context-dependent mismatches
            window = context.left_context[-3:] + context.base + context.right_context[:3]
            base_sig = len(set(window)) / len(window) if window else 0.0
            if context.technology == 'pacbio_clr':
                pacbio_clr_sig = base_sig * 1.5  # CLR has higher error than HiFi
            else:
                pacbio_hifi_sig = base_sig
        
        # Illumina-specific: quality decay at read ends
        illumina_sig = 0.0
        if context.technology == 'illumina':
            illumina_sig = (context.quality_score or 30) / 40.0
        
        # Ancient DNA-specific: deamination damage signature (C->T, G->A)
        # NOTE: For ASSEMBLY, deamination should be CORRECTED (it's damage, not true genome)
        # This feature helps detect deamination for annotation/metadata purposes
        # But the model learns to classify deamination as 'error' (to be corrected)
        ancient_sig = 0.0
        if context.technology == 'ancient_dna':
            # C->T deamination at 5' end, G->A at 3' end
            dist_to_left = len(context.left_context)
            if context.base in ['C', 'T'] and dist_to_left < 50:
                # Near 5' end: T is likely from C->T deamination (should be corrected to C)
                ancient_sig = 1.0 - (dist_to_left / 50.0)  # Higher near 5' end
            elif context.base in ['G', 'A']:
                # Near 3' end: A is likely from G->A deamination (should be corrected to G)
                # Estimate distance to 3' end (assume ~100bp reads)
                ancient_sig = min(dist_to_left / 50.0, 1.0)  # Higher near 3' end
        
        return {
            'error_sig_tech_ont_r9': 1.0 if tech_idx == 0 else 0.0,
            'error_sig_tech_ont_r10': 1.0 if tech_idx == 1 else 0.0,
            'error_sig_tech_pacbio_hifi': 1.0 if tech_idx == 2 else 0.0,
            'error_sig_tech_pacbio_clr': 1.0 if tech_idx == 3 else 0.0,
            'error_sig_tech_illumina': 1.0 if tech_idx == 4 else 0.0,
            'error_sig_tech_ancient': 1.0 if tech_idx == 5 else 0.0,
            'error_sig_ont_r9_homopoly': ont_r9_sig,
            'error_sig_ont_r10_homopoly': ont_r10_sig,
            'error_sig_pacbio_hifi_context': pacbio_hifi_sig,
            'error_sig_pacbio_clr_context': pacbio_clr_sig,
            'error_sig_illumina_quality': illumina_sig,
            'error_sig_ancient_deamination': ancient_sig,
            'error_sig_low_quality': 1.0 if (context.quality_score or 30) < 20 else 0.0,
            'error_sig_low_coverage': 1.0 if (context.kmer_coverage or 30) < 10 else 0.0,
            'error_sig_high_coverage': 1.0 if (context.kmer_coverage or 30) > 100 else 0.0,
            'error_sig_n_base': 1.0 if context.base == 'N' else 0.0
        }
    
    def _extract_position_features(self, context: BaseContext) -> Dict[str, float]:
        """Extract position-related features."""
        # Technology-specific read length assumptions
        tech_lengths = {
            'ont_r9': 10000, 'ont_r10': 10000, 'ont': 10000,
            'pacbio_hifi': 15000, 'pacbio_clr': 8000, 'pacbio': 15000,
            'illumina': 150,
            'ancient_dna': 60  # Ancient DNA fragments are typically very short
        }
        assumed_read_len = tech_lengths.get(context.technology, 10000)
        
        dist_to_left = len(context.left_context)
        dist_to_right_est = assumed_read_len - dist_to_left
        relative_pos = dist_to_left / assumed_read_len
        
        return {
            'pos_dist_to_left': float(dist_to_left),
            'pos_dist_to_right_est': float(dist_to_right_est),
            'pos_relative': relative_pos,
            'pos_near_left_end': 1.0 if dist_to_left < 50 else 0.0,
            'pos_near_right_end': 1.0 if dist_to_right_est < 50 else 0.0,
            'pos_in_middle': 1.0 if 0.3 < relative_pos < 0.7 else 0.0,
            'pos_log_left': np.log10(dist_to_left + 1),
            'pos_log_right': np.log10(dist_to_right_est + 1)
        }
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array in consistent order."""
        # Expected 48 features in specific order
        feature_order = [
            # Quality & Coverage (2)
            'quality_score', 'kmer_coverage',
            # Homopolymer (8)
            'homopoly_in_run', 'homopoly_run_length', 'homopoly_position',
            'homopoly_base_A', 'homopoly_base_T', 'homopoly_base_G', 'homopoly_base_C',
            'homopoly_relative_pos',
            # Dinucleotide (4)
            'dinuc_left_idx', 'dinuc_right_idx', 'dinuc_at_rich_left', 'dinuc_gc_rich_right',
            # Window (16)
            'window_gc_content', 'window_entropy', 'window_purine_pyrimidine_ratio', 'window_complexity',
            'window_length', 'window_a_freq', 'window_t_freq', 'window_g_freq', 'window_c_freq', 'window_n_freq',
            'window_gc_skew', 'window_at_skew', 'window_has_n', 'window_left_gc', 'window_right_gc', 'window_asymmetry',
            # Error signatures (15 - expanded for all technologies)
            'error_sig_tech_ont_r9', 'error_sig_tech_ont_r10', 'error_sig_tech_pacbio_hifi',
            'error_sig_tech_pacbio_clr', 'error_sig_tech_illumina', 'error_sig_tech_ancient',
            'error_sig_ont_r9_homopoly', 'error_sig_ont_r10_homopoly',
            'error_sig_pacbio_hifi_context', 'error_sig_pacbio_clr_context',
            'error_sig_illumina_quality', 'error_sig_ancient_deamination',
            'error_sig_low_quality', 'error_sig_low_coverage', 'error_sig_high_coverage',
            # Position (8)
            'pos_dist_to_left', 'pos_dist_to_right_est', 'pos_relative',
            'pos_near_left_end', 'pos_near_right_end', 'pos_in_middle', 'pos_log_left', 'pos_log_right'
        ]
        
        # Handle both dict and list inputs
        if isinstance(features, (list, np.ndarray)):
            return np.array(features, dtype=float)
        
        return np.array([features.get(f, 0.0) for f in feature_order], dtype=float)
    
    def classify_base(self, context: BaseContext) -> BaseErrorPrediction:
        """
        Classify a single base.
        
        Args:
            context: Base and surrounding context
        
        Returns:
            BaseErrorPrediction with classification
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_base_features(context)
        feature_array = self._dict_to_array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_array)
        
        # Predict
        class_idx = self.model.predict(scaled_features)[0]
        error_class = self.idx_to_class[class_idx]
        
        # Get probabilities
        probs = self.model.predict_proba(scaled_features)[0]
        confidence = float(probs[class_idx])
        
        error_probs = {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)}
        
        # Suggest correction if error
        suggested = None
        if error_class == 'error':
            # Simple heuristic: suggest most common base in context
            window = context.left_context[-3:] + context.right_context[:3]
            if window:
                base_counts = Counter(window)
                suggested = base_counts.most_common(1)[0][0]
        
        return BaseErrorPrediction(
            read_id=context.read_id,
            position=context.position,
            base=context.base,
            error_class=error_class,
            confidence=confidence,
            error_probabilities=error_probs,
            suggested_correction=suggested
        )
    
    def classify_read(
        self,
        read_id: str,
        sequence: str,
        quality_scores: Optional[List[int]] = None,
        kmer_coverage: Optional[Dict[int, int]] = None
    ) -> List[BaseErrorPrediction]:
        """Classify all bases in a read."""
        predictions = []
        seq_upper = sequence.upper()
        
        for i, base in enumerate(seq_upper):
            # Create context
            left = seq_upper[max(0, i-10):i]
            right = seq_upper[i+1:min(len(seq_upper), i+11)]
            
            context = BaseContext(
                read_id=read_id,
                position=i,
                base=base,
                quality_score=quality_scores[i] if quality_scores else None,
                left_context=left,
                right_context=right,
                kmer_coverage=kmer_coverage.get(i) if kmer_coverage else None
            )
            
            pred = self.classify_base(context)
            predictions.append(pred)
        
        return predictions
    
    def get_error_mask(
        self,
        sequence: str,
        confidence_threshold: float = 0.8
    ) -> List[bool]:
        """Get binary mask of likely errors."""
        preds = self.classify_read("temp", sequence)
        return [
            p.error_class == 'error' and p.confidence >= confidence_threshold
            for p in preds
        ]
    
    def suggest_corrections(
        self,
        read_id: str,
        sequence: str,
        quality_scores: Optional[List[int]] = None,
        technology: str = 'ont_r10'
    ) -> Dict[int, str]:
        """Suggest corrections for likely errors."""
        preds = self.classify_read(read_id, sequence, quality_scores, technology)
        
        corrections = {}
        for pred in preds:
            if pred.error_class == 'error' and pred.confidence >= 0.7 and pred.suggested_correction:
                corrections[pred.position] = pred.suggested_correction
        
        return corrections
    
    def get_deamination_annotations(
        self,
        read_id: str,
        sequence: str,
        quality_scores: Optional[List[int]] = None,
        technology: str = 'ancient_dna'
    ) -> Dict[str, Any]:
        """
        Get deamination damage annotations for ancient DNA.
        
        This provides metadata about deamination patterns for authentication
        and analysis purposes, while still correcting the errors for assembly.
        
        Returns:
            Dictionary with deamination statistics and positions
        """
        if technology != 'ancient_dna':
            return {'deamination_detected': False, 'reason': 'Not ancient DNA'}
        
        preds = self.classify_read(read_id, sequence, quality_scores, technology)
        
        deamination_5p = []  # Likely C->T at 5' end
        deamination_3p = []  # Likely G->A at 3' end
        
        for pred in preds:
            # Extract deamination signature from features
            features = self.extract_base_features(BaseContext(
                read_id=read_id,
                position=pred.position,
                base=sequence[pred.position],
                quality_score=quality_scores[pred.position] if quality_scores else 30,
                left_context=sequence[max(0, pred.position-20):pred.position],
                right_context=sequence[pred.position+1:min(len(sequence), pred.position+21)],
                kmer_coverage=30,
                technology=technology
            ))
            
            ancient_sig = features.get('error_sig_ancient_deamination', 0.0)
            
            # Detect likely deamination
            if ancient_sig > 0.5:
                if pred.position < 10 and sequence[pred.position] == 'T':
                    deamination_5p.append({
                        'position': pred.position,
                        'base': 'T',
                        'likely_original': 'C',
                        'confidence': ancient_sig,
                        'will_be_corrected': pred.error_class == 'error'
                    })
                elif pred.position >= len(sequence) - 10 and sequence[pred.position] == 'A':
                    deamination_3p.append({
                        'position': pred.position,
                        'base': 'A',
                        'likely_original': 'G',
                        'confidence': ancient_sig,
                        'will_be_corrected': pred.error_class == 'error'
                    })
        
        return {
            'deamination_detected': len(deamination_5p) + len(deamination_3p) > 0,
            'fragment_length': len(sequence),
            'deamination_5p_count': len(deamination_5p),
            'deamination_3p_count': len(deamination_3p),
            'deamination_5p_rate': len(deamination_5p) / min(10, len(sequence)) if len(sequence) > 0 else 0,
            'deamination_3p_rate': len(deamination_3p) / min(10, len(sequence)) if len(sequence) > 0 else 0,
            'deamination_5p_positions': deamination_5p,
            'deamination_3p_positions': deamination_3p,
            'note': 'Deamination will be corrected for assembly but tracked for authentication'
        }
    
    def get_correction_priority(
        self,
        predictions: List[BaseErrorPrediction]
    ) -> List[Tuple[int, float]]:
        """Get correction priority order."""
        errors = [
            (pred.position, pred.error_probabilities.get('error', 0.0))
            for pred in predictions
            if pred.error_class == 'error'
        ]
        
        # Sort by error confidence (descending)
        errors.sort(key=lambda x: x[1], reverse=True)
        
        return errors
    
    def train(
        self,
        train_data: List[Tuple[Any, str]],
        val_data: Optional[List[Tuple[Any, str]]] = None,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 150,
        random_seed: int = 42,
        verbose: int = 20
    ) -> Dict[str, Any]:
        """
        Train the base error classifier.
        
        Args:
            train_data: List of (features, error_class) tuples
            val_data: Optional validation data
            max_depth: Max tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            random_seed: Random seed
            verbose: Logging verbosity
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_name} on {len(train_data)} examples")
        
        # Extract features and labels
        X_train = []
        y_train = []
        
        for features, error_class in train_data:
            if isinstance(features, dict):
                feature_array = self._dict_to_array(features)
            else:
                feature_array = np.array(features, dtype=float)
            
            X_train.append(feature_array)
            
            # Normalize label
            label = error_class.lower()
            if label in self.class_to_idx:
                y_train.append(self.class_to_idx[label])
            else:
                # Default to ambiguous
                y_train.append(self.class_to_idx['ambiguous'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data
        eval_set = []
        if val_data:
            X_val = []
            y_val = []
            for features, error_class in val_data:
                if isinstance(features, dict):
                    feature_array = self._dict_to_array(features)
                else:
                    feature_array = np.array(features, dtype=float)
                X_val.append(feature_array)
                
                label = error_class.lower()
                if label in self.class_to_idx:
                    y_val.append(self.class_to_idx[label])
                else:
                    y_val.append(self.class_to_idx['ambiguous'])
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train XGBoost classifier
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='multi:softprob',
            num_class=len(self.error_classes),
            random_state=random_seed,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Compute metrics
        train_acc = float(self.model.score(X_train_scaled, y_train))
        
        metrics = {
            'train_accuracy': train_acc,
            'num_train_samples': len(train_data),
            'num_classes': len(self.error_classes)
        }
        
        if val_data:
            val_acc = float(self.model.score(X_val_scaled, y_val))
            metrics['val_accuracy'] = val_acc
            metrics['num_val_samples'] = len(val_data)
        
        self.training_metadata = metrics
        logger.info(f"Training complete: {metrics}")
        
        return metrics
    
    def predict(self, contexts: List[BaseContext]) -> List[BaseErrorPrediction]:
        """Classify multiple bases."""
        return [self.classify_base(ctx) for ctx in contexts]
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_file = path / "base_error_classifier_model.json"
        self.model.save_model(str(model_file))
        
        # Save scaler and metadata
        metadata = {
            'version': self.version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'error_classes': self.error_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'base_encoding': self.base_encoding
        }
        
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'metadata': metadata
            }, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        # Load XGBoost model
        model_file = path / "base_error_classifier_model.json"
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))
        
        # Load scaler and metadata
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            metadata = data['metadata']
        
        self.version = metadata['version']
        self.is_trained = metadata['is_trained']
        self.training_metadata = metadata['training_metadata']
        self.error_classes = metadata['error_classes']
        self.class_to_idx = metadata['class_to_idx']
        self.idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}
        self.base_encoding = metadata['base_encoding']
        
        logger.info(f"Loaded {self.model_name} from {path}")
