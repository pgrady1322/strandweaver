"""
Adaptive k-mer size prediction using machine learning.

This module implements multi-stage k-mer prediction, where different k values
are selected for different assembly operations:
- DBG construction: Balance connectivity vs specificity
- UL overlaps: High specificity for long-range spanning
- Extension: Gap bridging and path resolution
- Polishing: Error correction precision

The predictor uses XGBoost models trained on diverse assembly datasets.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KmerPrediction:
    """
    Multi-stage k-mer size predictions with confidence scores.
    
    Each assembly stage gets its own optimal k-mer size:
    - dbg: de Bruijn graph construction from HiFi/Illumina reads
    - ul_overlap: Ultralong read overlap detection
    - extension: Contig extension and gap bridging
    - polish: Final polishing and error correction
    """
    
    dbg_k: int
    ul_overlap_k: int
    extension_k: int
    polish_k: int
    
    # Confidence scores (0-1) for each prediction
    dbg_confidence: float = 1.0
    ul_overlap_confidence: float = 1.0
    extension_confidence: float = 1.0
    polish_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, int]:
        """Return k values as dictionary for easy access."""
        return {
            'dbg': self.dbg_k,
            'ul_overlap': self.ul_overlap_k,
            'extension': self.extension_k,
            'polish': self.polish_k,
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"K-mer predictions:\n"
            f"  DBG construction: k={self.dbg_k} (confidence: {self.dbg_confidence:.2f})\n"
            f"  UL overlaps: k={self.ul_overlap_k} (confidence: {self.ul_overlap_confidence:.2f})\n"
            f"  Extension: k={self.extension_k} (confidence: {self.extension_confidence:.2f})\n"
            f"  Polish: k={self.polish_k} (confidence: {self.polish_confidence:.2f})"
        )


class AdaptiveKmerPredictor:
    """
    Predicts optimal k-mer sizes for each assembly stage using ML.
    
    Uses separate XGBoost models for each stage, trained on:
    - DBG k: Optimized for graph connectivity and bubble frequency
    - UL overlap k: Optimized for spanning accuracy and specificity
    - Extension k: Optimized for gap closure and mis-join avoidance
    - Polish k: Optimized for base quality improvement
    
    Example:
        from strandweaver.read_correction import FeatureExtractor, AdaptiveKmerPredictor
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_from_file('reads.fastq')
        
        # Predict k values
        predictor = AdaptiveKmerPredictor()
        prediction = predictor.predict(features)
        
        print(prediction)
        # K-mer predictions:
        #   DBG construction: k=31 (confidence: 0.92)
        #   UL overlaps: k=1001 (confidence: 0.87)
        #   Extension: k=55 (confidence: 0.89)
        #   Polish: k=77 (confidence: 0.85)
    """
    
    def __init__(self, model_dir: Optional[Path] = None, use_ml: bool = True):
        """
        Initialize predictor with trained models.
        
        Args:
            model_dir: Directory containing trained XGBoost models.
                      If None, uses default models from package data.
            use_ml: If True, attempt to load ML models. If False or loading fails,
                   falls back to rule-based predictions.
        """
        self.model_dir = model_dir
        self.use_ml = use_ml
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """
        Load trained XGBoost models for each stage.
        
        Looks for models in:
        1. Specified model_dir
        2. Package default location (strandweaver/ai/training/trained_models/)
        3. Falls back to rule-based if not found
        """
        if not self.use_ml:
            logger.info("ML models disabled - using rule-based predictions")
            self.models = None
            return
        
        logger.info("Loading adaptive k-mer models...")
        
        # Determine model directory
        if self.model_dir:
            model_path = Path(self.model_dir)
        else:
            # Default: look in package data
            model_path = Path(__file__).parent / 'training' / 'trained_models'
        
        # Check if models exist
        model_files = {
            'dbg': model_path / 'dbg_model.pkl',
            'ul_overlap': model_path / 'ul_overlap_model.pkl',
            'extension': model_path / 'extension_model.pkl',
            'polish': model_path / 'polish_model.pkl',
        }
        
        # Try to load each model
        loaded_count = 0
        for stage, model_file in model_files.items():
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[stage] = pickle.load(f)
                    logger.info(f"  Loaded {stage} model from {model_file}")
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"  Failed to load {stage} model: {e}")
            else:
                logger.debug(f"  Model not found: {model_file}")
        
        if loaded_count == 0:
            logger.warning("No trained models found - using rule-based defaults")
            logger.info(f"  Looked in: {model_path}")
            logger.info("  To train models, run: python workflow_train_models.py")
            self.models = None
        elif loaded_count < 4:
            logger.warning(f"Only {loaded_count}/4 models loaded - some predictions will use rules")
        else:
            logger.info(f"✓ Successfully loaded all 4 ML models from {model_path}")
    
    def predict(self, features) -> KmerPrediction:
        """
        Predict optimal k-mer sizes for all assembly stages.
        
        Args:
            features: ReadFeatures object from FeatureExtractor
        
        Returns:
            KmerPrediction with k values for each stage
        """
        if self.models is None or len(self.models) == 0:
            # Use rule-based defaults if no models loaded
            logger.debug("Using rule-based k-mer prediction (no ML models)")
            return self._predict_rule_based(features)
        
        # Use trained ML models
        logger.debug("Using ML-based k-mer prediction")
        return self._predict_ml_based(features)
    
    def _predict_rule_based(self, features) -> KmerPrediction:
        """
        Rule-based k-mer selection (fallback until ML models trained).
        
        Rules based on industry best practices:
        - HiFi reads: DBG=31, UL_overlap=1001, ext=55, polish=77
        - ONT reads: DBG=21, UL_overlap=1001, ext=41, polish=55
        - Illumina: DBG=31, UL_overlap=N/A, ext=55, polish=77
        """
        read_type = features.read_type
        mean_len = features.mean_read_length
        error_rate = features.estimated_error_rate
        
        logger.info(f"Using rule-based k-mer selection for {read_type} reads")
        
        if read_type == 'hifi':
            # High-quality long reads - can use larger k
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=1001 if mean_len > 20000 else 501,
                extension_k=55,
                polish_k=77,
                dbg_confidence=0.8,
                ul_overlap_confidence=0.8,
                extension_confidence=0.8,
                polish_confidence=0.8,
            )
        elif read_type == 'ont':
            # Higher error rate - need smaller k for DBG
            return KmerPrediction(
                dbg_k=21 if error_rate > 0.05 else 31,
                ul_overlap_k=1001 if mean_len > 50000 else 501,
                extension_k=41,
                polish_k=55,
                dbg_confidence=0.7,
                ul_overlap_confidence=0.8,
                extension_confidence=0.7,
                polish_confidence=0.7,
            )
        elif read_type == 'illumina':
            # Short reads - moderate k
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=31,  # No ultralong reads
                extension_k=55,
                polish_k=77,
                dbg_confidence=0.8,
                ul_overlap_confidence=0.5,  # Lower confidence - no UL reads
                extension_confidence=0.8,
                polish_confidence=0.8,
            )
        else:
            # Unknown - conservative defaults
            logger.warning(f"Unknown read type, using conservative k values")
            return KmerPrediction(
                dbg_k=31,
                ul_overlap_k=501,
                extension_k=41,
                polish_k=55,
                dbg_confidence=0.5,
                ul_overlap_confidence=0.5,
                extension_confidence=0.5,
                polish_confidence=0.5,
            )
    
    def _predict_ml_based(self, features) -> KmerPrediction:
        """
        ML-based k-mer prediction using trained XGBoost models.
        
        Uses separate trained models for each assembly stage.
        Falls back to rule-based for any missing models.
        """
        # Convert features to numpy array
        X = features.to_feature_vector().reshape(1, -1)
        
        # Get rule-based prediction as fallback
        rule_based = self._predict_rule_based(features)
        
        # Predict k for each stage (use rule-based if model missing)
        k_dbg = int(self.models['dbg'].predict(X)[0]) if 'dbg' in self.models else rule_based.dbg_k
        k_ul = int(self.models['ul_overlap'].predict(X)[0]) if 'ul_overlap' in self.models else rule_based.ul_overlap_k
        k_ext = int(self.models['extension'].predict(X)[0]) if 'extension' in self.models else rule_based.extension_k
        k_pol = int(self.models['polish'].predict(X)[0]) if 'polish' in self.models else rule_based.polish_k
        
        # Confidence scores
        # For now, use high confidence for ML predictions, lower for fallbacks
        dbg_conf = 0.95 if 'dbg' in self.models else rule_based.dbg_confidence
        ul_conf = 0.95 if 'ul_overlap' in self.models else rule_based.ul_overlap_confidence
        ext_conf = 0.95 if 'extension' in self.models else rule_based.extension_confidence
        pol_conf = 0.95 if 'polish' in self.models else rule_based.polish_confidence
        
        logger.info(f"ML predictions: DBG={k_dbg}, UL={k_ul}, ext={k_ext}, pol={k_pol}")
        
        return KmerPrediction(
            dbg_k=k_dbg,
            ul_overlap_k=k_ul,
            extension_k=k_ext,
            polish_k=k_pol,
            dbg_confidence=dbg_conf,
            ul_overlap_confidence=ul_conf,
            extension_confidence=ext_conf,
            polish_confidence=pol_conf,
        )
    
    def predict_single_k(self, features, stage: str) -> int:
        """
        Predict k for a single assembly stage.
        
        Args:
            features: ReadFeatures object
            stage: One of 'dbg', 'ul_overlap', 'extension', 'polish'
        
        Returns:
            Optimal k-mer size for that stage
        """
        prediction = self.predict(features)
        k_dict = prediction.to_dict()
        
        if stage not in k_dict:
            raise ValueError(f"Invalid stage: {stage}. "
                           f"Must be one of {list(k_dict.keys())}")
        
        return k_dict[stage]
    
    def predict_from_file(self, reads_file: Path) -> KmerPrediction:
        """
        Convenience method: extract features and predict k values from reads file.
        
        Args:
            reads_file: Path to FASTQ or FASTA file
        
        Returns:
            KmerPrediction with k values for each stage
        """
        from strandweaver.read_correction.feature_extraction import FeatureExtractor
        
        logger.info(f"Extracting features from {reads_file}")
        extractor = FeatureExtractor()
        features = extractor.extract_from_file(reads_file)
        
        logger.info(f"Predicting k-mer sizes for {features.read_type} reads")
        return self.predict(features)
