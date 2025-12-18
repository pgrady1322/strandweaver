#!/usr/bin/env python3
"""
Train K-Weaver Model (Adaptive K-mer Selector)

Uses Random Forest or XGBoost to predict optimal k-mer size for error correction
based on read characteristics.

GPU/MPS acceleration used for data preprocessing where applicable.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load k-mer selector training data from PKL files."""
    logger.info(f"Loading training data from {data_dir}")
    
    features_list = []
    labels_list = []
    tech_list = []
    
    # Technologies to load
    technologies = ['ont_r9', 'ont_r10', 'pacbio_hifi', 'pacbio_clr', 'illumina', 'ancient_dna']
    
    for tech in technologies:
        tech_files = sorted(data_dir.glob(f'adaptive_kmer_{tech}_batch_*.pkl'))
        logger.info(f"  {tech}: {len(tech_files)} batches")
        
        for pkl_file in tech_files:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                # batch_data is a list of (ReadContext, optimal_k) tuples
                for read_ctx, optimal_k in batch_data:
                    # Extract features from ReadContext
                    # Features: error_rate, coverage, gc_content, homopolymer_len, read_length, tech encodings
                    features = [
                        getattr(read_ctx, 'error_rate', 0.1),
                        getattr(read_ctx, 'coverage', 30.0),
                        getattr(read_ctx, 'gc_content', 0.5),
                        getattr(read_ctx, 'homopolymer_length', 3.0) if hasattr(read_ctx, 'homopolymer_length') else 3.0,
                        len(read_ctx.sequence),
                        1 if tech == 'ont_r9' else 0,
                        1 if tech == 'ont_r10' else 0,
                        1 if tech == 'pacbio_hifi' else 0,
                        1 if tech == 'pacbio_clr' else 0,
                        1 if tech == 'illumina' else 0,
                        1 if tech == 'ancient_dna' else 0
                    ]
                    features_list.append(features)
                    labels_list.append(optimal_k)
                tech_list.extend([tech] * len(batch_data))
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    logger.info(f"Loaded {len(features)} examples from {len(technologies)} technologies")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Label shape: {labels.shape}")
    
    return features, labels, tech_list


def train_model(features: np.ndarray, labels: np.ndarray, model_type: str = 'xgboost'):
    """Train k-mer selector model."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train)} examples")
    logger.info(f"Test set: {len(X_test)} examples")
    
    if model_type == 'xgboost':
        try:
            import xgboost as xgb
            
            # Use GPU if available
            gpu_params = {}
            try:
                import torch
                if torch.backends.mps.is_available():
                    logger.info("MPS GPU detected, but XGBoost uses CPU (MPS not supported)")
                elif torch.cuda.is_available():
                    logger.info("CUDA GPU detected - enabling XGBoost GPU acceleration")
                    gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            except:
                pass
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=len(np.unique(labels)),
                random_state=42,
                n_jobs=-1,
                **gpu_params
            )
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            model_type = 'random_forest'
    
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    # Train
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = ['error_rate', 'coverage', 'gc_content', 'homopolymer_len', 
                        'read_length', 'tech_ont_r9', 'tech_ont_r10', 'tech_hifi',
                        'tech_clr', 'tech_illumina', 'tech_ancient_dna']
        
        logger.info("\nTop 10 Feature Importances:")
        importance_pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        for name, imp in importance_pairs[:10]:
            logger.info(f"  {name}: {imp:.4f}")
    
    return model, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train K-Weaver Model (Adaptive K-mer Selector)')
    parser.add_argument('--data', type=str, default='training_data/read_correction/adaptive_kmer',
                       help='Path to training data directory')
    parser.add_argument('--output', type=str, default='models/kmer_selector.pkl',
                       help='Output model path')
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest'],
                       help='Model type to train')
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    features, labels, tech_list = load_training_data(data_dir)
    
    # Train model
    model, accuracy = train_model(features, labels, args.model_type)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'model_type': args.model_type,
            'accuracy': accuracy,
            'num_examples': len(features),
            'num_technologies': 6
        }, f)
    
    logger.info("✅ K-mer selector model training complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Test accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
