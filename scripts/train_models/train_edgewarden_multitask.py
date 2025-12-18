#!/usr/bin/env python3
"""
Training pipeline for EdgeWarden Multi-Task Learning model.

This script trains the multi-task learning model with main task (overlap
classification) and 5 auxiliary tasks (technology, quality, anomaly, homopolymer,
repeat membership).

Usage:
    python3 train_edgewarden_multitask.py --help
    python3 train_edgewarden_multitask.py --training-data <path> --output-models <path>
"""

import argparse
import logging
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Any

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from edgewarden_multitask import (
    MultiTaskLearner, MultiTaskPipeline, AuxiliaryTaskSet,
    prepare_auxiliary_labels
)

logger = logging.getLogger(__name__)


class EdgeWardenMultiTaskTrainer:
    """End-to-end trainer for multi-task EdgeWarden model."""
    
    def __init__(self, output_dir: str):
        """Initialize trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Load training data from file.
        
        Expected format:
        - features.npy: (N, 26) array of features
        - labels.npy: (N,) array of main task labels (0/1 for valid/invalid overlap)
        - metadata.json: List of dicts with auxiliary task information
        """
        data_dir = Path(data_path)
        
        # Load features and labels
        features_file = data_dir / 'features.npy'
        labels_file = data_dir / 'labels.npy'
        metadata_file = data_dir / 'metadata.json'
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        X = np.load(features_file)
        y = np.load(labels_file)
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Prepare auxiliary labels
        auxiliary_labels = prepare_auxiliary_labels(metadata)
        
        logger.info(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Main task label distribution: {np.bincount(y)}")
        
        for task_name, labels in auxiliary_labels.items():
            logger.info(f"{task_name} distribution: {np.bincount(labels)}")
        
        return X, y, auxiliary_labels
    
    def train(self, training_data_path: str, 
              validation_split: float = 0.1,
              feature_dim: int = 26) -> Dict[str, Any]:
        """
        Train the multi-task model.
        
        Args:
            training_data_path: Path to training data directory
            validation_split: Fraction of data for validation
            feature_dim: Dimension of input features
        
        Returns:
            Dictionary with training results
        """
        logger.info("Loading training data...")
        X, y, auxiliary_labels = self.load_training_data(training_data_path)
        
        logger.info("Initializing multi-task pipeline...")
        pipeline = MultiTaskPipeline(feature_dim=feature_dim)
        
        logger.info("Training multi-task model...")
        start_time = datetime.now()
        results = pipeline.learner.train(
            X, y, auxiliary_labels,
            validation_split=validation_split,
            verbose=True
        )
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        results['training_time_seconds'] = elapsed_time
        results['data_size'] = {
            'total_samples': X.shape[0],
            'feature_dimension': X.shape[1],
            'training_samples': int(X.shape[0] * (1 - validation_split)),
            'validation_samples': int(X.shape[0] * validation_split)
        }
        
        logger.info(f"Training completed in {elapsed_time:.1f} seconds")
        
        # Save models
        logger.info(f"Saving models to {self.models_dir}...")
        pipeline.learner.save_models(str(self.models_dir))
        
        # Save results
        results_file = self.results_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert(item) for item in obj]
                return obj
            
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Print summary
        self._print_summary(results, elapsed_time)
        
        return results
    
    def _print_summary(self, results: Dict, elapsed_time: float) -> None:
        """Print training summary."""
        print("\n" + "="*70)
        print("MULTI-TASK LEARNING TRAINING SUMMARY")
        print("="*70)
        
        print(f"\nTraining Time: {elapsed_time:.1f} seconds")
        print(f"Data Size: {results['data_size']['total_samples']} samples")
        
        print("\nMain Task (Overlap Classification):")
        main_metrics = results['main_task']
        print(f"  Accuracy:  {main_metrics['accuracy']:.4f}")
        print(f"  Precision: {main_metrics['precision']:.4f}")
        print(f"  Recall:    {main_metrics['recall']:.4f}")
        print(f"  F1-Score:  {main_metrics['f1']:.4f}")
        
        print("\nAuxiliary Tasks:")
        for task_name, metrics in results['auxiliary_tasks'].items():
            print(f"\n  {task_name}:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\nCombined Weighted Loss: {results['combined_loss']:.4f}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train EdgeWarden multi-task learning model'
    )
    parser.add_argument(
        '--training-data',
        required=True,
        help='Path to training data directory with features.npy, labels.npy, metadata.json'
    )
    parser.add_argument(
        '--output-models',
        required=True,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.1,
        help='Fraction of data to use for validation (default: 0.1)'
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=26,
        help='Feature dimension (default: 26)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting EdgeWarden multi-task training")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output directory: {args.output_models}")
    
    # Train
    trainer = EdgeWardenMultiTaskTrainer(args.output_models)
    results = trainer.train(
        args.training_data,
        validation_split=args.validation_split,
        feature_dim=args.feature_dim
    )
    
    logger.info("Training completed successfully")
    return 0


if __name__ == '__main__':
    exit(main())
