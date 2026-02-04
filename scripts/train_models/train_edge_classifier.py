#!/usr/bin/env python3
"""
Train EdgeWarden edge classifier (XGBoost)

This trains a lightweight XGBoost model to classify assembly graph edges.
Suitable for training on Colab or MacBook (2-3 hours on small dataset).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_training_data(data_dir):
    """Load edge classification training data."""
    data_dir = Path(data_dir)
    
    features = []
    labels = []
    
    # Load all edge feature files
    feature_files = list(data_dir.glob("**/edge_features_*.npz"))
    
    if not feature_files:
        print(f"Error: No edge feature files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(feature_files)} feature files")
    
    for feature_file in feature_files:
        data = np.load(feature_file)
        features.append(data['features'])
        labels.append(data['labels'])
    
    X = np.vstack(features)
    y = np.hstack(labels)
    
    print(f"Loaded {len(X)} edge examples")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost edge classifier."""
    
    # XGBoost parameters (optimized for edge classification)
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'multi:softmax',
        'num_class': 6,  # TRUE, REPEAT, CHIMERIC, ALLELIC, SV_BREAK, UNKNOWN
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',  # Fast histogram-based method
        'random_state': 42,
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    print("\nTraining XGBoost edge classifier...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Class names
    class_names = ['TRUE', 'REPEAT', 'CHIMERIC', 'ALLELIC', 'SV_BREAK', 'UNKNOWN']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Compute accuracy per class
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"{name}: {acc:.3f} ({mask.sum()} examples)")
    
    # Overall accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    return accuracy


def save_model(model, output_path, metadata):
    """Save trained model and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    model.save_model(str(output_path))
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EdgeWarden edge classifier")
    parser.add_argument('--data-dir', required=True, help='Training data directory')
    parser.add_argument('--output', default='models/edge_classifier_v0.1.model', 
                       help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.15, 
                       help='Test set proportion')
    parser.add_argument('--val-size', type=float, default=0.15, 
                       help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    print("Loading training data...")
    X, y = load_training_data(args.data_dir)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    val_proportion = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_proportion, random_state=args.seed, stratify=y_temp
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    metadata = {
        'version': '0.1',
        'model_type': 'xgboost_edge_classifier',
        'num_features': X.shape[1],
        'num_classes': 6,
        'accuracy': float(accuracy),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'data_dir': str(args.data_dir),
    }
    
    save_model(model, args.output, metadata)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
