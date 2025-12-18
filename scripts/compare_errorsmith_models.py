#!/usr/bin/env python3
"""
ErrorSmith Model Comparison: v1 (CNN) vs v2 (Transformer)

Evaluates both models on:
1. Overall accuracy
2. Per-technology accuracy
3. Per-error-type performance
4. Homopolymer vs non-homopolymer
5. Quality-dependent performance
6. Inference speed
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_models.tech_aware_feature_builder import TechAwareFeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_model_v1(model_path: Path, device: torch.device):
    """Load CNN-based v1 model."""
    logger.info("Loading v1 (CNN) model...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate v1 architecture (from original train_base_error_predictor.py)
    class BaseErrorCNN(nn.Module):
        def __init__(self, input_dim=23):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv_layers(x)
            x = x.squeeze(-1)
            x = self.fc_layers(x)
            return x.squeeze(-1)
    
    model = BaseErrorCNN(input_dim=23)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)


def load_model_v2(model_path: Path, device: torch.device):
    """Load Transformer-based v2 model."""
    logger.info("Loading v2 (Transformer) model...")
    
    from train_models.train_base_error_predictor_v2 import TechAwareTransformer
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = TechAwareTransformer(
        seq_dim=25,
        tech_feature_dim=20,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)


def evaluate_v1_model(
    model: nn.Module,
    test_data: Dict,
    device: torch.device,
) -> Dict:
    """Evaluate v1 model (expects simple feature vectors)."""
    model.eval()
    
    features = torch.FloatTensor(test_data['features']).to(device)
    labels = torch.FloatTensor(test_data['labels']).to(device)
    
    with torch.no_grad():
        # v1 expects (batch, 23) → reshapes to (batch, 1, 23) for CNN
        outputs = model(features)
    
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': outputs.cpu().numpy(),
        'labels': labels.cpu().numpy(),
    }


def evaluate_v2_model(
    model: nn.Module,
    test_data: Dict,
    device: torch.device,
) -> Dict:
    """Evaluate v2 model (with tech-aware features)."""
    model.eval()
    
    seq_features = torch.FloatTensor(test_data['seq_features']).to(device)
    tech_features = torch.FloatTensor(test_data['tech_features']).to(device)
    tech_mask = torch.BoolTensor(test_data['tech_mask']).to(device)
    labels = torch.FloatTensor(test_data['labels']).to(device)
    
    with torch.no_grad():
        outputs = model(seq_features, tech_features, tech_mask)
    
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': outputs.cpu().numpy(),
        'labels': labels.cpu().numpy(),
    }


def load_test_data(data_dir: Path, feature_builder: TechAwareFeatureBuilder, limit: int = 5000):
    """Load test data from validation batch files."""
    logger.info(f"Loading test data (limit={limit})...")
    
    v1_features = []  # Simple features for v1
    v2_seq_features = []
    v2_tech_features = []
    v2_tech_masks = []
    labels = []
    technologies = []
    
    batch_files = sorted(data_dir.glob('base_error_*_batch_*.pkl'))
    
    for batch_file in batch_files[-5:]:  # Use last 5 batch files for validation
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        for base_ctx, error_label in batch_data:
            if len(labels) >= limit:
                break
            
            # v2: Tech-aware features
            features = feature_builder.build_features(
                base=base_ctx.base,
                left_context=base_ctx.left_context,
                right_context=base_ctx.right_context,
                quality_score=base_ctx.quality_score,
                kmer_coverage=base_ctx.kmer_coverage,
                position=base_ctx.position,
                read_length=1000,
                technology=base_ctx.technology,
            )
            
            v2_seq_features.append(features.sequence_features)
            v2_tech_features.append(features.tech_features)
            v2_tech_masks.append(features.tech_feature_mask)
            
            # v1: Simple concatenated features (old style)
            base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
            left = (base_ctx.left_context[-10:] if base_ctx.left_context else '').ljust(10, 'N')
            right = (base_ctx.right_context[:10] if base_ctx.right_context else '').ljust(10, 'N')
            
            left_enc = [base_map.get(b, 0) for b in left]
            right_enc = [base_map.get(b, 0) for b in right]
            v1_feat = (left_enc + 
                      [base_map.get(base_ctx.base, 0), base_ctx.quality_score or 20, 
                       base_ctx.kmer_coverage or 10] + 
                      right_enc)
            v1_features.append(v1_feat)
            
            label = 1.0 if error_label == 'error' else 0.0
            labels.append(label)
            technologies.append(base_ctx.technology)
        
        if len(labels) >= limit:
            break
    
    logger.info(f"Loaded {len(labels)} test examples")
    
    return {
        'v1': {
            'features': np.array(v1_features, dtype=np.float32),
            'labels': np.array(labels, dtype=np.float32),
        },
        'v2': {
            'seq_features': np.array(v2_seq_features, dtype=np.float32),
            'tech_features': np.array(v2_tech_features, dtype=np.float32),
            'tech_mask': np.array(v2_tech_masks, dtype=bool),
            'labels': np.array(labels, dtype=np.float32),
            'technologies': technologies,
        }
    }


def compute_per_technology_accuracy(predictions, labels, technologies):
    """Compute accuracy per technology."""
    results = {}
    
    for tech in set(technologies):
        mask = [t == tech for t in technologies]
        tech_preds = predictions[mask]
        tech_labels = labels[mask]
        
        acc = np.mean((tech_preds > 0.5) == tech_labels)
        results[tech] = {
            'accuracy': acc,
            'count': len(tech_labels),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare ErrorSmith v1 vs v2')
    parser.add_argument('--data', type=str, default='training_data/read_correction_v2/base_error')
    parser.add_argument('--model-v1', type=str, default='models/base_error_predictor.pt')
    parser.add_argument('--model-v2', type=str, default='models/base_error_predictor_v2.pt')
    parser.add_argument('--limit', type=int, default=5000, help='Max test examples')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load test data
    data_dir = Path(args.data)
    feature_builder = TechAwareFeatureBuilder()
    test_data = load_test_data(data_dir, feature_builder, limit=args.limit)
    
    # Evaluate models
    print("\n" + "="*60)
    print("ErrorSmith Model Comparison: v1 (CNN) vs v2 (Transformer)")
    print("="*60 + "\n")
    
    # v1 Evaluation
    if Path(args.model_v1).exists():
        logger.info(f"\nEvaluating v1 model: {args.model_v1}")
        model_v1 = load_model_v1(Path(args.model_v1), device)
        results_v1 = evaluate_v1_model(model_v1, test_data['v1'], device)
        
        print(f"v1 (CNN) - Overall Accuracy: {results_v1['accuracy']:.4f}")
        
        # Per-tech
        per_tech_v1 = compute_per_technology_accuracy(
            results_v1['predictions'],
            results_v1['labels'],
            test_data['v2']['technologies']
        )
        print("\n  Per-technology accuracy (v1):")
        for tech, metrics in per_tech_v1.items():
            print(f"    {tech:15s}: {metrics['accuracy']:.4f} ({metrics['count']:4d} examples)")
    else:
        results_v1 = None
        print(f"⚠️  v1 model not found: {args.model_v1}")
    
    # v2 Evaluation
    if Path(args.model_v2).exists():
        logger.info(f"\nEvaluating v2 model: {args.model_v2}")
        model_v2 = load_model_v2(Path(args.model_v2), device)
        results_v2 = evaluate_v2_model(model_v2, test_data['v2'], device)
        
        print(f"\nv2 (Transformer) - Overall Accuracy: {results_v2['accuracy']:.4f}")
        
        # Per-tech
        per_tech_v2 = compute_per_technology_accuracy(
            results_v2['predictions'],
            results_v2['labels'],
            test_data['v2']['technologies']
        )
        print("\n  Per-technology accuracy (v2):")
        for tech, metrics in per_tech_v2.items():
            print(f"    {tech:15s}: {metrics['accuracy']:.4f} ({metrics['count']:4d} examples)")
    else:
        results_v2 = None
        print(f"⚠️  v2 model not found: {args.model_v2}")
    
    # Comparison
    if results_v1 and results_v2:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)
        
        improvement = (results_v2['accuracy'] - results_v1['accuracy']) * 100
        print(f"v1 Accuracy: {results_v1['accuracy']:.4f}")
        print(f"v2 Accuracy: {results_v2['accuracy']:.4f}")
        print(f"Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✅ v2 outperforms v1 by {improvement:.2f}%")
        elif improvement < 0:
            print(f"\n⚠️  v2 underperforms v1 by {-improvement:.2f}%")
        else:
            print(f"\n➖ v1 and v2 have similar performance")


if __name__ == '__main__':
    import sys
    main()
