#!/usr/bin/env python3
"""
Tech-Specific Heads Fine-Tuning Script

Loads trained ErrorSmith v2 baseline and fine-tunes technology-specific heads.
Each technology gets its own classification head while sharing the Transformer encoder.

Process:
1. Load baseline Transformer encoder from trained v2 model
2. For each technology:
   - Create a technology-specific head MLP
   - Fine-tune on technology-specific training data
   - Validate with technology-specific metrics
   - Save checkpoint
3. Create final MultiHeadErrorPredictor with all heads
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_models.tech_aware_feature_builder import TechAwareFeatureBuilder, TechAwareFeatures
from train_models.tech_specific_heads import (
    MultiHeadErrorPredictor,
    TechSpecificCalibration,
    create_tech_specific_heads,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


class TechAwareTransformer(nn.Module):
    """Baseline Transformer encoder (from v2 training)."""
    
    def __init__(
        self,
        seq_dim: int = 21,
        tech_feature_dim: int = 20,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.tech_feature_dim = tech_feature_dim
        self.embedding_dim = embedding_dim
        
        # Sequence branch
        self.seq_embedding = nn.Linear(seq_dim, embedding_dim)
        self.seq_pos_embedding = nn.Embedding(seq_dim, embedding_dim)
        
        # Tech feature branch
        self.tech_embedding = nn.Sequential(
            nn.Linear(tech_feature_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Attention for pooling
        self.attention_weights = nn.Linear(embedding_dim, 1)
    
    def forward(
        self,
        seq_features: torch.Tensor,
        tech_features: torch.Tensor,
        tech_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Returns:
            x_pooled: (batch, embedding_dim) pooled representation
            tech_emb: (batch, embedding_dim//2) tech embedding for classifier
        """
        batch_size = seq_features.shape[0]
        
        # Sequence embedding
        seq_emb = self.seq_embedding(seq_features)
        seq_emb = seq_emb.unsqueeze(1)
        
        pos_idx = torch.arange(seq_features.shape[1], device=seq_features.device)
        pos_emb = self.seq_pos_embedding(pos_idx)
        
        if seq_emb.shape[1] == 1:
            seq_emb = seq_emb + pos_emb[10:11].unsqueeze(0)
        
        # Tech feature embedding
        tech_features_masked = tech_features * tech_mask.float()
        tech_emb = self.tech_embedding(tech_features_masked)
        tech_emb = tech_emb.unsqueeze(1)
        
        # Pad to embedding_dim
        tech_emb_padded = torch.cat([
            tech_emb,
            torch.zeros(batch_size, 1, self.embedding_dim - 32, device=tech_emb.device)
        ], dim=2)
        
        # Concatenate branches
        x = torch.cat([seq_emb, tech_emb_padded], dim=1)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Adaptive pooling
        attn_logits = self.attention_weights(x)
        attn_weights = torch.softmax(attn_logits, dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)
        
        return x_pooled, tech_emb.squeeze(1)


class TechSpecificDataset(Dataset):
    """Dataset for a specific technology."""
    
    def __init__(
        self,
        batch_pkl_files: List[Path],
        feature_builder: TechAwareFeatureBuilder,
        technology: str,
    ):
        self.feature_builder = feature_builder
        self.technology = technology
        self.examples = []
        
        # Load batches and filter by technology
        for pkl_file in batch_pkl_files:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                
                for base_ctx, error_label in batch_data:
                    # Filter by technology
                    if hasattr(base_ctx, 'technology') and base_ctx.technology != technology:
                        continue
                    
                    # Build features
                    features = self.feature_builder.build_features(
                        base=base_ctx.base,
                        left_context=base_ctx.left_context,
                        right_context=base_ctx.right_context,
                        quality_score=base_ctx.quality_score,
                        kmer_coverage=base_ctx.kmer_coverage,
                        position=base_ctx.position,
                        read_length=1000,
                        technology=base_ctx.technology,
                    )
                    
                    label = 1.0 if error_label == 'error' else 0.0
                    
                    self.examples.append({
                        'seq_features': features.sequence_features,
                        'tech_features': features.tech_features,
                        'tech_mask': features.tech_feature_mask,
                        'label': label,
                    })
        
        logger.info(
            f"Loaded {len(self.examples)} examples for technology: {technology}"
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'seq_features': torch.FloatTensor(ex['seq_features']),
            'tech_features': torch.FloatTensor(ex['tech_features']),
            'tech_mask': torch.BoolTensor(ex['tech_mask']),
            'label': torch.FloatTensor([ex['label']]),
        }


def collate_fn(batch):
    """Custom collate function."""
    seq_features = torch.stack([ex['seq_features'] for ex in batch])
    tech_features = torch.stack([ex['tech_features'] for ex in batch])
    tech_mask = torch.stack([ex['tech_mask'] for ex in batch])
    labels = torch.cat([ex['label'] for ex in batch])
    
    return {
        'seq_features': seq_features,
        'tech_features': tech_features,
        'tech_mask': tech_mask,
        'label': labels,
    }


def load_baseline_encoder(checkpoint_path: Path, device: torch.device) -> TechAwareTransformer:
    """Load trained v2 Transformer encoder."""
    logger.info(f"Loading baseline encoder from: {checkpoint_path}")
    
    model = TechAwareTransformer()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load only encoder weights if full model state
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('seq_embedding') or k.startswith('seq_pos_embedding') or \
           k.startswith('tech_embedding') or k.startswith('transformer') or \
           k.startswith('attention_weights'):
            encoder_state[k] = v
    
    model.load_state_dict(encoder_state, strict=False)
    model = model.to(device)
    return model


def train_tech_head(
    encoder: TechAwareTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    technology: str,
    device: torch.device,
    learning_rate: float = 1e-4,
    epochs: int = 20,
    output_dir: Path = None,
) -> Dict:
    """
    Fine-tune a technology-specific head.
    
    Args:
        encoder: Frozen Transformer encoder
        train_loader: Training data loader
        val_loader: Validation data loader
        technology: Technology name
        device: GPU device
        learning_rate: Learning rate for head
        epochs: Number of epochs
        output_dir: Directory to save checkpoints
        
    Returns:
        Dictionary with training metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning head for: {technology}")
    logger.info(f"{'='*60}")
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Create head (simple MLP)
    embedding_dim = encoder.embedding_dim
    head = nn.Sequential(
        nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(embedding_dim // 2, 1),
        nn.Sigmoid(),
    ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(head.parameters(), lr=learning_rate)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_val_accuracy = 0.0
    best_head_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        head.train()
        train_loss = 0.0
        
        for batch in train_loader:
            seq_features = batch['seq_features'].to(device)
            tech_features = batch['tech_features'].to(device)
            tech_mask = batch['tech_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward through encoder
            with torch.no_grad():
                x_pooled, tech_emb = encoder(seq_features, tech_features, tech_mask)
            
            # Forward through head
            x_final = torch.cat([x_pooled, tech_emb], dim=1)
            predictions = head(x_final).squeeze(-1)
            
            # Loss and backward
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                seq_features = batch['seq_features'].to(device)
                tech_features = batch['tech_features'].to(device)
                tech_mask = batch['tech_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward through encoder
                x_pooled, tech_emb = encoder(seq_features, tech_features, tech_mask)
                
                # Forward through head
                x_final = torch.cat([x_pooled, tech_emb], dim=1)
                predictions = head(x_final).squeeze(-1)
                
                # Loss and metrics
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                preds_binary = (predictions > 0.5).long()
                correct += (preds_binary == labels.long()).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0
        
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Accuracy: {val_accuracy:.4f}"
        )
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_head_state = head.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = output_dir / f"head_{technology}.pt"
                torch.save({
                    'state_dict': best_head_state,
                    'technology': technology,
                    'accuracy': best_val_accuracy,
                }, ckpt_path)
                logger.info(f"✓ Saved checkpoint: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best state
    if best_head_state:
        head.load_state_dict(best_head_state)
    
    logger.info(f"✓ Fine-tuning complete for {technology}")
    logger.info(f"  Best validation accuracy: {best_val_accuracy:.4f}\n")
    
    return {
        'technology': technology,
        'best_accuracy': best_val_accuracy,
        'head_state': best_head_state,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune technology-specific error prediction heads'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='models/error_predictor_v2.pt',
        help='Path to trained baseline v2 model',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='training_data/read_correction_v2/base_error',
        help='Path to training data directory',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/tech_specific_heads',
        help='Output directory for checkpoints',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Epochs per technology',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for head fine-tuning',
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: auto (auto-detect), mps (Apple Silicon), cuda (NVIDIA), or cpu',
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device_str = 'mps'
        elif torch.cuda.is_available():
            device_str = 'cuda'
        else:
            device_str = 'cpu'
    else:
        device_str = args.device
    
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    if device_str == 'mps':
        logger.info("  Backend: Apple Silicon (Metal Performance Shaders)")
    elif device_str == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device_str == 'cpu':
        logger.info(f"  CPUs: {multiprocessing.cpu_count()}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load baseline encoder
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        logger.error(f"Baseline model not found: {baseline_path}")
        sys.exit(1)
    
    encoder = load_baseline_encoder(baseline_path, device)
    
    # Load training data
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    batch_files = sorted(data_dir.glob('base_error_*_batch_*.pkl'))
    if not batch_files:
        logger.error(f"No batch files found in {data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    # Feature builder
    feature_builder = TechAwareFeatureBuilder()
    
    # Technologies to fine-tune
    technologies = ['ont_r9', 'ont_r10', 'hifi', 'illumina', 'adna']
    
    results = {}
    
    # Fine-tune each technology
    for tech in technologies:
        # Create dataset
        dataset = TechSpecificDataset(batch_files, feature_builder, tech)
        
        if len(dataset) < 100:
            logger.warning(f"Skipping {tech}: too few examples ({len(dataset)})")
            continue
        
        # Split into train/val
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        # Fine-tune head
        result = train_tech_head(
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            technology=tech,
            device=device,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            output_dir=output_dir,
        )
        
        results[tech] = result
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINE-TUNING SUMMARY")
    logger.info(f"{'='*60}")
    
    for tech, result in results.items():
        logger.info(f"{tech:15} - Accuracy: {result['best_accuracy']:.4f}")
    
    logger.info(f"\n✅ All heads saved to: {output_dir}")
    logger.info(f"✅ Next: Use these heads with MultiHeadErrorPredictor for inference")
    
    # Save summary
    import json
    summary = {
        'baseline': str(baseline_path),
        'output_dir': str(output_dir),
        'results': {k: {'accuracy': v['best_accuracy']} for k, v in results.items()},
        'hyperparameters': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
        },
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
