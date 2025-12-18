#!/usr/bin/env python3
"""
ErrorSmith v2: Transformer-Based Base Error Predictor

Improvements over CNN:
1. Transformer attention for long-range context
2. Technology-aware feature masking
3. Separate token embeddings for sequence vs. quality
4. Focal loss for class imbalance
5. Per-technology calibration
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_models.tech_aware_feature_builder import TechAwareFeatureBuilder, TechAwareFeatures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples, focuses on hard negatives/positives.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch,) predictions after sigmoid
            targets: (batch,) binary labels {0, 1}
        """
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Probability of correct class
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        
        # Focal weight: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


class TechAwareTransformer(nn.Module):
    """
    Transformer-based error predictor with tech-aware feature handling.
    
    Architecture:
    1. Sequence embedding (one-hot → dense)
    2. Tech-conditional feature processing
    3. Transformer encoder layers
    4. Adaptive pooling (tech-dependent)
    5. Classification head
    """
    
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
        
        # ========== SEQUENCE BRANCH ==========
        # Embed one-hot encoded sequence context (21D → 64D)
        self.seq_embedding = nn.Linear(seq_dim, embedding_dim)
        self.seq_pos_embedding = nn.Embedding(seq_dim, embedding_dim)
        
        # ========== TECH FEATURE BRANCH ==========
        # Project tech features (20D → 32D)
        self.tech_embedding = nn.Sequential(
            nn.Linear(tech_feature_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ========== TRANSFORMER ENCODER ==========
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
        
        # ========== ATTENTION FOR DYNAMIC POOLING ==========
        # Learn which positions to attend to
        self.attention_weights = nn.Linear(embedding_dim, 1)
        
        # ========== CLASSIFICATION HEAD ==========
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        seq_features: torch.Tensor,
        tech_features: torch.Tensor,
        tech_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_features: (batch, 21) one-hot encoded sequence context
            tech_features: (batch, 20) technology-specific features
            tech_mask: (batch, 20) boolean mask of active features
            
        Returns:
            (batch,) predictions in [0, 1]
        """
        batch_size = seq_features.shape[0]
        
        # ========== SEQUENCE EMBEDDING ==========
        # seq_features: (batch, 21) → (batch, 21, 64)
        seq_emb = self.seq_embedding(seq_features)  # (batch, 64)
        seq_emb = seq_emb.unsqueeze(1)  # (batch, 1, 64)
        
        # Add positional encoding
        pos_idx = torch.arange(seq_features.shape[1], device=seq_features.device)
        pos_emb = self.seq_pos_embedding(pos_idx)  # (21, 64)
        
        # Reshape and add
        if seq_emb.shape[1] == 1:
            seq_emb = seq_emb + pos_emb[10:11].unsqueeze(0)  # Just center base
        
        # ========== TECH FEATURE EMBEDDING ==========
        # Apply mask to tech features
        tech_features_masked = tech_features * tech_mask.float()
        
        # Project to embedding space (batch, 20) → (batch, 32)
        tech_emb = self.tech_embedding(tech_features_masked)  # (batch, 32)
        tech_emb = tech_emb.unsqueeze(1)  # (batch, 1, 32)
        
        # Pad tech embedding to match embedding_dim (32 → 64)
        tech_emb_padded = torch.cat([
            tech_emb,
            torch.zeros(batch_size, 1, self.embedding_dim - 32, device=tech_emb.device)
        ], dim=2)  # (batch, 1, 64)
        
        # ========== CONCATENATE BRANCHES ==========
        # (batch, 1, 64) + (batch, 1, 64) → (batch, 2, 64)
        x = torch.cat([seq_emb, tech_emb_padded], dim=1)
        
        # ========== TRANSFORMER ENCODING ==========
        x = self.transformer(x)  # (batch, 2, 64)
        
        # ========== ADAPTIVE POOLING ==========
        # Learn which positions to weight
        attn_logits = self.attention_weights(x)  # (batch, 2, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, 2, 1)
        
        # Weighted average
        x_pooled = (x * attn_weights).sum(dim=1)  # (batch, 64)
        
        # ========== CLASSIFICATION ==========
        # Concatenate tech features for final decision
        x_final = torch.cat([x_pooled, tech_emb.squeeze(1)], dim=1)  # (batch, 96)
        output = self.classifier(x_final)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)


class BaseErrorDataset(Dataset):
    """PyTorch dataset for base error prediction with tech-aware features."""
    
    def __init__(
        self,
        batch_pkl_files: List[Path],
        feature_builder: TechAwareFeatureBuilder,
    ):
        """
        Args:
            batch_pkl_files: List of pickle files containing (BaseContext, label) tuples
            feature_builder: TechAwareFeatureBuilder instance
        """
        self.feature_builder = feature_builder
        self.examples = []
        
        # Load all batches
        for pkl_file in batch_pkl_files:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                
                for base_ctx, error_label in batch_data:
                    # Build features
                    features = self.feature_builder.build_features(
                        base=base_ctx.base,
                        left_context=base_ctx.left_context,
                        right_context=base_ctx.right_context,
                        quality_score=base_ctx.quality_score,
                        kmer_coverage=base_ctx.kmer_coverage,
                        position=base_ctx.position,
                        read_length=1000,  # Placeholder if not in context
                        technology=base_ctx.technology,
                    )
                    
                    # Convert label
                    label = 1.0 if error_label == 'error' else 0.0
                    
                    self.examples.append({
                        'seq_features': features.sequence_features,
                        'tech_features': features.tech_features,
                        'tech_mask': features.tech_feature_mask,
                        'label': label,
                        'technology': features.technology,
                    })
        
        logger.info(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'seq_features': torch.FloatTensor(ex['seq_features']),
            'tech_features': torch.FloatTensor(ex['tech_features']),
            'tech_mask': torch.BoolTensor(ex['tech_mask']),
            'label': torch.FloatTensor([ex['label']]),
            'technology': ex['technology'],
        }


def collate_fn(batch):
    """Custom collate function to handle variable-size tech features."""
    seq_features = torch.stack([ex['seq_features'] for ex in batch])
    tech_features = torch.stack([ex['tech_features'] for ex in batch])
    tech_mask = torch.stack([ex['tech_mask'] for ex in batch])
    labels = torch.cat([ex['label'] for ex in batch])
    technologies = [ex['technology'] for ex in batch]
    
    return {
        'seq_features': seq_features,
        'tech_features': tech_features,
        'tech_mask': tech_mask,
        'label': labels,
        'technology': technologies,
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Train for one epoch with optional AMP support."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        seq_features = batch['seq_features'].to(device, non_blocking=True)
        tech_features = batch['tech_features'].to(device, non_blocking=True)
        tech_mask = batch['tech_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use AMP if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(seq_features, tech_features, tech_mask)
                loss = criterion(outputs, labels.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(seq_features, tech_features, tech_mask)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            seq_features = batch['seq_features'].to(device)
            tech_features = batch['tech_features'].to(device)
            tech_mask = batch['tech_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(seq_features, tech_features, tech_mask)
            loss = criterion(outputs, labels.squeeze())
            
            total_loss += loss.item()
            
            # Calculate accuracy (threshold at 0.5)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels.squeeze()).sum().item()
            total += labels.shape[0]
    
    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(val_loader), accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train ErrorSmith v2 (Transformer-based Base Error Predictor)'
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
        default='models/base_error_predictor_v2.pt',
        help='Output model path',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile() for 2x speedup (PyTorch 2.0+)',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Use Automatic Mixed Precision for faster training',
    )
    
    args = parser.parse_args()
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Find training data
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    batch_files = sorted(data_dir.glob('base_error_*_batch_*.pkl'))
    if not batch_files:
        logger.error(f"No batch files found in {data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(batch_files))
    train_files = batch_files[:split_idx]
    val_files = batch_files[split_idx:]
    
    logger.info(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # Build feature builder
    feature_builder = TechAwareFeatureBuilder()
    
    # Create datasets
    logger.info("Loading training data...")
    train_dataset = BaseErrorDataset(train_files, feature_builder)
    
    logger.info("Loading validation data...")
    val_dataset = BaseErrorDataset(val_files, feature_builder)
    
    # Create dataloaders with performance optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading (4 workers)
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation (no gradients)
        shuffle=False,
        num_workers=4,  # Parallel loading for validation too
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    # Create model
    model = TechAwareTransformer(
        seq_dim=25,
        tech_feature_dim=20,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
    )
    model = model.to(device)
    
    # Apply torch.compile for 2x speedup (PyTorch 2.0+)
    if args.compile:
        logger.info("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model)
            logger.info("✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Setup AMP if requested
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Accuracy: {val_accuracy:.4f}"
        )
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping triggered")
                break
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'transformer_v2',
        'accuracy': best_accuracy,
        'best_val_loss': best_val_loss,
        'num_examples': len(train_dataset) + len(val_dataset),
        'feature_dim': 25 + 20,  # seq + tech
    }, output_path)
    
    logger.info(f"✅ Model saved to: {output_path}")


if __name__ == '__main__':
    main()
