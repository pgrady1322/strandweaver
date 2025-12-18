#!/usr/bin/env python3
"""
Knowledge Distillation Training Script

Trains a compact student model by distilling knowledge from the trained ErrorSmith v2.

Features:
1. Teacher: Trained ErrorSmith v2 Transformer (156.6K params)
2. Student options: MLP (5K params) or CNN (500 params)
3. Knowledge distillation with soft targets
4. Temperature-based probability softening
5. Early stopping and checkpoint management

Expected results:
- MLP: 6-10x speedup, 95-99% accuracy retention
- CNN: 8-12x speedup, 90-95% accuracy retention
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
import time

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_models.tech_aware_feature_builder import TechAwareFeatureBuilder, TechAwareFeatures
from train_models.knowledge_distillation import (
    DistillationTrainer,
    KnowledgeDistillationLoss,
    create_student_model,
    StudentNetworkMLP,
    StudentNetworkCNN,
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
    """Teacher Transformer (from v2 training)."""
    
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )
    
    def forward(
        self,
        seq_features: torch.Tensor,
        tech_features: torch.Tensor,
        tech_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            (batch,) logits (before sigmoid)
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
        
        # Classification
        x_final = torch.cat([x_pooled, tech_emb.squeeze(1)], dim=1)
        output = self.classifier(x_final)
        
        return output.squeeze(-1)


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    
    def __init__(
        self,
        batch_pkl_files: List[Path],
        feature_builder: TechAwareFeatureBuilder,
    ):
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
                        read_length=1000,
                        technology=base_ctx.technology,
                    )
                    
                    # Hard target
                    label = 1.0 if error_label == 'error' else 0.0
                    
                    self.examples.append({
                        'seq_features': features.sequence_features,
                        'tech_features': features.tech_features,
                        'tech_mask': features.tech_feature_mask,
                        'label': label,
                    })
        
        logger.info(f"Loaded {len(self.examples)} distillation examples")
    
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


def load_teacher(checkpoint_path: Path, device: torch.device) -> TechAwareTransformer:
    """Load trained teacher model."""
    logger.info(f"Loading teacher from: {checkpoint_path}")
    
    teacher = TechAwareTransformer()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    teacher.load_state_dict(state_dict, strict=False)
    teacher = teacher.to(device)
    return teacher


def generate_teacher_predictions(
    teacher: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Generate teacher predictions for all training data."""
    logger.info("Generating teacher predictions for knowledge distillation...")
    
    teacher.eval()
    teacher_preds = []
    labels = []
    
    with torch.no_grad():
        for batch in train_loader:
            seq_features = batch['seq_features'].to(device)
            tech_features = batch['tech_features'].to(device)
            tech_mask = batch['tech_mask'].to(device)
            batch_labels = batch['label'].to(device)
            
            # Teacher forward
            logits = teacher(seq_features, tech_features, tech_mask)
            probs = torch.sigmoid(logits)
            
            teacher_preds.append(probs.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    teacher_preds = np.concatenate(teacher_preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    logger.info(f"Generated {len(teacher_preds)} teacher predictions")
    
    return {
        'teacher_preds': teacher_preds,
        'labels': labels,
    }


def train_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.3,
    output_dir: Path = None,
) -> Dict:
    """
    Train student model with knowledge distillation.
    
    Args:
        teacher: Trained teacher model
        student: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: GPU device
        epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature for softening
        alpha: Weight for hard targets (1-alpha for soft)
        output_dir: Directory to save checkpoints
        
    Returns:
        Dictionary with training metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING STUDENT WITH KNOWLEDGE DISTILLATION")
    logger.info(f"{'='*60}")
    
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Optimizer and loss
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    distill_loss_fn = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=alpha,
    )
    
    best_val_accuracy = 0.0
    best_student_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        student.train()
        train_loss = 0.0
        
        for batch in train_loader:
            seq_features = batch['seq_features'].to(device)
            tech_features = batch['tech_features'].to(device)
            tech_mask = batch['tech_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = teacher(seq_features, tech_features, tech_mask)
            
            # Student predictions
            student_logits = student(seq_features, tech_features, tech_mask)
            
            # Distillation loss
            loss = distill_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=labels,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                seq_features = batch['seq_features'].to(device)
                tech_features = batch['tech_features'].to(device)
                tech_mask = batch['tech_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Teacher predictions
                teacher_logits = teacher(seq_features, tech_features, tech_mask)
                
                # Student predictions
                student_logits = student(seq_features, tech_features, tech_mask)
                
                # Loss
                loss = distill_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=labels,
                )
                val_loss += loss.item()
                
                # Accuracy
                student_probs = torch.sigmoid(student_logits)
                preds_binary = (student_probs > 0.5).long()
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
            best_student_state = student.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                student_type = 'mlp' if isinstance(student, StudentNetworkMLP) else 'cnn'
                ckpt_path = output_dir / f"student_{student_type}.pt"
                torch.save({
                    'state_dict': best_student_state,
                    'type': student_type,
                    'accuracy': best_val_accuracy,
                    'temperature': temperature,
                    'alpha': alpha,
                }, ckpt_path)
                logger.info(f"✓ Saved checkpoint: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best state
    if best_student_state:
        student.load_state_dict(best_student_state)
    
    logger.info(f"✓ Training complete")
    logger.info(f"  Best validation accuracy: {best_val_accuracy:.4f}\n")
    
    return {
        'best_accuracy': best_val_accuracy,
        'student_state': best_student_state,
    }


def benchmark_inference_speed(
    teacher: nn.Module,
    student: nn.Module,
    batch_size: int = 32,
    num_batches: int = 100,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """Benchmark inference speed."""
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SPEED BENCHMARK")
    logger.info("="*60)
    
    teacher.eval()
    student.eval()
    
    # Create dummy data
    dummy_seq = torch.randn(batch_size, 21, device=device)
    dummy_tech = torch.randn(batch_size, 20, device=device)
    dummy_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)
    
    # Benchmark teacher
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_batches):
            _ = teacher(dummy_seq, dummy_tech, dummy_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    teacher_time = (time.time() - start) / num_batches * 1000  # ms per batch
    
    # Benchmark student
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_batches):
            _ = student(dummy_seq, dummy_tech, dummy_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    student_time = (time.time() - start) / num_batches * 1000  # ms per batch
    
    speedup = teacher_time / student_time if student_time > 0 else 0
    
    logger.info(f"Teacher inference time: {teacher_time:.2f} ms/batch")
    logger.info(f"Student inference time: {student_time:.2f} ms/batch")
    logger.info(f"Speedup: {speedup:.1f}x")
    
    return {
        'teacher_time_ms': teacher_time,
        'student_time_ms': student_time,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train student model with knowledge distillation'
    )
    parser.add_argument(
        '--teacher',
        type=str,
        default='models/error_predictor_v2.pt',
        help='Path to trained teacher model',
    )
    parser.add_argument(
        '--student-type',
        type=str,
        choices=['mlp', 'cnn'],
        default='mlp',
        help='Type of student model',
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
        default='models/student_models',
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
        default=50,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=4.0,
        help='Temperature for probability softening',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Weight for hard targets (1-alpha for soft)',
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
    logger.info(f"Student type: {args.student_type}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load teacher
    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        logger.error(f"Teacher model not found: {teacher_path}")
        sys.exit(1)
    
    teacher = load_teacher(teacher_path, device)
    
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
    
    # Create dataset
    dataset = DistillationDataset(batch_files, feature_builder)
    
    if len(dataset) < 100:
        logger.error(f"Too few examples ({len(dataset)}), need at least 100")
        sys.exit(1)
    
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
    
    # Create student
    student = create_student_model(args.student_type, 45).to(device)
    
    logger.info(f"\nStudent model parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Train student
    result = train_student(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        output_dir=output_dir,
    )
    
    # Benchmark
    benchmark_result = benchmark_inference_speed(
        teacher=teacher,
        student=student,
        batch_size=args.batch_size,
        num_batches=100,
        device=device,
    )
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DISTILLATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Student accuracy: {result['best_accuracy']:.4f}")
    logger.info(f"Speedup: {benchmark_result['speedup']:.1f}x")
    logger.info(f"✅ Model saved to: {output_dir}")
    
    # Save summary
    import json
    summary = {
        'teacher': str(teacher_path),
        'student_type': args.student_type,
        'output_dir': str(output_dir),
        'accuracy': float(result['best_accuracy']),
        'speedup': float(benchmark_result['speedup']),
        'inference_time_ms': {
            'teacher': float(benchmark_result['teacher_time_ms']),
            'student': float(benchmark_result['student_time_ms']),
        },
        'hyperparameters': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'temperature': args.temperature,
            'alpha': args.alpha,
            'batch_size': args.batch_size,
        },
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
