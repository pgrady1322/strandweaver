#!/usr/bin/env python3
"""
Train Base Error Predictor Model

Uses 1D CNN or Bidirectional LSTM to predict per-base error probabilities
based on sequence context and read features.

GPU/MPS acceleration via PyTorch.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BaseErrorDataset(Dataset):
    """PyTorch dataset for base error prediction."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BaseErrorCNN(nn.Module):
    """1D CNN for base error prediction."""
    
    def __init__(self, input_dim: int = 100, num_classes: int = 1):
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
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv_layers(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.fc_layers(x)  # (batch, 1)
        return x.squeeze(-1)  # (batch,)


class BaseErrorLSTM(nn.Module):
    """Bidirectional LSTM for base error prediction."""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        lstm_out, _ = self.lstm(x)  # (batch, features, hidden*2)
        x = lstm_out[:, -1, :]  # Take last timestep (batch, hidden*2)
        x = self.fc_layers(x)  # (batch, 1)
        return x.squeeze(-1)  # (batch,)


def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load base error predictor training data from PKL files."""
    logger.info(f"Loading training data from {data_dir}")
    
    features_list = []
    labels_list = []
    
    # Technologies to load
    technologies = ['ont_r9', 'ont_r10', 'pacbio_hifi', 'pacbio_clr', 'illumina', 'ancient_dna']
    
    for tech in technologies:
        tech_files = sorted(data_dir.glob(f'base_error_{tech}_batch_*.pkl'))
        logger.info(f"  {tech}: {len(tech_files)} batches")
        
        for pkl_file in tech_files:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                # batch_data is a list of (BaseContext, error_label) tuples
                for base_ctx, error_label in batch_data:
                    # Extract features from BaseContext
                    # Create features from base context: position, quality, left/right context
                    left = base_ctx.left_context[-10:] if base_ctx.left_context else ''
                    right = base_ctx.right_context[:10] if base_ctx.right_context else ''
                    
                    # Encode context as integers (A=1, C=2, G=3, T=4, else=0)
                    base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
                    left_enc = [base_map.get(b, 0) for b in left]
                    right_enc = [base_map.get(b, 0) for b in right]
                    
                    # Pad to size
                    left_enc = ([0] * (10 - len(left_enc))) + left_enc
                    right_enc = right_enc + ([0] * (10 - len(right_enc)))
                    
                    features = left_enc + [base_map.get(base_ctx.base, 0), base_ctx.quality_score, base_ctx.kmer_coverage] + right_enc
                    features_list.append(features)
                    labels_list.append(1.0 if error_label == 'error' else 0.0)
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    logger.info(f"Loaded {len(features)} examples from {len(technologies)} technologies")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Label shape: {labels.shape}")
    
    return features, labels


def train_model(features: np.ndarray, labels: np.ndarray, model_type: str = 'cnn', 
                epochs: int = 20, batch_size: int = 256):
    """Train base error predictor model with GPU/MPS acceleration."""
    from sklearn.model_selection import train_test_split
    
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} examples")
    logger.info(f"Test set: {len(X_test)} examples")
    
    # Create datasets and dataloaders
    train_dataset = BaseErrorDataset(X_train, y_train)
    test_dataset = BaseErrorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    input_dim = features.shape[1]
    
    if model_type == 'cnn':
        model = BaseErrorCNN(input_dim=input_dim)
    elif model_type == 'lstm':
        model = BaseErrorLSTM(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Model architecture: {model_type.upper()}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Calculate accuracy (threshold at 0.5)
                predicted = (outputs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_loss /= len(test_loader)
        accuracy = correct / total
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
    
    logger.info(f"\n✅ Training complete! Best validation loss: {best_loss:.4f}")
    
    return model, accuracy, device


def main():
    parser = argparse.ArgumentParser(description='Train ErrorSmith Model (Base Error Predictor)')
    parser.add_argument('--data', type=str, default='training_data/read_correction/base_error',
                       help='Path to training data directory')
    parser.add_argument('--output', type=str, default='models/base_error_predictor.pt',
                       help='Output model path')
    parser.add_argument('--model-type', type=str, default='cnn',
                       choices=['cnn', 'lstm'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    features, labels = load_training_data(data_dir)
    
    # Train model
    model, accuracy, device = train_model(
        features, labels, 
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'accuracy': accuracy,
        'num_examples': len(features),
        'num_technologies': 6,
        'input_dim': features.shape[1]
    }, output_path)
    
    logger.info("✅ ErrorSmith model training complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Test accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
