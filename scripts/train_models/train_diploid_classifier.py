#!/usr/bin/env python3
"""
Train DiploidAI haplotype classifier (PyTorch MLP)

This trains a simple MLP for haplotype assignment (Hap A/B/Both/Repeat/Unknown).
Suitable for training on Colab with GPU (1-2 hours on small dataset).
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class HaplotypeDataset(Dataset):
    """Dataset for haplotype classification."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DiploidClassifier(nn.Module):
    """Simple MLP for haplotype classification."""
    
    def __init__(self, input_dim=42, hidden_dims=[64, 32], num_classes=5):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_training_data(data_dir):
    """Load diploid classification training data."""
    data_dir = Path(data_dir)
    
    features = []
    labels = []
    
    # Load all diploid feature files
    feature_files = list(data_dir.glob("**/diploid_features_*.npz"))
    
    if not feature_files:
        print(f"Error: No diploid feature files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(feature_files)} feature files")
    
    for feature_file in feature_files:
        data = np.load(feature_file)
        features.append(data['features'])
        labels.append(data['labels'])
    
    X = np.vstack(features)
    y = np.hstack(labels)
    
    print(f"Loaded {len(X)} node examples")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=0.001):
    """Train diploid classifier."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = HaplotypeDataset(X_train, y_train)
    val_dataset = HaplotypeDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    input_dim = X_train.shape[1]
    model = DiploidClassifier(input_dim=input_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    print("\nTraining diploid classifier...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model, device


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model on test set."""
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = outputs.max(1)
        y_pred = y_pred.cpu().numpy()
    
    # Class names
    class_names = ['HAP_A', 'HAP_B', 'BOTH', 'REPEAT', 'UNKNOWN']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Overall accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    return accuracy


def save_model(model, output_path, metadata):
    """Save trained model and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), output_path)
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DiploidAI haplotype classifier")
    parser.add_argument('--data-dir', required=True, help='Training data directory')
    parser.add_argument('--output', default='models/diploid_classifier_v0.1.pth', 
                       help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set proportion')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    model, device = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, device)
    
    # Save model
    metadata = {
        'version': '0.1',
        'model_type': 'pytorch_diploid_classifier',
        'input_dim': X.shape[1],
        'num_classes': 5,
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
