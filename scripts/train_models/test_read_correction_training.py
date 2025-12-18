#!/usr/bin/env python3
"""
Test Read Correction Model Training (GPU/MPS Verification)

Quick test to verify:
1. Training data can be loaded
2. GPU/MPS acceleration is working
3. Models can train on small subset

This test uses only a small subset of data to verify configuration
before launching full training runs.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_gpu_availability():
    """Test GPU availability for both XGBoost and PyTorch."""
    logger.info("=" * 80)
    logger.info("Testing GPU Availability")
    logger.info("=" * 80)
    
    # Test PyTorch MPS
    mps_available = False
    cuda_available = False
    
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("✅ PyTorch MPS (Apple Silicon GPU) available")
            # Test MPS operations
            x = torch.randn(100, 100, device='mps')
            y = torch.matmul(x, x.T)
            logger.info("✅ MPS operations working correctly")
            mps_available = True
        elif torch.cuda.is_available():
            logger.info("✅ PyTorch CUDA (NVIDIA GPU) available")
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            cuda_available = True
        else:
            logger.warning("⚠️ No GPU detected - PyTorch will use CPU")
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    
    # Test XGBoost
    try:
        import xgboost as xgb
        logger.info(f"✅ XGBoost installed: {xgb.__version__}")
        if cuda_available:
            logger.info("   XGBoost can use CUDA GPU")
        else:
            logger.info("   XGBoost will use CPU (MPS not supported)")
    except ImportError:
        logger.error("❌ XGBoost not installed")
        return False
    
    return True


def test_kmer_selector_data():
    """Test loading k-mer selector training data."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing K-mer Selector Data Loading")
    logger.info("=" * 80)
    
    data_dir = Path('training_data/read_correction/adaptive_kmer')
    
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Count files per technology
    technologies = ['ont_r9', 'ont_r10', 'pacbio_hifi', 'pacbio_clr', 'illumina', 'ancient_dna']
    total_files = 0
    
    for tech in technologies:
        tech_files = list(data_dir.glob(f'adaptive_kmer_{tech}_batch_*.pkl'))
        logger.info(f"  {tech}: {len(tech_files)} batches")
        total_files += len(tech_files)
    
    logger.info(f"✅ Total batches: {total_files}")
    
    # Load a small sample
    import pickle
    sample_file = list(data_dir.glob('adaptive_kmer_*.pkl'))[0]
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    
    logger.info(f"✅ Sample batch loaded: {len(sample_data)} examples")
    
    return True


def test_base_error_data():
    """Test loading ErrorSmith (base error predictor) training data."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing ErrorSmith Data Loading")
    logger.info("=" * 80)
    
    data_dir = Path('training_data/read_correction/base_error')
    
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Count files per technology
    technologies = ['ont_r9', 'ont_r10', 'pacbio_hifi', 'pacbio_clr', 'illumina', 'ancient_dna']
    total_files = 0
    
    for tech in technologies:
        tech_files = list(data_dir.glob(f'base_error_{tech}_batch_*.pkl'))
        logger.info(f"  {tech}: {len(tech_files)} batches")
        total_files += len(tech_files)
    
    logger.info(f"✅ Total batches: {total_files}")
    
    # Load a small sample
    import pickle
    sample_file = list(data_dir.glob('base_error_*.pkl'))[0]
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    
    logger.info(f"✅ Sample batch loaded: {len(sample_data)} examples")
    
    return True


def test_kmer_selector_training():
    """Test k-mer selector model training on small subset."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing K-mer Selector Model Training (Small Subset)")
    logger.info("=" * 80)
    
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Load small subset from first 5 batches
    data_dir = Path('training_data/read_correction/adaptive_kmer')
    batch_files = sorted(data_dir.glob('adaptive_kmer_*.pkl'))[:5]
    
    features_list = []
    labels_list = []
    
    for pkl_file in batch_files:
        with open(pkl_file, 'rb') as f:
            batch_data = pickle.load(f)
            for read_ctx, optimal_k in batch_data:
                # Extract basic features
                features = [
                    getattr(read_ctx, 'error_rate', 0.1),
                    getattr(read_ctx, 'coverage', 30.0),
                    getattr(read_ctx, 'gc_content', 0.5),
                    getattr(read_ctx, 'homopolymer_length', 3.0) if hasattr(read_ctx, 'homopolymer_length') else 3.0,
                    len(read_ctx.sequence),
                ]
                features_list.append(features)
                labels_list.append(optimal_k)
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    logger.info(f"Loaded {len(features)} examples")
    
    # Train tiny XGBoost model
    try:
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Use regression mode since k-mer values are integers (not classes 0-4)
        model = xgb.XGBRegressor(
            n_estimators=10,  # Very small for testing
            max_depth=4,
            random_state=42
        )
        
        logger.info("Training small XGBoost model...")
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        logger.info(f"✅ Model trained successfully! Test R² score: {score:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False


def test_base_error_training():
    """Test ErrorSmith model training on small subset with GPU/MPS."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing ErrorSmith Training (Small Subset, GPU/MPS)")
    logger.info("=" * 80)
    
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ Using Apple Silicon GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ Using CPU (no GPU detected)")
    
    # Load small subset from first 5 batches
    data_dir = Path('training_data/read_correction/base_error')
    batch_files = sorted(data_dir.glob('base_error_*.pkl'))[:5]
    
    features_list = []
    labels_list = []
    
    base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    
    for pkl_file in batch_files:
        with open(pkl_file, 'rb') as f:
            batch_data = pickle.load(f)
            for base_ctx, error_label in batch_data:
                # Simple feature extraction
                left = base_ctx.left_context[-10:] if base_ctx.left_context else ''
                right = base_ctx.right_context[:10] if base_ctx.right_context else ''
                
                left_enc = [base_map.get(b, 0) for b in left]
                right_enc = [base_map.get(b, 0) for b in right]
                
                left_enc = ([0] * (10 - len(left_enc))) + left_enc
                right_enc = right_enc + ([0] * (10 - len(right_enc)))
                
                features = left_enc + [base_map.get(base_ctx.base, 0), base_ctx.quality_score, base_ctx.kmer_coverage] + right_enc
                features_list.append(features)
                labels_list.append(1.0 if error_label == 'error' else 0.0)
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    logger.info(f"Loaded {len(features)} examples")
    
    # Simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.layers(x).squeeze(-1)
    
    # Split and create dataloaders
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create and train model
    model = SimpleModel(features.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info("Training small neural network on GPU/MPS...")
    
    try:
        # Train for 3 epochs
        for epoch in range(3):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            logger.info(f"  Epoch {epoch+1}/3 - Loss: {train_loss:.4f}")
        
        # Test
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total
        logger.info(f"✅ Model trained successfully on {device}! Test accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Read Correction Model Training - GPU/MPS Verification Test")
    logger.info("=" * 80)
    logger.info("")
    
    results = {
        'GPU Availability': test_gpu_availability(),
        'K-mer Data Loading': test_kmer_selector_data(),
        'Base Error Data Loading': test_base_error_data(),
        'K-mer Selector Training': test_kmer_selector_training(),
        'Base Error Training (GPU/MPS)': test_base_error_training(),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - Ready for full training!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("  1. Train K-Weaver: python scripts/train_models/train_kmer_selector.py")
        logger.info("  2. Train ErrorSmith: python scripts/train_models/train_base_error_predictor.py")
        logger.info("  3. Or run both: bash scripts/train_models/train_all_read_correction_models.sh")
    else:
        logger.error("❌ SOME TESTS FAILED - Fix issues before full training")
        logger.info("=" * 80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
