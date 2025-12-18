"""
XGBoost model training for adaptive k-mer prediction.

Trains separate models for each assembly stage:
1. DBG k prediction
2. UL overlap k prediction  
3. Extension k prediction
4. Polish k prediction

Uses cross-validation and saves trained models for deployment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

logger = logging.getLogger(__name__)


class KmerModelTrainer:
    """
    Trains XGBoost models to predict optimal k-mer sizes.
    
    Strategy:
    - Separate model for each stage (DBG, UL overlap, extension, polish)
    - Regression task (predict k value)
    - Cross-validation for robust evaluation
    - Hyperparameter tuning via grid search
    
    Example:
        trainer = KmerModelTrainer(output_dir='models')
        trainer.train_from_csv('training_data/labeled_data.csv')
        trainer.save_models()
    """
    
    def __init__(self,
                 output_dir: Path,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            output_dir: Directory to save trained models
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Trained models (one per stage)
        self.models = {}
        
        # Training history
        self.training_history = {
            'dbg': {},
            'ul_overlap': {},
            'extension': {},
            'polish': {},
        }
    
    def train_from_csv(self, data_path: Path):
        """
        Train all models from labeled CSV data.
        
        Expected CSV columns:
        - feat_* columns (features)
        - optimal_dbg_k (target)
        - optimal_ul_overlap_k (target)
        - optimal_extension_k (target)
        - optimal_polish_k (target)
        """
        logger.info(f"Loading training data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Extract features
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        X = df[feature_cols].values
        
        logger.info(f"Training with {len(X)} samples, {len(feature_cols)} features")
        
        # Train model for each stage
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            target_col = f'optimal_{stage}_k'
            
            if target_col not in df.columns:
                logger.warning(f"No target column {target_col}, skipping {stage}")
                continue
            
            y = df[target_col].values
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {stage} k predictor")
            logger.info(f"{'='*60}")
            
            model = self._train_single_model(X, y, stage)
            self.models[stage] = model
    
    def _train_single_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           stage: str) -> xgb.XGBRegressor:
        """
        Train XGBoost model for a single stage.
        
        Uses grid search for hyperparameter tuning.
        """
        logger.info(f"Target range: [{y.min()}, {y.max()}], mean: {y.mean():.1f}")
        
        # Define hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        # Grid search with cross-validation
        logger.info("Running grid search for optimal hyperparameters...")
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=-1,
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Evaluate with cross-validation
        self._evaluate_model(best_model, X, y, stage)
        
        # Train final model on all data
        logger.info("Training final model on full dataset...")
        best_model.fit(X, y)
        
        # Feature importance
        self._log_feature_importance(best_model, stage)
        
        return best_model
    
    def _evaluate_model(self,
                       model: xgb.XGBRegressor,
                       X: np.ndarray,
                       y: np.ndarray,
                       stage: str):
        """Evaluate model with cross-validation."""
        logger.info(f"\nCross-validation evaluation ({self.cv_folds} folds):")
        
        # MAE (Mean Absolute Error)
        mae_scores = -cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error'
        )
        
        # R² score
        r2_scores = cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='r2'
        )
        
        # RMSE (Root Mean Squared Error)
        mse_scores = -cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(mse_scores)
        
        logger.info(f"  MAE:  {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
        logger.info(f"  RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        logger.info(f"  R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
        
        # Store in history
        self.training_history[stage] = {
            'mae_mean': float(mae_scores.mean()),
            'mae_std': float(mae_scores.std()),
            'rmse_mean': float(rmse_scores.mean()),
            'rmse_std': float(rmse_scores.std()),
            'r2_mean': float(r2_scores.mean()),
            'r2_std': float(r2_scores.std()),
            'cv_folds': self.cv_folds,
        }
        
        # Check if MAE is acceptable (within 5 of optimal)
        if mae_scores.mean() > 5:
            logger.warning(f"⚠️  MAE > 5 - model may need improvement")
        else:
            logger.info(f"✓ MAE < 5 - good prediction accuracy!")
    
    def _log_feature_importance(self, model: xgb.XGBRegressor, stage: str):
        """Log and save feature importance."""
        importance = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        logger.info(f"\nTop 10 most important features for {stage}:")
        for i, idx in enumerate(indices[:10], 1):
            logger.info(f"  {i}. Feature {idx}: {importance[idx]:.4f}")
        
        # Save to history
        self.training_history[stage]['feature_importance'] = {
            int(i): float(imp) for i, imp in enumerate(importance)
        }
    
    def save_models(self):
        """Save trained models and training history."""
        # Save each model
        for stage, model in self.models.items():
            model_path = self.output_dir / f'{stage}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {stage} model to {model_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save metadata
        metadata = {
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'stages': list(self.models.keys()),
        }
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_models(self):
        """Load previously trained models."""
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            model_path = self.output_dir / f'{stage}_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[stage] = pickle.load(f)
                logger.info(f"Loaded {stage} model from {model_path}")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load training history
        history_path = self.output_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict k values for all stages.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Dict mapping stage → predictions
        """
        predictions = {}
        for stage, model in self.models.items():
            predictions[stage] = model.predict(X)
        return predictions
    
    def generate_report(self) -> str:
        """Generate a summary report of training results."""
        report = []
        report.append("=" * 70)
        report.append("ADAPTIVE K-MER PREDICTION: TRAINING REPORT")
        report.append("=" * 70)
        report.append("")
        
        for stage in ['dbg', 'ul_overlap', 'extension', 'polish']:
            if stage not in self.training_history:
                continue
            
            hist = self.training_history[stage]
            
            report.append(f"\n{stage.upper()} K PREDICTION")
            report.append("-" * 40)
            report.append(f"MAE:  {hist['mae_mean']:.2f} ± {hist['mae_std']:.2f}")
            report.append(f"RMSE: {hist['rmse_mean']:.2f} ± {hist['rmse_std']:.2f}")
            report.append(f"R²:   {hist['r2_mean']:.3f} ± {hist['r2_std']:.3f}")
            
            # Success criteria
            if hist['mae_mean'] < 5:
                report.append("✓ PASSED: MAE < 5 (within 5 of optimal k)")
            else:
                report.append("⚠️  WARNING: MAE > 5 (may need more training data)")
            
            if hist['r2_mean'] > 0.7:
                report.append("✓ PASSED: R² > 0.7 (good predictive power)")
            else:
                report.append("⚠️  WARNING: R² < 0.7 (weak predictions)")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """Example training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train adaptive k-mer prediction models')
    parser.add_argument('--data', required=True, help='Path to labeled training data CSV')
    parser.add_argument('--output', default='models', help='Output directory for models')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models
    trainer = KmerModelTrainer(
        output_dir=Path(args.output),
        cv_folds=args.cv_folds
    )
    
    trainer.train_from_csv(Path(args.data))
    trainer.save_models()
    
    # Print report
    print("\n" + trainer.generate_report())


if __name__ == '__main__':
    main()
