"""
Benchmark validation for adaptive k-mer prediction models.

Tests trained models on held-out benchmark datasets and compares
predictions to actual assembly quality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from strandweaver.preprocessing.kweaver_module import FeatureExtractor
from strandweaver.training.model_trainer import KmerModelTrainer
from strandweaver.training.data_generator import TrainingDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from validating model on a single dataset."""
    dataset_name: str
    
    # Predicted k values
    predicted_dbg_k: int
    predicted_ul_k: int
    predicted_ext_k: int
    predicted_pol_k: int
    
    # Actual optimal k values (from ground truth)
    optimal_dbg_k: int
    optimal_ul_k: int
    optimal_ext_k: int
    optimal_pol_k: int
    
    # Assembly quality with predicted k
    n50_with_predicted: int
    n50_with_optimal: int
    n50_with_default: int  # Baseline (k=31 for all)
    
    # Prediction errors
    dbg_error: int
    ul_error: int
    ext_error: int
    pol_error: int
    
    def improvement_over_default(self) -> float:
        """Calculate % improvement over default k=31."""
        if self.n50_with_default == 0:
            return 0.0
        return (self.n50_with_predicted - self.n50_with_default) / self.n50_with_default * 100
    
    def gap_to_optimal(self) -> float:
        """Calculate % gap between predicted and optimal."""
        if self.n50_with_optimal == 0:
            return 0.0
        return (self.n50_with_optimal - self.n50_with_predicted) / self.n50_with_optimal * 100


class ModelValidator:
    """
    Validates trained k-mer prediction models on benchmark datasets.
    
    Compares:
    1. Predicted k vs optimal k (error in k-mer choice)
    2. Assembly quality with predicted k vs optimal k
    3. Assembly quality with predicted k vs default k=31
    
    Example:
        validator = ModelValidator(models_dir='models')
        results = validator.validate_on_dataset(
            reads_file='test_data/yeast_hifi.fastq',
            dataset_name='yeast_hifi_40x'
        )
        validator.print_summary()
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize validator.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        
        # Load trained models
        self.trainer = KmerModelTrainer(output_dir=models_dir)
        self.trainer.load_models()
        
        # Validation results
        self.results: List[ValidationResult] = []
    
    def validate_on_dataset(self,
                           reads_file: Path,
                           dataset_name: str,
                           ground_truth_k: Optional[Dict[str, int]] = None,
                           ul_reads_file: Optional[Path] = None) -> ValidationResult:
        """
        Validate model on a single dataset.
        
        Args:
            reads_file: Path to reads for validation
            dataset_name: Name of this dataset
            ground_truth_k: Known optimal k values (if available)
            ul_reads_file: Optional ultralong reads
        
        Returns:
            ValidationResult with prediction accuracy and assembly quality
        """
        logger.info(f"Validating on {dataset_name}")
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_from_file(reads_file)
        
        # Predict k values
        X = features.to_feature_vector().reshape(1, -1)
        predictions = self.trainer.predict(X)
        
        predicted_k = {
            'dbg': int(predictions['dbg'][0]),
            'ul_overlap': int(predictions['ul_overlap'][0]),
            'extension': int(predictions['extension'][0]),
            'polish': int(predictions['polish'][0]),
        }
        
        logger.info(f"Predicted k values: {predicted_k}")
        
        # If ground truth not provided, run assemblies to find optimal
        if ground_truth_k is None:
            logger.info("No ground truth provided, running assemblies to find optimal k...")
            ground_truth_k = self._find_optimal_k(
                reads_file,
                ul_reads_file,
                dataset_name
            )
        
        logger.info(f"Ground truth k values: {ground_truth_k}")
        
        # Run assemblies with different k choices
        n50_predicted = self._run_assembly_and_get_n50(
            reads_file, ul_reads_file, predicted_k
        )
        n50_optimal = self._run_assembly_and_get_n50(
            reads_file, ul_reads_file, ground_truth_k
        )
        n50_default = self._run_assembly_and_get_n50(
            reads_file, ul_reads_file,
            {'dbg': 31, 'ul_overlap': 31, 'extension': 31, 'polish': 31}
        )
        
        # Calculate errors
        result = ValidationResult(
            dataset_name=dataset_name,
            predicted_dbg_k=predicted_k['dbg'],
            predicted_ul_k=predicted_k['ul_overlap'],
            predicted_ext_k=predicted_k['extension'],
            predicted_pol_k=predicted_k['polish'],
            optimal_dbg_k=ground_truth_k['dbg'],
            optimal_ul_k=ground_truth_k['ul_overlap'],
            optimal_ext_k=ground_truth_k['extension'],
            optimal_pol_k=ground_truth_k['polish'],
            n50_with_predicted=n50_predicted,
            n50_with_optimal=n50_optimal,
            n50_with_default=n50_default,
            dbg_error=abs(predicted_k['dbg'] - ground_truth_k['dbg']),
            ul_error=abs(predicted_k['ul_overlap'] - ground_truth_k['ul_overlap']),
            ext_error=abs(predicted_k['extension'] - ground_truth_k['extension']),
            pol_error=abs(predicted_k['polish'] - ground_truth_k['polish']),
        )
        
        self.results.append(result)
        
        logger.info(f"Validation complete for {dataset_name}")
        logger.info(f"  N50 (predicted): {n50_predicted:,}")
        logger.info(f"  N50 (optimal):   {n50_optimal:,}")
        logger.info(f"  N50 (default):   {n50_default:,}")
        logger.info(f"  Improvement over default: {result.improvement_over_default():.1f}%")
        logger.info(f"  Gap to optimal: {result.gap_to_optimal():.1f}%")
        
        return result
    
    def _find_optimal_k(self,
                       reads_file: Path,
                       ul_reads_file: Optional[Path],
                       dataset_name: str) -> Dict[str, int]:
        """
        Run grid search to find optimal k values.
        
        Uses TrainingDataGenerator to test different k combinations.
        """
        generator = TrainingDataGenerator(
            output_dir=Path('validation_temp'),
            k_ranges={
                'dbg': [21, 31, 41, 51],
                'ul_overlap': [201, 501, 1001],
                'extension': [31, 41, 55, 77],
                'polish': [41, 55, 77, 101],
            }
        )
        
        samples = generator.generate_for_dataset(
            reads_file=reads_file,
            dataset_name=dataset_name,
            ul_reads_file=ul_reads_file
        )
        
        # Find best k for each stage
        df = pd.DataFrame([s.to_dict() for s in samples])
        
        # DBG: maximize connectivity, minimize bubbles
        dbg_scores = (
            df['metric_dbg_connectivity'] * 0.6 +
            (1 - df['metric_dbg_bubble_count'] / 200) * 0.4
        )
        optimal_dbg = int(df.loc[dbg_scores.idxmax(), 'dbg_k'])
        
        # UL: maximize mapping rate and specificity
        ul_scores = (
            df['metric_ul_mapping_rate'] * 0.4 +
            df['metric_ul_anchor_specificity'] * 0.6
        )
        optimal_ul = int(df.loc[ul_scores.idxmax(), 'ul_overlap_k'])
        
        # Extension/Polish: maximize N50
        optimal_ext = int(df.loc[df['metric_n50'].idxmax(), 'extension_k'])
        optimal_pol = int(df.loc[df['metric_n50'].idxmax(), 'polish_k'])
        
        return {
            'dbg': optimal_dbg,
            'ul_overlap': optimal_ul,
            'extension': optimal_ext,
            'polish': optimal_pol,
        }
    
    def _run_assembly_and_get_n50(self,
                                  reads_file: Path,
                                  ul_reads_file: Optional[Path],
                                  k_values: Dict[str, int]) -> int:
        """
        Run assembly with given k values and return N50.
        
        TODO: Replace with actual assembly run.
        For now, simulates based on k-mer biology.
        """
        # Simplified simulation
        # In reality, would call build_hybrid_assembly_graph()
        
        # DBG quality
        dbg_k = k_values['dbg']
        if 21 <= dbg_k <= 41:
            dbg_q = 1.0
        else:
            dbg_q = 0.8
        
        # UL quality  
        ul_k = k_values['ul_overlap']
        if 501 <= ul_k <= 1001:
            ul_q = 1.0
        else:
            ul_q = 0.7
        
        # Overall quality
        quality = (dbg_q + ul_q) / 2
        
        # Simulate N50
        base_n50 = 50000
        n50 = int(base_n50 * quality * (1 + np.random.normal(0, 0.05)))
        
        return max(1000, n50)
    
    def print_summary(self):
        """Print validation summary across all datasets."""
        if not self.results:
            print("No validation results yet")
            return
        
        print("\n" + "=" * 80)
        print("ADAPTIVE K-MER PREDICTION: VALIDATION SUMMARY")
        print("=" * 80)
        
        # Per-dataset results
        print(f"\n{'Dataset':<20} {'DBG err':>8} {'UL err':>8} {'Ext err':>8} {'Pol err':>8} {'vs Default':>12} {'vs Optimal':>12}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result.dataset_name:<20} "
                  f"{result.dbg_error:>8} "
                  f"{result.ul_error:>8} "
                  f"{result.ext_error:>8} "
                  f"{result.pol_error:>8} "
                  f"{result.improvement_over_default():>11.1f}% "
                  f"{result.gap_to_optimal():>11.1f}%")
        
        # Aggregate statistics
        print("\n" + "-" * 80)
        print("AGGREGATE STATISTICS")
        print("-" * 80)
        
        avg_dbg_err = np.mean([r.dbg_error for r in self.results])
        avg_ul_err = np.mean([r.ul_error for r in self.results])
        avg_ext_err = np.mean([r.ext_error for r in self.results])
        avg_pol_err = np.mean([r.pol_error for r in self.results])
        
        avg_improvement = np.mean([r.improvement_over_default() for r in self.results])
        avg_gap = np.mean([r.gap_to_optimal() for r in self.results])
        
        print(f"Average k-mer prediction error:")
        print(f"  DBG:       {avg_dbg_err:.1f}")
        print(f"  UL:        {avg_ul_err:.1f}")
        print(f"  Extension: {avg_ext_err:.1f}")
        print(f"  Polish:    {avg_pol_err:.1f}")
        
        print(f"\nAssembly quality:")
        print(f"  Improvement over default k=31: {avg_improvement:.1f}%")
        print(f"  Gap to optimal k:              {avg_gap:.1f}%")
        
        # Success criteria
        print("\n" + "-" * 80)
        print("SUCCESS CRITERIA")
        print("-" * 80)
        
        if avg_dbg_err < 5 and avg_ul_err < 100 and avg_ext_err < 5 and avg_pol_err < 5:
            print("✓ PASSED: Average k-mer errors within acceptable range")
        else:
            print("⚠️  WARNING: Some k-mer errors exceed thresholds")
        
        if avg_improvement > 15:
            print(f"✓ PASSED: {avg_improvement:.1f}% improvement > 15% target")
        else:
            print(f"⚠️  WARNING: {avg_improvement:.1f}% improvement < 15% target")
        
        if avg_gap < 10:
            print(f"✓ PASSED: {avg_gap:.1f}% gap to optimal < 10%")
        else:
            print(f"⚠️  INFO: {avg_gap:.1f}% gap to optimal (room for improvement)")
        
        print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate k-mer prediction models')
    parser.add_argument('--models', required=True, help='Directory with trained models')
    parser.add_argument('--reads', required=True, help='Test reads file')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--ul-reads', help='Optional ultralong reads')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    validator = ModelValidator(models_dir=Path(args.models))
    
    validator.validate_on_dataset(
        reads_file=Path(args.reads),
        dataset_name=args.dataset,
        ul_reads_file=Path(args.ul_reads) if args.ul_reads else None
    )
    
    validator.print_summary()
