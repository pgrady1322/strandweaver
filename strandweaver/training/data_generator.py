"""
Training data generation for adaptive k-mer prediction.

This module generates labeled training data by:
1. Running assemblies with different k-mer combinations
2. Evaluating quality metrics for each stage
3. Labeling optimal k values based on assembly quality
4. Saving features + labels for ML training

The training data includes:
- Read features (from FeatureExtractor)
- Assembly quality metrics per k combination
- Optimal k labels for each stage (DBG, UL overlap, extension, polish)
"""

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from strandweaver.read_correction.feature_extraction import FeatureExtractor, ReadFeatures

logger = logging.getLogger(__name__)


@dataclass
class AssemblyMetrics:
    """
    Quality metrics for a single assembly run.
    
    These metrics are used to determine optimal k values for each stage.
    """
    # General assembly quality
    n50: int
    n90: int
    num_contigs: int
    total_length: int
    largest_contig: int
    
    # DBG-specific metrics
    dbg_connectivity: float  # % reads in largest component
    dbg_node_count: int
    dbg_edge_density: float
    dbg_bubble_count: int  # Lower is better
    
    # UL overlap metrics
    ul_mapping_rate: float  # % UL reads successfully mapped
    ul_anchor_specificity: float  # Unique vs multi-mapped
    ul_spanning_accuracy: float  # Validated spans (if ref available)
    ul_false_positive_rate: float
    
    # Extension metrics (if implemented)
    gap_closure_rate: Optional[float] = None
    extension_accuracy: Optional[float] = None
    mis_join_rate: Optional[float] = None
    
    # Polish metrics (if implemented)
    qv_improvement: Optional[float] = None
    error_rate_reduction: Optional[float] = None
    
    # Reference-based metrics (if available)
    na50: Optional[int] = None  # N50 aligned to reference
    genome_fraction: Optional[float] = None  # % reference covered
    num_misassemblies: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingSample:
    """
    Single training sample: features + k-mer choices + quality metrics.
    """
    # Dataset identifier
    dataset_name: str
    
    # Read features (input to ML)
    features: ReadFeatures
    
    # K-mer choices for this run
    dbg_k: int
    ul_overlap_k: int
    extension_k: int
    polish_k: int
    
    # Assembly quality metrics (output)
    metrics: AssemblyMetrics
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for CSV/DataFrame."""
        result = {
            'dataset_name': self.dataset_name,
            'dbg_k': self.dbg_k,
            'ul_overlap_k': self.ul_overlap_k,
            'extension_k': self.extension_k,
            'polish_k': self.polish_k,
        }
        
        # Add features (prefixed with 'feat_')
        for k, v in self.features.to_dict().items():
            result[f'feat_{k}'] = v
        
        # Add metrics (prefixed with 'metric_')
        for k, v in self.metrics.to_dict().items():
            result[f'metric_{k}'] = v
        
        return result


class TrainingDataGenerator:
    """
    Generates training data by running assemblies with different k combinations.
    
    Strategy:
    1. Grid search over reasonable k ranges
    2. Run assembly for each k combination
    3. Evaluate quality metrics per stage
    4. Save results for labeling
    
    Example:
        generator = TrainingDataGenerator(
            output_dir='training_data',
            reference_genome='ecoli_ref.fa'  # Optional for validation
        )
        
        samples = generator.generate_for_dataset(
            reads_file='ecoli_hifi.fastq',
            dataset_name='ecoli_hifi_40x'
        )
    """
    
    def __init__(self, 
                 output_dir: Path,
                 reference_genome: Optional[Path] = None,
                 k_ranges: Optional[Dict[str, List[int]]] = None,
                 max_workers: int = 4):
        """
        Initialize training data generator.
        
        Args:
            output_dir: Directory to save training data
            reference_genome: Optional reference for validation metrics
            k_ranges: K-mer ranges to test for each stage
            max_workers: Number of parallel assembly jobs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_genome = Path(reference_genome) if reference_genome else None
        self.max_workers = max_workers
        
        # Default k-mer ranges to test
        self.k_ranges = k_ranges or {
            'dbg': [21, 31, 41, 51],
            'ul_overlap': [201, 501, 1001],
            'extension': [31, 41, 55, 77],
            'polish': [41, 55, 77, 101],
        }
    
    def generate_for_dataset(self,
                            reads_file: Path,
                            dataset_name: str,
                            ul_reads_file: Optional[Path] = None) -> List[TrainingSample]:
        """
        Generate training samples for a single dataset.
        
        Args:
            reads_file: Path to accurate reads (HiFi/Illumina)
            dataset_name: Name for this dataset
            ul_reads_file: Optional ultralong reads
        
        Returns:
            List of training samples with different k combinations
        """
        logger.info(f"Generating training data for {dataset_name}")
        
        # Extract features once (same for all k combinations)
        extractor = FeatureExtractor()
        features = extractor.extract_from_file(reads_file)
        
        logger.info(f"Extracted features: {features.read_type}, "
                   f"{features.num_reads:,} reads, "
                   f"mean length {features.mean_read_length:.0f} bp")
        
        # Generate all k combinations to test
        k_combinations = self._generate_k_combinations()
        logger.info(f"Testing {len(k_combinations)} k-mer combinations")
        
        # Run assemblies in parallel
        samples = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_assembly_with_k,
                    reads_file,
                    ul_reads_file,
                    k_combo,
                    features,
                    dataset_name
                ): k_combo
                for k_combo in k_combinations
            }
            
            for future in as_completed(futures):
                k_combo = futures[future]
                try:
                    sample = future.result()
                    samples.append(sample)
                    logger.info(f"Completed k={k_combo}: N50={sample.metrics.n50:,}")
                except Exception as e:
                    logger.error(f"Failed for k={k_combo}: {e}")
        
        # Save samples
        self._save_samples(samples, dataset_name)
        
        return samples
    
    def _generate_k_combinations(self) -> List[Dict[str, int]]:
        """
        Generate all combinations of k values to test.
        
        For efficiency, we don't test full cartesian product.
        Instead, we use:
        1. Fixed combinations that make biological sense
        2. Vary one stage at a time to measure impact
        """
        combinations = []
        
        # Strategy 1: Test each DBG k with reasonable defaults for other stages
        for dbg_k in self.k_ranges['dbg']:
            combinations.append({
                'dbg': dbg_k,
                'ul_overlap': 501,  # Medium default
                'extension': 55,    # Standard
                'polish': 77,       # High quality
            })
        
        # Strategy 2: Test each UL k with reasonable defaults
        for ul_k in self.k_ranges['ul_overlap']:
            combinations.append({
                'dbg': 31,          # Standard
                'ul_overlap': ul_k,
                'extension': 55,
                'polish': 77,
            })
        
        # Strategy 3: Test each extension k
        for ext_k in self.k_ranges['extension']:
            combinations.append({
                'dbg': 31,
                'ul_overlap': 501,
                'extension': ext_k,
                'polish': 77,
            })
        
        # Strategy 4: Test each polish k
        for pol_k in self.k_ranges['polish']:
            combinations.append({
                'dbg': 31,
                'ul_overlap': 501,
                'extension': 55,
                'polish': pol_k,
            })
        
        # Remove duplicates
        unique_combos = []
        seen = set()
        for combo in combinations:
            key = (combo['dbg'], combo['ul_overlap'], combo['extension'], combo['polish'])
            if key not in seen:
                unique_combos.append(combo)
                seen.add(key)
        
        return unique_combos
    
    def _run_assembly_with_k(self,
                            reads_file: Path,
                            ul_reads_file: Optional[Path],
                            k_combo: Dict[str, int],
                            features: ReadFeatures,
                            dataset_name: str) -> TrainingSample:
        """
        Run assembly with specific k combination and collect metrics.
        
        NOTE: This is a placeholder that will be implemented with actual
        assembly code. For initial training, we can use simulated metrics
        based on literature and best practices.
        """
        # TODO: Replace with actual assembly run
        # For now, simulate metrics based on k-mer biology
        
        metrics = self._simulate_assembly_metrics(k_combo, features)
        
        return TrainingSample(
            dataset_name=dataset_name,
            features=features,
            dbg_k=k_combo['dbg'],
            ul_overlap_k=k_combo['ul_overlap'],
            extension_k=k_combo['extension'],
            polish_k=k_combo['polish'],
            metrics=metrics,
        )
    
    def _simulate_assembly_metrics(self,
                                   k_combo: Dict[str, int],
                                   features: ReadFeatures) -> AssemblyMetrics:
        """
        Simulate assembly metrics based on k-mer biology.
        
        This is a temporary placeholder until we can run real assemblies.
        Uses heuristics based on:
        - Literature (SPAdes, Verkko papers)
        - Read characteristics
        - K-mer theory
        """
        # Base quality from read type
        base_quality = {
            'hifi': 1.0,
            'ont': 0.7,
            'illumina': 0.85,
            'unknown': 0.6,
        }.get(features.read_type, 0.6)
        
        # DBG k affects connectivity
        dbg_k = k_combo['dbg']
        if dbg_k < 21:
            dbg_quality = 0.6  # Too small, promiscuous
        elif dbg_k <= 31:
            dbg_quality = 1.0  # Optimal
        elif dbg_k <= 51:
            dbg_quality = 0.9  # Still good
        else:
            dbg_quality = 0.7  # Too large, fragmented
        
        # UL overlap k affects spanning
        ul_k = k_combo['ul_overlap']
        if ul_k < 201:
            ul_quality = 0.5  # Too promiscuous
        elif ul_k <= 501:
            ul_quality = 0.8  # Good
        elif ul_k <= 1001:
            ul_quality = 1.0  # Optimal for long reads
        else:
            ul_quality = 0.9  # Very specific
        
        # Combined quality score
        overall_quality = base_quality * 0.5 + dbg_quality * 0.3 + ul_quality * 0.2
        
        # Simulate N50 (larger is better)
        base_n50 = 50000  # Base contig size
        n50 = int(base_n50 * overall_quality * (1 + np.random.normal(0, 0.1)))
        
        # Simulate other metrics
        return AssemblyMetrics(
            n50=max(1000, n50),
            n90=max(500, n50 // 3),
            num_contigs=int(100 / overall_quality),
            total_length=int(5_000_000 * overall_quality),
            largest_contig=int(n50 * 2),
            
            # DBG metrics
            dbg_connectivity=min(1.0, dbg_quality * 0.95),
            dbg_node_count=int(10000 / dbg_k),
            dbg_edge_density=dbg_quality * 0.8,
            dbg_bubble_count=int(100 * (1 - dbg_quality)),
            
            # UL metrics
            ul_mapping_rate=min(1.0, ul_quality * 0.9),
            ul_anchor_specificity=ul_quality,
            ul_spanning_accuracy=ul_quality * 0.95,
            ul_false_positive_rate=(1 - ul_quality) * 0.1,
        )
    
    def _save_samples(self, samples: List[TrainingSample], dataset_name: str):
        """Save training samples to disk."""
        # Convert to DataFrame
        df = pd.DataFrame([s.to_dict() for s in samples])
        
        # Save CSV
        csv_path = self.output_dir / f"{dataset_name}_samples.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(samples)} samples to {csv_path}")
        
        # Save JSON (more detailed)
        json_path = self.output_dir / f"{dataset_name}_samples.json"
        with open(json_path, 'w') as f:
            json.dump([s.to_dict() for s in samples], f, indent=2, default=str)
        logger.info(f"Saved detailed JSON to {json_path}")


class TrainingDataLabeler:
    """
    Labels training data with optimal k values for each stage.
    
    Given assembly results with different k combinations, determines
    which k was optimal for each stage based on stage-specific metrics.
    """
    
    def __init__(self):
        pass
    
    def label_optimal_k(self, samples: List[TrainingSample]) -> pd.DataFrame:
        """
        Determine optimal k for each stage based on metrics.
        
        Returns:
            DataFrame with features → optimal k labels
        """
        # Group by dataset
        df = pd.DataFrame([s.to_dict() for s in samples])
        
        # For each stage, find k that maximizes relevant metrics
        labeled_data = []
        
        for dataset in df['dataset_name'].unique():
            dataset_samples = df[df['dataset_name'] == dataset]
            
            # Find optimal DBG k (maximize connectivity, minimize bubbles)
            dbg_scores = (
                dataset_samples['metric_dbg_connectivity'] * 0.6 +
                (1 - dataset_samples['metric_dbg_bubble_count'] / 200) * 0.4
            )
            optimal_dbg_k = dataset_samples.loc[dbg_scores.idxmax(), 'dbg_k']
            
            # Find optimal UL k (maximize mapping rate and specificity)
            ul_scores = (
                dataset_samples['metric_ul_mapping_rate'] * 0.4 +
                dataset_samples['metric_ul_anchor_specificity'] * 0.6
            )
            optimal_ul_k = dataset_samples.loc[ul_scores.idxmax(), 'ul_overlap_k']
            
            # Find optimal extension k (maximize N50)
            optimal_ext_k = dataset_samples.loc[dataset_samples['metric_n50'].idxmax(), 'extension_k']
            
            # Find optimal polish k (maximize N50 for now)
            optimal_pol_k = dataset_samples.loc[dataset_samples['metric_n50'].idxmax(), 'polish_k']
            
            # Create labeled sample (use first row's features)
            labeled = dataset_samples.iloc[0].copy()
            labeled['optimal_dbg_k'] = optimal_dbg_k
            labeled['optimal_ul_overlap_k'] = optimal_ul_k
            labeled['optimal_extension_k'] = optimal_ext_k
            labeled['optimal_polish_k'] = optimal_pol_k
            
            labeled_data.append(labeled)
        
        return pd.DataFrame(labeled_data)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    generator = TrainingDataGenerator(
        output_dir=Path('training_data'),
        k_ranges={
            'dbg': [21, 31, 41],
            'ul_overlap': [201, 501, 1001],
            'extension': [41, 55],
            'polish': [55, 77],
        }
    )
    
    # Would generate data for real datasets
    # samples = generator.generate_for_dataset(
    #     reads_file=Path('data/ecoli_hifi.fastq'),
    #     dataset_name='ecoli_hifi'
    # )
    
    print("Training data generator ready!")
    print("Use generate_for_dataset() with real read files to create training data")
