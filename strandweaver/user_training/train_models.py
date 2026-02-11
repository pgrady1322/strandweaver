#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Training Runner — trains five graph-related XGBoost models from CSV data,
evaluates with k-fold cross-validation, and saves weights.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("strandweaver.train_models")

# ═══════════════════════════════════════════════════════════════════════
#  OPTIONAL HEAVY DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════

_HAS_NUMPY = _HAS_XGB = _HAS_SKLEARN = _HAS_TORCH = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    pass

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    pass

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, KFold, cross_val_score,
    )
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_squared_error, r2_score,
    )
    from sklearn.pipeline import Pipeline
    _HAS_SKLEARN = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE-COLUMN CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
# Canonical order — must match graph_training_data.py export columns.

try:
    from .graph_training_data import (
        EDGE_AI_FEATURES,
        PATH_GNN_FEATURES,
        NODE_SIGNAL_FEATURES,
        UL_ROUTE_FEATURES,
        SV_DETECT_FEATURES,
    )
except ImportError:
    # Standalone fallback — keep in sync with graph_training_data.py
    EDGE_AI_FEATURES = [
        'overlap_length', 'overlap_identity', 'read1_length', 'read2_length',
        'coverage_r1', 'coverage_r2', 'gc_content_r1', 'gc_content_r2',
        'repeat_fraction_r1', 'repeat_fraction_r2',
        'kmer_diversity_r1', 'kmer_diversity_r2',
        'branching_factor_r1', 'branching_factor_r2',
        'hic_support', 'mapping_quality_r1', 'mapping_quality_r2',
    ]
    PATH_GNN_FEATURES = [
        'overlap_length', 'overlap_identity', 'coverage_consistency',
        'gc_similarity', 'repeat_match', 'branching_score',
        'path_support', 'hic_contact', 'mapping_quality',
        'kmer_match', 'sequence_complexity', 'orientation_score',
        'distance_score', 'topology_score', 'ul_support', 'sv_evidence',
    ]
    NODE_SIGNAL_FEATURES = [
        'coverage', 'gc_content', 'repeat_fraction', 'kmer_diversity',
        'branching_factor', 'hic_contact_density', 'allele_frequency',
        'heterozygosity', 'phase_consistency', 'mappability',
    ]
    UL_ROUTE_FEATURES = [
        'path_length', 'num_branches', 'coverage_mean', 'coverage_std',
        'sequence_identity', 'mapping_quality', 'num_gaps', 'gap_size_mean',
        'kmer_consistency', 'orientation_consistency', 'ul_span',
        'route_complexity',
    ]
    SV_DETECT_FEATURES = [
        'coverage_mean', 'coverage_std', 'coverage_median',
        'gc_content', 'repeat_fraction', 'kmer_diversity',
        'branching_complexity', 'hic_disruption_score',
        'ul_support', 'mapping_quality',
        'region_length', 'breakpoint_precision',
        'allele_balance', 'phase_switch_rate',
    ]


# ═══════════════════════════════════════════════════════════════════════
#  MODEL SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════

# Technologies for which EdgeWarden expects separate model files.
EDGEWARDEN_TECHNOLOGIES = ['hifi', 'ont_r9', 'ont_r10', 'illumina', 'adna']

# Maps each model type to its CSV pattern, features, label column,
# default XGBoost hyperparameters, and pipeline save layout.
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    'edge_ai': {
        'csv_glob': '**/edge_ai_training_g*.csv',
        'features': EDGE_AI_FEATURES,
        'label_col': 'label',
        'task': 'multiclass',
        'save_subdir': 'edgewarden',
        'xgb_defaults': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        'desc': 'Edge classification  (TRUE / ALLELIC / REPEAT / SV_BREAK / CHIMERIC)',
    },
    'path_gnn': {
        'csv_glob': '**/path_gnn_training_g*.csv',
        'features': PATH_GNN_FEATURES,
        'label_col': 'in_correct_path',
        'task': 'binary',
        'save_subdir': 'pathgnn',
        'xgb_defaults': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        'desc': 'Path edge scoring    (binary: in correct path or not)',
    },
    'diploid_ai': {
        'csv_glob': '**/diploid_ai_training_g*.csv',
        'features': NODE_SIGNAL_FEATURES,
        'label_col': 'haplotype_label',
        'task': 'multiclass',
        'label_transform': lambda lbl: lbl.replace('HAP_', ''),  # HAP_A→A
        'save_subdir': 'diploid',
        'xgb_defaults': {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 200},
        'desc': 'Node haplotype assign (A / B / BOTH / REPEAT / UNKNOWN)',
    },
    'ul_routing': {
        'csv_glob': '**/ul_route_training_g*.csv',
        'features': UL_ROUTE_FEATURES,
        'label_col': 'route_score',
        'task': 'regression',
        'save_subdir': 'ul_routing',
        'xgb_defaults': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        'desc': 'UL-read route scoring (regression 0.0 – 1.0)',
    },
    'sv_ai': {
        'csv_glob': '**/sv_detect_training_g*.csv',
        'features': SV_DETECT_FEATURES,
        'label_col': 'sv_type',
        'task': 'multiclass',
        'save_subdir': 'sv_detector',
        'xgb_defaults': {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150},
        'desc': 'SV type detection     (del / ins / inv / dup / trans / none)',
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelTrainingConfig:
    """Configuration for the training runner."""

    # Paths
    data_dir: str = "."
    output_dir: str = "trained_models"

    # Model selection (subset of MODEL_SPECS keys)
    models: List[str] = field(
        default_factory=lambda: list(MODEL_SPECS.keys())
    )

    # Hyperparameter overrides (None → use per-model defaults)
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    n_estimators: Optional[int] = None

    # Training regime
    validation_split: float = 0.15
    n_folds: int = 5
    early_stopping_rounds: int = 10
    random_seed: int = 42

    # EdgeWarden: which technology slots to save
    edgewarden_technologies: List[str] = field(
        default_factory=lambda: list(EDGEWARDEN_TECHNOLOGIES)
    )

    # Minimum number of samples required to train a model
    min_samples: int = 10

    verbose: bool = False


# ═══════════════════════════════════════════════════════════════════════
#  CSV LOADING
# ═══════════════════════════════════════════════════════════════════════

def find_csvs(data_dir: Path, pattern: str) -> List[Path]:
    """Find CSV files matching a glob pattern recursively under *data_dir*."""
    return sorted(data_dir.glob(pattern))


def load_csv(
    csv_path: Path,
    feature_columns: List[str],
    label_column: str,
    label_transform: Optional[Callable] = None,
) -> Tuple[List[List[float]], List[Any]]:
    """
    Load a single CSV into parallel lists ``(features, labels)``.

    Each feature row is a list of floats in *feature_columns* order.
    Labels are left as strings (or transformed by *label_transform*).
    """
    features: List[List[float]] = []
    labels: List[Any] = []

    with open(csv_path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                feat = [float(row[col]) for col in feature_columns]
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed row in %s: %s", csv_path.name, exc)
                continue
            lbl = row.get(label_column, '')
            if label_transform is not None:
                lbl = label_transform(lbl)
            features.append(feat)
            labels.append(lbl)

    return features, labels


def load_all_csvs(
    data_dir: Path,
    csv_glob: str,
    feature_columns: List[str],
    label_column: str,
    label_transform: Optional[Callable] = None,
) -> Tuple[List[List[float]], List[Any]]:
    """Load and merge CSVs from all matching files under *data_dir*."""
    all_feats: List[List[float]] = []
    all_labels: List[Any] = []

    for path in find_csvs(data_dir, csv_glob):
        feats, lbls = load_csv(path, feature_columns, label_column, label_transform)
        all_feats.extend(feats)
        all_labels.extend(lbls)
        logger.info("  Loaded %d rows from %s", len(feats), path.name)

    return all_feats, all_labels


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING CORE
# ═══════════════════════════════════════════════════════════════════════

def _resolve_xgb_params(
    spec: Dict[str, Any],
    config: ModelTrainingConfig,
) -> Dict[str, Any]:
    """Merge per-model XGBoost defaults with user overrides."""
    params = dict(spec['xgb_defaults'])
    if config.max_depth is not None:
        params['max_depth'] = config.max_depth
    if config.learning_rate is not None:
        params['learning_rate'] = config.learning_rate
    if config.n_estimators is not None:
        params['n_estimators'] = config.n_estimators
    return params


def _train_classifier(
    X: "np.ndarray",
    y: "np.ndarray",
    xgb_params: Dict[str, Any],
    config: ModelTrainingConfig,
    *,
    is_binary: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an XGBoost classifier with train/val split.

    Returns ``(model, metrics_dict)``.
    """
    stratify = y if len(set(y.tolist())) > 1 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=config.validation_split,
        random_state=config.random_seed,
        stratify=stratify,
    )

    params: Dict[str, Any] = {
        'max_depth': xgb_params['max_depth'],
        'learning_rate': xgb_params['learning_rate'],
        'n_estimators': xgb_params['n_estimators'],
        'random_state': config.random_seed,
        'use_label_encoder': False,
        'verbosity': 0,
        'early_stopping_rounds': config.early_stopping_rounds,
    }
    if is_binary:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = int(len(set(y.tolist())))

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_pred = model.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average='weighted', zero_division=0)

    metrics = {
        'val_accuracy': round(float(acc), 4),
        'val_f1_weighted': round(float(f1), 4),
        'train_size': int(len(X_tr)),
        'val_size': int(len(X_va)),
        'best_iteration': int(getattr(model, 'best_iteration', xgb_params['n_estimators'])),
    }
    return model, metrics


def _train_regressor(
    X: "np.ndarray",
    y: "np.ndarray",
    xgb_params: Dict[str, Any],
    config: ModelTrainingConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an XGBoost regressor with train/val split.

    Returns ``(model, metrics_dict)``.
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=config.validation_split,
        random_state=config.random_seed,
    )

    model = xgb.XGBRegressor(
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        n_estimators=xgb_params['n_estimators'],
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=config.random_seed,
        verbosity=0,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_pred = model.predict(X_va)
    mse = float(mean_squared_error(y_va, y_pred))
    r2 = float(r2_score(y_va, y_pred))

    metrics = {
        'val_rmse': round(mse ** 0.5, 4),
        'val_r2': round(r2, 4),
        'train_size': int(len(X_tr)),
        'val_size': int(len(X_va)),
        'best_iteration': int(getattr(model, 'best_iteration', xgb_params['n_estimators'])),
    }
    return model, metrics


# ── Cross-validation ───────────────────────────────────────────────────

def _cv_classifier(
    X: "np.ndarray",
    y: "np.ndarray",
    xgb_params: Dict[str, Any],
    config: ModelTrainingConfig,
    *,
    is_binary: bool = False,
) -> Dict[str, Any]:
    """K-fold cross-validation for a classifier.  Returns summary dict."""
    params: Dict[str, Any] = {
        'max_depth': xgb_params['max_depth'],
        'learning_rate': xgb_params['learning_rate'],
        'n_estimators': xgb_params['n_estimators'],
        'random_state': config.random_seed,
        'use_label_encoder': False,
        'verbosity': 0,
    }
    if is_binary:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = int(len(set(y.tolist())))

    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(**params)),
    ])

    n_folds = min(config.n_folds, len(set(y.tolist())))
    if n_folds < 2:
        return {'cv_accuracy_mean': 0.0, 'cv_accuracy_std': 0.0, 'cv_fold_scores': []}

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
    try:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')
    except ValueError:
        # Fallback if some folds have only one class
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')

    return {
        'cv_accuracy_mean': round(float(scores.mean()), 4),
        'cv_accuracy_std': round(float(scores.std()), 4),
        'cv_fold_scores': [round(float(s), 4) for s in scores],
    }


def _cv_regressor(
    X: "np.ndarray",
    y: "np.ndarray",
    xgb_params: Dict[str, Any],
    config: ModelTrainingConfig,
) -> Dict[str, Any]:
    """K-fold cross-validation for a regressor.  Returns summary dict."""
    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', xgb.XGBRegressor(
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            n_estimators=xgb_params['n_estimators'],
            objective='reg:squarederror',
            random_state=config.random_seed,
            verbosity=0,
        )),
    ])

    cv = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='r2')

    return {
        'cv_r2_mean': round(float(scores.mean()), 4),
        'cv_r2_std': round(float(scores.std()), 4),
        'cv_fold_scores': [round(float(s), 4) for s in scores],
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODEL SAVING
# ═══════════════════════════════════════════════════════════════════════

def _save_edgewarden(
    model: Any,
    scaler: Any,
    label_encoder: Any,
    feature_columns: List[str],
    output_dir: Path,
    technologies: List[str],
    metrics: Dict[str, Any],
) -> List[str]:
    """Save EdgeAI model in pipeline-compatible ``edgewarden/`` format.

    Replicates the same model for every requested technology slot so
    that the pipeline's per-technology loader finds its files.
    """
    ew_dir = output_dir / 'edgewarden'
    ew_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for tech in technologies:
        m_path = ew_dir / f"edgewarden_{tech}.pkl"
        s_path = ew_dir / f"scaler_{tech}.pkl"
        with open(m_path, 'wb') as fh:
            pickle.dump(model, fh)
        with open(s_path, 'wb') as fh:
            pickle.dump(scaler, fh)
        saved.extend([str(m_path), str(s_path)])

    # Also save rich metadata for inspection / debugging.
    meta = {
        'feature_columns': feature_columns,
        'label_classes': list(label_encoder.classes_) if label_encoder is not None else [],
        'technologies': technologies,
        'metrics': metrics,
    }
    meta_path = ew_dir / 'training_metadata.json'
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    saved.append(str(meta_path))

    return saved


def _save_pathgnn(
    model: Any,
    scaler: Any,
    feature_columns: List[str],
    output_dir: Path,
    metrics: Dict[str, Any],
) -> List[str]:
    """Save PathGNN scorer (XGBoost binary classifier).

    Always saves ``pathgnn_scorer.pkl``.  If PyTorch is available, also
    creates a minimal ``pathgnn_model.pt`` scaffold so the pipeline's
    GNN loader doesn't crash on missing file.
    """
    pg_dir = output_dir / 'pathgnn'
    pg_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    # Primary model (XGBoost)
    pkl_path = pg_dir / 'pathgnn_scorer.pkl'
    with open(pkl_path, 'wb') as fh:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'model_type': 'xgboost_binary',
        }, fh)
    saved.append(str(pkl_path))

    # Optional torch scaffold for pipeline compat
    if _HAS_TORCH:
        try:
            in_dim = len(feature_columns)
            net = _SimpleMLPScorer(in_dim)
            pt_path = pg_dir / 'pathgnn_model.pt'
            torch.save({
                'model_type': 'SimpleGNN',
                'config': {'in_channels': in_dim, 'out_channels': 2},
                'model_state': net.state_dict(),
                'note': ('Scaffold MLP — use pathgnn_scorer.pkl for the '
                         'XGBoost model trained on real data.'),
            }, pt_path)
            saved.append(str(pt_path))
        except Exception as exc:
            logger.debug("Torch scaffold save skipped: %s", exc)

    meta_path = pg_dir / 'training_metadata.json'
    with open(meta_path, 'w') as fh:
        json.dump({
            'feature_columns': feature_columns,
            'model_type': 'xgboost_binary_classifier',
            'metrics': metrics,
            'note': ('Primary model is pathgnn_scorer.pkl (XGBoost).  '
                     'pathgnn_model.pt is an untrained scaffold for '
                     'pipeline integration.'),
        }, fh, indent=2)
    saved.append(str(meta_path))

    return saved


def _save_pickle_model(
    model: Any,
    scaler: Any,
    label_encoder: Any,
    feature_columns: List[str],
    output_dir: Path,
    subdir: str,
    filename: str,
    metrics: Dict[str, Any],
) -> List[str]:
    """Save model + scaler + label encoder as a single pickle bundle."""
    model_dir = output_dir / subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
    }
    if label_encoder is not None:
        payload['label_encoder'] = label_encoder

    m_path = model_dir / filename
    with open(m_path, 'wb') as fh:
        pickle.dump(payload, fh)

    meta_path = model_dir / 'training_metadata.json'
    meta: Dict[str, Any] = {
        'feature_columns': feature_columns,
        'metrics': metrics,
    }
    if label_encoder is not None:
        meta['label_classes'] = list(label_encoder.classes_)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    return [str(m_path), str(meta_path)]


# ── tiny MLP for PathGNN .pt scaffold ──────────────────────────────────

if _HAS_TORCH:
    class _SimpleMLPScorer(nn.Module):
        """Minimal 2-layer MLP matching SimpleGNN state-dict shape."""

        def __init__(self, in_channels: int = 16, hidden: int = 64,
                     out_channels: int = 2):
            super().__init__()
            self.fc1 = nn.Linear(in_channels, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, out_channels)
            self.relu = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: LOAD A TRAINED MODEL
# ═══════════════════════════════════════════════════════════════════════

def load_trained_model(
    model_path: str | Path,
) -> Dict[str, Any]:
    """Load a model bundle saved by this training runner.

    Returns a dict with at least ``'model'``, ``'scaler'``, and
    ``'feature_columns'`` keys.  Classification models also include
    ``'label_encoder'``.

    Example::

        bundle = load_trained_model("trained_models/diploid/diploid_model.pkl")
        model  = bundle['model']
        scaler = bundle['scaler']
        X_new  = scaler.transform([feature_vector])
        pred   = model.predict(X_new)
    """
    with open(model_path, 'rb') as fh:
        payload = pickle.load(fh)

    # EdgeWarden saves bare model objects (not dicts).
    if not isinstance(payload, dict):
        return {'model': payload, 'scaler': None, 'feature_columns': None}

    return payload


# ═══════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

# Maps model name → (save subdir, save filename).
_SAVE_FILENAMES: Dict[str, Tuple[str, str]] = {
    'diploid_ai': ('diploid', 'diploid_model.pkl'),
    'ul_routing':  ('ul_routing', 'ul_routing_model.pkl'),
    'sv_ai':       ('sv_detector', 'sv_detector_model.pkl'),
}


def train_all_models(config: ModelTrainingConfig) -> Dict[str, Any]:
    """Main entry point.

    Discovers CSV files under *config.data_dir*, trains every requested
    model, runs cross-validation, saves weights, and returns a training
    report dict.
    """
    _check_dependencies()

    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        'config': asdict(config),
        'models': {},
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'status': 'success',
    }

    trained = 0
    skipped = 0
    t_start = time.time()

    for model_name in config.models:
        if model_name not in MODEL_SPECS:
            logger.warning("Unknown model type '%s', skipping", model_name)
            skipped += 1
            continue

        spec = MODEL_SPECS[model_name]
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training: %s — %s", model_name, spec['desc'])
        logger.info("=" * 60)

        # ── 1. Load data ───────────────────────────────────────────
        label_transform = spec.get('label_transform')
        features, labels = load_all_csvs(
            data_dir, spec['csv_glob'],
            spec['features'], spec['label_col'],
            label_transform=label_transform,
        )

        if len(features) < config.min_samples:
            reason = f"insufficient data ({len(features)} rows, need {config.min_samples})"
            logger.warning("  ⊘ Skipping %s: %s", model_name, reason)
            report['models'][model_name] = {'status': 'skipped', 'reason': reason}
            skipped += 1
            continue

        feature_cols = spec['features']
        X_raw = np.array(features, dtype=np.float32)

        # ── 2. Scale features ──────────────────────────────────────
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # ── 3. Prepare labels ──────────────────────────────────────
        label_encoder: Optional[LabelEncoder] = None
        label_dist = Counter(labels)

        if spec['task'] == 'regression':
            y = np.array([float(v) for v in labels], dtype=np.float32)
            logger.info("  Data  : %d samples × %d features", len(y), X.shape[1])
            logger.info("  Target: regression, range [%.3f, %.3f]", y.min(), y.max())
        elif spec['task'] == 'binary':
            y = np.array([int(v) for v in labels], dtype=np.int32)
            logger.info("  Data  : %d samples × %d features", len(y), X.shape[1])
            logger.info("  Labels: %s", dict(label_dist))
        else:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
            logger.info("  Data  : %d samples × %d features", len(y), X.shape[1])
            logger.info("  Labels: %s", dict(label_dist))

        # ── 4. Train ───────────────────────────────────────────────
        xgb_params = _resolve_xgb_params(spec, config)
        n_classes = len(set(y.tolist()))
        is_binary = (spec['task'] == 'binary') or (n_classes == 2)

        if spec['task'] == 'regression':
            model, metrics = _train_regressor(X, y, xgb_params, config)
            cv_metrics = _cv_regressor(X_raw, y, xgb_params, config)
            logger.info("  Val   : RMSE=%.4f  R²=%.4f",
                        metrics['val_rmse'], metrics['val_r2'])
            logger.info("  CV R² : %.4f ± %.4f",
                        cv_metrics['cv_r2_mean'], cv_metrics['cv_r2_std'])
        else:
            model, metrics = _train_classifier(X, y, xgb_params, config, is_binary=is_binary)
            cv_metrics = _cv_classifier(X_raw, y, xgb_params, config, is_binary=is_binary)
            logger.info("  Val   : acc=%.4f  F1=%.4f",
                        metrics['val_accuracy'], metrics['val_f1_weighted'])
            logger.info("  CV acc: %.4f ± %.4f",
                        cv_metrics['cv_accuracy_mean'], cv_metrics['cv_accuracy_std'])

        metrics.update(cv_metrics)
        metrics['label_distribution'] = {str(k): int(v) for k, v in label_dist.items()}

        # ── 5. Save ────────────────────────────────────────────────
        if model_name == 'edge_ai':
            saved = _save_edgewarden(
                model, scaler, label_encoder, feature_cols,
                output_dir, config.edgewarden_technologies, metrics,
            )
        elif model_name == 'path_gnn':
            saved = _save_pathgnn(model, scaler, feature_cols, output_dir, metrics)
        else:
            subdir, fname = _SAVE_FILENAMES.get(model_name, (model_name, f'{model_name}_model.pkl'))
            saved = _save_pickle_model(
                model, scaler, label_encoder, feature_cols,
                output_dir, subdir, fname, metrics,
            )

        report['models'][model_name] = {
            'status': 'trained',
            'metrics': metrics,
            'saved_files': saved,
            'num_samples': len(features),
            'num_features': len(feature_cols),
        }
        trained += 1
        logger.info("  Saved : %d files", len(saved))

    elapsed = round(time.time() - t_start, 1)

    # ── Training report ────────────────────────────────────────────
    report_path = output_dir / 'training_report.json'
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    report['summary'] = {
        'models_trained': trained,
        'models_skipped': skipped,
        'elapsed_seconds': elapsed,
        'report_path': str(report_path),
    }
    return report


def _check_dependencies() -> None:
    """Fail fast with a clear message if required packages are missing."""
    missing: List[str] = []
    if not _HAS_NUMPY:
        missing.append('numpy')
    if not _HAS_XGB:
        missing.append('xgboost')
    if not _HAS_SKLEARN:
        missing.append('scikit-learn')
    if missing:
        raise ImportError(
            f"Model training requires: {', '.join(missing)}.  "
            f"Install with:  pip install {' '.join(missing)}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='strandweaver-train-models',
        description='Train StrandWeaver ML models from graph training CSV data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Train all five model types from generated CSV data
              %(prog)s --data-dir training_output/ --output-dir trained_models/

              # Train only EdgeAI and DiploidAI with custom hyperparameters
              %(prog)s --data-dir training_output/ --output-dir models/ \\
                  --models edge_ai diploid_ai --max-depth 8 --n-estimators 200

              # Quick 3-fold CV with verbose logging
              %(prog)s --data-dir training_output/ --output-dir models/ --n-folds 3 -v

              # End-to-end: generate data then train
              python -m strandweaver.user_training.generate_training_data \\
                  --graph-training -o training_output/ ...
              %(prog)s --data-dir training_output/ --output-dir trained_models/
        """),
    )

    # I/O
    io_grp = parser.add_argument_group('Input / Output')
    io_grp.add_argument(
        '--data-dir', required=True,
        help='Directory containing graph training CSVs (searched recursively)',
    )
    io_grp.add_argument(
        '--output-dir', default='trained_models',
        help='Directory to save trained model weights  (default: trained_models/)',
    )

    # Model selection
    m_grp = parser.add_argument_group('Model Selection')
    m_grp.add_argument(
        '--models', nargs='+',
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        metavar='MODEL',
        help=f"Which models to train (default: all).  Choices: {', '.join(MODEL_SPECS)}",
    )

    # Hyperparameters
    hp_grp = parser.add_argument_group('Hyperparameters')
    hp_grp.add_argument('--max-depth', type=int,
                        help='XGBoost max tree depth (overrides per-model default)')
    hp_grp.add_argument('--learning-rate', type=float,
                        help='XGBoost learning rate')
    hp_grp.add_argument('--n-estimators', type=int,
                        help='Number of XGBoost boosting rounds')
    hp_grp.add_argument('--n-folds', type=int, default=5,
                        help='Cross-validation folds (default: 5)')
    hp_grp.add_argument('--val-split', type=float, default=0.15,
                        help='Hold-out validation fraction (default: 0.15)')
    hp_grp.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # EdgeWarden
    ew_grp = parser.add_argument_group('EdgeWarden Options')
    ew_grp.add_argument(
        '--edgewarden-techs', nargs='+',
        default=EDGEWARDEN_TECHNOLOGIES,
        metavar='TECH',
        help=f'Technology slots for EdgeWarden  (default: {" ".join(EDGEWARDEN_TECHNOLOGIES)})',
    )

    # Misc
    out_grp = parser.add_argument_group('Output Options')
    out_grp.add_argument('-v', '--verbose', action='store_true',
                         help='Verbose (DEBUG) logging')

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    config = ModelTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.models,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        n_folds=args.n_folds,
        validation_split=args.val_split,
        random_seed=args.seed,
        edgewarden_technologies=args.edgewarden_techs,
        verbose=args.verbose,
    )

    # ── Banner ─────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            StrandWeaver  ·  Model Training Runner          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Data dir    : {args.data_dir}")
    print(f"║  Output dir  : {args.output_dir}")
    print(f"║  Models      : {', '.join(args.models)}")
    print(f"║  CV folds    : {args.n_folds}")
    print(f"║  Val split   : {args.val_split}")
    print(f"║  Random seed : {args.seed}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    try:
        report = train_all_models(config)
    except ImportError as exc:
        print(f"\n✗ Missing dependencies: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────
    summary = report.get('summary', {})
    print()
    print("═" * 62)
    print(f"  Training complete in {summary.get('elapsed_seconds', '?')}s")
    print(f"  Models trained : {summary.get('models_trained', 0)}")
    print(f"  Models skipped : {summary.get('models_skipped', 0)}")
    print("═" * 62)

    for name, info in report.get('models', {}).items():
        status = info.get('status', 'unknown')
        if status == 'trained':
            m = info.get('metrics', {})
            if 'val_accuracy' in m:
                print(f"  {name:15s} ✓  acc={m['val_accuracy']:.4f}  "
                      f"f1={m['val_f1_weighted']:.4f}  "
                      f"CV={m.get('cv_accuracy_mean', 0):.4f}"
                      f"±{m.get('cv_accuracy_std', 0):.4f}")
            else:
                print(f"  {name:15s} ✓  RMSE={m.get('val_rmse', 0):.4f}  "
                      f"R²={m.get('val_r2', 0):.4f}  "
                      f"CV R²={m.get('cv_r2_mean', 0):.4f}"
                      f"±{m.get('cv_r2_std', 0):.4f}")
        elif status == 'skipped':
            print(f"  {name:15s} ⊘  {info.get('reason', '')}")

    report_path = summary.get('report_path')
    if report_path:
        print(f"\n  Full report → {report_path}")
    print(f"  Model weights → {args.output_dir}/\n")


if __name__ == '__main__':
    main()

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
