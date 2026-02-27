#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

SVScribe: Two-stage SV detection inference module.

Implements the S2 two-stage architecture:
  Stage 1: Binary classifier (SV vs. no SV)
  Stage 2: Multiclass SV-type classifier (del / ins / inv / dup)

Falls back to single-model inference if two-stage models are not available.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("strandweaver.sv_detection")

# ═══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════


class SVScribeDetector:
    """Two-stage SV detection model.

    Stage 1: Binary (SV vs. no-SV) — high recall, filters out easy negatives.
    Stage 2: Multiclass (del / ins / inv / dup) — runs on stage-1 positives.

    Falls back to single-model mode if only ``sv_detector_model.pkl`` exists.

    Usage::

        detector = SVScribeDetector.load("trained_models/sv_detector/")
        predictions = detector.predict(feature_matrix)
        probabilities = detector.predict_proba(feature_matrix)
    """

    def __init__(
        self,
        binary_model: Optional[Any] = None,
        binary_scaler: Optional[Any] = None,
        subtype_model: Optional[Any] = None,
        subtype_scaler: Optional[Any] = None,
        subtype_label_encoder: Optional[Any] = None,
        single_model: Optional[Any] = None,
        single_scaler: Optional[Any] = None,
        single_label_encoder: Optional[Any] = None,
        feature_columns: Optional[List[str]] = None,
        subtype_feature_columns: Optional[List[str]] = None,
        binary_threshold: float = 0.5,
    ):
        self.binary_model = binary_model
        self.binary_scaler = binary_scaler
        self.subtype_model = subtype_model
        self.subtype_scaler = subtype_scaler
        self.subtype_label_encoder = subtype_label_encoder
        self.single_model = single_model
        self.single_scaler = single_scaler
        self.single_label_encoder = single_label_encoder
        self.feature_columns = feature_columns
        self.subtype_feature_columns = subtype_feature_columns
        self.binary_threshold = binary_threshold
        self.is_two_stage = binary_model is not None and subtype_model is not None

        # v3.1: Precompute column index mapping if S2 uses a different feature set
        self._s2_col_indices: Optional[List[int]] = None
        if (self.is_two_stage
                and self.subtype_feature_columns
                and self.feature_columns
                and self.subtype_feature_columns != self.feature_columns):
            col_map = {name: i for i, name in enumerate(self.feature_columns)}
            self._s2_col_indices = [
                col_map[name] for name in self.subtype_feature_columns
                if name in col_map
            ]
            logger.info(
                "  S2 uses %d/%d features (metadata excluded)",
                len(self._s2_col_indices), len(self.feature_columns),
            )

    @classmethod
    def load(cls, model_dir: str | Path) -> "SVScribeDetector":
        """Load models from the sv_detector/ directory.

        Attempts to load two-stage models first. If not found, falls
        back to single-model mode.

        Parameters
        ----------
        model_dir : str or Path
            Path to the ``sv_detector/`` directory.

        Returns
        -------
        SVScribeDetector
            Loaded detector instance.
        """
        model_dir = Path(model_dir)

        # Try two-stage models
        binary_path = model_dir / 'sv_binary_model.pkl'
        subtype_path = model_dir / 'sv_subtype_model.pkl'

        binary_model = binary_scaler = None
        subtype_model = subtype_scaler = subtype_le = None
        single_model = single_scaler = single_le = None
        feature_columns = None
        subtype_feature_columns = None

        if binary_path.exists() and subtype_path.exists():
            logger.info("Loading two-stage SVScribe models...")

            with open(binary_path, 'rb') as f:
                binary_bundle = pickle.load(f)
            binary_model = binary_bundle['model']
            binary_scaler = binary_bundle.get('scaler')
            feature_columns = binary_bundle.get('feature_columns')
            logger.info("  Stage 1 (binary): loaded from %s (%d features)",
                        binary_path.name, len(feature_columns) if feature_columns else 0)

            with open(subtype_path, 'rb') as f:
                subtype_bundle = pickle.load(f)
            subtype_model = subtype_bundle['model']
            subtype_scaler = subtype_bundle.get('scaler')
            subtype_le = subtype_bundle.get('label_encoder')
            # v3.1: S2 may use a different (smaller) feature set
            subtype_feature_columns = subtype_bundle.get('feature_columns')
            logger.info("  Stage 2 (subtype): loaded from %s (%d features)",
                        subtype_path.name,
                        len(subtype_feature_columns) if subtype_feature_columns else 0)

        # Fallback: single model
        single_path = model_dir / 'sv_detector_model.pkl'
        if single_path.exists():
            with open(single_path, 'rb') as f:
                single_bundle = pickle.load(f)
            if isinstance(single_bundle, dict):
                single_model = single_bundle['model']
                single_scaler = single_bundle.get('scaler')
                single_le = single_bundle.get('label_encoder')
                if feature_columns is None:
                    feature_columns = single_bundle.get('feature_columns')
            else:
                single_model = single_bundle
            logger.info("  Single model: loaded from %s", single_path.name)

        detector = cls(
            binary_model=binary_model,
            binary_scaler=binary_scaler,
            subtype_model=subtype_model,
            subtype_scaler=subtype_scaler,
            subtype_label_encoder=subtype_le,
            single_model=single_model,
            single_scaler=single_scaler,
            single_label_encoder=single_le,
            feature_columns=feature_columns,
            subtype_feature_columns=subtype_feature_columns,
        )

        mode = "two-stage" if detector.is_two_stage else "single-model"
        logger.info("SVScribe loaded in %s mode", mode)
        return detector

    def predict(
        self,
        X,
        binary_threshold: Optional[float] = None,
    ) -> List[str]:
        """Predict SV types for input feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (raw, will be scaled internally).
        binary_threshold : float or None
            Threshold for stage-1 binary classifier. Default uses
            self.binary_threshold (0.5).

        Returns
        -------
        list of str
            Predicted SV types: 'none', 'deletion', 'insertion',
            'inversion', 'duplication'.
        """
        import numpy as np

        X = np.asarray(X, dtype=np.float32)
        threshold = binary_threshold or self.binary_threshold

        if self.is_two_stage:
            return self._predict_two_stage(X, threshold)
        elif self.single_model is not None:
            return self._predict_single(X)
        else:
            raise RuntimeError("No SVScribe model loaded")

    def _predict_two_stage(self, X, threshold: float) -> List[str]:
        """Two-stage prediction: binary filter → subtype classification."""
        import numpy as np

        n_samples = X.shape[0]
        results = ['none'] * n_samples

        # Stage 1: Binary SV vs. no-SV (uses full feature set)
        X_scaled = self.binary_scaler.transform(X) if self.binary_scaler else X
        binary_proba = self.binary_model.predict_proba(X_scaled)

        # binary_proba[:, 1] = P(SV)
        if binary_proba.ndim == 2 and binary_proba.shape[1] == 2:
            sv_proba = binary_proba[:, 1]
        else:
            sv_proba = binary_proba.ravel()

        sv_mask = sv_proba >= threshold
        sv_indices = np.where(sv_mask)[0]

        if len(sv_indices) == 0:
            return results

        # Stage 2: Subtype classification on positives only
        X_sv = X[sv_indices]

        # v3.1: Select S2-only features if subtype uses a smaller feature set
        if self._s2_col_indices is not None:
            X_sv = X_sv[:, self._s2_col_indices]

        X_sv_scaled = self.subtype_scaler.transform(X_sv) if self.subtype_scaler else X_sv
        subtype_preds = self.subtype_model.predict(X_sv_scaled)

        # Decode labels
        if self.subtype_label_encoder is not None:
            subtype_labels = self.subtype_label_encoder.inverse_transform(subtype_preds)
        else:
            subtype_labels = [str(p) for p in subtype_preds]

        for idx, label in zip(sv_indices, subtype_labels):
            results[idx] = label

        return results

    def _predict_single(self, X) -> List[str]:
        """Single-model prediction (legacy fallback)."""
        X_scaled = self.single_scaler.transform(X) if self.single_scaler else X
        preds = self.single_model.predict(X_scaled)

        if self.single_label_encoder is not None:
            return list(self.single_label_encoder.inverse_transform(preds))
        return [str(p) for p in preds]

    def predict_proba(self, X) -> Dict[str, Any]:
        """Return prediction probabilities.

        Returns
        -------
        dict with keys:
            'binary_proba': array of P(SV) for each sample (two-stage only)
            'subtype_proba': array of P(type|SV) for SV-positive samples
            'labels': class labels for subtype probabilities
        """
        import numpy as np

        X = np.asarray(X, dtype=np.float32)
        result: Dict[str, Any] = {}

        if self.is_two_stage:
            X_scaled = self.binary_scaler.transform(X) if self.binary_scaler else X
            binary_proba = self.binary_model.predict_proba(X_scaled)
            result['binary_proba'] = binary_proba

            # Subtype probas on all samples (for analysis)
            # v3.1: Select S2-only features if subtype uses a smaller set
            X_sub = X
            if self._s2_col_indices is not None:
                X_sub = X_sub[:, self._s2_col_indices]
            X_sub = self.subtype_scaler.transform(X_sub) if self.subtype_scaler else X_sub
            subtype_proba = self.subtype_model.predict_proba(X_sub)
            result['subtype_proba'] = subtype_proba

            if self.subtype_label_encoder is not None:
                result['labels'] = list(self.subtype_label_encoder.classes_)
        elif self.single_model is not None:
            X_scaled = self.single_scaler.transform(X) if self.single_scaler else X
            result['proba'] = self.single_model.predict_proba(X_scaled)
            if self.single_label_encoder is not None:
                result['labels'] = list(self.single_label_encoder.classes_)

        return result

    @property
    def mode(self) -> str:
        """Return 'two_stage' or 'single'."""
        return 'two_stage' if self.is_two_stage else 'single'


# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
