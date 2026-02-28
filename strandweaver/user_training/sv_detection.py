#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

SVScribe: Two-stage SV detection inference module.

Implements the S2 two-stage architecture:
  Stage 1: Binary classifier (SV vs. no SV)  — XGBoost
  Stage 2: Multiclass SV-type classifier     — XGBoost + LightGBM ensemble

Supports three model formats:
  v5: Ensemble bundle  (model_xgb + model_lgbm + blend_weight + class_scales)
  v4: Single model     (model + bio_prior_rules + bio_prior_alpha)
  Legacy: Single .pkl  (sv_detector_model.pkl)

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("strandweaver.sv_detection")

# ═══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════


class SVScribeDetector:
    """Two-stage SV detection with v5 XGBoost + LightGBM ensemble.

    Stage 1: Binary (SV vs. no-SV) — high recall, filters out easy negatives.
    Stage 2: Multiclass (del / ins / inv / dup) — XGBoost + LightGBM
             ensemble with per-class threshold scaling.

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
        # v5: ensemble models
        subtype_model_xgb: Optional[Any] = None,
        subtype_model_lgbm: Optional[Any] = None,
        blend_weight: float = 0.5,
        class_scales: Optional[List[float]] = None,
        # v4 compat: single subtype model
        subtype_model: Optional[Any] = None,
        subtype_scaler: Optional[Any] = None,
        subtype_label_encoder: Optional[Any] = None,
        # Legacy single-model
        single_model: Optional[Any] = None,
        single_scaler: Optional[Any] = None,
        single_label_encoder: Optional[Any] = None,
        # Feature info
        feature_columns: Optional[List[str]] = None,
        subtype_feature_columns: Optional[List[str]] = None,
        binary_threshold: float = 0.5,
        # v4 compat: bio calibration (kept for backwards compat)
        bio_prior_rules: Optional[Dict[str, Dict[str, float]]] = None,
        bio_prior_alpha: float = 0.0,
    ):
        self.binary_model = binary_model
        self.binary_scaler = binary_scaler

        # v5 ensemble
        self.subtype_model_xgb = subtype_model_xgb
        self.subtype_model_lgbm = subtype_model_lgbm
        self.blend_weight = blend_weight
        self.class_scales = (
            np.array(class_scales, dtype=np.float64)
            if class_scales is not None
            else np.ones(4, dtype=np.float64)
        )

        # v4 compat or ensemble primary model
        self.subtype_model = subtype_model
        self.subtype_scaler = subtype_scaler
        self.subtype_label_encoder = subtype_label_encoder

        # Legacy
        self.single_model = single_model
        self.single_scaler = single_scaler
        self.single_label_encoder = single_label_encoder

        # Features
        self.feature_columns = feature_columns
        self.subtype_feature_columns = subtype_feature_columns
        self.binary_threshold = binary_threshold

        # v4 compat
        self.bio_prior_rules = bio_prior_rules
        self.bio_prior_alpha = bio_prior_alpha

        # Determine mode
        self.is_ensemble = (
            subtype_model_xgb is not None and subtype_model_lgbm is not None
        )
        self.is_two_stage = (
            binary_model is not None
            and (self.is_ensemble or subtype_model is not None)
        )

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

        if self.is_ensemble:
            logger.info(
                "  v5 ensemble: XGB(%.2f) + LGBM(%.2f), scales=%s",
                self.blend_weight, 1 - self.blend_weight,
                [f"{s:.2f}" for s in self.class_scales],
            )
        elif self.bio_prior_rules and self.bio_prior_alpha > 0:
            logger.info(
                "  v4 bio calibration active (α=%.3f, %d type rules)",
                self.bio_prior_alpha,
                sum(len(v) for v in self.bio_prior_rules.values()),
            )

    @classmethod
    def load(cls, model_dir: str | Path) -> "SVScribeDetector":
        """Load models from the sv_detector/ directory.

        Supports v5 (ensemble), v4 (bio calibration), and legacy formats.

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
        subtype_model_xgb = subtype_model_lgbm = None
        subtype_model = subtype_scaler = subtype_le = None
        single_model = single_scaler = single_le = None
        feature_columns = None
        subtype_feature_columns = None
        blend_weight = 0.5
        class_scales = None
        bio_prior_rules = None
        bio_prior_alpha = 0.0

        if binary_path.exists() and subtype_path.exists():
            logger.info("Loading two-stage SVScribe models...")

            with open(binary_path, 'rb') as f:
                binary_bundle = pickle.load(f)
            binary_model = binary_bundle['model']
            binary_scaler = binary_bundle.get('scaler')
            feature_columns = binary_bundle.get('feature_columns')
            logger.info(
                "  Stage 1 (binary): loaded from %s (%d features)",
                binary_path.name,
                len(feature_columns) if feature_columns else 0,
            )

            with open(subtype_path, 'rb') as f:
                subtype_bundle = pickle.load(f)

            subtype_scaler = subtype_bundle.get('scaler')
            subtype_le = subtype_bundle.get('label_encoder')
            subtype_feature_columns = subtype_bundle.get('feature_columns')

            # v5 ensemble format
            if 'model_xgb' in subtype_bundle and 'model_lgbm' in subtype_bundle:
                subtype_model_xgb = subtype_bundle['model_xgb']
                subtype_model_lgbm = subtype_bundle['model_lgbm']
                blend_weight = subtype_bundle.get('blend_weight', 0.5)
                class_scales = subtype_bundle.get('class_scales')
                logger.info(
                    "  Stage 2 (v5 ensemble): XGB+LGBM, blend=%.2f, "
                    "%d features",
                    blend_weight,
                    len(subtype_feature_columns) if subtype_feature_columns else 0,
                )
            else:
                # v4 or earlier: single model
                subtype_model = subtype_bundle['model']
                bio_prior_rules = subtype_bundle.get('bio_prior_rules')
                bio_prior_alpha = subtype_bundle.get('bio_prior_alpha', 0.0)
                logger.info(
                    "  Stage 2 (single): loaded from %s (%d features)",
                    subtype_path.name,
                    len(subtype_feature_columns) if subtype_feature_columns else 0,
                )

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
            subtype_model_xgb=subtype_model_xgb,
            subtype_model_lgbm=subtype_model_lgbm,
            blend_weight=blend_weight,
            class_scales=class_scales,
            subtype_model=subtype_model,
            subtype_scaler=subtype_scaler,
            subtype_label_encoder=subtype_le,
            single_model=single_model,
            single_scaler=single_scaler,
            single_label_encoder=single_le,
            feature_columns=feature_columns,
            subtype_feature_columns=subtype_feature_columns,
            bio_prior_rules=bio_prior_rules if binary_model else None,
            bio_prior_alpha=bio_prior_alpha if binary_model else 0.0,
        )

        mode = "two-stage-ensemble" if detector.is_ensemble else (
            "two-stage" if detector.is_two_stage else "single-model"
        )
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
        X = np.asarray(X, dtype=np.float32)
        threshold = binary_threshold or self.binary_threshold

        if self.is_two_stage:
            return self._predict_two_stage(X, threshold)
        elif self.single_model is not None:
            return self._predict_single(X)
        else:
            raise RuntimeError("No SVScribe model loaded")

    # ── v5: Ensemble subtype prediction ───────────────────────────

    def _ensemble_predict_proba(self, X_scaled):
        """Blended XGBoost + LightGBM probability prediction.

        Returns raw blended probabilities (before class_scales).
        """
        xgb_proba = self.subtype_model_xgb.predict_proba(X_scaled)
        lgbm_proba = self.subtype_model_lgbm.predict_proba(X_scaled)
        blended = (
            self.blend_weight * xgb_proba
            + (1 - self.blend_weight) * lgbm_proba
        )
        return blended

    def _ensemble_predict(self, X_scaled):
        """Blended prediction with per-class threshold scaling."""
        blended = self._ensemble_predict_proba(X_scaled)
        adjusted = blended * self.class_scales[np.newaxis, :]
        return adjusted.argmax(axis=1), blended

    # ── v4 compat: Biology-informed prior calibration ─────────────

    def _compute_bio_prior(self, X_scaled, feature_names: List[str]):
        """Compute bio prior matrix from scaled features."""
        type_order = ['deletion', 'insertion', 'inversion', 'duplication']
        feat_idx = {name: i for i, name in enumerate(feature_names)}
        n = X_scaled.shape[0]
        bio = np.zeros((n, 4), dtype=np.float64)

        for t, sv_type in enumerate(type_order):
            rules = self.bio_prior_rules.get(sv_type, {})
            for feat_name, weight in rules.items():
                if feat_name not in feat_idx:
                    continue
                col = X_scaled[:, feat_idx[feat_name]]
                bio[:, t] += weight * np.clip(col, 0, None)

        return bio

    def _apply_bio_calibration(self, model_proba, bio_prior):
        """Bayesian calibration: P_cal ∝ P_model × exp(α × bio_prior)."""
        alpha = self.bio_prior_alpha
        if alpha <= 0 or bio_prior is None:
            return model_proba

        bio_row_max = bio_prior.max(axis=1, keepdims=True)
        bio_centered = bio_prior - bio_row_max

        calibrated = model_proba * np.exp(alpha * bio_centered)
        row_sums = calibrated.sum(axis=1, keepdims=True) + 1e-12
        return calibrated / row_sums

    # ── Two-stage prediction ──────────────────────────────────────

    def _predict_two_stage(self, X, threshold: float) -> List[str]:
        """Two-stage prediction: binary filter → subtype classification."""
        n_samples = X.shape[0]
        results = ['none'] * n_samples

        # Stage 1: Binary SV vs. no-SV (uses full feature set)
        X_scaled = self.binary_scaler.transform(X) if self.binary_scaler else X
        binary_proba = self.binary_model.predict_proba(X_scaled)

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

        X_sv_scaled = (
            self.subtype_scaler.transform(X_sv)
            if self.subtype_scaler else X_sv
        )

        # v5: Ensemble predictions
        if self.is_ensemble:
            subtype_preds, _ = self._ensemble_predict(X_sv_scaled)
        # v4: Bio-calibrated predictions
        elif self.bio_prior_rules and self.bio_prior_alpha > 0:
            feat_names = self.subtype_feature_columns or self.feature_columns
            raw_proba = self.subtype_model.predict_proba(X_sv_scaled)
            bio_prior = self._compute_bio_prior(X_sv_scaled, feat_names)
            cal_proba = self._apply_bio_calibration(raw_proba, bio_prior)
            subtype_preds = cal_proba.argmax(axis=1)
        else:
            subtype_preds = self.subtype_model.predict(X_sv_scaled)

        # Decode labels
        if self.subtype_label_encoder is not None:
            subtype_labels = self.subtype_label_encoder.inverse_transform(
                subtype_preds
            )
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
        X = np.asarray(X, dtype=np.float32)
        result: Dict[str, Any] = {}

        if self.is_two_stage:
            X_scaled = (
                self.binary_scaler.transform(X)
                if self.binary_scaler else X
            )
            binary_proba = self.binary_model.predict_proba(X_scaled)
            result['binary_proba'] = binary_proba

            # v3.1: Select S2-only features
            X_sub = X
            if self._s2_col_indices is not None:
                X_sub = X_sub[:, self._s2_col_indices]
            X_sub = (
                self.subtype_scaler.transform(X_sub)
                if self.subtype_scaler else X_sub
            )

            # v5: Ensemble probabilities
            if self.is_ensemble:
                subtype_proba = self._ensemble_predict_proba(X_sub)
                result['ensemble'] = True
                result['blend_weight'] = self.blend_weight
            # v4: Bio calibration
            elif self.bio_prior_rules and self.bio_prior_alpha > 0:
                subtype_proba = self.subtype_model.predict_proba(X_sub)
                feat_names = (
                    self.subtype_feature_columns or self.feature_columns
                )
                bio_prior = self._compute_bio_prior(X_sub, feat_names)
                subtype_proba = self._apply_bio_calibration(
                    subtype_proba, bio_prior
                )
                result['bio_calibrated'] = True
            else:
                subtype_proba = self.subtype_model.predict_proba(X_sub)

            result['subtype_proba'] = subtype_proba

            if self.subtype_label_encoder is not None:
                result['labels'] = list(self.subtype_label_encoder.classes_)
        elif self.single_model is not None:
            X_scaled = (
                self.single_scaler.transform(X)
                if self.single_scaler else X
            )
            result['proba'] = self.single_model.predict_proba(X_scaled)
            if self.single_label_encoder is not None:
                result['labels'] = list(self.single_label_encoder.classes_)

        return result

    @property
    def mode(self) -> str:
        """Return 'ensemble', 'two_stage', or 'single'."""
        if self.is_ensemble:
            return 'ensemble'
        elif self.is_two_stage:
            return 'two_stage'
        return 'single'


# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
