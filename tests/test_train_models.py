#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0-dev

Train-models unit tests — MODEL_SPECS structure, ModelTrainingConfig defaults,
CSV loading/writing, XGBoost parameter resolution, and resampling functions.

Tests that require numpy/sklearn/xgboost are skipped gracefully if those
packages are not installed.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial)
"""

import csv
from pathlib import Path

import pytest

from strandweaver.user_training.train_models import (
    EDGEWARDEN_TECHNOLOGIES,
    MODEL_SPECS,
    ModelTrainingConfig,
    _HAS_NUMPY,
    _HAS_SKLEARN,
    _HAS_XGB,
    _resolve_xgb_params,
    find_csvs,
    load_csv,
    load_all_csvs,
)

# Conditional imports for resampling tests
if _HAS_NUMPY:
    import numpy as np
    from strandweaver.user_training.train_models import (
        _compute_sample_weights,
        _hybrid_resample,
        _oversample,
        _undersample,
    )

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
needs_sklearn = pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
needs_xgb = pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")


# ═══════════════════════════════════════════════════════════════════════
#  MODEL_SPECS REGISTRY
# ═══════════════════════════════════════════════════════════════════════

class TestModelSpecs:
    """Validate the MODEL_SPECS constant structure."""

    EXPECTED_MODELS = {
        "edge_ai", "path_gnn", "diploid_ai", "ul_routing", "sv_ai",
        "kweaver_dbg", "kweaver_ul", "kweaver_extension", "kweaver_polish",
        "errorsmith",
    }

    def test_all_models_present(self):
        assert set(MODEL_SPECS.keys()) == self.EXPECTED_MODELS

    @pytest.mark.parametrize("model_name", list(EXPECTED_MODELS))
    def test_required_keys(self, model_name):
        spec = MODEL_SPECS[model_name]
        for key in ("csv_glob", "features", "label_col", "task", "save_subdir", "xgb_defaults", "desc"):
            assert key in spec, f"Missing key '{key}' in MODEL_SPECS['{model_name}']"

    @pytest.mark.parametrize("model_name", list(EXPECTED_MODELS))
    def test_features_non_empty(self, model_name):
        assert len(MODEL_SPECS[model_name]["features"]) > 0

    @pytest.mark.parametrize("model_name", list(EXPECTED_MODELS))
    def test_task_valid(self, model_name):
        assert MODEL_SPECS[model_name]["task"] in ("binary", "multiclass", "regression")

    @pytest.mark.parametrize("model_name", list(EXPECTED_MODELS))
    def test_xgb_defaults_dict(self, model_name):
        defaults = MODEL_SPECS[model_name]["xgb_defaults"]
        assert isinstance(defaults, dict)
        assert "max_depth" in defaults
        assert "learning_rate" in defaults
        assert "n_estimators" in defaults

    def test_diploid_ai_has_advanced_params(self):
        """DiploidAI should have additional regularisation params."""
        params = MODEL_SPECS["diploid_ai"]["xgb_defaults"]
        for key in ("subsample", "colsample_bytree", "min_child_weight", "gamma"):
            assert key in params, f"Missing '{key}' in diploid_ai xgb_defaults"

    def test_edgewarden_technologies_non_empty(self):
        assert len(EDGEWARDEN_TECHNOLOGIES) >= 3


# ═══════════════════════════════════════════════════════════════════════
#  ModelTrainingConfig
# ═══════════════════════════════════════════════════════════════════════

class TestModelTrainingConfig:

    def test_defaults(self):
        cfg = ModelTrainingConfig()
        assert cfg.data_dir == "."
        assert cfg.output_dir == "trained_models"
        assert cfg.validation_split == 0.15
        assert cfg.n_folds == 5
        assert cfg.random_seed == 42
        assert cfg.min_samples == 10

    def test_default_models_match_specs(self):
        cfg = ModelTrainingConfig()
        assert set(cfg.models) == set(MODEL_SPECS.keys())

    def test_default_technologies_match_constant(self):
        cfg = ModelTrainingConfig()
        assert cfg.edgewarden_technologies == EDGEWARDEN_TECHNOLOGIES

    def test_override_hyperparams(self):
        cfg = ModelTrainingConfig(max_depth=10, learning_rate=0.05, n_estimators=200)
        assert cfg.max_depth == 10
        assert cfg.learning_rate == 0.05
        assert cfg.n_estimators == 200

    def test_models_list_independent(self):
        a = ModelTrainingConfig()
        b = ModelTrainingConfig()
        a.models.append("custom")
        assert "custom" not in b.models


# ═══════════════════════════════════════════════════════════════════════
#  XGBoost PARAMETER RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

class TestResolveXGBParams:

    def test_uses_spec_defaults(self):
        spec = MODEL_SPECS["edge_ai"]
        config = ModelTrainingConfig()
        params = _resolve_xgb_params(spec, config)
        assert params["max_depth"] == spec["xgb_defaults"]["max_depth"]
        assert params["learning_rate"] == spec["xgb_defaults"]["learning_rate"]

    def test_cli_overrides_take_precedence(self):
        spec = MODEL_SPECS["edge_ai"]
        config = ModelTrainingConfig(max_depth=20, learning_rate=0.01, n_estimators=500)
        params = _resolve_xgb_params(spec, config)
        assert params["max_depth"] == 20
        assert params["learning_rate"] == 0.01
        assert params["n_estimators"] == 500

    def test_partial_override(self):
        spec = MODEL_SPECS["edge_ai"]
        config = ModelTrainingConfig(max_depth=15)
        params = _resolve_xgb_params(spec, config)
        assert params["max_depth"] == 15
        # Other params should stay at spec defaults
        assert params["learning_rate"] == spec["xgb_defaults"]["learning_rate"]

    def test_preserves_regularisation_keys(self):
        """DiploidAI's advanced params should pass through."""
        spec = MODEL_SPECS["diploid_ai"]
        config = ModelTrainingConfig()
        params = _resolve_xgb_params(spec, config)
        assert "subsample" in params
        assert "colsample_bytree" in params


# ═══════════════════════════════════════════════════════════════════════
#  CSV LOADING
# ═══════════════════════════════════════════════════════════════════════

def _write_csv(path, header, rows):
    """Helper: write a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


class TestFindCSVs:

    def test_finds_matching(self, tmp_path):
        (tmp_path / "edge_ai_training_g0000.csv").touch()
        (tmp_path / "edge_ai_training_g0001.csv").touch()
        (tmp_path / "other.txt").touch()
        found = find_csvs(tmp_path, "edge_ai_training_g*.csv")
        assert len(found) == 2

    def test_no_matches(self, tmp_path):
        found = find_csvs(tmp_path, "nonexistent_*.csv")
        assert len(found) == 0


class TestLoadCSV:

    def test_basic_load(self, tmp_path):
        header = ["feat_a", "feat_b", "label"]
        rows = [
            [1.0, 2.0, "TRUE"],
            [3.0, 4.0, "FALSE"],
        ]
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, header, rows)

        features, labels = load_csv(csv_path, ["feat_a", "feat_b"], "label")
        assert len(features) == 2
        assert len(labels) == 2
        assert features[0] == [1.0, 2.0]
        assert labels[0] == "TRUE"

    def test_skips_malformed_rows(self, tmp_path):
        header = ["feat_a", "feat_b", "label"]
        rows = [
            [1.0, 2.0, "OK"],
            [1.0, "not_a_number", "BAD"],
            [3.0, 4.0, "OK"],
        ]
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, header, rows)

        features, labels = load_csv(csv_path, ["feat_a", "feat_b"], "label")
        assert len(features) == 2  # skipped the malformed row

    def test_with_metadata_columns(self, tmp_path):
        """v2.0 CSVs have metadata columns that should be ignored."""
        header = ["genome_id", "genome_size", "feat_a", "feat_b", "provenance_x", "label"]
        rows = [
            [0, 1000000, 1.0, 2.0, "prov", "TRUE"],
        ]
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, header, rows)

        features, labels = load_csv(csv_path, ["feat_a", "feat_b"], "label")
        assert features == [[1.0, 2.0]]
        assert labels == ["TRUE"]

    def test_label_transform(self, tmp_path):
        header = ["feat_a", "label"]
        rows = [[1.0, "HAP_A"], [2.0, "HAP_B"]]
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, header, rows)

        features, labels = load_csv(
            csv_path, ["feat_a"], "label",
            label_transform=lambda lbl: lbl.replace("HAP_", ""),
        )
        assert labels == ["A", "B"]

    def test_empty_csv(self, tmp_path):
        header = ["feat_a", "label"]
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, header, [])
        features, labels = load_csv(csv_path, ["feat_a"], "label")
        assert features == []
        assert labels == []


class TestLoadAllCSVs:

    def test_merges_multiple_files(self, tmp_path):
        header = ["feat_a", "label"]
        for i in range(3):
            _write_csv(
                tmp_path / f"edge_ai_training_g{i:04d}.csv",
                header,
                [[float(i), f"CLASS_{i}"]],
            )
        features, labels = load_all_csvs(
            tmp_path, "edge_ai_training_g*.csv", ["feat_a"], "label",
        )
        assert len(features) == 3
        assert len(labels) == 3

    def test_no_files(self, tmp_path):
        features, labels = load_all_csvs(
            tmp_path, "missing_*.csv", ["feat_a"], "label",
        )
        assert features == []
        assert labels == []


# ═══════════════════════════════════════════════════════════════════════
#  RESAMPLING FUNCTIONS (require numpy)
# ═══════════════════════════════════════════════════════════════════════

@needs_numpy
class TestUndersample:

    def test_reduces_majority(self):
        # 1000 class-0, 50 class-1
        X = np.vstack([np.ones((1000, 2)), np.zeros((50, 2))])
        y = np.array([0] * 1000 + [1] * 50)
        X_us, y_us = _undersample(X, y, max_per_class=100)
        # Class 0 should be capped at 100, class 1 kept at 50
        assert (y_us == 0).sum() <= 100
        assert (y_us == 1).sum() == 50

    def test_no_change_if_below_threshold(self):
        X = np.ones((30, 2))
        y = np.array([0] * 15 + [1] * 15)
        X_us, y_us = _undersample(X, y, max_per_class=100)
        assert len(y_us) == 30


@needs_numpy
class TestOversample:

    def test_boosts_minority(self):
        X = np.vstack([np.ones((100, 2)), np.zeros((10, 2))])
        y = np.array([0] * 100 + [1] * 10)
        X_os, y_os = _oversample(X, y, target_count=100)
        assert (y_os == 1).sum() >= 100

    def test_default_target_is_median(self):
        X = np.vstack([np.ones((200, 2)), np.zeros((50, 2)), np.ones((100, 2)) * 2])
        y = np.array([0] * 200 + [1] * 50 + [2] * 100)
        X_os, y_os = _oversample(X, y)
        # Median class size = 100, so class-1 (50) should be boosted to ≥100
        assert (y_os == 1).sum() >= 100


@needs_numpy
class TestHybridResample:

    def test_combined_effect(self):
        X = np.vstack([np.ones((5000, 2)), np.zeros((20, 2))])
        y = np.array([0] * 5000 + [1] * 20)
        X_hr, y_hr = _hybrid_resample(X, y, max_majority=200)
        # Majority should be reduced, minority should be boosted
        assert (y_hr == 0).sum() <= 200
        assert (y_hr == 1).sum() > 20

    def test_deterministic(self):
        X = np.vstack([np.ones((500, 2)), np.zeros((10, 2))])
        y = np.array([0] * 500 + [1] * 10)
        rng = np.random.default_rng(99)
        X_a, y_a = _hybrid_resample(X, y, rng=np.random.default_rng(99))
        X_b, y_b = _hybrid_resample(X, y, rng=np.random.default_rng(99))
        assert np.array_equal(y_a, y_b)


@needs_numpy
class TestComputeSampleWeights:

    def test_shape(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        w = _compute_sample_weights(y)
        assert w.shape == y.shape

    def test_minority_gets_higher_weight(self):
        y = np.array([0] * 100 + [1] * 10)
        w = _compute_sample_weights(y)
        # All class-1 samples should have higher weight than class-0
        assert w[100] > w[0]

    def test_capped_at_max_weight(self):
        y = np.array([0] * 10000 + [1] * 1)
        w = _compute_sample_weights(y, max_weight=50.0)
        assert w.max() <= 50.0
