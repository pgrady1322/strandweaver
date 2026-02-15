#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeAI Class-Imbalance Strategy Comparison
===========================================

Trains EdgeAI (EdgeWarden) under multiple rebalancing strategies on
the existing training_data_10x/ CSVs and prints a head-to-head
comparison with per-class metrics.

Strategies tested
-----------------
1. baseline          – current code (inverse-freq weights, max_weight=50)
2. high_weight       – inverse-freq weights, max_weight=500
3. uncapped_weight   – inverse-freq weights, no cap
4. undersample_maj   – downsample CHIMERIC/SV_BREAK to 100k each
5. oversample_min    – random oversample TRUE/ALLELIC to match median class
6. hybrid            – undersample maj + oversample min
7. binary_true       – binary classifier: TRUE vs everything else

Usage
-----
    cd strandweaver-dev
    python3 scripts/compare_edgeai_strategies.py [--data-dir training_data_10x]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("edgeai_compare")

# ── Feature columns (must match graph_training_data.py v2.0) ──────────
EDGE_AI_FEATURES = [
    "overlap_length", "overlap_identity", "read1_length", "read2_length",
    "coverage_r1", "coverage_r2", "gc_content_r1", "gc_content_r2",
    "repeat_fraction_r1", "repeat_fraction_r2",
    "kmer_diversity_r1", "kmer_diversity_r2",
    "branching_factor_r1", "branching_factor_r2",
    "hic_support", "mapping_quality_r1", "mapping_quality_r2",
    "clustering_coeff_r1", "clustering_coeff_r2", "component_size",
    "entropy_r1", "entropy_r2", "homopolymer_max_r1", "homopolymer_max_r2",
]
LABEL_COL = "label"


# ═════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════

def load_edge_ai_csvs(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all edge_ai CSVs, return (X, labels_str)."""
    csvs = sorted(data_dir.glob("**/edge_ai_training_g*.csv"))
    if not csvs:
        sys.exit(f"No edge_ai CSVs found under {data_dir}")

    all_feats: List[List[float]] = []
    all_labels: List[str] = []
    for p in csvs:
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    feat = [float(row[c]) for c in EDGE_AI_FEATURES]
                except (KeyError, ValueError):
                    continue
                all_feats.append(feat)
                all_labels.append(row[LABEL_COL])

    log.info("Loaded %d rows from %d CSVs", len(all_feats), len(csvs))
    X = np.array(all_feats, dtype=np.float32)
    y_str = np.array(all_labels)
    return X, y_str


# ═════════════════════════════════════════════════════════════════════
#  REBALANCING HELPERS
# ═════════════════════════════════════════════════════════════════════

def compute_sample_weights(
    y: np.ndarray, *, max_weight: float = 50.0
) -> np.ndarray:
    """Inverse-frequency weights, capped at max_weight."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    nc = len(classes)
    cw = {}
    for cls, cnt in zip(classes, counts):
        w = n / (nc * cnt)
        cw[cls] = min(w, max_weight) if max_weight > 0 else w
    return np.array([cw[l] for l in y], dtype=np.float32)


def undersample(X: np.ndarray, y: np.ndarray,
                max_per_class: int = 100_000,
                rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample majority classes to max_per_class, keep minorities intact."""
    if rng is None:
        rng = np.random.default_rng(42)
    indices: List[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        indices.append(idx)
    sel = np.concatenate(indices)
    rng.shuffle(sel)
    return X[sel], y[sel]


def oversample(X: np.ndarray, y: np.ndarray,
               target_count: int | None = None,
               rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Random oversample minority classes up to target_count (default = median class size)."""
    if rng is None:
        rng = np.random.default_rng(42)
    classes, counts = np.unique(y, return_counts=True)
    if target_count is None:
        target_count = int(np.median(counts))
    new_X = [X]
    new_y = [y]
    for cls, cnt in zip(classes, counts):
        if cnt < target_count:
            idx = np.where(y == cls)[0]
            need = target_count - cnt
            extra = rng.choice(idx, need, replace=True)
            new_X.append(X[extra])
            new_y.append(y[extra])
    return np.concatenate(new_X), np.concatenate(new_y)


def hybrid_resample(X: np.ndarray, y: np.ndarray,
                    max_maj: int = 100_000,
                    rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample majority, then oversample minority to meet new median."""
    X2, y2 = undersample(X, y, max_per_class=max_maj, rng=rng)
    return oversample(X2, y2, rng=rng)


# ═════════════════════════════════════════════════════════════════════
#  TRAINING + EVALUATION
# ═════════════════════════════════════════════════════════════════════

XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "verbosity": 0,
    "random_state": 42,
    "early_stopping_rounds": 10,
}

XGB_PARAMS_BINARY = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "verbosity": 0,
    "random_state": 42,
    "early_stopping_rounds": 10,
}


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    is_binary: bool = False,
    strategy_name: str = "",
    n_cv_folds: int = 3,
) -> Dict[str, Any]:
    """Train/val split + k-fold CV, return rich metrics."""

    params = dict(XGB_PARAMS_BINARY if is_binary else XGB_PARAMS)
    if not is_binary:
        params["num_class"] = int(len(np.unique(y)))

    # ── Stratified train/val split ──────────────────────────────
    stratify = y if len(set(y.tolist())) > 1 else None
    if sample_weight is not None:
        X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
            X, y, sample_weight,
            test_size=0.15, random_state=42, stratify=stratify,
        )
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=stratify,
        )
        w_tr = w_va = None

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # ── Train ──────────────────────────────────────────────────
    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    fit_kw: Dict[str, Any] = {"eval_set": [(X_va_s, y_va)], "verbose": False}
    if w_tr is not None:
        fit_kw["sample_weight"] = w_tr
        fit_kw["sample_weight_eval_set"] = [w_va]
    model.fit(X_tr_s, y_tr, **fit_kw)
    train_sec = time.time() - t0

    # ── Val metrics ────────────────────────────────────────────
    y_pred = model.predict(X_va_s)
    acc = accuracy_score(y_va, y_pred)
    f1_w = f1_score(y_va, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_va, y_pred, average="macro", zero_division=0)
    prec_macro = precision_score(y_va, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_va, y_pred, average="macro", zero_division=0)

    # Per-class report
    target_names = label_names or [str(c) for c in sorted(np.unique(np.concatenate([y_va, y_pred])))]
    report = classification_report(
        y_va, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # ── Quick 3-fold CV for variance estimate ──────────────────
    cv_scores = []
    try:
        n_folds = min(n_cv_folds, len(np.unique(y)))
        if n_folds >= 2:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            for tr_idx, te_idx in skf.split(X, y):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr_idx])
                Xte = sc.transform(X[te_idx])
                clf = xgb.XGBClassifier(**{k: v for k, v in params.items()
                                           if k != "early_stopping_rounds"})
                fkw: Dict[str, Any] = {}
                if sample_weight is not None:
                    fkw["sample_weight"] = sample_weight[tr_idx]
                clf.fit(Xtr, y[tr_idx], **fkw)
                yp = clf.predict(Xte)
                cv_scores.append(float(f1_score(y[te_idx], yp, average="macro", zero_division=0)))
    except Exception:
        pass

    result = {
        "strategy": strategy_name,
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "cv_f1_macro_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
        "cv_f1_macro_std": round(float(np.std(cv_scores)), 4) if cv_scores else None,
        "per_class": {},
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_va)),
        "train_seconds": round(train_sec, 1),
        "best_iteration": int(getattr(model, "best_iteration", params.get("n_estimators", 100))),
        "label_distribution_train": {str(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
    }

    for cls_name in target_names:
        if cls_name in report:
            result["per_class"][cls_name] = {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"]),
            }

    return result


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY DEFINITIONS
# ═════════════════════════════════════════════════════════════════════

def run_all_strategies(
    X_raw: np.ndarray,
    y_str: np.ndarray,
) -> List[Dict[str, Any]]:
    """Run all rebalancing strategies and return list of result dicts."""

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    label_names = list(le.classes_)
    results: List[Dict[str, Any]] = []

    rng = np.random.default_rng(42)

    # ── Strategy 1: Baseline (current code) ─────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 1/7: BASELINE  (inverse-freq, max_weight=50)")
    log.info("━" * 65)
    sw = compute_sample_weights(y_enc, max_weight=50.0)
    r = train_and_evaluate(X_raw, y_enc, sample_weight=sw,
                           label_names=label_names,
                           strategy_name="baseline_w50")
    results.append(r)
    _print_summary(r)

    # ── Strategy 2: High weight cap ─────────────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 2/7: HIGH WEIGHT  (inverse-freq, max_weight=500)")
    log.info("━" * 65)
    sw = compute_sample_weights(y_enc, max_weight=500.0)
    r = train_and_evaluate(X_raw, y_enc, sample_weight=sw,
                           label_names=label_names,
                           strategy_name="high_weight_w500")
    results.append(r)
    _print_summary(r)

    # ── Strategy 3: Uncapped weights ────────────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 3/7: UNCAPPED WEIGHT  (inverse-freq, no cap)")
    log.info("━" * 65)
    sw = compute_sample_weights(y_enc, max_weight=0)
    r = train_and_evaluate(X_raw, y_enc, sample_weight=sw,
                           label_names=label_names,
                           strategy_name="uncapped_weight")
    results.append(r)
    _print_summary(r)

    # ── Strategy 4: Undersample majority ────────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 4/7: UNDERSAMPLE MAJORITY  (cap at 100k per class)")
    log.info("━" * 65)
    X_us, y_us = undersample(X_raw, y_enc, max_per_class=100_000, rng=rng)
    log.info("  Resampled: %d → %d samples", len(X_raw), len(X_us))
    r = train_and_evaluate(X_us, y_us, label_names=label_names,
                           strategy_name="undersample_100k")
    results.append(r)
    _print_summary(r)

    # ── Strategy 5: Oversample minority ─────────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 5/7: OVERSAMPLE MINORITY  (up to median class size)")
    log.info("━" * 65)
    X_os, y_os = oversample(X_raw, y_enc, rng=rng)
    log.info("  Resampled: %d → %d samples", len(X_raw), len(X_os))
    r = train_and_evaluate(X_os, y_os, label_names=label_names,
                           strategy_name="oversample_median")
    results.append(r)
    _print_summary(r)

    # ── Strategy 6: Hybrid (undersample + oversample) ───────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 6/7: HYBRID  (undersample maj to 100k + oversample min)")
    log.info("━" * 65)
    X_hy, y_hy = hybrid_resample(X_raw, y_enc, max_maj=100_000, rng=rng)
    log.info("  Resampled: %d → %d samples", len(X_raw), len(X_hy))
    r = train_and_evaluate(X_hy, y_hy, label_names=label_names,
                           strategy_name="hybrid")
    results.append(r)
    _print_summary(r)

    # ── Strategy 7: Binary (TRUE vs rest) ───────────────────────
    log.info("")
    log.info("━" * 65)
    log.info("Strategy 7/7: BINARY  (TRUE vs everything else)")
    log.info("━" * 65)
    true_idx = list(le.classes_).index("TRUE") if "TRUE" in le.classes_ else -1
    y_bin = (y_enc == true_idx).astype(np.int32) if true_idx >= 0 else y_enc
    binary_names = ["NOT_TRUE", "TRUE"]
    sw_bin = compute_sample_weights(y_bin, max_weight=0)
    r = train_and_evaluate(X_raw, y_bin, sample_weight=sw_bin,
                           label_names=binary_names,
                           is_binary=True,
                           strategy_name="binary_true_vs_rest")
    results.append(r)
    _print_summary(r)

    return results


# ═════════════════════════════════════════════════════════════════════
#  PRINTING
# ═════════════════════════════════════════════════════════════════════

def _print_summary(r: Dict[str, Any]) -> None:
    """Print single-strategy summary."""
    log.info("  → acc=%.4f  F1w=%.4f  F1m=%.4f  train=%ds",
             r["accuracy"], r["f1_weighted"], r["f1_macro"], r["train_seconds"])
    if r["cv_f1_macro_mean"] is not None:
        log.info("    CV F1-macro: %.4f ± %.4f", r["cv_f1_macro_mean"], r["cv_f1_macro_std"])
    for cls, m in r["per_class"].items():
        log.info("    %-12s  P=%.3f  R=%.3f  F1=%.3f  n=%d",
                 cls, m["precision"], m["recall"], m["f1"], m["support"])


def print_comparison_table(results: List[Dict[str, Any]], label_names: List[str]) -> None:
    """Print a pretty comparison table to stdout."""
    print()
    print("=" * 100)
    print("  EDGEAI STRATEGY COMPARISON")
    print("=" * 100)

    # Header
    hdr = f"{'Strategy':<25s} {'Acc':>6s} {'F1w':>6s} {'F1m':>6s} {'CVF1m':>8s}"
    for ln in label_names:
        hdr += f" {'F1-' + ln[:6]:>10s}"
    hdr += f" {'Time':>6s} {'Rows':>8s}"
    print(hdr)
    print("─" * len(hdr))

    for r in results:
        line = f"{r['strategy']:<25s} {r['accuracy']:6.4f} {r['f1_weighted']:6.4f} {r['f1_macro']:6.4f}"
        cv = f"{r['cv_f1_macro_mean']:.4f}" if r["cv_f1_macro_mean"] is not None else "  —   "
        line += f" {cv:>8s}"
        for ln in label_names:
            pc = r["per_class"].get(ln, {})
            f1v = pc.get("f1", 0.0)
            line += f" {f1v:10.4f}"
        line += f" {r['train_seconds']:5.0f}s"
        line += f" {r['train_size']:>8d}"
        print(line)

    print("─" * len(hdr))
    print()

    # ── Best-strategy summary ──────────────────────────────────
    best_f1m = max(results, key=lambda r: r["f1_macro"])
    best_acc = max(results, key=lambda r: r["accuracy"])
    print(f"  Best F1-macro:  {best_f1m['strategy']}  ({best_f1m['f1_macro']:.4f})")
    print(f"  Best accuracy:  {best_acc['strategy']}  ({best_acc['accuracy']:.4f})")
    print()

    # ── Per-class best ─────────────────────────────────────────
    multiclass_results = [r for r in results if r["strategy"] != "binary_true_vs_rest"]
    if multiclass_results:
        print("  Per-class best F1 (multiclass strategies only):")
        for ln in label_names:
            best = max(multiclass_results,
                       key=lambda r, _ln=ln: r["per_class"].get(_ln, {}).get("f1", 0.0))
            f1v = best["per_class"].get(ln, {}).get("f1", 0.0)
            print(f"    {ln:<12s}  → {best['strategy']:<25s}  F1={f1v:.4f}")
    print()


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare EdgeAI class-imbalance strategies",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="training_data_10x",
        help="Root directory with training CSVs (default: training_data_10x)",
    )
    parser.add_argument(
        "--output", "-o",
        default="edgeai_comparison_results.json",
        help="Path to save JSON results (default: edgeai_comparison_results.json)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    log.info("Loading EdgeAI training data from %s ...", data_dir)
    X, y_str = load_edge_ai_csvs(data_dir)

    dist = Counter(y_str.tolist())
    log.info("Label distribution:")
    total = sum(dist.values())
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        log.info("  %-12s  %8d  (%.1f%%)", lbl, cnt, cnt / total * 100)

    results = run_all_strategies(X, y_str)

    label_names = sorted(dist.keys())
    print_comparison_table(results, label_names)

    # Save detailed results
    out_path = Path(args.output)
    with open(out_path, "w") as fh:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_dir": str(data_dir),
            "total_samples": int(len(X)),
            "label_distribution": {k: int(v) for k, v in dist.items()},
            "strategies": results,
        }, fh, indent=2)
    log.info("Detailed results saved to %s", out_path)


if __name__ == "__main__":
    main()
