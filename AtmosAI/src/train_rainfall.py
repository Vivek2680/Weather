"""
src/train_rainfall.py
═══════════════════════════════════════════════════════════════════════════════
Trains binary classification models for next-day rainfall occurrence.

CLASS IMBALANCE
────────────────
In many locations, rainy days are the minority class (e.g. 30% of days).
If ignored, a model that always predicts "no rain" gets 70% accuracy but
has 0% recall — useless for the actual prediction task.

We address this with:
  • class_weight='balanced' in LogisticRegression and RandomForest
    (up-weights the minority class during training)
  • Reporting F1 + recall (not just accuracy) so the imbalance effect is visible

RAINFALL THRESHOLD
──────────────────
The standard WMO/IMD threshold for "measurable rain" is 0.1 mm.
Any day with rainfall > 0.1 mm is labelled RainTomorrow = 1.
This is documented in the metadata so the user always knows the definition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    MIN_TRAINING_ROWS, RAIN_META_PATH, RAIN_MODEL_PATH,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO,
)
from src.evaluation import (
    classification_metrics, compare_classification_models,
    majority_class_baseline,
)
from src.feature_engineering import (
    build_rainfall_features, chronological_split, get_feature_names,
)

logger = logging.getLogger(__name__)

RAIN_THRESHOLD_MM = 0.1   # WMO/IMD standard for "measurable rainfall"


def train_rainfall_models(
    df: pd.DataFrame,
    available_columns: List[str],
    force_retrain: bool = False,
) -> Optional[Dict]:
    """
    Full rainfall classification pipeline.

    Returns None if rainfall data is unavailable or insufficient.
    """
    # Load existing model if compatible
    if not force_retrain and RAIN_MODEL_PATH.exists() and RAIN_META_PATH.exists():
        existing = _load_model_safe()
        if existing:
            logger.info("Loading existing rainfall model from disk.")
            return existing

    # ── Build features ─────────────────────────────────────────────────────
    feat_df, target_name = build_rainfall_features(df, available_columns, RAIN_THRESHOLD_MM)
    if feat_df is None:
        logger.warning("Rainfall feature construction returned None.")
        return None

    feature_names = get_feature_names(feat_df, target_name)

    if len(feat_df) < MIN_TRAINING_ROWS:
        logger.warning(
            "Too few feature rows (%d) for rainfall model.", len(feat_df)
        )
        return None

    # ── Chronological split ────────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(feat_df, TRAIN_RATIO, VAL_RATIO)

    X_train = train_df[feature_names].values
    y_train = train_df[target_name].values
    X_val   = val_df[feature_names].values
    y_val   = val_df[target_name].values
    X_test  = test_df[feature_names].values
    y_test  = test_df[target_name].values

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline_val = majority_class_baseline(y_val)

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Train models ────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }

    val_results: List[Dict] = [baseline_val]
    trained: Dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred_val = model.predict(X_val_s)
        prob_val = (
            model.predict_proba(X_val_s)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        result = classification_metrics(y_val, pred_val, prob_val, name)
        val_results.append(result)
        trained[name] = model
        logger.info(
            "[VAL] %s  P=%.3f  R=%.3f  F1=%.3f",
            name, result["precision"], result["recall"], result["f1"],
        )

    # ── Select by validation F1 ────────────────────────────────────────────
    ml_val = [r for r in val_results if r["model"] != "Majority Class Baseline"]
    best_name = max(ml_val, key=lambda r: r["f1"])["model"]
    best_model = trained[best_name]
    logger.info("Best rainfall model (by val F1): %s", best_name)

    # ── Test evaluation (ONCE) ─────────────────────────────────────────────
    test_pred = best_model.predict(X_test_s)
    test_prob = (
        best_model.predict_proba(X_test_s)[:, 1]
        if hasattr(best_model, "predict_proba") else None
    )
    test_result = classification_metrics(y_test, test_pred, test_prob, best_name)
    logger.info(
        "[TEST] %s  P=%.3f  R=%.3f  F1=%.3f",
        best_name, test_result["precision"], test_result["recall"], test_result["f1"],
    )

    # ── Persist ────────────────────────────────────────────────────────────
    metadata = {
        "model_type":       best_name,
        "task":             "rainfall_classification",
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "feature_names":    feature_names,
        "train_start":      str(train_df.index.min().date()),
        "train_end":        str(train_df.index.max().date()),
        "test_start":       str(test_df.index.min().date()),
        "test_end":         str(test_df.index.max().date()),
        "val_metrics":      {r["model"]: {k: v for k, v in r.items()
                             if k != "confusion_matrix"} for r in val_results},
        "test_metrics":     {k: v for k, v in test_result.items()
                             if k != "confusion_matrix"},
        "n_train":          len(X_train),
        "n_val":            len(X_val),
        "n_test":           len(X_test),
        "version":          "1.0",
    }

    joblib.dump({"model": best_model, "scaler": scaler}, RAIN_MODEL_PATH)
    joblib.dump(metadata, RAIN_META_PATH)
    logger.info("Rainfall model saved to %s", RAIN_MODEL_PATH)

    return {
        "best_model":      best_model,
        "scaler":          scaler,
        "feature_names":   feature_names,
        "metadata":        metadata,
        "val_results":     val_results,
        "test_result":     test_result,
    }


def _load_model_safe() -> Optional[Dict]:
    try:
        payload  = joblib.load(RAIN_MODEL_PATH)
        metadata = joblib.load(RAIN_META_PATH)
        return {
            "best_model":    payload["model"],
            "scaler":        payload["scaler"],
            "feature_names": metadata["feature_names"],
            "metadata":      metadata,
            "val_results":   list(metadata.get("val_metrics", {}).values()),
            "test_result":   metadata.get("test_metrics", {}),
        }
    except Exception as exc:
        logger.warning("Could not load existing rainfall model: %s", exc)
        return None
