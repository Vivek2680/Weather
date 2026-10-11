"""
src/train_temperature.py
═══════════════════════════════════════════════════════════════════════════════
Trains regression models for next-day temperature prediction.

MODELS COMPARED
───────────────
1. Persistence Baseline — tomorrow = today (free benchmark)
2. Linear Regression    — simplest parametric model; interpretable
3. Ridge Regression     — linear + L2 regularization; handles multicollinear
                          features better than plain OLS (inspired by
                          Adhokshaj04/weather_predict which used Ridge on NOAA)
4. Random Forest        — non-linear ensemble; usually best but less
                          interpretable

MODEL SELECTION STRATEGY
─────────────────────────
• Train all models on the training set.
• Choose the best model using VALIDATION MAE (not test MAE!).
• The test set is only evaluated ONCE for the winning model.
• Never retrain using test labels.

WHY NOT GRID SEARCH?
─────────────────────
GridSearchCV with cross-validation on time series is tricky — standard k-fold
leaks temporal order.  We use TimeSeriesSplit later if needed.  For now,
sensible default hyperparameters are sufficient for a student portfolio.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.config import (
    MIN_TRAINING_ROWS, MODELS_DIR, RANDOM_SEED,
    TEMP_META_PATH, TEMP_MODEL_PATH, TRAIN_RATIO, VAL_RATIO,
)
from src.evaluation import compare_regression_models, regression_metrics
from src.feature_engineering import (
    build_temperature_features, chronological_split, get_feature_names,
)

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────

def train_temperature_models(
    df: pd.DataFrame,
    available_columns: List[str],
    target_col: str = "tmax",
    force_retrain: bool = False,
) -> Dict:
    """
    Full temperature training pipeline.

    Parameters
    ----------
    df               : preprocessed DataFrame indexed by date
    available_columns: internal column names present in df
    target_col       : 'tmax', 'tmin', or 'tavg'
    force_retrain    : if False, return existing model if compatible

    Returns
    -------
    dict with keys:
        best_model, scaler, feature_names, metadata,
        val_results, test_result, all_val_results
    """
    # Load existing model if compatible
    if not force_retrain and TEMP_MODEL_PATH.exists() and TEMP_META_PATH.exists():
        existing = _load_model_safe()
        if existing and existing["metadata"].get("target_col") == target_col:
            logger.info("Loading existing temperature model from disk.")
            return existing

    # ── Build features ─────────────────────────────────────────────────────
    feat_df, target_name = build_temperature_features(df, available_columns, target_col)
    feature_names = get_feature_names(feat_df, target_name)

    if len(feat_df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Only {len(feat_df)} feature rows after removing NaN "
            f"(minimum: {MIN_TRAINING_ROWS}). Add more historical data."
        )

    # ── Chronological split ────────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(feat_df, TRAIN_RATIO, VAL_RATIO)

    X_train = train_df[feature_names].values
    y_train = train_df[target_name].values
    X_val   = val_df[feature_names].values
    y_val   = val_df[target_name].values
    X_test  = test_df[feature_names].values
    y_test  = test_df[target_name].values

    # ── Persistence baseline ────────────────────────────────────────────────
    # "Predict tomorrow = the last lag-1 value" — strictly causal
    lag1_col = f"{target_col}_lag1"
    if lag1_col in feature_names:
        lag1_idx = feature_names.index(lag1_col)
        pers_val_pred  = val_df[feature_names].values[:, lag1_idx]
        pers_test_pred = test_df[feature_names].values[:, lag1_idx]
    else:
        pers_val_pred  = np.full_like(y_val,  float(y_train[-1]))
        pers_test_pred = np.full_like(y_test, float(y_train[-1]))

    pers_val_result = regression_metrics(y_val, pers_val_pred, "Persistence Baseline")

    # ── Scale features ─────────────────────────────────────────────────────
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Train all models ───────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200,
                                 max_depth=10,
                                 min_samples_leaf=5,
                                 random_state=RANDOM_SEED,
                                 n_jobs=-1,
                             ),
    }

    val_results: List[Dict] = [pers_val_result]
    trained: Dict[str, Any] = {}

    for name, model in models.items():
        # Random Forest doesn't benefit from scaling but it's harmless
        model.fit(X_train_s, y_train)
        pred_val = model.predict(X_val_s)
        result = regression_metrics(y_val, pred_val, name)
        val_results.append(result)
        trained[name] = model
        logger.info("[VAL] %s  MAE=%.3f  RMSE=%.3f  R²=%.3f",
                    name, result["MAE"], result["RMSE"], result["R2"])

    # ── Select best model by validation MAE ────────────────────────────────
    ml_val = [r for r in val_results if r["model"] != "Persistence Baseline"]
    best_name = min(ml_val, key=lambda r: r["MAE"])["model"]
    best_model = trained[best_name]
    logger.info("Best model (by val MAE): %s", best_name)

    # ── Evaluate ONCE on untouched test set ────────────────────────────────
    test_pred   = best_model.predict(X_test_s)
    test_result = regression_metrics(y_test, test_pred, best_name)
    pers_test_result = regression_metrics(y_test, pers_test_pred, "Persistence Baseline")
    logger.info("[TEST] %s  MAE=%.3f  RMSE=%.3f  R²=%.3f",
                best_name, test_result["MAE"], test_result["RMSE"], test_result["R2"])

    # ── Feature importances (for dashboard display) ────────────────────────
    importance: Optional[Dict] = None
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feature_names, best_model.feature_importances_))

    # ── Persist model + metadata ────────────────────────────────────────────
    metadata = {
        "model_type":    best_name,
        "target_col":    target_col,
        "feature_names": feature_names,
        "train_start":   str(train_df.index.min().date()),
        "train_end":     str(train_df.index.max().date()),
        "val_start":     str(val_df.index.min().date()),
        "val_end":       str(val_df.index.max().date()),
        "test_start":    str(test_df.index.min().date()),
        "test_end":      str(test_df.index.max().date()),
        "val_metrics":   {r["model"]: r for r in val_results},
        "test_metrics":  {
            best_name:             test_result,
            "Persistence Baseline": pers_test_result,
        },
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_test":        len(X_test),
        "feature_importance": importance,
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_std":    scaler.scale_.tolist(),
        "version":       "1.0",
    }

    _save_model(best_model, scaler, metadata)

    return {
        "best_model":      best_model,
        "scaler":          scaler,
        "feature_names":   feature_names,
        "metadata":        metadata,
        "val_results":     val_results,
        "test_result":     test_result,
        "all_val_results": val_results,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _save_model(model: Any, scaler: StandardScaler, metadata: Dict) -> None:
    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, TEMP_MODEL_PATH)
    joblib.dump(metadata, TEMP_META_PATH)
    logger.info("Temperature model saved to %s", TEMP_MODEL_PATH)


def _load_model_safe() -> Optional[Dict]:
    try:
        payload  = joblib.load(TEMP_MODEL_PATH)
        metadata = joblib.load(TEMP_META_PATH)
        return {
            "best_model":    payload["model"],
            "scaler":        payload["scaler"],
            "feature_names": metadata["feature_names"],
            "metadata":      metadata,
            "val_results":   list(metadata.get("val_metrics", {}).values()),
            "test_result":   list(metadata.get("test_metrics", {}).values())[0]
                             if metadata.get("test_metrics") else {},
            "all_val_results": list(metadata.get("val_metrics", {}).values()),
        }
    except Exception as exc:
        logger.warning("Could not load existing temperature model: %s", exc)
        return None
