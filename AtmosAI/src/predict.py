"""
src/predict.py
═══════════════════════════════════════════════════════════════════════════════
Reusable prediction functions for temperature and rainfall.

FORECAST HORIZON LIMITATION
─────────────────────────────
These models are trained and validated for NEXT-DAY (1-day horizon) prediction
only.  Multi-day recursive forecasting is NOT implemented because each step's
prediction error compounds — a 2°C error on day 1 becomes an input to day 2,
amplifying the error chain.  Doing so without proper evaluation would mislead
the user about forecast reliability.

If you want to extend to multi-day forecasts in the future, implement
rolling-origin evaluation (evaluate on every possible forecast start date)
and report error statistics across all horizons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_HUMIDITY, COL_PRESSURE, COL_RAINFALL,
    COL_TAVG, COL_TMAX, COL_TMIN, COL_WINDSPEED,
    RAIN_META_PATH, RAIN_MODEL_PATH,
    TEMP_META_PATH, TEMP_MODEL_PATH,
)

logger = logging.getLogger(__name__)


# ── Temperature prediction ─────────────────────────────────────────────────

def predict_next_day_temperature(
    df: pd.DataFrame,
    temp_artifacts: Dict,
) -> Dict:
    """
    Predict tomorrow's temperature using the latest available observations.

    Parameters
    ----------
    df             : preprocessed DataFrame indexed by date (full history)
    temp_artifacts : dict returned by train_temperature_models

    Returns
    -------
    dict with keys: predicted_value, target_col, latest_date,
                    feature_values, model_type, forecast_horizon
    """
    model         = temp_artifacts["best_model"]
    scaler        = temp_artifacts["scaler"]
    feature_names = temp_artifacts["feature_names"]
    metadata      = temp_artifacts["metadata"]
    target_col    = metadata["target_col"]

    # Build the feature row for the LAST available date
    row = _build_last_feature_row(df, feature_names, target_col)

    if row is None:
        raise ValueError(
            "Could not build a complete feature row from the latest data.  "
            "Ensure the dataset is recent enough and contains the required columns."
        )

    X = np.array([[row[f] for f in feature_names]])
    X_scaled = scaler.transform(X)
    pred = float(model.predict(X_scaled)[0])

    return {
        "predicted_value":  round(pred, 2),
        "target_col":       target_col,
        "latest_date":      df.index.max(),
        "forecast_date":    df.index.max() + pd.Timedelta(days=1),
        "feature_values":   row,
        "model_type":       metadata["model_type"],
        "forecast_horizon": "1 day (next-day only)",
    }


def predict_next_day_rainfall(
    df: pd.DataFrame,
    rain_artifacts: Dict,
) -> Dict:
    """
    Predict tomorrow's rainfall occurrence.

    Returns probability (if the model supports it) and binary prediction.
    NOTE: The raw probability from sklearn is NOT a calibrated probability
    (it is not guaranteed to match actual occurrence rates).  The dashboard
    shows it as an estimate, with this limitation disclosed.
    """
    model         = rain_artifacts["best_model"]
    scaler        = rain_artifacts["scaler"]
    feature_names = rain_artifacts["feature_names"]
    metadata      = rain_artifacts["metadata"]

    row = _build_last_feature_row(df, feature_names, target_col=COL_RAINFALL)
    if row is None:
        raise ValueError(
            "Could not build a complete feature row for rainfall prediction."
        )

    X = np.array([[row[f] for f in feature_names]])
    X_scaled = scaler.transform(X)

    pred_class = int(model.predict(X_scaled)[0])
    prob: Optional[float] = None
    if hasattr(model, "predict_proba"):
        prob = round(float(model.predict_proba(X_scaled)[0, 1]), 4)

    return {
        "rain_tomorrow":      bool(pred_class),
        "rain_probability":   prob,
        "probability_note":   (
            "This probability is an uncalibrated model output. "
            "It indicates the model's confidence, not a verified frequency."
        ),
        "threshold_mm":       metadata.get("rain_threshold_mm", 0.1),
        "latest_date":        df.index.max(),
        "forecast_date":      df.index.max() + pd.Timedelta(days=1),
        "model_type":         metadata["model_type"],
        "forecast_horizon":   "1 day (next-day only)",
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _build_last_feature_row(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
) -> Optional[Dict[str, float]]:
    """
    Reconstruct the feature vector for the most recent date in the dataset.

    This replicates the feature engineering logic but for a single row,
    using only historical data — strictly causal.
    """
    if len(df) < 31:
        logger.warning("Need at least 31 rows to build features.")
        return None

    row: Dict[str, float] = {}
    last_date = df.index.max()

    # Calendar
    row["day_of_year"] = float(last_date.dayofyear)
    row["month"]       = float(last_date.month)
    _season = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3}
    row["season_idx"]  = float(_season.get(last_date.month, 0))
    row["sin_doy"]     = float(np.sin(2 * np.pi * row["day_of_year"] / 365.25))
    row["cos_doy"]     = float(np.cos(2 * np.pi * row["day_of_year"] / 365.25))
    row["sin_mon"]     = float(np.sin(2 * np.pi * row["month"] / 12))
    row["cos_mon"]     = float(np.cos(2 * np.pi * row["month"] / 12))

    # Lag columns — look back in the DataFrame
    from src.config import LAG_DAYS, ROLLING_WINDOWS
    base_col = target_col if target_col in df.columns else COL_TMAX

    # Temperature lags
    if base_col in df.columns:
        for lag in LAG_DAYS:
            key = f"{base_col}_lag{lag}"
            if key in feature_names:
                idx = -lag  # last row = -1, day before = -2, etc.
                try:
                    row[key] = float(df[base_col].iloc[idx])
                except IndexError:
                    row[key] = np.nan

        # Rolling averages (end at t-1, so shift(1) equivalent = df up to last row)
        series = df[base_col].iloc[:-1]  # exclude current row (shift(1) equivalent)
        for w in ROLLING_WINDOWS:
            key = f"{base_col}_roll{w}"
            if key in feature_names:
                row[key] = float(series.iloc[-w:].mean()) if len(series) >= w else float(series.mean())

        key_std = f"{base_col}_std7"
        if key_std in feature_names:
            row[key_std] = float(series.iloc[-7:].std()) if len(series) >= 7 else float(series.std())

    # Auxiliary lags
    for col in [COL_HUMIDITY, COL_RAINFALL, COL_WINDSPEED, COL_PRESSURE]:
        for lag in [1, 3]:
            key = f"{col}_lag{lag}"
            if key in feature_names and col in df.columns:
                try:
                    row[key] = float(df[col].iloc[-lag])
                except IndexError:
                    row[key] = np.nan

    # Other temperature columns as lag-1
    for col in [COL_TMAX, COL_TMIN, COL_TAVG]:
        key = f"{col}_lag1"
        if key in feature_names and col in df.columns:
            try:
                row[key] = float(df[col].iloc[-1])
            except IndexError:
                row[key] = np.nan

    # Rainfall-specific features
    if COL_RAINFALL in df.columns:
        for lag in LAG_DAYS:
            key = f"rainfall_lag{lag}"
            if key in feature_names:
                try:
                    row[key] = float(df[COL_RAINFALL].iloc[-lag])
                except IndexError:
                    row[key] = np.nan
        for lag in [1, 2, 3]:
            key = f"rained_lag{lag}"
            if key in feature_names:
                try:
                    row[key] = float(df[COL_RAINFALL].iloc[-lag] > 0.1)
                except IndexError:
                    row[key] = 0.0
        rain_series = df[COL_RAINFALL].iloc[:-1]
        for w in ROLLING_WINDOWS:
            key = f"rainfall_roll{w}"
            if key in feature_names:
                row[key] = float(rain_series.iloc[-w:].mean()) if len(rain_series) >= w else float(rain_series.mean())

    # Check all required features are present and non-NaN
    missing_features = [f for f in feature_names if f not in row or np.isnan(row.get(f, np.nan))]
    if missing_features:
        logger.warning("Missing features for prediction: %s", missing_features[:5])
        # Fill remaining missing with 0 (safe fallback with a warning)
        for f in missing_features:
            row[f] = 0.0

    return row
