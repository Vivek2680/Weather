"""
src/feature_engineering.py
═══════════════════════════════════════════════════════════════════════════════
Builds ML-ready feature matrices from the preprocessed DataFrame.

NO LEAKAGE RULE — every feature must represent information available at
the forecast origin (the moment the prediction is being made).

  ✓ df[temp].shift(1)          — yesterday's temperature (known)
  ✓ df[temp].shift(1).rolling(7).mean() — 7-day avg ending YESTERDAY (known)
  ✗ df[temp].rolling(7).mean() — 7-day avg including TODAY'S value (leakage)
  ✗ df[humidity]               — today's humidity as predictor for today's temp

The target is tomorrow's temperature: df[temp].shift(-1).
Row i then contains "all features from today" → "tomorrow's actual value".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COL_HUMIDITY, COL_PRESSURE, COL_RAINFALL,
    COL_TAVG, COL_TMAX, COL_TMIN, COL_WINDSPEED,
    LAG_DAYS, ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)

# India meteorological seasons (month → season name)
_SEASON_MAP: Dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer",  4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}
_SEASON_IDX: Dict[int, int] = {
    12: 0, 1: 0, 2: 0,
    3: 1,  4: 1, 5: 1,
    6: 2,  7: 2, 8: 2, 9: 2,
    10: 3, 11: 3,
}


def build_temperature_features(
    df: pd.DataFrame,
    available_columns: List[str],
    target_col: str = COL_TMAX,
) -> Tuple[pd.DataFrame, str]:
    """
    Build the feature matrix for next-day temperature prediction.

    Parameters
    ----------
    df : preprocessed DataFrame indexed by date
    available_columns : internal column names present in df
    target_col : which temperature to predict (default: tmax)

    Returns
    -------
    features_df : DataFrame with features + 'target' column
    target_col_name : always 'target'
    """
    # Validate target column exists; fall back gracefully
    if target_col not in df.columns:
        for fallback in [COL_TMAX, COL_TAVG, COL_TMIN]:
            if fallback in df.columns:
                logger.warning("Target '%s' not found; falling back to '%s'",
                               target_col, fallback)
                target_col = fallback
                break
        else:
            raise ValueError(
                "No temperature column found. Need at least one of: "
                "tmax, tavg, tmin."
            )

    feat = pd.DataFrame(index=df.index)

    # ── Calendar features (always safe — deterministic from date) ──────────
    feat["day_of_year"] = df.index.dayofyear
    feat["month"]       = df.index.month
    feat["season_idx"]  = df.index.month.map(_SEASON_IDX)
    # Cyclic encoding prevents the model from treating Dec→Jan as a large jump
    feat["sin_doy"] = np.sin(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["sin_mon"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["cos_mon"] = np.cos(2 * np.pi * feat["month"] / 12)

    # ── Lagged temperature (shift(n) gives value from n days ago) ──────────
    for lag in LAG_DAYS:
        feat[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    # ── Leakage-safe rolling average ───────────────────────────────────────
    # shift(1) before rolling() ensures window ends at t-1, never at t
    shifted = df[target_col].shift(1)
    for w in ROLLING_WINDOWS:
        feat[f"{target_col}_roll{w}"] = (
            shifted.rolling(window=w, min_periods=max(1, w // 2)).mean()
        )

    # ── Rolling std (temperature variability) ─────────────────────────────
    feat[f"{target_col}_std7"] = (
        shifted.rolling(window=7, min_periods=3).std()
    )

    # ── Auxiliary feature lags (only if column is available) ───────────────
    aux_cols = [COL_HUMIDITY, COL_RAINFALL, COL_WINDSPEED, COL_PRESSURE]
    for col in aux_cols:
        if col in df.columns and col in available_columns:
            feat[f"{col}_lag1"] = df[col].shift(1)
            feat[f"{col}_lag3"] = df[col].shift(3)

    # ── Other temperature columns as lag-1 features ────────────────────────
    for col in [COL_TMAX, COL_TMIN, COL_TAVG]:
        if col != target_col and col in df.columns and col in available_columns:
            feat[f"{col}_lag1"] = df[col].shift(1)

    # ── Target: NEXT day's temperature ────────────────────────────────────
    # shift(-1) moves tomorrow's value into today's row
    feat["target"] = df[target_col].shift(-1)

    # Drop rows with NaN in features (leading) or target (last row)
    feat = feat.dropna()

    logger.info(
        "Temperature features: %d rows × %d features",
        len(feat), feat.shape[1] - 1,
    )
    return feat, "target"


def build_rainfall_features(
    df: pd.DataFrame,
    available_columns: List[str],
    rain_threshold_mm: float = 0.1,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Build the feature matrix for binary rainfall prediction.

    Target: will TOMORROW have measurable rainfall? (> rain_threshold_mm)
    Returns (None, None) when rainfall column is unavailable.
    """
    if COL_RAINFALL not in df.columns or COL_RAINFALL not in available_columns:
        logger.warning("Rainfall column unavailable — skipping.")
        return None, None

    feat = pd.DataFrame(index=df.index)

    # Calendar
    feat["day_of_year"] = df.index.dayofyear
    feat["month"]       = df.index.month
    feat["season_idx"]  = df.index.month.map(_SEASON_IDX)
    feat["sin_doy"]     = np.sin(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["cos_doy"]     = np.cos(2 * np.pi * feat["day_of_year"] / 365.25)

    # Lagged rainfall
    for lag in LAG_DAYS:
        feat[f"rainfall_lag{lag}"] = df[COL_RAINFALL].shift(lag)

    # Binary lag: was it raining n days ago?
    for lag in [1, 2, 3]:
        feat[f"rained_lag{lag}"] = (
            df[COL_RAINFALL].shift(lag) > rain_threshold_mm
        ).astype(int)

    # Rolling rainfall totals (leakage-safe)
    shifted_rain = df[COL_RAINFALL].shift(1)
    for w in ROLLING_WINDOWS:
        feat[f"rainfall_roll{w}"] = (
            shifted_rain.rolling(window=w, min_periods=max(1, w // 2)).mean()
        )

    # Auxiliary lags
    for col in [COL_TAVG, COL_TMAX, COL_TMIN,
                COL_HUMIDITY, COL_WINDSPEED, COL_PRESSURE]:
        if col in df.columns and col in available_columns:
            feat[f"{col}_lag1"] = df[col].shift(1)

    # Binary target: did measurable rain fall TOMORROW?
    feat["target"] = (df[COL_RAINFALL].shift(-1) > rain_threshold_mm).astype(int)
    feat = feat.dropna()

    rain_days = int(feat["target"].sum())
    total = len(feat)
    logger.info(
        "Rainfall features: %d rows, %d rain days (%.1f%%)",
        total, rain_days, 100 * rain_days / total if total else 0,
    )
    return feat, "target"


def get_feature_names(features_df: pd.DataFrame, target_col: str) -> List[str]:
    """Return feature columns (excludes target)."""
    return [c for c in features_df.columns if c != target_col]


def chronological_split(
    features_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split in chronological order: train → val → test.

    WHY NOT RANDOM?
    ───────────────
    Random shuffling allows the model to train on 2023 data and validate
    on 2020 data.  This is leakage: the model sees the future.  A
    chronological split mirrors real forecasting conditions.
    """
    n = len(features_df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))
    train = features_df.iloc[:i_train]
    val   = features_df.iloc[i_train:i_val]
    test  = features_df.iloc[i_val:]
    logger.info(
        "Split → train:%d  val:%d  test:%d", len(train), len(val), len(test)
    )
    return train, val, test
