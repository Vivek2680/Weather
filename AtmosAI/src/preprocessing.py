"""
src/preprocessing.py
═══════════════════════════════════════════════════════════════════════════════
Transforms the validated DataFrame into a form ready for feature engineering.

LEAKAGE-SAFE IMPUTATION
────────────────────────
We use forward-fill (ffill) + backward-fill (bfill):
  • ffill: carry the last KNOWN value forward — strictly causal.
  • bfill: only used for the very first rows where ffill has nothing to copy.
  • Mean/median imputation is NOT used because it would incorporate future
    observations into past gaps (subtle leakage for time-series data).

When a column is > 50 % missing, imputation is skipped — imputing sparse
data creates fictional observations that mislead the model.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from src.config import COL_DATE, COL_TAVG, COL_TMAX, COL_TMIN, VALID_RANGES

logger = logging.getLogger(__name__)

_MAX_IMPUTE_MISSING = 0.50


def preprocess(
    df: pd.DataFrame,
    available_columns: List[str],
    clip_outliers: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Apply all preprocessing steps.

    Returns
    -------
    clean_df : pd.DataFrame
        Preprocessed DataFrame indexed by date.
    impute_log : dict[str, str]
        Maps column name → strategy used (for display in the dashboard).
    """
    df = df.copy()
    df = df.set_index(COL_DATE).sort_index()
    numeric_cols = [c for c in available_columns if c != COL_DATE and c in df.columns]

    # 1. Clip outliers to valid range
    if clip_outliers:
        for col in numeric_cols:
            if col in VALID_RANGES:
                lo, hi = VALID_RANGES[col]
                n = int(((df[col] < lo) | (df[col] > hi)).sum())
                if n:
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    logger.info("Clipped %d outlier(s) in '%s'", n, col)

    # 2. Impute missing values
    impute_log: Dict[str, str] = {}
    for col in numeric_cols:
        missing_frac = df[col].isna().mean()
        if missing_frac == 0:
            impute_log[col] = "none (complete)"
        elif missing_frac > _MAX_IMPUTE_MISSING:
            impute_log[col] = (
                f"SKIPPED — {missing_frac*100:.1f}% missing (too sparse)"
            )
        else:
            df[col] = df[col].ffill().bfill()
            impute_log[col] = f"ffill+bfill ({missing_frac*100:.1f}% was missing)"

    return df, impute_log


def derive_tavg_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    If tavg is absent but tmax and tmin are present, derive it as
    (tmax + tmin) / 2.  This is a standard meteorological approximation.
    """
    df = df.copy()
    if COL_TAVG not in df.columns:
        if COL_TMAX in df.columns and COL_TMIN in df.columns:
            df[COL_TAVG] = (df[COL_TMAX] + df[COL_TMIN]) / 2.0
            logger.info("Derived tavg = (tmax + tmin) / 2")
    return df
