"""
src/data_validation.py
═══════════════════════════════════════════════════════════════════════════════
Post-load validation: detects implausible values, date gaps, and
high-missingness columns — WITHOUT modifying the DataFrame.

WHY SEPARATE VALIDATION FROM PREPROCESSING?
────────────────────────────────────────────
• Validation produces human-readable diagnostic reports for the dashboard.
• Preprocessing modifies data.  Keeping them separate prevents silent fixes
  that hide data quality issues from the user.
• Tests can call either module independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import (
    COL_DATE, COL_TAVG, COL_TMAX, COL_TMIN,
    MISSING_VALUE_WARN_THRESHOLD, VALID_RANGES,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    missing_pct: Dict[str, float] = field(default_factory=dict)
    date_gap_days: List[int] = field(default_factory=list)
    largest_gap_days: int = 0


def validate_dataframe(
    df: pd.DataFrame,
    available_columns: List[str],
) -> ValidationReport:
    """Run all validation checks and return a ValidationReport."""
    report = ValidationReport()
    if df.empty:
        report.is_valid = False
        report.errors.append("DataFrame is empty.")
        return report

    _check_row_count(df, report)
    _check_missing_values(df, available_columns, report)
    _check_date_continuity(df, report)
    _check_plausible_ranges(df, available_columns, report)
    _check_temperature_consistency(df, available_columns, report)
    return report


# ── Checks ─────────────────────────────────────────────────────────────────

def _check_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
    n = len(df)
    if n < 30:
        report.is_valid = False
        report.errors.append(f"Only {n} rows — too few for any analysis.")
    elif n < 365:
        report.warnings.append(
            f"{n} rows (< 1 year). Seasonal patterns may be unreliable."
        )


def _check_missing_values(
    df: pd.DataFrame, available_columns: List[str], report: ValidationReport
) -> None:
    for col in available_columns:
        if col == COL_DATE or col not in df.columns:
            continue
        pct = df[col].isna().mean()
        report.missing_pct[col] = round(pct * 100, 2)
        if pct >= MISSING_VALUE_WARN_THRESHOLD:
            report.warnings.append(
                f"'{col}' is {pct*100:.1f}% missing — predictions using this "
                "feature may be unreliable."
            )


def _check_date_continuity(df: pd.DataFrame, report: ValidationReport) -> None:
    if COL_DATE not in df.columns or len(df) < 2:
        return
    dates = df[COL_DATE].sort_values().dropna()
    gaps = dates.diff().dt.days.dropna()
    large = gaps[gaps > 3]
    if not large.empty:
        report.date_gap_days = sorted([int(g) for g in large], reverse=True)
        report.largest_gap_days = report.date_gap_days[0]
        report.warnings.append(
            f"{len(large)} gap(s) in the date series. "
            f"Largest: {report.largest_gap_days} days. "
            "Gaps can affect lag features and rolling averages."
        )


def _check_plausible_ranges(
    df: pd.DataFrame, available_columns: List[str], report: ValidationReport
) -> None:
    for col in available_columns:
        if col not in VALID_RANGES or col not in df.columns:
            continue
        lo, hi = VALID_RANGES[col]
        s = df[col].dropna()
        n_out = int(((s < lo) | (s > hi)).sum())
        report.outlier_counts[col] = n_out
        if n_out:
            bad = s[(s < lo) | (s > hi)].head(3).tolist()
            report.warnings.append(
                f"'{col}': {n_out} value(s) outside [{lo}, {hi}]. "
                f"Sample: {bad}"
            )


def _check_temperature_consistency(
    df: pd.DataFrame, available_columns: List[str], report: ValidationReport
) -> None:
    has = lambda c: c in available_columns and c in df.columns  # noqa: E731
    if has(COL_TMAX) and has(COL_TMIN):
        bad = int((df[COL_TMAX] < df[COL_TMIN]).sum())
        if bad:
            report.warnings.append(f"{bad} row(s) where tmax < tmin.")
    if has(COL_TMAX) and has(COL_TAVG):
        bad = int((df[COL_TAVG] > df[COL_TMAX] + 1).sum())
        if bad:
            report.warnings.append(f"{bad} row(s) where tavg > tmax.")
    if has(COL_TMIN) and has(COL_TAVG):
        bad = int((df[COL_TAVG] < df[COL_TMIN] - 1).sum())
        if bad:
            report.warnings.append(f"{bad} row(s) where tavg < tmin.")
