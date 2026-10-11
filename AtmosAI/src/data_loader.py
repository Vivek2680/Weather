"""
src/data_loader.py
═══════════════════════════════════════════════════════════════════════════════
Reads a raw CSV file and returns a standardised, sorted DataFrame.

KEY DESIGN DECISIONS
─────────────────────
1. Column alias resolution — maps raw CSV names to internal standard names
   so all downstream code works with the same names regardless of dataset.

2. Multiple date-format fallbacks — weather CSVs from NOAA, IMD, Kaggle,
   and BMKG all use different date formats.  We try automatic inference
   first, then a list of explicit formats, then pandas "mixed" mode.

3. Structured LoadResult — every problem is returned in a typed object
   instead of raising exceptions or printing to stdout.  The Streamlit
   layer can display these warnings and errors to the user cleanly.

4. No silent data loss — every dropped row is counted and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import (
    ALL_KNOWN_COLUMNS, COL_DATE, COLUMN_ALIASES, REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)

_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%Y/%m/%d", "%d-%b-%Y", "%b %d %Y", "%Y%m%d",
    "%d %b %Y", "%B %d, %Y",
]


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class LoadResult:
    """Carries both data and load-time diagnostics."""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    column_map: Dict[str, str] = field(default_factory=dict)
    available_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)


# ── Public API ─────────────────────────────────────────────────────────────

def load_weather_csv(filepath: Path | str) -> LoadResult:
    """
    Load a historical weather CSV and return a standardised DataFrame.

    Always returns a LoadResult — inspect .success and .errors before using .df
    """
    result = LoadResult()
    filepath = Path(filepath)

    # 1. File checks
    if not filepath.exists():
        result.errors.append(
            f"File not found: {filepath}\n"
            "Place your weather CSV in data/raw/ and name it weather_data.csv"
        )
        return result
    if filepath.stat().st_size == 0:
        result.errors.append(f"File is empty: {filepath}")
        return result

    # 2. Read CSV
    try:
        raw_df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        result.errors.append(f"pandas could not read the CSV: {exc}")
        return result

    if raw_df.empty:
        result.errors.append("CSV parsed successfully but contains zero rows.")
        return result

    logger.info("Read %d rows × %d cols from %s", *raw_df.shape, filepath)

    # 3. Column resolution
    raw_df, column_map, missing = _resolve_columns(raw_df)
    result.column_map = column_map
    result.available_columns = list(column_map.keys())
    result.missing_columns = missing

    for req in REQUIRED_COLUMNS:
        if req not in column_map:
            result.errors.append(
                f"Required column '{req}' not found.\n"
                f"CSV columns: {list(raw_df.columns)}\n"
                "Add an alias to COLUMN_ALIASES in src/config.py"
            )
            return result

    # 4. Date parsing
    raw_df, date_warnings = _parse_dates(raw_df)
    result.warnings.extend(date_warnings)

    if raw_df[COL_DATE].isna().all():
        result.errors.append(
            "No dates could be parsed. Check your date column format and "
            "add a format string to _DATE_FORMATS in data_loader.py"
        )
        return result

    dropped = int(raw_df[COL_DATE].isna().sum())
    if dropped:
        result.warnings.append(f"{dropped} rows with unparseable dates were dropped.")
        raw_df = raw_df.dropna(subset=[COL_DATE])

    # 5. Sort chronologically
    raw_df = raw_df.sort_values(COL_DATE).reset_index(drop=True)

    # 6. Deduplicate dates
    dups = int(raw_df.duplicated(subset=[COL_DATE]).sum())
    if dups:
        result.warnings.append(
            f"{dups} duplicate date(s) found — keeping first occurrence."
        )
        raw_df = raw_df.drop_duplicates(subset=[COL_DATE], keep="first")

    # 7. Coerce numeric columns
    numeric_cols = [c for c in result.available_columns if c != COL_DATE]
    for col in numeric_cols:
        if col not in raw_df.columns:
            continue
        before = int(raw_df[col].isna().sum())
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        after = int(raw_df[col].isna().sum())
        if after - before:
            result.warnings.append(
                f"'{col}': {after - before} non-numeric value(s) → NaN"
            )

    # 8. Dataset size warning
    if len(raw_df) < 365:
        result.warnings.append(
            f"Only {len(raw_df)} rows — recommend ≥ 365 for reliable predictions."
        )

    result.df = raw_df
    result.success = True
    logger.info("Load complete: %d usable rows", len(raw_df))
    return result


def get_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date)."""
    return df[COL_DATE].min(), df[COL_DATE].max()


def get_dataset_summary(df: pd.DataFrame, column_map: Dict[str, str]) -> Dict:
    """Produce a summary dict for display in the dashboard."""
    start, end = get_date_range(df)
    return {
        "total_rows":       len(df),
        "date_range_start": start,
        "date_range_end":   end,
        "date_span_days":   (end - start).days,
        "available_features": list(column_map.keys()),
        "missing_counts":   df.isnull().sum().to_dict(),
        "missing_pct":      (df.isnull().mean() * 100).round(2).to_dict(),
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _resolve_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Map CSV column names → internal standard names via COLUMN_ALIASES."""
    reverse: Dict[str, str] = {}
    for internal, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            reverse[alias.lower().strip()] = internal

    rename_map: Dict[str, str] = {}
    column_map: Dict[str, str] = {}

    for csv_col in df.columns:
        internal = reverse.get(csv_col.lower().strip())
        if internal and internal not in column_map:
            rename_map[csv_col] = internal
            column_map[internal] = csv_col

    df = df.rename(columns=rename_map)
    # Keep only known internal columns (drop unrecognised ones)
    keep = [c for c in df.columns if c in ALL_KNOWN_COLUMNS]
    df = df[keep]

    missing = [c for c in ALL_KNOWN_COLUMNS if c not in column_map]
    logger.debug("Resolved columns: %s", column_map)
    return df, column_map, missing


def _parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convert the date column to pd.Timestamp with multiple fallbacks."""
    warnings_out: List[str] = []
    original = df[COL_DATE].copy()

    # Attempt 1: automatic inference
    try:
        parsed = pd.to_datetime(original, errors="coerce")
        if parsed.notna().sum() > len(parsed) * 0.9:
            df[COL_DATE] = parsed
            return df, warnings_out
    except Exception:
        pass

    # Attempt 2: explicit formats
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(original, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(parsed) * 0.9:
            df[COL_DATE] = parsed
            warnings_out.append(f"Dates parsed using format '{fmt}'")
            return df, warnings_out

    # Attempt 3: mixed mode (slow but permissive)
    df[COL_DATE] = pd.to_datetime(original, format="mixed", errors="coerce")
    warnings_out.append(
        "Dates parsed in 'mixed' mode — verify a sample looks correct."
    )
    return df, warnings_out
