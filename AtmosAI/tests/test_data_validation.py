"""
tests/test_data_validation.py
────────────────────────────────────────────────────────────────────────────
Unit tests for data_loader and data_validation using small synthetic datasets.

WHY SYNTHETIC DATA IN TESTS?
  • Tests must be fast, deterministic, and require no external files.
  • Synthetic data lets us precisely control edge cases (e.g. exactly one
    duplicate, exactly 25% missing values).
  • We never claim synthetic data is real weather data.

Run with:
    pytest tests/test_data_validation.py -v
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_weather_csv, LoadResult
from src.data_validation import validate_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_csv(df: pd.DataFrame) -> Path:
    """Write a DataFrame to a temp CSV and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return Path(tmp.name)


def _synthetic_df(n: int = 400) -> pd.DataFrame:
    """Generate a minimal synthetic daily weather DataFrame (400 rows ≈ 1 year+)."""
    dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date":      dates.strftime("%Y-%m-%d"),
        "tmax":      25 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tmin":      15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tavg":      20 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 0.5, n),
        "humidity":  rng.uniform(40, 90, n),
        "rainfall":  rng.exponential(2, n),
        "windspeed": rng.uniform(5, 30, n),
    })


# ── data_loader tests ──────────────────────────────────────────────────────

class TestLoadWeatherCsv:

    def test_valid_csv_succeeds(self):
        df = _synthetic_df()
        path = _make_csv(df)
        result = load_weather_csv(path)
        assert result.success is True
        assert not result.df.empty
        assert "date" in result.available_columns

    def test_missing_file_returns_error(self):
        result = load_weather_csv(Path("/nonexistent/path/file.csv"))
        assert result.success is False
        assert any("not found" in e.lower() or "File not found" in e for e in result.errors)

    def test_empty_file_returns_error(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        result = load_weather_csv(empty)
        assert result.success is False

    def test_missing_date_column_returns_error(self, tmp_path):
        df = pd.DataFrame({"temperature": [25, 26, 27]})
        path = tmp_path / "no_date.csv"
        df.to_csv(path, index=False)
        result = load_weather_csv(path)
        assert result.success is False
        assert any("date" in e.lower() for e in result.errors)

    def test_duplicate_dates_are_deduplicated(self, tmp_path):
        df = _synthetic_df(10)
        df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        path = tmp_path / "dup.csv"
        df_dup.to_csv(path, index=False)
        result = load_weather_csv(path)
        assert result.success is True
        # After deduplication, no duplicate dates
        assert result.df["date"].nunique() == len(result.df)

    def test_output_is_sorted_chronologically(self):
        df = _synthetic_df(100)
        df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        path = _make_csv(df_shuffled)
        result = load_weather_csv(path)
        assert result.success is True
        dates = pd.to_datetime(result.df["date"])
        assert dates.is_monotonic_increasing

    def test_non_numeric_values_become_nan(self, tmp_path):
        # Write the CSV directly as text so the "N/A" string appears in the file
        # without triggering pandas dtype incompatibility warnings in the test itself.
        df = _synthetic_df(50)
        csv_text = df.to_csv(index=False)
        rows = csv_text.split("\n")
        # Row index 6 = row 5 (0-indexed) of data (row 0 is header, row 1 is first data)
        if len(rows) > 6:
            parts = rows[6].split(",")
            if len(parts) > 1:
                parts[1] = "CORRUPT_NUM"   # replace tmax value with non-numeric
            rows[6] = ",".join(parts)
        bad_csv = "\n".join(rows)
        path = tmp_path / "bad_numeric.csv"
        path.write_text(bad_csv)
        result = load_weather_csv(path)
        assert result.success is True
        assert any("non-numeric" in w.lower() or "NaN" in w for w in result.warnings)

    def test_column_aliases_resolved(self, tmp_path):
        """Test that a CSV with 'MaxTemp' is mapped to internal name 'tmax'."""
        df = pd.DataFrame({
            "date": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "MaxTemp": [30, 31, 29],
            "MinTemp": [20, 21, 19],
        })
        path = tmp_path / "alias.csv"
        df.to_csv(path, index=False)
        result = load_weather_csv(path)
        assert result.success is True
        assert "tmax" in result.available_columns
        assert "tmin" in result.available_columns

    def test_various_date_formats(self, tmp_path):
        """Test that common date formats are correctly parsed."""
        for fmt, sample in [
            ("%Y-%m-%d", "2022-01-15"),
            ("%d/%m/%Y", "15/01/2022"),
            ("%d-%m-%Y", "15-01-2022"),
        ]:
            df = pd.DataFrame({
                "date": pd.date_range("2022-01-01", periods=50).strftime(fmt),
                "tmax": np.random.uniform(20, 35, 50),
            })
            path = tmp_path / f"dates_{fmt.replace('%','').replace('/','_').replace('-','_')}.csv"
            df.to_csv(path, index=False)
            result = load_weather_csv(path)
            assert result.success is True, f"Failed for format {fmt}: {result.errors}"


# ── data_validation tests ──────────────────────────────────────────────────

class TestValidateDataframe:

    def test_valid_data_passes(self):
        df = _synthetic_df()
        result = load_weather_csv(_make_csv(df))
        loaded_df = result.df.copy()
        loaded_df["date"] = pd.to_datetime(loaded_df["date"])
        report = validate_dataframe(loaded_df, result.available_columns)
        assert report.is_valid is True

    def test_empty_dataframe_is_invalid(self):
        report = validate_dataframe(pd.DataFrame(), [])
        assert report.is_valid is False

    def test_too_few_rows_invalid(self):
        df = _synthetic_df(10)
        result = load_weather_csv(_make_csv(df))
        loaded_df = result.df.copy()
        loaded_df["date"] = pd.to_datetime(loaded_df["date"])
        report = validate_dataframe(loaded_df, result.available_columns)
        assert report.is_valid is False

    def test_outlier_detection(self):
        df = _synthetic_df(400)
        result = load_weather_csv(_make_csv(df))
        loaded_df = result.df.copy()
        loaded_df["date"] = pd.to_datetime(loaded_df["date"])
        loaded_df.loc[0, "tmax"] = 999.0  # physically impossible
        report = validate_dataframe(loaded_df, result.available_columns)
        assert report.outlier_counts.get("tmax", 0) > 0

    def test_missing_value_warning_triggered(self):
        df = _synthetic_df(400)
        # Set > 20% of tmax to NaN
        df.loc[:100, "tmax"] = np.nan
        result = load_weather_csv(_make_csv(df))
        loaded_df = result.df.copy()
        loaded_df["date"] = pd.to_datetime(loaded_df["date"])
        report = validate_dataframe(loaded_df, result.available_columns)
        assert any("tmax" in w for w in report.warnings)

    def test_temperature_consistency_check(self):
        df = _synthetic_df(400)
        result = load_weather_csv(_make_csv(df))
        loaded_df = result.df.copy()
        loaded_df["date"] = pd.to_datetime(loaded_df["date"])
        # Make tmax < tmin for several rows
        loaded_df.loc[0, "tmax"] = 5.0
        loaded_df.loc[0, "tmin"] = 25.0
        report = validate_dataframe(loaded_df, result.available_columns)
        assert any("tmax < tmin" in w for w in report.warnings)
