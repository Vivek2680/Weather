"""
tests/test_feature_engineering.py
────────────────────────────────────────────────────────────────────────────
Tests for the feature engineering module — focusing on leakage prevention
and correct target construction.

The most important invariant tested here:
  At row i, all lag-n features must contain the value from row i-n.
  This guarantees the model never sees "the future" during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_temperature_features,
    build_rainfall_features,
    chronological_split,
    get_feature_names,
)


# ── Synthetic preprocessed DataFrame ──────────────────────────────────────

def _make_processed_df(n: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "tmax":      30 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tmin":      18 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tavg":      24 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 0.5, n),
        "humidity":  rng.uniform(40, 90, n),
        "rainfall":  np.abs(rng.normal(2, 3, n)),
        "windspeed": rng.uniform(5, 25, n),
    }, index=dates)
    return df


AVAIL_COLS = ["tmax", "tmin", "tavg", "humidity", "rainfall", "windspeed"]


class TestBuildTemperatureFeatures:

    def test_returns_dataframe_with_target(self):
        df = _make_processed_df()
        feat_df, tname = build_temperature_features(df, AVAIL_COLS, "tmax")
        assert isinstance(feat_df, pd.DataFrame)
        assert tname == "target"
        assert "target" in feat_df.columns

    def test_no_nan_in_output(self):
        df = _make_processed_df()
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        assert feat_df.isnull().sum().sum() == 0, "Feature matrix must contain no NaN"

    def test_lag1_feature_is_correct_value(self):
        """tmax_lag1 at row i must equal tmax at row i-1 in the original df."""
        df = _make_processed_df(200)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        # The feature df has fewer rows (leading NaN dropped), so realign by date
        for date in feat_df.index[10:20]:  # sample middle rows
            expected_lag1_date = date - pd.Timedelta(days=1)
            if expected_lag1_date in df.index:
                expected = df.loc[expected_lag1_date, "tmax"]
                actual   = feat_df.loc[date, "tmax_lag1"]
                assert abs(actual - expected) < 1e-9, (
                    f"Lag-1 leakage detected at {date}: "
                    f"expected {expected}, got {actual}"
                )

    def test_target_is_next_day_temperature(self):
        """target at row i must equal tmax at row i+1 in the original df."""
        df = _make_processed_df(200)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        for date in feat_df.index[5:15]:
            next_date = date + pd.Timedelta(days=1)
            if next_date in df.index:
                expected = df.loc[next_date, "tmax"]
                actual   = feat_df.loc[date, "target"]
                assert abs(actual - expected) < 1e-9, (
                    f"Target construction error at {date}: "
                    f"expected {expected}, got {actual}"
                )

    def test_feature_matrix_has_expected_columns(self):
        df = _make_processed_df()
        feat_df, tname = build_temperature_features(df, AVAIL_COLS, "tmax")
        feature_names = get_feature_names(feat_df, tname)
        # Must include at least lag-1
        assert "tmax_lag1" in feature_names
        # Must include rolling features
        assert any("roll" in f for f in feature_names)
        # Must include calendar features
        assert "day_of_year" in feature_names
        assert "month" in feature_names

    def test_fallback_when_target_missing(self):
        """If tmax is missing, should fall back to tavg without raising."""
        df = _make_processed_df()
        df_no_tmax = df.drop(columns=["tmax"])
        avail_no_tmax = [c for c in AVAIL_COLS if c != "tmax"]
        feat_df, tname = build_temperature_features(df_no_tmax, avail_no_tmax, "tmax")
        assert feat_df is not None
        assert len(feat_df) > 0

    def test_rolling_window_ends_at_t_minus_1(self):
        """
        Verify leakage prevention: tmax_roll3 at row i must equal the mean
        of rows i-3, i-2, i-1 (not including row i).
        """
        df = _make_processed_df(200)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        check_date = feat_df.index[40]   # pick a row well away from edges
        d1 = check_date - pd.Timedelta(days=1)
        d2 = check_date - pd.Timedelta(days=2)
        d3 = check_date - pd.Timedelta(days=3)
        if all(d in df.index for d in [d1, d2, d3]):
            expected = df.loc[[d3, d2, d1], "tmax"].mean()
            actual   = feat_df.loc[check_date, "tmax_roll3"]
            assert abs(actual - expected) < 1e-6, (
                f"Rolling-3 leakage: expected {expected}, got {actual}"
            )


class TestBuildRainfallFeatures:

    def test_returns_none_when_rainfall_missing(self):
        df = _make_processed_df()
        df_no_rain = df.drop(columns=["rainfall"])
        avail = [c for c in AVAIL_COLS if c != "rainfall"]
        feat_df, tname = build_rainfall_features(df_no_rain, avail)
        assert feat_df is None
        assert tname is None

    def test_binary_target_construction(self):
        """target must be 1 where TOMORROW's rainfall > 0.1 mm."""
        df = _make_processed_df(200)
        feat_df, tname = build_rainfall_features(df, AVAIL_COLS, rain_threshold_mm=0.1)
        assert feat_df is not None
        for date in feat_df.index[5:15]:
            next_date = date + pd.Timedelta(days=1)
            if next_date in df.index:
                expected_target = int(df.loc[next_date, "rainfall"] > 0.1)
                actual_target   = int(feat_df.loc[date, "target"])
                assert actual_target == expected_target, (
                    f"Rainfall target error at {date}: "
                    f"expected {expected_target}, got {actual_target}"
                )

    def test_target_is_binary(self):
        df = _make_processed_df()
        feat_df, tname = build_rainfall_features(df, AVAIL_COLS)
        assert feat_df is not None
        unique_values = set(feat_df["target"].unique())
        assert unique_values.issubset({0, 1}), f"Non-binary target values: {unique_values}"


class TestChronologicalSplit:

    def test_split_ratios(self):
        df = _make_processed_df(1000)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        train, val, test = chronological_split(feat_df, 0.70, 0.15)
        n = len(feat_df)
        assert abs(len(train)/n - 0.70) < 0.02
        assert abs(len(val)/n   - 0.15) < 0.02

    def test_chronological_order_preserved(self):
        df = _make_processed_df(500)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        train, val, test = chronological_split(feat_df, 0.70, 0.15)
        assert train.index.max() < val.index.min(), "Train bleeds into val"
        assert val.index.max() < test.index.min(), "Val bleeds into test"

    def test_no_overlap_between_splits(self):
        df = _make_processed_df(500)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        train, val, test = chronological_split(feat_df, 0.70, 0.15)
        train_dates = set(train.index)
        val_dates   = set(val.index)
        test_dates  = set(test.index)
        assert len(train_dates & val_dates)  == 0, "Overlap between train and val"
        assert len(train_dates & test_dates) == 0, "Overlap between train and test"
        assert len(val_dates   & test_dates) == 0, "Overlap between val and test"

    def test_covers_all_data(self):
        df = _make_processed_df(500)
        feat_df, _ = build_temperature_features(df, AVAIL_COLS, "tmax")
        train, val, test = chronological_split(feat_df, 0.70, 0.15)
        assert len(train) + len(val) + len(test) == len(feat_df)
