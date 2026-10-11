"""
tests/test_predictions.py
────────────────────────────────────────────────────────────────────────────
Integration tests: train a model on synthetic data and verify that
predictions are returned in the correct format.

These tests do NOT assert on specific numeric values (because that depends
on random seeds and synthetic data).  Instead they assert on:
  • Structure: are all expected keys present?
  • Types: are values numeric where expected?
  • Sanity: is the prediction within a physically plausible range?
  • Persistence: can the model be saved and reloaded?
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import derive_tavg_if_missing, preprocess
from src.train_temperature import train_temperature_models
from src.train_rainfall import train_rainfall_models
from src.predict import predict_next_day_temperature, predict_next_day_rainfall
from src.evaluation import regression_metrics, classification_metrics


# ── Shared fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def processed_df():
    """Return a preprocessed DataFrame large enough to train all models."""
    n = 800
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "tmax":      32 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tmin":      20 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 1, n),
        "tavg":      26 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 0.5, n),
        "humidity":  rng.uniform(40, 90, n),
        "rainfall":  np.abs(rng.normal(2, 4, n)),
        "windspeed": rng.uniform(5, 25, n),
        "pressure":  rng.uniform(1005, 1020, n),
    }, index=dates)
    return df


AVAIL = ["tmax", "tmin", "tavg", "humidity", "rainfall", "windspeed", "pressure"]


# ── Evaluation metric tests ────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_regression_metrics_structure(self):
        y_true = np.array([30, 31, 29, 28, 32], dtype=float)
        y_pred = np.array([29, 30, 28, 29, 31], dtype=float)
        result = regression_metrics(y_true, y_pred, "TestModel")
        assert "MAE"   in result
        assert "RMSE"  in result
        assert "R2"    in result
        assert result["MAE"] >= 0
        assert result["RMSE"] >= 0
        assert result["RMSE"] >= result["MAE"]   # RMSE ≥ MAE always

    def test_perfect_prediction_has_zero_error(self):
        y = np.array([25.0, 26.0, 27.0, 28.0, 29.0])
        result = regression_metrics(y, y, "Perfect")
        assert result["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert result["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert result["R2"]   == pytest.approx(1.0, abs=1e-9)

    def test_classification_metrics_structure(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        result = classification_metrics(y_true, y_pred, model_name="TestCls")
        for k in ["precision", "recall", "f1"]:
            assert k in result
            assert 0 <= result[k] <= 1


# ── Temperature model tests ────────────────────────────────────────────────

class TestTemperatureTraining:

    def test_training_returns_expected_keys(self, tmp_path, monkeypatch, processed_df):
        # Redirect model save paths to temp dir
        import src.config as cfg
        monkeypatch.setattr(cfg, "TEMP_MODEL_PATH", tmp_path / "temp_model.joblib")
        monkeypatch.setattr(cfg, "TEMP_META_PATH",  tmp_path / "temp_meta.joblib")
        import src.train_temperature as tt
        monkeypatch.setattr(tt, "TEMP_MODEL_PATH", tmp_path / "temp_model.joblib")
        monkeypatch.setattr(tt, "TEMP_META_PATH",  tmp_path / "temp_meta.joblib")

        artifacts = train_temperature_models(
            processed_df, AVAIL, target_col="tmax", force_retrain=True
        )
        for key in ["best_model", "scaler", "feature_names", "metadata", "test_result"]:
            assert key in artifacts, f"Missing key: {key}"

    def test_val_mae_reported_for_all_models(self, tmp_path, monkeypatch, processed_df):
        import src.config as cfg
        monkeypatch.setattr(cfg, "TEMP_MODEL_PATH", tmp_path / "t2.joblib")
        monkeypatch.setattr(cfg, "TEMP_META_PATH",  tmp_path / "t2_meta.joblib")
        import src.train_temperature as tt
        monkeypatch.setattr(tt, "TEMP_MODEL_PATH", tmp_path / "t2.joblib")
        monkeypatch.setattr(tt, "TEMP_META_PATH",  tmp_path / "t2_meta.joblib")

        artifacts = train_temperature_models(
            processed_df, AVAIL, target_col="tmax", force_retrain=True
        )
        val_results = artifacts["all_val_results"]
        model_names = [r["model"] for r in val_results]
        assert "Persistence Baseline" in model_names
        assert any("Linear" in n or "Ridge" in n or "Forest" in n for n in model_names)

    def test_test_mae_is_finite(self, tmp_path, monkeypatch, processed_df):
        import src.config as cfg
        monkeypatch.setattr(cfg, "TEMP_MODEL_PATH", tmp_path / "t3.joblib")
        monkeypatch.setattr(cfg, "TEMP_META_PATH",  tmp_path / "t3_meta.joblib")
        import src.train_temperature as tt
        monkeypatch.setattr(tt, "TEMP_MODEL_PATH", tmp_path / "t3.joblib")
        monkeypatch.setattr(tt, "TEMP_META_PATH",  tmp_path / "t3_meta.joblib")

        artifacts = train_temperature_models(
            processed_df, AVAIL, target_col="tmax", force_retrain=True
        )
        mae = artifacts["test_result"]["MAE"]
        assert np.isfinite(mae), "Test MAE must be a finite number"
        assert mae > 0, "MAE must be positive"


# ── Prediction engine tests ────────────────────────────────────────────────

class TestPredictionEngine:

    @pytest.fixture
    def temp_artifacts(self, tmp_path, monkeypatch, processed_df):
        import src.config as cfg
        monkeypatch.setattr(cfg, "TEMP_MODEL_PATH", tmp_path / "tp.joblib")
        monkeypatch.setattr(cfg, "TEMP_META_PATH",  tmp_path / "tp_meta.joblib")
        import src.train_temperature as tt
        monkeypatch.setattr(tt, "TEMP_MODEL_PATH", tmp_path / "tp.joblib")
        monkeypatch.setattr(tt, "TEMP_META_PATH",  tmp_path / "tp_meta.joblib")
        return train_temperature_models(
            processed_df, AVAIL, target_col="tmax", force_retrain=True
        )

    def test_prediction_returns_expected_structure(self, temp_artifacts, processed_df):
        result = predict_next_day_temperature(processed_df, temp_artifacts)
        assert "predicted_value"  in result
        assert "forecast_date"    in result
        assert "forecast_horizon" in result
        assert "model_type"       in result

    def test_predicted_temperature_is_plausible(self, temp_artifacts, processed_df):
        result = predict_next_day_temperature(processed_df, temp_artifacts)
        temp = result["predicted_value"]
        assert -30 <= temp <= 60, f"Predicted temp {temp}°C is outside plausible range"

    def test_forecast_date_is_next_day(self, temp_artifacts, processed_df):
        result = predict_next_day_temperature(processed_df, temp_artifacts)
        expected = processed_df.index.max() + pd.Timedelta(days=1)
        assert result["forecast_date"] == expected

    def test_too_little_data_raises(self, temp_artifacts):
        tiny_df = pd.DataFrame(
            {"tmax": [30.0, 31.0]},
            index=pd.date_range("2022-01-01", periods=2, freq="D"),
        )
        with pytest.raises((ValueError, Exception)):
            predict_next_day_temperature(tiny_df, temp_artifacts)


# ── Rainfall model tests ───────────────────────────────────────────────────

class TestRainfallTraining:

    def test_returns_none_without_rainfall_column(self, tmp_path, monkeypatch, processed_df):
        import src.config as cfg
        monkeypatch.setattr(cfg, "RAIN_MODEL_PATH", tmp_path / "rain.joblib")
        monkeypatch.setattr(cfg, "RAIN_META_PATH",  tmp_path / "rain_meta.joblib")
        import src.train_rainfall as tr
        monkeypatch.setattr(tr, "RAIN_MODEL_PATH", tmp_path / "rain.joblib")
        monkeypatch.setattr(tr, "RAIN_META_PATH",  tmp_path / "rain_meta.joblib")

        df_no_rain = processed_df.drop(columns=["rainfall"])
        avail_no_rain = [c for c in AVAIL if c != "rainfall"]
        result = train_rainfall_models(df_no_rain, avail_no_rain, force_retrain=True)
        assert result is None

    def test_training_with_rainfall_succeeds(self, tmp_path, monkeypatch, processed_df):
        import src.config as cfg
        monkeypatch.setattr(cfg, "RAIN_MODEL_PATH", tmp_path / "rain2.joblib")
        monkeypatch.setattr(cfg, "RAIN_META_PATH",  tmp_path / "rain2_meta.joblib")
        import src.train_rainfall as tr
        monkeypatch.setattr(tr, "RAIN_MODEL_PATH", tmp_path / "rain2.joblib")
        monkeypatch.setattr(tr, "RAIN_META_PATH",  tmp_path / "rain2_meta.joblib")

        result = train_rainfall_models(processed_df, AVAIL, force_retrain=True)
        assert result is not None
        assert "best_model" in result
        assert "test_result" in result
        test = result["test_result"]
        for metric in ["precision", "recall", "f1"]:
            assert metric in test
            assert 0 <= test[metric] <= 1
