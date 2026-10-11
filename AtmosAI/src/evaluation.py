"""
src/evaluation.py
═══════════════════════════════════════════════════════════════════════════════
Computes and formats ML evaluation metrics for both regression and
classification tasks.

METRIC INTERPRETATION GUIDE
────────────────────────────
MAE  (Mean Absolute Error): average prediction error in °C.
     Easy to interpret. Outlier-robust. MAE = 2.0 means predictions are
     off by 2°C on average.

RMSE (Root Mean Squared Error): penalises large errors more than MAE.
     If RMSE >> MAE, the model makes occasional large errors.
     Compare RMSE to MAE, not to MAE from a different task.

R²   (coefficient of determination): fraction of target variance explained.
     R² = 1.0: perfect. R² = 0: model is no better than predicting the mean.
     R² < 0: model is worse than predicting the mean (a bad sign).

For classification:
Precision: of predicted rain events, how many were real?
Recall:    of real rain events, how many did we catch?
F1:        harmonic mean of precision and recall.
ROC-AUC:   area under the ROC curve (1.0 = perfect, 0.5 = random).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ── Regression ─────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Return MAE, RMSE, R² for a regression prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model":  model_name,
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":     round(float(r2_score(y_true, y_pred)), 4),
        "n":      len(y_true),
    }


def persistence_baseline(
    y_true: np.ndarray | pd.Series,
    y_true_shifted: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """
    Compute metrics for the persistence baseline:
    "predict tomorrow = today's observed value".

    y_true         — tomorrow's actual temperature (test set)
    y_true_shifted — today's actual temperature (used as prediction)
    """
    return regression_metrics(y_true, y_true_shifted, model_name="Persistence Baseline")


def compare_regression_models(
    results: List[Dict[str, float]],
) -> pd.DataFrame:
    """
    Accept a list of regression_metrics dicts and return a sorted DataFrame.
    Best model = lowest MAE on the validation set.
    """
    df = pd.DataFrame(results)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_values("MAE")


# ── Classification ─────────────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: Optional[np.ndarray | pd.Series] = None,
    model_name: str = "Model",
) -> Dict:
    """Return precision, recall, F1, and optionally ROC-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics: Dict = {
        "model":     model_name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)),        4),
        "n":         len(y_true),
        "rain_days": int(y_true.sum()),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(
                float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float))), 4
            )
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def compare_classification_models(results: List[Dict]) -> pd.DataFrame:
    """Return a sorted comparison DataFrame by F1 score."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "confusion_matrix"}
        rows.append(row)
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_values("f1", ascending=False)


def majority_class_baseline(
    y_true: np.ndarray | pd.Series,
) -> Dict:
    """
    Predict the majority class for every sample.
    This is the simplest possible baseline for classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    return classification_metrics(
        y_true, y_pred, model_name="Majority Class Baseline"
    )
