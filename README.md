# AtmosAI: Weather Prediction and Analysis System

AtmosAI is a machine learning system that predicts next-day maximum temperature and rainfall occurrence from historical daily observations. It runs fully offline, requires no API keys, and is designed to prevent temporal data leakage.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![pytest](https://img.shields.io/badge/pytest-8.0%2B-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Machine Learning Formulation](#machine-learning-formulation)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Dataset Sources](#dataset-sources)
8. [Testing](#testing)
9. [Design Decisions](#design-decisions)
10. [License](#license)

---

## Overview

Many weather applications retrieve forecasts from third-party APIs and display them in a user interface. AtmosAI takes a different approach: it learns patterns directly from historical observations and evaluates its predictions against statistical baselines.

**Design principles**

- **No live weather API.** All predictions are derived from historical data supplied by the user.
- **Temporal integrity.** Data is split chronologically, lag features are built from past values only, and rolling statistics end at day *t-1*. This avoids the leakage that random splits introduce in time-series problems.
- **Baseline benchmarking.** Each model is compared with a naive baseline (persistence for regression, majority class for classification) to show whether it adds real predictive skill.

---

## Key Features

| Capability | Description |
|---|---|
| CSV ingestion | A case-insensitive alias map normalizes 30+ column names (for example `MaxTemp`, `tmax`, `PRCP`, `precipitation`) to a unified schema. |
| Data validation | Detects duplicate dates, missing intervals, statistical outliers (IQR method), and physical inconsistencies such as T<sub>max</sub> < T<sub>min</sub>. |
| Preprocessing | Imputes missing values using historical data only, and derives T<sub>avg</sub> from T<sub>max</sub> and T<sub>min</sub> when absent. |
| Feature engineering | Multi-day lags, backward-looking rolling statistics, and cyclical day-of-year encodings. |
| Temperature prediction | Persistence baseline, Ordinary Least Squares, Ridge Regression, and Random Forest Regressor. |
| Rainfall classification | Majority-class baseline, class-balanced Logistic Regression, and Random Forest Classifier (rain threshold of 0.1 mm). |
| Evaluation | MAE, RMSE, R², precision, recall, F1-score, and confusion matrices on a held-out test set. |
| Model persistence | Trained models, feature schemas, scalers, and metadata are serialized with `joblib` for offline inference. |
| Dashboard | Seven-page Streamlit interface with interactive Matplotlib and Seaborn visualizations. |
| Test suite | 41 unit tests covering data loading, chronological splitting, leakage prevention, and model training. |

---

## System Architecture

```text
 historical_weather.csv
          │
          ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. Ingestion and Validation (src/data_loader.py)       │
 │    - Column alias matching (NOAA, IMD, Kaggle formats) │
 │    - Date normalization and chronological sorting      │
 │    - Anomaly and physical consistency checks           │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Preprocessing and Imputation (src/preprocessing.py) │
 │    - Leakage-safe imputation                           │
 │    - Derivation of Tavg from Tmax/Tmin when omitted    │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Feature Engineering (src/feature_engineering.py)    │
 │    - Lags: t-1, t-2, t-3, t-7                          │
 │    - Rolling mean and std: window [t-7, t-1]           │
 │    - Seasonality: sin(2πd/365), cos(2πd/365)           │
 │    - Chronological split: 70% train / 15% val / 15% test│
 └─────────────┬────────────────────────────┬─────────────┘
               │                            │
               ▼                            ▼
 ┌──────────────────────────┐ ┌──────────────────────────┐
 │ 4a. Temperature Pipeline │ │ 4b. Rainfall Pipeline    │
 │ (src/train_temperature)  │ │ (src/train_rainfall)     │
 │                          │ │                          │
 │ • Persistence baseline   │ │ • Majority baseline      │
 │ • Linear Regression      │ │ • Logistic Regression    │
 │ • Ridge Regression (L2)  │ │ • Random Forest          │
 │ • Random Forest          │ │   (200 est., balanced)   │
 │   (200 estimators)       │ │                          │
 └─────────────┬────────────┘ └─────────────┬────────────┘
               │                            │
               └──────────────┬─────────────┘
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ 5. Inference and Presentation (app.py)                 │
 │    - Next-day forecast                                 │
 │    - Model comparison tables and charts                │
 │    - Fully local execution                             │
 └────────────────────────────────────────────────────────┘
```

---

## Machine Learning Formulation

### Temperature Prediction (Regression)

- **Target:** y<sub>t+1</sub> = T<sub>max</sub>(t+1)
- **Feature vector:**

$$X_t = \left[ T(t), T(t-1), \dots, T(t-7), \mu_{7}(t-1), \sigma_{7}(t-1), \sin\left(\frac{2\pi d}{365}\right), \cos\left(\frac{2\pi d}{365}\right), \text{Humidity}_t, \text{Pressure}_t \right]$$

- **Baseline:** ŷ<sub>persistence</sub> = T<sub>max</sub>(t)
- **Model selection:** lowest Mean Absolute Error (MAE) on the chronological validation set.

### Rainfall Occurrence (Classification)

- **Target:**

$$y_{t+1} = \begin{cases} 1 & \text{if } \text{Rainfall}(t+1) \ge 0.1 \text{ mm} \\ 0 & \text{otherwise} \end{cases}$$

- **Class imbalance:** rainfall datasets are often dominated by dry days (commonly 70–85%). Models use balanced class weights:

$$w_c = \frac{N}{2 \cdot N_c}$$

- **Model selection:** highest F1-score on the chronological validation set.

---

## Project Structure

```text
AtmosAI/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .gitignore
│
├── data/
│   ├── raw/                    # Input CSVs (e.g. weather_data.csv)
│   └── processed/              # Cached feature sets
│
├── models/                     # Serialized models and metadata
│   ├── temperature_model.joblib
│   └── rainfall_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, column aliases, constants, hyperparameters
│   ├── data_loader.py          # CSV loading with alias mapping
│   ├── data_validation.py      # Outlier detection and sanity checks
│   ├── preprocessing.py        # Imputation and sanitization
│   ├── feature_engineering.py  # Lag/rolling features and chronological splits
│   ├── data_analysis.py        # Visualizations (heatmaps, seasonality, YoY)
│   ├── train_temperature.py    # Regression training pipeline
│   ├── train_rainfall.py       # Classification training pipeline
│   ├── evaluation.py           # MAE, RMSE, R², F1, confusion matrix
│   └── predict.py              # Single-step inference
│
└── tests/
    ├── __init__.py
    ├── test_data_validation.py
    ├── test_feature_engineering.py
    └── test_predictions.py
```

---

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- `pip` and `venv`

### Installation

```bash
git clone https://github.com/Vivek2680/AtmosAI.git
cd AtmosAI

python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Dataset Sources

AtmosAI accepts any standard daily weather CSV. The following sources are free and require no API key:

1. [Open-Meteo Historical Archive](https://open-meteo.com/en/docs/historical-weather-api): select a location, choose daily variables (max/min temperature, precipitation, relative humidity, wind speed), and download the CSV.
2. [Visual Crossing Weather](https://www.visualcrossing.com/weather-history): free daily historical exports worldwide.
3. [Kaggle Datasets](https://www.kaggle.com/datasets): search for daily climate datasets (for example, Delhi, Seattle, or Austin weather).

**Usage:** place the CSV at `data/raw/weather_data.csv`, or upload it through the dashboard sidebar.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

The suite verifies loader robustness, chronological split ordering, leakage prevention, and model training and inference.

---

## Design Decisions

1. **Chronological splitting.** Random shuffling (`train_test_split(shuffle=True)`) lets the model interpolate between adjacent days, which inflates scores. AtmosAI slices the data in time order (`iloc[:n]`).
2. **Rolling window alignment.** Rolling statistics are computed with `df[col].shift(1).rolling(w)`, so a feature vector never contains information from the day it is predicting from.
3. **Scaler isolation.** `StandardScaler` is fitted on the training split only and then applied to the validation and test splits.
4. **Flexible ingestion.** The loader handles varied meteorological column naming conventions without manual configuration.
