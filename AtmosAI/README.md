# 🌤️ AtmosAI — AI Weather Prediction and Analysis System

> An educational machine learning application that predicts next-day temperature
> and rainfall using historical weather observations — no API, no live data, fully offline.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

Weather forecasting typically relies on numerical weather prediction (NWP) systems
that model atmospheric dynamics using supercomputers and live sensor feeds.
This project demonstrates that even simple historical data can be used to train
machine learning models that outperform a naive persistence baseline on
next-day temperature prediction — a meaningful educational result.

---

## 🎯 Objectives

1. Load and validate historical weather data from a local CSV file.
2. Perform exploratory data analysis with interactive visualizations.
3. Predict next-day maximum temperature using multiple ML models.
4. Classify whether measurable rainfall will occur tomorrow.
5. Evaluate models against proper baselines using chronological splits.
6. Serve an interactive Streamlit dashboard.
7. Save trained models for subsequent use without retraining.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 CSV loading | Auto-detects column names from 30+ common aliases |
| 🔍 Data validation | Flags outliers, gaps, and temperature inconsistencies |
| 🧹 Preprocessing | Leakage-safe forward-fill imputation |
| 📊 EDA | Temperature, rainfall, humidity, wind, YoY comparison |
| 🌡️ Temperature prediction | Persistence + LinearRegression + Ridge + RandomForest |
| 🌧️ Rainfall classification | Majority baseline + LogisticRegression + RandomForest |
| 📈 Model evaluation | MAE, RMSE, R², Precision, Recall, F1, Confusion matrix |
| 💾 Model persistence | Joblib save/load with metadata and compatibility checks |
| ⬇️ CSV download | Download test-set predictions |
| 🔒 Offline | Runs without internet after initial install |

---

## 🏗️ Architecture

```
data/raw/weather_data.csv
        │
        ▼
src/data_loader.py          Column alias resolution, date parsing, deduplication
src/data_validation.py      Outlier detection, gap analysis, temperature consistency
src/preprocessing.py        Forward-fill imputation, outlier clipping
src/feature_engineering.py  Lag features, rolling averages, calendar encoding
        │
        ├──▶ src/train_temperature.py  → models/temperature_model.joblib
        └──▶ src/train_rainfall.py     → models/rainfall_model.joblib
        │
src/predict.py              Build feature vector from latest obs → inference
src/evaluation.py           MAE / RMSE / R² / Precision / Recall / F1 / AUC
        │
        ▼
app.py (Streamlit)          7-page interactive dashboard
```

---

## 📂 Project Structure

```
AtmosAI/
├── app.py                      Streamlit dashboard (UI only)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── weather_data.csv    ← place your dataset here
│   └── processed/
│
├── models/                     trained model files (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── config.py               paths, column aliases, thresholds, split ratios
│   ├── data_loader.py          CSV loading with alias resolution
│   ├── data_validation.py      data quality checks
│   ├── preprocessing.py        imputation and clipping
│   ├── feature_engineering.py  lag/rolling features, chronological split
│   ├── data_analysis.py        matplotlib EDA figures
│   ├── train_temperature.py    regression training pipeline
│   ├── train_rainfall.py       classification training pipeline
│   ├── evaluation.py           metric computation
│   └── predict.py              inference engine
│
└── tests/
    ├── test_data_validation.py
    ├── test_feature_engineering.py
    └── test_predictions.py
```

---

## 📥 Dataset

AtmosAI does **not** use any weather API.

### Recommended free datasets (no API key required):

| Source | URL | Notes |
|---|---|---|
| **Visual Crossing** | [visualcrossing.com/weather-history](https://www.visualcrossing.com/weather-history) | Free 1000 records/day, daily history, global |
| **Kaggle: Daily Climate Delhi** | [kaggle.com/datasets/sumanthvrao](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) | 2013–2017, meantemp, humidity, wind, pressure |
| **Open-Meteo Historical** | [open-meteo.com](https://open-meteo.com) | Free, no signup, CSV download, global |
| **NOAA CDO** | [ncei.noaa.gov/cdo-web](https://www.ncei.noaa.gov/cdo-web/) | GHCND daily, free, requires registration |

### Column naming
AtmosAI auto-detects column names using a configurable alias map in
`src/config.py → COLUMN_ALIASES`. Supported synonyms include:
`tmax / MaxTemp / max_temp / TMAX`, `rainfall / prcp / PRCP / precipitation`, etc.

---

## ⚙️ Installation (macOS)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AtmosAI.git
cd AtmosAI
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
```bash
# Place your CSV here:
data/raw/weather_data.csv

# OR use the sidebar upload in the dashboard
```

---

## 🚀 Running AtmosAI

### Launch the dashboard
```bash
source .venv/bin/activate
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Run the test suite
```bash
pytest tests/ -v
```

### Run tests with coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🤖 Machine Learning Methods

### Temperature Prediction (Regression)

| Model | Description | Strength |
|---|---|---|
| Persistence Baseline | Predict tomorrow = today | Free benchmark |
| Linear Regression | Ordinary least squares | Interpretable |
| Ridge Regression | OLS + L2 regularization | Handles correlated features |
| Random Forest | 200-tree ensemble | Non-linear patterns |

**Selection**: best model chosen by validation MAE (not test MAE).
**Evaluation**: test set used exactly once for the winning model.

### Rainfall Prediction (Classification)

| Model | Description |
|---|---|
| Majority Class Baseline | Always predict the most frequent class |
| Logistic Regression | Linear decision boundary, calibrated |
| Random Forest | Ensemble classifier |

**Threshold**: 0.1 mm (WMO/IMD standard for measurable rainfall).
**Class imbalance**: handled with `class_weight='balanced'`.
**Selection**: best model chosen by validation F1 score.

---

## 📊 Leakage Prevention

| Rule | Implementation |
|---|---|
| Lag features use only past values | `df[col].shift(n)` |
| Rolling averages end at t-1 | `df[col].shift(1).rolling(w)` |
| Train/val/test split is chronological | `features_df.iloc[:n]` |
| Scaler fitted on training data only | `scaler.fit_transform(X_train)` |
| Test set evaluated once | Single evaluation after model selection |
| No future info in imputation | Forward-fill only |

---

## 📈 Evaluation Methodology

```
Full Dataset (chronological)
│
├── TRAIN (70%) — model training and scaler fitting
├── VALIDATION (15%) — model selection (MAE / F1)
└── TEST (15%) — final evaluation (evaluated ONCE)
```

Metrics reported on the **test set** are the honest performance estimates.
Validation metrics are shown for comparison but were used in selection.

---

## ⚠️ Known Limitations

1. **Forecast horizon**: next-day (1-day) prediction only.
2. **No NWP**: historical patterns cannot capture real-time atmospheric dynamics.
3. **Local data**: accuracy depends entirely on the quality and length of your CSV.
4. **No calibration**: rainfall probability is uncalibrated model output.
5. **Extreme events**: the model cannot generalize to weather patterns absent from training.
6. **Single location**: the model is trained on one location's history.

---

## 🔮 Future Improvements

- [ ] Rolling-origin evaluation for multi-day forecasts
- [ ] Probability calibration for rainfall (Platt scaling / isotonic regression)
- [ ] XGBoost / LightGBM as additional models
- [ ] Multiple location support
- [ ] Automated NOAA/Open-Meteo data download script
- [ ] Docker containerization

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

## 👨‍💻 About

Built as a portfolio project for B.Tech Computer Science (AI/ML specialization).
Demonstrates a complete ML engineering workflow: data ingestion → validation →
feature engineering → model training → evaluation → deployment (local Streamlit).

> **This application uses historically observed data only.
> It does not provide live weather forecasts and should not be used
> for safety-critical decisions.**
