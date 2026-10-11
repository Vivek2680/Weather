"""
src/config.py
═══════════════════════════════════════════════════════════════════════════════
Central configuration for AtmosAI.

WHY A SEPARATE CONFIG MODULE?
──────────────────────────────
All "magic strings" — column names, file paths, thresholds — live here.
When you switch to a new dataset, you change ONE file.  Every other module
imports from this module; nothing is hard-coded in business logic.

COLUMN ALIASES
──────────────
Different weather CSVs use different column names for the same measurement.
COLUMN_ALIASES maps each internal standard name to a list of common aliases
found in real datasets (NOAA, IMD, Kaggle, BMKG, etc.).  The data loader
checks these aliases case-insensitively and renames columns at load time.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).resolve().parent.parent
DATA_DIR          = ROOT_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DATA_DIR= DATA_DIR / "processed"
MODELS_DIR        = ROOT_DIR / "models"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = RAW_DATA_DIR / "weather_data.csv"

# ── Internal column names (the ONLY names used after loading) ──────────────
COL_DATE      = "date"
COL_TMAX      = "tmax"        # Maximum temperature  (°C)
COL_TMIN      = "tmin"        # Minimum temperature  (°C)
COL_TAVG      = "tavg"        # Average temperature  (°C)
COL_HUMIDITY  = "humidity"    # Relative humidity    (%)
COL_RAINFALL  = "rainfall"    # Daily rainfall       (mm)
COL_WINDSPEED = "windspeed"   # Wind speed           (km/h)
COL_PRESSURE  = "pressure"    # Atmospheric pressure (hPa)

ALL_KNOWN_COLUMNS: List[str] = [
    COL_DATE, COL_TMAX, COL_TMIN, COL_TAVG,
    COL_HUMIDITY, COL_RAINFALL, COL_WINDSPEED, COL_PRESSURE,
]
REQUIRED_COLUMNS: List[str] = [COL_DATE]

# ── Column alias map (internal name → list of CSV synonyms) ───────────────
COLUMN_ALIASES: Dict[str, List[str]] = {
    COL_DATE:      ["date","Date","DATE","dt","time","Time","observation_date",
                    "report_date","day","YEAR"],
    COL_TMAX:      ["tmax","Tmax","TMAX","max_temp","MaxTemp","temp_max",
                    "maximum_temperature","tempmax","max_temperature","TMAX"],
    COL_TMIN:      ["tmin","Tmin","TMIN","min_temp","MinTemp","temp_min",
                    "minimum_temperature","tempmin","min_temperature","TMIN"],
    COL_TAVG:      ["tavg","Tavg","TAVG","avg_temp","AvgTemp","temp_avg",
                    "mean_temp","MeanTemp","mean_temperature","tempmean",
                    "temperature","TAVG","meantemp","Mean Temp (°C)","TMEAN"],
    COL_HUMIDITY:  ["humidity","Humidity","HUMIDITY","rhum","relative_humidity",
                    "rh","RH","Humidity(%)","mean_humidity"],
    COL_RAINFALL:  ["rainfall","Rainfall","RAINFALL","prcp","PRCP",
                    "precipitation","Precipitation","rain","Rain","precip",
                    "Precip","meanprecip","Precipitation (mm)"],
    COL_WINDSPEED: ["windspeed","WindSpeed","WINDSPEED","wspd","WSPD",
                    "wind_speed","wind speed","wdsp","Mean Wind Speed",
                    "wind_speed_kmh","meanwindspeed"],
    COL_PRESSURE:  ["pressure","Pressure","PRESSURE","pres","PRES",
                    "atm_pressure","sea_level_pressure","stp","STP",
                    "mean_pressure","MeanPressure"],
}

# ── Physical plausibility ranges (flag, don't drop) ───────────────────────
VALID_RANGES: Dict[str, tuple] = {
    COL_TMAX:      (-50.0,  60.0),
    COL_TMIN:      (-60.0,  55.0),
    COL_TAVG:      (-55.0,  57.0),
    COL_HUMIDITY:  (  0.0, 100.0),
    COL_RAINFALL:  (  0.0, 1200.0),
    COL_WINDSPEED: (  0.0,  250.0),
    COL_PRESSURE:  (870.0, 1090.0),
}

MISSING_VALUE_WARN_THRESHOLD = 0.20   # warn if > 20 % missing

# ── Train / Val / Test split (chronological, ratios must sum to 1) ─────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

MIN_TRAINING_ROWS = 365   # require at least 1 year of daily data

# ── Feature engineering parameters ────────────────────────────────────────
LAG_DAYS       : List[int] = [1, 2, 3, 7, 14]
ROLLING_WINDOWS: List[int] = [3, 7, 14, 30]
RANDOM_SEED    : int       = 42

# ── Saved model paths ──────────────────────────────────────────────────────
TEMP_MODEL_PATH = MODELS_DIR / "temperature_model.joblib"
TEMP_META_PATH  = MODELS_DIR / "temperature_meta.joblib"
RAIN_MODEL_PATH = MODELS_DIR / "rainfall_model.joblib"
RAIN_META_PATH  = MODELS_DIR / "rainfall_meta.joblib"

# ── Streamlit app settings ─────────────────────────────────────────────────
APP_TITLE = "AtmosAI — AI Weather Prediction"
APP_ICON  = "🌤️"
