"""
app.py
═══════════════════════════════════════════════════════════════════════════════
AtmosAI — Streamlit Dashboard
Integrates data loading, EDA, temperature prediction, rainfall prediction,
and model evaluation into a multi-page interactive application.

Architecture note:
  This file contains ONLY presentation logic (st.* calls).
  All ML, data processing, and analysis functions live in src/.
  app.py calls them; it does not implement them.
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="AtmosAI — AI Weather Prediction",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore")

# ── Imports from src/ ──────────────────────────────────────────────────────
from src.config import (
    APP_ICON, APP_TITLE, COL_DATE, COL_RAINFALL,
    COL_TMAX, COL_TMIN, COL_TAVG, DEFAULT_CSV,
)
from src.data_loader import LoadResult, load_weather_csv, get_dataset_summary
from src.data_validation import validate_dataframe
from src.preprocessing import preprocess, derive_tavg_if_missing
from src.data_analysis import (
    compute_summary_stats,
    plot_correlation_heatmap,
    plot_humidity_trend,
    plot_monthly_temperature,
    plot_rainfall_distribution,
    plot_seasonal_temperature,
    plot_temperature_history,
    plot_wind_speed,
    plot_year_over_year,
)
from src.train_temperature import train_temperature_models
from src.train_rainfall import train_rainfall_models
from src.predict import predict_next_day_temperature, predict_next_day_rainfall
from src.evaluation import compare_regression_models, compare_classification_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CSS — custom dark theme
# ══════════════════════════════════════════════════════════════════════════════
def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp { background: #0E1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1D27 0%, #252836 100%);
        border: 1px solid #2D3142;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4C9BE8;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9DA3B4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-unit {
        font-size: 0.75rem;
        color: #6B7280;
    }

    /* Section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #E8ECF4;
        border-left: 4px solid #4C9BE8;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Info / warning banners */
    .info-banner {
        background: rgba(76, 155, 232, 0.12);
        border: 1px solid #4C9BE8;
        border-radius: 8px;
        padding: 12px 16px;
        color: #A8C8F0;
        font-size: 0.88rem;
    }
    .warn-banner {
        background: rgba(232, 124, 76, 0.12);
        border: 1px solid #E87C4C;
        border-radius: 8px;
        padding: 12px 16px;
        color: #F0C4A8;
        font-size: 0.88rem;
    }
    .success-banner {
        background: rgba(76, 232, 197, 0.12);
        border: 1px solid #4CE8C5;
        border-radius: 8px;
        padding: 12px 16px;
        color: #A8F0E4;
        font-size: 0.88rem;
    }

    /* Prediction result card */
    .pred-card {
        background: linear-gradient(135deg, #1A1D27, #252836);
        border: 2px solid #4C9BE8;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .pred-temp {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4C9BE8, #4CE8C5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }

    /* Sidebar */
    .css-1d391kg { background: #13151F; }
    section[data-testid="stSidebar"] { background: #13151F; }

    /* Remove default Streamlit padding on columns */
    div[data-testid="column"] { padding: 4px !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session-state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "load_result":  None,
        "clean_df":     None,
        "impute_log":   None,
        "val_report":   None,
        "temp_artifacts": None,
        "rain_artifacts": None,
        "data_source":  "default",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _mc(value: Any, label: str, unit: str = "", colour: str = "#4C9BE8") -> str:
    """Render a metric card HTML snippet."""
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{colour};">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Data loading / caching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_and_prepare(csv_bytes: bytes, filename: str) -> Dict:
    """Load, validate, and preprocess a CSV — result is cached per file content."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = Path(tmp.name)

    result: LoadResult = load_weather_csv(tmp_path)
    tmp_path.unlink(missing_ok=True)

    if not result.success:
        return {"success": False, "errors": result.errors, "warnings": result.warnings}

    report = validate_dataframe(result.df, result.available_columns)

    df = derive_tavg_if_missing(result.df)
    if COL_TAVG not in result.available_columns and COL_TAVG in df.columns:
        result.available_columns.append(COL_TAVG)

    clean_df, impute_log = preprocess(df, result.available_columns)
    summary = get_dataset_summary(result.df, result.column_map)
    stats   = compute_summary_stats(clean_df, result.available_columns)

    return {
        "success":           True,
        "errors":            result.errors,
        "warnings":          result.warnings + report.warnings,
        "val_errors":        report.errors,
        "available_columns": result.available_columns,
        "column_map":        result.column_map,
        "missing_columns":   result.missing_columns,
        "clean_df":          clean_df,
        "impute_log":        impute_log,
        "summary":           summary,
        "stats":             stats,
        "outlier_counts":    report.outlier_counts,
        "missing_pct":       report.missing_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(f"## {APP_ICON} AtmosAI")
        st.markdown("*AI Weather Prediction System*")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Home", "📊 Historical Analysis", "🌡️ Temperature Prediction",
             "🌧️ Rainfall Prediction", "📈 Model Evaluation",
             "📁 Dataset Management", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**📂 Data Source**")

        use_default = DEFAULT_CSV.exists()
        upload_tab = st.file_uploader(
            "Upload weather CSV", type=["csv"], key="csv_upload",
            help="CSV must contain a date column and at least one weather variable."
        )

        if upload_tab is not None:
            data = _load_and_prepare(upload_tab.read(), upload_tab.name)
            if data["success"]:
                st.session_state.update({
                    "load_result": data,
                    "clean_df":    data["clean_df"],
                    "impute_log":  data["impute_log"],
                    "val_report":  None,
                    "temp_artifacts": None,
                    "rain_artifacts": None,
                    "data_source": upload_tab.name,
                })
                st.success(f"✅ Loaded: {upload_tab.name}")
            else:
                for e in data["errors"]:
                    st.error(e)

        elif use_default and st.session_state.get("load_result") is None:
            with open(DEFAULT_CSV, "rb") as f:
                data = _load_and_prepare(f.read(), DEFAULT_CSV.name)
            if data["success"]:
                st.session_state.update({
                    "load_result": data,
                    "clean_df":    data["clean_df"],
                    "impute_log":  data["impute_log"],
                    "data_source": DEFAULT_CSV.name,
                })

        if st.session_state.get("load_result"):
            lr = st.session_state["load_result"]
            s  = lr.get("summary", {})
            st.markdown(f"""
            <div style="font-size:0.8rem; color:#9DA3B4; margin-top:8px;">
            📅 {s.get('date_range_start','?')} → {s.get('date_range_end','?')}<br>
            📋 {s.get('total_rows','?'):,} daily observations<br>
            🔬 Features: {', '.join(s.get('available_features',[]))}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warn-banner">⚠️ No dataset loaded.<br>'
                'Upload a CSV or place <code>weather_data.csv</code> in '
                '<code>data/raw/</code></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("🔒 Offline · No API · No live weather data")

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════

def _page_home() -> None:
    st.markdown("# 🌤️ AtmosAI")
    st.markdown(
        "**AI-powered weather prediction and analysis** using historical "
        "observations — no API required."
    )

    lr = st.session_state.get("load_result")
    if lr is None:
        st.markdown("""
        <div class="warn-banner">
        📁 <strong>No dataset loaded.</strong> Use the sidebar to upload a weather CSV
        or place <code>weather_data.csv</code> in <code>data/raw/</code>.
        </div>
        """, unsafe_allow_html=True)
        _show_dataset_help()
        return

    s    = lr["summary"]
    stats = lr["stats"]
    avail = lr["available_columns"]

    # ── Key metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_mc(f"{s['total_rows']:,}", "Total Records", "daily observations"), unsafe_allow_html=True)
    with c2:
        span = s.get("date_span_days", 0)
        st.markdown(_mc(f"{span//365}y {(span%365)//30}m", "Date Span", "historical coverage"), unsafe_allow_html=True)
    with c3:
        n_feat = len([c for c in avail if c != COL_DATE])
        st.markdown(_mc(n_feat, "Features", "weather variables"), unsafe_allow_html=True)
    with c4:
        status = "✅ Trained" if st.session_state.get("temp_artifacts") else "⏳ Untrained"
        st.markdown(_mc(status, "Model Status", ""), unsafe_allow_html=True)

    # ── Temperature summary cards ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Historical Summary</div>', unsafe_allow_html=True)
    temp_stats = {k: v for k, v in stats.items()
                  if k in [COL_TMAX, COL_TMIN, COL_TAVG]}
    if temp_stats:
        cols = st.columns(len(temp_stats))
        labels = {COL_TMAX: "Max Temp", COL_TMIN: "Min Temp", COL_TAVG: "Avg Temp"}
        colours = {COL_TMAX: "#E87C4C", COL_TMIN: "#4CE8C5", COL_TAVG: "#4C9BE8"}
        for col, (k, v) in zip(cols, temp_stats.items()):
            with col:
                st.markdown(_mc(f"{v['mean']}°C", labels[k],
                                f"Range: {v['min']}–{v['max']}°C", colours[k]),
                            unsafe_allow_html=True)

    # ── Quick temperature chart ────────────────────────────────────────────
    clean_df = st.session_state.get("clean_df")
    if clean_df is not None:
        fig = plot_temperature_history(clean_df, avail)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    # ── Warnings ──────────────────────────────────────────────────────────
    if lr.get("warnings"):
        with st.expander(f"⚠️ {len(lr['warnings'])} data quality notice(s)"):
            for w in lr["warnings"]:
                st.warning(w)

    # ── Capabilities ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Available Predictions</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        if any(c in avail for c in [COL_TMAX, COL_TMIN, COL_TAVG]):
            st.success("✅ Temperature Prediction\n\nNext-day temperature forecasting")
        else:
            st.error("❌ No temperature columns found")
    with cb:
        if COL_RAINFALL in avail:
            st.success("✅ Rainfall Prediction\n\nNext-day rainfall classification")
        else:
            st.warning("⚠️ Rainfall column missing\n\nRainfall prediction unavailable")
    with cc:
        st.info("📊 Historical Analysis\n\nEDA, trends, correlations, YoY comparison")


def _show_dataset_help() -> None:
    st.markdown('<div class="section-header">📥 How to Get Weather Data</div>', unsafe_allow_html=True)
    st.markdown("""
**Recommended free datasets (no API required):**

| Source | URL | Notes |
|---|---|---|
| **Visual Crossing** | [visualcrossing.com/weather-history](https://www.visualcrossing.com/weather-history) | Free tier, daily history, global locations |
| **Kaggle: Daily Climate Delhi** | [kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) | Daily mean temp, humidity, wind, pressure |
| **NOAA Climate Data Online** | [ncei.noaa.gov/cdo-web](https://www.ncei.noaa.gov/cdo-web/) | GHCND daily station data, free download |
| **Open-Meteo (historical)** | [open-meteo.com](https://open-meteo.com) | Free, no API key needed, CSV download |

**After downloading:**
1. Place the CSV in `data/raw/`
2. Rename it `weather_data.csv`, OR upload via the sidebar
3. The app auto-detects column names (temperature, humidity, rainfall, etc.)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Historical Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _page_analysis() -> None:
    st.markdown("# 📊 Historical Weather Analysis")
    clean_df = st.session_state.get("clean_df")
    lr = st.session_state.get("load_result")

    if clean_df is None or lr is None:
        st.warning("Load a dataset first.")
        return

    avail = lr["available_columns"]
    stats = lr.get("stats", compute_summary_stats(clean_df, avail))

    # ── Date range filter ──────────────────────────────────────────────────
    min_date = clean_df.index.min().date()
    max_date = clean_df.index.max().date()
    col_a, col_b = st.columns(2)
    with col_a:
        start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with col_b:
        end = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    mask = (clean_df.index.date >= start) & (clean_df.index.date <= end)
    view_df = clean_df.loc[mask]
    if view_df.empty:
        st.error("No data in selected range.")
        return

    # ── Stats ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
    stat_df = pd.DataFrame(stats).T
    if not stat_df.empty:
        st.dataframe(stat_df.style.format("{:.2f}"), use_container_width=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    charts = [
        ("🌡️ Temperature History",     plot_temperature_history),
        ("📅 Monthly Distribution",    plot_monthly_temperature),
        ("🍂 Seasonal Comparison",     plot_seasonal_temperature),
        ("🌧️ Rainfall Distribution",   plot_rainfall_distribution),
        ("💧 Humidity Trend",          plot_humidity_trend),
        ("💨 Wind Speed",              plot_wind_speed),
        ("📐 Year-over-Year",          plot_year_over_year),
    ]
    for title, fn in charts:
        fig = fn(view_df, avail)
        if fig:
            st.markdown(f"**{title}**")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption(f"{title}: required columns not available in this dataset.")

    # ── Correlation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">⚠️ <strong>Correlation ≠ Causation.</strong> '
        'A strong correlation between two variables does not imply that one causes '
        'the other.</div>',
        unsafe_allow_html=True,
    )
    fig_corr = plot_correlation_heatmap(view_df, avail)
    if fig_corr:
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Temperature Prediction
# ══════════════════════════════════════════════════════════════════════════════

def _page_temperature() -> None:
    st.markdown("# 🌡️ Temperature Prediction")
    clean_df = st.session_state.get("clean_df")
    lr       = st.session_state.get("load_result")

    if clean_df is None or lr is None:
        st.warning("Load a dataset first.")
        return

    avail = lr["available_columns"]
    temp_cols = [c for c in [COL_TMAX, COL_TMIN, COL_TAVG] if c in avail]
    if not temp_cols:
        st.error("No temperature column found in dataset.")
        return

    col_labels = {COL_TMAX: "Max Temp (°C)", COL_TMIN: "Min Temp (°C)", COL_TAVG: "Avg Temp (°C)"}
    target_col = st.selectbox(
        "Select target temperature",
        options=temp_cols,
        format_func=lambda c: col_labels.get(c, c),
    )

    if st.button("🚀 Train / Load Temperature Model", type="primary"):
        with st.spinner("Training models (this may take 30–60 seconds)…"):
            try:
                artifacts = train_temperature_models(
                    clean_df, avail, target_col=target_col, force_retrain=True
                )
                st.session_state["temp_artifacts"] = artifacts
                st.success("✅ Model training complete!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

    artifacts = st.session_state.get("temp_artifacts")
    if artifacts is None:
        st.markdown(
            '<div class="info-banner">Click <strong>Train</strong> to train models '
            'or load an existing saved model.</div>',
            unsafe_allow_html=True,
        )
        return

    meta = artifacts["metadata"]
    if meta.get("target_col") != target_col:
        st.warning(
            f"Loaded model was trained for '{meta['target_col']}'. "
            "Re-train for the selected target."
        )

    # ── Prediction ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Next-Day Forecast</div>', unsafe_allow_html=True)
    try:
        pred = predict_next_day_temperature(clean_df, artifacts)
        forecast_date = pred["forecast_date"].strftime("%A, %d %b %Y")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="pred-card">
                <div style="color:#9DA3B4; font-size:0.9rem; margin-bottom:8px;">
                    Forecast for {forecast_date}
                </div>
                <div class="pred-temp">{pred['predicted_value']}°C</div>
                <div style="color:#9DA3B4; font-size:0.85rem; margin-top:12px;">
                    {col_labels.get(target_col, target_col)}
                </div>
                <div style="color:#6B7280; font-size:0.75rem; margin-top:8px;">
                    Model: {pred['model_type']}<br>
                    Horizon: {pred['forecast_horizon']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            # Show recent history vs prediction
            recent = clean_df[target_col].dropna().tail(60)
            fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#1A1D27")
            ax.plot(recent.index, recent.values, color="#4C9BE8",
                    linewidth=1.5, label="Historical")
            forecast_dt = pred["forecast_date"]
            ax.scatter([forecast_dt], [pred["predicted_value"]], color="#E87C4C",
                       s=120, zorder=5, label=f"Predicted ({pred['predicted_value']}°C)")
            ax.axvline(forecast_dt, color="#E87C4C", linestyle="--",
                       linewidth=0.8, alpha=0.5)
            ax.set_title("Recent History + Tomorrow's Forecast", color="white", fontsize=12)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.set_ylabel("Temperature (°C)")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            ax.legend(facecolor="#1A1D27", labelcolor="white", framealpha=0.7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as exc:
        st.error(f"Prediction error: {exc}")

    # ── Model info ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="warn-banner">
    ⚠️ <strong>Limitation:</strong> This is a next-day (1-day horizon) prediction only.
    The model is trained on historical patterns and cannot capture real-time atmospheric
    dynamics.  Use it as an educational forecasting demonstration, not an operational forecast.
    </div>
    """, unsafe_allow_html=True)

    # ── Downloadable predictions ───────────────────────────────────────────
    if artifacts:
        _download_temperature_predictions(clean_df, artifacts, avail, target_col)


def _download_temperature_predictions(
    clean_df: pd.DataFrame, artifacts: Dict, avail: List[str], target_col: str
) -> None:
    """Generate downloadable CSV of predictions on the test period."""
    try:
        from src.feature_engineering import (
            build_temperature_features, chronological_split, get_feature_names
        )
        from src.config import TRAIN_RATIO, VAL_RATIO
        feat_df, tname = build_temperature_features(clean_df, avail, target_col)
        _, _, test_df = chronological_split(feat_df, TRAIN_RATIO, VAL_RATIO)
        if test_df.empty:
            return
        fnames = get_feature_names(feat_df, tname)
        X_test = artifacts["scaler"].transform(test_df[fnames].values)
        preds  = artifacts["best_model"].predict(X_test)
        out = pd.DataFrame({
            "date":     test_df.index,
            "actual":   test_df[tname].values,
            "predicted": preds,
            "error":    preds - test_df[tname].values,
        })
        csv = out.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Test Predictions (CSV)",
            data=csv,
            file_name=f"atmosai_temp_predictions_{target_col}.csv",
            mime="text/csv",
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Rainfall Prediction
# ══════════════════════════════════════════════════════════════════════════════

def _page_rainfall() -> None:
    st.markdown("# 🌧️ Rainfall Prediction")
    clean_df = st.session_state.get("clean_df")
    lr       = st.session_state.get("load_result")

    if clean_df is None or lr is None:
        st.warning("Load a dataset first.")
        return

    avail = lr["available_columns"]
    if COL_RAINFALL not in avail:
        st.error(
            "The loaded dataset does not contain a rainfall column.  "
            "Rainfall prediction requires daily precipitation data."
        )
        return

    st.markdown("""
    <div class="info-banner">
    The rainfall model predicts <strong>whether measurable rainfall will occur
    tomorrow</strong> (binary classification: rain / no rain).<br>
    Threshold: &gt; 0.1 mm (WMO / IMD standard for "measurable rainfall").<br>
    This is <em>not</em> a prediction of rainfall <em>amount</em>.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train / Load Rainfall Model", type="primary"):
        with st.spinner("Training rainfall models…"):
            try:
                artifacts = train_rainfall_models(
                    clean_df, avail, force_retrain=True
                )
                if artifacts:
                    st.session_state["rain_artifacts"] = artifacts
                    st.success("✅ Rainfall model trained!")
                else:
                    st.error("Training returned None — check dataset has sufficient rainfall data.")
                    return
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

    artifacts = st.session_state.get("rain_artifacts")
    if artifacts is None:
        st.info("Click Train to build the rainfall classifier.")
        return

    # ── Prediction ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Next-Day Rainfall Forecast</div>',
                unsafe_allow_html=True)
    try:
        pred = predict_next_day_rainfall(clean_df, artifacts)
        forecast_date = pred["forecast_date"].strftime("%A, %d %b %Y")

        c1, c2 = st.columns(2)
        with c1:
            icon  = "🌧️" if pred["rain_tomorrow"] else "☀️"
            label = "Rain Expected" if pred["rain_tomorrow"] else "No Rain Expected"
            colour = "#4C9BE8" if pred["rain_tomorrow"] else "#E8C84C"
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:3.5rem;">{icon}</div>
                <div class="pred-temp" style="color:{colour};">{label}</div>
                <div style="color:#9DA3B4; font-size:0.85rem; margin-top:12px;">
                    {forecast_date}
                </div>
                <div style="color:#6B7280; font-size:0.75rem; margin-top:8px;">
                    Model: {pred['model_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if pred.get("rain_probability") is not None:
                prob = pred["rain_probability"]
                st.metric("Rain Probability (uncalibrated)", f"{prob*100:.1f}%")
                fig_gauge, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
                fig_gauge.patch.set_facecolor("#1A1D27")
                ax.set_facecolor("#1A1D27")
                bar_colour = "#4C9BE8" if prob < 0.5 else "#E87C4C"
                ax.barh([""], [prob], color=bar_colour, alpha=0.8, height=0.4)
                ax.barh([""], [1 - prob], left=[prob], color="#2D3142", height=0.4)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", color="white", fontsize=9)
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333")
                ax.set_title(f"Rain Probability: {prob*100:.1f}%", color="white", fontsize=10)
                fig_gauge.tight_layout()
                st.pyplot(fig_gauge)
                plt.close(fig_gauge)
                st.markdown(f'<div class="info-banner">⚠️ {pred["probability_note"]}</div>',
                            unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Prediction error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _page_evaluation() -> None:
    st.markdown("# 📈 Model Evaluation")

    temp_art = st.session_state.get("temp_artifacts")
    rain_art = st.session_state.get("rain_artifacts")

    if temp_art is None and rain_art is None:
        st.warning("Train at least one model to see evaluation results.")
        return

    # ── Temperature results ────────────────────────────────────────────────
    if temp_art:
        st.markdown('<div class="section-header">🌡️ Temperature Model Results</div>',
                    unsafe_allow_html=True)
        meta = temp_art["metadata"]
        st.caption(
            f"Train: {meta['train_start']} → {meta['train_end']}  |  "
            f"Val: {meta['val_start']} → {meta['val_end']}  |  "
            f"Test: {meta['test_start']} → {meta['test_end']}"
        )

        # Validation comparison table
        st.markdown("**Validation Set — All Models**")
        val_rows = temp_art.get("all_val_results", [])
        if val_rows:
            df_val = pd.DataFrame(val_rows).set_index("model")
            df_val = df_val[["MAE","RMSE","R2","n"]].sort_values("MAE")
            st.dataframe(
                df_val.style.highlight_min(subset=["MAE","RMSE"], color="#1a3a1a")
                            .highlight_max(subset=["R2"],         color="#1a3a1a")
                            .format({"MAE":"{:.3f}", "RMSE":"{:.3f}", "R2":"{:.3f}"}),
                use_container_width=True,
            )

        # Test result
        test_r = temp_art.get("test_result", {})
        if test_r:
            st.markdown("**Test Set — Best Model (evaluated once, never during selection)**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("MAE", f"{test_r.get('MAE','?')} °C")
            col_b.metric("RMSE", f"{test_r.get('RMSE','?')} °C")
            col_c.metric("R²", f"{test_r.get('R2','?')}")

        # Feature importances
        if meta.get("feature_importance"):
            fi = pd.Series(meta["feature_importance"]).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#1A1D27")
            ax.barh(fi.index[::-1], fi.values[::-1], color="#4C9BE8", alpha=0.8)
            ax.set_title("Top 15 Feature Importances (Random Forest)", color="white", fontsize=12)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Metric explanations
        with st.expander("📖 Metric Definitions"):
            st.markdown("""
| Metric | Meaning |
|---|---|
| **MAE** | Average prediction error in °C. A MAE of 2.0 means predictions are off by 2°C on average. |
| **RMSE** | Root mean squared error. Penalises large errors more than MAE. If RMSE >> MAE, the model makes occasional large mistakes. |
| **R²** | Fraction of temperature variance explained. R²=1.0 is perfect; R²=0 means the model is no better than predicting the mean. |
            """)

    # ── Rainfall results ───────────────────────────────────────────────────
    if rain_art:
        st.markdown('<div class="section-header">🌧️ Rainfall Classifier Results</div>',
                    unsafe_allow_html=True)
        rain_meta = rain_art["metadata"]
        st.caption(
            f"Train: {rain_meta.get('train_start','?')} → {rain_meta.get('train_end','?')}  |  "
            f"Test: {rain_meta.get('test_start','?')} → {rain_meta.get('test_end','?')}"
        )

        val_rows = rain_art.get("val_results", [])
        if val_rows:
            st.markdown("**Validation Set — All Classifiers**")
            rows_clean = [{k: v for k, v in r.items() if k != "confusion_matrix"}
                          for r in val_rows]
            df_rain = pd.DataFrame(rows_clean).set_index("model")
            display_cols = [c for c in ["precision","recall","f1","roc_auc","n"]
                            if c in df_rain.columns]
            st.dataframe(
                df_rain[display_cols].sort_values("f1", ascending=False)
                    .style.format({c: "{:.3f}" for c in display_cols if c != "n"}),
                use_container_width=True,
            )

        test_r = rain_art.get("test_result", {})
        if test_r and isinstance(test_r, dict):
            st.markdown("**Test Set — Best Classifier**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{test_r.get('precision','?'):.3f}")
            c2.metric("Recall",    f"{test_r.get('recall','?'):.3f}")
            c3.metric("F1 Score",  f"{test_r.get('f1','?'):.3f}")

            cm = test_r.get("confusion_matrix")
            if cm is not None:
                fig_cm, ax = plt.subplots(figsize=(5, 4), dpi=100)
                fig_cm.patch.set_facecolor("#0E1117")
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No Rain","Rain"],
                            yticklabels=["No Rain","Rain"], ax=ax)
                ax.set_title("Confusion Matrix (Test Set)", color="white", fontsize=11)
                ax.tick_params(colors="white")
                ax.title.set_color("white")
                fig_cm.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Management
# ══════════════════════════════════════════════════════════════════════════════

def _page_dataset() -> None:
    st.markdown("# 📁 Dataset Management")
    lr = st.session_state.get("load_result")
    clean_df = st.session_state.get("clean_df")
    impute_log = st.session_state.get("impute_log")

    if lr is None:
        st.warning("No dataset loaded. Upload a CSV via the sidebar.")
        _show_dataset_help()
        return

    s = lr["summary"]

    # ── Dataset info ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",  f"{s['total_rows']:,}")
    c2.metric("Date Range",  f"{s['date_range_start']} → {s['date_range_end']}")
    c3.metric("Span",        f"{s['date_span_days']} days")

    # ── Column mapping ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Column Mapping</div>', unsafe_allow_html=True)
    if lr.get("column_map"):
        cmap_df = pd.DataFrame([
            {"Internal Name": k, "CSV Column": v}
            for k, v in lr["column_map"].items()
        ])
        st.dataframe(cmap_df, use_container_width=True)
    if lr.get("missing_columns"):
        st.info(f"Not found in CSV: {', '.join(lr['missing_columns'])}")

    # ── Imputation log ─────────────────────────────────────────────────────
    if impute_log:
        st.markdown('<div class="section-header">🧹 Preprocessing Report</div>',
                    unsafe_allow_html=True)
        imp_df = pd.DataFrame([
            {"Column": k, "Action": v} for k, v in impute_log.items()
        ])
        st.dataframe(imp_df, use_container_width=True)

    # ── Missing values ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">❓ Missing Values</div>', unsafe_allow_html=True)
    miss = lr.get("missing_pct", {})
    if miss:
        miss_df = pd.DataFrame([
            {"Column": k, "Missing (%)": v} for k, v in miss.items()
        ]).sort_values("Missing (%)", ascending=False)
        st.dataframe(miss_df, use_container_width=True)

    # ── Data preview ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">👁️ Data Preview</div>', unsafe_allow_html=True)
    if clean_df is not None:
        n_rows = st.slider("Rows to preview", 5, 50, 10)
        st.dataframe(clean_df.head(n_rows), use_container_width=True)

    # ── Warnings / errors ──────────────────────────────────────────────────
    for w in lr.get("warnings", []):
        st.warning(w)
    for e in lr.get("val_errors", []):
        st.error(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════

def _page_about() -> None:
    st.markdown("# ℹ️ About AtmosAI")
    st.markdown("""
## Project Overview
AtmosAI is an educational AI weather prediction system built without any live
weather API.  It loads historical observations from a local CSV file and uses
scikit-learn machine learning models to predict next-day temperature and rainfall.

## Architecture

```
CSV (data/raw/)
    ↓
data_loader.py   — column resolution, date parsing, deduplication
data_validation.py — gap detection, range checks, temperature consistency
preprocessing.py — ffill imputation, outlier clipping
feature_engineering.py — lag features, rolling averages, calendar encoding
    ↓
train_temperature.py  — Persistence + LinearRegression + Ridge + RandomForest
train_rainfall.py     — Majority baseline + LogisticRegression + RandomForest
evaluation.py         — MAE / RMSE / R² / Precision / Recall / F1 / AUC
predict.py            — builds feature vector from latest obs, runs inference
    ↓
app.py (Streamlit) — interactive dashboard
```

## Machine Learning Methods

| Task | Models | Selection Criterion |
|---|---|---|
| Temperature regression | Persistence, LinearRegression, Ridge, RandomForest | Lowest validation MAE |
| Rainfall classification | Majority baseline, LogisticRegression, RandomForest | Highest validation F1 |

## Leakage Prevention
- All lag and rolling features use `.shift(n)` to ensure only past information is used
- Rolling windows are computed on `series.shift(1)` so the window ends at t-1
- Training uses a chronological split (never random)
- Preprocessing is fitted on the training set only
- The test set is evaluated exactly once

## Known Limitations
1. **Forecast horizon**: next-day (1-day) prediction only
2. **Local data**: predictions are only as good as the historical dataset
3. **No NWP**: the models do not simulate atmospheric dynamics
4. **Calibration**: rainfall probability is uncalibrated model output
5. **Extrapolation**: the model cannot predict during extreme weather patterns
   absent from the training set

## Technology Stack
- Python 3 · Pandas · NumPy · scikit-learn · Matplotlib · Streamlit · Joblib

## Data Disclaimer
This system uses historically observed data. It does not provide live weather
data and should not be used for safety-critical decisions.

## Author
Built as a portfolio project for B.Tech CSE (AI/ML specialization).
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _inject_css()
    _init_state()
    page = _sidebar()

    page_map = {
        "🏠 Home":                    _page_home,
        "📊 Historical Analysis":     _page_analysis,
        "🌡️ Temperature Prediction":  _page_temperature,
        "🌧️ Rainfall Prediction":     _page_rainfall,
        "📈 Model Evaluation":        _page_evaluation,
        "📁 Dataset Management":      _page_dataset,
        "ℹ️ About":                   _page_about,
    }
    fn = page_map.get(page, _page_home)
    fn()


if __name__ == "__main__":
    main()
