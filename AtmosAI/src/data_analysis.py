"""
src/data_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Generates all EDA visualizations for the dashboard.

Design contract:
  • Every function RETURNS a matplotlib Figure.
  • Functions never call plt.show() — the Streamlit layer decides when to render.
  • All figures use a dark theme to match the dashboard aesthetic.
  • Correlation heatmaps are explicitly labelled "association ≠ causation".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COL_HUMIDITY, COL_PRESSURE, COL_RAINFALL,
    COL_TAVG, COL_TMAX, COL_TMIN, COL_WINDSPEED,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="darkgrid", palette="muted")

# Colour palette
_BG    = "#0E1117"
_PANEL = "#1A1D27"
_BLUE  = "#4C9BE8"
_WARM  = "#E87C4C"
_COOL  = "#4CE8C5"
_PURP  = "#B56CE8"
_GRÜN  = "#9BE84C"


def _styled_fig(figsize=(13, 5)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    return fig, ax


# ── Temperature history ────────────────────────────────────────────────────

def plot_temperature_history(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    cols_cfg = [
        (COL_TMAX, "Max Temp", _WARM),
        (COL_TMIN, "Min Temp", _COOL),
        (COL_TAVG, "Avg Temp", _BLUE),
    ]
    present = [(c, lbl, clr) for c, lbl, clr in cols_cfg
               if c in available_columns and c in df.columns]
    if not present:
        return None

    fig, ax = _styled_fig(figsize=(14, 5))
    for col, label, colour in present:
        ax.plot(df.index, df[col], linewidth=0.8, alpha=0.85,
                label=label, color=colour)
    ax.set_title("Historical Temperature Record", fontsize=14, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(facecolor=_PANEL, labelcolor="white", framealpha=0.7)
    fig.tight_layout()
    return fig


# ── Monthly temperature boxplot ────────────────────────────────────────────

def plot_monthly_temperature(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    target = next(
        (c for c in [COL_TAVG, COL_TMAX, COL_TMIN]
         if c in available_columns and c in df.columns), None
    )
    if target is None:
        return None

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    data = [df[target][df.index.month == m].dropna().values for m in range(1, 13)]

    fig, ax = _styled_fig(figsize=(12, 5))
    bp = ax.boxplot(data, labels=month_names, patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(_BLUE)
        patch.set_alpha(0.6)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#aaa")
    for f in bp["fliers"]:
        f.set(marker="o", color=_WARM, alpha=0.4, markersize=3)
    col_label = {"tavg":"Avg","tmax":"Max","tmin":"Min"}.get(target, target)
    ax.set_title(f"Monthly {col_label} Temperature Distribution", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    fig.tight_layout()
    return fig


# ── Seasonal bar chart ─────────────────────────────────────────────────────

def plot_seasonal_temperature(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    target = next(
        (c for c in [COL_TAVG, COL_TMAX, COL_TMIN]
         if c in available_columns and c in df.columns), None
    )
    if target is None:
        return None

    season_map = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Summer",4:"Summer",5:"Summer",
                  6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
                  10:"Post-Monsoon",11:"Post-Monsoon"}
    order = ["Winter","Summer","Monsoon","Post-Monsoon"]
    colours = [_COOL, _WARM, _BLUE, _PURP]

    season_series = df.index.month.map(season_map)
    means = df[target].groupby(season_series).mean().reindex(order)

    fig, ax = _styled_fig(figsize=(8, 5))
    bars = ax.bar(means.index, means.values, color=colours, alpha=0.85,
                  edgecolor="#222", linewidth=0.5)
    for bar, val in zip(bars, means.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}°C", ha="center", va="bottom",
                    color="white", fontsize=10)
    ax.set_title("Mean Temperature by Indian Meteorological Season", fontsize=13)
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean Temperature (°C)")
    fig.tight_layout()
    return fig


# ── Rainfall distribution ──────────────────────────────────────────────────

def plot_rainfall_distribution(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    if COL_RAINFALL not in available_columns or COL_RAINFALL not in df.columns:
        return None

    rain = df[COL_RAINFALL].dropna()
    rainy = rain[rain > 0.1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    axes[0].hist(rain, bins=50, color=_BLUE, alpha=0.8, edgecolor="#222")
    axes[0].set_title("Daily Rainfall — All Days", fontsize=12)
    axes[0].set_xlabel("Rainfall (mm)")
    axes[0].set_ylabel("Count")
    pct = 100 * len(rainy) / len(rain) if len(rain) else 0
    axes[0].text(0.97, 0.95, f"Rainy days: {pct:.1f}%",
                 transform=axes[0].transAxes, ha="right", va="top",
                 color="white", fontsize=9)

    if len(rainy) > 0:
        axes[1].hist(rainy, bins=40, color=_WARM, alpha=0.8, edgecolor="#222")
        axes[1].set_title("Rainfall — Rainy Days Only (> 0.1 mm)", fontsize=12)
        axes[1].set_xlabel("Rainfall (mm)")
    else:
        axes[1].text(0.5, 0.5, "No rainy days recorded",
                     ha="center", va="center", color="white",
                     fontsize=12, transform=axes[1].transAxes)
    fig.tight_layout()
    return fig


# ── Humidity trend ─────────────────────────────────────────────────────────

def plot_humidity_trend(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    if COL_HUMIDITY not in available_columns or COL_HUMIDITY not in df.columns:
        return None
    monthly = df[COL_HUMIDITY].resample("ME").mean()
    fig, ax = _styled_fig(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, color=_COOL, linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=_COOL)
    ax.set_title("Monthly Mean Relative Humidity", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Humidity (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


# ── Wind speed ─────────────────────────────────────────────────────────────

def plot_wind_speed(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    if COL_WINDSPEED not in available_columns or COL_WINDSPEED not in df.columns:
        return None
    monthly = df[COL_WINDSPEED].resample("ME").mean()
    fig, ax = _styled_fig(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, color=_GRÜN, linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=_GRÜN)
    ax.set_title("Monthly Mean Wind Speed", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wind Speed (km/h)")
    fig.tight_layout()
    return fig


# ── Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    numeric_cols = [
        c for c in available_columns
        if c != "date" and c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr()
    size = len(numeric_cols)
    fig, ax = plt.subplots(figsize=(size + 2, size + 1), dpi=110)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor="#222",
        ax=ax, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation  (association ≠ causation)", fontsize=13)
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("white")
    fig.tight_layout()
    return fig


# ── Year-over-year comparison ──────────────────────────────────────────────

def plot_year_over_year(
    df: pd.DataFrame, available_columns: List[str]
) -> Optional[plt.Figure]:
    target = next(
        (c for c in [COL_TAVG, COL_TMAX, COL_TMIN]
         if c in available_columns and c in df.columns), None
    )
    if target is None:
        return None

    years = sorted(df.index.year.unique())
    if len(years) < 2:
        return None

    fig, ax = _styled_fig(figsize=(13, 5))
    cmap = plt.cm.plasma  # noqa: E1101
    for i, year in enumerate(years):
        mask = df.index.year == year
        year_data = df.loc[mask, target]
        doy = df.index[mask].dayofyear
        smooth = year_data.rolling(7, center=True, min_periods=1).mean()
        colour = cmap(i / max(len(years) - 1, 1))
        ax.plot(doy, smooth.values, linewidth=1.0, alpha=0.75,
                label=str(year), color=colour)

    ax.set_title("Year-over-Year Temperature (7-day smooth)", fontsize=13)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(facecolor=_PANEL, labelcolor="white", framealpha=0.7,
              ncol=max(1, len(years)//5), fontsize=8)
    fig.tight_layout()
    return fig


# ── Summary statistics dict ────────────────────────────────────────────────

def compute_summary_stats(
    df: pd.DataFrame, available_columns: List[str]
) -> Dict:
    stats: Dict = {}
    for col in available_columns:
        if col == "date" or col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "mean":        round(float(s.mean()), 2),
            "median":      round(float(s.median()), 2),
            "min":         round(float(s.min()), 2),
            "max":         round(float(s.max()), 2),
            "std":         round(float(s.std()), 2),
            "count":       int(s.count()),
            "missing_pct": round(df[col].isna().mean() * 100, 2),
        }
    return stats
