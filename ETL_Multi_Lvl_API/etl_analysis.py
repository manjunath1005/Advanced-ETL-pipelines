"""
Read air quality data from Supabase (or staged CSV) and produce:
A. KPI Metrics
   - City with highest average PM2.5
   - City with highest severity score
   - Percentage of High/Moderate/Low risk hours
   - Hour of day with worst AQI (by avg PM2.5)
B. City Pollution Trend Report
   - For each city: time -> pm2_5, pm10, ozone (saved to pollution_trends.csv)
C. Export Outputs -> save CSVs into data/processed/
   - summary_metrics.csv
   - city_risk_distribution.csv
   - pollution_trends.csv
D. Visualizations (PNG in data/processed/plots)
   - Histogram of PM2.5
   - Bar chart of risk flags per city
   - Line chart of hourly PM2.5 trends (cities)
   - Scatter: severity_score vs pm2_5
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Optional supabase client
try:
    from supabase import create_client
except Exception:
    create_client = None

load_dotenv()

#  CONFIG 
USE_STAGED = os.getenv("USE_STAGED", "False").lower() in ("1", "true", "yes")
STAGED_DIR = Path(os.getenv("STAGED_DIR", "data/staged"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
PLOTS_DIR = PROCESSED_DIR / "plots"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "air_quality")

# plotting style
sns.set(style="whitegrid", rc={"figure.dpi": 150})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#  Utilities 
def severity_score_row(row: pd.Series) -> float:
    """Compute severity using the weighted formula."""
    def safe(x):
        try:
            return float(x) if pd.notna(x) else 0.0
        except Exception:
            return 0.0
    return (
        safe(row.get("pm2_5")) * 5.0
        + safe(row.get("pm10")) * 3.0
        + safe(row.get("nitrogen_dioxide")) * 4.0
        + safe(row.get("sulphur_dioxide")) * 4.0
        + safe(row.get("carbon_monoxide")) * 2.0
        + safe(row.get("ozone")) * 3.0
    )


def risk_class_from_severity(sev: float) -> str:
    """Return risk class from severity."""
    if sev > 400:
        return "High Risk"
    if sev > 200:
        return "Moderate Risk"
    return "Low Risk"


def aqi_category_pm25(v) -> str:
    try:
        if pd.isna(v):
            return "Unknown"
        v = float(v)
    except Exception:
        return "Unknown"
    if v <= 50:
        return "Good"
    if 51 <= v <= 100:
        return "Moderate"
    if 101 <= v <= 200:
        return "Unhealthy"
    if 201 <= v <= 300:
        return "Very Unhealthy"
    if v > 300:
        return "Hazardous"
    return "Unknown"


#  Data loading 
def load_from_staged() -> pd.DataFrame:
    files = sorted(STAGED_DIR.glob("air_quality_t_*.csv"))
    if not files:
        raise FileNotFoundError(f"No staged files found in {STAGED_DIR}")
    path = files[-1]
    logging.info("Loading staged CSV: %s", path)
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def load_from_supabase(limit: Optional[int] = None) -> pd.DataFrame:
    if create_client is None:
        raise RuntimeError("supabase package not installed. Set USE_STAGED=True or install supabase.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set to load from Supabase.")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    q = supabase.table(SUPABASE_TABLE).select("*")
    if limit:
        q = q.limit(limit)
    res = q.execute()
    # handle both return shapes
    data = None
    if isinstance(res, dict):  # older/newer shape might differ
        data = res.get("data", res)
    else:
        data = getattr(res, "data", None) or res
    df = pd.DataFrame(data)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def load_data() -> pd.DataFrame:
    if USE_STAGED:
        return load_from_staged()
    else:
        return load_from_supabase()


#  Analysis computations 
def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute KPI metrics described in the spec."""
    # Ensure numeric pollutant columns exist
    for c in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "uv_index"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure severity present or compute
    if "severity" not in df.columns or df["severity"].isna().all():
        logging.info("Computing severity_score because column missing or NaN.")
        df["severity"] = df.apply(severity_score_row, axis=1)

    # Risk class
    if "risk_class" not in df.columns or df["risk_class"].isna().all():
        df["risk_class"] = df["severity"].apply(risk_class_from_severity)

    # AQI based on pm2_5 (category column)
    if "aqi_pm25" not in df.columns or df["aqi_pm25"].isna().all():
        df["aqi_pm25"] = df["pm2_5"].apply(aqi_category_pm25)

    # A. KPIs
    kpis = {}

    # City with highest average PM2.5
    pm25_by_city = df.groupby("city")["pm2_5"].mean().dropna()
    if not pm25_by_city.empty:
        top_city_pm25 = pm25_by_city.idxmax()
        top_city_pm25_val = pm25_by_city.max()
    else:
        top_city_pm25 = None
        top_city_pm25_val = None
    kpis["city_highest_avg_pm2_5"] = top_city_pm25
    kpis["city_highest_avg_pm2_5_value"] = float(top_city_pm25_val) if top_city_pm25_val is not None else None

    # City with highest average severity score
    sev_by_city = df.groupby("city")["severity"].mean().dropna()
    if not sev_by_city.empty:
        top_city_sev = sev_by_city.idxmax()
        top_city_sev_val = sev_by_city.max()
    else:
        top_city_sev = None
        top_city_sev_val = None
    kpis["city_highest_severity"] = top_city_sev
    kpis["city_highest_severity_value"] = float(top_city_sev_val) if top_city_sev_val is not None else None

    # Percentage of High/Moderate/Low risk hours (overall)
    risk_counts = df["risk_class"].value_counts(dropna=True)
    total = risk_counts.sum() if len(risk_counts) > 0 else 0
    pct = {}
    for cls in ["High Risk", "Moderate Risk", "Low Risk"]:
        pct[cls] = float(risk_counts.get(cls, 0) / total * 100) if total > 0 else 0.0
    kpis["risk_percentages"] = pct

    # Hour of day with worst AQI (based on average pm2.5 aggregated by hour across all cities)
    if "time" in df.columns:
        # extract hour (localize if naive? we'll use time.hour if present)
        df_time = df[df["time"].notna()].copy()
        if not df_time.empty:
            df_time["hour_of_day"] = df_time["time"].dt.hour
            hour_pm25 = df_time.groupby("hour_of_day")["pm2_5"].mean().dropna()
            if not hour_pm25.empty:
                worst_hour = int(hour_pm25.idxmax())
                worst_hour_val = float(hour_pm25.max())
            else:
                worst_hour = None
                worst_hour_val = None
        else:
            worst_hour = None
            worst_hour_val = None
    else:
        worst_hour = None
        worst_hour_val = None

    kpis["hour_with_worst_avg_pm2_5"] = worst_hour
    kpis["hour_with_worst_avg_pm2_5_value"] = worst_hour_val

    # Return KPIs and the possibly augmented dataframe (for use downstream)
    return kpis, df


def build_city_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with city, high/moderate/low counts and percentages."""
    # Count per city per risk class
    counts = df.groupby(["city", "risk_class"]).size().unstack(fill_value=0)
    # Ensure columns exist
    for col in ["High Risk", "Moderate Risk", "Low Risk"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total_hours"] = counts.sum(axis=1)
    counts["pct_high"] = counts["High Risk"] / counts["total_hours"] * 100
    counts["pct_moderate"] = counts["Moderate Risk"] / counts["total_hours"] * 100
    counts["pct_low"] = counts["Low Risk"] / counts["total_hours"] * 100
    counts = counts.reset_index()
    # reorder columns
    cols = ["city", "High Risk", "Moderate Risk", "Low Risk", "total_hours", "pct_high", "pct_moderate", "pct_low"]
    for c in cols:
        if c not in counts.columns:
            counts[c] = 0
    return counts[cols]


def build_pollution_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with city, time, pm2_5, pm10, ozone for every observed record.
    We will sort by city/time; if there are multiple rows for same city-time we aggregate mean.
    """
    cols = ["city", "time", "pm2_5", "pm10", "ozone"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df_sub = df[cols].copy()
    # ensure time parsed
    df_sub["time"] = pd.to_datetime(df_sub["time"], errors="coerce")
    # aggregate duplicates
    df_grouped = df_sub.groupby(["city", "time"], as_index=False).agg({"pm2_5": "mean", "pm10": "mean", "ozone": "mean"})
    df_grouped = df_grouped.sort_values(["city", "time"])
    return df_grouped


#  Visualizations 
def plot_histogram_pm25(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["pm2_5"].dropna(), bins=40, kde=False)
    plt.title("Histogram of PM2.5")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved histogram: %s", out_path)


def plot_risk_flags_per_city(city_risk_df: pd.DataFrame, out_path: Path) -> None:
    # create a stacked bar chart of counts
    plot_df = city_risk_df.copy()
    plot_df = plot_df.set_index("city")
    plot_df[["High Risk", "Moderate Risk", "Low Risk"]] = plot_df[["High Risk", "Moderate Risk", "Low Risk"]].astype(float)
    plot_df = plot_df.sort_values("High Risk", ascending=False)
    plot_df[["High Risk", "Moderate Risk", "Low Risk"]].plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Risk flags per city (counts of hours)")
    plt.xlabel("City")
    plt.ylabel("Hours")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved bar chart of risk flags: %s", out_path)


def plot_hourly_pm25_trends(pollution_trends_df: pd.DataFrame, out_path: Path, top_n_cities: int = 6) -> None:
    """
    Plot hourly PM2.5 trends for top N cities.
    Correctly resamples only numeric columns and avoids aggregating non-numeric data.
    """
    # ensure time is datetime
    if "time" in pollution_trends_df.columns:
        pollution_trends_df["time"] = pd.to_datetime(pollution_trends_df["time"], errors="coerce"   , utc=True)

    # pick top N cities by data count
    counts = pollution_trends_df["city"].value_counts().head(top_n_cities).index.tolist()

    plt.figure(figsize=(12, 6))
    for city in counts:
        city_df = pollution_trends_df[pollution_trends_df["city"] == city].sort_values("time")
        if city_df.empty:
            continue
        # Set time as index
        city_df = city_df.set_index("time")

        # Select only the numeric column pm2_5 (avoid string columns like city)
        if "pm2_5" not in city_df.columns:
            continue

        # Resample hourly computing mean of numeric columns only
        # use 'h' (lowercase) to avoid FutureWarning
        hourly = city_df["pm2_5"].resample("h").mean()

        # plot
        plt.plot(hourly.index, hourly.values, label=city)

    plt.legend()
    plt.title("Hourly PM2.5 trends (top cities)")
    plt.xlabel("time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved hourly PM2.5 trends: %s", out_path)

def plot_severity_vs_pm25(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="pm2_5", y="severity", hue="city", data=df.sample(min(len(df), 2000)))  # sample for performance
    plt.title("Severity score vs PM2.5")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Severity Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved scatter severity vs pm2_5: %s", out_path)

#  Main flow 
def run_etl_analysis(limit: Optional[int] = None):
    logging.info("Starting ETL analysis. USE_STAGED=%s", USE_STAGED)
    df = load_data() if USE_STAGED else load_from_supabase(limit=limit)
    logging.info("Loaded rows: %d", len(df))

    # normalize city names
    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()

    # compute KPIs (this also computes severity/risk if missing)
    kpis, df = compute_kpis(df)

    # Build outputs
    summary_metrics = {
        "metric": [
            "city_highest_avg_pm2_5",
            "city_highest_avg_pm2_5_value",
            "city_highest_severity",
            "city_highest_severity_value",
            "hour_with_worst_avg_pm2_5",
            "hour_with_worst_avg_pm2_5_value",
        ],
        "value": [
            kpis.get("city_highest_avg_pm2_5"),
            kpis.get("city_highest_avg_pm2_5_value"),
            kpis.get("city_highest_severity"),
            kpis.get("city_highest_severity_value"),
            kpis.get("hour_with_worst_avg_pm2_5"),
            kpis.get("hour_with_worst_avg_pm2_5_value"),
        ],
    }
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = PROCESSED_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Saved summary metrics to %s", summary_csv)

    # City risk distribution
    city_risk_df = build_city_risk_distribution(df)
    city_risk_csv = PROCESSED_DIR / "city_risk_distribution.csv"
    city_risk_df.to_csv(city_risk_csv, index=False)
    logging.info("Saved city risk distribution to %s", city_risk_csv)

    # Pollution trends
    trends_df = build_pollution_trends(df)
    pollution_trends_csv = PROCESSED_DIR / "pollution_trends.csv"
    trends_df.to_csv(pollution_trends_csv, index=False)
    logging.info("Saved pollution trends to %s", pollution_trends_csv)

    # Visualizations
    # Histogram of PM2.5
    plot_histogram_pm25_path = PLOTS_DIR / "hist_pm2_5.png"
    plot_histogram_pm25(df, plot_histogram_pm25_path)

    # Bar chart of risk flags per city
    plot_risk_flags_path = PLOTS_DIR / "risk_flags_per_city.png"
    plot_risk_flags_per_city(city_risk_df, plot_risk_flags_path)

    # Line chart of hourly PM2.5 trends
    plot_hourly_pm25_path = PLOTS_DIR / "hourly_pm2_5_trends.png"
    plot_hourly_pm25_trends(trends_df, plot_hourly_pm25_path)

    # Scatter: severity_score vs pm2_5 (sampled)
    plot_scatter_path = PLOTS_DIR / "severity_vs_pm2_5.png"
    plot_severity_vs_pm25(df, plot_scatter_path)

    logging.info("ETL analysis complete. Processed outputs in %s", PROCESSED_DIR)


if __name__ == "__main__":
    run_etl_analysis()
