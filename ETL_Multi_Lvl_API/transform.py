"""
Transform step for Urban Air Quality Monitoring ETL.
Handles two raw JSON shapes:
 - hourly-array shape (payload["hourly"]["time"], payload["hourly"]["pm2_5"], ...)
 - measurement-list shape (OpenAQ v3 /locations results with 'parameters'/'measurements')

Produces one row per city per hour with required columns and derived features,
drops rows with all pollutant values missing, and saves staged CSV to data/staged/.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

RAW_DIR = Path("data") / "raw"
STAGED_DIR = Path("data") / "staged"
STAGED_DIR.mkdir(parents=True, exist_ok=True)

TS_FMT = "%Y%m%dT%H%M%SZ"
NOW_TS = datetime.now(timezone.utc).strftime(TS_FMT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# target pollutant columns
POLLUTANT_COLS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "uv_index",
]

OUT_COLS = ["city", "time"] + POLLUTANT_COLS

# helper: AQI category from pm2.5
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

# severity formula
def severity_score(row: pd.Series) -> float:
    def safe(x):
        return float(x) if pd.notna(x) else 0.0
    s = (
        safe(row.get("pm2_5")) * 5.0
        + safe(row.get("pm10")) * 3.0
        + safe(row.get("nitrogen_dioxide")) * 4.0
        + safe(row.get("sulphur_dioxide")) * 4.0
        + safe(row.get("carbon_monoxide")) * 2.0
        + safe(row.get("ozone")) * 3.0
    )
    return s

def risk_class_from_severity(sev: float) -> str:
    if sev > 400:
        return "High Risk"
    if sev > 200:
        return "Moderate Risk"
    return "Low Risk"

#  Parsing helpers 
def _discover_raw_files() -> List[Path]:
    if not RAW_DIR.exists():
        logging.warning("Raw dir %s does not exist.", RAW_DIR)
        return []
    return sorted(RAW_DIR.glob("*_raw_*.json"))

def _parse_hourly_array_payload(payload: Dict[str, Any], filename_hint: str = "") -> pd.DataFrame:
    """
    Parse payload that contains payload["hourly"] with 'time' array and pollutant arrays.
    Returns DataFrame with columns: city, time (datetime UTC), pollutant columns.
    """
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return pd.DataFrame(columns=OUT_COLS)

    times = hourly.get("time") or hourly.get("times")
    if not times or not isinstance(times, list):
        return pd.DataFrame(columns=OUT_COLS)

    # get city hint (some hourly payloads lack explicit city)
    city = payload.get("city") or payload.get("name") or ""
    if not city and filename_hint:
        # filename like "mumbai_raw_20251211T060616Z"
        city = filename_hint.split("_")[0]

    # Build dict of arrays aligned by index
    data = {"time": times}
    for col in POLLUTANT_COLS:
        # payload keys may be pm2_5 or pm2.5 etc. try a few variants
        possible_keys = [col, col.replace("_", "."), col.replace("_", ""), col.replace("_", "").lower()]
        found = None
        for k in possible_keys:
            if k in hourly:
                found = k
                break
        data[col] = hourly.get(found) if found else [None] * len(times)

    df = pd.DataFrame(data)
    df["city"] = city or "unknown"
    # ensure time as timezone-aware UTC
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(timezone.utc, ambiguous="NaT", nonexistent="NaT") \
        .where(pd.notnull(pd.to_datetime(df["time"], errors="coerce")), pd.NaT)
    # convert pollutant columns to numeric (coerce)
    for c in POLLUTANT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # reorder columns
    df = df[["city", "time"] + POLLUTANT_COLS]
    return df

def _parse_measurement_list_payload(payload: Dict[str, Any], filename_hint: str = "") -> pd.DataFrame:
    """
    Parse OpenAQ /locations-style payloads with payload["results"] list containing
    parameters/measurements/latest fields (defensive).
    Returns DataFrame with city, time, parameter, value and then pivoted to pollutant columns.
    """
    rows = []
    results = payload.get("results") if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
    if not results:
        return pd.DataFrame(columns=OUT_COLS)

    for loc in results:
        city = loc.get("city") or loc.get("name") or loc.get("location") or ""
        # try parameters
        params = loc.get("parameters") or loc.get("measurements") or loc.get("latest") or loc.get("latestMeasurements")
        if isinstance(params, list):
            for p in params:
                param = p.get("parameter") or p.get("param") or p.get("name")
                # some use lastValue / value / avg
                val = p.get("lastValue") if "lastValue" in p else p.get("value") if "value" in p else p.get("avg") if "avg" in p else None
                ts = p.get("lastUpdated") or p.get("lastUpdatedAt") or p.get("date") or p.get("time")
                # if ts is dict like {"utc":...}
                if isinstance(ts, dict):
                    ts = ts.get("utc") or ts.get("local")
                rows.append({"city": city or filename_hint.split("_")[0], "time": ts, "parameter": param, "value": val})
        else:
            # attempt to find top-level pollutant keys
            for possible_param in ["pm25", "pm2_5", "pm10", "co", "no2", "so2", "o3", "uv", "uv_index","carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]:
                if possible_param in loc:
                    rows.append({"city": city or filename_hint.split("_")[0], "time": loc.get("lastUpdated") or loc.get("updated_at"), "parameter": possible_param, "value": loc.get(possible_param)})

    if not rows:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.DataFrame(rows)
    # parse times and values
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_convert(timezone.utc) if df["time"].notna().any() else pd.NaT
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # normalize parameter names to our pollutant columns
    def map_param(p):
        if not isinstance(p, str):
            return None
        p = p.strip().lower()
        if p in ("pm25", "pm2.5", "pm2_5"):
            return "pm2_5"
        if p in ("pm10",):
            return "pm10"
        if p in ("co", "carbon_monoxide", "carbonmonoxide"):
            return "carbon_monoxide"
        if p in ("no2", "nitrogen_dioxide"):
            return "nitrogen_dioxide"
        if p in ("so2", "sulphur_dioxide"):
            return "sulphur_dioxide"
        if p in ("o3", "ozone"):
            return "ozone"
        if p in ("uv", "uv_index"):
            return "uv_index"
        return None

    df["param_col"] = df["parameter"].map(map_param)
    df = df[df["param_col"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=OUT_COLS)
    # pivot by city + time to have pollutant columns
    pivot = (
        df.groupby(["city", "time", "param_col"])["value"]
        .mean()
        .reset_index()
        .pivot_table(index=["city", "time"], columns="param_col", values="value")
        .reset_index()
    )
    # ensure columns present
    for c in POLLUTANT_COLS:
        if c not in pivot.columns:
            pivot[c] = np.nan
    # reorder
    pivot = pivot[["city", "time"] + POLLUTANT_COLS]
    return pivot

#  Main transform 
def run_transform(save_path: Path = None) -> Path:
    files = _discover_raw_files()
    if not files:
        logging.warning("No raw files found.")
    fragments = []
    for f in files:
        try:
            payload = json.load(f.open("r", encoding="utf-8"))
        except Exception as e:
            logging.warning("Failed to parse %s: %s", f, e)
            continue

        hint = f.stem  # filename without extension
        # detect shape: hourly-array style has 'hourly' key with 'time' list
        if isinstance(payload, dict) and "hourly" in payload and isinstance(payload["hourly"], dict) and "time" in payload["hourly"]:
            df_frag = _parse_hourly_array_payload(payload, filename_hint=hint)
            logging.info("Parsed hourly-array shape from %s -> rows=%d", f.name, len(df_frag))
        else:
            df_frag = _parse_measurement_list_payload(payload, filename_hint=hint)
            logging.info("Parsed measurement-list shape from %s -> rows=%d", f.name, len(df_frag))
        if not df_frag.empty:
            fragments.append(df_frag)

    if not fragments:
        # save empty CSV with headers
        staged = STAGED_DIR / f"air_quality_t_{NOW_TS}.csv"
        pd.DataFrame(columns=["city","time"] + POLLUTANT_COLS + ["hour","aqi_pm25","severity","risk_class"]).to_csv(staged, index=False)
        logging.warning("No measurement rows extracted. Wrote empty staged file: %s", staged)
        return staged

    df_all = pd.concat(fragments, ignore_index=True)
    # ensure time is datetime UTC
    df_all["time"] = pd.to_datetime(df_all["time"], errors="coerce").dt.tz_convert(timezone.utc)
    # convert pollutant columns numeric
    for c in POLLUTANT_COLS:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # remove rows where ALL pollutant readings are missing
    df_all = df_all[df_all[POLLUTANT_COLS].notna().any(axis=1)].copy()

    if df_all.empty:
        staged = STAGED_DIR / f"air_quality_t_{NOW_TS}.csv"
        pd.DataFrame(columns=["city","time"] + POLLUTANT_COLS + ["hour","aqi_pm25","severity","risk_class"]).to_csv(staged, index=False)
        logging.warning("After dropping all-missing rows, nothing remains. Wrote empty staged file: %s", staged)
        return staged

    # If there are multiple records per city+time, aggregate by mean
    df_all = (
        df_all.groupby(["city", "time"], as_index=False)[POLLUTANT_COLS]
        .mean()
    )

    # derived features
    df_all["hour"] = df_all["time"].dt.hour
    df_all["aqi_pm25"] = df_all["pm2_5"].apply(aqi_category_pm25)
    df_all["severity"] = df_all.apply(severity_score, axis=1)
    df_all["risk_class"] = df_all["severity"].apply(risk_class_from_severity)

    # reorder and save
    out_cols = ["city", "time", "hour"] + POLLUTANT_COLS + ["aqi_pm25", "severity", "risk_class"]
    df_out = df_all[out_cols]
    staged_path = save_path or STAGED_DIR / f"air_quality_t_{NOW_TS}.csv"
    df_out.to_csv(staged_path, index=False)
    logging.info("Saved staged file %s (rows=%d)", staged_path, len(df_out))
    return staged_path

if __name__ == "__main__":
    logging.info("Starting transform...")
    staged = run_transform()
    logging.info("Transform finished. Staged: %s", staged)