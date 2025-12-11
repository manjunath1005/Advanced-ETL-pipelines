"""
Load staged air quality CSV into Supabase (Postgres) using upsert on (city, time).

Environment variables:
  SUPABASE_URL     - your Supabase project URL (e.g. https://xyz.supabase.co)
  SUPABASE_KEY     - service_role key (recommended) or anon key (ensure RLS policy allows inserts)
  STAGED_DIR       - optional override for staged directory (default: data/staged)
  BATCH_SIZE       - number of rows per insert batch (default: 200)
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from supabase import create_client  # supabase-py

from dotenv import load_dotenv
load_dotenv()

# -- Config --
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("SUPABASE_URL and SUPABASE_KEY must be set in environment (.env or envvars)")

STAGED_DIR = Path(os.getenv("STAGED_DIR", "data/staged"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
TABLE_NAME = os.getenv("SUPABASE_TABLE", "air_quality")
MAX_INSERT_RETRIES = int(os.getenv("MAX_INSERT_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = int(os.getenv("RETRY_BACKOFF_SECONDS", "3"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _find_latest_staged() -> Path:
    files = sorted(STAGED_DIR.glob("air_quality_t_*.csv"))
    if not files:
        raise FileNotFoundError(f"No staged files found in {STAGED_DIR}")
    return files[-1]


def _prepare_records_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert dataframe rows into JSON-serializable dicts compatible with Supabase.
    Ensures:
      - time is ISO string (with timezone) or None
      - numeric types are Python floats/ints or None
      - NaN / numpy.nan are converted to None
    """
    df = df.copy()

    # Ensure time is timezone-aware; if not, parse and assume UTC
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["time"] = df["time"].apply(lambda t: t.isoformat() if pd.notnull(t) else None)

    # Ensure expected columns exist
    expected_cols = [
        "city",
        "time",
        "hour",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "uv_index",
        "aqi_pm25",
        "severity",
        "risk_class",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # Replace NaN/inf with None (works for numpy.nan / pandas NA)
    df = df.replace([pd.NA, pd.NaT], None)
    df = df.where(pd.notnull(df), None)

    # Convert numpy types to native python types in each record
    records: List[Dict[str, Any]] = df[expected_cols].to_dict(orient="records")
    clean_records: List[Dict[str, Any]] = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            # convert numpy types to native python types
            if v is None:
                clean_rec[k] = None
                continue
            # handle pandas/numpy scalars
            try:
                # This will turn numpy.int64 / numpy.float64 into Python int/float
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            # Guard against remaining NaN or inf
            if isinstance(v, float):
                if pd.isna(v) or v != v:  # v != v is True for NaN
                    clean_rec[k] = None
                    continue
                if v == float("inf") or v == float("-inf"):
                    clean_rec[k] = None
                    continue
            clean_rec[k] = v
        clean_records.append(clean_rec)

    return clean_records


def _insert_batch_upsert(records: List[Dict[str, Any]], on_conflict: str = "city,time") -> None:
    """
    Upsert a batch of records into Supabase table.
    Retries on transient errors.
    """
    attempt = 0
    while attempt < MAX_INSERT_RETRIES:
        attempt += 1
        try:
            # supabase.table(table).upsert(records, on_conflict="city,time").execute()
            res = supabase.table(TABLE_NAME).upsert(records, on_conflict=on_conflict).execute()
            # older versions returned res.error; newer return dict with 'error' and 'status_code'
            # examine both possibilities
            err = getattr(res, "error", None) if hasattr(res, "__dict__") else (res.get("error") if isinstance(res, dict) else None)
            if err:
                logging.error("Supabase returned error on upsert: %s", err)
                raise Exception(f"Upsert error: {err}")
            # success
            logging.info("Inserted batch of %d rows (attempt %d)", len(records), attempt)
            return
        except Exception as e:
            logging.warning("Upsert attempt %d failed: %s", attempt, e)
            if attempt >= MAX_INSERT_RETRIES:
                logging.error("Upsert failed after %d attempts. Error: %s", attempt, e)
                raise
            backoff = RETRY_BACKOFF_SECONDS * attempt
            logging.info("Retrying upsert in %d seconds...", backoff)
            time.sleep(backoff)


def load_staged_to_supabase(staged_path: Path | None = None) -> None:
    staged_path = staged_path or _find_latest_staged()
    logging.info("Loading staged CSV: %s", staged_path)
    df = pd.read_csv(staged_path)

    if df.empty:
        logging.warning("Staged CSV is empty. Nothing to load.")
        return

    records = _prepare_records_from_df(df)
    logging.info("Prepared %d records for upsert.", len(records))

    # Insert in batches
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        logging.info("Upserting records %d - %d ...", i, i + len(batch) - 1)
        _insert_batch_upsert(batch, on_conflict="city,time")

    logging.info("Load complete.")


if __name__ == "__main__":
    logging.info("Starting load step to Supabase table: %s", TABLE_NAME)
    try:
        load_staged_to_supabase(None)
    except Exception as e:
        logging.exception("Load failed: %s", e)
        raise 