"""

Extract step for Urban Air Quality Monitoring ETL.
 
- Calls OpenAQ public API (no auth) for a list of cities.

- Implements retry with exponential backoff (default 3 attempts).

- Saves each raw response to data/raw/<city>_raw_<timestamp>.json

- Returns a list of saved file paths.
 
Usage:

    from extract import fetch_all_cities

    saved = fetch_all_cities()  # returns list of dicts [{'city':'Delhi','raw_path':'...'}, ...]

    # Or run from CLI: python extract.py

"""
 
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv
 
load_dotenv()
 
# Config (override via .env)

RAW_DIR = Path(os.getenv("RAW_DIR", Path(__file__).resolve().parents[0] / "data" / "raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)
API_BASE = os.getenv("OPENAQ_API_BASE", "https://api.openaq.org/v2/latest")
DEFAULT_CITIES = os.getenv("AQ_CITIES", "Delhi,Bengaluru,Hyderabad,Mumbai,Kolkata").split(",")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "10"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.5"))  # polite pause between requests
 
def _now_ts() -> str:
    """UTC compact timestamp used in filenames."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
 
def _save_raw(payload: object, city: str) -> str:
    """
    Save JSON-serializable payload to RAW_DIR and return absolute path.
    If payload isn't JSON-serializable, fall back to writing string repr.
    """
    ts = _now_ts()
    filename = f"{city.replace(' ', '_').lower()}_raw_{ts}.json"
    path = RAW_DIR / filename

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    except Exception:
        # fallback to plain text
        path = RAW_DIR / f"{city.replace(' ', '_').lower()}_raw_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(repr(payload))
    return str(path.resolve())
 
def _fetch_city(city: str, max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SECONDS) -> Dict[str, Optional[str]]:
    """
    Fetch OpenAQ latest measurements for a single city with retry logic.
    Returns dict with keys: city, success (bool), raw_path or error.
    """
    params = {"city": city}
    attempt = 0
    last_error: Optional[str] = None
    while attempt < max_retries:
        attempt += 1
        try:
            resp = requests.get(API_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            # parse JSON if possible
            try:
                payload = resp.json()
            except ValueError:
                payload = {"raw_text": resp.text}

            saved = _save_raw(payload, city)
            print(f"✅ [{city}] fetched and saved to: {saved}")
            return {"city": city, "success": "true", "raw_path": saved}
        
        except requests.RequestException as e:
            last_error = str(e)
            print(f"⚠️ [{city}] attempt {attempt}/{max_retries} failed: {e}")

        except Exception as e:
            last_error = str(e)
            print(f"⚠️ [{city}] unexpected error on attempt {attempt}: {e}")
 
        # exponential backoff before next attempt
        backoff = 2 ** (attempt - 1)
        print(f"⏳ [{city}] retrying in {backoff}s ...")
        time.sleep(backoff)

    # exhausted retries
    print(f"❌ [{city}] failed after {max_retries} attempts. Last error: {last_error}")
    return {"city": city, "success": "false", "error": last_error}
 
def fetch_all_cities(cities: Optional[List[str]] = None) -> List[Dict[str, Optional[str]]]:
    """
    Fetch data for all cities (default list in DEFAULT_CITIES).
    Returns list of results dicts for each city.
    """
    if cities is None:
        cities = [c.strip() for c in DEFAULT_CITIES if c.strip()]

    results: List[Dict[str, Optional[str]]] = []
    for city in cities:
        res = _fetch_city(city)
        results.append(res)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results
 
if __name__ == "__main__":
    print("Starting extraction for OpenAQ (no auth required)")
    cities_env = [c.strip() for c in DEFAULT_CITIES if c.strip()]
    print(f"Cities: {cities_env}")
    out = fetch_all_cities(cities_env)
    print("Extraction complete. Summary:")
    for r in out:
        if r.get("success") == "true":
            print(f" - {r['city']}: saved -> {r['raw_path']}")
        else:
            print(f" - {r['city']}: ERROR -> {r.get('error')}")