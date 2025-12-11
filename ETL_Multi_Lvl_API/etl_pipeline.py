
# run_pipeline.py
"""
Run the full ETL pipeline: extract -> transform -> load -> analysis.

Behavior:
 - Preferred: import functions from local modules:
     extract.fetch_all_cities()
     transform.run_transform()
     load.load_staged_to_supabase()
     etl_analysis.run_etl_analysis()  (or analysis.run_analysis())
 - Fallback: call scripts with subprocess: python extract.py, python transform.py, etc.
 - Logs durations and basic success/failure info.
 - Exit code != 0 on error.

Usage:
    python run_pipeline.py
Environment:
    If you want to skip load step set SKIP_LOAD=1
    If you want to skip analysis set SKIP_ANALYSIS=1
"""
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _call_or_subprocess(module_name: str, func_name: str, script_name: str, *args, **kwargs):
    """
    Try to import module and call func_name(*args, **kwargs).
    If import fails or function not found, run `python script_name` as a subprocess.
    Returns (success: bool, result_or_output: any)
    """
    try:
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name, None)
        if callable(func):
            logging.info("Calling %s.%s(...)", module_name, func_name)
            res = func(*args, **kwargs)
            return True, res
        else:
            logging.warning("Module %s loaded but function %s not found. Falling back to subprocess.", module_name, func_name)
    except Exception as e:
        logging.info("Could not import %s (%s). Falling back to subprocess.", module_name, e)

    # Fallback to subprocess
    logging.info("Running script fallback: python %s", script_name)
    try:
        proc = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        logging.info("Subprocess %s finished (returncode=%d).", script_name, proc.returncode)
        return True, proc.stdout + proc.stderr
    except subprocess.CalledProcessError as e:
        logging.error("Subprocess %s failed (returncode=%s). Stderr:\n%s", script_name, e.returncode, e.stderr)
        return False, e.stderr


def step(name: str, runner: Callable[[], tuple], fail_on_error: bool = True):
    logging.info("=== STEP: %s ===", name)
    start = time.time()
    try:
        ok, res = runner()
    except Exception as e:
        logging.exception("Exception while running step %s: %s", name, e)
        ok = False
        res = str(e)
    duration = time.time() - start
    if ok:
        logging.info("STEP %s completed in %.2fs", name, duration)
    else:
        logging.error("STEP %s FAILED in %.2fs — error summary: %s", name, duration, res)
        if fail_on_error:
            logging.error("Aborting pipeline due to failure in step: %s", name)
            sys.exit(1)
    return ok, res


def run_extract():
    # prefer extract.fetch_all_cities -> returns list of dicts or file paths
    return _call_or_subprocess("extract", "fetch_all_cities", "extract.py")


def run_transform():
    # prefer transform.run_transform -> returns staged path
    # allow optional argless call
    return _call_or_subprocess("transform", "run_transform", "transform.py")


def run_load():
    # prefer load.load_staged_to_supabase -> no args
    # older script used load_staged_to_supabase(staged_path=None)
    return _call_or_subprocess("load", "load_staged_to_supabase", "load.py")


def run_analysis():
    # prefer etl_analysis.run_etl_analysis else analysis.run_analysis
    ok, res = _call_or_subprocess("etl_analysis", "run_etl_analysis", "etl_analysis.py")
    if ok:
        return True, res
    # fallback to analysis.py
    return _call_or_subprocess("analysis", "run_analysis", "analysis.py")


def main():
    logging.info("Starting full ETL pipeline...")
    pipeline_start = time.time()

    # Step 1: Extract
    step("extract", lambda: run_extract())

    # Step 2: Transform
    step("transform", lambda: run_transform())

    # Step 3: Load (skip if SKIP_LOAD set)
    import os

    if os.getenv("SKIP_LOAD", "0") in ("1", "true", "True"):
        logging.info("SKIP_LOAD is set — skipping load step.")
    else:
        step("load", lambda: run_load())

    # Step 4: Analysis (skip if SKIP_ANALYSIS set)
    if os.getenv("SKIP_ANALYSIS", "0") in ("1", "true", "True"):
        logging.info("SKIP_ANALYSIS is set — skipping analysis step.")
    else:
        step("analysis", lambda: run_analysis())

    total = time.time() - pipeline_start
    logging.info("ETL pipeline finished successfully in %.2fs", total)


if __name__ == "__main__":
    main()
