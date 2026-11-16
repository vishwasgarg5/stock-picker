"""One-off source-backed OHLCV recovery; does not run predictions or trading."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .historical_coverage_audit import run as run_coverage_audit
from .repair_historical_gaps import repair_historical_gaps
from .pipeline import HISTORY_FILE, load_universe, update_history

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = DATA / "historical_recovery_run.json"


def run() -> dict:
    symbols = load_universe()
    try:
        before = pd.read_csv(HISTORY_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        before = pd.DataFrame()

    history = update_history(symbols)
    gap_repair = repair_historical_gaps()
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    coverage = run_coverage_audit()
    symbols_with_history = (
        set(history["symbol"].astype(str).str.upper().str.strip())
        if not history.empty else set()
    )
    missing = sorted(set(symbols) - symbols_with_history)
    result = {
        "status": "PASS" if not missing else "PARTIAL_REVIEW_REQUIRED",
        "source": "Yahoo Finance via src.pipeline.download_history",
        "requested_start_date": "2018-01-01",
        "symbols_requested": len(symbols),
        "symbols_with_history": len(symbols_with_history),
        "symbols_without_history": len(missing),
        "symbols_missing_history": missing,
        "rows_before": int(len(before)),
        "rows_after": int(len(history)),
        "rows_added_or_refreshed": int(max(0, len(history) - len(before))),
        "history_start": str(pd.to_datetime(history["date"]).min().date()) if not history.empty else None,
        "history_end": str(pd.to_datetime(history["date"]).max().date()) if not history.empty else None,
        "coverage_audit": coverage,
        "gap_repair": gap_repair,
        "repair_policy": "Only source-returned OHLCV bars are stored; no price interpolation or synthetic history.",
    }
    REPORT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    run()
