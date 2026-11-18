"""Per-symbol historical OHLCV coverage audit for the active stock universe.

The audit measures observed bars, expected NSE sessions, history-depth bands and
minimum feature readiness. It never fills or fabricates historical observations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNIVERSE_FILE = DATA / "universe.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
LISTING_MASTER_FILE = DATA / "nse_listing_master.csv"
REPORT_JSON = DATA / "historical_coverage_audit.json"
REPORT_CSV = DATA / "historical_coverage_audit.csv"
from common.historical_coverage import (
    MIN_TRAINING_SESSIONS,
    SESSIONS_PER_YEAR,
    _add_listing_coverage,
    _expected_sessions,
    audit_history,
)


def run() -> dict[str, Any]:
    universe = pd.read_csv(UNIVERSE_FILE)
    try:
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        history = pd.DataFrame()
    from .listing_master import refresh_listing_master
    symbols = universe["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    listing_refresh = refresh_listing_master(symbols)
    try:
        listing_master = pd.read_csv(LISTING_MASTER_FILE, dtype=str)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        listing_master = pd.DataFrame(columns=["symbol", "listing_date"])
    summary, detail = audit_history(universe, history, listing_master)
    summary["listing_master_refresh"] = listing_refresh
    summary["listing_master_file"] = str(LISTING_MASTER_FILE.relative_to(ROOT))
    summary["detail_file"] = str(REPORT_CSV.relative_to(ROOT))
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    detail.to_csv(REPORT_CSV, index=False)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
