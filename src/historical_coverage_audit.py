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
REPORT_JSON = DATA / "historical_coverage_audit.json"
REPORT_CSV = DATA / "historical_coverage_audit.csv"
SESSIONS_PER_YEAR = 252
MIN_TRAINING_SESSIONS = 50


def _expected_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return NSE sessions in the observed interval; fail explicitly if calendar unavailable."""
    calendar = mcal.get_calendar("NSE")
    schedule = calendar.schedule(start_date=start.date(), end_date=end.date())
    return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()


def audit_history(
    universe: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if "symbol" not in universe.columns:
        raise ValueError("Universe file must contain a symbol column")
    symbols = (
        universe["symbol"].dropna().astype(str).str.upper().str.strip()
        .loc[lambda values: values.ne("") & ~values.str.startswith("DUMMY")]
        .drop_duplicates().tolist()
    )
    if not symbols:
        raise ValueError("Universe contains no valid symbols")

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required.difference(history.columns))
    if missing_columns:
        history = pd.DataFrame(columns=sorted(required))
    else:
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        history["symbol"] = history["symbol"].astype(str).str.upper().str.strip()
        for col in ["open", "high", "low", "close", "volume"]:
            history[col] = pd.to_numeric(history[col], errors="coerce")
        history = history.dropna(subset=["date", "symbol"])

    relevant = history[history["symbol"].isin(symbols)].copy() if not history.empty else history
    valid = relevant
    if not valid.empty and {"open", "high", "low", "close", "volume"}.issubset(valid.columns):
        prices = valid[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        valid = valid.loc[np.isfinite(prices).all(axis=1) & (prices[:, :4] > 0).all(axis=1) & (prices[:, 4] >= 0)].copy()
        valid = valid.loc[
            valid["high"].ge(valid[["open", "close"]].max(axis=1))
            & valid["low"].le(valid[["open", "close"]].min(axis=1))
        ].copy()

    if valid.empty:
        rows = [{
            "symbol": symbol, "first_date": None, "last_date": None,
            "observed_sessions": 0, "expected_sessions": 0, "missing_sessions": None,
            "session_coverage_pct": 0.0, "history_years": 0.0,
            "at_least_1y": False, "at_least_3y": False,
            "at_least_5y": False, "at_least_8y": False,
            "training_ready_50_sessions": False, "status": "NO_HISTORY",
        } for symbol in symbols]
        detail = pd.DataFrame(rows)
        summary = {
            "status": "BLOCKED_NO_HISTORY", "universe_symbols": len(symbols),
            "symbols_with_history": 0, "symbols_without_history": len(symbols),
            "history_start": None, "history_end": None, "total_valid_rows": 0,
            "minimum_training_sessions": MIN_TRAINING_SESSIONS,
            "depth_bands": {"1y": 0, "3y": 0, "5y": 0, "8y": 0},
            "missing_universe_symbols": symbols,
            "invalid_or_excluded_rows": int(len(relevant)),
            "calendar": "NSE exchange sessions",
            "note": "No OHLCV observations found; no history was inferred or imputed.",
        }
        return summary, detail

    overall_end = valid["date"].max()
    overall_start = valid["date"].min()
    calendar = mcal.get_calendar("NSE")
    expected_all = _expected_sessions(overall_start, overall_end)
    records = []
    for symbol in symbols:
        stock = valid.loc[valid["symbol"] == symbol].drop_duplicates("date").sort_values("date")
        if stock.empty:
            records.append({
                "symbol": symbol, "first_date": None, "last_date": None,
                "observed_sessions": 0, "expected_sessions": 0, "missing_sessions": None,
                "session_coverage_pct": 0.0, "history_years": 0.0,
                "at_least_1y": False, "at_least_3y": False,
                "at_least_5y": False, "at_least_8y": False,
                "training_ready_50_sessions": False, "status": "NO_HISTORY",
            })
            continue
        first = stock["date"].min()
        last = stock["date"].max()
        expected = _expected_sessions(first, last)
        observed_dates = pd.DatetimeIndex(stock["date"].drop_duplicates())
        observed_in_range = observed_dates.intersection(expected)
        expected_count = len(expected)
        observed_count = len(observed_in_range)
        count = len(stock)
        years = count / SESSIONS_PER_YEAR
        records.append({
            "symbol": symbol,
            "first_date": first.date().isoformat(),
            "last_date": last.date().isoformat(),
            "observed_sessions": count,
            "expected_sessions": expected_count,
            "missing_sessions": max(0, expected_count - observed_count),
            "session_coverage_pct": round(observed_count / expected_count * 100, 2) if expected_count else 0.0,
            "history_years": round(years, 2),
            "at_least_1y": count >= SESSIONS_PER_YEAR,
            "at_least_3y": count >= 3 * SESSIONS_PER_YEAR,
            "at_least_5y": count >= 5 * SESSIONS_PER_YEAR,
            "at_least_8y": count >= 8 * SESSIONS_PER_YEAR,
            "training_ready_50_sessions": count >= MIN_TRAINING_SESSIONS,
            "status": "PASS" if count >= MIN_TRAINING_SESSIONS else "INSUFFICIENT_HISTORY",
        })

    detail = pd.DataFrame(records).sort_values(
        ["observed_sessions", "symbol"], ascending=[True, True]
    ).reset_index(drop=True)
    counts = {
        f"{years}y": int(detail[f"at_least_{years}y"].sum())
        for years in (1, 3, 5, 8)
    }
    with_history = int((detail["observed_sessions"] > 0).sum())
    summary = {
        "status": "PASS" if with_history == len(symbols) and bool(detail["training_ready_50_sessions"].all()) else "REVIEW_REQUIRED",
        "universe_symbols": len(symbols),
        "symbols_with_history": with_history,
        "symbols_without_history": len(symbols) - with_history,
        "symbols_training_ready_50_sessions": int(detail["training_ready_50_sessions"].sum()),
        "history_start": overall_start.date().isoformat(),
        "history_end": overall_end.date().isoformat(),
        "total_valid_rows": int(len(valid)),
        "minimum_training_sessions": MIN_TRAINING_SESSIONS,
        "depth_bands": counts,
        "median_observed_sessions": int(detail["observed_sessions"].median()),
        "median_history_years": round(float(detail["history_years"].median()), 2),
        "missing_universe_symbols": detail.loc[detail["observed_sessions"] == 0, "symbol"].tolist(),
        "invalid_or_excluded_rows": int(len(relevant) - len(valid)),
        "calendar": "NSE exchange sessions",
        "note": "Year bands use 252 observed trading sessions per year; session gaps use the NSE calendar.",
    }
    return summary, detail


def run() -> dict[str, Any]:
    universe = pd.read_csv(UNIVERSE_FILE)
    try:
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        history = pd.DataFrame()
    summary, detail = audit_history(universe, history)
    summary["detail_file"] = str(REPORT_CSV.relative_to(ROOT))
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    detail.to_csv(REPORT_CSV, index=False)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
