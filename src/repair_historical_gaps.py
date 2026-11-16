"""Repair missing NSE-session OHLCV bars using Yahoo Finance source data only."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .historical_coverage_audit import UNIVERSE_FILE, _expected_sessions, run as run_coverage_audit
from .pipeline import HISTORY_FILE, download_history


def find_missing_sessions(universe: pd.DataFrame, history: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    symbols = (
        universe["symbol"].dropna().astype(str).str.upper().str.strip()
        .loc[lambda x: x.ne("") & ~x.str.startswith("DUMMY")]
        .drop_duplicates().tolist()
    )
    if history.empty or not {"date", "symbol"}.issubset(history.columns):
        return {symbol: [] for symbol in symbols}
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data = data.dropna(subset=["date"])
    result: dict[str, list[pd.Timestamp]] = {}
    for symbol in symbols:
        observed = pd.DatetimeIndex(data.loc[data["symbol"] == symbol, "date"].drop_duplicates()).sort_values()
        if observed.empty:
            result[symbol] = []
            continue
        expected = _expected_sessions(observed.min(), observed.max())
        result[symbol] = list(expected.difference(observed))
    return result


def _valid_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(rows.columns):
        return rows.iloc[0:0].copy()
    out = rows.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    prices = out[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    valid = np.isfinite(prices).all(axis=1) & (prices[:, :4] > 0).all(axis=1) & (prices[:, 4] >= 0)
    valid &= out["high"].ge(out[["open", "close"]].max(axis=1)).to_numpy()
    valid &= out["low"].le(out[["open", "close"]].min(axis=1)).to_numpy()
    return out.loc[valid].copy()


def repair_historical_gaps() -> dict[str, Any]:
    universe = pd.read_csv(UNIVERSE_FILE)
    try:
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        history = pd.DataFrame()
    if history.empty:
        raise RuntimeError("OHLCV history is empty; run the full historical recovery first")

    missing = find_missing_sessions(universe, history)
    requested = {(symbol, day.normalize()) for symbol, days in missing.items() for day in days}
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol, days in missing.items():
        if days:
            groups[min(days).strftime("%Y-%m-%d")].append(symbol)

    recovered_parts = []
    for start, symbols in sorted(groups.items()):
        fresh = download_history(sorted(symbols), start)
        if fresh.empty:
            continue
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        fresh["symbol"] = fresh["symbol"].astype(str).str.upper().str.strip()
        fresh = _valid_source_rows(fresh)
        fresh = fresh.loc[
            [(str(row.symbol), pd.Timestamp(row.date).normalize()) in requested for row in fresh.itertuples()]
        ]
        if not fresh.empty:
            recovered_parts.append(fresh)

    recovered = pd.concat(recovered_parts, ignore_index=True) if recovered_parts else pd.DataFrame(columns=history.columns)
    if not recovered.empty:
        combined = pd.concat([history, recovered], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
        combined = combined.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
        temporary = Path(HISTORY_FILE).with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(HISTORY_FILE)

    after = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    remaining = find_missing_sessions(universe, after)
    remaining_keys = {(symbol, day.normalize()) for symbol, days in remaining.items() for day in days}
    recovered_keys = {
        (str(row.symbol), pd.Timestamp(row.date).normalize())
        for row in recovered.itertuples()
    }
    return {
        "status": "PASS" if not remaining_keys else "REVIEW_REQUIRED",
        "symbols_with_gaps_before": sum(bool(days) for days in missing.values()),
        "missing_sessions_before": len(requested),
        "source_rows_recovered": len(recovered),
        "missing_sessions_recovered": len(recovered_keys & requested),
        "missing_sessions_remaining": len(remaining_keys),
        "symbols_still_with_gaps": sorted({symbol for symbol, _ in remaining_keys}),
        "policy": "Only validated Yahoo Finance bars for missing NSE sessions are inserted; no interpolation or synthetic OHLCV.",
        "coverage_audit": run_coverage_audit(),
    }
