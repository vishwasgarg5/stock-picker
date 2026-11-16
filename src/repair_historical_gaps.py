"""Repair missing NSE-session OHLCV bars from Yahoo and official NSE bhavcopies."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
import time
import re

import numpy as np
import pandas as pd
import requests

from .historical_coverage_audit import UNIVERSE_FILE, _expected_sessions, run as run_coverage_audit
from .pipeline import HISTORY_FILE, download_history

NSE_UDIFF_CUTOVER = date(2024, 7, 8)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockPickerHistoricalAudit/1.0)",
    "Accept": "text/csv,application/zip,*/*",
    "Referer": "https://www.nseindia.com/",
}


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


def nse_bhavcopy_url(session: date | pd.Timestamp) -> str:
    """Return the official NSE daily cash-market bhavcopy archive URL."""
    day = pd.Timestamp(session).date()
    if day < NSE_UDIFF_CUTOVER:
        month = day.strftime("%b").upper()
        return (
            "https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{day:%Y}/{month}/cm{day:%d}{month}{day:%Y}bhav.csv.zip"
        )
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    )


def _parse_nse_trade_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates as year-first and legacy NSE dates as day-first."""
    parsed = []
    for value in values:
        raw = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            parsed.append(pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce"))
        elif re.fullmatch(r"\d{4}/\d{2}/\d{2}", raw):
            parsed.append(pd.to_datetime(raw, format="%Y/%m/%d", errors="coerce"))
        else:
            parsed.append(pd.to_datetime(raw, errors="coerce", dayfirst=True))
    return pd.Series(pd.to_datetime(parsed), index=values.index)


def normalise_nse_bhavcopy(frame: pd.DataFrame, session: date | pd.Timestamp) -> pd.DataFrame:
    """Map legacy and UDiFF NSE bhavcopy schemas after verifying the file date."""
    columns = {str(column).strip().lower(): column for column in frame.columns}
    requested_day = pd.Timestamp(session).normalize()
    date_column = next((columns[name] for name in ("traddt", "timestamp", "date") if name in columns), None)
    if date_column is None:
        raise ValueError("NSE bhavcopy is missing its trade-date field")
    source_dates = _parse_nse_trade_dates(frame[date_column]).dropna()
    source_dates = source_dates.dt.tz_localize(None).dt.normalize()
    if source_dates.empty or source_dates.nunique() != 1 or source_dates.iloc[0] != requested_day:
        observed = sorted({value.date().isoformat() for value in source_dates})
        raise ValueError(f"NSE bhavcopy trade date mismatch: requested {requested_day.date()}, observed {observed}")
    aliases = {
        "symbol": ("symbol", "tckrsymb"),
        "series": ("series", "sctysrs"),
        "open": ("open", "opnpric"),
        "high": ("high", "hghpric"),
        "low": ("low", "lwpric"),
        "close": ("close", "clspric"),
        "volume": ("tottrdqty", "ttltradgvol"),
    }
    selected: dict[str, Any] = {}
    for target, candidates in aliases.items():
        source = next((columns[name] for name in candidates if name in columns), None)
        if source is None and target != "series":
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        if source is not None:
            selected[target] = source

    result = pd.DataFrame()
    for target, source in selected.items():
        result[target] = frame[source]
    if "series" in result.columns:
        result["series"] = result["series"].astype(str).str.upper().str.strip()
        result = result.loc[result["series"].isin({"EQ", "BE", "BZ"})].copy()
        result["_series_priority"] = result["series"].map({"EQ": 0, "BE": 1, "BZ": 2}).fillna(9)
        result = result.sort_values(["symbol", "_series_priority"]).drop_duplicates("symbol", keep="first")
        result = result.drop(columns=["series", "_series_priority"])
    result["date"] = pd.Timestamp(session).normalize()
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    for column in ["dividends", "stock_splits"]:
        result[column] = 0.0
    return _valid_source_rows(result)


def fetch_nse_bhavcopy(session: date | pd.Timestamp) -> pd.DataFrame:
    """Fetch and parse one source-backed NSE cash-market session with bounded retries."""
    response = None
    for attempt in range(3):
        try:
            response = requests.get(nse_bhavcopy_url(session), headers=NSE_HEADERS, timeout=45)
            if response.status_code == 404:
                response.raise_for_status()
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2 or (response is not None and response.status_code == 404):
                raise
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"NSE bhavcopy request failed for {session}")
    try:
        with ZipFile(BytesIO(response.content)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"NSE bhavcopy archive for {session} contains no CSV")
            with archive.open(csv_names[0]) as source:
                raw = pd.read_csv(source)
    except BadZipFile as exc:
        raise ValueError(f"NSE bhavcopy archive for {session} was not a valid ZIP") from exc
    return normalise_nse_bhavcopy(raw, session)


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

    yahoo_parts = []
    yahoo_rows_downloaded = 0
    for start, symbols in sorted(groups.items()):
        fresh = download_history(sorted(symbols), start)
        yahoo_rows_downloaded += len(fresh)
        if fresh.empty:
            continue
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        fresh["symbol"] = fresh["symbol"].astype(str).str.upper().str.strip()
        fresh = _valid_source_rows(fresh)
        fresh = fresh.loc[
            [(str(row.symbol), pd.Timestamp(row.date).normalize()) in requested for row in fresh.itertuples()]
        ]
        if not fresh.empty:
            yahoo_parts.append(fresh)

    yahoo_recovered = pd.concat(yahoo_parts, ignore_index=True) if yahoo_parts else pd.DataFrame()
    yahoo_keys = {
        (str(row.symbol), pd.Timestamp(row.date).normalize())
        for row in yahoo_recovered.itertuples()
    } if not yahoo_recovered.empty else set()
    still_requested = requested - yahoo_keys

    # NSE publishes one full-market bhavcopy per trading day. Fetch each
    # missing date once and extract only the requested symbols for that date.
    nse_parts = []
    nse_dates_checked = 0
    nse_dates_unavailable: list[str] = []
    nse_fetch_failures: list[dict[str, str]] = []
    for day in sorted({day for _, day in still_requested}):
        nse_dates_checked += 1
        try:
            market = fetch_nse_bhavcopy(day)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                nse_dates_unavailable.append(day.date().isoformat())
            else:
                nse_fetch_failures.append({"date": day.date().isoformat(), "error": str(exc)[:300]})
            continue
        except Exception as exc:
            nse_fetch_failures.append({"date": day.date().isoformat(), "error": str(exc)[:300]})
            continue
        if market.empty:
            nse_dates_unavailable.append(day.date().isoformat())
            continue
        keys = [
            (str(row.symbol), pd.Timestamp(row.date).normalize()) in still_requested
            for row in market.itertuples()
        ]
        selected = market.loc[keys].copy()
        if not selected.empty:
            nse_parts.append(selected)

    nse_recovered = pd.concat(nse_parts, ignore_index=True) if nse_parts else pd.DataFrame()
    parts = [part for part in [yahoo_recovered, nse_recovered] if not part.empty]
    recovered = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=history.columns)
    if not recovered.empty:
        for column in history.columns:
            if column not in recovered.columns:
                recovered[column] = 0.0 if column in {"dividends", "stock_splits"} else np.nan
        recovered = recovered[history.columns]
        combined = pd.concat([history, recovered], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
        # Existing observations always win; repairs only fill absent keys.
        combined = combined.drop_duplicates(["date", "symbol"], keep="first").sort_values(["symbol", "date"])
        temporary = Path(HISTORY_FILE).with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(HISTORY_FILE)

    after = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    remaining = find_missing_sessions(universe, after)
    remaining_keys = {(symbol, day.normalize()) for symbol, days in remaining.items() for day in days}
    recovered_keys = {
        (str(row.symbol), pd.Timestamp(row.date).normalize())
        for row in recovered.itertuples()
    } if not recovered.empty else set()
    return {
        "status": "PASS" if not remaining_keys else "REVIEW_REQUIRED",
        "symbols_with_gaps_before": sum(bool(days) for days in missing.values()),
        "missing_sessions_before": len(requested),
        "yahoo_rows_downloaded": int(yahoo_rows_downloaded),
        "yahoo_sessions_recovered": len(yahoo_keys & requested),
        "nse_sessions_checked": nse_dates_checked,
        "nse_sessions_recovered": int(len(nse_recovered)),
        "nse_dates_unavailable": nse_dates_unavailable,
        "nse_fetch_failures": nse_fetch_failures,
        "source_rows_recovered": len(recovered),
        "missing_sessions_recovered": len(recovered_keys & requested),
        "missing_sessions_remaining": len(remaining_keys),
        "symbols_still_with_gaps": sorted({symbol for symbol, _ in remaining_keys}),
        "policy": "Only validated source-returned Yahoo Finance or official NSE bhavcopy bars are inserted; no interpolation or synthetic OHLCV.",
        "coverage_audit": run_coverage_audit(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(repair_historical_gaps(), indent=2, default=str))
