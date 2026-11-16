from __future__ import annotations

"""Point-in-time NSE index OHLCV, constituent and constituent-weight intelligence.

NSE's public constituent endpoint is used for current membership. When it
publishes a constituent weight it is retained as official; otherwise a
free-float-market-cap proxy is explicitly labelled ESTIMATED. Snapshots are
never backfilled into dates before they were observed.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_HISTORY_FILE = DATA / "index_ohlcv.csv"
CONSTITUENTS_FILE = DATA / "index_constituents_history.csv"
FEATURES_FILE = DATA / "index_constituent_features.csv"
AUDIT_FILE = DATA / "index_constituent_intelligence_audit.json"

# Only indexes with a stable Yahoo Finance symbol are enabled for OHLCV
# collection. Missing vendor history remains missing; it is never fabricated.
INDEX_TICKERS = {
    "NIFTY50": ["^NSEI"],
    "NIFTYNEXT50": ["^NSMIDCP", "NIFTY_NEXT_50.NS"],
    "NIFTY100": ["^CNX100"],
    "NIFTY200": ["^CNX200"],
    "NIFTY500": ["^CRSLDX"],
    "BANKNIFTY": ["^NSEBANK"],
    "NIFTYIT": ["^CNXIT"],
    "NIFTYAUTO": ["^CNXAUTO"],
    "NIFTYFIN": ["^CNXFINANCE", "NIFTY_FIN_SERVICE.NS"],
    "NIFTYPHARMA": ["^CNXPHARMA"],
    "NIFTYFMCG": ["^CNXFMCG"],
}
NSE_INDEX_NAMES = {
    "NIFTY50": "NIFTY 50",
    "NIFTYNEXT50": "NIFTY NEXT 50",
    "NIFTY100": "NIFTY 100",
    "NIFTY200": "NIFTY 200",
    "NIFTY500": "NIFTY 500",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTYIT": "NIFTY IT",
    "NIFTYAUTO": "NIFTY AUTO",
    "NIFTYFIN": "NIFTY FINANCIAL SERVICES",
    "NIFTYPHARMA": "NIFTY PHARMA",
    "NIFTYFMCG": "NIFTY FMCG",
}
NSE_HOME = "https://www.nseindia.com/"
NSE_API = "https://www.nseindia.com/api/equity-stock-indices?index={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}
INDEX_COLUMNS = ["date", "index", "ticker", "open", "high", "low", "close", "volume", "source"]
MEMBER_COLUMNS = ["index", "symbol", "as_of_date", "weight_pct", "weight_method", "ffmc", "source"]
FEATURE_COLUMNS = ["date", "index", "symbol", "weight_pct", "weight_method", "index_return_1d_pct", "index_return_5d_pct", "stock_return_1d_pct", "rolling_corr_60d", "rolling_beta_60d", "weighted_stock_return_contribution_pct", "as_of_date"]


def _normalise_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()


def normalise_index_bars(index: str, ticker: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    x = raw.copy()
    x.columns = [str(c).lower().replace(" ", "_") for c in x.columns]
    x["date"] = pd.to_datetime(x.index, errors="coerce", utc=True).tz_convert(None).normalize()
    required = ["open", "high", "low", "close"]
    if not set(required).issubset(x.columns):
        return pd.DataFrame(columns=INDEX_COLUMNS)
    for col in required + (["volume"] if "volume" in x else []):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if "volume" not in x:
        x["volume"] = np.nan
    x["index"] = index
    x["ticker"] = ticker
    x["source"] = "Yahoo Finance daily index OHLCV"
    x = x.dropna(subset=["date", "open", "high", "low", "close"])
    x = x[(x[["open", "high", "low", "close"]] > 0).all(axis=1)]
    x = x.drop_duplicates(["date", "index"], keep="last")
    return x[INDEX_COLUMNS].sort_values("date")


def download_index_history(ticker_map: dict | None = None, downloader=None) -> tuple[pd.DataFrame, dict]:
    tickers = ticker_map or INDEX_TICKERS
    download = downloader or (lambda ticker: yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False))
    rows, audit = [], {}
    for index, alternatives in tickers.items():
        errors, found = [], False
        for ticker in alternatives:
            try:
                raw = download(ticker)
                bars = normalise_index_bars(index, ticker, raw)
                if not bars.empty:
                    rows.append(bars)
                    audit[index] = {"status": "PASS", "ticker": ticker, "rows_downloaded": int(len(bars)),
                                    "first_date": bars.date.min().date().isoformat(),
                                    "last_date": bars.date.max().date().isoformat()}
                    found = True
                    break
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        if not found:
            audit[index] = {"status": "SOURCE_UNAVAILABLE", "rows_downloaded": 0, "errors": errors}
    downloaded = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=INDEX_COLUMNS)
    if INDEX_HISTORY_FILE.exists():
        old = pd.read_csv(INDEX_HISTORY_FILE)
        if not old.empty and set(INDEX_COLUMNS).issubset(old.columns):
            downloaded = pd.concat([old[INDEX_COLUMNS], downloaded], ignore_index=True)
    if not downloaded.empty:
        downloaded["date"] = _normalise_date(downloaded["date"])
        downloaded = downloaded.dropna(subset=["date"]).drop_duplicates(["date", "index"], keep="last")
        downloaded = downloaded.sort_values(["index", "date"])
        INDEX_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        downloaded.to_csv(INDEX_HISTORY_FILE, index=False)
    return downloaded, audit


def normalise_constituent_payload(index: str, payload: dict, as_of_date: str | None = None, source_url: str | None = None) -> pd.DataFrame:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        raise ValueError(f"NSE constituent response for {index} has no data list")
    snapshot_date = pd.Timestamp(as_of_date or datetime.now(ZoneInfo("Asia/Kolkata")).date()).normalize()
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        symbol = str(item.get("symbol") or meta.get("symbol") or "").strip().upper()
        if not symbol or symbol in {index, NSE_INDEX_NAMES.get(index, "").upper(), "NIFTY 50"}:
            continue
        ffmc_raw = item.get("ffmc") or item.get("marketCap") or item.get("market_cap")
        try:
            ffmc = float(str(ffmc_raw).replace(",", "")) if ffmc_raw not in (None, "") else np.nan
        except (TypeError, ValueError):
            ffmc = np.nan
        weight_raw = item.get("weightage") or item.get("weight") or item.get("weight_pct")
        try:
            weight = float(str(weight_raw).replace("%", "").replace(",", "")) if weight_raw not in (None, "") else np.nan
        except (TypeError, ValueError):
            weight = np.nan
        records.append({"index": index, "symbol": symbol, "as_of_date": snapshot_date,
                        "weight_pct": weight, "weight_method": "OFFICIAL_NSE_WEIGHT" if np.isfinite(weight) else "PENDING_WEIGHT",
                        "ffmc": ffmc, "source": source_url or NSE_API.format(NSE_INDEX_NAMES[index].replace(" ", "%20"))})
    out = pd.DataFrame(records, columns=MEMBER_COLUMNS)
    if out.empty:
        return out
    out = out.drop_duplicates(["index", "symbol"], keep="last")
    # NSE's free-float market-cap field can provide a useful proxy for
    # free-float-cap-weighted indices. It is explicitly not represented as
    # an official weight, and is not substituted for capped/equal-weight rules.
    cap_weighted = index in {"NIFTY50", "NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500", "BANKNIFTY", "NIFTYIT", "NIFTYAUTO", "NIFTYFIN", "NIFTYPHARMA", "NIFTYFMCG"}
    pending = out["weight_pct"].isna()
    total_ffmc = pd.to_numeric(out["ffmc"], errors="coerce").sum(min_count=1)
    official_total = pd.to_numeric(out.loc[~pending, "weight_pct"], errors="coerce").sum(min_count=1)
    residual_weight = max(0.0, 100.0 - float(official_total or 0.0)) if pd.notna(official_total) else 100.0
    if cap_weighted and pd.notna(total_ffmc) and total_ffmc > 0 and residual_weight > 0:
        pending_ffmc = pd.to_numeric(out.loc[pending, "ffmc"], errors="coerce")
        pending_total = pending_ffmc.sum(min_count=1)
        if pd.notna(pending_total) and pending_total > 0:
            out.loc[pending, "weight_pct"] = pending_ffmc / pending_total * residual_weight
        out.loc[pending & out["weight_pct"].notna(), "weight_method"] = "ESTIMATED_FFMC_PROXY"
    return out


def refresh_constituent_snapshots(session=None, indexes: dict | None = None, as_of_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    client = session or requests.Session()
    client.headers.update(HEADERS)
    try:
        client.get(NSE_HOME, timeout=20)
    except Exception as exc:
        return pd.DataFrame(columns=MEMBER_COLUMNS), {"status": "SOURCE_UNAVAILABLE", "error": str(exc)}
    all_rows, audit = [], {}
    for index, nse_name in (indexes or NSE_INDEX_NAMES).items():
        errors = []
        frame = pd.DataFrame(columns=MEMBER_COLUMNS)
        # NSE has exposed both route spellings in different generations of its
        # public site. Try the current hyphenated route first, then legacy.
        endpoints = [
            "https://www.nseindia.com/api/equity-stockIndices?index={}",
            NSE_API,
        ]
        for endpoint in endpoints:
            try:
                request_url = endpoint.format(nse_name.replace(" ", "%20"))
                response = client.get(request_url, timeout=25, headers=HEADERS)
                response.raise_for_status()
                frame = normalise_constituent_payload(index, response.json(), as_of_date, source_url=request_url)
                if frame.empty:
                    raise ValueError("NSE response contained no constituents")
                break
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
        if not frame.empty:
            all_rows.append(frame)
            audit[index] = {"status": "PASS", "constituents": int(len(frame)),
                            "official_weights": int(frame.weight_method.eq("OFFICIAL_NSE_WEIGHT").sum()),
                            "estimated_weights": int(frame.weight_method.eq("ESTIMATED_FFMC_PROXY").sum()),
                            "missing_weights": int(frame.weight_pct.isna().sum())}
        else:
            audit[index] = {"status": "SOURCE_UNAVAILABLE", "errors": errors}
        time.sleep(0.15)
    fresh = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=MEMBER_COLUMNS)
    if CONSTITUENTS_FILE.exists():
        old = pd.read_csv(CONSTITUENTS_FILE)
        if not old.empty and set(MEMBER_COLUMNS).issubset(old.columns):
            fresh = pd.concat([old[MEMBER_COLUMNS], fresh], ignore_index=True)
    if not fresh.empty:
        fresh["as_of_date"] = _normalise_date(fresh["as_of_date"])
        fresh["symbol"] = fresh["symbol"].astype(str).str.upper().str.strip()
        fresh = fresh.dropna(subset=["as_of_date"]).drop_duplicates(["index", "symbol", "as_of_date"], keep="last")
        fresh = fresh.sort_values(["index", "as_of_date", "symbol"])
    # Always persist the schema, even when the source is unavailable. An empty
    # but valid artifact is distinguishable from a missing/malformed file.
    CONSTITUENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fresh.reindex(columns=MEMBER_COLUMNS).to_csv(CONSTITUENTS_FILE, index=False)
    return fresh, audit


def build_constituent_features(stock_history: pd.DataFrame, index_history: pd.DataFrame,
                               membership: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    """Create date-effective, no-look-ahead stock/index relationship features."""
    output = []
    required_stock = {"date", "symbol", "close"}
    required_index = {"date", "index", "close"}
    if stock_history.empty or index_history.empty or membership.empty:
        return pd.DataFrame()
    if not required_stock.issubset(stock_history) or not required_index.issubset(index_history):
        raise ValueError("Stock and index history require date, symbol/index, and close")
    stocks = stock_history.copy()
    indexes = index_history.copy()
    members = membership.copy()
    stocks["date"] = _normalise_date(stocks["date"])
    indexes["date"] = _normalise_date(indexes["date"])
    members["as_of_date"] = _normalise_date(members["as_of_date"])
    stocks["symbol"] = stocks["symbol"].astype(str).str.upper().str.strip()
    indexes["index"] = indexes["index"].astype(str).str.upper().str.strip()
    members["symbol"] = members["symbol"].astype(str).str.upper().str.strip()
    members["index"] = members["index"].astype(str).str.upper().str.strip()
    for frame in (stocks, indexes):
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    members["weight_pct"] = pd.to_numeric(members["weight_pct"], errors="coerce")
    stocks = stocks.dropna(subset=["date", "close"]).sort_values(["symbol", "date"])
    indexes = indexes.dropna(subset=["date", "close"]).sort_values(["index", "date"])
    members = members.dropna(subset=["as_of_date"]).sort_values(["index", "symbol", "as_of_date"])
    index_key_map = {k.upper(): k for k in INDEX_TICKERS}
    for (index, symbol), group in members.groupby(["index", "symbol"]):
        idx_key = index_key_map.get(index)
        if not idx_key:
            continue
        stock = stocks.loc[stocks.symbol.eq(symbol), ["date", "close"]].drop_duplicates("date", keep="last").copy()
        idx = indexes.loc[indexes["index"].eq(idx_key), ["date", "close"]].drop_duplicates("date", keep="last").copy()
        if stock.empty or idx.empty:
            continue
        stock["stock_return_1d_pct"] = stock.close.pct_change() * 100
        idx["index_return_1d_pct"] = idx.close.pct_change() * 100
        idx["index_return_5d_pct"] = idx.close.pct_change(5) * 100
        joined = stock.merge(idx, on="date", how="inner", suffixes=("", "_index"))
        joined = joined.sort_values("date")
        if joined.empty:
            continue
        joined["rolling_corr_60d"] = joined.stock_return_1d_pct.rolling(60, min_periods=min_periods).corr(joined.index_return_1d_pct)
        covariance = joined.stock_return_1d_pct.rolling(60, min_periods=min_periods).cov(joined.index_return_1d_pct)
        variance = joined.index_return_1d_pct.rolling(60, min_periods=min_periods).var()
        joined["rolling_beta_60d"] = covariance / variance.replace(0, np.nan)
        # As-of merge ensures a constituent/weight snapshot is never applied
        # before the date on which the system actually observed it.
        snapshots = group[["as_of_date", "weight_pct", "weight_method"]].drop_duplicates("as_of_date").sort_values("as_of_date")
        joined = pd.merge_asof(joined.sort_values("date"), snapshots,
                               left_on="date", right_on="as_of_date", direction="backward")
        joined = joined.dropna(subset=["as_of_date"])
        joined["index"] = index
        joined["symbol"] = symbol
        joined["weighted_stock_return_contribution_pct"] = joined.stock_return_1d_pct * joined.weight_pct / 100.0
        output.append(joined[["date", "index", "symbol", "weight_pct", "weight_method",
                              "index_return_1d_pct", "index_return_5d_pct", "stock_return_1d_pct",
                              "rolling_corr_60d", "rolling_beta_60d",
                              "weighted_stock_return_contribution_pct", "as_of_date"]])
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def run_index_constituent_intelligence(stock_history: pd.DataFrame | None = None) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    index_history, index_audit = download_index_history()
    _, member_audit = refresh_constituent_snapshots()
    if stock_history is None:
        stock_history = pd.read_csv(ROOT / "data" / "ohlcv.csv") if (ROOT / "data" / "ohlcv.csv").exists() else pd.DataFrame()
    membership = pd.read_csv(CONSTITUENTS_FILE) if CONSTITUENTS_FILE.exists() else pd.DataFrame(columns=MEMBER_COLUMNS)
    features = build_constituent_features(stock_history, index_history, membership)
    if features.empty:
        features = pd.DataFrame(columns=FEATURE_COLUMNS)
    features.to_csv(FEATURES_FILE, index=False)
    report = {
        "updated_at_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "index_ohlcv": index_audit,
        "constituents": member_audit,
        "index_history_file": str(INDEX_HISTORY_FILE.relative_to(ROOT)),
        "constituents_file": str(CONSTITUENTS_FILE.relative_to(ROOT)),
        "features_file": str(FEATURES_FILE.relative_to(ROOT)),
        "feature_rows": int(len(features)),
        "weight_disclaimer": "Only exchange-provided weights are official. Free-float-market-cap proxies are estimates and must not be represented as exact index weights.",
        "history_disclaimer": "Constituent snapshots are point-in-time from the first observed date. Current members/weights are never projected backwards to create historical membership.",
    }
    AUDIT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_index_constituent_intelligence()
