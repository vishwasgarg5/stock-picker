from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

UNIVERSE_FILE = DATA / "universe.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
RANKING_FILE = DATA / "rankings.csv"
FUNDAMENTALS_FILE = DATA / "fundamentals.csv"
PREDICTIONS_FILE = DATA / "predictions.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"

FEATURE_COLUMNS = ["return_1d", "return_5d", "return_20d", "sma20", "sma50", "ema20", "ema50", "rsi14", "volume_ratio"]
TARGETS = {"open": "target_open_return", "high": "target_high_return", "low": "target_low_return", "close": "target_close_return"}
EMPTY_HISTORY = ["date", "symbol", "open", "high", "low", "close", "volume"]


def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_FILE)
    symbols = df["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    if len(symbols) < 100:
        raise RuntimeError(f"Universe contains only {len(symbols)} symbols; refusing to run")
    return symbols


def _normalise_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EMPTY_HISTORY)
    df = df.reset_index().rename(columns={"Date":"date", "Datetime":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
    required = set(EMPTY_HISTORY) - {"symbol"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=EMPTY_HISTORY)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df[EMPTY_HISTORY].dropna(subset=["date", "close"])


def download_history(symbols: list[str], start: str) -> pd.DataFrame:
    rows = []
    for batch_start in range(0, len(symbols), 25):
        batch = symbols[batch_start:batch_start + 25]
        try:
            raw = yf.download([f"{s}.NS" for s in batch], start=start, interval="1d", auto_adjust=False, group_by="ticker", progress=False, threads=True)
        except Exception as exc:
            print(f"Download batch failed: {exc}")
            continue
        if raw.empty:
            continue
        for symbol in batch:
            ticker = f"{symbol}.NS"
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    levels = raw.columns.get_level_values(0)
                    if ticker in levels:
                        part = raw[ticker].copy()
                    elif symbol in levels:
                        part = raw[symbol].copy()
                    else:
                        continue
                else:
                    if len(batch) != 1:
                        continue
                    part = raw.copy()
                x = _normalise_history(part, symbol)
                if not x.empty:
                    rows.append(x)
            except Exception as exc:
                print(f"Skipping {symbol}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EMPTY_HISTORY)


def update_history(symbols: list[str]) -> pd.DataFrame:
    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    else:
        existing = pd.DataFrame(columns=EMPTY_HISTORY)
    existing["date"] = pd.to_datetime(existing.get("date"), errors="coerce")
    existing = existing.dropna(subset=["date", "symbol", "close"])
    starts = {}
    for symbol in symbols:
        rows = existing.loc[existing["symbol"].astype(str).str.upper() == symbol]
        starts[symbol] = "2018-01-01" if rows.empty else (rows["date"].max() - timedelta(days=10)).strftime("%Y-%m-%d")
    fresh_parts = [download_history([s for s, value in starts.items() if value == start], start) for start in sorted(set(starts.values()))]
    fresh = pd.concat(fresh_parts, ignore_index=True) if fresh_parts else pd.DataFrame(columns=EMPTY_HISTORY)
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "symbol", "close"]).drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    combined.to_csv(HISTORY_FILE, index=False)
    return combined


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    df["return_1d"] = g["close"].pct_change()
    df["return_5d"] = g["close"].pct_change(5)
    df["return_20d"] = g["close"].pct_change(20)
    df["sma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["sma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
    df["ema20"] = g["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["ema50"] = g["close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["rsi14"] = g["close"].transform(rsi)
    df["volume_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)
    return df


def technical_score(latest: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=latest.index)
    score += latest["return_20d"].rank(pct=True) * 25
    score += latest["return_5d"].rank(pct=True) * 15
    score += latest["volume_ratio"].rank(pct=True) * 10
    score += latest["rsi14"].rank(pct=True) * 10
    score += (latest["close"] > latest["ema20"]).astype(float) * 10
    score += (latest["ema20"] > latest["ema50"]).astype(float) * 10
    return score


FUNDAMENTAL_FIELDS = {
    "returnOnEquity": (0.20, True),
    "profitMargins": (0.15, True),
    "revenueGrowth": (0.15, True),
    "earningsGrowth": (0.15, True),
    "trailingPE": (0.10, False),
    "priceToBook": (0.10, False),
    "debtToEquity": (0.10, False),
    "dividendYield": (0.05, True),
}


def _safe_number(value: object) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def update_fundamentals(symbols: list[str], max_age_days: int = 7) -> pd.DataFrame:
    """Fetch/cache current company fundamentals used by the ranking model.

    Fundamentals are refreshed weekly so the daily workflow does not repeatedly
    make 150 metadata requests. Failed individual lookups are retained from the
    previous cache when possible.
    """
    columns = ["symbol", "updated_at"] + list(FUNDAMENTAL_FIELDS)
    if FUNDAMENTALS_FILE.exists():
        cached = pd.read_csv(FUNDAMENTALS_FILE)
        cached["symbol"] = cached.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    else:
        cached = pd.DataFrame(columns=columns)

    now = pd.Timestamp.now(tz="UTC")
    cached_dates = pd.to_datetime(cached.get("updated_at"), errors="coerce", utc=True)
    cache_map = {row["symbol"]: row for _, row in cached.iterrows() if pd.notna(row.get("symbol"))}
    refreshed = []

    for symbol in symbols:
        old = cache_map.get(symbol)
        old_date = pd.to_datetime(old.get("updated_at"), errors="coerce", utc=True) if old is not None else pd.NaT
        stale = pd.isna(old_date) or (now - old_date > pd.Timedelta(days=max_age_days))
        if not stale:
            refreshed.append({c: old.get(c, np.nan) for c in columns})
            continue

        row = {"symbol": symbol, "updated_at": now.isoformat(timespec="seconds")}
        for field in FUNDAMENTAL_FIELDS:
            row[field] = np.nan
        try:
            info = yf.Ticker(f"{symbol}.NS").get_info()
            for field in FUNDAMENTAL_FIELDS:
                row[field] = _safe_number(info.get(field))
            print(f"Fundamentals updated: {symbol}")
        except Exception as exc:
            print(f"Fundamentals lookup failed for {symbol}: {exc}")
            if old is not None:
                row = {c: old.get(c, np.nan) for c in columns}
            else:
                row = {"symbol": symbol, "updated_at": now.isoformat(timespec="seconds"), **{f: np.nan for f in FUNDAMENTAL_FIELDS}}
        refreshed.append(row)

    result = pd.DataFrame(refreshed, columns=columns)
    result = result.drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)
    result.to_csv(FUNDAMENTALS_FILE, index=False)
    return result


def fundamental_score(fundamentals: pd.DataFrame) -> pd.Series:
    """Score fundamentals from 0-20 using cross-sectional percentile ranks.

    Higher ROE/margins/growth/dividend yield score higher; lower PE/PB/debt
    score higher. Missing metrics are ignored rather than turning every score
    into zero. A stock with no usable fundamental data receives the neutral
    midpoint (10/20), making data availability visible without dominating the
    ranking.
    """
    result = pd.Series(0.0, index=fundamentals.index)
    weight_used = pd.Series(0.0, index=fundamentals.index)
    for field, (weight, higher_is_better) in FUNDAMENTAL_FIELDS.items():
        values = pd.to_numeric(fundamentals[field], errors="coerce")
        valid = values.notna()
        if field in {"trailingPE", "priceToBook", "debtToEquity"}:
            valid &= values > 0
        if valid.sum() < 2:
            continue
        ranks = values[valid].rank(pct=True)
        if not higher_is_better:
            ranks = 1.0 - ranks + (1.0 / valid.sum())
        result.loc[valid] += ranks * weight
        weight_used.loc[valid] += weight

    scored = pd.Series(10.0, index=fundamentals.index)
    usable = weight_used > 0
    scored.loc[usable] = (result.loc[usable] / weight_used.loc[usable]) * 20.0
    return scored.clip(0, 20)


def rank_stocks(df: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
    latest = df.sort_values("date").groupby("symbol", as_index=False).tail(1).dropna(subset=FEATURE_COLUMNS).copy()
    latest["technical_score"] = technical_score(latest)
    if fundamentals is None:
        fundamentals = pd.DataFrame({"symbol": latest["symbol"]})
    fundamental_cols = ["symbol"] + list(FUNDAMENTAL_FIELDS)
    available = fundamentals[[c for c in fundamental_cols if c in fundamentals.columns]].copy()
    latest = latest.merge(available, on="symbol", how="left")
    latest["fundamental_score"] = fundamental_score(latest)
    latest["total_score"] = latest["technical_score"] + latest["fundamental_score"]
    latest["rank"] = latest["total_score"].rank(ascending=False, method="first").astype(int)
    return latest.sort_values("rank")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    base = df["close"]
    for field in ["open", "high", "low", "close"]:
        df[f"target_{field}_return"] = g[field].shift(-1) / base - 1
    return df


def train(df: pd.DataFrame) -> None:
    work = add_targets(features(df)).dropna(subset=FEATURE_COLUMNS + list(TARGETS.values()))
    if len(work) < 500:
        raise RuntimeError(f"Not enough training rows: {len(work)}")
    for name, target in TARGETS.items():
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(work[FEATURE_COLUMNS], work[target])
        joblib.dump(model, MODELS / f"{name}.joblib")


def models_ready() -> bool:
    return all((MODELS / f"{name}.joblib").exists() for name in TARGETS)


def _next_trading_date(last_date: pd.Timestamp, history_dates: pd.Series) -> pd.Timestamp:
    dates = pd.to_datetime(history_dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    future = dates[dates > pd.Timestamp(last_date).normalize()]
    if not future.empty:
        return future.iloc[0]
    candidate = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate


def predict_top10(df: pd.DataFrame, ranking: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    ranked = ranking.head(10)[["symbol", "date", "close", "rank", "total_score"]].copy()
    latest = features(df).sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest["symbol"].isin(ranked["symbol"])].dropna(subset=FEATURE_COLUMNS)
    if not models_ready():
        raise RuntimeError("Prediction models are missing")
    for name in TARGETS:
        model = joblib.load(MODELS / f"{name}.joblib")
        latest[f"predicted_{name}"] = latest["close"] * (1 + model.predict(latest[FEATURE_COLUMNS]))
    latest["predicted_high"] = latest[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
    latest["predicted_low"] = latest[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)
    out = latest[["date", "symbol", "close", "predicted_open", "predicted_high", "predicted_low", "predicted_close"]].copy()
    out = out.rename(columns={"date": "prediction_date", "close": "base_close"})
    lookup = ranked.set_index("symbol")
    out["rank"] = out["symbol"].map(lookup["rank"])
    out["score"] = out["symbol"].map(lookup["total_score"])
    out["target_date"] = pd.Timestamp(target_date).normalize()
    out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out.sort_values("rank")


def run_morning() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    fundamentals = update_fundamentals(symbols)
    ranking = rank_stocks(features(hist), fundamentals)
    ranking.to_csv(RANKING_FILE, index=False)
    if not models_ready():
        train(hist)

    last_date = pd.to_datetime(hist["date"], errors="coerce").max().normalize()
    target_date = _next_trading_date(last_date, hist["date"])

    existing = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"]) if PREDICTIONS_FILE.exists() else pd.DataFrame()
    if not existing.empty and "target_date" in existing.columns:
        same_target = existing[existing["target_date"].dt.normalize() == target_date]
        if not same_target.empty:
            print(f"Predictions already exist for {target_date.date()}; keeping existing output unchanged.")
            return

    predictions = predict_top10(hist, ranking, target_date)
    combined = pd.concat([existing, predictions], ignore_index=True) if not existing.empty else predictions
    combined = combined.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    combined.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Morning run complete: target session {target_date.date()}, {len(predictions)} predictions created")


def run_evening() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    if not PREDICTIONS_FILE.exists():
        print("No predictions file; nothing to evaluate.")
        train(hist)
        return
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    actual = hist.sort_values(["symbol", "date"]).copy()
    actual = actual.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"})
    evals = predictions.merge(actual[["symbol", "target_date", "actual_open", "actual_high", "actual_low", "actual_close"]], on=["symbol", "target_date"], how="inner")
    evals = evals.dropna(subset=["actual_open", "actual_high", "actual_low", "actual_close"])
    if evals.empty:
        print("No completed prediction sessions to evaluate yet.")
        train(hist)
        return
    for field in ["open", "high", "low", "close"]:
        pred = evals[f"predicted_{field}"]
        real = evals[f"actual_{field}"]
        evals[f"{field}_error"] = real - pred
        evals[f"{field}_abs_pct_error"] = (real - pred).abs() / real.abs().replace(0, np.nan)
    keep = ["prediction_date", "target_date", "symbol", "rank", "score", "predicted_open", "actual_open", "open_error", "open_abs_pct_error", "predicted_high", "actual_high", "high_error", "high_abs_pct_error", "predicted_low", "actual_low", "low_error", "low_abs_pct_error", "predicted_close", "actual_close", "close_error", "close_abs_pct_error"]
    evals = evals[keep]
    if EVALUATIONS_FILE.exists():
        old = pd.read_csv(EVALUATIONS_FILE)
        evals = pd.concat([old, evals], ignore_index=True)
    evals = evals.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    evals.to_csv(EVALUATIONS_FILE, index=False)
    train(hist)
    print(f"Evening run complete: {len(evals)} total evaluations and models retrained")


if __name__ == "__main__":
    run_morning()
