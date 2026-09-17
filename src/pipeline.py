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
PREDICTIONS_FILE = DATA / "predictions.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"

FEATURE_COLUMNS = ["return_1d", "return_5d", "return_20d", "sma20", "sma50", "ema20", "ema50", "rsi14", "volume_ratio"]
TARGETS = {"open": "target_open_return", "high": "target_high_return", "low": "target_low_return", "close": "target_close_return"}
EMPTY_HISTORY = ["date", "symbol", "open", "high", "low", "close", "volume"]


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError("data/universe.csv is missing. Run update_universe.py first.")
    df = pd.read_csv(UNIVERSE_FILE)
    symbols = df["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    if len(symbols) < 100:
        raise RuntimeError(f"Universe contains only {len(symbols)} symbols; refusing to run.")
    return symbols


def _normalise_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EMPTY_HISTORY)
    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=EMPTY_HISTORY)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "close"])


def download_history(symbols: list[str], start: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for batch_start in range(0, len(symbols), 40):
        batch = symbols[batch_start:batch_start + 40]
        tickers = [f"{s}.NS" for s in batch]
        raw = yf.download(tickers, start=start, interval="1d", auto_adjust=False, group_by="ticker", progress=False, threads=True)
        if raw.empty:
            continue
        for symbol in batch:
            ticker = f"{symbol}.NS"
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    part = raw[ticker].copy()
                else:
                    if len(batch) != 1:
                        continue
                    part = raw.copy()
                rows.append(_normalise_history(part, symbol))
            except Exception as exc:
                print(f"Skipping {symbol}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EMPTY_HISTORY)


def update_history(symbols: list[str]) -> pd.DataFrame:
    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        last = existing["date"].max() if not existing.empty else pd.Timestamp("2018-01-01")
        start = (last - timedelta(days=10)).strftime("%Y-%m-%d")
    else:
        existing = pd.DataFrame(columns=EMPTY_HISTORY)
        start = "2018-01-01"
    fresh = download_history(symbols, start)
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["date", "symbol", "close"])
    combined = combined.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
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
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
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


def rank_stocks(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
    latest = latest.dropna(subset=["return_20d", "return_5d", "volume_ratio", "rsi14", "ema20", "ema50"])
    latest["technical_score"] = technical_score(latest)
    latest["fundamental_score"] = 0.0
    latest["total_score"] = latest["technical_score"]
    latest["rank"] = latest["total_score"].rank(ascending=False, method="first").astype(int)
    return latest.sort_values("rank")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    base = df["close"]
    df["target_open_return"] = g["open"].shift(-1) / base - 1
    df["target_high_return"] = g["high"].shift(-1) / base - 1
    df["target_low_return"] = g["low"].shift(-1) / base - 1
    df["target_close_return"] = g["close"].shift(-1) / base - 1
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


def predict_top10(df: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    ranked = ranking.head(10)[["symbol", "date", "close", "rank", "total_score"]].copy()
    latest = features(df).sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest.symbol.isin(ranked.symbol)].dropna(subset=FEATURE_COLUMNS)
    if not models_ready():
        raise RuntimeError("Prediction models are missing")
    for name in TARGETS:
        model = joblib.load(MODELS / f"{name}.joblib")
        latest[f"pred_{name}_return"] = model.predict(latest[FEATURE_COLUMNS])
        latest[f"predicted_{name}"] = latest["close"] * (1 + latest[f"pred_{name}_return"])
    out = latest[["date", "symbol", "close", "predicted_open", "predicted_high", "predicted_low", "predicted_close"]].copy()
    out = out.rename(columns={"date": "prediction_date", "close": "base_close"})
    lookup = ranked.set_index("symbol")
    out["rank"] = out.symbol.map(lookup["rank"])
    out["score"] = out.symbol.map(lookup["total_score"])
    out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out.sort_values("rank")


def run_morning() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    ranking = rank_stocks(features(hist))
    ranking.to_csv(RANKING_FILE, index=False)
    if not models_ready():
        train(hist)
    predictions = predict_top10(hist, ranking)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Morning run complete: {len(ranking)} ranked, {len(predictions)} predictions")


def run_evening() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    if not PREDICTIONS_FILE.exists():
        print("No predictions file; nothing to evaluate.")
        return
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date"])
    hist["date"] = pd.to_datetime(hist["date"])
    actual = hist.sort_values(["symbol", "date"]).copy()
    actual["actual_date"] = actual.groupby("symbol")["date"].shift(-1)
    for field in ["open", "high", "low", "close"]:
        actual[f"next_{field}"] = actual.groupby("symbol")[field].shift(-1)
    actual = actual[["symbol", "date", "actual_date", "next_open", "next_high", "next_low", "next_close"]]
    evals = predictions.merge(actual, left_on=["symbol", "prediction_date"], right_on=["symbol", "date"], how="inner")
    if evals.empty:
        print("No completed prediction sessions to evaluate yet.")
        return
    for field in ["open", "high", "low", "close"]:
        pred = evals[f"predicted_{field}"]
        real = evals[f"next_{field}"]
        evals[f"{field}_error"] = real - pred
        evals[f"{field}_abs_pct_error"] = (real - pred).abs() / real.abs().replace(0, np.nan)
    evals = evals.rename(columns={"date": "prediction_date"})
    keep = ["prediction_date", "actual_date", "symbol", "rank", "score", "predicted_open", "next_open", "open_error", "open_abs_pct_error", "predicted_high", "next_high", "high_error", "high_abs_pct_error", "predicted_low", "next_low", "low_error", "low_abs_pct_error", "predicted_close", "next_close", "close_error", "close_abs_pct_error"]
    evals = evals[keep]
    if EVALUATIONS_FILE.exists():
        old = pd.read_csv(EVALUATIONS_FILE)
        evals = pd.concat([old, evals], ignore_index=True)
    evals = evals.drop_duplicates(["prediction_date", "symbol"], keep="last").sort_values(["prediction_date", "rank"])
    evals.to_csv(EVALUATIONS_FILE, index=False)
    train(hist)
    print(f"Evening run complete: evaluated {len(evals)} rows and retrained models")


if __name__ == "__main__":
    run_morning()
