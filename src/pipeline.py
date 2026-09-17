from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

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


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Create {UNIVERSE_FILE} with columns symbol,company_name")
    return pd.read_csv(UNIVERSE_FILE)["symbol"].dropna().astype(str).unique().tolist()


def download_history(symbols: list[str], start: str = "2018-01-01") -> pd.DataFrame:
    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, start=start, interval="1d", auto_adjust=False, group_by="ticker", progress=False, threads=True)
    rows = []
    for symbol in symbols:
        ticker = f"{symbol}.NS"
        if ticker not in raw.columns.get_level_values(0):
            continue
        df = raw[ticker].copy().reset_index()
        df["symbol"] = symbol
        df = df.rename(columns={"Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
        rows.append(df[["date","symbol","open","high","low","close","volume"]])
    if not rows:
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])
    return pd.concat(rows, ignore_index=True)


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
    latest["technical_score"] = technical_score(latest)
    # Fundamental score is intentionally zero until point-in-time fundamental data is supplied.
    latest["fundamental_score"] = 0.0
    latest["total_score"] = latest["technical_score"]
    latest["rank"] = latest["total_score"].rank(ascending=False, method="first").astype(int)
    return latest.sort_values("rank")


FEATURE_COLUMNS = ["return_1d","return_5d","return_20d","sma20","sma50","ema20","ema50","rsi14","volume_ratio"]
TARGETS = {"open":"target_open_return", "high":"target_high_return", "low":"target_low_return", "close":"target_close_return"}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol","date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    next_close = g["close"].shift(-1)
    next_open = g["open"].shift(-1)
    next_high = g["high"].shift(-1)
    next_low = g["low"].shift(-1)
    df["target_open_return"] = next_open / df["close"] - 1
    df["target_high_return"] = next_high / df["close"] - 1
    df["target_low_return"] = next_low / df["close"] - 1
    df["target_close_return"] = next_close / df["close"] - 1
    return df


def train(df: pd.DataFrame) -> None:
    work = add_targets(features(df)).dropna(subset=FEATURE_COLUMNS + list(TARGETS.values()))
    for name, target in TARGETS.items():
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(work[FEATURE_COLUMNS], work[target])
        joblib.dump(model, MODELS / f"{name}.joblib")


def predict_top10(df: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    ranked = ranking.head(10)[["symbol","date","close","rank","total_score"]].copy()
    latest = features(df).sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest.symbol.isin(ranked.symbol)]
    for name in TARGETS:
        model = joblib.load(MODELS / f"{name}.joblib")
        latest[f"pred_{name}_return"] = model.predict(latest[FEATURE_COLUMNS])
        latest[f"predicted_{name}"] = latest["close"] * (1 + latest[f"pred_{name}_return"])
    out = latest[["date","symbol","close","predicted_open","predicted_high","predicted_low","predicted_close"]].copy()
    out = out.rename(columns={"date":"prediction_date","close":"base_close"})
    out["rank"] = out.symbol.map(ranked.set_index("symbol")["rank"])
    out["score"] = out.symbol.map(ranked.set_index("symbol")["total_score"])
    return out


def run_initial() -> None:
    symbols = load_universe()
    hist = download_history(symbols)
    hist.to_csv(HISTORY_FILE, index=False)
    feat = features(hist)
    ranking = rank_stocks(feat)
    ranking.to_csv(RANKING_FILE, index=False)
    train(hist)
    predictions = predict_top10(hist, ranking)
    predictions.to_csv(PREDICTIONS_FILE, index=False)


if __name__ == "__main__":
    run_initial()
