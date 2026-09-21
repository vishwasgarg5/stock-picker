from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
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

FEATURE_COLUMNS = [\n    "return_1d", "return_5d", "return_20d",\n    "sma20", "sma50", "ema20", "ema50",\n    "rsi14", "volume_ratio",\n    "atr14_pct", "macd", "macd_signal",\n    "bb_position", "range_pct", "close_sma20_gap",\n    "close_sma50_gap", "volatility20", "volume_trend5",\n]
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
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
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
    """Build strictly historical, next-day-safe features for each stock."""
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    # Momentum and trend.
    df["return_1d"] = g["close"].pct_change()
    df["return_5d"] = g["close"].pct_change(5)
    df["return_20d"] = g["close"].pct_change(20)
    df["sma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["sma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
    df["ema20"] = g["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["ema50"] = g["close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["rsi14"] = g["close"].transform(rsi)

    # Volatility/range: all values are calculated using current and prior bars only.
    prev_close = g["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14_pct"] = (
        true_range.groupby(df["symbol"], group_keys=False)
        .transform(lambda x: x.rolling(14).mean())
        / df["close"].replace(0, np.nan)
    )
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["volatility20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())

    # MACD trend strength.
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )

    # Position inside the 20-day Bollinger band.
    rolling_std20 = g["close"].transform(lambda x: x.rolling(20).std())
    df["bb_position"] = (df["close"] - df["sma20"]) / (
        2.0 * rolling_std20.replace(0, np.nan)
    )

    # Relative price/trend features.
    df["close_sma20_gap"] = df["close"] / df["sma20"].replace(0, np.nan) - 1.0
    df["close_sma50_gap"] = df["close"] / df["sma50"].replace(0, np.nan) - 1.0

    # Volume regime.
    df["volume_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)
    df["volume_ma5"] = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_trend5"] = df["volume_ma5"] / df["volume_ma20"].replace(0, np.nan)

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
    columns = ["symbol", "updated_at"] + list(FUNDAMENTAL_FIELDS)
    if FUNDAMENTALS_FILE.exists():
        cached = pd.read_csv(FUNDAMENTALS_FILE)
        cached["symbol"] = cached.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    else:
        cached = pd.DataFrame(columns=columns)
    now = pd.Timestamp.now(tz="UTC")
    cache_map = {row["symbol"]: row for _, row in cached.iterrows() if pd.notna(row.get("symbol"))}
    refreshed = []
    for symbol in symbols:
        old = cache_map.get(symbol)
        old_date = pd.to_datetime(old.get("updated_at"), errors="coerce", utc=True) if old is not None else pd.NaT
        stale = pd.isna(old_date) or (now - old_date > pd.Timedelta(days=max_age_days))
        if not stale:
            refreshed.append({c: old.get(c, np.nan) for c in columns})
            continue
        row = {"symbol": symbol, "updated_at": now.isoformat(timespec="seconds"), **{f: np.nan for f in FUNDAMENTAL_FIELDS}}
        try:
            info = yf.Ticker(f"{symbol}.NS").get_info()
            for field in FUNDAMENTAL_FIELDS:
                row[field] = _safe_number(info.get(field))
            print(f"Fundamentals updated: {symbol}")
        except Exception as exc:
            print(f"Fundamentals lookup failed for {symbol}: {exc}")
            if old is not None:
                row = {c: old.get(c, np.nan) for c in columns}
        refreshed.append(row)
    result = pd.DataFrame(refreshed, columns=columns).drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)
    result.to_csv(FUNDAMENTALS_FILE, index=False)
    return result


def fundamental_score(fundamentals: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.0, index=fundamentals.index)
    weight_used = pd.Series(0.0, index=fundamentals.index)
    for field, (weight, higher_is_better) in FUNDAMENTAL_FIELDS.items():
        if field not in fundamentals.columns:
            continue
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
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    latest_date = work["date"].max()
    latest = work[work["date"] == latest_date].dropna(subset=FEATURE_COLUMNS).copy()
    latest["technical_score"] = technical_score(latest)
    if fundamentals is None:
        fundamentals = pd.DataFrame({"symbol": latest["symbol"]})
    fundamental_cols = ["symbol"] + list(FUNDAMENTAL_FIELDS)
    available = fundamentals[[c for c in fundamental_cols if c in fundamentals.columns]].copy()
    latest = latest.merge(available, on="symbol", how="left")
    latest["fundamental_score"] = fundamental_score(latest)
    latest["total_score"] = latest["technical_score"] + latest["fundamental_score"]
    latest = latest.sort_values(["total_score", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    latest["rank"] = np.arange(1, len(latest) + 1, dtype=int)
    if latest["rank"].duplicated().any() or latest["rank"].tolist() != list(range(1, len(latest) + 1)):
        raise RuntimeError("Ranking validation failed: ranks are not unique and sequential")
    return latest


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
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(work[FEATURE_COLUMNS], work[target])
        joblib.dump(model, MODELS / f"{name}.joblib")


def models_ready() -> bool:
    return all((MODELS / f"{name}.joblib").exists() for name in TARGETS)


def _next_trading_date(last_date: pd.Timestamp, history_dates: pd.Series) -> pd.Timestamp:
    """Return the next NSE session, including exchange holidays rather than only skipping weekends."""
    last = pd.Timestamp(last_date).normalize()
    schedule = mcal.get_calendar("XNSE").schedule(start_date=last + pd.Timedelta(days=1), end_date=last + pd.Timedelta(days=14))
    if schedule.empty:
        raise RuntimeError(f"Could not determine next NSE trading session after {last.date()}")
    return pd.Timestamp(schedule.index[0]).normalize()


def predict_top10(df: pd.DataFrame, ranking: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    ranked = ranking.head(10)[["symbol", "date", "close", "rank", "total_score"]].copy()
    if len(ranked) != 10:
        raise RuntimeError(f"Expected 10 ranked stocks, found {len(ranked)}")
    latest = features(df).sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest["symbol"].isin(ranked["symbol"])].dropna(subset=FEATURE_COLUMNS)
    if len(latest) != 10:
        raise RuntimeError(f"Expected features for 10 top stocks, found {len(latest)}")
    if not models_ready():
        raise RuntimeError("Prediction models are missing")
    for name in TARGETS:
        model = joblib.load(MODELS / f"{name}.joblib")
        latest[f"predicted_{name}"] = latest["close"] * (1 + model.predict(latest[FEATURE_COLUMNS]))
    latest["predicted_high"] = latest[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
    latest["predicted_low"] = latest[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)
    out = latest[["date", "symbol", "close", "predicted_open", "predicted_high", "predicted_low", "predicted_close"]].copy().rename(columns={"date": "prediction_date", "close": "base_close"})
    lookup = ranked.set_index("symbol")
    out["rank"] = out["symbol"].map(lookup["rank"]).astype(int)
    out["score"] = out["symbol"].map(lookup["total_score"])
    out["target_date"] = pd.Timestamp(target_date).normalize()
    out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out.sort_values("rank")


def _save_next_session_prediction(hist: pd.DataFrame, ranking: pd.DataFrame, target_date: pd.Timestamp, existing: pd.DataFrame) -> None:
    if not existing.empty and "target_date" in existing.columns:
        existing_dates = pd.to_datetime(existing["target_date"], errors="coerce").dt.normalize()
        if (existing_dates == target_date.normalize()).any():
            return
    prediction = predict_top10(hist, ranking, target_date)
    combined = pd.concat([existing, prediction], ignore_index=True) if not existing.empty else prediction
    combined = combined.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    combined.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Backfilled next prediction session: {target_date.date()} ({len(prediction)} stocks)")


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
    print(f"Morning session: latest completed market date={last_date.date()}, prediction target={target_date.date()}")
    existing = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"]) if PREDICTIONS_FILE.exists() else pd.DataFrame()
    if not existing.empty and "target_date" in existing.columns:
        existing["target_date"] = pd.to_datetime(existing["target_date"], errors="coerce").dt.normalize()
        same_target = existing[existing["target_date"] == target_date]
        if not same_target.empty:
            if len(same_target) != 10:
                raise RuntimeError(f"Prediction session {target_date.date()} exists but has {len(same_target)} rows; refusing partial session")
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
        print("No predictions file; creating a next-session prediction if models are available.")
        if not models_ready():
            train(hist)
        fundamentals = update_fundamentals(symbols)
        ranking = rank_stocks(features(hist), fundamentals)
        ranking.to_csv(RANKING_FILE, index=False)
        target_date = _next_trading_date(pd.to_datetime(hist["date"], errors="coerce").max().normalize(), hist["date"])
        _save_next_session_prediction(hist, ranking, target_date, pd.DataFrame())
        return

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    latest_actual_date = hist["date"].max()
    completed_predictions = predictions[predictions["target_date"] <= latest_actual_date].copy()
    print(f"Evening session: latest actual market date={latest_actual_date.date()}, completed prediction rows available={len(completed_predictions)}")

    if completed_predictions.empty:
        if not models_ready():
            train(hist)
        fundamentals = update_fundamentals(symbols)
        ranking = rank_stocks(features(hist), fundamentals)
        ranking.to_csv(RANKING_FILE, index=False)
        target_date = _next_trading_date(latest_actual_date, hist["date"])
        _save_next_session_prediction(hist, ranking, target_date, predictions)
        print("No completed prediction session to evaluate; ensured next session is predicted.")
        return

    actual = hist.sort_values(["symbol", "date"]).rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"})
    evals = completed_predictions.merge(actual[["symbol", "target_date", "actual_open", "actual_high", "actual_low", "actual_close"]], on=["symbol", "target_date"], how="inner")
    evals = evals.dropna(subset=["actual_open", "actual_high", "actual_low", "actual_close", "base_close"])
    if evals.empty:
        print("Completed target dates exist, but actual OHLC rows are not available yet.")
        return

    for field in ["open", "high", "low", "close"]:
        pred = pd.to_numeric(evals[f"predicted_{field}"], errors="coerce")
        real = pd.to_numeric(evals[f"actual_{field}"], errors="coerce")
        baseline = pd.to_numeric(evals["base_close"], errors="coerce")
        evals[f"{field}_error"] = real - pred
        evals[f"{field}_abs_pct_error"] = (real - pred).abs() / real.abs().replace(0, np.nan)
        evals[f"baseline_{field}_error"] = real - baseline
        evals[f"baseline_{field}_abs_pct_error"] = (real - baseline).abs() / real.abs().replace(0, np.nan)

    predicted_return = pd.to_numeric(evals["predicted_close"], errors="coerce") / pd.to_numeric(evals["base_close"], errors="coerce") - 1
    actual_return = pd.to_numeric(evals["actual_close"], errors="coerce") / pd.to_numeric(evals["base_close"], errors="coerce") - 1
    evals["predicted_close_direction"] = np.sign(predicted_return).astype(int)
    evals["actual_close_direction"] = np.sign(actual_return).astype(int)
    evals["close_direction_correct"] = (evals["predicted_close_direction"] == evals["actual_close_direction"]).astype(int)

    keep = [
        "prediction_date", "target_date", "symbol", "rank", "score", "base_close",
        "predicted_open", "actual_open", "open_error", "open_abs_pct_error", "baseline_open_error", "baseline_open_abs_pct_error",
        "predicted_high", "actual_high", "high_error", "high_abs_pct_error", "baseline_high_error", "baseline_high_abs_pct_error",
        "predicted_low", "actual_low", "low_error", "low_abs_pct_error", "baseline_low_error", "baseline_low_abs_pct_error",
        "predicted_close", "actual_close", "close_error", "close_abs_pct_error", "baseline_close_error", "baseline_close_abs_pct_error",
        "predicted_close_direction", "actual_close_direction", "close_direction_correct",
    ]
    evals = evals[keep]
    if EVALUATIONS_FILE.exists():
        old = pd.read_csv(EVALUATIONS_FILE)
        # CSV dates come back as strings; normalize both sides before concat/sort.
        if "target_date" in old.columns:
            old["target_date"] = pd.to_datetime(old["target_date"], errors="coerce").dt.normalize()
        if "prediction_date" in old.columns:
            old["prediction_date"] = pd.to_datetime(old["prediction_date"], errors="coerce").dt.normalize()
        evals = pd.concat([old, evals], ignore_index=True)
    evals["target_date"] = pd.to_datetime(evals["target_date"], errors="coerce").dt.normalize()
    evals["prediction_date"] = pd.to_datetime(evals["prediction_date"], errors="coerce").dt.normalize()
    evals = evals.dropna(subset=["target_date", "symbol"])
    evals = evals.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    evals.to_csv(EVALUATIONS_FILE, index=False)
    train(hist)

    target_date = _next_trading_date(latest_actual_date, hist["date"])
    fundamentals = update_fundamentals(symbols)
    ranking = rank_stocks(features(hist), fundamentals)
    ranking.to_csv(RANKING_FILE, index=False)
    _save_next_session_prediction(hist, ranking, target_date, predictions)

    sessions = evals["target_date"].nunique()
    latest_session = evals["target_date"].max().date()
    print(f"Evening run complete: latest evaluated session={latest_session}, total evaluations={len(evals)}, sessions={sessions}, next target={target_date.date()}, models retrained={models_ready()}")


if __name__ == "__main__":
    run_morning()
