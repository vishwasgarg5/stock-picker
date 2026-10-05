from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

UNIVERSE_FILE = DATA / "universe.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
RANKING_FILE = DATA / "rankings.csv"
FUNDAMENTALS_FILE = DATA / "fundamentals.csv"
FUNDAMENTALS_HISTORY_FILE = DATA / "fundamentals_history.csv"
PREDICTIONS_FILE = DATA / "predictions.csv"
CANDIDATES_FILE = DATA / "prediction_candidates.csv"
CANDIDATE_HISTORY_FILE = DATA / "prediction_candidates_history.csv"
CONFIDENCE_ANALYSIS_FILE = DATA / "confidence_analysis.csv"
SELECTION_VALIDATION_FILE = DATA / "selection_validation.csv"
NEW_LISTINGS_FILE = DATA / "new_listings.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"
PAPER_TRADES_FILE = DATA / "paper_trades.csv"
PORTFOLIO_FILE = DATA / "portfolio_daily.csv"
PAPER_TRADE_TOP_N = 5
PAPER_CAPITAL = 100000.0

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d",
    "sma20", "sma50", "ema20", "ema50",
    "rsi14", "volume_ratio",
    "atr14_pct", "macd", "macd_signal",
    "bb_position", "range_pct", "close_sma20_gap",
    "close_sma50_gap", "volatility20", "volume_trend5",
]
TARGETS = {"open": "target_open_return", "high": "target_high_return", "low": "target_low_return", "close": "target_close_return"}
EMPTY_HISTORY = ["date", "symbol", "open", "high", "low", "close", "volume"]

# Reliability gates for production data and predictions.
MIN_OHLCV_COVERAGE = 0.95
MAX_PREDICTED_MOVE = 0.40


def validate_data_quality(hist: pd.DataFrame, symbols: list[str], context: str = "") -> dict:
    if hist.empty:
        raise RuntimeError("Data-quality gate failed: OHLCV history is empty")
    x = hist.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    latest_date = x["date"].max()
    if pd.isna(latest_date):
        raise RuntimeError("Data-quality gate failed: no valid OHLCV date")
    universe = {str(s).upper().strip() for s in symbols}
    latest = x[(x["date"] == latest_date) & x["symbol"].isin(universe)].drop_duplicates("symbol", keep="last")
    coverage = len(latest) / max(len(universe), 1)
    prices = latest[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    bad_prices = int(prices.isna().any(axis=1).sum())
    invalid_ohlc = int(((prices["high"] < prices[["open", "close"]].max(axis=1)) |
                        (prices["low"] > prices[["open", "close"]].min(axis=1)) |
                        (prices["close"] <= 0)).sum())
    print(f"Data quality [{context}]: latest={latest_date.date()}, fresh={len(latest)}/{len(universe)} ({coverage:.1%}), bad_rows={bad_prices}, invalid_ohlc={invalid_ohlc}")
    if coverage < MIN_OHLCV_COVERAGE:
        raise RuntimeError(f"Data-quality gate failed: fresh OHLCV coverage {coverage:.1%} ({len(latest)}/{len(universe)}), minimum {MIN_OHLCV_COVERAGE:.1%}")
    if bad_prices or invalid_ohlc:
        raise RuntimeError(f"Data-quality gate failed: {bad_prices} bad price rows and {invalid_ohlc} invalid OHLC rows")
    return {"latest_date": latest_date, "fresh_symbols": len(latest), "universe_symbols": len(universe), "coverage": coverage}


def _validate_prediction_session(prediction: pd.DataFrame, target_date: pd.Timestamp) -> None:
    target = pd.Timestamp(target_date).normalize()
    if len(prediction) != 10:
        raise RuntimeError(f"Prediction integrity gate failed: expected 10 stocks, found {len(prediction)}")
    if prediction["symbol"].astype(str).str.upper().duplicated().any():
        raise RuntimeError("Prediction integrity gate failed: duplicate symbols")
    stored_target = pd.to_datetime(prediction["target_date"], errors="coerce").dt.normalize()
    if stored_target.isna().any() or not (stored_target == target).all():
        raise RuntimeError(f"Prediction integrity gate failed: target date mismatch; expected {target.date()}")
    numeric = ["base_close", "predicted_open", "predicted_high", "predicted_low", "predicted_close"]
    values = prediction[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise RuntimeError("Prediction integrity gate failed: non-finite or non-positive OHLC prediction")
    if (prediction["predicted_high"] < prediction[["predicted_open", "predicted_close"]].max(axis=1)).any():
        raise RuntimeError("Prediction integrity gate failed: predicted high is below open/close")
    if (prediction["predicted_low"] > prediction[["predicted_open", "predicted_close"]].min(axis=1)).any():
        raise RuntimeError("Prediction integrity gate failed: predicted low is above open/close")
    base = prediction["base_close"].abs().replace(0, np.nan)
    moves = prediction[["predicted_open", "predicted_high", "predicted_low", "predicted_close"]].sub(prediction["base_close"], axis=0).abs().div(base, axis=0).max(axis=1)
    if (moves > MAX_PREDICTED_MOVE).any():
        offenders = prediction.loc[moves > MAX_PREDICTED_MOVE, "symbol"].astype(str).tolist()
        raise RuntimeError(f"Prediction integrity gate failed: predicted move exceeds {MAX_PREDICTED_MOVE:.0%} for {offenders}")
    if prediction["rank"].nunique() != 10:
        raise RuntimeError("Prediction integrity gate failed: ranks are not unique")
    print(f"Prediction integrity gate passed: {target.date()}, 10 unique stocks")



def load_universe() -> list[str]:
    """Return Nifty 500 plus sufficiently mature recent listings.

    Recent listings are tracked separately by src.universe. The first 19 calendar
    days remain watch/limited-data only; from day 20 they can enter the normal
    ML pipeline with the same ranking machinery. This prevents a handful of
    post-IPO observations from being treated like established stocks.
    """
    df = pd.read_csv(UNIVERSE_FILE)
    core = df["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()

    eligible_new = []
    if NEW_LISTINGS_FILE.exists():
        recent = pd.read_csv(NEW_LISTINGS_FILE)
        if not recent.empty and {"symbol", "calendar_age"}.issubset(recent.columns):
            recent["symbol"] = recent["symbol"].astype(str).str.upper().str.strip()
            recent["calendar_age"] = pd.to_numeric(recent["calendar_age"], errors="coerce")
            eligible_new = recent.loc[
                recent["calendar_age"].ge(20) & recent["symbol"].ne(""),
                "symbol"
            ].dropna().tolist()

    symbols = list(dict.fromkeys(core + eligible_new))
    if len(core) < 100:
        raise RuntimeError(f"Core Nifty universe contains only {len(core)} symbols; refusing to run")
    if eligible_new:
        print(f"Recent listings eligible for ML pipeline: {len(eligible_new)}")
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


def _download_yahoo_batch(batch: list[str], start: str) -> pd.DataFrame:
    """Download one batch with retries and no yfinance worker threads.

    yfinance can use a local sqlite-backed cookie/crumb cache. Concurrent
    downloads occasionally collide on that database and raise
    'database is locked'. Serialising the request inside each batch avoids
    that failure mode while retries handle transient network/rate-limit errors.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return yf.download(
                [f"{s}.NS" for s in batch],
                start=start,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=False,
            )
        except Exception as exc:
            last_error = exc
            print(f"Yahoo batch {batch[0]}..{batch[-1]} attempt {attempt + 1}/3 failed: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Yahoo batch failed after 3 attempts: {last_error}")


def download_history(symbols: list[str], start: str) -> pd.DataFrame:
    rows = []
    # Smaller batches reduce the amount of work lost when Yahoo has a transient
    # failure and reduce pressure on yfinance's local cache.
    batch_size = 15
    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        try:
            raw = _download_yahoo_batch(batch, start)
        except Exception as exc:
            print(f"Download batch failed after retries; isolating symbols: {exc}")
            # A failed batch must not discard good data for the other symbols.
            for symbol in batch:
                try:
                    raw = _download_yahoo_batch([symbol], start)
                    if raw.empty:
                        continue
                    x = _normalise_history(raw, symbol)
                    if not x.empty:
                        rows.append(x)
                except Exception as symbol_exc:
                    print(f"Skipping {symbol} after isolated retries: {symbol_exc}")
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
    df["return_1d"] = g["close"].pct_change()
    df["return_5d"] = g["close"].pct_change(5)
    df["return_20d"] = g["close"].pct_change(20)
    df["sma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["sma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
    df["ema20"] = g["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["ema50"] = g["close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["rsi14"] = g["close"].transform(rsi)
    prev_close = g["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr14_pct"] = true_range.groupby(df["symbol"], group_keys=False).transform(lambda x: x.rolling(14).mean()) / df["close"].replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["volatility20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    rolling_std20 = g["close"].transform(lambda x: x.rolling(20).std())
    df["bb_position"] = (df["close"] - df["sma20"]) / (2.0 * rolling_std20.replace(0, np.nan))
    df["close_sma20_gap"] = df["close"] / df["sma20"].replace(0, np.nan) - 1.0
    df["close_sma50_gap"] = df["close"] / df["sma50"].replace(0, np.nan) - 1.0
    df["volume_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)
    df["volume_ma5"] = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_trend5"] = df["volume_ma5"] / df["volume_ma20"].replace(0, np.nan)
    return df


def market_regime(df: pd.DataFrame) -> str:
    """Classify the broad market regime using only completed historical bars."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    daily = x.groupby("date")["close"].median().sort_index().dropna()
    if len(daily) < 50:
        return "NEUTRAL"
    sma20 = daily.rolling(20).mean().iloc[-1]
    sma50 = daily.rolling(50).mean().iloc[-1]
    ret20 = daily.iloc[-1] / daily.iloc[-21] - 1.0
    if daily.iloc[-1] > sma20 > sma50 and ret20 >= 0.03:
        return "BULL"
    if daily.iloc[-1] < sma20 < sma50 and ret20 <= -0.03:
        return "BEAR"
    return "NEUTRAL"


def market_regime_score(latest: pd.DataFrame, regime: str) -> pd.Series:
    """Small regime-aware adjustment; ranking remains primarily technical+fundamental."""
    score = pd.Series(0.0, index=latest.index)
    momentum = latest["return_20d"].rank(pct=True)
    if regime == "BULL":
        score = momentum * 2.0
    elif regime == "BEAR":
        score = 0.0
    return score


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
    columns = ["symbol", "updated_at", "sector"] + list(FUNDAMENTAL_FIELDS)
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
        row = {"symbol": symbol, "updated_at": now.isoformat(timespec="seconds"), "sector": np.nan, **{f: np.nan for f in FUNDAMENTAL_FIELDS}}
        try:
            info = yf.Ticker(f"{symbol}.NS").get_info()
            row["sector"] = info.get("sector") or np.nan
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


def record_fundamentals_snapshot(fundamentals: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
    """Persist the fundamentals actually used for a market-date ranking."""
    if fundamentals is None or fundamentals.empty:
        return
    columns = ["as_of_date", "symbol", "sector"] + list(FUNDAMENTAL_FIELDS)
    snapshot = fundamentals.copy()
    snapshot["as_of_date"] = pd.Timestamp(as_of_date).normalize()
    if "sector" not in snapshot.columns:
        snapshot["sector"] = np.nan
    for field in FUNDAMENTAL_FIELDS:
        if field not in snapshot.columns:
            snapshot[field] = np.nan
    snapshot = snapshot[columns]
    old = pd.read_csv(FUNDAMENTALS_HISTORY_FILE) if FUNDAMENTALS_HISTORY_FILE.exists() else pd.DataFrame(columns=columns)
    combined = pd.concat([old, snapshot], ignore_index=True)
    combined["as_of_date"] = pd.to_datetime(combined["as_of_date"], errors="coerce").dt.normalize()
    combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
    combined = combined.dropna(subset=["as_of_date", "symbol"]).drop_duplicates(
        ["as_of_date", "symbol"], keep="last"
    ).sort_values(["as_of_date", "symbol"])
    combined.to_csv(FUNDAMENTALS_HISTORY_FILE, index=False)


def fundamental_score(fundamentals: pd.DataFrame) -> pd.Series:
    """Score fundamentals while shrinking incomplete records toward neutral.

    Missing/invalid fields must not be renormalized into a full 20-point score:
    otherwise one available metric can dominate the entire fundamental block.
    """
    total_weight = float(sum(weight for weight, _ in FUNDAMENTAL_FIELDS.values()))
    result = pd.Series(0.0, index=fundamentals.index, dtype=float)
    weight_used = pd.Series(0.0, index=fundamentals.index, dtype=float)

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

    # Full coverage maps to 0-20. Partial coverage shrinks the deviation
    # from neutral (10) in proportion to usable fundamental weight.
    raw = (result / total_weight) * 20.0
    coverage = (weight_used / total_weight).clip(0.0, 1.0)
    scored = 10.0 + (raw - 10.0) * coverage
    return scored.clip(0, 20)


def rank_stocks(df: pd.DataFrame, fundamentals: pd.DataFrame | None = None, use_market_regime: bool = False) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    latest_date = work["date"].max()
    latest = work[work["date"] == latest_date].dropna(subset=FEATURE_COLUMNS).copy()
    latest["technical_score"] = technical_score(latest)
    regime = market_regime(df)
    latest["market_regime"] = regime
    latest["market_regime_score"] = market_regime_score(latest, regime) if use_market_regime else 0.0
    latest["technical_score"] = (latest["technical_score"] + latest["market_regime_score"]).clip(0, 80)
    if fundamentals is None:
        fundamentals = pd.DataFrame({"symbol": latest["symbol"]})
    fundamental_cols = ["symbol"] + list(FUNDAMENTAL_FIELDS)
    available = fundamentals[[c for c in fundamental_cols if c in fundamentals.columns]].copy()
    latest = latest.merge(available, on="symbol", how="left")
    latest["fundamental_score"] = fundamental_score(latest)
    total_weight = float(sum(weight for weight, _ in FUNDAMENTAL_FIELDS.values()))
    latest["fundamental_coverage_pct"] = (
        latest[list(FUNDAMENTAL_FIELDS)].notna().mul(
            pd.Series({k: v[0] for k, v in FUNDAMENTAL_FIELDS.items()})
        ).sum(axis=1) / total_weight * 100.0
    ).clip(0, 100)
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


def _fit_ensemble(x: pd.DataFrame, y: pd.Series) -> dict:
    hist = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    extra = ExtraTreesRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=42)
    hist.fit(x, y)
    extra.fit(x, y)
    return {"models": [hist, extra], "weights": [0.70, 0.30], "version": "ensemble_v1"}


def _ensemble_predict(bundle: dict, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    models = bundle["models"]
    weights = np.asarray(bundle.get("weights", [1.0 / len(models)] * len(models)), dtype=float)
    weights = weights / weights.sum()
    preds = np.column_stack([m.predict(x) for m in models])
    return preds @ weights, preds.std(axis=1)


def train(df: pd.DataFrame) -> None:
    work = add_targets(features(df)).dropna(subset=FEATURE_COLUMNS + list(TARGETS.values()))
    if len(work) < 500:
        raise RuntimeError(f"Not enough training rows: {len(work)}")
    for name, target in TARGETS.items():
        bundle = _fit_ensemble(work[FEATURE_COLUMNS], work[target])
        joblib.dump(bundle, MODELS / f"{name}.joblib")


def models_ready() -> bool:
    if not all((MODELS / f"{name}.joblib").exists() for name in TARGETS):
        return False
    try:
        expected = len(FEATURE_COLUMNS)
        for name in TARGETS:
            bundle = joblib.load(MODELS / f"{name}.joblib")
            if not isinstance(bundle, dict) or bundle.get("version") != "ensemble_v1":
                print(f"Model {name} is legacy/non-ensemble; retraining.")
                return False
            models = bundle.get("models", [])
            if len(models) != 2 or any(getattr(m, "n_features_in_", None) != expected for m in models):
                print(f"Model {name} does not match the current feature set; retraining.")
                return False
        return True
    except Exception as exc:
        print(f"Stored model validation failed; retraining: {exc}")
        return False


def _next_trading_date(last_date: pd.Timestamp, history_dates: pd.Series) -> pd.Timestamp:
    last = pd.Timestamp(last_date).normalize()
    schedule = mcal.get_calendar("XNSE").schedule(start_date=last + pd.Timedelta(days=1), end_date=last + pd.Timedelta(days=14))
    if schedule.empty:
        raise RuntimeError(f"Could not determine next NSE trading session after {last.date()}")
    return pd.Timestamp(schedule.index[0]).normalize()


def predict_top10(df: pd.DataFrame, ranking: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    ranked = ranking.head(20)[["symbol", "date", "close", "rank", "total_score", "technical_score", "fundamental_score", "market_regime"]].copy()
    if len(ranked) < 10:
        raise RuntimeError(f"Expected at least 10 ranked stocks, found {len(ranked)}")
    latest = features(df).sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest["symbol"].isin(ranked["symbol"])].dropna(subset=FEATURE_COLUMNS)
    if len(latest) < 10:
        raise RuntimeError(f"Expected features for at least 10 candidates, found {len(latest)}")
    if not models_ready():
        raise RuntimeError("Prediction models are missing")
    spreads = []
    for name in TARGETS:
        bundle = joblib.load(MODELS / f"{name}.joblib")
        pred, spread = _ensemble_predict(bundle, latest[FEATURE_COLUMNS])
        latest[f"predicted_{name}"] = latest["close"] * (1 + pred)
        spreads.append(spread)
    latest["prediction_spread"] = np.mean(np.column_stack(spreads), axis=1)
    latest["predicted_high"] = latest[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
    latest["predicted_low"] = latest[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)
    out = latest[["date", "symbol", "close", "predicted_open", "predicted_high", "predicted_low", "predicted_close", "prediction_spread"]].copy().rename(columns={"date": "prediction_date", "close": "base_close"})
    lookup = ranked.set_index("symbol")
    out["rank"] = out["symbol"].map(lookup["rank"]).astype(int)
    out["score"] = out["symbol"].map(lookup["total_score"])
    out["technical_score"] = out["symbol"].map(lookup["technical_score"])
    out["fundamental_score"] = out["symbol"].map(lookup["fundamental_score"])
    expected_move = (out["predicted_close"] / out["base_close"] - 1).abs()
    uncertainty_ratio = out["prediction_spread"] / expected_move.replace(0, np.nan)
    out["confidence_score"] = (100 / (1 + uncertainty_ratio)).clip(0, 100).fillna(0)
    out["target_date"] = pd.Timestamp(target_date).normalize()
    out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidates = out.sort_values("rank").copy()
    selection_method = "ranking_top10"
    selected_symbols = set(candidates.head(10)["symbol"])

    # Apply the recent-error selector only after its own out-of-sample
    # A/B validation gate has demonstrated improvement over pure rank selection.
    # Until then, production stays on the safer rank-based Top-10.
    try:
        validation = pd.read_csv(SELECTION_VALIDATION_FILE) if SELECTION_VALIDATION_FILE.exists() else pd.DataFrame()
        selector_validated = (
            not validation.empty
            and "recent_error_promotion_evidence" in validation.columns
            and bool(validation["recent_error_promotion_evidence"].fillna(False).astype(bool).any())
        )
        if selector_validated and EVALUATIONS_FILE.exists():
            ev = pd.read_csv(EVALUATIONS_FILE)
            ev["target_date"] = pd.to_datetime(ev["target_date"], errors="coerce").dt.normalize()
            ev["close_abs_pct_error"] = pd.to_numeric(ev["close_abs_pct_error"], errors="coerce")
            ev = ev.dropna(subset=["target_date", "symbol", "close_abs_pct_error"])
            if not ev.empty:
                cutoff = ev["target_date"].max() - pd.Timedelta(days=21)
                recent = ev[ev["target_date"] >= cutoff]
                if len(recent) >= 30:
                    recent_error = recent.groupby("symbol")["close_abs_pct_error"].mean()
                    candidates["recent_close_error"] = candidates["symbol"].map(recent_error)
                    median_error = float(recent_error.median())
                    excess = ((candidates["recent_close_error"] / max(median_error, 1e-6)) - 1.0).clip(lower=0, upper=2)
                    candidates["selection_penalty"] = (excess * 0.75).fillna(0.0)
                    # No recent history means no evidence of persistent error; do not penalize it.
                    candidates["selection_priority"] = candidates["rank"] + candidates["selection_penalty"]
                    selected_symbols = set(candidates.sort_values(["selection_priority", "rank"]).head(10)["symbol"])
                    selection_method = "ranking_recent_error_adjusted"
                else:
                    candidates["recent_close_error"] = np.nan
                    candidates["selection_penalty"] = 0.0
    except Exception as exc:
        print(f"Recent-error selector unavailable; retaining Top-10 ranking: {exc}")

    try:
        analysis = pd.read_csv(SELECTION_VALIDATION_FILE)
        validated = (
            not analysis.empty
            and "confidence_promotion_evidence" in analysis.columns
            and analysis["confidence_promotion_evidence"].fillna(False).astype(bool).any()
        )
        if validated:
            candidates["selection_priority"] = candidates["rank"] + ((100.0 - candidates["confidence_score"]) / 100.0)
            selected_symbols = set(candidates.sort_values(["selection_priority", "rank"]).head(10)["symbol"])
            selection_method = "validated_confidence_tiebreak"
    except Exception:
        print("Confidence selector unavailable or not validated; retaining current selection")
    candidates["selection_method"] = selection_method
    candidates["selected"] = candidates["symbol"].isin(selected_symbols).astype(int)
    candidates.to_csv(CANDIDATES_FILE, index=False)
    existing_candidates = pd.read_csv(CANDIDATE_HISTORY_FILE) if CANDIDATE_HISTORY_FILE.exists() else pd.DataFrame()
    history_candidates = pd.concat([existing_candidates, candidates], ignore_index=True)
    history_candidates = history_candidates.drop_duplicates(["target_date", "symbol"], keep="first")
    history_candidates.to_csv(CANDIDATE_HISTORY_FILE, index=False)
    selected = candidates[candidates["selected"] == 1].copy()
    if len(selected) != 10:
        raise RuntimeError(f"Confidence selector produced {len(selected)} stocks; expected 10")
    selected = selected.sort_values("rank").reset_index(drop=True)
    _validate_prediction_session(selected, target_date)
    return selected


def _save_next_session_prediction(hist: pd.DataFrame, ranking: pd.DataFrame, target_date: pd.Timestamp, existing: pd.DataFrame) -> None:
    if not existing.empty and "target_date" in existing.columns:
        existing_dates = pd.to_datetime(existing["target_date"], errors="coerce").dt.normalize()
        if (existing_dates == target_date.normalize()).any():
            same_target = existing[existing_dates == target_date.normalize()].copy()
            _validate_prediction_session(same_target, target_date)
            return
    prediction = predict_top10(hist, ranking, target_date)
    combined = pd.concat([existing, prediction], ignore_index=True) if not existing.empty else prediction
    combined = combined.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    combined.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Backfilled next prediction session: {target_date.date()} ({len(prediction)} stocks)")


def run_morning() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    validate_data_quality(hist, symbols, "morning")
    fundamentals = update_fundamentals(symbols)
    record_fundamentals_snapshot(fundamentals, pd.to_datetime(hist["date"]).max())
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
            _validate_prediction_session(same_target, target_date)
            print(f"Predictions already exist for {target_date.date()}; keeping existing output unchanged.")
            return
    predictions = predict_top10(hist, ranking, target_date)
    combined = pd.concat([existing, predictions], ignore_index=True) if not existing.empty else predictions
    combined = combined.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank"])
    combined.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Morning run complete: target session {target_date.date()}, {len(predictions)} predictions created")



def _paper_trade_completed_predictions(predictions: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the Top-5 paper portfolio for every newly completed prediction session.

    The paper portfolio is deliberately separate from model training. The model
    learns from actual OHLC targets, while paper trading measures the practical
    result of the selected Top-5 basket.
    """
    if predictions.empty:
        return pd.DataFrame()

    p = predictions.copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["rank"] = pd.to_numeric(p["rank"], errors="coerce")
    p = p.dropna(subset=["target_date", "symbol", "rank"])

    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    actual = h.rename(columns={
        "date": "target_date",
        "open": "actual_open",
        "high": "actual_high",
        "low": "actual_low",
        "close": "actual_close",
    })[["symbol", "target_date", "actual_open", "actual_high", "actual_low", "actual_close"]]

    rows = p.merge(actual, on=["symbol", "target_date"], how="inner")
    rows = rows.dropna(subset=["actual_open", "actual_close"])
    rows = rows[rows["target_date"] <= h["date"].max()].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["signal"] = np.where(rows["rank"] <= PAPER_TRADE_TOP_N, "BUY", "SKIP")
    rows["entry_price"] = np.where(rows["signal"].eq("BUY"), rows["actual_open"], np.nan)
    rows["exit_price"] = np.where(rows["signal"].eq("BUY"), rows["actual_close"], np.nan)
    rows["return_pct"] = np.where(
        rows["signal"].eq("BUY") & rows["entry_price"].ne(0),
        (rows["exit_price"] / rows["entry_price"] - 1.0) * 100.0,
        np.nan,
    )

    # Fixed-share paper portfolio: determine integer quantity from the
    # latest completed close (base_close) before the target session.
    # This prevents hindsight from using the actual next-day open to size shares.
    allocation = PAPER_CAPITAL / PAPER_TRADE_TOP_N
    rows["reference_price"] = pd.to_numeric(rows.get("base_close"), errors="coerce")
    rows["quantity"] = np.where(
        rows["signal"].eq("BUY") & rows["reference_price"].gt(0),
        np.floor(allocation / rows["reference_price"]).astype(int),
        0,
    )
    rows["planned_capital"] = np.where(
        rows["signal"].eq("BUY"),
        rows["quantity"] * rows["reference_price"],
        0.0,
    )
    rows["position_value"] = np.where(
        rows["signal"].eq("BUY"),
        rows["quantity"] * pd.to_numeric(rows["entry_price"], errors="coerce"),
        0.0,
    )
    rows["profit_loss"] = np.where(
        rows["signal"].eq("BUY"),
        rows["quantity"] * (
            pd.to_numeric(rows["exit_price"], errors="coerce")
            - pd.to_numeric(rows["entry_price"], errors="coerce")
        ),
        0.0,
    )

    keep = [
        "prediction_date", "target_date", "symbol", "rank", "score",
        "signal", "reference_price", "quantity", "planned_capital",
        "entry_price", "exit_price", "return_pct",
        "position_value", "profit_loss",
    ]
    rows = rows[keep]

    if PAPER_TRADES_FILE.exists():
        old = pd.read_csv(PAPER_TRADES_FILE)
        combined = pd.concat([old, rows], ignore_index=True)
    else:
        combined = rows

    combined["target_date"] = pd.to_datetime(combined["target_date"], errors="coerce").dt.normalize()
    combined["prediction_date"] = pd.to_datetime(combined["prediction_date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["target_date", "symbol"])
    combined = combined.drop_duplicates(["target_date", "symbol"], keep="first")
    combined = combined.sort_values(["target_date", "rank"]).reset_index(drop=True)
    combined.to_csv(PAPER_TRADES_FILE, index=False)

    # Daily portfolio return is actual fixed-share P&L divided by the
    # fixed ₹100,000 paper capital. This includes the cash left unused by
    # integer-share sizing and avoids fractional-share assumptions.
    buy = combined[combined["signal"].eq("BUY")].copy()
    daily = buy.groupby("target_date", as_index=False).agg(
        trades=("symbol", "count"),
        daily_profit_loss=("profit_loss", "sum"),
    )
    daily["daily_return_pct"] = daily["daily_profit_loss"] / PAPER_CAPITAL * 100.0
    if daily.empty:
        return combined

    daily = daily.sort_values("target_date")
    existing_portfolio = pd.read_csv(PORTFOLIO_FILE) if PORTFOLIO_FILE.exists() else pd.DataFrame()
    if not existing_portfolio.empty:
        existing_portfolio["target_date"] = pd.to_datetime(existing_portfolio["target_date"], errors="coerce").dt.normalize()

    history = daily.copy()
    history["portfolio_value"] = np.nan
    history["cumulative_return_pct"] = np.nan
    prior_dates = set(existing_portfolio["target_date"].dropna()) if not existing_portfolio.empty else set()

    # Position sizing is fixed from the original ₹100,000 paper capital, so
    # portfolio P&L is additive rather than compounded. Recompute the
    # cumulative value from the fixed starting capital for all known sessions.
    combined_daily = pd.concat(
        [
            existing_portfolio[["target_date", "daily_profit_loss"]]
            if not existing_portfolio.empty and "daily_profit_loss" in existing_portfolio.columns
            else pd.DataFrame(columns=["target_date", "daily_profit_loss"]),
            history[["target_date", "daily_profit_loss"]],
        ],
        ignore_index=True,
    )
    combined_daily["target_date"] = pd.to_datetime(combined_daily["target_date"], errors="coerce").dt.normalize()
    combined_daily["daily_profit_loss"] = pd.to_numeric(combined_daily["daily_profit_loss"], errors="coerce").fillna(0.0)
    combined_daily = combined_daily.dropna(subset=["target_date"]).drop_duplicates("target_date", keep="last").sort_values("target_date")
    combined_daily["portfolio_value"] = PAPER_CAPITAL + combined_daily["daily_profit_loss"].cumsum()
    combined_daily["cumulative_return_pct"] = (combined_daily["portfolio_value"] / PAPER_CAPITAL - 1.0) * 100.0

    history = history.drop(columns=["portfolio_value", "cumulative_return_pct"], errors="ignore").merge(
        combined_daily[["target_date", "portfolio_value", "cumulative_return_pct"]],
        on="target_date",
        how="left",
    )

    history["target_date"] = pd.to_datetime(history["target_date"]).dt.normalize()
    history = pd.concat([existing_portfolio, history], ignore_index=True)
    history = history.drop_duplicates(["target_date"], keep="last").sort_values("target_date")
    history.to_csv(PORTFOLIO_FILE, index=False)
    print(
        f"Paper trading updated: Top-{PAPER_TRADE_TOP_N}, "
        f"sessions={len(history)}, latest={history['target_date'].max().date()}, "
        f"portfolio=₹{float(history.iloc[-1]['portfolio_value']):,.2f}"
    )
    return combined


def run_evening() -> None:
    symbols = load_universe()
    hist = update_history(symbols)
    validate_data_quality(hist, symbols, "evening")
    if not PREDICTIONS_FILE.exists():
        print("No stored prediction; training on all available history and creating the next-session prediction.")
        if not models_ready():
            train(hist)
        fundamentals = update_fundamentals(symbols)
        record_fundamentals_snapshot(fundamentals, pd.to_datetime(hist["date"]).max())
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
        record_fundamentals_snapshot(fundamentals, pd.to_datetime(hist["date"]).max())
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

    # Paper-trade the Top-5 from the same predictions whose actual market
    # session is now complete. This is measurement only; P&L is never used
    # directly as a training target.
    _paper_trade_completed_predictions(predictions, hist)

    # Retrain on all realized actual market data, including the newly completed
    # prediction target session.
    train(hist)

    target_date = _next_trading_date(latest_actual_date, hist["date"])
    fundamentals = update_fundamentals(symbols)
    record_fundamentals_snapshot(fundamentals, pd.to_datetime(hist["date"]).max())
    ranking = rank_stocks(features(hist), fundamentals)
    ranking.to_csv(RANKING_FILE, index=False)
    _save_next_session_prediction(hist, ranking, target_date, predictions)

    sessions = evals["target_date"].nunique()
    latest_session = evals["target_date"].max().date()
    print(f"Evening run complete: latest evaluated session={latest_session}, total evaluations={len(evals)}, sessions={sessions}, next target={target_date.date()}, models retrained={models_ready()}")


if __name__ == "__main__":
    run_morning()
