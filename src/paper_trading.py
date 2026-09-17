from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
PREDICTIONS_FILE = DATA / "predictions.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
TRADES_FILE = DATA / "paper_trades.csv"
PORTFOLIO_FILE = DATA / "portfolio_daily.csv"
STRATEGY_FILE = DATA / "trading_strategy_metrics.csv"
ENTRY_MODEL = MODELS / "entry_model.joblib"
MFE_MODEL = MODELS / "mfe_model.joblib"
MAE_MODEL = MODELS / "mae_model.joblib"

FEATURES = [
    "predicted_return",
    "predicted_upside",
    "predicted_downside",
    "rank",
    "score",
]


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"], errors="coerce").dt.normalize()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    return predictions, history


def _build_training_rows(predictions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    actual = history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"})
    cols = ["symbol", "target_date", "actual_open", "actual_high", "actual_low", "actual_close"]
    rows = predictions.merge(actual[cols], on=["symbol", "target_date"], how="inner")
    rows = rows.dropna(subset=["base_close", "predicted_open", "predicted_high", "predicted_low", "predicted_close", "actual_open", "actual_high", "actual_low", "actual_close"])
    rows["predicted_return"] = rows["predicted_close"] / rows["base_close"] - 1
    rows["predicted_upside"] = rows["predicted_high"] / rows["base_close"] - 1
    rows["predicted_downside"] = rows["predicted_low"] / rows["base_close"] - 1
    rows["mfe"] = rows["actual_high"] / rows["actual_open"] - 1
    rows["mae"] = rows["actual_low"] / rows["actual_open"] - 1
    rows["close_return"] = rows["actual_close"] / rows["actual_open"] - 1
    rows["profitable_close"] = (rows["close_return"] > 0).astype(int)
    return rows.sort_values(["target_date", "rank", "symbol"]).reset_index(drop=True)


def _train_models(history_rows: pd.DataFrame) -> None:
    if len(history_rows) < 100:
        return
    x = history_rows[FEATURES].astype(float)
    entry = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    entry.fit(x, history_rows["profitable_close"])
    mfe = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    mfe.fit(x, history_rows["mfe"])
    mae = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    mae.fit(x, history_rows["mae"])
    joblib.dump(entry, ENTRY_MODEL)
    joblib.dump(mfe, MFE_MODEL)
    joblib.dump(mae, MAE_MODEL)


def _simulate_day(row: pd.Series, entry_model, mfe_model, mae_model, learned: bool) -> dict:
    base = float(row["base_close"])
    actual_open = float(row["actual_open"])
    actual_high = float(row["actual_high"])
    actual_low = float(row["actual_low"])
    actual_close = float(row["actual_close"])
    x = pd.DataFrame([{f: float(row[f]) for f in FEATURES}])
    probability = float(entry_model.predict_proba(x)[0, 1]) if entry_model is not None else 1.0

    # Do not force a trade when history is too short. Once trained, the model
    # learns whether the next-day entry historically had a positive close.
    enter = (not learned) or probability >= 0.55
    if not enter:
        return {
            "symbol": row["symbol"], "target_date": row["target_date"], "rank": int(row["rank"]),
            "signal": "SKIP", "entry_price": np.nan, "exit_price": np.nan, "exit_reason": "learned_entry_filter",
            "profit_loss": 0.0, "return_pct": 0.0, "entry_probability": probability,
        }

    if learned:
        target_pct = max(0.003, min(0.08, float(mfe_model.predict(x)[0])))
        stop_pct = max(0.003, min(0.05, abs(float(mae_model.predict(x)[0]))))
    else:
        target_pct = max(0.005, float(row["predicted_high"]) / base - 1)
        stop_pct = max(0.005, 1 - float(row["predicted_low"]) / base)

    target = actual_open * (1 + target_pct)
    stop = actual_open * (1 - stop_pct)

    hit_target = actual_high >= target
    hit_stop = actual_low <= stop
    # Daily OHLC has no intraday order. If both levels were reached, use the
    # conservative stop outcome rather than assuming the profitable level first.
    if hit_stop and hit_target:
        exit_price, reason = stop, "stop_and_target_same_day_conservative"
    elif hit_target:
        exit_price, reason = target, "target"
    elif hit_stop:
        exit_price, reason = stop, "stop_loss"
    else:
        exit_price, reason = actual_close, "close"

    pnl = exit_price - actual_open
    return {
        "symbol": row["symbol"], "target_date": row["target_date"], "rank": int(row["rank"]),
        "signal": "BUY", "entry_price": actual_open, "exit_price": exit_price, "exit_reason": reason,
        "target_pct": target_pct * 100, "stop_pct": stop_pct * 100,
        "profit_loss": pnl, "return_pct": pnl / actual_open * 100, "entry_probability": probability,
    }


def run_paper_trading() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        print("Paper trading skipped: predictions or history file missing.")
        return pd.DataFrame()

    predictions, history = _load()
    rows = _build_training_rows(predictions, history)
    if rows.empty:
        print("Paper trading skipped: no completed prediction sessions yet.")
        return pd.DataFrame()

    completed_dates = sorted(rows["target_date"].dropna().unique())
    old = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    done_keys = set(zip(pd.to_datetime(old.get("target_date", pd.Series(dtype=str)), errors="coerce").dt.strftime("%Y-%m-%d"), old.get("symbol", pd.Series(dtype=str)))) if not old.empty else set()

    all_trades = []
    # Walk forward: each session's entry/exit model is trained only on sessions
    # strictly before that session. This prevents future information leakage.
    for target_date in completed_dates:
        prior = rows[rows["target_date"] < target_date]
        learned = len(prior) >= 100
        if learned:
            _train_models(prior)
            entry_model = joblib.load(ENTRY_MODEL)
            mfe_model = joblib.load(MFE_MODEL)
            mae_model = joblib.load(MAE_MODEL)
        else:
            entry_model = mfe_model = mae_model = None

        day_rows = rows[rows["target_date"] == target_date]
        for _, row in day_rows.iterrows():
            key = (pd.Timestamp(target_date).strftime("%Y-%m-%d"), str(row["symbol"]))
            if key in done_keys:
                continue
            trade = _simulate_day(row, entry_model, mfe_model, mae_model, learned)
            trade["model_learned"] = learned
            all_trades.append(trade)

    if all_trades:
        new = pd.DataFrame(all_trades)
        if old.empty:
            trades = new
        else:
            trades = pd.concat([old, new], ignore_index=True)
        trades["target_date"] = pd.to_datetime(trades["target_date"], errors="coerce").dt.normalize()
        trades = trades.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank", "symbol"])
        trades.to_csv(TRADES_FILE, index=False)
    else:
        trades = old

    if trades.empty:
        return trades

    # Equal-weight portfolio accounting for a simple, reproducible paper test.
    settled = trades[trades["signal"] == "BUY"].copy()
    daily = settled.groupby("target_date", as_index=False).agg(
        trades=("symbol", "count"),
        winners=("profit_loss", lambda s: int((s > 0).sum())),
        losers=("profit_loss", lambda s: int((s < 0).sum())),
        avg_return_pct=("return_pct", "mean"),
        daily_return_pct=("return_pct", "mean"),
        daily_pnl_per_100k=("profit_loss", lambda s: float(s.sum() / max(len(s), 1) * 10)),
    )
    daily["win_rate_pct"] = daily["winners"] / daily["trades"] * 100
    daily["portfolio_value"] = 100000 * (1 + daily["daily_return_pct"] / 100).cumprod()
    daily["cumulative_return_pct"] = (daily["portfolio_value"] / 100000 - 1) * 100
    daily["peak_value"] = daily["portfolio_value"].cummax()
    daily["drawdown_pct"] = (daily["portfolio_value"] / daily["peak_value"] - 1) * 100
    daily.to_csv(PORTFOLIO_FILE, index=False)

    metrics = pd.DataFrame([{
        "as_of": daily["target_date"].max(),
        "sessions": len(daily),
        "trades": int(len(settled)),
        "win_rate_pct": float((settled["profit_loss"] > 0).mean() * 100),
        "total_return_pct": float(daily["cumulative_return_pct"].iloc[-1]),
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "avg_trade_return_pct": float(settled["return_pct"].mean()),
        "profit_factor": float(settled.loc[settled["profit_loss"] > 0, "profit_loss"].sum() / max(abs(settled.loc[settled["profit_loss"] < 0, "profit_loss"].sum()), 1e-9)),
    }])
    metrics.to_csv(STRATEGY_FILE, index=False)
    print(f"Paper trading complete: {len(settled)} settled trades, {len(daily)} sessions, return={metrics.iloc[0]['total_return_pct']:.2f}%")
    return trades


if __name__ == "__main__":
    run_paper_trading()
