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
LEARNING_FILE = DATA / "paper_trading_learning.csv"

STARTING_CAPITAL = 100_000.0
ROUND_TRIP_COST_PCT = 0.001  # 0.10% of allocated capital per completed trade.

FEATURES = [
    "predicted_return",
    "predicted_upside",
    "predicted_downside",
    "prediction_spread",
    "rank",
    "score",
    "technical_score",
    "fundamental_score",
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
    if "prediction_spread" not in rows.columns:
        rows["prediction_spread"] = 0.0
    rows["prediction_spread"] = pd.to_numeric(rows["prediction_spread"], errors="coerce").fillna(0.0)
    rows["mfe"] = rows["actual_high"] / rows["actual_open"] - 1
    rows["mae"] = rows["actual_low"] / rows["actual_open"] - 1
    rows["close_return"] = rows["actual_close"] / rows["actual_open"] - 1
    rows["profitable_close"] = (rows["close_return"] > 0).astype(int)
    rows = rows.sort_values(["target_date", "rank", "symbol"]).reset_index(drop=True)
    learning_cols = ["prediction_date","target_date","symbol","rank","score","base_close",
        "predicted_return","predicted_upside","predicted_downside","prediction_spread",
        "actual_open","actual_high","actual_low","actual_close","mfe","mae","close_return",
        "profitable_close"]
    existing = pd.read_csv(LEARNING_FILE) if LEARNING_FILE.exists() else pd.DataFrame()
    learning = rows[[x for x in learning_cols if x in rows.columns]].copy()
    if not existing.empty:
        learning = pd.concat([existing, learning], ignore_index=True)
    if not learning.empty:
        for col in ["prediction_date","target_date"]:
            if col in learning.columns:
                learning[col] = pd.to_datetime(learning[col], errors="coerce").dt.normalize()
        learning = learning.drop_duplicates(["target_date","symbol"], keep="first")
        learning = learning.sort_values(["target_date","rank","symbol"])
        learning.to_csv(LEARNING_FILE, index=False)
    return rows


def _train_models(history_rows: pd.DataFrame) -> None:
    if len(history_rows) < 100 or history_rows["profitable_close"].nunique() < 2:
        return
    x = history_rows[FEATURES].copy()
    for col in ["technical_score", "fundamental_score"]:
        if col not in x.columns:
            x[col] = 0.0
    x = x.astype(float)
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
    values = {f: (float(row[f]) if f in row.index and pd.notna(row[f]) else 0.0) for f in FEATURES}
    x = pd.DataFrame([values])
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

    # Keep the per-share move here; rupee P/L is assigned after equal-capital
    # sizing in _build_portfolio so it is consistent with the portfolio value.
    pnl_per_share = exit_price - actual_open
    return {
        "symbol": row["symbol"], "target_date": row["target_date"], "rank": int(row["rank"]),
        "signal": "BUY", "entry_price": actual_open, "exit_price": exit_price, "exit_reason": reason,
        "target_pct": target_pct * 100, "stop_pct": stop_pct * 100,
        "pnl_per_share": pnl_per_share, "profit_loss": pnl_per_share,
        "return_pct": pnl_per_share / actual_open * 100, "entry_probability": probability,
    }


def _build_portfolio(trades: pd.DataFrame) -> pd.DataFrame:
    """Apply equal-capital sizing sequentially, with a fixed round-trip cost."""
    settled = trades[trades["signal"] == "BUY"].copy()
    if settled.empty:
        return pd.DataFrame()

    settled["target_date"] = pd.to_datetime(settled["target_date"], errors="coerce").dt.normalize()
    settled = settled.sort_values(["target_date", "rank", "symbol"]).reset_index()

    portfolio = STARTING_CAPITAL
    daily_rows = []
    net_pnl_by_index: dict[int, float] = {}
    shares_by_index: dict[int, float] = {}
    allocation_by_index: dict[int, float] = {}
    costs_by_index: dict[int, float] = {}

    for target_date, day in settled.groupby("target_date", sort=True):
        allocation = portfolio / len(day)
        day_gross_pnl = 0.0
        day_costs = 0.0

        for idx, row in day.iterrows():
            entry = float(row["entry_price"])
            exit_price = float(row["exit_price"])
            shares = allocation / entry
            gross_pnl = shares * (exit_price - entry)
            costs = allocation * ROUND_TRIP_COST_PCT
            net_pnl = gross_pnl - costs

            shares_by_index[idx] = shares
            allocation_by_index[idx] = allocation
            costs_by_index[idx] = costs
            net_pnl_by_index[idx] = net_pnl
            day_gross_pnl += gross_pnl
            day_costs += costs

        day_net_pnl = day_gross_pnl - day_costs
        daily_return_pct = day_net_pnl / portfolio * 100
        portfolio += day_net_pnl
        daily_rows.append({
            "target_date": target_date,
            "trades": len(day),
            "winners": int(sum(net_pnl_by_index[idx] > 0 for idx in day.index)),
            "losers": int(sum(net_pnl_by_index[idx] < 0 for idx in day.index)),
            "gross_pnl": day_gross_pnl,
            "costs": day_costs,
            "net_pnl": day_net_pnl,
            "daily_return_pct": daily_return_pct,
            "portfolio_value": portfolio,
        })

    settled["shares"] = [shares_by_index[i] for i in settled["index"]]
    settled["allocated_capital"] = [allocation_by_index[i] for i in settled["index"]]
    settled["costs"] = [costs_by_index[i] for i in settled["index"]]
    settled["gross_profit_loss"] = settled["shares"] * (settled["exit_price"] - settled["entry_price"])
    settled["profit_loss"] = [net_pnl_by_index[i] for i in settled["index"]]
    settled["return_pct"] = settled["profit_loss"] / settled["allocated_capital"] * 100

    # Copy the updated accounting fields back to the full trade table.
    trades = trades.copy()
    for column in ["shares", "allocated_capital", "costs", "gross_profit_loss", "profit_loss", "return_pct"]:
        if column not in trades.columns:
            trades[column] = np.nan
    for i, row in settled.iterrows():
        original_index = int(row["index"])
        for column in ["shares", "allocated_capital", "costs", "gross_profit_loss", "profit_loss", "return_pct"]:
            trades.loc[original_index, column] = row[column]

    daily = pd.DataFrame(daily_rows)
    daily["cumulative_return_pct"] = (daily["portfolio_value"] / STARTING_CAPITAL - 1) * 100
    daily["peak_value"] = daily["portfolio_value"].cummax()
    daily["drawdown_pct"] = (daily["portfolio_value"] / daily["peak_value"] - 1) * 100
    daily["win_rate_pct"] = daily["winners"] / daily["trades"] * 100
    return trades, daily


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
        learned = len(prior) >= 100 and prior["profitable_close"].nunique() >= 2
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
        trades = trades.drop_duplicates(["target_date", "symbol"], keep="first").sort_values(["target_date", "rank", "symbol"]).reset_index(drop=True)
    else:
        trades = old

    if trades.empty:
        return trades

    # Rebuild portfolio from all settled trades so the result is deterministic
    # even when this job is rerun. Equal capital is allocated across BUY signals
    # for each session using the portfolio value entering that session.
    trades, daily = _build_portfolio(trades)
    trades.to_csv(TRADES_FILE, index=False)
    daily.to_csv(PORTFOLIO_FILE, index=False)

    settled = trades[trades["signal"] == "BUY"].copy()
    final = daily.iloc[-1]
    metrics = pd.DataFrame([{
        "as_of": final["target_date"],
        "sessions": len(daily),
        "trades": int(len(settled)),
        "win_rate_pct": float((settled["profit_loss"] > 0).mean() * 100),
        "total_return_pct": float(final["cumulative_return_pct"]),
        "max_drawdown_pct": float(daily["drawdown_pct"].min()),
        "avg_trade_return_pct": float(settled["return_pct"].mean()),
        "profit_factor": float(
            settled.loc[settled["profit_loss"] > 0, "profit_loss"].sum()
            / max(abs(settled.loc[settled["profit_loss"] < 0, "profit_loss"].sum()), 1e-9)
        ),
        "starting_capital": STARTING_CAPITAL,
        "ending_capital": float(final["portfolio_value"]),
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT * 100,
        "learning_rows": int(len(rows)),
        "learned_model_active": bool(len(rows) >= 100 and rows["profitable_close"].nunique() >= 2),
    }])
    metrics.to_csv(STRATEGY_FILE, index=False)
    print(
        f"Paper trading complete: {len(settled)} settled trades, {len(daily)} sessions, "
        f"net return={metrics.iloc[0]['total_return_pct']:.2f}%"
    )
    return trades


if __name__ == "__main__":
    run_paper_trading()
