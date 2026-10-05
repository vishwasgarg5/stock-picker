from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
CONF_SUMMARY = DATA / "confidence_validation_summary.csv"
TRADES_FILE = DATA / "paper_trades_v2.csv"
PORTFOLIO_FILE = DATA / "portfolio_v2_daily.csv"

CAPITAL = 100000.0
MAX_TRADES = 5
MIN_RISK_REWARD = 1.25
MAX_POSITION_RISK_PCT = 0.75
MAX_PORTFOLIO_RISK_PCT = 3.0
COST_RATE = 0.0005


def _confidence_gate() -> bool:
    if not CONF_SUMMARY.exists():
        return False
    try:
        x = pd.read_csv(CONF_SUMMARY)
        if x.empty:
            return False
        return str(x.iloc[-1].get("confidence_promotion_evidence", "False")).strip().lower() == "true"
    except Exception:
        return False


def _atr14(hist: pd.DataFrame) -> pd.DataFrame:
    x = hist.sort_values(["symbol", "date"]).copy()
    prev = x.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr14"] = tr.groupby(x["symbol"]).transform(lambda s: s.rolling(14).mean())
    return x[["symbol", "date", "atr14"]]


def run_paper_trading_v2() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        return pd.DataFrame()

    p = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    h = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if p.empty or h.empty or "confidence_score" not in p.columns:
        return pd.DataFrame()

    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    for col in ["rank", "confidence_score", "base_close", "predicted_high", "predicted_low", "predicted_close"]:
        p[col] = pd.to_numeric(p.get(col), errors="coerce")

    actual = h.rename(columns={"date": "target_date", "open": "actual_open", "close": "actual_close"})[
        ["symbol", "target_date", "actual_open", "actual_close"]
    ].copy()
    actual["symbol"] = actual["symbol"].astype(str).str.upper().str.strip()

    atr = _atr14(h)
    latest_atr = atr.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
    latest_atr = latest_atr.rename(columns={"date": "prediction_date", "atr14": "atr14"})
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    x = p.merge(actual, on=["symbol", "target_date"], how="inner")
    x = x.merge(latest_atr, on=["symbol", "prediction_date"], how="left")
    x = x.dropna(subset=["target_date", "confidence_score", "actual_open", "actual_close", "base_close"])
    x = x[x["target_date"] <= h["date"].max()].copy()
    if x.empty:
        return pd.DataFrame()

    evidence = _confidence_gate()
    x["strategy"] = "V2_WAITING_FOR_EVIDENCE"
    x["signal"] = "SKIP"

    if evidence:
        x["confidence_pct"] = x.groupby("target_date")["confidence_score"].rank(pct=True, method="first")
        eligible = x[x["confidence_pct"] >= 0.8].copy()

        # Conservative entry/target/stop estimates use only information that
        # existed before the target session. Actual OHLC is used solely for
        # retrospective paper-trading evaluation.
        x["expected_return_pct"] = (x["predicted_close"] / x["base_close"] - 1.0) * 100.0
        x["atr_pct"] = (x["atr14"] / x["base_close"] * 100.0).replace([np.inf, -np.inf], np.nan)
        x["stop_distance_pct"] = np.maximum(1.5 * x["atr_pct"], 1.0)
        x["target_return_pct"] = x["expected_return_pct"].clip(lower=0.0)
        x["risk_reward"] = x["target_return_pct"] / x["stop_distance_pct"].replace(0, np.nan)

        eligible = eligible.merge(
            x[["symbol", "target_date", "expected_return_pct", "atr_pct", "stop_distance_pct", "target_return_pct", "risk_reward"]],
            on=["symbol", "target_date"], how="left", suffixes=("", "_risk")
        )
        eligible = eligible[
            eligible["expected_return_pct"].gt(0) &
            eligible["risk_reward"].ge(MIN_RISK_REWARD) &
            eligible["atr_pct"].notna()
        ].copy()

        # Never allocate more than 0.75% of portfolio capital to initial stop risk.
        eligible["risk_budget"] = CAPITAL * MAX_POSITION_RISK_PCT / 100.0
        eligible["risk_per_share"] = eligible["actual_open"] * eligible["stop_distance_pct"] / 100.0
        eligible["risk_quantity"] = np.floor(
            eligible["risk_budget"] / eligible["risk_per_share"].replace(0, np.nan)
        ).fillna(0).astype(int)

        # Capital constraint remains roughly equal-weighted; risk sizing can only reduce size.
        allocation = CAPITAL / MAX_TRADES
        eligible["capital_quantity"] = np.floor(allocation / eligible["base_close"].replace(0, np.nan)).fillna(0).astype(int)
        eligible["quantity_candidate"] = np.minimum(eligible["risk_quantity"], eligible["capital_quantity"])
        eligible = eligible[eligible["quantity_candidate"] > 0].copy()

        selected = eligible.sort_values(
            ["target_date", "risk_reward", "confidence_score"],
            ascending=[True, False, False]
        ).groupby("target_date", group_keys=False).head(MAX_TRADES).copy()

        # Cap aggregate initial stop risk at 3% of portfolio capital.
        selected["risk_value"] = selected["quantity_candidate"] * selected["risk_per_share"]
        selected["risk_value_cum"] = selected.groupby("target_date")["risk_value"].cumsum()
        selected = selected[selected["risk_value_cum"] <= CAPITAL * MAX_PORTFOLIO_RISK_PCT / 100.0].copy()

        x.loc[:, "signal"] = "SKIP"
        x.loc[selected.index, "signal"] = "BUY"
        x.loc[selected.index, "strategy"] = "V2_CONFIDENCE_RISK_FILTERED"
        x.loc[selected.index, "quantity"] = selected["quantity_candidate"].to_numpy()
        x.loc[selected.index, "risk_reward"] = selected["risk_reward"].to_numpy()
        x.loc[selected.index, "stop_distance_pct"] = selected["stop_distance_pct"].to_numpy()
        x.loc[selected.index, "expected_return_pct"] = selected["expected_return_pct"].to_numpy()
    else:
        x["strategy"] = "V2_WAITING_FOR_EVIDENCE"

    for col in ["quantity", "risk_reward", "stop_distance_pct", "expected_return_pct"]:
        if col not in x:
            x[col] = np.nan

    x["reference_price"] = x["base_close"]
    x["quantity"] = np.where(x["signal"].eq("BUY"), x["quantity"].fillna(0), 0).astype(int)
    x["planned_capital"] = np.where(x["signal"].eq("BUY"), x["quantity"] * x["reference_price"], 0.0)
    x["entry_price"] = np.where(x["signal"].eq("BUY"), x["actual_open"], np.nan)
    x["exit_price"] = np.where(x["signal"].eq("BUY"), x["actual_close"], np.nan)
    x["gross_profit_loss"] = np.where(
        x["signal"].eq("BUY"), x["quantity"] * (x["exit_price"] - x["entry_price"]), 0.0
    )
    x["trading_cost"] = np.where(
        x["signal"].eq("BUY"),
        (x["entry_price"] * x["quantity"] + x["exit_price"] * x["quantity"]) * COST_RATE,
        0.0
    )
    x["profit_loss"] = x["gross_profit_loss"] - x["trading_cost"]
    x["return_pct"] = np.where(
        x["signal"].eq("BUY") & x["entry_price"].ne(0),
        (x["exit_price"] / x["entry_price"] - 1.0) * 100.0, np.nan
    )

    keep = [
        "prediction_date", "target_date", "symbol", "rank", "score", "confidence_score",
        "strategy", "signal", "reference_price", "quantity", "planned_capital",
        "expected_return_pct", "risk_reward", "stop_distance_pct",
        "entry_price", "exit_price", "return_pct", "gross_profit_loss",
        "trading_cost", "profit_loss"
    ]
    x = x[keep]

    old = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    combined = pd.concat([old, x], ignore_index=True)
    combined["target_date"] = pd.to_datetime(combined["target_date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["target_date", "symbol"]).drop_duplicates(
        ["target_date", "symbol"], keep="first"
    ).sort_values(["target_date", "confidence_score"], ascending=[True, False])
    combined.to_csv(TRADES_FILE, index=False)

    buy = combined[combined["signal"].eq("BUY")].copy()
    daily = buy.groupby("target_date", as_index=False).agg(
        trades=("symbol", "count"), gross_profit_loss=("gross_profit_loss", "sum"),
        trading_cost=("trading_cost", "sum"), daily_profit_loss=("profit_loss", "sum")
    )
    if daily.empty:
        return combined

    daily["daily_return_pct"] = daily["daily_profit_loss"] / CAPITAL * 100.0
    daily = daily.sort_values("target_date")
    daily["portfolio_value"] = CAPITAL + daily["daily_profit_loss"].cumsum()
    daily["cumulative_return_pct"] = (daily["portfolio_value"] / CAPITAL - 1.0) * 100.0
    daily["peak_value"] = daily["portfolio_value"].cummax()
    daily["drawdown_pct"] = (daily["portfolio_value"] / daily["peak_value"] - 1.0) * 100.0
    daily.to_csv(PORTFOLIO_FILE, index=False)

    print(
        f"Paper V2 risk-filtered: evidence={evidence}, sessions={len(daily)}, "
        f"portfolio=₹{float(daily.iloc[-1]['portfolio_value']):,.2f}"
    )
    return combined


if __name__ == "__main__":
    run_paper_trading_v2()
