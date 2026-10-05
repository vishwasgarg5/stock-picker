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
MIN_CONFIDENCE_EVIDENCE = True


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


def run_paper_trading_v2() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        return pd.DataFrame()

    p = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    h = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if p.empty or h.empty or "confidence_score" not in p.columns:
        return pd.DataFrame()

    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    p["rank"] = pd.to_numeric(p["rank"], errors="coerce")
    p["confidence_score"] = pd.to_numeric(p["confidence_score"], errors="coerce")
    p["base_close"] = pd.to_numeric(p["base_close"], errors="coerce")

    actual = h.rename(columns={
        "date": "target_date", "open": "actual_open", "close": "actual_close"
    })[["symbol", "target_date", "actual_open", "actual_close"]]
    actual["symbol"] = actual["symbol"].astype(str).str.upper().str.strip()

    x = p.merge(actual, on=["symbol", "target_date"], how="inner")
    x = x.dropna(subset=["target_date", "confidence_score", "actual_open", "actual_close", "base_close"])
    x = x[x["target_date"] <= h["date"].max()].copy()
    if x.empty:
        return pd.DataFrame()

    # V2 only activates after the independent confidence-validation gate.
    evidence = _confidence_gate()
    x["strategy"] = "V2_CONFIDENCE_FILTERED" if evidence else "V2_WAITING_FOR_EVIDENCE"
    x["signal"] = "SKIP"
    if evidence:
        # Within each session, trade the highest-confidence names, capped at five.
        x["confidence_pct"] = x.groupby("target_date")["confidence_score"].rank(pct=True, method="first")
        eligible = x[x["confidence_pct"] >= 0.8].copy()
        # In BEAR regimes, require above-median model rank as an additional
        # protection. This changes only V2; V1 remains the benchmark.
        if "market_regime" in eligible.columns:
            bear = eligible["market_regime"].astype(str).str.upper().eq("BEAR")
            if bear.any():
                eligible.loc[bear, "rank_cut"] = eligible.loc[bear].groupby("target_date")["rank"].transform("median")
                eligible = eligible[~bear | (eligible["rank"] <= eligible["rank_cut"])]
        selected = eligible.sort_values(
            ["target_date", "confidence_score"], ascending=[True, False]
        ).groupby("target_date", group_keys=False).head(MAX_TRADES).copy()
        x.loc[selected.index, "signal"] = "BUY"

    allocation = CAPITAL / MAX_TRADES
    x["reference_price"] = x["base_close"]
    x["quantity"] = np.where(
        x["signal"].eq("BUY") & x["reference_price"].gt(0),
        np.floor(allocation / x["reference_price"]).astype(int), 0
    )
    x["planned_capital"] = np.where(x["signal"].eq("BUY"), x["quantity"] * x["reference_price"], 0.0)
    x["entry_price"] = np.where(x["signal"].eq("BUY"), x["actual_open"], np.nan)
    x["exit_price"] = np.where(x["signal"].eq("BUY"), x["actual_close"], np.nan)
    x["profit_loss"] = np.where(
        x["signal"].eq("BUY"),
        x["quantity"] * (x["exit_price"] - x["entry_price"]), 0.0
    )
    x["return_pct"] = np.where(
        x["signal"].eq("BUY") & x["entry_price"].ne(0),
        (x["exit_price"] / x["entry_price"] - 1.0) * 100.0, np.nan
    )

    keep = [
        "prediction_date", "target_date", "symbol", "rank", "score",
        "confidence_score", "strategy", "signal", "reference_price",
        "quantity", "planned_capital", "entry_price", "exit_price",
        "return_pct", "profit_loss"
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
        trades=("symbol", "count"), daily_profit_loss=("profit_loss", "sum")
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
        f"Paper V2 updated: evidence={evidence}, sessions={len(daily)}, "
        f"portfolio=₹{float(daily.iloc[-1]['portfolio_value']):,.2f}"
    )
    return combined


if __name__ == "__main__":
    run_paper_trading_v2()
