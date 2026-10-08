from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .risk_management import MAX_TRADES
from .trade_quality_model import latest_trade_quality_scores
from .phase4_optimizer import apply_phase4

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
CONF_SUMMARY = DATA / "confidence_validation_summary.csv"
PHASE2_FILE = DATA / "phase2_optimized_candidates.csv"
TRADES_FILE = DATA / "paper_trades_v2.csv"
PORTFOLIO_FILE = DATA / "portfolio_v2_daily.csv"

CAPITAL = 100000.0
COST_RATE = 0.0005
SHADOW_CONFIDENCE_PERCENTILE = 0.50
MAX_SHADOW_ATR_PCT = 8.0


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


def _phase2_confidence(p: pd.DataFrame) -> pd.DataFrame:
    if not PHASE2_FILE.exists():
        return p
    try:
        q = pd.read_csv(PHASE2_FILE)
        required = {"prediction_date", "target_date", "symbol"}
        if q.empty or not required.issubset(q.columns):
            return p
        q["prediction_date"] = pd.to_datetime(q["prediction_date"], errors="coerce").dt.normalize()
        q["target_date"] = pd.to_datetime(q["target_date"], errors="coerce").dt.normalize()
        q["symbol"] = q["symbol"].astype(str).str.upper().str.strip()
        keep = [c for c in [
            "prediction_date", "target_date", "symbol", "phase2_rank",
            "phase2_selected", "phase2_score", "confidence_v3", "direction_score_v3"
        ] if c in q.columns]
        q = q[keep].drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
        return p.merge(q, on=["prediction_date", "target_date", "symbol"], how="left")
    except Exception:
        return p


def run_paper_trading_v2() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        return pd.DataFrame()

    p = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    h = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if p.empty or h.empty or "confidence_score" not in p.columns:
        return pd.DataFrame()

    for c in ["target_date", "prediction_date"]:
        p[c] = pd.to_datetime(p[c], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()

    for col in ["rank", "score", "confidence_score", "base_close", "predicted_high", "predicted_low", "predicted_close"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    p = _phase2_confidence(p)
    if "confidence_v3" not in p.columns:
        p["confidence_v3"] = p["confidence_score"]
    p["confidence_v3"] = pd.to_numeric(p["confidence_v3"], errors="coerce").fillna(
        pd.to_numeric(p["confidence_score"], errors="coerce")
    )
    p["phase2_score"] = pd.to_numeric(p.get("phase2_score"), errors="coerce")
    p["phase2_selected"] = pd.to_numeric(p.get("phase2_selected"), errors="coerce").fillna(0)

    actual = h.rename(columns={
        "date": "target_date",
        "open": "actual_open",
        "close": "actual_close"
    })[["symbol", "target_date", "actual_open", "actual_close"]].copy()
    actual["symbol"] = actual["symbol"].astype(str).str.upper().str.strip()

    atr = _atr14(h)
    latest_atr = atr.rename(columns={"date": "prediction_date"})
    p = p.dropna(subset=["target_date", "prediction_date", "symbol"])
    x = p.merge(actual, on=["symbol", "target_date"], how="inner")
    x = x.merge(latest_atr, on=["symbol", "prediction_date"], how="left")
    x = x.dropna(subset=["confidence_score", "actual_open", "actual_close", "base_close"])
    x = x[x["target_date"] <= h["date"].max()].copy()
    if x.empty:
        return pd.DataFrame()

    confidence_promoted = _confidence_gate()

    # Evidence collection is independent of production promotion. V2 benchmarks
    # the Phase-2 selector itself; production remains protected by the governor.
    x["confidence_pct"] = x.groupby("target_date")["confidence_v3"].rank(pct=True, method="first")
    eligible = x[
        (x["confidence_pct"] >= SHADOW_CONFIDENCE_PERCENTILE)
        & x["phase2_selected"].eq(1)
    ].copy()

    eligible["expected_return_pct"] = (
        eligible["predicted_close"] / eligible["base_close"].replace(0, np.nan) - 1.0
    ) * 100.0
    eligible["atr_pct"] = (
        eligible["atr14"] / eligible["base_close"].replace(0, np.nan) * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    eligible["stop_distance_pct"] = np.maximum(1.5 * eligible["atr_pct"].fillna(0.0), 1.0)
    eligible["risk_reward"] = eligible["expected_return_pct"] / eligible["stop_distance_pct"].replace(0, np.nan)

    quality = latest_trade_quality_scores(eligible)
    if not quality.empty:
        eligible = eligible.merge(quality, on=["symbol", "target_date"], how="left")
    if "trade_quality_probability" not in eligible.columns:
        eligible["trade_quality_probability"] = 0.50
    eligible["trade_quality_probability"] = pd.to_numeric(
        eligible["trade_quality_probability"], errors="coerce"
    ).fillna(0.50)

    # Phase 4 is an adaptive overlay learned only from already-realized data.
    # It improves signal thresholds, symbol weighting, regime handling and
    # losing-symbol cooldowns while remaining downstream of Phase 2.
    eligible["confidence_pct"] = eligible.groupby("target_date")["confidence_v3"].rank(
        pct=True, method="first"
    )
    eligible = apply_phase4(eligible)

    # Shadow benchmark is not a production risk gate. Apply only the Phase-4
    # eligibility rules and a hard ATR sanity bound.
    eligible = eligible[
        eligible["phase4_eligible"]
        & (eligible["atr_pct"].isna() | eligible["atr_pct"].le(MAX_SHADOW_ATR_PCT))
    ].copy()
    selected = eligible.sort_values(
        ["target_date", "phase4_score", "phase2_score", "confidence_v3"],
        ascending=[True, False, False, False]
    ).groupby("target_date", group_keys=False).head(MAX_TRADES).copy()

    x["strategy"] = "V2_SHADOW_BENCHMARK" if not confidence_promoted else "V2_CONFIDENCE_BENCHMARK"
    x["signal"] = "SKIP"
    x["no_trade_reason"] = "shadow_selection_filter"
    if not selected.empty:
        selected_idx = pd.MultiIndex.from_frame(selected[["symbol", "target_date"]])
        x_idx = pd.MultiIndex.from_frame(x[["symbol", "target_date"]])
        mask = x_idx.isin(selected_idx)
        x.loc[mask, "signal"] = "BUY"
        x.loc[mask, "no_trade_reason"] = ""

        selected_lookup = selected.set_index(["symbol", "target_date"])
        for col in ["risk_reward", "stop_distance_pct", "expected_return_pct",
                    "trade_quality_probability", "phase4_score", "phase4_weight",
                    "phase4_stop_atr_multiplier", "phase4_target_stop_multiple"]:
            if col in selected.columns:
                old_series = x[col] if col in x.columns else pd.Series(np.nan, index=x.index)
                x[col] = [
                    selected_lookup.loc[(sym, dt), col]
                    if (sym, dt) in selected_lookup.index else old
                    for sym, dt, old in zip(x["symbol"], x["target_date"], old_series)
                ]
    else:
        for col in ["risk_reward", "stop_distance_pct", "expected_return_pct", "trade_quality_probability"]:
            if col not in x:
                x[col] = np.nan

    x["reference_price"] = x["base_close"]
    x["no_trade_reason"] = x["no_trade_reason"].fillna("").astype(str)
    # Phase-4 weight can only reduce the equal-allocation budget; it cannot
    # increase it. This keeps the portfolio-capital invariant intact.
    if "phase4_weight" in x.columns:
        weight = pd.to_numeric(x["phase4_weight"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        weight = pd.Series(1.0, index=x.index, dtype=float)
    x["quantity"] = np.where(
        x["signal"].eq("BUY"),
        np.floor((CAPITAL / MAX_TRADES) * weight / x["reference_price"].replace(0, np.nan)).fillna(0),
        0,
    ).astype(int)
    x["planned_capital"] = np.where(x["signal"].eq("BUY"), x["quantity"] * x["reference_price"], 0.0)
    x["entry_price"] = np.where(x["signal"].eq("BUY"), x["actual_open"], np.nan)
    x["exit_price"] = np.where(x["signal"].eq("BUY"), x["actual_close"], np.nan)
    x["gross_profit_loss"] = np.where(
        x["signal"].eq("BUY"), x["quantity"] * (x["exit_price"] - x["entry_price"]), 0.0
    )
    x["trading_cost"] = np.where(
        x["signal"].eq("BUY"),
        (x["entry_price"] * x["quantity"] + x["exit_price"] * x["quantity"]) * COST_RATE,
        0.0,
    )
    x["profit_loss"] = x["gross_profit_loss"] - x["trading_cost"]
    x["return_pct"] = np.where(
        x["signal"].eq("BUY") & x["entry_price"].ne(0),
        (x["exit_price"] / x["entry_price"] - 1.0) * 100.0,
        np.nan,
    )

    keep = [
        "prediction_date", "target_date", "symbol", "rank", "score", "confidence_score",
        "confidence_v3", "direction_score_v3", "strategy", "signal", "reference_price",
        "quantity", "planned_capital", "expected_return_pct", "risk_reward",
        "stop_distance_pct", "trade_quality_probability", "phase4_score",
        "phase4_weight", "phase4_stop_atr_multiplier", "phase4_target_stop_multiple",
        "no_trade_reason",
        "entry_price", "exit_price", "return_pct", "gross_profit_loss",
        "trading_cost", "profit_loss",
    ]
    x = x[[c for c in keep if c in x.columns]]

    old = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    combined = pd.concat([old, x], ignore_index=True)
    combined["target_date"] = pd.to_datetime(combined["target_date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["target_date", "symbol"]).drop_duplicates(
        ["target_date", "symbol"], keep="last"
    ).sort_values(["target_date", "confidence_v3"], ascending=[True, False])
    combined.to_csv(TRADES_FILE, index=False)

    buy = combined[combined["signal"].astype(str).str.upper().eq("BUY")].copy()
    daily = buy.groupby("target_date", as_index=False).agg(
        trades=("symbol", "count"),
        gross_profit_loss=("gross_profit_loss", "sum"),
        trading_cost=("trading_cost", "sum"),
        daily_profit_loss=("profit_loss", "sum"),
    )
    sessions = combined[["target_date"]].dropna().drop_duplicates().sort_values("target_date")
    daily = sessions.merge(daily, on="target_date", how="left")
    for col in ["trades", "gross_profit_loss", "trading_cost", "daily_profit_loss"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)

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
        f"Paper V2: confidence_promoted={confidence_promoted}, "
        f"shadow_selector={SHADOW_CONFIDENCE_PERCENTILE:.0%}, "
        f"sessions={len(daily)}, BUY trades={int(buy.shape[0])}, "
        f"portfolio=₹{float(daily.iloc[-1]['portfolio_value']):,.2f}"
    )
    return combined


if __name__ == "__main__":
    run_paper_trading_v2()
