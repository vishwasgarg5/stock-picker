from __future__ import annotations

import numpy as np
import pandas as pd

CAPITAL = 100000.0
MAX_POSITION_RISK_PCT = 0.75
MAX_PORTFOLIO_RISK_PCT = 3.0
MIN_EXPECTED_RETURN_PCT = 0.50
MIN_RISK_REWARD = 1.25
MAX_ATR_PCT = 6.0
MIN_TRADE_QUALITY = 0.50
MAX_TRADES = 5


def apply_risk_gate(candidates: pd.DataFrame, capital: float = CAPITAL) -> pd.DataFrame:
    x = candidates.copy()
    for c in ["base_close", "expected_return_pct", "atr_pct", "trade_quality_probability"]:
        if c not in x:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["trade_quality_probability"] = x["trade_quality_probability"].fillna(0.50)
    x["stop_distance_pct"] = np.maximum(1.5 * x["atr_pct"], 1.0)
    x["risk_reward"] = x["expected_return_pct"] / x["stop_distance_pct"].replace(0, np.nan)
    x["risk_per_share"] = x["base_close"] * x["stop_distance_pct"] / 100.0
    x["risk_quantity"] = np.floor(
        (capital * MAX_POSITION_RISK_PCT / 100.0) /
        x["risk_per_share"].replace(0, np.nan)
    ).fillna(0).astype(int)
    x["capital_quantity"] = np.floor(
        (capital / MAX_TRADES) / x["base_close"].replace(0, np.nan)
    ).fillna(0).astype(int)
    x["quantity"] = np.minimum(x["risk_quantity"], x["capital_quantity"])
    x["risk_value"] = x["quantity"] * x["risk_per_share"]

    x["trade_decision"] = "NO_TRADE"
    eligible = (
        x["expected_return_pct"].ge(MIN_EXPECTED_RETURN_PCT)
        & x["risk_reward"].ge(MIN_RISK_REWARD)
        & x["atr_pct"].notna()
        & x["atr_pct"].le(MAX_ATR_PCT)
        & x["trade_quality_probability"].ge(MIN_TRADE_QUALITY)
        & x["quantity"].gt(0)
    )
    selected = x[eligible].sort_values(
        ["risk_reward", "trade_quality_probability"], ascending=False
    ).head(MAX_TRADES).copy()
    if not selected.empty:
        selected["risk_cum"] = selected["risk_value"].cumsum()
        selected = selected[
            selected["risk_cum"] <= capital * MAX_PORTFOLIO_RISK_PCT / 100.0
        ]
        x.loc[selected.index, "trade_decision"] = "BUY"

    x["no_trade_reason"] = ""
    x.loc[x["expected_return_pct"].fillna(0) < MIN_EXPECTED_RETURN_PCT, "no_trade_reason"] = "expected_return_below_min"
    x.loc[x["atr_pct"].isna(), "no_trade_reason"] = "atr_unavailable"
    x.loc[x["atr_pct"].gt(MAX_ATR_PCT), "no_trade_reason"] = "volatility_too_high"
    x.loc[x["trade_quality_probability"].lt(MIN_TRADE_QUALITY), "no_trade_reason"] = "trade_quality_below_min"
    x.loc[(x["risk_reward"].fillna(0) < MIN_RISK_REWARD) & x["no_trade_reason"].eq(""), "no_trade_reason"] = "risk_reward_below_min"
    x.loc[x["trade_decision"].eq("BUY"), "no_trade_reason"] = ""
    return x


if __name__ == "__main__":
    print("Risk management module loaded: explicit NO_TRADE gate, capped position risk, portfolio risk and volatility limits.")
