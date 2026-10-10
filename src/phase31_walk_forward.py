from __future__ import annotations

"""Phase 31: chronological comparison of recorded V1/V2 portfolio outcomes.

This is an analysis-only replay of already recorded daily portfolio outcomes.
It does not refit models, make predictions, change rankings/risk, or promote V2.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAILY_REPORT = DATA / "phase31_chronological_comparison.csv"
SUMMARY_REPORT = DATA / "phase31_summary.json"
INITIAL_CAPITAL = 100_000.0
MIN_MATCHED_SESSIONS = 20
MIN_V2_EXECUTED_TRADES = 50


def _prepare_daily(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty or "target_date" not in frame.columns:
        return pd.DataFrame(columns=["target_date", f"{prefix}_net_pnl"])
    x = frame.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    pnl_col = next((c for c in ("daily_profit_loss", "net_pnl", "gross_pnl") if c in x.columns), None)
    if pnl_col is None:
        return pd.DataFrame(columns=["target_date", f"{prefix}_net_pnl"])
    x[f"{prefix}_net_pnl"] = pd.to_numeric(x[pnl_col], errors="coerce")
    x = x.dropna(subset=["target_date", f"{prefix}_net_pnl"])
    return x.groupby("target_date", as_index=False)[f"{prefix}_net_pnl"].sum()


def _executed_trade_count(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    x = trades.copy()
    if "signal" in x.columns:
        x = x[~x["signal"].astype(str).str.strip().str.upper().isin(["SKIP", "HOLD", "NONE", ""])]
    if "quantity" in x.columns:
        quantity = pd.to_numeric(x["quantity"], errors="coerce")
        x = x[quantity.fillna(0) > 0]
    elif "shares" in x.columns:
        quantity = pd.to_numeric(x["shares"], errors="coerce")
        x = x[quantity.fillna(0) > 0]
    return int(len(x))


def chronological_comparison(
    v1_daily: pd.DataFrame,
    v2_daily: pd.DataFrame,
    v1_trades: pd.DataFrame | None = None,
    v2_trades: pd.DataFrame | None = None,
    min_matched_sessions: int = MIN_MATCHED_SESSIONS,
    min_v2_executed_trades: int = MIN_V2_EXECUTED_TRADES,
) -> tuple[pd.DataFrame, dict]:
    """Compare only matched dates in chronological order and enforce an evidence gate."""
    v1_trades = v1_trades if v1_trades is not None else pd.DataFrame()
    v2_trades = v2_trades if v2_trades is not None else pd.DataFrame()
    v1 = _prepare_daily(v1_daily, "v1")
    v2 = _prepare_daily(v2_daily, "v2")
    matched = v1.merge(v2, on="target_date", how="inner").sort_values("target_date").reset_index(drop=True)

    executed_v1 = _executed_trade_count(v1_trades)
    executed_v2 = _executed_trade_count(v2_trades)
    if matched.empty:
        summary = {
            "matched_sessions": 0,
            "v1_net_pnl": None,
            "v2_net_pnl": None,
            "v2_minus_v1_pnl": None,
            "v1_executed_trades": executed_v1,
            "v2_executed_trades": executed_v2,
            "holdout_sessions": 0,
            "holdout_start_date": None,
            "holdout_v1_net_pnl": None,
            "holdout_v2_net_pnl": None,
            "evidence_gate": "INSUFFICIENT_EVIDENCE",
            "production_champion": "V1",
            "v2_promoted": False,
            "interpretation": "No matched daily portfolio outcomes; no performance comparison is possible.",
        }
        return pd.DataFrame(columns=[
            "target_date", "v1_net_pnl", "v2_net_pnl", "v2_minus_v1_pnl",
            "v1_cumulative_pnl", "v2_cumulative_pnl", "difference_cumulative_pnl", "period"
        ]), summary

    matched["v2_minus_v1_pnl"] = matched["v2_net_pnl"] - matched["v1_net_pnl"]
    matched["v1_cumulative_pnl"] = matched["v1_net_pnl"].cumsum()
    matched["v2_cumulative_pnl"] = matched["v2_net_pnl"].cumsum()
    matched["difference_cumulative_pnl"] = matched["v2_minus_v1_pnl"].cumsum()

    # Strict chronological holdout: earliest 60% is context; latest 40% is holdout.
    holdout_start_idx = min(len(matched) - 1, max(0, int(np.floor(len(matched) * 0.60))))
    matched["period"] = np.where(
        np.arange(len(matched)) < holdout_start_idx, "EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT"
    )
    holdout = matched.iloc[holdout_start_idx:]
    enough = len(matched) >= min_matched_sessions and executed_v2 >= min_v2_executed_trades
    summary = {
        "matched_sessions": int(len(matched)),
        "first_matched_date": matched["target_date"].min().date().isoformat(),
        "last_matched_date": matched["target_date"].max().date().isoformat(),
        "v1_net_pnl": float(matched["v1_net_pnl"].sum()),
        "v2_net_pnl": float(matched["v2_net_pnl"].sum()),
        "v2_minus_v1_pnl": float(matched["v2_minus_v1_pnl"].sum()),
        "v1_positive_pnl_sessions_pct": float((matched["v1_net_pnl"] > 0).mean() * 100),
        "v2_positive_pnl_sessions_pct": float((matched["v2_net_pnl"] > 0).mean() * 100),
        "v1_executed_trades": executed_v1,
        "v2_executed_trades": executed_v2,
        "holdout_sessions": int(len(holdout)),
        "holdout_start_date": holdout["target_date"].min().date().isoformat(),
        "holdout_v1_net_pnl": float(holdout["v1_net_pnl"].sum()),
        "holdout_v2_net_pnl": float(holdout["v2_net_pnl"].sum()),
        "holdout_v2_minus_v1_pnl": float(holdout["v2_minus_v1_pnl"].sum()),
        "minimum_matched_sessions_required": int(min_matched_sessions),
        "minimum_v2_executed_trades_required": int(min_v2_executed_trades),
        "evidence_gate": "EVIDENCE_SUFFICIENT_FOR_REVIEW" if enough else "COLLECTING_MATCHED_SESSIONS",
        "production_champion": "V1",
        "v2_promoted": False,
        "interpretation": (
            "Recorded-outcome replay only; not a model-refit walk-forward test. "
            "Review evidence only when both sample gates pass; this report never promotes a model."
        ),
    }
    return matched, summary


def run() -> dict:
    DATA.mkdir(exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        path = DATA / name
        try:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()
        except (OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()

    daily, summary = chronological_comparison(
        read("portfolio_daily.csv"),
        read("portfolio_v2_daily.csv"),
        read("paper_trades.csv"),
        read("paper_trades_v2.csv"),
    )
    daily.to_csv(DAILY_REPORT, index=False)
    summary["phase"] = 31
    summary["as_of"] = str(pd.Timestamp.now().date())
    summary["production_model_changed"] = False
    summary["ranking_or_risk_changed"] = False
    SUMMARY_REPORT.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
