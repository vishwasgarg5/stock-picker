from __future__ import annotations

"""Phase 105: exit-reason attribution and loss accounting audit.

Diagnostic only. Uses recorded paper-trade P&L and existing exit metadata; never
infers a missing exit reason or changes production selection/risk behaviour.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REASON_CANDIDATES = ("exit_reason", "exit_type", "reason")
PNL_CANDIDATES = ("profit_loss", "net_pnl", "daily_profit_loss")
METADATA_CANDIDATES = (
    "entry_price", "exit_price", "stop_loss", "stop_loss_price",
    "target_price", "exit_date", "holding_days", "hold_days",
    "exit_signal", "trade_status",
)


def evaluate_exit_attribution(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    safe = {
        "diagnostic_only": True,
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "thresholds_changed": False,
        "automatic_promotion": False,
    }
    if trades.empty:
        return pd.DataFrame(), {**safe, "status": "INSUFFICIENT_DATA", "executed_trade_rows": 0}

    x = trades.copy()
    if "signal" in x.columns:
        x = x[x["signal"].astype(str).str.upper().str.strip().isin(["BUY", "LONG", "EXECUTED"])].copy()
    pnl_col = next((c for c in PNL_CANDIDATES if c in x.columns), None)
    if pnl_col is None or "symbol" not in x.columns:
        return pd.DataFrame(), {
            **safe, "status": "INSUFFICIENT_DATA", "executed_trade_rows": int(len(x)),
            "reason": "Missing symbol or recorded P&L column",
        }

    x["recorded_pnl"] = pd.to_numeric(x[pnl_col], errors="coerce")
    x = x.dropna(subset=["recorded_pnl"]).copy()
    if x.empty:
        return pd.DataFrame(), {**safe, "status": "INSUFFICIENT_DATA", "executed_trade_rows": 0}

    reason_col = next((c for c in REASON_CANDIDATES if c in x.columns), None)
    if reason_col:
        raw = x[reason_col].fillna("").astype(str).str.strip()
        x["exit_reason_group"] = raw.where(raw.ne(""), "UNKNOWN")
        x.loc[x["exit_reason_group"].str.upper().isin(["UNKNOWN", "N/A", "NA", "NONE", "NAN"]), "exit_reason_group"] = "UNKNOWN"
    else:
        x["exit_reason_group"] = "NOT_RECORDED"

    grouped = x.groupby("exit_reason_group", dropna=False).agg(
        trades=("recorded_pnl", "size"),
        net_pnl=("recorded_pnl", "sum"),
        mean_pnl=("recorded_pnl", "mean"),
        losing_trades=("recorded_pnl", lambda s: int((s < 0).sum())),
        winning_trades=("recorded_pnl", lambda s: int((s > 0).sum())),
        gross_loss=("recorded_pnl", lambda s: float(-s[s < 0].sum())),
        gross_profit=("recorded_pnl", lambda s: float(s[s > 0].sum())),
    ).reset_index()
    grouped["loss_rate_pct"] = (grouped["losing_trades"] / grouped["trades"] * 100).round(4)
    total_gross_loss = float(-x.loc[x["recorded_pnl"] < 0, "recorded_pnl"].sum())
    grouped["share_of_gross_loss_pct"] = (
        grouped["gross_loss"] / total_gross_loss * 100 if total_gross_loss > 0 else 0.0
    )
    grouped = grouped.sort_values(["gross_loss", "trades"], ascending=[False, False]).reset_index(drop=True)

    unknown_mask = x["exit_reason_group"].str.upper().isin(["UNKNOWN", "NOT_RECORDED"])
    unknown = x.loc[unknown_mask]
    metadata_presence = {
        col: int(x[col].notna().sum())
        for col in METADATA_CANDIDATES if col in x.columns
    }
    blank_metadata = {
        col: int(x.loc[unknown_mask, col].notna().sum())
        for col in METADATA_CANDIDATES if col in x.columns
    }
    unknown_gross_loss = float(-unknown.loc[unknown["recorded_pnl"] < 0, "recorded_pnl"].sum())
    summary = {
        **safe,
        "status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE",
        "executed_trade_rows": int(len(x)),
        "pnl_source_field": pnl_col,
        "exit_reason_field_used": reason_col,
        "reason_groups": int(grouped["exit_reason_group"].nunique()),
        "gross_loss_magnitude": total_gross_loss,
        "unknown_or_unrecorded_reason_rows": int(unknown_mask.sum()),
        "unknown_or_unrecorded_reason_pct": float(unknown_mask.mean() * 100),
        "unknown_reason_losing_rows": int((unknown["recorded_pnl"] < 0).sum()),
        "unknown_reason_net_pnl": float(unknown["recorded_pnl"].sum()),
        "unknown_reason_gross_loss": unknown_gross_loss,
        "unknown_reason_share_of_gross_loss_pct": float(unknown_gross_loss / total_gross_loss * 100) if total_gross_loss else None,
        "metadata_non_null_counts_all_trades": metadata_presence,
        "metadata_non_null_counts_unknown_reason_trades": blank_metadata,
        "costs_subtracted_again": False,
        "missing_exit_reasons_inferred": False,
        "interpretation": "Prioritize fixing exit-reason recording and inspect available metadata. This report does not infer unknown exits, blacklist symbols, or alter trading rules.",
    }
    return grouped, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "paper_trades.csv"
    try:
        trades = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        trades = pd.DataFrame()
    by_reason, summary = evaluate_exit_attribution(trades)
    by_reason.to_csv(DATA / "phase105_exit_attribution_by_reason.csv", index=False)
    (DATA / "phase105_exit_attribution_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
