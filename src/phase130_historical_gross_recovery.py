from __future__ import annotations

"""Phase 130: source-backed recovery of legacy paper-trade gross fields.

This report validates alternate fields already recorded in paper_trades.csv.
It never edits that ledger and never estimates transaction costs, net P&L,
or exit reasons.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOLERANCE = 0.05


def evaluate_historical_gross_recovery(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "signal", "symbol", "target_date", "entry_price", "exit_price",
        "profit_loss", "shares", "allocated_capital", "gross_profit_loss",
        "costs", "quantity", "position_value", "planned_capital",
        "reference_price", "pre_split_quantity", "split_ratio",
    ]
    missing_columns = [c for c in required if c not in trades.columns]
    if missing_columns:
        return pd.DataFrame(), {
            "status": "BLOCKED_MISSING_SOURCE_COLUMNS",
            "missing_columns": missing_columns,
            "source_ledger_mutated": False,
            "missing_facts_inferred": False,
        }

    buys = trades.loc[
        trades["signal"].astype(str).str.upper().str.strip().eq("BUY")
    ].copy()
    numeric_cols = [
        "entry_price", "exit_price", "profit_loss", "shares",
        "allocated_capital", "gross_profit_loss", "costs", "quantity",
        "position_value", "planned_capital", "reference_price",
        "pre_split_quantity", "split_ratio",
    ]
    n = buys[numeric_cols].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(np.isfinite(n.to_numpy(dtype=float)).all(axis=1), index=buys.index)
    canonical_complete = n[["shares", "allocated_capital", "gross_profit_loss", "costs"]].notna().all(axis=1)
    source_complete = n[[
        "entry_price", "exit_price", "profit_loss", "quantity",
        "position_value", "planned_capital", "reference_price",
        "pre_split_quantity", "split_ratio",
    ]].notna().all(axis=1)
    positive = (
        n["entry_price"].gt(0) & n["exit_price"].gt(0) & n["quantity"].gt(0)
        & n["position_value"].gt(0) & n["planned_capital"].gt(0)
        & n["reference_price"].gt(0) & n["pre_split_quantity"].gt(0)
        & n["split_ratio"].gt(0)
    )
    shares_delta = n["pre_split_quantity"] * n["split_ratio"] - n["quantity"]
    capital_delta = n["quantity"] * n["entry_price"] - n["position_value"]
    planned_delta = n["pre_split_quantity"] * n["reference_price"] - n["planned_capital"]
    gross_delta = n["quantity"] * (n["exit_price"] - n["entry_price"]) - n["profit_loss"]
    valid = (
        finite & source_complete & positive
        & shares_delta.abs().le(TOLERANCE)
        & capital_delta.abs().le(TOLERANCE)
        & planned_delta.abs().le(TOLERANCE)
        & gross_delta.abs().le(TOLERANCE)
    )
    recoverable = ~canonical_complete & valid
    reason = pd.Series("SOURCE_IDENTITY_MISMATCH", index=buys.index, dtype="string")
    reason.loc[~finite | ~source_complete] = "MISSING_OR_NONFINITE_SOURCE"
    reason.loc[canonical_complete] = "CANONICAL_FIELDS_ALREADY_PRESENT"
    reason.loc[recoverable] = "GROSS_FIELDS_SOURCE_VALIDATED"
    detail = pd.DataFrame({
        "source_row_index": buys.index,
        "symbol": buys["symbol"].astype(str),
        "target_date": buys["target_date"].astype(str),
        "recovery_status": reason,
        "source_quantity": n["quantity"],
        "validated_shares": n["quantity"].where(valid),
        "validated_allocated_capital": n["position_value"].where(valid),
        "validated_gross_profit_loss": (
            n["quantity"] * (n["exit_price"] - n["entry_price"])
        ).where(valid),
        "costs_recovered": False,
        "net_profit_loss_recovered": False,
        "exit_reason_recovered": False,
        "shares_delta": shares_delta,
        "capital_delta": capital_delta,
        "planned_capital_delta": planned_delta,
        "gross_pnl_delta": gross_delta,
    })
    recoverable_count = int(recoverable.sum())
    summary = {
        "status": "GROSS_SOURCE_FIELDS_RECOVERABLE_WITH_NET_ACCOUNTING_BLOCKED" if recoverable_count else "BLOCKED_NO_VALID_SOURCE_ROWS",
        "source": "data/paper_trades.csv",
        "buy_rows": int(len(buys)),
        "canonical_complete_rows": int(canonical_complete.sum()),
        "gross_source_recoverable_rows": recoverable_count,
        "gross_source_unresolved_rows": int(len(buys) - canonical_complete.sum() - recoverable_count),
        "validated_source_identities": {
            "shares": "pre_split_quantity * split_ratio = quantity",
            "allocated_capital": "quantity * entry_price = position_value",
            "planned_capital": "pre_split_quantity * reference_price = planned_capital",
            "gross_profit_loss": "quantity * (exit_price - entry_price) = recorded profit_loss",
        },
        "costs_recovered_rows": 0,
        "net_profit_loss_recovered_rows": 0,
        "exit_reason_recovered_rows": 0,
        "canonical_ledger_backfilled": False,
        "source_ledger_mutated": False,
        "missing_facts_inferred": False,
        "promotion_authorized": False,
        "interpretation": "Validated shares, actual entry notional, and gross P&L are emitted as separate source-backed evidence only. Costs, net P&L, and exit reasons remain unresolved; the canonical ledger and promotion gate remain unchanged.",
    }
    return detail, summary


def run() -> dict[str, Any]:
    source = DATA / "paper_trades.csv"
    trades = pd.read_csv(source, low_memory=False) if source.exists() else pd.DataFrame()
    detail, summary = evaluate_historical_gross_recovery(trades)
    detail.to_csv(DATA / "phase130_historical_gross_recovery_detail.csv", index=False)
    (DATA / "phase130_historical_gross_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
