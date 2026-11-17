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
        "costs", "position_value", "return_pct",
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
        "allocated_capital", "gross_profit_loss", "costs", "position_value",
    ]
    for optional in ("quantity", "planned_capital", "reference_price",
                     "pre_split_quantity", "split_ratio"):
        if optional in buys.columns:
            numeric_cols.append(optional)
    n = buys[numeric_cols].apply(pd.to_numeric, errors="coerce")
    canonical_complete = n[["shares", "allocated_capital", "gross_profit_loss", "costs"]].notna().all(axis=1)

    # Support both ledger generations without treating field presence as proof:
    # newer rows record shares/allocated_capital and gross_profit_loss, while
    # older rows record quantity/position_value and keep gross P&L in profit_loss.
    entry = n["entry_price"]
    exit_price = n["exit_price"]
    allocated = n["allocated_capital"]
    position_value = n["position_value"]
    capital = allocated.where(allocated.gt(0), position_value)
    stored_net_or_legacy_pnl = n["profit_loss"]
    stored_gross = n["gross_profit_loss"].where(
        n["gross_profit_loss"].notna(), stored_net_or_legacy_pnl
    )
    observed_quantity = n["quantity"] if "quantity" in n else pd.Series(np.nan, index=buys.index)
    observed_shares = n["shares"] if "shares" in n else pd.Series(np.nan, index=buys.index)
    implied_from_capital = capital / entry.replace(0, np.nan)
    # Prefer an explicitly recorded quantity, then recorded shares, and only
    # then derive shares from recorded capital / entry price.
    shares = observed_quantity.where(
        observed_quantity.gt(0),
        observed_shares.where(observed_shares.gt(0), implied_from_capital),
    )
    source_values = pd.DataFrame({
        "entry": entry, "exit": exit_price, "capital": capital,
        "stored_gross": stored_gross, "stored_pnl": stored_net_or_legacy_pnl,
        "shares": shares,
    }, index=buys.index)
    finite = pd.Series(np.isfinite(source_values.to_numpy(dtype=float)).all(axis=1), index=buys.index)
    positive = entry.gt(0) & exit_price.gt(0) & capital.gt(0) & shares.gt(0)
    share_identity = (shares - implied_from_capital).abs().le(TOLERANCE)
    # Do not divide by the price move to infer shares: a valid flat-price
    # trade has a zero move but still has a verifiable share count.
    gross_calculated = shares * (exit_price - entry)
    gross_identity = (gross_calculated - stored_gross).abs().le(TOLERANCE)
    # return_pct has historically been calculated from profit_loss, which is
    # net P&L in the newer schema and gross P&L in the older schema.
    return_calculated = stored_net_or_legacy_pnl / capital * 100.0
    if "return_pct" in buys.columns:
        recorded_return = pd.to_numeric(buys["return_pct"], errors="coerce")
    else:
        recorded_return = pd.Series(np.nan, index=buys.index)
    return_identity = (return_calculated - recorded_return).abs().le(TOLERANCE)

    optional_identity = pd.Series(True, index=buys.index)
    if {"pre_split_quantity", "split_ratio", "quantity"}.issubset(n.columns):
        pre = n["pre_split_quantity"]
        ratio = n["split_ratio"]
        supplied = pre.notna() | ratio.notna()
        paired_valid = pre.gt(0) & ratio.gt(0) & n["quantity"].gt(0)
        split_ok = (pre * ratio - n["quantity"]).abs().le(TOLERANCE)
        optional_identity &= ~supplied | (paired_valid & split_ok)
    if {"planned_capital", "reference_price", "quantity"}.issubset(n.columns):
        supplied = n["planned_capital"].notna() | n["reference_price"].notna()
        planned_ok = (n["quantity"] * n["reference_price"] - n["planned_capital"]).abs().le(TOLERANCE)
        optional_identity &= ~supplied | (n["planned_capital"].gt(0) & n["reference_price"].gt(0) & planned_ok)

    valid = (
        finite & positive & share_identity
        & gross_identity & return_identity & optional_identity
    )
    canonical_validated = canonical_complete & valid
    recoverable = ~canonical_complete & valid
    reason = pd.Series("SOURCE_IDENTITY_MISMATCH", index=buys.index, dtype="string")
    reason.loc[~finite] = "MISSING_OR_NONFINITE_SOURCE"
    reason.loc[finite & ~positive] = "NON_POSITIVE_SOURCE_VALUE"
    reason.loc[recoverable] = "GROSS_FIELDS_SOURCE_VALIDATED"
    reason.loc[canonical_validated] = "CANONICAL_FIELDS_SOURCE_VALIDATED"
    reason.loc[canonical_complete & ~valid] = "CANONICAL_FIELDS_PRESENT_IDENTITY_MISMATCH"
    detail = pd.DataFrame({
        "source_row_index": buys.index,
        "symbol": buys["symbol"].astype(str),
        "target_date": buys["target_date"].astype(str),
        "recovery_status": reason,
        "source_quantity": observed_quantity,
        "source_shares": observed_shares,
        "source_allocated_capital": allocated,
        "source_position_value": position_value,
        "gross_reference_field": pd.Series(
            np.where(n["gross_profit_loss"].notna(), "gross_profit_loss", "profit_loss"),
            index=buys.index,
        ),
        "validated_shares": shares.where(valid),
        "validated_allocated_capital": capital.where(valid),
        "validated_gross_profit_loss": gross_calculated.where(valid),
        "costs_recovered": False,
        "net_profit_loss_recovered": False,
        "exit_reason_recovered": False,
        "share_identity_delta": (shares - implied_from_capital),
        "capital_implied_share_delta": (shares - implied_from_capital),
        "gross_pnl_delta": (gross_calculated - stored_gross),
        "return_delta_pct_points": (return_calculated - pd.to_numeric(buys.get("return_pct"), errors="coerce")),
    })
    recoverable_count = int(recoverable.sum())
    summary = {
        "status": "GROSS_SOURCE_FIELDS_RECOVERABLE_WITH_NET_ACCOUNTING_BLOCKED" if recoverable_count else "BLOCKED_NO_VALID_SOURCE_ROWS",
        "source": "data/paper_trades.csv",
        "buy_rows": int(len(buys)),
        "canonical_complete_rows": int(canonical_complete.sum()),
        "canonical_source_validated_rows": int(canonical_validated.sum()),
        "gross_source_recoverable_rows": recoverable_count,
        "gross_source_unresolved_rows": int(len(buys) - canonical_validated.sum() - recoverable_count),
        "validated_source_identities": {
            "shares": "recorded quantity or shares, cross-checked against allocated_capital or position_value divided by entry_price",
            "allocated_capital": "recorded allocated_capital, falling back to recorded position_value for legacy rows",
            "gross_profit_loss": "validated_shares * (exit_price - entry_price) = recorded gross_profit_loss, falling back to legacy profit_loss",
            "return": "recorded profit_loss / validated capital * 100 = recorded return_pct",
            "optional_quantity": "recorded quantity must agree with the independently implied shares when present",
            "optional_split": "pre_split_quantity * split_ratio = quantity when split metadata is supplied",
            "optional_planned_capital": "quantity * reference_price = planned_capital when both are supplied",
        },
        "costs_recovered_rows": 0,
        "net_profit_loss_recovered_rows": 0,
        "exit_reason_recovered_rows": 0,
        "canonical_ledger_backfilled": False,
        "source_ledger_mutated": False,
        "missing_facts_inferred": False,
        "promotion_authorized": False,
        "interpretation": "Both canonical and legacy rows are independently checked against recorded share, capital, gross-P&L, and return identities. Costs, net P&L reconstruction, and exit reasons remain unresolved; the canonical ledger and promotion gate remain unchanged.",
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
