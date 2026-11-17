from __future__ import annotations

"""Phase 131: historical execution-cost provenance audit.

Read-only diagnostics. Recorded costs are never reconstructed from gross/net P&L,
estimated from rates, or written back to the canonical ledger.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COST_COLUMNS = ("costs", "trading_cost", "transaction_costs")
GROSS_COLUMNS = ("gross_profit_loss", "gross_pnl")
NET_COLUMNS = ("profit_loss", "net_pnl")
COMPONENT_COLUMNS = (
    "brokerage", "stt", "gst", "exchange_charges",
    "sebi_charges", "stamp_duty", "other_charges",
)


def _number(row: pd.Series, name: str) -> float | None:
    if name not in row.index or pd.isna(row[name]) or str(row[name]).strip() == "":
        return None
    try:
        value = float(row[name])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _first_recorded(row: pd.Series, names: tuple[str, ...]) -> tuple[float | None, str | None]:
    present = [(name, _number(row, name)) for name in names]
    present = [(name, value) for name, value in present if value is not None]
    if not present:
        return None, None
    return present[0][1], present[0][0]


def evaluate_execution_cost_provenance(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify executed rows using recorded cost fields and independent identities."""
    if trades.empty:
        detail = pd.DataFrame(columns=[
            "row_index", "symbol", "target_date", "recorded_costs",
            "cost_field", "gross_pnl", "net_pnl", "cost_identity_delta",
            "component_breakdown_status", "cost_evidence_reference", "status",
        ])
        return detail, {
            "status": "NO_EXECUTED_TRADES", "executed_rows": 0,
            "ledger_cost_rows": 0, "component_verified_rows": 0,
            "source_referenced_rows": 0, "blocked_rows": 0,
            "mismatch_rows": 0, "costs_inferred": False,
            "canonical_ledger_mutated": False,
        }

    signal = trades.get("signal", pd.Series("BUY", index=trades.index)).astype(str).str.upper().str.strip()
    qty_name = "quantity" if "quantity" in trades.columns else "shares"
    qty = pd.to_numeric(trades.get(qty_name, pd.Series(np.nan, index=trades.index)), errors="coerce")
    executed = signal.isin(["BUY", "LONG", "EXECUTED"]) & qty.gt(0)
    rows = trades.loc[executed]
    output = []
    for idx, row in rows.iterrows():
        cost, cost_field = _first_recorded(row, COST_COLUMNS)
        gross, gross_field = _first_recorded(row, GROSS_COLUMNS)
        net, net_field = _first_recorded(row, NET_COLUMNS)
        all_cost_values = [(name, _number(row, name)) for name in COST_COLUMNS]
        all_cost_values = [(name, value) for name, value in all_cost_values if value is not None]
        conflicting_cost_fields = bool(all_cost_values and any(
            abs(value - all_cost_values[0][1]) > 0.02 for _, value in all_cost_values[1:]
        ))
        delta = gross - cost - net if gross is not None and cost is not None and net is not None else None

        supplied_components = [(name, _number(row, name)) for name in COMPONENT_COLUMNS]
        supplied_components = [(name, value) for name, value in supplied_components if value is not None]
        component_names_present = [name for name in COMPONENT_COLUMNS if name in row.index]
        components_complete = bool(component_names_present) and all(
            name in dict(supplied_components) for name in component_names_present
        )
        component_total = sum(value for _, value in supplied_components) if components_complete else None
        component_delta = component_total - cost if component_total is not None and cost is not None else None
        component_verified = component_delta is not None and abs(component_delta) <= 0.02
        evidence_ref = next((
            str(row[name]).strip() for name in ("cost_evidence_reference", "cost_evidence_url", "broker_contract_note_id")
            if name in row.index and pd.notna(row[name]) and str(row[name]).strip()
        ), "")
        if conflicting_cost_fields or (delta is not None and abs(delta) > 0.02):
            status = "FAIL_RECONCILIATION_MISMATCH"
        elif cost is None or gross is None or net is None:
            status = "BLOCKED_MISSING_RECORDED_ACCOUNTING"
        elif cost < 0:
            status = "FAIL_INVALID_NEGATIVE_COST"
        elif component_verified and evidence_ref:
            status = "SOURCE_REFERENCED_COMPONENTS_RECONCILED"
        elif component_verified:
            status = "COMPONENTS_RECONCILED_REFERENCE_MISSING"
        else:
            status = "LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED"
        output.append({
            "row_index": idx,
            "symbol": str(row.get("symbol", "")),
            "target_date": str(row.get("target_date", "")),
            "recorded_costs": cost,
            "cost_field": cost_field,
            "gross_pnl": gross,
            "gross_field": gross_field,
            "net_pnl": net,
            "net_field": net_field,
            "cost_identity_delta": delta,
            "component_total": component_total,
            "component_identity_delta": component_delta,
            "component_breakdown_status": "VERIFIED" if component_verified else (
                "INCOMPLETE" if not components_complete else "MISMATCH"
            ),
            "cost_evidence_reference": evidence_ref,
            "status": status,
        })
    detail = pd.DataFrame(output)
    statuses = detail.get("status", pd.Series(dtype=str))
    counts = statuses.value_counts().to_dict()
    source_verified = int(statuses.eq("SOURCE_REFERENCED_COMPONENTS_RECONCILED").sum())
    component_verified = int(statuses.isin([
        "SOURCE_REFERENCED_COMPONENTS_RECONCILED",
        "COMPONENTS_RECONCILED_REFERENCE_MISSING",
    ]).sum())
    mismatch = int(statuses.str.startswith("FAIL_").sum())
    blocked = int(statuses.eq("BLOCKED_MISSING_RECORDED_ACCOUNTING").sum())
    ledger_only = int(statuses.eq("LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED").sum())
    summary = {
        "status": "PASS" if len(detail) and source_verified == len(detail) and mismatch == 0 and blocked == 0 else (
            "FAIL" if mismatch else "BLOCKED" if blocked else "REVIEW"
        ),
        "executed_rows": int(len(detail)),
        "ledger_cost_rows": int(detail["recorded_costs"].notna().sum()),
        "component_verified_rows": component_verified,
        "source_referenced_rows": source_verified,
        "ledger_arithmetic_only_rows": ledger_only,
        "blocked_rows": blocked,
        "mismatch_rows": mismatch,
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "costs_inferred": False,
        "costs_estimated": False,
        "costs_zero_filled": False,
        "canonical_ledger_mutated": False,
        "policy": "A recorded total that satisfies gross minus costs equals net is arithmetic reconciliation only. Source-backed verification additionally requires a complete recorded component breakdown and a source reference. No rate-based reconstruction or backfill is performed.",
    }
    return detail, summary


def build_report() -> dict[str, Any]:
    path = DATA / "paper_trades.csv"
    try:
        trades = pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        trades = pd.DataFrame()
    detail, summary = evaluate_execution_cost_provenance(trades)
    DATA.mkdir(exist_ok=True)
    detail.to_csv(DATA / "phase131_execution_cost_provenance_detail.csv", index=False)
    report = {**summary, "source": "data/paper_trades.csv"}
    (DATA / "phase131_execution_cost_provenance_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    build_report()
