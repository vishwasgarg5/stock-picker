from __future__ import annotations

"""Phase 131: historical execution-cost provenance audit.

Read-only diagnostics. Recorded costs are never reconstructed from gross/net P&L,
estimated from rates, or written back to the canonical ledger.
"""
import json
import re
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


def configured_cost_model() -> dict[str, Any]:
    """Read both simulation assumptions; neither is proof of historical broker fees."""
    specifications = (
        ("src/pipeline.py", "PAPER_ROUND_TRIP_COST_PCT"),
        ("src/paper_trading.py", "ROUND_TRIP_COST_PCT"),
    )
    models = []
    for relative_path, constant in specifications:
        source = ROOT / relative_path
        if not source.is_file():
            models.append({"source": relative_path, "constant": constant, "rate": None, "status": "SOURCE_UNAVAILABLE"})
            continue
        source_text = source.read_text(encoding="utf-8")
        match = re.search(r"^" + re.escape(constant) + r"[ ]*=[ ]*([0-9]+([.][0-9]+)?)", source_text, re.MULTILINE)
        models.append({
            "source": relative_path,
            "constant": constant,
            "rate": float(match.group(1)) if match else None,
            "status": "FIXED_SIMULATION_ASSUMPTION_NOT_BROKER_EVIDENCE" if match else "RATE_NOT_FOUND",
        })
    primary = next((model for model in models if model["rate"] is not None), {})
    return {
        "rate": primary.get("rate"),
        "source": primary.get("source"),
        "models": models,
        "status": "SIMULATION_ASSUMPTIONS_ONLY",
        "actual_broker_cost_evidence": False,
    }
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
    """Audit every BUY-like candidate; never silently drop rows missing quantity."""
    cost_model = configured_cost_model()
    columns = [
        "row_index", "symbol", "target_date", "recorded_costs", "allocated_capital",
        "recorded_cost_rate", "matches_current_simulation_rate", "cost_field",
        "gross_pnl", "gross_field", "net_pnl", "net_field", "cost_identity_delta",
        "component_total", "component_identity_delta", "component_breakdown_status",
        "cost_evidence_reference", "execution_quantity", "execution_quantity_status", "status",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns), {
            "status": "NO_EXECUTED_TRADES", "candidate_rows": 0, "executed_rows": 0,
            "candidate_rows_missing_quantity": 0, "quantity_blocked_rows": 0,
            "accounting_blocked_rows": 0,
            "configured_cost_model": cost_model, "rows_matching_current_simulation_rate": 0,
            "ledger_cost_rows": 0, "component_verified_rows": 0,
            "source_reference_present_rows": 0, "source_documents_verified": False,
            "ledger_arithmetic_only_rows": 0, "blocked_rows": 0, "mismatch_rows": 0,
            "costs_inferred": False, "costs_estimated": False,
            "costs_zero_filled": False, "canonical_ledger_mutated": False,
        }

    signal = trades.get("signal", pd.Series("BUY", index=trades.index)).astype(str).str.upper().str.strip()
    candidate = signal.isin(["BUY", "LONG", "EXECUTED"])
    # Keep all BUY-like candidates in the audit. Positive recorded quantity/shares
    # establishes an execution row; missing quantity is itself a blocker, not a
    # reason to silently omit the candidate from accounting coverage.
    rows = trades.loc[candidate]
    model_rates = [model["rate"] for model in cost_model.get("models", []) if model.get("rate") is not None]
    output = []

    for idx, row in rows.iterrows():
        cost, cost_field = _first_recorded(row, COST_COLUMNS)
        gross, gross_field = _first_recorded(row, GROSS_COLUMNS)
        net, net_field = _first_recorded(row, NET_COLUMNS)
        allocated = _number(row, "allocated_capital")
        quantity_value = _number(row, "quantity")
        shares_value = _number(row, "shares")
        execution_quantity = quantity_value if quantity_value is not None and quantity_value > 0 else shares_value
        quantity_verified = execution_quantity is not None and execution_quantity > 0
        recorded_rate = cost / allocated if cost is not None and allocated is not None and allocated > 0 else None

        all_cost_values = [(name, _number(row, name)) for name in COST_COLUMNS]
        all_cost_values = [(name, value) for name, value in all_cost_values if value is not None]
        conflicting_cost_fields = bool(all_cost_values and any(
            abs(value - all_cost_values[0][1]) > 0.02 for _, value in all_cost_values[1:]
        ))
        delta = gross - cost - net if gross is not None and cost is not None and net is not None else None

        components_complete = all(
            name in row.index and _number(row, name) is not None
            for name in COMPONENT_COLUMNS
        )
        component_total = sum(_number(row, name) for name in COMPONENT_COLUMNS) if components_complete else None
        component_delta = component_total - cost if component_total is not None and cost is not None else None
        component_verified = component_delta is not None and abs(component_delta) <= 0.02
        evidence_ref = next((
            str(row[name]).strip()
            for name in ("cost_evidence_reference", "cost_evidence_url", "broker_contract_note_id")
            if name in row.index and pd.notna(row[name]) and str(row[name]).strip()
        ), "")

        if conflicting_cost_fields or (delta is not None and abs(delta) > 0.02):
            status = "FAIL_RECONCILIATION_MISMATCH"
        elif cost is None or gross is None or net is None:
            status = "BLOCKED_MISSING_RECORDED_ACCOUNTING"
        elif cost < 0:
            status = "FAIL_INVALID_NEGATIVE_COST"
        elif not quantity_verified:
            status = "BLOCKED_MISSING_EXECUTION_QUANTITY"
        elif component_verified and evidence_ref:
            status = "REFERENCE_PRESENT_COMPONENTS_RECONCILED"
        elif component_verified:
            status = "COMPONENTS_RECONCILED_REFERENCE_MISSING"
        else:
            status = "LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED"

        output.append({
            "row_index": idx,
            "symbol": str(row.get("symbol", "")),
            "target_date": str(row.get("target_date", "")),
            "recorded_costs": cost,
            "allocated_capital": allocated,
            "recorded_cost_rate": recorded_rate,
            "matches_current_simulation_rate": bool(
                recorded_rate is not None and any(abs(recorded_rate - rate) <= 1e-9 for rate in model_rates)
            ),
            "cost_field": cost_field,
            "gross_pnl": gross,
            "gross_field": gross_field,
            "net_pnl": net,
            "net_field": net_field,
            "cost_identity_delta": delta,
            "component_total": component_total,
            "component_identity_delta": component_delta,
            "component_breakdown_status": "RECONCILED" if component_verified else (
                "INCOMPLETE" if not components_complete else "MISMATCH"
            ),
            "cost_evidence_reference": evidence_ref,
            "execution_quantity": execution_quantity if quantity_verified else None,
            "execution_quantity_status": "RECORDED_POSITIVE_QUANTITY" if quantity_verified else "MISSING_OR_INVALID_RECORDED_QUANTITY",
            "status": status,
        })

    detail = pd.DataFrame(output, columns=columns)
    statuses = detail["status"]
    counts = statuses.value_counts().to_dict()
    reference_present = int(statuses.eq("REFERENCE_PRESENT_COMPONENTS_RECONCILED").sum())
    component_verified = int(statuses.isin([
        "REFERENCE_PRESENT_COMPONENTS_RECONCILED",
        "COMPONENTS_RECONCILED_REFERENCE_MISSING",
    ]).sum())
    mismatch = int(statuses.str.startswith("FAIL_").sum())
    accounting_blocked = int(statuses.eq("BLOCKED_MISSING_RECORDED_ACCOUNTING").sum())
    quantity_blocked = int(detail["execution_quantity_status"].ne("RECORDED_POSITIVE_QUANTITY").sum())
    blocked = int(statuses.isin(["BLOCKED_MISSING_RECORDED_ACCOUNTING", "BLOCKED_MISSING_EXECUTION_QUANTITY"]).sum())
    ledger_only = int(statuses.eq("LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED").sum())
    summary = {
        "status": "NO_EXECUTED_TRADES" if detail.empty else (
            "FAIL" if mismatch else "BLOCKED" if blocked else "REVIEW"
        ),
        "candidate_rows": int(len(detail)),
        "executed_rows": int(detail["execution_quantity_status"].eq("RECORDED_POSITIVE_QUANTITY").sum()),
        "candidate_rows_missing_quantity": quantity_blocked,
        "quantity_blocked_rows": quantity_blocked,
        "accounting_blocked_rows": accounting_blocked,
        "configured_cost_model": cost_model,
        "rows_matching_current_simulation_rate": int(detail["matches_current_simulation_rate"].sum()),
        "ledger_cost_rows": int(detail["recorded_costs"].notna().sum()),
        "component_verified_rows": component_verified,
        "source_reference_present_rows": reference_present,
        "source_documents_verified": False,
        "ledger_arithmetic_only_rows": ledger_only,
        "blocked_rows": blocked,
        "mismatch_rows": mismatch,
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "costs_inferred": False,
        "costs_estimated": False,
        "costs_zero_filled": False,
        "canonical_ledger_mutated": False,
        "policy": (
            "A recorded total that satisfies gross minus costs equals net is arithmetic reconciliation only. "
            "A component breakdown and reference string are reported as supplied, but the referenced document "
            "is not fetched or authenticated by this audit. BUY-like candidates without positive recorded "
            "quantity/shares remain visible as blocked rows rather than being excluded. The current fixed "
            "round-trip simulation rate is reported separately and is never applied to historical rows. "
            "No rate-based reconstruction or backfill is performed."
        ),
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
