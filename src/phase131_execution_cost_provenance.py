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


def configured_cost_model() -> dict[str, Any]:
    """Read the current simulation assumption; it is not proof of historical broker fees."""
    source = ROOT / "src" / "paper_trading.py"
    if not source.is_file():
        return {"rate": None, "source": "src/paper_trading.py", "status": "SOURCE_UNAVAILABLE"}
    text = source.read_text(encoding="utf-8")
    match = re.search(r"^ROUND_TRIP_COST_PCT[ ]*=[ ]*([0-9]+([.][0-9]+)?)", text, re.MULTILINE)
    if not match:
        return {"rate": None, "source": "src/paper_trading.py", "status": "RATE_NOT_FOUND"}
    return {
        "rate": float(match.group(1)),
        "source": "src/paper_trading.py:ROUND_TRIP_COST_PCT",
        "status": "FIXED_SIMULATION_ASSUMPTION_NOT_BROKER_EVIDENCE",
        "actual_broker_cost_evidence": False,
    }
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
            "row_index", "symbol", "target_date", "recorded_costs", "allocated_capital", "recorded_cost_rate",
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
    quantity = pd.to_numeric(trades.get("quantity", pd.Series(np.nan, index=trades.index)), errors="coerce")
    shares = pd.to_numeric(trades.get("shares", pd.Series(np.nan, index=trades.index)), errors="coerce")
    qty = quantity.where(quantity.gt(0), shares)
    executed = signal.isin(["BUY", "LONG", "EXECUTED"]) & qty.gt(0)
    rows = trades.loc[executed]
    cost_model = configured_cost_model()
    model_rate = cost_model.get("rate")
    output = []
    for idx, row in rows.iterrows():
        cost, cost_field = _first_recorded(row, COST_COLUMNS)
        allocated = _number(row, "allocated_capital")
        recorded_rate = cost / allocated if cost is not None and allocated is not None and allocated > 0 else None
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
        components_complete = all(
            name in row.index and _number(row, name) is not None
            for name in COMPONENT_COLUMNS
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
            "matches_current_simulation_rate": bool(recorded_rate is not None and model_rate is not None and abs(recorded_rate - model_rate) <= 1e-9),
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
    detail = pd.DataFrame(output, columns=[
        "row_index", "symbol", "target_date", "recorded_costs", "allocated_capital",
        "recorded_cost_rate", "matches_current_simulation_rate", "cost_field",
        "gross_pnl", "gross_field", "net_pnl", "net_field", "cost_identity_delta",
        "component_total", "component_identity_delta", "component_breakdown_status",
        "cost_evidence_reference", "status",
    ])
    statuses = detail.get("status", pd.Series(dtype=str))
    counts = statuses.value_counts().to_dict()
    reference_present = int(statuses.eq("REFERENCE_PRESENT_COMPONENTS_RECONCILED").sum())
    component_verified = int(statuses.isin([
        "REFERENCE_PRESENT_COMPONENTS_RECONCILED",
        "COMPONENTS_RECONCILED_REFERENCE_MISSING",
    ]).sum())
    mismatch = int(statuses.str.startswith("FAIL_").sum())
    blocked = int(statuses.eq("BLOCKED_MISSING_RECORDED_ACCOUNTING").sum())
    ledger_only = int(statuses.eq("LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED").sum())
    summary = {
        "status": "NO_EXECUTED_TRADES" if detail.empty else "FAIL" if mismatch else "BLOCKED" if blocked else "REVIEW",
        "executed_rows": int(len(detail)),
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
        "policy": "A recorded total that satisfies gross minus costs equals net is arithmetic reconciliation only. A component breakdown and reference string are reported as supplied, but the referenced document is not fetched or authenticated by this audit. The current fixed round-trip simulation rate is reported separately and is never applied to historical rows. No rate-based reconstruction or backfill is performed.",
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
        json.dumps(report, indent=2, allow_nan=False) + "\\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    build_report()
            "FAIL" if mismatch else "BLOCKED" if blocked else "REVIEW"
        ),
        "executed_rows": int(len(detail)),
        "configured_cost_model": cost_model,
        "rows_matching_current_simulation_rate": int(detail["matches_current_simulation_rate"].sum()),
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
        "policy": "A recorded total that satisfies gross minus costs equals net is arithmetic reconciliation only. Source-backed verification additionally requires a complete recorded component breakdown and a source reference. The current fixed round-trip simulation rate is reported separately and is never applied to historical rows. No rate-based reconstruction or backfill is performed.",
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
