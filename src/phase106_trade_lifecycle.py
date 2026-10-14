from __future__ import annotations

"""Phase 106: paper-trade lifecycle data-quality audit.

Descriptive only. This module measures available and invalid lifecycle fields;
it never fills missing trade facts or changes trading behaviour.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FIELD_ALIASES = {
    "symbol": ("symbol", "ticker"),
    "entry_price": ("entry_price", "buy_price", "actual_entry_price"),
    "exit_price": ("exit_price", "sell_price", "actual_exit_price"),
    "entry_date": ("entry_date", "buy_date", "trade_date", "target_date"),
    "exit_date": ("exit_date", "sell_date", "closed_date"),
    "quantity": ("quantity", "qty", "shares"),
    "pnl": ("profit_loss", "net_pnl", "daily_profit_loss"),
    "exit_reason": ("exit_reason", "exit_type", "reason"),
    "status": ("trade_status", "status", "signal"),
}


def _field(frame: pd.DataFrame, logical_name: str) -> str | None:
    return next((name for name in FIELD_ALIASES[logical_name] if name in frame.columns), None)


def evaluate_trade_lifecycle(trades: pd.DataFrame) -> dict[str, Any]:
    safe = {
        "diagnostic_only": True,
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "risk_or_sizing_changed": False,
        "missing_values_filled": False,
        "automatic_promotion": False,
    }
    if trades.empty:
        return {
            **safe, "status": "INSUFFICIENT_DATA", "rows": 0,
            "field_presence": {}, "field_non_null_counts": {},
            "invalid_price_rows": 0, "invalid_quantity_rows": 0,
            "entry_after_exit_rows": 0, "duplicate_symbol_entry_date_rows": 0,
        }

    x = trades.copy()
    fields = {logical: _field(x, logical) for logical in FIELD_ALIASES}
    presence = {logical: (column is not None) for logical, column in fields.items()}
    non_null = {
        logical: int(x[column].notna().sum()) if column else 0
        for logical, column in fields.items()
    }
    price_bad = pd.Series(False, index=x.index)
    for logical in ("entry_price", "exit_price"):
        column = fields[logical]
        if column:
            values = pd.to_numeric(x[column], errors="coerce")
            price_bad |= values.notna() & (values <= 0)
    quantity_bad = pd.Series(False, index=x.index)
    quantity_col = fields["quantity"]
    if quantity_col:
        quantities = pd.to_numeric(x[quantity_col], errors="coerce")
        quantity_bad = quantities.notna() & (quantities <= 0)

    entry_after_exit = pd.Series(False, index=x.index)
    entry_col, exit_col = fields["entry_date"], fields["exit_date"]
    if entry_col and exit_col:
        entry_dates = pd.to_datetime(x[entry_col], errors="coerce", utc=True)
        exit_dates = pd.to_datetime(x[exit_col], errors="coerce", utc=True)
        entry_after_exit = entry_dates.notna() & exit_dates.notna() & (entry_dates > exit_dates)

    duplicate_rows = 0
    symbol_col = fields["symbol"]
    if symbol_col and entry_col:
        keys = pd.DataFrame({
            "symbol": x[symbol_col].astype("string").str.strip().str.upper(),
            "entry_date": pd.to_datetime(x[entry_col], errors="coerce", utc=True),
        })
        valid_keys = keys["symbol"].notna() & keys["symbol"].ne("") & keys["entry_date"].notna()
        duplicate_rows = int(keys.loc[valid_keys].duplicated(["symbol", "entry_date"], keep=False).sum())

    status_col = fields["status"]
    status_counts = {}
    if status_col:
        status_counts = (
            x[status_col].fillna("MISSING").astype(str).str.strip()
            .replace("", "MISSING").value_counts(dropna=False).to_dict()
        )
    completed_mask = pd.Series(False, index=x.index)
    if status_col:
        completed_mask = x[status_col].astype(str).str.upper().str.strip().isin(
            ["CLOSED", "COMPLETED", "EXITED", "SELL", "SOLD"]
        )
    closed_rows = int(completed_mask.sum())
    missing_exit_price_on_closed = int(
        (completed_mask & x[fields["exit_price"]].isna()).sum()
    ) if fields["exit_price"] else closed_rows

    required_core = ("symbol", "entry_price", "entry_date", "pnl")
    missing_core_fields = [name for name in required_core if not presence[name]]
    rows = int(len(x))
    summary = {
        **safe,
        "status": "SCORED" if rows >= 20 else "SMALL_SAMPLE",
        "rows": rows,
        "field_presence": presence,
        "field_non_null_counts": non_null,
        "field_completeness_pct": {
            logical: round(count / rows * 100, 4) for logical, count in non_null.items()
        },
        "missing_core_fields": missing_core_fields,
        "invalid_price_rows": int(price_bad.sum()),
        "invalid_quantity_rows": int(quantity_bad.sum()),
        "entry_after_exit_rows": int(entry_after_exit.sum()),
        "duplicate_symbol_entry_date_rows": duplicate_rows,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "closed_trade_rows_detected": closed_rows,
        "closed_rows_missing_exit_price": missing_exit_price_on_closed,
        "costs_subtracted_again": False,
        "missing_facts_inferred": False,
        "interpretation": (
            "Use completeness gaps to improve upstream recording. Missing lifecycle values are not "
            "reconstructed; duplicate keys are flagged for review and are not automatically deleted."
        ),
    }
    return summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "paper_trades.csv"
    try:
        trades = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        trades = pd.DataFrame()
    summary = evaluate_trade_lifecycle(trades)
    (DATA / "phase106_trade_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
