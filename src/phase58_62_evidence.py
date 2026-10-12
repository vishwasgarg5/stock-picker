from __future__ import annotations

"""Phases 58-62: evidence integrity and operational diagnostics.

Analysis-only. Does not alter model scores, production ranking, risk limits, or
promotion status. Unknown accounting values remain unknown.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def accounting_diagnostics() -> tuple[dict[str, Any], pd.DataFrame]:
    """Explain missing ledger fields; never infer or fill costs/net P&L."""
    frame = _read("paper_trades.csv")
    if frame.empty:
        return {"status": "MISSING_LEDGER", "rows_examined": 0, "incomplete_rows": None,
                "missing_cost_rows": None, "missing_net_pnl_rows": None}, pd.DataFrame()
    signal = frame.get("signal", pd.Series("BUY", index=frame.index)).astype(str).str.upper().str.strip()
    quantity = _numeric(frame, "quantity").where(_numeric(frame, "quantity").gt(0), _numeric(frame, "shares"))
    executed = ~signal.isin(["SKIP", "HOLD", "NONE", ""]) & quantity.fillna(0).gt(0)
    x = frame.loc[executed].copy()
    quantity = quantity.loc[executed]
    columns = {
        "quantity": quantity,
        "entry_price": _numeric(x, "entry_price"),
        "exit_price": _numeric(x, "exit_price"),
        "gross_profit_loss": _numeric(x, "gross_profit_loss"),
        "costs": _numeric(x, "costs"),
        "profit_loss": _numeric(x, "profit_loss"),
    }
    detail = pd.DataFrame({
        "row_index": x.index,
        "symbol": x.get("symbol", pd.Series("", index=x.index)).astype(str).values,
        "target_date": x.get("target_date", pd.Series("", index=x.index)).astype(str).values,
    })
    for name, values in columns.items():
        detail[f"missing_{name}"] = values.isna().to_numpy()
    detail["missing_fields"] = [
        ",".join(name for name, values in columns.items() if pd.isna(values.loc[idx]))
        for idx in x.index
    ]
    detail["entry_exit_prices_available"] = (
        columns["entry_price"].gt(0) & columns["exit_price"].gt(0) & quantity.gt(0)
    ).to_numpy()
    detail["gross_reconstruction_possible"] = (
        columns["gross_profit_loss"].isna() & detail["entry_exit_prices_available"]
    )
    detail["accounting_status"] = np.where(detail["missing_fields"].eq(""), "FIELDS_PRESENT",
                                             "INCOMPLETE_REQUIRES_SOURCE_TRACE")
    missing = {name: int(values.isna().sum()) for name, values in columns.items()}
    summary = {
        "status": "PASS" if detail.empty or detail["missing_fields"].eq("").all() else "INCOMPLETE",
        "rows_examined": int(len(frame)),
        "executed_rows_examined": int(len(x)),
        "incomplete_rows": int(detail["missing_fields"].ne("").sum()),
        "missing_field_counts": missing,
        "missing_cost_rows": missing["costs"],
        "missing_net_pnl_rows": missing["profit_loss"],
        "gross_reconstruction_possible_rows": int(detail["gross_reconstruction_possible"].sum()),
        "values_imputed": False,
        "net_pnl_claim_safe": bool(detail.empty or detail["missing_fields"].eq("").all()),
    }
    return summary, detail


def forecast_diagnostics() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = _read("phase33_chronological_prediction_scores.csv")
    required = {"target_date", "close_ape", "baseline_close_ape"}
    if frame.empty or not required.issubset(frame.columns):
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0,
                "model_close_mape_pct": None, "baseline_close_mape_pct": None,
                "beats_baseline": False}, pd.DataFrame(), pd.DataFrame()
    x = frame.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce")
    for col in ["close_ape", "baseline_close_ape", "close_direction_correct"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    if x.empty:
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0,
                "model_close_mape_pct": None, "baseline_close_mape_pct": None,
                "beats_baseline": False}, pd.DataFrame(), pd.DataFrame()
    x["close_ape_pct"] = x["close_ape"] * 100.0
    x["baseline_close_ape_pct"] = x["baseline_close_ape"] * 100.0
    by_date = x.groupby("target_date", as_index=False).agg(
        scored_rows=("close_ape", "size"),
        model_close_mape_pct=("close_ape_pct", "mean"),
        baseline_close_mape_pct=("baseline_close_ape_pct", "mean"),
    )
    if "close_direction_correct" in x.columns:
        direction = x.groupby("target_date")["close_direction_correct"].mean().mul(100).rename("directional_accuracy_pct")
        by_date = by_date.merge(direction, left_on="target_date", right_index=True, how="left")
    symbol_col = "symbol" if "symbol" in x.columns else None
    if symbol_col:
        by_symbol = x.groupby("symbol", as_index=False).agg(
            scored_rows=("close_ape", "size"),
            model_close_mape_pct=("close_ape_pct", "mean"),
            baseline_close_mape_pct=("baseline_close_ape_pct", "mean"),
        )
        by_symbol["error_delta_pct_points"] = by_symbol["model_close_mape_pct"] - by_symbol["baseline_close_mape_pct"]
        by_symbol = by_symbol.sort_values("error_delta_pct_points", ascending=False)
    else:
        by_symbol = pd.DataFrame()
    model = float(x["close_ape"].mean() * 100)
    baseline = float(x["baseline_close_ape"].mean() * 100)
    summary = {
        "status": "PASS",
        "scored_rows": int(len(x)),
        "model_close_mape_pct": model,
        "baseline_close_mape_pct": baseline,
        "error_delta_pct_points": model - baseline,
        "beats_baseline": bool(model < baseline),
        "chronological_dates": int(x["target_date"].nunique()),
        "symbol_breakdown_available": bool(symbol_col),
        "evaluation_source": "phase33_chronological_prediction_scores.csv",
        "production_model_changed": False,
    }
    return summary, by_date, by_symbol


def v2_diagnostics() -> tuple[dict[str, Any], pd.DataFrame]:
    frame = _read("paper_trades_v2.csv")
    if frame.empty or "signal" not in frame.columns:
        return {"status": "MISSING_OR_INVALID_V2_LEDGER", "rows": 0, "buy_rows": 0,
                "skip_rows": 0, "missing_skip_reason_rows": None}, pd.DataFrame()
    signal = frame["signal"].astype(str).str.upper().str.strip()
    skips = frame.loc[signal.eq("SKIP")].copy()
    reason = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip()
    phase4 = skips.get("phase4_reject_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip()
    skips["primary_reason"] = reason.where(reason.ne(""), "MISSING_REASON")
    skips["phase4_reason"] = phase4.where(phase4.ne(""), "NOT_RECORDED")
    by_reason = skips.groupby(["primary_reason", "phase4_reason"], dropna=False).size().reset_index(name="rows")
    summary = {
        "status": "PASS" if not reason.eq("").any() else "INCOMPLETE_REASONS",
        "rows": int(len(frame)),
        "buy_rows": int(signal.eq("BUY").sum()),
        "skip_rows": int(signal.eq("SKIP").sum()),
        "missing_skip_reason_rows": int(reason.eq("").sum()),
        "primary_reason_counts": {str(k): int(v) for k, v in skips["primary_reason"].value_counts().items()},
        "phase4_reason_counts": {str(k): int(v) for k, v in skips["phase4_reason"].value_counts().items()},
        "production_filters_changed": False,
    }
    return summary, by_reason


def telegram_diagnostics() -> dict[str, Any]:
    files = ["telegram_sent.csv", "telegram_evening_sent.csv", "telegram_paper_sent.csv", "telegram_paper_v2_sent.csv"]
    total = confirmed = legacy = missing_status = 0
    per_file: dict[str, Any] = {}
    for name in files:
        frame = _read(name)
        if frame.empty:
            per_file[name] = {"rows": 0, "confirmed": 0, "legacy_unverified": 0, "missing_status": 0}
            continue
        status = frame.get("delivery_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        counts = {
            "rows": int(len(frame)),
            "confirmed": int(status.eq("TELEGRAM_API_CONFIRMED").sum()),
            "legacy_unverified": int(status.eq("LEGACY_UNVERIFIED").sum()),
            "missing_status": int(status.eq("").sum()),
        }
        per_file[name] = counts
        total += counts["rows"]
        confirmed += counts["confirmed"]
        legacy += counts["legacy_unverified"]
        missing_status += counts["missing_status"]
    return {"status": "PASS" if confirmed > 0 else "NO_CONFIRMED_RECEIPTS",
            "rows_across_logs": total, "api_confirmed_rows": confirmed,
            "legacy_unverified_rows": legacy, "missing_status_rows": missing_status,
            "by_file": per_file, "human_read_receipt_proven": False}


def build_reports() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    ledger_summary, ledger_detail = accounting_diagnostics()
    forecast_summary, by_date, by_symbol = forecast_diagnostics()
    v2_summary, v2_reasons = v2_diagnostics()
    telegram_summary = telegram_diagnostics()
    if not ledger_detail.empty:
        ledger_detail.to_csv(DATA / "phase58_ledger_gap_diagnostics.csv", index=False)
    if not by_date.empty:
        by_date.to_csv(DATA / "phase59_forecast_by_date.csv", index=False)
    if not by_symbol.empty:
        by_symbol.to_csv(DATA / "phase59_forecast_by_symbol.csv", index=False)
    if not v2_reasons.empty:
        v2_reasons.to_csv(DATA / "phase60_v2_rejection_breakdown.csv", index=False)
    existing_gate = _read("phase52_promotion_gate.json")
    gate = {}
    gate_path = DATA / "phase52_promotion_gate.json"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            gate = {}
    report = {
        "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
        "phase58_accounting": ledger_summary,
        "phase59_forecast": forecast_summary,
        "phase60_v2": v2_summary,
        "phase61_telegram": telegram_summary,
        "phase62_promotion": {
            "existing_phase52_gate_readable": bool(gate),
            "existing_promotion_safe": bool(gate.get("promotion_safe", False)),
            "production_champion": "V1",
            "v2_status": "SHADOW_ONLY",
            "automatic_promotion": False,
            "production_settings_changed": False,
            "missing_accounting_blocks_net_pnl_claim": not bool(ledger_summary.get("net_pnl_claim_safe", False)),
        },
    }
    (DATA / "phase58_62_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    build_reports()
