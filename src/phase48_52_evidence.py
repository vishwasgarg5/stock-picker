from __future__ import annotations

"""Phases 48-52: accounting, forecast, funnel, stress and promotion evidence.

Analysis-only. Never fills missing ledger values, changes production filters, or
promotes V2. V1 remains champion unless a separate human-approved process acts.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_MATCHED_SESSIONS = 20
MIN_V2_EXECUTED_TRADES = 50


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _ledger(frame: pd.DataFrame, model: str, gross_col: str, cost_col: str, net_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if frame.empty:
        empty = pd.DataFrame(columns=["model", "row_index", "symbol", "target_date", "gross_pnl", "recorded_costs", "net_pnl", "reconciliation_delta", "status"])
        return {"model": model, "rows": 0, "executed_trades": 0, "incomplete_rows": 0, "ledger_status": "MISSING_LEDGER", "gross_pnl": None, "net_pnl": None, "recorded_costs": None}, empty
    x = frame.copy().reset_index(drop=True)
    signal = x.get("signal", pd.Series("BUY", index=x.index)).astype(str).str.upper().str.strip()
    # Resolve quantity row-by-row: newer rows use quantity, older rows use shares.
    qty = _num(x, "quantity").where(_num(x, "quantity").gt(0), _num(x, "shares"))
    if "quantity" not in x.columns and "shares" not in x.columns:
        qty = pd.Series(np.nan, index=x.index, dtype=float)
    executed = ~signal.isin(["SKIP", "HOLD", "NONE", ""])
    executed &= qty.fillna(0).gt(0)
    x = x.loc[executed].copy()
    qty = qty.loc[executed]
    gross, cost, net = _num(x, gross_col), _num(x, cost_col), _num(x, net_col)
    gross_source = pd.Series("RECORDED", index=x.index, dtype=object)
    # A missing gross value can be reconstructed from recorded entry/exit prices and quantity.
    price_gross = (_num(x, "exit_price") - _num(x, "entry_price")) * qty
    reconstructable = gross.isna() & price_gross.notna() & _num(x, "entry_price").gt(0) & _num(x, "exit_price").gt(0)
    gross = gross.where(~reconstructable, price_gross)
    gross_source.loc[reconstructable] = "DERIVED_FROM_RECORDED_ENTRY_EXIT_AND_QUANTITY"
    missing = gross.isna() | cost.isna() | net.isna()
    delta = gross - cost - net
    status = np.where(missing, "MISSING_ACCOUNTING_FIELD",
                      np.where(delta.abs() > 0.02, "RECONCILIATION_MISMATCH",
                               np.where(gross_source.eq("RECORDED"), "RECONCILED", "RECONCILED_WITH_DERIVED_GROSS")))
    detail = pd.DataFrame({
        "model": model, "row_index": x.index, "symbol": x.get("symbol", pd.Series("", index=x.index)).astype(str),
        "target_date": x.get("target_date", pd.Series("", index=x.index)).astype(str),
        "quantity_used": qty, "gross_pnl": gross, "gross_pnl_source": gross_source,
        "recorded_costs": cost, "net_pnl": net,
        "reconciliation_delta": delta, "status": status,
    })
    incomplete = int(missing.sum())
    mismatch = int((~missing & delta.abs().gt(0.02)).sum())
    complete = not x.empty and incomplete == 0 and mismatch == 0
    summary = {
        "model": model, "rows": int(len(frame)), "executed_trades": int(len(x)),
        "incomplete_rows": incomplete, "reconciliation_mismatch_rows": mismatch,
        "gross_rows_reconstructed_from_prices": int(reconstructable.sum()),
        "ledger_status": "RECONCILED" if complete else ("NO_EXECUTED_TRADES" if x.empty else "INCOMPLETE_OR_MISMATCH"),
        "gross_pnl": float(gross.sum()) if gross.notna().all() and len(gross) else (0.0 if x.empty else None),
        "net_pnl": float(net.sum()) if net.notna().all() and len(net) else (0.0 if x.empty else None),
        "recorded_costs": float(cost.sum()) if cost.notna().all() and len(cost) else (0.0 if x.empty else None),
        "missing_values_imputed": False,
        "derived_gross_values_are_flagged": True,
    }
    return summary, detail


def build_reports() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    v1, v1_detail = _ledger(_read("paper_trades.csv"), "V1", "gross_profit_loss", "costs", "profit_loss")
    v2, v2_detail = _ledger(_read("paper_trades_v2.csv"), "V2", "gross_profit_loss", "trading_cost", "profit_loss")
    ledger_detail = pd.concat([v1_detail, v2_detail], ignore_index=True)
    ledger_detail.to_csv(DATA / "phase48_ledger_reconciliation.csv", index=False)

    forecast = _read("phase33_chronological_prediction_scores.csv")
    forecast_summary: dict[str, Any] = {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "forecast_beats_baseline": False}
    forecast_report = pd.DataFrame()
    symbol_report = pd.DataFrame()
    if not forecast.empty and {"target_date", "close_ape", "baseline_close_ape"}.issubset(forecast.columns):
        f = forecast.copy()
        numeric_cols = [c for c in ("open_ape", "high_ape", "low_ape", "close_ape", "baseline_close_ape",
                                     "predicted_close", "base_close", "actual_close", "close_direction_correct")
                        if c in f.columns]
        for col in numeric_cols:
            f[col] = pd.to_numeric(f[col], errors="coerce")
        f["target_date"] = pd.to_datetime(f["target_date"], errors="coerce")
        f = f.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
        if not f.empty:
            grouped = f.groupby("target_date", as_index=False)
            aggregations = {"scored_rows": ("close_ape", "size"),
                            "close_mape_pct": ("close_ape", lambda s: float(s.mean() * 100)),
                            "baseline_close_mape_pct": ("baseline_close_ape", lambda s: float(s.mean() * 100))}
            for col in ("open_ape", "high_ape", "low_ape"):
                if col in f.columns:
                    aggregations[f"{col.removesuffix('_ape')}_mape_pct"] = (col, lambda s: float(s.mean() * 100))
            if "close_direction_correct" in f.columns:
                aggregations["directional_accuracy_pct"] = ("close_direction_correct", lambda s: float(pd.to_numeric(s, errors="coerce").mean() * 100))
            forecast_report = grouped.agg(**aggregations)
            forecast_report["forecast_beats_baseline"] = forecast_report["close_mape_pct"] < forecast_report["baseline_close_mape_pct"]
            if "symbol" in f.columns:
                symbol_aggs = {"scored_rows": ("close_ape", "size"),
                               "close_mape_pct": ("close_ape", lambda s: float(s.mean() * 100)),
                               "baseline_close_mape_pct": ("baseline_close_ape", lambda s: float(s.mean() * 100))}
                for col in ("open_ape", "high_ape", "low_ape"):
                    if col in f.columns:
                        symbol_aggs[f"{col.removesuffix('_ape')}_mape_pct"] = (col, lambda s: float(s.mean() * 100))
                symbol_report = f.groupby("symbol", as_index=False).agg(**symbol_aggs).sort_values("close_mape_pct")
                symbol_report["forecast_beats_baseline"] = symbol_report["close_mape_pct"] < symbol_report["baseline_close_mape_pct"]
            forecast_summary = {
                "status": "SCORED" if len(f) >= 20 else "SMALL_SAMPLE", "scored_rows": int(len(f)),
                "target_dates": int(f["target_date"].nunique()),
                "close_mape_pct": float(f["close_ape"].mean() * 100),
                "baseline_close_mape_pct": float(f["baseline_close_ape"].mean() * 100),
                "forecast_beats_baseline": bool(f["close_ape"].mean() < f["baseline_close_ape"].mean()),
                "ohlc_mape_pct": {col.removesuffix("_ape"): float(f[col].mean() * 100)
                                  for col in ("open_ape", "high_ape", "low_ape", "close_ape") if col in f.columns},
                "directional_accuracy_pct": (float(f["close_direction_correct"].mean() * 100)
                                             if "close_direction_correct" in f.columns else None),
                "method": "saved chronological predictions versus saved base-close baseline; no refit or production model changes",
            }
    forecast_report.to_csv(DATA / "phase49_forecast_accuracy.csv", index=False)
    symbol_report.to_csv(DATA / "phase49_forecast_by_symbol.csv", index=False)

    trades2 = _read("paper_trades_v2.csv")
    reason_rows = []
    if not trades2.empty:
        for col in ("no_trade_reason", "phase4_reject_reason"):
            if col in trades2:
                vals = trades2[col].fillna("").astype(str).str.strip()
                vals = vals[vals.ne("")]
                if len(vals):
                    counts = vals.value_counts()
                    reason_rows.extend({"reason_source": col, "reason": str(k), "rows": int(v)} for k, v in counts.items())
    reason_report = pd.DataFrame(reason_rows, columns=["reason_source", "reason", "rows"])
    reason_report.to_csv(DATA / "phase50_v2_rejection_reasons.csv", index=False)

    stress_rows = []
    for item in (v1, v2):
        net, costs = item.get("net_pnl"), item.get("recorded_costs")
        for extra_cost_pct in (0, 0.5, 1.0):
            # Stress applies only additional assumed costs; recorded net P&L already includes recorded costs.
            stressed = (float(net) - float(costs) * extra_cost_pct) if net is not None and costs is not None else None
            stress_rows.append({"model": item["model"], "additional_cost_multiple": extra_cost_pct,
                                "recorded_net_pnl": net, "recorded_costs": costs,
                                "stressed_net_pnl": stressed,
                                "note": "additional cost stress only; no recorded cost deducted twice"})
    stress_report = pd.DataFrame(stress_rows)
    stress_report.to_csv(DATA / "phase51_return_stress.csv", index=False)

    prior = _read("phase31_chronological_comparison.csv")
    matched_sessions = int(pd.to_datetime(prior.get("target_date", pd.Series(dtype=object)), errors="coerce").nunique()) if not prior.empty else 0
    v2_executed = int(v2["executed_trades"])
    ledgers_complete = v1["ledger_status"] == "RECONCILED" and v2["ledger_status"] == "RECONCILED"
    p_and_l_favorable = v1.get("net_pnl") is not None and v2.get("net_pnl") is not None and v2["net_pnl"] > v1["net_pnl"]
    gate_checks = {
        "minimum_matched_sessions_met": matched_sessions >= MIN_MATCHED_SESSIONS,
        "minimum_v2_executed_trades_met": v2_executed >= MIN_V2_EXECUTED_TRADES,
        "both_ledgers_complete": bool(ledgers_complete),
        "v2_net_pnl_exceeds_v1": bool(p_and_l_favorable),
        "forecast_beats_baseline": bool(forecast_summary.get("forecast_beats_baseline", False)),
    }
    eligible = all(gate_checks.values())
    gate = {
        "phase": 52, "status": "ELIGIBLE_FOR_HUMAN_REVIEW_ONLY" if eligible else "NOT_ELIGIBLE",
        "gate_checks": gate_checks, "matched_sessions": matched_sessions,
        "minimum_matched_sessions": MIN_MATCHED_SESSIONS, "v2_executed_trades": v2_executed,
        "minimum_v2_executed_trades": MIN_V2_EXECUTED_TRADES,
        "v1_ledger_status": v1["ledger_status"], "v2_ledger_status": v2["ledger_status"],
        "production_champion": "V1", "v2_promoted": False, "automatic_promotion": False,
        "production_settings_changed": False, "ranking_or_risk_changed": False,
        "warning": "Eligibility only permits human review; it never deploys a model automatically.",
    }
    (DATA / "phase52_promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    summary = {
        "phases": [48, 49, 50, 51, 52], "phase48_ledger": {"v1": v1, "v2": v2},
        "phase49_forecast": forecast_summary, "phase50_rejection_reason_rows": int(len(reason_report)),
        "phase51_stress_scenarios": int(len(stress_report)), "phase52_promotion_gate": gate,
        "production_champion": "V1", "v2_promoted": False, "production_settings_changed": False,
    }
    (DATA / "phase48_52_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    build_reports()
