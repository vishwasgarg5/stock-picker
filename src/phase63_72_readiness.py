from __future__ import annotations

"""Phases 63-72: production-readiness evidence audit (analysis-only).

This module reports evidence gaps; it never changes stock ranks, risk limits,
trade signals, champion selection, or promotion settings.
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


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def prediction_report_audit() -> dict[str, Any]:
    predictions = _read("predictions.csv")
    if predictions.empty:
        return {"status": "INSUFFICIENT_DATA", "rows": 0, "target_date": None,
                "top10_complete": False, "invalid_ohlc_rows": None, "duplicate_symbol_rows": None}
    required = {"symbol", "target_date", "rank", "predicted_open", "predicted_high",
                "predicted_low", "predicted_close"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        return {"status": "INVALID_SCHEMA", "rows": len(predictions), "missing_columns": missing,
                "top10_complete": False}
    x = predictions.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["rank"] = pd.to_numeric(x["rank"], errors="coerce")
    target = x["target_date"].dropna().max()
    if pd.isna(target):
        return {"status": "INVALID_DATES", "rows": len(x), "top10_complete": False}
    day = x[x["target_date"].eq(target)].copy()
    symbols = day["symbol"].astype(str).str.upper().str.strip()
    price_cols = ["predicted_open", "predicted_high", "predicted_low", "predicted_close"]
    for col in price_cols:
        day[col] = pd.to_numeric(day[col], errors="coerce")
    invalid_prices = day[price_cols].isna().any(axis=1) | day[price_cols].le(0).any(axis=1)
    bad_ohlc = (
        day["predicted_high"].lt(day[["predicted_open", "predicted_close", "predicted_low"]].max(axis=1))
        | day["predicted_low"].gt(day[["predicted_open", "predicted_close", "predicted_high"]].min(axis=1))
    )
    top = day.sort_values("rank").head(10)
    top_symbols = top["symbol"].astype(str).str.upper().str.strip()
    top_invalid = top[price_cols].isna().any(axis=1) | top[price_cols].le(0).any(axis=1)
    duplicate_symbols = int(symbols.duplicated().sum())
    return {
        "status": "PASS" if len(top) == 10 and not top_invalid.any() and not invalid_prices.any() and not bad_ohlc.any() and duplicate_symbols == 0 else "BLOCK",
        "target_date": str(target.date()),
        "rows_for_target_date": int(len(day)),
        "top10_rows": int(len(top)),
        "top10_complete": bool(len(top) == 10 and not top_invalid.any()),
        "invalid_price_rows": int(invalid_prices.sum()),
        "invalid_ohlc_rows": int(bad_ohlc.sum()),
        "duplicate_symbol_rows": duplicate_symbols,
        "top10_symbols": top_symbols.tolist(),
        "ranked_candidates_are_not_buy_approval": True,
    }


def forecast_baseline_audit() -> dict[str, Any]:
    x = _read("phase33_chronological_prediction_scores.csv")
    needed = {"target_date", "close_ape", "baseline_close_ape"}
    if x.empty or not needed.issubset(x.columns):
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "beats_baseline": False}
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["close_ape"] = pd.to_numeric(x["close_ape"], errors="coerce")
    x["baseline_close_ape"] = pd.to_numeric(x["baseline_close_ape"], errors="coerce")
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    if x.empty:
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "beats_baseline": False}
    model = float(x["close_ape"].mean() * 100)
    baseline = float(x["baseline_close_ape"].mean() * 100)
    direction = _num(x, "close_direction_correct").dropna() if "close_direction_correct" in x else pd.Series(dtype=float)
    return {
        "status": "PASS" if model < baseline else "BASELINE_NOT_BEATEN",
        "scored_rows": int(len(x)),
        "unique_dates": int(x["target_date"].nunique()),
        "model_close_mape_pct": model,
        "baseline_close_mape_pct": baseline,
        "error_delta_pct_points": model - baseline,
        "beats_baseline": bool(model < baseline),
        "directional_accuracy_pct": float(direction.mean() * 100) if not direction.empty else None,
        "same_scored_rows_for_comparison": True,
        "production_model_changed": False,
    }


def ledger_audit() -> dict[str, Any]:
    x = _read("paper_trades.csv")
    required = {"signal", "entry_price", "exit_price", "shares", "allocated_capital",
                "gross_profit_loss", "costs", "profit_loss"}
    if x.empty or not required.issubset(x.columns):
        return {"status": "INSUFFICIENT_DATA", "rows": len(x),
                "missing_columns": sorted(required.difference(x.columns)), "unknown_values_imputed": False}
    signal = x["signal"].astype(str).str.upper().str.strip()
    executed = ~signal.isin(["SKIP", "HOLD", "NONE", ""])
    fields = {name: _num(x, name) for name in required if name != "signal"}
    complete = pd.Series(True, index=x.index)
    for values in fields.values():
        complete &= values.notna()
    relevant = executed & _num(x, "shares").gt(0)
    gross = _num(x, "gross_profit_loss")
    costs = _num(x, "costs")
    net = _num(x, "profit_loss")
    mismatch = relevant & gross.notna() & costs.notna() & net.notna() & ((gross - costs - net).abs() > 0.02)
    return {
        "status": "PASS" if not (relevant & ~complete).any() and not mismatch.any() else "BLOCK",
        "rows": int(len(x)),
        "executed_rows": int(relevant.sum()),
        "incomplete_executed_rows": int((relevant & ~complete).sum()),
        "net_pnl_reconciliation_mismatch_rows": int(mismatch.sum()),
        "unknown_values_imputed": False,
        "net_pnl_claim_safe": bool(not (relevant & ~complete).any() and not mismatch.any()),
    }


def telegram_receipt_audit() -> dict[str, Any]:
    names = ["telegram_sent.csv", "telegram_evening_sent.csv", "telegram_paper_sent.csv", "telegram_paper_v2_sent.csv"]
    per_file: dict[str, Any] = {}
    duplicate_keys = 0
    confirmed = 0
    total = 0
    for name in names:
        x = _read(name)
        if x.empty:
            per_file[name] = {"rows": 0, "confirmed": 0, "duplicate_message_keys": 0}
            continue
        status = x.get("delivery_status", pd.Series("", index=x.index)).fillna("").astype(str).str.strip()
        hashes = x.get("message_hash", pd.Series("", index=x.index)).fillna("").astype(str)
        dates = x.get("target_date", pd.Series("", index=x.index)).fillna("").astype(str)
        dup = int(pd.DataFrame({"date": dates, "hash": hashes}).duplicated().sum())
        nconfirmed = int(status.eq("TELEGRAM_API_CONFIRMED").sum())
        per_file[name] = {"rows": int(len(x)), "confirmed": nconfirmed,
                          "legacy_unverified": int(status.eq("LEGACY_UNVERIFIED").sum()),
                          "duplicate_message_keys": dup}
        total += len(x)
        confirmed += nconfirmed
        duplicate_keys += dup
    return {"status": "PASS" if confirmed > 0 and duplicate_keys == 0 else "REVIEW",
            "rows": int(total), "api_confirmed_rows": int(confirmed),
            "duplicate_message_keys": int(duplicate_keys), "by_file": per_file,
            "human_read_receipt_proven": False}


def v2_evidence_audit() -> dict[str, Any]:
    x = _read("paper_trades_v2.csv")
    if x.empty or "signal" not in x:
        return {"status": "INSUFFICIENT_DATA", "rows": len(x), "promotion_ready": False}
    signal = x["signal"].astype(str).str.upper().str.strip()
    buys = x[signal.eq("BUY")]
    skips = x[signal.eq("SKIP")]
    reasons = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip()
    sessions = pd.to_datetime(x.get("target_date", pd.Series(dtype=str)), errors="coerce").nunique()
    gate = {}
    path = DATA / "phase52_promotion_gate.json"
    if path.exists():
        try:
            gate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            gate = {}
    return {
        "status": "PASS" if len(skips) == 0 or not reasons.eq("").any() else "BLOCK",
        "rows": int(len(x)), "unique_sessions": int(sessions),
        "buy_rows": int(len(buys)), "skip_rows": int(len(skips)),
        "skip_rows_without_reason": int(reasons.eq("").sum()),
        "existing_promotion_safe": bool(gate.get("promotion_safe", False)),
        "promotion_ready": False,
        "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        "automatic_promotion": False,
    }


def build_report() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    sections = {
        "prediction_report": prediction_report_audit(),
        "forecast_baseline": forecast_baseline_audit(),
        "paper_ledger": ledger_audit(),
        "telegram_receipts": telegram_receipt_audit(),
        "v2_evidence": v2_evidence_audit(),
    }
    blocking = [name for name, section in sections.items()
                if section.get("status") in {"BLOCK", "BASELINE_NOT_BEATEN"}]
    report = {
        "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "REVIEW_REQUIRED" if blocking else "DIAGNOSTICS_RECORDED",
        "blocking_or_review_sections": blocking,
        "sections": sections,
        "safety": {
            "production_champion": "V1",
            "v2_mode": "SHADOW_ONLY",
            "automatic_promotion": False,
            "production_ranking_or_risk_changed": False,
            "storage_settings_changed": False,
            "unknown_accounting_values_imputed": False,
        },
    }
    (DATA / "phase63_72_readiness_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame([{"section": name, **{k: v for k, v in section.items() if not isinstance(v, (dict, list))}}
                  for name, section in sections.items()]).to_csv(DATA / "phase63_72_readiness_summary.csv", index=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    build_report()
