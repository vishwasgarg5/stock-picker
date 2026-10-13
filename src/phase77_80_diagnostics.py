from __future__ import annotations

"""Phases 77-80: operational delivery, V2 funnel, forecast evidence and storage plan.

All diagnostics are read-only with respect to production selection and source OHLCV.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TELEGRAM_FILES = [
    "telegram_sent.csv",
    "telegram_evening_sent.csv",
    "telegram_paper_sent.csv",
    "telegram_paper_v2_sent.csv",
]


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def delivery_audit() -> tuple[dict[str, Any], pd.DataFrame]:
    per_file: dict[str, Any] = {}
    detail: list[dict[str, Any]] = []
    totals = {"rows": 0, "confirmed": 0, "legacy_unverified": 0, "missing_status": 0, "duplicates": 0}
    for name in TELEGRAM_FILES:
        frame = _read(name)
        if frame.empty:
            per_file[name] = {"rows": 0, "confirmed": 0, "legacy_unverified": 0, "missing_status": 0, "duplicate_keys": 0}
            continue
        status = frame.get("delivery_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        key = frame.get("message_hash", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        usable_key = key.ne("")
        duplicate = usable_key & key.duplicated(keep=False)
        counts = {
            "rows": int(len(frame)),
            "confirmed": int(status.eq("TELEGRAM_API_CONFIRMED").sum()),
            "legacy_unverified": int(status.eq("LEGACY_UNVERIFIED").sum()),
            "missing_status": int(status.eq("").sum()),
            "duplicate_keys": int((usable_key & key.duplicated()).sum()),
            "rows_missing_message_key": int((~usable_key).sum()),
        }
        per_file[name] = counts
        for i in frame.index:
            detail.append({
                "file": name,
                "row_index": int(i),
                "delivery_status": status.loc[i] or "MISSING_STATUS",
                "message_hash_present": bool(usable_key.loc[i]),
                "duplicate_message_key": bool(duplicate.loc[i]),
                "telegram_message_id_present": bool(str(frame.get("telegram_message_id", pd.Series("", index=frame.index)).loc[i]).strip()),
            })
        for key_name, total_name in [
            ("rows", "rows"), ("confirmed", "confirmed"), ("legacy_unverified", "legacy_unverified"),
            ("missing_status", "missing_status"), ("duplicate_keys", "duplicates")
        ]:
            totals[total_name] += counts[key_name]
    coverage = totals["confirmed"] / totals["rows"] if totals["rows"] else None
    report = {
        "status": "NO_RECEIPT_ROWS" if totals["rows"] == 0 else ("RECEIPT_COVERAGE_COMPLETE" if totals["confirmed"] == totals["rows"] else "RECEIPT_AUDIT_INCOMPLETE"),
        **totals,
        "confirmed_receipt_pct": round(coverage * 100, 2) if coverage is not None else None,
        "per_file": per_file,
        "interpretation": "Only TELEGRAM_API_CONFIRMED counts as API-confirmed; legacy or missing status does not prove delivery failure.",
        "production_behavior_changed": False,
    }
    return report, pd.DataFrame(detail)


def v2_funnel_audit() -> tuple[dict[str, Any], pd.DataFrame]:
    trades = _read("paper_trades_v2.csv")
    funnel = _read("phase32_v2_selection_funnel.csv")
    if trades.empty or "signal" not in trades.columns:
        return {"status": "INSUFFICIENT_DATA", "rows": len(trades), "buy_rows": 0, "production_filters_changed": False}, funnel
    signal = trades["signal"].fillna("").astype(str).str.upper().str.strip()
    skips = trades.loc[signal.eq("SKIP")]
    reasons = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip().replace("", "MISSING_REASON")
    funnel_valid = not funnel.empty and {"stage", "input_rows", "output_rows"}.issubset(funnel.columns)
    stage_detail = pd.DataFrame()
    if funnel_valid:
        stage_detail = funnel.copy()
        stage_detail["input_rows"] = pd.to_numeric(stage_detail["input_rows"], errors="coerce")
        stage_detail["output_rows"] = pd.to_numeric(stage_detail["output_rows"], errors="coerce")
        stage_detail["calculated_drop_rows"] = (stage_detail["input_rows"] - stage_detail["output_rows"]).clip(lower=0)
    report = {
        "status": "ZERO_BUY" if not signal.eq("BUY").any() else "BUY_ACTIVITY_PRESENT",
        "rows": int(len(trades)),
        "buy_rows": int(signal.eq("BUY").sum()),
        "skip_rows": int(signal.eq("SKIP").sum()),
        "skip_reason_counts": {str(k): int(v) for k, v in reasons.value_counts().items()},
        "missing_skip_reason_rows": int(reasons.eq("MISSING_REASON").sum()),
        "funnel_rows": int(len(funnel)),
        "funnel_schema_valid": bool(funnel_valid),
        "largest_recorded_drop_stage": None,
        "production_filters_changed": False,
        "thresholds_deployed": False,
    }
    if not stage_detail.empty and stage_detail["calculated_drop_rows"].notna().any():
        largest = stage_detail.loc[stage_detail["calculated_drop_rows"].idxmax()]
        report["largest_recorded_drop_stage"] = str(largest["stage"])
        report["largest_recorded_drop_rows"] = int(largest["calculated_drop_rows"])
    return report, stage_detail


def forecast_evidence() -> tuple[dict[str, Any], pd.DataFrame]:
    frame = _read("phase33_chronological_prediction_scores.csv")
    needed = {"target_date", "close_ape", "baseline_close_ape"}
    if frame.empty or not needed.issubset(frame.columns):
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "beats_baseline": False, "model_changed": False}, pd.DataFrame()
    x = frame.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    for col in ("close_ape", "baseline_close_ape"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    x = x[x["close_ape"].ge(0) & x["baseline_close_ape"].ge(0)].sort_values("target_date")
    if x.empty:
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "beats_baseline": False, "model_changed": False}, pd.DataFrame()
    x["model_ape_pct"] = x["close_ape"] * 100
    x["baseline_ape_pct"] = x["baseline_close_ape"] * 100
    by_date = x.groupby("target_date", as_index=False).agg(
        scored_rows=("close_ape", "size"),
        model_mape_pct=("model_ape_pct", "mean"),
        baseline_mape_pct=("baseline_ape_pct", "mean"),
    ).sort_values("target_date")
    by_date["delta_pct_points"] = by_date["model_mape_pct"] - by_date["baseline_mape_pct"]
    recent = by_date.tail(5)
    model = float(x["model_ape_pct"].mean())
    baseline = float(x["baseline_ape_pct"].mean())
    recent_model = float(recent["model_mape_pct"].mean()) if len(recent) else None
    recent_baseline = float(recent["baseline_mape_pct"].mean()) if len(recent) else None
    report = {
        "status": "MODEL_BEATS_BASELINE" if model < baseline else "BASELINE_NOT_BEATEN",
        "scored_rows": int(len(x)),
        "target_dates": int(x["target_date"].nunique()),
        "model_mape_pct": model,
        "baseline_mape_pct": baseline,
        "delta_pct_points": model - baseline,
        "beats_baseline": bool(model < baseline),
        "recent_sessions": int(len(recent)),
        "recent_model_mape_pct": recent_model,
        "recent_baseline_mape_pct": recent_baseline,
        "recent_delta_pct_points": (recent_model - recent_baseline) if recent_model is not None else None,
        "recent_beats_baseline": bool(recent_model < recent_baseline) if recent_model is not None else False,
        "same_scored_rows": True,
        "model_changed": False,
        "promotion_evidence_only": True,
    }
    return report, by_date


def storage_migration_plan() -> dict[str, Any]:
    path = DATA / "ohlcv.csv"
    if not path.exists():
        return {"status": "SOURCE_MISSING", "source_modified": False, "migration_executed": False}
    size = path.stat().st_size
    return {
        "status": "PLAN_REQUIRED" if size > 50 * 1024 * 1024 else "NO_SIZE_TRIGGER",
        "source_path": "data/ohlcv.csv",
        "source_bytes": size,
        "source_mib": round(size / (1024 * 1024), 2),
        "steps": [
            "Create a checksum and row/date/symbol baseline before any change.",
            "Prototype compressed or partitioned output beside the original file.",
            "Round-trip compare schema, unique symbol/date keys, OHLC values and row counts.",
            "Test a clean workflow run against the prototype and measure runtime/storage.",
            "Switch readers only after parity checks; retain rollback to the original source.",
        ],
        "migration_executed": False,
        "source_modified": False,
        "workflow_triggers_changed": False,
    }


def run() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    delivery, delivery_rows = delivery_audit()
    v2, funnel = v2_funnel_audit()
    forecast, by_date = forecast_evidence()
    storage = storage_migration_plan()
    if not delivery_rows.empty:
        delivery_rows.to_csv(DATA / "phase77_telegram_delivery_audit.csv", index=False)
    if not funnel.empty:
        funnel.to_csv(DATA / "phase78_v2_funnel_audit.csv", index=False)
    if not by_date.empty:
        by_date.assign(target_date=by_date["target_date"].dt.strftime("%Y-%m-%d")).to_csv(DATA / "phase79_forecast_evidence.csv", index=False)
    report = {
        "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "phases": [77, 78, 79, 80],
        "phase77_delivery": delivery,
        "phase78_v2_funnel": v2,
        "phase79_forecast_evidence": forecast,
        "phase80_storage_migration_plan": storage,
        "safety": {"production_champion": "V1", "v2_mode": "SHADOW_ONLY", "automatic_promotion": False,
                   "ranking_or_risk_changed": False, "source_ohlcv_modified": False},
    }
    (DATA / "phase77_80_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False, default=str))
    return report


if __name__ == "__main__":
    run()
