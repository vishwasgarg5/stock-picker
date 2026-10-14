from __future__ import annotations

"""Phases 89-92: delivery reconciliation, segmented forecast diagnostics,
V2 rejection analysis, and a non-destructive storage migration checklist.
All outputs are diagnostic; no production strategy or execution behavior changes.
"""
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECEIPT_FILES = ["telegram_sent.csv", "telegram_evening_sent.csv",
                 "telegram_paper_sent.csv", "telegram_paper_v2_sent.csv"]


def read_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def phase89_receipt_reconciliation() -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for name in RECEIPT_FILES:
        df = read_csv(name)
        if df.empty:
            continue
        for col in ["message_hash", "delivery_status", "telegram_message_id"]:
            if col not in df:
                df[col] = ""
        for idx, row in df.iterrows():
            rows.append({
                "file": name, "row_index": int(idx),
                "message_hash": str(row.get("message_hash", "") or "").strip(),
                "delivery_status": str(row.get("delivery_status", "") or "").strip(),
                "telegram_message_id": str(row.get("telegram_message_id", "") or "").strip(),
            })
    detail = pd.DataFrame(rows, columns=["file", "row_index", "message_hash", "delivery_status", "telegram_message_id"])
    if detail.empty:
        return {"status": "NO_RECEIPTS", "rows": 0, "repeated_hash_groups": 0,
                "confirmed_repeated_hash_groups": 0, "missing_message_id_rows": 0}, detail
    detail["hash_present"] = detail["message_hash"].ne("")
    detail["message_id_present"] = detail["telegram_message_id"].ne("")
    detail["api_confirmed"] = detail["delivery_status"].eq("TELEGRAM_API_CONFIRMED")
    detail["repeated_hash"] = detail["hash_present"] & detail.duplicated(["file", "message_hash"], keep=False)
    repeated = detail[detail["repeated_hash"]]
    groups = repeated.groupby(["file", "message_hash"], dropna=False)
    group_rows = []
    for (filename, message_hash), group in groups:
        ids = group.loc[group["message_id_present"], "telegram_message_id"].nunique()
        group_rows.append({
            "file": filename, "message_hash": message_hash, "rows": int(len(group)),
            "api_confirmed_rows": int(group["api_confirmed"].sum()),
            "distinct_message_ids": int(ids),
            "classification": ("MULTIPLE_CONFIRMED_MESSAGE_IDS" if ids > 1 else
                               "REPEATED_HASH_WITH_CONFIRMATION" if group["api_confirmed"].any() else
                               "REPEATED_HASH_UNVERIFIED"),
        })
    groups_df = pd.DataFrame(group_rows)
    detail = detail.merge(groups_df[["file", "message_hash", "classification"]], on=["file", "message_hash"], how="left")
    detail["classification"] = detail["classification"].fillna("HASH_UNIQUE")
    missing_ids = int((~detail["message_id_present"]).sum())
    return {
        "status": "REVIEW_REQUIRED" if not groups_df.empty or missing_ids else "RECONCILED",
        "rows": int(len(detail)), "repeated_hash_groups": int(len(groups_df)),
        "confirmed_repeated_hash_groups": int(groups_df["api_confirmed_rows"].gt(0).sum()) if not groups_df.empty else 0,
        "multiple_confirmed_message_id_groups": int(groups_df["classification"].eq("MULTIPLE_CONFIRMED_MESSAGE_IDS").sum()) if not groups_df.empty else 0,
        "missing_message_id_rows": missing_ids,
        "note": "Repeated hashes are evidence for review, not proof of accidental duplicate delivery. No messages are resent.",
        "send_logic_changed": False, "receipt_writer_changed": False,
    }, detail


def phase90_segmented_forecast() -> tuple[dict[str, Any], pd.DataFrame]:
    df = read_csv("phase33_chronological_prediction_scores.csv")
    required = {"target_date", "symbol", "close_ape", "baseline_close_ape"}
    if df.empty or not required.issubset(df.columns):
        return {"status": "INSUFFICIENT_DATA", "rows": 0}, pd.DataFrame()
    x = df.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce")
    for col in ["close_ape", "baseline_close_ape"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "symbol", "close_ape", "baseline_close_ape"])
    x = x[(x["close_ape"] >= 0) & (x["baseline_close_ape"] >= 0)]
    x["target_date"] = x["target_date"].dt.strftime("%Y-%m-%d")
    grouped = x.groupby("symbol", as_index=False).agg(
        scored_rows=("close_ape", "size"),
        model_mape_pct=("close_ape", lambda s: float(s.mean() * 100)),
        baseline_mape_pct=("baseline_close_ape", lambda s: float(s.mean() * 100)),
        target_dates=("target_date", "nunique"),
    )
    grouped["delta_pct_points"] = grouped["model_mape_pct"] - grouped["baseline_mape_pct"]
    grouped["segment_status"] = grouped["delta_pct_points"].map(
        lambda d: "MODEL_BEATS_BASELINE" if d < 0 else "BASELINE_BEATS_OR_TIES")
    grouped = grouped.sort_values(["delta_pct_points", "symbol"]).reset_index(drop=True)
    return {
        "status": "SEGMENTED_DIAGNOSTIC_ONLY" if not grouped.empty else "INSUFFICIENT_DATA",
        "symbols_scored": int(len(grouped)), "rows_scored": int(len(x)),
        "symbols_model_beats_baseline": int(grouped["delta_pct_points"].lt(0).sum()),
        "symbols_baseline_beats_or_ties": int(grouped["delta_pct_points"].ge(0).sum()),
        "model_or_threshold_changed": False, "promotion_evidence_only": True,
        "note": "Per-symbol results are descriptive; no symbol-specific ranking or filter is changed.",
    }, grouped


def phase91_v2_rejection_analysis() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trades = read_csv("paper_trades_v2.csv")
    funnel = read_csv("phase32_v2_selection_funnel.csv")
    if trades.empty or "signal" not in trades:
        return {"status": "INSUFFICIENT_DATA", "rows": 0}, pd.DataFrame(), funnel
    signal = trades["signal"].fillna("").astype(str).str.upper().str.strip()
    skips = trades.loc[signal.eq("SKIP")].copy()
    reason_col = "no_trade_reason" if "no_trade_reason" in skips else None
    if reason_col:
        skips["diagnostic_reason"] = skips[reason_col].fillna("").astype(str).str.strip().replace("", "UNSPECIFIED")
    else:
        skips["diagnostic_reason"] = "UNSPECIFIED"
    reason_counts = skips.groupby("diagnostic_reason", dropna=False).size().rename("rows").reset_index()
    reason_counts = reason_counts.sort_values(["rows", "diagnostic_reason"], ascending=[False, True])
    stage_counts = funnel.copy()
    if not stage_counts.empty and "stage" in stage_counts:
        for col in ["input_rows", "output_rows", "rejected_rows"]:
            if col in stage_counts:
                stage_counts[col] = pd.to_numeric(stage_counts[col], errors="coerce")
    buys = int(signal.eq("BUY").sum())
    return {
        "status": "DIAGNOSTIC_ONLY", "rows": int(len(trades)), "buy_rows": buys,
        "skip_rows": int(len(skips)), "skip_rows_with_reason": int(skips["diagnostic_reason"].ne("UNSPECIFIED").sum()),
        "unique_skip_reasons": int(reason_counts["diagnostic_reason"].nunique()),
        "funnel_stages": int(len(stage_counts)),
        "risk_thresholds_changed": False, "signals_changed": False,
        "note": "This report identifies rejection concentrations; it does not relax gates or create trades.",
    }, reason_counts, stage_counts


def phase92_storage_migration_readiness() -> dict[str, Any]:
    source = DATA / "ohlcv.csv"
    if not source.exists():
        return {"status": "SOURCE_MISSING", "migration_executed": False}
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    size = source.stat().st_size
    checklist = {
        "source_exists": True,
        "source_sha256_recorded": bool(digest),
        "source_size_bytes": int(size),
        "source_over_github_recommended_50mb": size > 50 * 1024 * 1024,
        "backup_and_restore_test_required": True,
        "reader_compatibility_test_required": True,
        "history_and_ci_access_test_required": True,
        "rollback_plan_required": True,
        "migration_executed": False,
        "production_reader_switched": False,
        "source_modified": False,
    }
    checklist["status"] = "MIGRATION_PLANNING_REQUIRED" if checklist["source_over_github_recommended_50mb"] else "MIGRATION_OPTIONAL"
    checklist["sha256"] = digest
    return checklist


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    receipt, receipt_detail = phase89_receipt_reconciliation()
    forecast, forecast_detail = phase90_segmented_forecast()
    v2, v2_reasons, v2_funnel = phase91_v2_rejection_analysis()
    storage = phase92_storage_migration_readiness()
    receipt_detail.to_csv(DATA / "phase89_receipt_reconciliation.csv", index=False)
    forecast_detail.to_csv(DATA / "phase90_symbol_forecast_diagnostics.csv", index=False)
    v2_reasons.to_csv(DATA / "phase91_v2_rejection_reasons.csv", index=False)
    v2_funnel.to_csv(DATA / "phase91_v2_funnel_snapshot.csv", index=False)
    report = {
        "phases": [89, 90, 91, 92],
        "phase89_receipt_reconciliation": receipt,
        "phase90_segmented_forecast": forecast,
        "phase91_v2_rejection_analysis": v2,
        "phase92_storage_migration_readiness": storage,
        "safety": {
            "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
            "automatic_promotion": False, "ranking_or_risk_changed": False,
            "telegram_send_logic_changed": False, "receipt_writer_changed": False,
            "production_reader_switched": False, "source_ohlcv_modified": False,
            "workflow_triggers_changed": False,
        },
    }
    (DATA / "phase89_92_diagnostics.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run()
