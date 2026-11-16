from __future__ import annotations

"""Phases 93-96: Telegram receipt audit, forecast error segmentation,
V2 earliest-gate tracing, and reversible OHLCV migration preflight.
Diagnostic only: no resends, production reader changes, or strategy changes.
"""
import gzip
import hashlib
import json
import shutil
import tempfile
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


def phase93_receipt_forensics() -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for filename in RECEIPT_FILES:
        df = read_csv(filename)
        if df.empty or "message_hash" not in df:
            continue
        for col in ("target_date", "delivery_status", "telegram_message_id", "telegram_date_utc"):
            if col not in df:
                df[col] = ""
        for idx, row in df.iterrows():
            hash_value = str(row.get("message_hash", "") or "").strip()
            if not hash_value:
                continue
            raw_ts = pd.to_numeric(pd.Series([row.get("telegram_date_utc")]), errors="coerce").iloc[0]
            ts = pd.to_datetime(raw_ts, unit="s", utc=True, errors="coerce") if pd.notna(raw_ts) else pd.NaT
            rows.append({
                "file": filename, "row_index": int(idx), "target_date": str(row.get("target_date", "") or ""),
                "message_hash": hash_value, "delivery_status": str(row.get("delivery_status", "") or ""),
                "telegram_message_id": str(row.get("telegram_message_id", "") or ""),
                "telegram_date_utc": ts.isoformat() if pd.notna(ts) else "",
                "_timestamp": ts,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"status": "NO_HASHED_RECEIPTS", "hashed_rows": 0, "repeated_hash_groups": 0,
                "groups_with_multiple_message_ids": 0, "resends_performed": False}, pd.DataFrame()
    grouped = []
    for (filename, hash_value), group in detail.groupby(["file", "message_hash"], sort=True):
        ids = sorted({v for v in group["telegram_message_id"].astype(str) if v and v.lower() != "nan"})
        times = group["_timestamp"].dropna().sort_values()
        seconds = float((times.iloc[-1] - times.iloc[0]).total_seconds()) if len(times) > 1 else None
        dates = sorted(set(group["target_date"].astype(str)))
        grouped.append({
            "file": filename, "message_hash": hash_value, "rows": int(len(group)),
            "target_dates": "|".join(dates), "distinct_message_ids": len(ids),
            "message_ids": "|".join(ids),
            "confirmed_rows": int(group["delivery_status"].astype(str).eq("TELEGRAM_API_CONFIRMED").sum()),
            "first_confirmed_utc": times.iloc[0].isoformat() if len(times) else "",
            "last_confirmed_utc": times.iloc[-1].isoformat() if len(times) else "",
            "elapsed_seconds": seconds,
            "classification": ("MULTIPLE_API_MESSAGE_IDS" if len(ids) > 1 else
                               "REPEATED_HASH_SAME_ID" if len(group) > 1 and len(ids) == 1 else
                               "REPEATED_HASH_UNVERIFIED" if len(group) > 1 else "UNIQUE_HASH"),
        })
    groups = pd.DataFrame(grouped)
    repeated = groups[groups["rows"].gt(1)]
    return {
        "status": "REVIEW_REQUIRED" if not repeated.empty else "NO_REPEATED_HASHES",
        "hashed_rows": int(len(detail)), "repeated_hash_groups": int(len(repeated)),
        "groups_with_multiple_message_ids": int(repeated["classification"].eq("MULTIPLE_API_MESSAGE_IDS").sum()),
        "groups_with_timestamp_intervals": int(repeated["elapsed_seconds"].notna().sum()),
        "resends_performed": False,
        "note": "Receipt logs lack full message bodies; same hash with multiple API message IDs warrants manual review but does not prove an unintended resend.",
    }, groups


def phase94_forecast_error_segments() -> tuple[dict[str, Any], pd.DataFrame]:
    df = read_csv("phase33_chronological_prediction_scores.csv")
    required = {"target_date", "symbol", "close_ape", "baseline_close_ape"}
    if df.empty or not required.issubset(df.columns):
        return {"status": "INSUFFICIENT_DATA", "rows_scored": 0}, pd.DataFrame()
    x = df.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce")
    for col in ("close_ape", "baseline_close_ape"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "symbol", "close_ape", "baseline_close_ape"])
    x = x[(x["close_ape"] >= 0) & (x["baseline_close_ape"] >= 0)]
    if "period" not in x:
        x["period"] = "UNSPECIFIED"
    x["period"] = x["period"].fillna("UNSPECIFIED").astype(str)
    x["close_ape_pct"] = x["close_ape"] * 100
    x["baseline_close_ape_pct"] = x["baseline_close_ape"] * 100
    x["delta_pct_points"] = x["close_ape_pct"] - x["baseline_close_ape_pct"]
    symbol = x.groupby("symbol", as_index=False).agg(
        rows=("close_ape", "size"), target_dates=("target_date", "nunique"),
        model_mape_pct=("close_ape_pct", "mean"), baseline_mape_pct=("baseline_close_ape_pct", "mean"),
        mean_delta_pct_points=("delta_pct_points", "mean"))
    symbol["segment_type"] = "SYMBOL"
    period = x.groupby("period", as_index=False).agg(
        rows=("close_ape", "size"), target_dates=("target_date", "nunique"),
        model_mape_pct=("close_ape_pct", "mean"), baseline_mape_pct=("baseline_close_ape_pct", "mean"),
        mean_delta_pct_points=("delta_pct_points", "mean")).rename(columns={"period": "symbol"})
    period["segment_type"] = "PERIOD"
    combined = pd.concat([symbol, period], ignore_index=True)
    combined["model_beats_baseline"] = combined["mean_delta_pct_points"] < 0
    combined = combined.sort_values(["segment_type", "mean_delta_pct_points"], ascending=[True, False]).reset_index(drop=True)
    worst = symbol.sort_values("mean_delta_pct_points", ascending=False).head(10)
    return {
        "status": "SEGMENTED_DIAGNOSTIC_ONLY", "rows_scored": int(len(x)),
        "symbols_scored": int(len(symbol)), "periods_scored": int(x["period"].nunique()),
        "symbols_worse_than_baseline": int(symbol["mean_delta_pct_points"].gt(0).sum()),
        "symbols_better_than_baseline": int(symbol["mean_delta_pct_points"].lt(0).sum()),
        "worst_symbols": [{"symbol": str(r.symbol), "delta_pct_points": round(float(r.mean_delta_pct_points), 4),
                           "rows": int(r.rows)} for r in worst.itertuples()],
        "chronological_source": "phase33_chronological_prediction_scores.csv",
        "ranking_or_threshold_changed": False,
    }, combined


def phase95_v2_earliest_rejection() -> tuple[dict[str, Any], pd.DataFrame]:
    funnel = read_csv("phase32_v2_selection_funnel.csv")
    if funnel.empty or not {"stage", "input_rows", "output_rows", "rejected_rows"}.issubset(funnel.columns):
        return {"status": "INSUFFICIENT_FUNNEL_DATA", "earliest_blocking_stage": None,
                "signals_changed": False}, pd.DataFrame()
    x = funnel.copy()
    for col in ("input_rows", "output_rows", "rejected_rows"):
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0).astype(int)
    x["stage_order"] = range(1, len(x) + 1)
    x["retention_pct"] = x.apply(lambda r: round(100 * r["output_rows"] / r["input_rows"], 2)
                                 if r["input_rows"] > 0 else None, axis=1)
    rejecting = x[x["rejected_rows"].gt(0)]
    first = rejecting.iloc[0] if not rejecting.empty else None
    x["is_earliest_rejection"] = False
    if first is not None:
        x.loc[x.index == first.name, "is_earliest_rejection"] = True
    return {
        "status": "DIAGNOSTIC_ONLY", "funnel_stages": int(len(x)),
        "earliest_blocking_stage": str(first["stage"]) if first is not None else None,
        "earliest_stage_input_rows": int(first["input_rows"]) if first is not None else 0,
        "earliest_stage_rejected_rows": int(first["rejected_rows"]) if first is not None else 0,
        "stages_with_rejections": int(len(rejecting)),
        "total_rejected_rows_stage_sum": int(x["rejected_rows"].sum()),
        "signals_changed": False, "risk_thresholds_changed": False,
        "note": "The earliest stage with rejections is a funnel bottleneck candidate; later-stage counts are not additive unique candidates.",
    }, x


def phase96_reversible_storage_preflight() -> dict[str, Any]:
    source = DATA / "ohlcv.csv"
    if not source.exists():
        return {"status": "SOURCE_MISSING", "migration_executed": False, "reader_switched": False}
    before_size = source.stat().st_size
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    compressed_size = None
    parity = False
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="stock-picker-storage-preflight-") as temp_dir:
            archive = Path(temp_dir) / "ohlcv.csv.gz"
            with source.open("rb") as src, gzip.open(archive, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            compressed_size = archive.stat().st_size
            original_sample = pd.read_csv(source, nrows=100, low_memory=False)
            gzip_sample = pd.read_csv(archive, nrows=100, low_memory=False)
            parity = original_sample.equals(gzip_sample)
    except (OSError, ValueError, EOFError, pd.errors.ParserError) as exc:
        error = type(exc).__name__
    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    ratio = (1 - compressed_size / before_size) * 100 if compressed_size is not None and before_size else None
    return {
        "status": "PREFLIGHT_PASSED_NO_MIGRATION" if compressed_size is not None and parity and before_hash == after_hash
                  else "PREFLIGHT_REVIEW_REQUIRED",
        "source_size_bytes": int(before_size), "source_sha256_before": before_hash,
        "source_sha256_after": after_hash, "gzip_size_bytes": compressed_size,
        "estimated_size_reduction_pct": round(ratio, 2) if ratio is not None else None,
        "first_100_rows_parity": parity, "source_unchanged": before_hash == after_hash,
        "migration_executed": False, "production_reader_switched": False,
        "backup_restore_tested": False, "rollback_tested": False, "error_type": error,
        "next_required_steps": ["full-file parity validation", "reader compatibility tests",
                                "CI/history access test", "backup and restore test", "rollback rehearsal"],
    }


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    receipt, receipt_groups = phase93_receipt_forensics()
    forecast, forecast_segments = phase94_forecast_error_segments()
    v2, funnel = phase95_v2_earliest_rejection()
    storage = phase96_reversible_storage_preflight()
    receipt_groups.to_csv(DATA / "phase93_receipt_forensics.csv", index=False)
    forecast_segments.to_csv(DATA / "phase94_forecast_error_segments.csv", index=False)
    funnel.to_csv(DATA / "phase95_v2_earliest_rejection.csv", index=False)
    report = {
        "phases": [93, 94, 95, 96],
        "phase93_receipt_forensics": receipt,
        "phase94_forecast_error_segments": forecast,
        "phase95_v2_earliest_rejection": v2,
        "phase96_storage_preflight": storage,
        "safety": {"production_champion": "V1", "v2_mode": "SHADOW_ONLY",
                   "automatic_promotion": False, "ranking_or_risk_changed": False,
                   "telegram_send_logic_changed": False, "messages_resent": False,
                   "source_ohlcv_modified": False, "production_reader_switched": False,
                   "workflow_triggers_changed": False},
    }
    (DATA / "phase93_96_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
