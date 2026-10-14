from __future__ import annotations

"""Phases 85-88: receipt anomaly classification, schema readiness, rolling forecast evidence,
and non-destructive storage performance benchmarking. Analysis only; no production decisions change.
"""
import gzip
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECEIPT_FILES = [
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


def receipt_anomaly_classification() -> tuple[dict[str, Any], pd.DataFrame]:
    """Classify repeated hashes without assuming every repeat is an accidental duplicate send."""
    frames: list[pd.DataFrame] = []
    per_file: dict[str, Any] = {}
    for name in RECEIPT_FILES:
        df = _read(name)
        if df.empty:
            per_file[name] = {"rows": 0, "hash_coverage_pct": None, "status_coverage_pct": None,
                              "message_id_coverage_pct": None}
            continue
        hashes = df.get("message_hash", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        status = df.get("delivery_status", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        msgid = df.get("telegram_message_id", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        valid_hash = hashes.ne("")
        repeated = valid_hash & hashes.duplicated(keep=False)
        per_file[name] = {
            "rows": int(len(df)),
            "hash_coverage_pct": round(100 * valid_hash.mean(), 2),
            "status_coverage_pct": round(100 * status.ne("").mean(), 2),
            "message_id_coverage_pct": round(100 * msgid.ne("").mean(), 2),
            "repeated_hash_rows": int(repeated.sum()),
            "unique_repeated_hashes": int(hashes[repeated].nunique()),
        }
        part = pd.DataFrame({
            "file": name,
            "row_index": df.index.astype(int),
            "message_hash": hashes.to_numpy(),
            "delivery_status": status.to_numpy(),
            "telegram_message_id": msgid.to_numpy(),
            "hash_repeated_in_file": repeated.to_numpy(),
        })
        part["classification"] = np.select(
            [~valid_hash.to_numpy(), repeated.to_numpy()],
            ["HASH_MISSING", "REPEATED_HASH_REVIEW"],
            default="HASH_UNIQUE",
        )
        frames.append(part)
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["file", "row_index", "message_hash", "delivery_status", "telegram_message_id",
                 "hash_repeated_in_file", "classification"])
    repeated_rows = int(detail["hash_repeated_in_file"].sum()) if not detail.empty else 0
    missing_hash = int(detail["classification"].eq("HASH_MISSING").sum()) if not detail.empty else 0
    total = int(len(detail))
    status = "REVIEW_REQUIRED" if repeated_rows or missing_hash else ("NO_RECEIPTS" if not total else "NO_HASH_ANOMALIES")
    return {
        "status": status, "rows": total, "repeated_hash_rows": repeated_rows,
        "missing_hash_rows": missing_hash, "hash_coverage_pct": round(100 * (total - missing_hash) / total, 2) if total else None,
        "per_file": per_file,
        "duplicate_interpretation": "Repeated hashes require content/time/retry review; they are not automatically classified as duplicate sends.",
        "telegram_send_logic_changed": False, "messages_resubmitted": False,
    }, detail


def receipt_schema_readiness() -> tuple[dict[str, Any], pd.DataFrame]:
    required = ["delivery_status", "message_hash", "telegram_message_id"]
    rows = []
    for name in RECEIPT_FILES:
        df = _read(name)
        present = set(df.columns)
        row = {"file": name, "rows": int(len(df)), "file_present": (DATA / name).exists()}
        for col in required:
            row[f"has_{col}"] = col in present
            row[f"{col}_nonempty_pct"] = round(100 * df[col].fillna("").astype(str).str.strip().ne("").mean(), 2) if col in present and len(df) else (0.0 if col in present else None)
        row["schema_ready_for_future_receipts"] = all(col in present for col in required)
        rows.append(row)
    detail = pd.DataFrame(rows)
    ready = int(detail["schema_ready_for_future_receipts"].sum()) if not detail.empty else 0
    return {
        "status": "SCHEMA_READY" if ready == len(RECEIPT_FILES) and len(detail) == len(RECEIPT_FILES) else "SCHEMA_GAPS",
        "files_checked": int(len(detail)), "files_with_required_columns": ready,
        "required_columns": required,
        "note": "Audit only: no Telegram send path or receipt writer is modified by this phase.",
        "send_path_changed": False, "receipt_writer_changed": False,
    }, detail


def rolling_forecast_evidence() -> tuple[dict[str, Any], pd.DataFrame]:
    df = _read("phase33_chronological_prediction_scores.csv")
    required = {"target_date", "close_ape", "baseline_close_ape"}
    if df.empty or not required.issubset(df.columns):
        return {"status": "INSUFFICIENT_DATA", "target_dates": 0, "minimum_dates_for_window": 5}, pd.DataFrame()
    x = df.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    for col in ("close_ape", "baseline_close_ape"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    x = x[(x["close_ape"] >= 0) & (x["baseline_close_ape"] >= 0)]
    dates = x.groupby("target_date", as_index=False).agg(
        scored_rows=("close_ape", "size"),
        model_mape_pct=("close_ape", lambda s: float(s.mean() * 100)),
        baseline_mape_pct=("baseline_close_ape", lambda s: float(s.mean() * 100)),
    ).sort_values("target_date").reset_index(drop=True)
    dates["delta_pct_points"] = dates["model_mape_pct"] - dates["baseline_mape_pct"]
    windows = []
    for end in range(4, len(dates)):
        w = dates.iloc[end - 4:end + 1]
        windows.append({
            "window_start": str(w.iloc[0]["target_date"].date()),
            "window_end": str(w.iloc[-1]["target_date"].date()),
            "sessions": int(len(w)),
            "model_mape_pct": float(w["model_mape_pct"].mean()),
            "baseline_mape_pct": float(w["baseline_mape_pct"].mean()),
            "delta_pct_points": float(w["delta_pct_points"].mean()),
            "model_beats_baseline": bool(w["delta_pct_points"].mean() < 0),
        })
    wdf = pd.DataFrame(windows)
    overall_delta = float(dates["delta_pct_points"].mean()) if not dates.empty else None
    return {
        "status": "BASELINE_NOT_BEATEN" if overall_delta is not None and overall_delta >= 0 else ("MODEL_BEATS_BASELINE" if overall_delta is not None else "INSUFFICIENT_DATA"),
        "target_dates": int(len(dates)), "scored_rows": int(len(x)),
        "overall_model_mape_pct": float(dates["model_mape_pct"].mean()) if not dates.empty else None,
        "overall_baseline_mape_pct": float(dates["baseline_mape_pct"].mean()) if not dates.empty else None,
        "overall_delta_pct_points": overall_delta,
        "five_session_windows": int(len(wdf)),
        "windows_model_beats_baseline": int(wdf["model_beats_baseline"].sum()) if not wdf.empty else 0,
        "windows_baseline_beats_or_ties": int((~wdf["model_beats_baseline"]).sum()) if not wdf.empty else 0,
        "model_or_threshold_changed": False, "promotion_evidence_only": True,
    }, wdf


def storage_performance_benchmark() -> dict[str, Any]:
    """Benchmark raw vs temporary gzip read; preserve source bytes and production reader."""
    source = DATA / "ohlcv.csv"
    if not source.exists():
        return {"status": "SOURCE_MISSING", "source_modified": False, "production_reader_switched": False}
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    before_hash = digest(source)
    source_bytes = source.stat().st_size
    try:
        with tempfile.TemporaryDirectory(prefix="stock-picker-phase88-") as temp:
            compressed_path = Path(temp) / "ohlcv.csv.gz"
            with source.open("rb") as inp, gzip.open(compressed_path, "wb", compresslevel=6) as out:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
            compressed_bytes = compressed_path.stat().st_size
            t0 = time.perf_counter()
            raw = pd.read_csv(source, low_memory=False)
            raw_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            zipped = pd.read_csv(compressed_path, compression="gzip", low_memory=False)
            gzip_seconds = time.perf_counter() - t1
            parity = list(raw.columns) == list(zipped.columns) and len(raw) == len(zipped) and raw.equals(zipped)
            raw_mem = int(raw.memory_usage(index=True, deep=True).sum())
            zipped_mem = int(zipped.memory_usage(index=True, deep=True).sum())
            rows = len(raw)
            columns = list(raw.columns)
        after_hash = digest(source)
        unchanged = before_hash == after_hash
        return {
            "status": "BENCHMARK_PARITY_PASS" if parity and unchanged else "BENCHMARK_REVIEW_REQUIRED",
            "source_path": "data/ohlcv.csv", "source_bytes": int(source_bytes),
            "gzip_bytes": int(compressed_bytes),
            "size_reduction_pct": round((1 - compressed_bytes / source_bytes) * 100, 2) if source_bytes else 0.0,
            "rows": int(rows), "columns": columns, "raw_read_seconds": round(raw_seconds, 4),
            "gzip_read_seconds": round(gzip_seconds, 4),
            "gzip_read_time_delta_pct": round((gzip_seconds / raw_seconds - 1) * 100, 2) if raw_seconds else None,
            "raw_dataframe_memory_bytes": raw_mem, "gzip_dataframe_memory_bytes": zipped_mem,
            "dataframe_memory_equal": raw_mem == zipped_mem, "dataframe_parity": bool(parity),
            "sha256_before": before_hash, "sha256_after": after_hash, "source_unchanged": bool(unchanged),
            "prototype_only": True, "production_reader_switched": False, "migration_executed": False,
        }
    except Exception as exc:
        return {"status": "BENCHMARK_FAILED", "source_path": "data/ohlcv.csv",
                "source_bytes": int(source_bytes), "source_modified": False,
                "production_reader_switched": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    anomalies, anomaly_rows = receipt_anomaly_classification()
    readiness, schema_rows = receipt_schema_readiness()
    forecast, windows = rolling_forecast_evidence()
    storage = storage_performance_benchmark()
    anomaly_rows.to_csv(DATA / "phase85_receipt_anomalies.csv", index=False)
    schema_rows.to_csv(DATA / "phase86_receipt_schema_readiness.csv", index=False)
    windows.to_csv(DATA / "phase87_rolling_forecast_windows.csv", index=False)
    report = {
        "phases": [85, 86, 87, 88],
        "phase85_receipt_anomaly_classification": anomalies,
        "phase86_receipt_schema_readiness": readiness,
        "phase87_rolling_forecast_evidence": forecast,
        "phase88_storage_performance": storage,
        "safety": {
            "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
            "automatic_promotion": False, "ranking_or_risk_changed": False,
            "telegram_send_logic_changed": False, "receipt_writer_changed": False,
            "production_reader_switched": False, "source_ohlcv_modified": False,
            "workflow_triggers_changed": False,
        },
    }
    (DATA / "phase85_88_diagnostics.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run()
