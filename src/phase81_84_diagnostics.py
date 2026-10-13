from __future__ import annotations

"""Phases 81-84: receipt reconciliation, V2 rejection diagnosis, forecast evidence,
and a non-destructive gzip storage prototype. No production decisions are changed.
"""
import hashlib
import json
import tempfile
import gzip
import shutil
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
OHLC_COLUMNS = {"open", "high", "low", "close"}


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def receipt_reconciliation() -> tuple[dict[str, Any], pd.DataFrame]:
    """Separate API-confirmed sends from legacy records and report id/hash anomalies."""
    details: list[dict[str, Any]] = []
    total = {"rows": 0, "confirmed": 0, "legacy_unverified": 0, "unknown_status": 0,
             "duplicate_hash_rows": 0, "confirmed_missing_message_id": 0}
    files: dict[str, Any] = {}
    for name in RECEIPT_FILES:
        frame = _read(name)
        if frame.empty:
            files[name] = {"rows": 0, "confirmed": 0, "legacy_unverified": 0,
                           "unknown_status": 0, "duplicate_hash_rows": 0}
            continue
        status = frame.get("delivery_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        hashes = frame.get("message_hash", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        ids = frame.get("telegram_message_id", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        confirmed = status.eq("TELEGRAM_API_CONFIRMED")
        legacy = status.eq("LEGACY_UNVERIFIED")
        unknown = ~(confirmed | legacy)
        dup = hashes.ne("") & hashes.duplicated(keep=False)
        missing_id = confirmed & ids.eq("")
        stats = {"rows": int(len(frame)), "confirmed": int(confirmed.sum()),
                 "legacy_unverified": int(legacy.sum()), "unknown_status": int(unknown.sum()),
                 "duplicate_hash_rows": int(dup.sum()),
                 "confirmed_missing_message_id": int(missing_id.sum())}
        files[name] = stats
        for key in total:
            total[key] += stats[key]
        for i in frame.index:
            details.append({"file": name, "row_index": int(i),
                            "status": status.loc[i] or "MISSING_OR_UNKNOWN",
                            "hash_present": bool(hashes.loc[i]),
                            "duplicate_hash": bool(dup.loc[i]),
                            "message_id_present": bool(ids.loc[i])})
    total["api_confirmed_pct"] = round(100 * total["confirmed"] / total["rows"], 2) if total["rows"] else None
    total["status"] = "RECEIPTS_NEED_RECONCILIATION" if (
        total["unknown_status"] or total["legacy_unverified"] or total["duplicate_hash_rows"]
        or total["confirmed_missing_message_id"]
    ) else ("NO_RECEIPTS" if total["rows"] == 0 else "RECEIPTS_CONSISTENT")
    total["per_file"] = files
    total["interpretation"] = "API confirmation is not proof a person read the message; legacy/unknown is not proof of delivery failure."
    total["production_send_behavior_changed"] = False
    return total, pd.DataFrame(details)


def v2_rejection_diagnosis() -> tuple[dict[str, Any], pd.DataFrame]:
    trades = _read("paper_trades_v2.csv")
    funnel = _read("phase32_v2_selection_funnel.csv")
    if trades.empty or "signal" not in trades:
        return {"status": "INSUFFICIENT_DATA", "rows": int(len(trades)),
                "production_filters_changed": False, "thresholds_deployed": False}, pd.DataFrame()
    signal = trades["signal"].fillna("").astype(str).str.upper().str.strip()
    skips = trades.loc[signal.eq("SKIP")].copy()
    reason = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip()
    reason = reason.replace("", "MISSING_REASON")
    reasons = reason.value_counts().rename_axis("skip_reason").reset_index(name="rows")
    reasons["share_of_skips_pct"] = (reasons["rows"] / max(1, len(skips)) * 100).round(2)
    if not funnel.empty and {"stage", "input_rows", "output_rows"}.issubset(funnel.columns):
        f = funnel.copy()
        f["input_rows"] = pd.to_numeric(f["input_rows"], errors="coerce")
        f["output_rows"] = pd.to_numeric(f["output_rows"], errors="coerce")
        f["drop_rows"] = (f["input_rows"] - f["output_rows"]).clip(lower=0)
        f["drop_pct_of_input"] = np.where(f["input_rows"] > 0, f["drop_rows"] / f["input_rows"] * 100, np.nan)
    else:
        f = pd.DataFrame(columns=["stage", "input_rows", "output_rows", "drop_rows", "drop_pct_of_input"])
    report = {
        "status": "ZERO_BUY" if not signal.eq("BUY").any() else "BUY_ACTIVITY_PRESENT",
        "rows": int(len(trades)), "buy_rows": int(signal.eq("BUY").sum()),
        "skip_rows": int(signal.eq("SKIP").sum()),
        "missing_reason_rows": int(reason.eq("MISSING_REASON").sum()),
        "skip_reason_counts": {str(row.skip_reason): int(row.rows) for row in reasons.itertuples()},
        "largest_reason": str(reasons.iloc[0]["skip_reason"]) if not reasons.empty else None,
        "largest_recorded_drop_stage": str(f.loc[f["drop_rows"].idxmax(), "stage"]) if not f.empty and f["drop_rows"].notna().any() else None,
        "largest_recorded_drop_rows": int(f["drop_rows"].max()) if not f.empty and f["drop_rows"].notna().any() else 0,
        "production_filters_changed": False, "thresholds_deployed": False,
        "warning": "Historical rejection counts are descriptive; do not relax filters without out-of-sample evidence."
    }
    return report, reasons.merge(f.assign(_join=1), how="cross") if False else f


def forecast_evidence() -> tuple[dict[str, Any], pd.DataFrame]:
    frame = _read("phase33_chronological_prediction_scores.csv")
    needed = {"target_date", "close_ape", "baseline_close_ape"}
    if frame.empty or not needed.issubset(frame.columns):
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "promotion_evidence_only": True}, pd.DataFrame()
    x = frame.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    for col in ("close_ape", "baseline_close_ape"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    x = x[x["close_ape"].ge(0) & x["baseline_close_ape"].ge(0)].sort_values("target_date")
    if x.empty:
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "promotion_evidence_only": True}, pd.DataFrame()
    x["model_ape_pct"] = x["close_ape"] * 100
    x["baseline_ape_pct"] = x["baseline_close_ape"] * 100
    by_date = x.groupby("target_date", as_index=False).agg(
        rows=("close_ape", "size"), model_mape_pct=("model_ape_pct", "mean"),
        baseline_mape_pct=("baseline_ape_pct", "mean")).sort_values("target_date")
    by_date["delta_pct_points"] = by_date["model_mape_pct"] - by_date["baseline_mape_pct"]
    recent = by_date.tail(5)
    deltas = by_date["delta_pct_points"].to_numpy(dtype=float)
    ci = None
    if len(deltas) >= 5:
        rng = np.random.default_rng(81084)
        samples = rng.choice(deltas, size=(4000, len(deltas)), replace=True).mean(axis=1)
        ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    model = float(x["model_ape_pct"].mean())
    baseline = float(x["baseline_ape_pct"].mean())
    recent_model = float(recent["model_mape_pct"].mean())
    recent_baseline = float(recent["baseline_mape_pct"].mean())
    report = {
        "status": "MODEL_BEATS_BASELINE" if model < baseline else "BASELINE_NOT_BEATEN",
        "scored_rows": int(len(x)), "target_dates": int(len(by_date)),
        "model_mape_pct": model, "baseline_mape_pct": baseline,
        "delta_pct_points": model - baseline, "beats_baseline": bool(model < baseline),
        "recent_sessions": int(len(recent)), "recent_model_mape_pct": recent_model,
        "recent_baseline_mape_pct": recent_baseline,
        "recent_delta_pct_points": recent_model - recent_baseline,
        "recent_beats_baseline": bool(recent_model < recent_baseline),
        "date_block_bootstrap_95pct_ci_delta_pct_points": ci,
        "ci_interpretation": "Negative delta favors the model; interval crossing zero is inconclusive.",
        "same_scored_rows": True, "model_changed": False, "promotion_evidence_only": True
    }
    return report, by_date


def storage_prototype() -> dict[str, Any]:
    """Test gzip round-trip in a temporary directory; never alter source or production readers."""
    path = DATA / "ohlcv.csv"
    if not path.exists():
        return {"status": "SOURCE_MISSING", "prototype_written_to_repo": False, "source_modified": False}
    before_hash = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            before_hash.update(block)
    original_bytes = path.stat().st_size
    try:
        with tempfile.TemporaryDirectory(prefix="stock-picker-phase84-") as temp:
            gz_path = Path(temp) / "ohlcv.csv.gz"
            with path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            raw = pd.read_csv(path, low_memory=False)
            compressed = pd.read_csv(gz_path, compression="gzip", low_memory=False)
            schema_equal = list(raw.columns) == list(compressed.columns)
            rows_equal = len(raw) == len(compressed)
            frame_equal = schema_equal and rows_equal and raw.equals(compressed)
            key_cols = {"date", "symbol"}
            keys_valid = key_cols.issubset(raw.columns)
            duplicates = int(raw.duplicated(["date", "symbol"]).sum()) if keys_valid else None
            compressed_bytes = gz_path.stat().st_size
            savings_pct = (1 - compressed_bytes / original_bytes) * 100 if original_bytes else 0.0
        after_hash = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                after_hash.update(block)
        unchanged = before_hash.hexdigest() == after_hash.hexdigest()
        return {
            "status": "ROUNDTRIP_PARITY_PASS" if frame_equal and unchanged else "ROUNDTRIP_PARITY_REVIEW",
            "source_path": "data/ohlcv.csv", "source_bytes": original_bytes,
            "prototype_compressed_bytes": compressed_bytes,
            "prototype_compression_ratio_pct": round(savings_pct, 2),
            "schema_equal": bool(schema_equal), "row_count_equal": bool(rows_equal),
            "dataframe_roundtrip_equal": bool(frame_equal), "unique_date_symbol_key_columns_present": bool(keys_valid),
            "duplicate_date_symbol_rows": duplicates, "source_sha256_before": before_hash.hexdigest(),
            "source_sha256_after": after_hash.hexdigest(), "source_unchanged": bool(unchanged),
            "prototype_written_to_repo": False, "migration_executed": False,
            "production_reader_switched": False, "rollback_required": True
        }
    except Exception as exc:
        return {"status": "PROTOTYPE_FAILED", "source_path": "data/ohlcv.csv",
                "source_bytes": original_bytes, "source_modified": False,
                "prototype_written_to_repo": False, "migration_executed": False,
                "error_type": type(exc).__name__, "error": str(exc)[:300]}


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    receipt, receipt_rows = receipt_reconciliation()
    v2, funnel = v2_rejection_diagnosis()
    forecast, dates = forecast_evidence()
    storage = storage_prototype()
    receipt_rows.to_csv(DATA / "phase81_receipt_reconciliation.csv", index=False)
    funnel.to_csv(DATA / "phase82_v2_funnel_detail.csv", index=False)
    dates.to_csv(DATA / "phase83_forecast_by_date.csv", index=False)
    report = {
        "phases": [81, 82, 83, 84],
        "phase81_telegram_receipts": receipt,
        "phase82_v2_rejection_diagnosis": v2,
        "phase83_forecast_evidence": forecast,
        "phase84_storage_prototype": storage,
        "safety": {"production_champion": "V1", "v2_mode": "SHADOW_ONLY",
                   "automatic_promotion": False, "ranking_or_risk_changed": False,
                   "telegram_send_logic_changed": False, "production_reader_switched": False,
                   "source_ohlcv_modified": False, "workflow_triggers_changed": False}
    }
    (DATA / "phase81_84_diagnostics.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run()
