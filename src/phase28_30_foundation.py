from __future__ import annotations

"""Phases 28-30: data integrity, historical dataset audit, and accuracy evidence.

These diagnostics are analysis-only. They do not retrain models, change rankings,
modify risk settings, select trades, or promote V2 over the V1 production champion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVALUATIONS = "evaluations.csv"
CANDIDATES = "prediction_candidates_history.csv"
OHLC_FIELDS = ("open", "high", "low", "close")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed legacy date formats consistently, including older pandas versions."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        # Per-value parsing avoids pandas' older first-value format inference
        # turning later valid values into NaT when formats differ.
        return values.map(lambda value: pd.to_datetime(value, errors="coerce"))


def read_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _symbols(frame: pd.DataFrame) -> pd.Series:
    if "symbol" not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame["symbol"].fillna("").astype(str).str.upper().str.strip()


def _fingerprint(name: str) -> dict:
    path = DATA / name
    if not path.exists():
        return {"exists": False, "sha256": None, "bytes": 0}
    raw = path.read_bytes()
    return {"exists": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def data_quality_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Return explicit, reproducible data-quality checks without changing source rows."""
    rows: list[dict] = []

    def add(dataset: str, check: str, status: str, affected: int, detail: str) -> None:
        rows.append({"dataset": dataset, "check": check, "status": status,
                     "affected_rows": int(affected), "detail": detail})

    for name, frame, date_col, key_date in [
        ("evaluations", evaluations, "prediction_date", "prediction_date"),
        ("candidates", candidates, "prediction_date", "target_date"),
    ]:
        if frame.empty:
            add(name, "input_available", "BLOCKED", 0, "input file is missing or empty")
            continue
        add(name, "input_available", "PASS", 0, f"{len(frame)} rows loaded")
        if "symbol" not in frame:
            add(name, "symbol_column", "BLOCKED", len(frame), "required symbol column is missing")
            continue
        symbol = _symbols(frame)
        invalid_symbol = int(symbol.isin(["", "NAN", "NONE", "NULL"]).sum())
        add(name, "symbol_values", "WARN" if invalid_symbol else "PASS", invalid_symbol,
            "blank or null-like symbols after normalization")
        if date_col not in frame:
            add(name, "prediction_date_column", "BLOCKED", len(frame),
                f"required {date_col} column is missing")
            continue
        dates = parse_dates(frame[date_col])
        invalid_dates = int(dates.isna().sum())
        add(name, "prediction_dates_parseable", "WARN" if invalid_dates else "PASS",
            invalid_dates, f"{invalid_dates}/{len(frame)} {date_col} values are invalid")
        if key_date not in frame:
            add(name, "business_date_column", "BLOCKED", len(frame),
                f"required business key date {key_date} is missing")
            continue
        business_dates = parse_dates(frame[key_date]).dt.normalize()
        invalid_business = int((business_dates.isna() | symbol.isin(["", "NAN", "NONE", "NULL"])).sum())
        valid = pd.DataFrame({"date": business_dates, "symbol": symbol}, index=frame.index)
        valid = valid.loc[business_dates.notna() & ~symbol.isin(["", "NAN", "NONE", "NULL"])]
        dup_rows = int(valid.duplicated(["date", "symbol"], keep=False).sum())
        excess = int(valid.duplicated(["date", "symbol"]).sum())
        add(name, "business_keys_valid", "WARN" if invalid_business else "PASS",
            invalid_business, f"{invalid_business}/{len(frame)} rows have invalid {key_date}/symbol keys")
        add(name, "business_keys_unique", "WARN" if dup_rows else "PASS", dup_rows,
            f"{excess} excess rows across duplicate valid {key_date}/symbol keys; invalid keys excluded")

    if not candidates.empty:
        if {"prediction_date", "target_date"}.issubset(candidates.columns):
            prediction = parse_dates(candidates["prediction_date"]).dt.normalize()
            target = parse_dates(candidates["target_date"]).dt.normalize()
            invalid_order = int((prediction.notna() & target.notna() & (prediction >= target)).sum())
            missing_order = int((prediction.isna() | target.isna()).sum())
            add("candidates", "prediction_precedes_target",
                "WARN" if invalid_order or missing_order else "PASS", invalid_order + missing_order,
                f"{invalid_order} rows have prediction_date >= target_date; {missing_order} rows have an unparseable date")
        else:
            add("candidates", "prediction_precedes_target", "BLOCKED", len(candidates),
                "prediction_date and target_date are required")

    if not evaluations.empty:
        if {"prediction_date", "target_date"}.issubset(evaluations.columns):
            prediction = parse_dates(evaluations["prediction_date"]).dt.normalize()
            target = parse_dates(evaluations["target_date"]).dt.normalize()
            invalid_order = int((prediction.notna() & target.notna() & (prediction >= target)).sum())
            missing_order = int((prediction.isna() | target.isna()).sum())
            add("evaluations", "prediction_precedes_target",
                "WARN" if invalid_order or missing_order else "PASS", invalid_order + missing_order,
                f"{invalid_order} rows have prediction_date >= target_date; {missing_order} rows have an unparseable date")
        else:
            add("evaluations", "prediction_precedes_target", "BLOCKED", len(evaluations),
                "prediction_date and target_date are required")
        for field in OHLC_FIELDS:
            predicted, actual = f"predicted_{field}", f"actual_{field}"
            if predicted not in evaluations or actual not in evaluations:
                add("evaluations", f"{field}_price_columns", "REVIEW", len(evaluations),
                    f"expected {predicted} and {actual} columns")
                continue
            p = pd.to_numeric(evaluations[predicted], errors="coerce")
            a = pd.to_numeric(evaluations[actual], errors="coerce")
            bad = int((p.isna() | a.isna() | ~np.isfinite(p) | ~np.isfinite(a)).sum())
            nonpositive_actual = int((a.notna() & (a <= 0)).sum())
            add("evaluations", f"{field}_price_values", "WARN" if bad or nonpositive_actual else "PASS",
                bad + nonpositive_actual,
                f"{bad} missing/non-finite predicted or actual values; {nonpositive_actual} non-positive actual prices")
    return pd.DataFrame(rows, columns=["dataset", "check", "status", "affected_rows", "detail"])


def historical_dataset_audit(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize historical coverage and key quality for both persisted datasets."""
    rows = []
    specs = [
        ("evaluations", evaluations, "prediction_date", "target_date", "prediction_date"),
        ("prediction_candidates_history", candidates, "prediction_date", "target_date", "target_date"),
    ]
    for name, frame, pred_col, target_col, key_col in specs:
        if frame.empty:
            rows.append({"dataset": name, "rows": 0, "symbols": 0,
                         "prediction_date_coverage_pct": np.nan, "target_date_coverage_pct": np.nan,
                         "valid_business_key_rows": 0, "duplicate_business_key_rows": 0,
                         "first_prediction_date": None, "latest_prediction_date": None,
                         "first_target_date": None, "latest_target_date": None,
                         "audit_status": "BLOCKED"})
            continue
        pred = parse_dates(frame[pred_col]) if pred_col in frame else pd.Series(pd.NaT, index=frame.index)
        target = parse_dates(frame[target_col]) if target_col in frame else pd.Series(pd.NaT, index=frame.index)
        symbols = _symbols(frame)
        key_dates = parse_dates(frame[key_col]).dt.normalize() if key_col in frame else pd.Series(pd.NaT, index=frame.index)
        valid_mask = key_dates.notna() & ~symbols.isin(["", "NAN", "NONE", "NULL"])
        keys = pd.DataFrame({"date": key_dates, "symbol": symbols}, index=frame.index).loc[valid_mask]
        duplicate_rows = int(keys.duplicated(["date", "symbol"], keep=False).sum())
        valid_rows = int(valid_mask.sum())
        rows.append({
            "dataset": name, "rows": int(len(frame)), "symbols": int(symbols[~symbols.isin(["", "NAN", "NONE", "NULL"])].nunique()),
            "prediction_date_coverage_pct": float(pred.notna().mean() * 100),
            "target_date_coverage_pct": float(target.notna().mean() * 100),
            "valid_business_key_rows": valid_rows, "duplicate_business_key_rows": duplicate_rows,
            "first_prediction_date": str(pred.min().date()) if pred.notna().any() else None,
            "latest_prediction_date": str(pred.max().date()) if pred.notna().any() else None,
            "first_target_date": str(target.min().date()) if target.notna().any() else None,
            "latest_target_date": str(target.max().date()) if target.notna().any() else None,
            "audit_status": (
                "PASS" if pred.notna().all() and target.notna().all()
                and valid_rows == len(frame) and duplicate_rows == 0 else "REVIEW"
            ),
        })
    return pd.DataFrame(rows)


def accuracy_metrics(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Calculate price-error metrics for persisted forecasts versus actuals and baseline.

    Existing evaluation rows are treated as historical forecasts; this function does
    not fit a model or claim that the underlying predictions came from a holdout split.
    """
    columns = ["scope", "target_date", "field", "n", "mae", "mape_pct",
               "baseline_mape_pct", "mape_improvement_pct", "direction_accuracy_pct", "evidence_status"]
    if evaluations.empty:
        return pd.DataFrame(columns=columns)
    frame = evaluations.copy()
    if "target_date" in frame:
        frame["target_date"] = parse_dates(frame["target_date"]).dt.strftime("%Y-%m-%d")
    else:
        frame["target_date"] = ""
    rows = []
    groups = [("overall", "", frame)]
    groups.extend(( "target_date", str(day), group) for day, group in frame.groupby("target_date", dropna=False))
    for scope, day, group in groups:
        for field in OHLC_FIELDS:
            predicted_col, actual_col = f"predicted_{field}", f"actual_{field}"
            if predicted_col not in group or actual_col not in group:
                continue
            predicted = pd.to_numeric(group[predicted_col], errors="coerce")
            actual = pd.to_numeric(group[actual_col], errors="coerce")
            valid = predicted.notna() & actual.notna() & np.isfinite(predicted) & np.isfinite(actual)
            valid &= actual.abs().gt(0)
            n = int(valid.sum())
            if n:
                mae = float((predicted[valid] - actual[valid]).abs().mean())
                mape = float(((predicted[valid] - actual[valid]).abs() / actual[valid].abs()).mean() * 100)
            else:
                mae, mape = np.nan, np.nan
            baseline_col = f"baseline_{field}_abs_pct_error"
            baseline = pd.to_numeric(group[baseline_col], errors="coerce") if baseline_col in group else pd.Series(np.nan, index=group.index)
            baseline = baseline[valid & baseline.notna() & np.isfinite(baseline)]
            baseline_mape = float(baseline.mean() * 100) if len(baseline) else np.nan
            improvement = ((baseline_mape - mape) / baseline_mape * 100
                           if np.isfinite(baseline_mape) and baseline_mape > 0 and np.isfinite(mape) else np.nan)
            direction = np.nan
            if field == "close" and {"predicted_close_direction", "actual_close_direction"}.issubset(group.columns):
                pdirection = pd.to_numeric(group.loc[valid, "predicted_close_direction"], errors="coerce")
                adirection = pd.to_numeric(group.loc[valid, "actual_close_direction"], errors="coerce")
                dvalid = pdirection.notna() & adirection.notna()
                if dvalid.any():
                    direction = float((pdirection[dvalid] == adirection[dvalid]).mean() * 100)
            rows.append({
                "scope": scope, "target_date": day, "field": field, "n": n, "mae": mae,
                "mape_pct": mape, "baseline_mape_pct": baseline_mape,
                "mape_improvement_pct": improvement, "direction_accuracy_pct": direction,
                "evidence_status": "INSUFFICIENT_SAMPLE" if n < 30 else "DESCRIPTIVE_ONLY",
            })
    return pd.DataFrame(rows, columns=columns)


def run() -> dict:
    DATA.mkdir(exist_ok=True)
    evaluations = read_csv(EVALUATIONS)
    candidates = read_csv(CANDIDATES)
    quality = data_quality_report(evaluations, candidates)
    audit = historical_dataset_audit(evaluations, candidates)
    metrics = accuracy_metrics(evaluations)
    quality.to_csv(DATA / "phase28_data_quality.csv", index=False)
    audit.to_csv(DATA / "phase29_dataset_audit.csv", index=False)
    metrics.to_csv(DATA / "phase30_accuracy_metrics.csv", index=False)
    summary = {
        "phases": [28, 29, 30],
        "evaluation_rows": int(len(evaluations)),
        "candidate_rows": int(len(candidates)),
        "quality_checks": int(len(quality)),
        "quality_warnings_or_blocks": int(quality["status"].isin(["WARN", "BLOCKED"]).sum()) if not quality.empty else 1,
        "datasets_needing_review": int((audit["audit_status"] != "PASS").sum()) if not audit.empty else 2,
        "accuracy_metric_rows": int(len(metrics)),
        "inputs": {EVALUATIONS: _fingerprint(EVALUATIONS), CANDIDATES: _fingerprint(CANDIDATES)},
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "v2_promoted": False,
        "interpretation": "Diagnostics only; accuracy metrics are descriptive and not proof of out-of-sample performance.",
    }
    (DATA / "phase28_30_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
