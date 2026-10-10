from __future__ import annotations

"""Phase 27: targeted data-integrity and accuracy-recovery diagnostics.

Reports are analysis-only. This module never changes ranking, model weights,
risk settings, production champion, or trading decisions.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed legacy date formats consistently across pandas versions."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(name: str) -> pd.DataFrame:
    try:
        path = DATA / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _keys(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    x = frame.copy()
    x[date_col] = _parse_dates(x[date_col]).dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    return x


def _candidate_date_column(frame: pd.DataFrame) -> str | None:
    """Prefer prediction date, but use target date for legacy rows with gaps."""
    for col in ("prediction_date", "target_date"):
        if col in frame.columns and _parse_dates(frame[col]).notna().any():
            return col
    return None


def _extract_regime(row: pd.Series) -> str | None:
    for col in ("index_market_regime", "market_regime", "regime"):
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().upper()
    explanation = row.get("selection_explanation")
    if isinstance(explanation, str) and explanation.strip():
        try:
            parsed = json.loads(explanation)
            value = parsed.get("regime") if isinstance(parsed, dict) else None
            if value and str(value).strip():
                return str(value).strip().upper()
        except (ValueError, TypeError):
            return None
    return None


def integrity_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Quantify usable keys, duplicate business keys, confidence coverage and leakage clues."""
    rows = []

    def add(check: str, status: str, count: int, detail: str) -> None:
        rows.append({"check": check, "status": status, "affected_rows": int(count), "detail": detail})

    if evaluations.empty:
        add("evaluation_data", "BLOCKED", 0, "input file is missing or empty")
    elif not {"prediction_date", "symbol"}.issubset(evaluations.columns):
        add("evaluation_key_columns", "BLOCKED", 0, "prediction_date and symbol are required for reliable joins")
    else:
        e = _keys(evaluations, "prediction_date")
        invalid = int((e["prediction_date"].isna() | e["symbol"].isin(["", "NAN", "NONE"])).sum())
        duplicate_rows = int(e.duplicated(["prediction_date", "symbol"], keep=False).sum())
        excess = int(e.duplicated(["prediction_date", "symbol"]).sum())
        add("evaluation_invalid_keys", "WARN" if invalid else "PASS", invalid,
            "rows with invalid prediction_date or blank symbol")
        add("evaluation_duplicate_key_rows", "WARN" if duplicate_rows else "PASS", duplicate_rows,
            f"{excess} excess rows across prediction_date/symbol keys")

    if candidates.empty:
        add("candidate_history_data", "BLOCKED", 0, "input file is missing or empty")
    else:
        date_col = _candidate_date_column(candidates)
        if date_col is None or "symbol" not in candidates:
            add("candidate_history_key_columns", "BLOCKED", 0,
                "no usable prediction_date/target_date and symbol columns")
        else:
            c = _keys(candidates, date_col)
            invalid_prediction = int((c[date_col].isna() | c["symbol"].isin(["", "NAN", "NONE"])).sum())
            # Pipeline history is intentionally unique by target_date + symbol.
            business_date_col = "target_date" if "target_date" in c.columns else date_col
            c[business_date_col] = _parse_dates(c[business_date_col]).dt.normalize()
            invalid_business_mask = c[business_date_col].isna() | c["symbol"].isin(["", "NAN", "NONE"])
            invalid_business = int(invalid_business_mask.sum())
            # Never count malformed rows as duplicate business keys: NaT/blank values
            # are data-quality failures, not evidence of repeated candidate records.
            valid_business = c.loc[~invalid_business_mask]
            dup_mask = valid_business.duplicated([business_date_col, "symbol"], keep=False)
            dup_rows = int(dup_mask.sum())
            excess = int(valid_business.duplicated([business_date_col, "symbol"]).sum())
            add("candidate_history_invalid_keys", "WARN" if invalid_business else "PASS", invalid_business,
                f"{invalid_business}/{len(c)} rows have an invalid {business_date_col}/symbol business key; "
                f"{invalid_prediction} rows have an invalid {date_col}/symbol prediction key")
            add("candidate_history_duplicate_key_rows", "WARN" if dup_rows else "PASS", dup_rows,
                f"{excess} excess rows among valid {business_date_col}/symbol keys; invalid keys excluded")

        confidence_col = next((c for c in (
            "adjusted_confidence_score", "confidence_score", "confidence"
        ) if c in candidates.columns), None)
        if confidence_col:
            vals = pd.to_numeric(candidates[confidence_col], errors="coerce")
            nonmissing = vals.dropna()
            if len(nonmissing) and nonmissing.between(0, 1).mean() >= 0.8:
                vals = vals * 100
            date_col = _candidate_date_column(candidates)
            if date_col:
                dates = _parse_dates(candidates[date_col])
                latest = dates.max()
                current_mask = dates.eq(latest)
            else:
                current_mask = pd.Series(True, index=candidates.index)
            current_missing = int(vals[current_mask].isna().sum())
            legacy_missing = int(vals[~current_mask].isna().sum())
            out_of_range = int((vals.notna() & ~vals.between(0, 100)).sum())
            add("candidate_confidence_current_cohort", "WARN" if current_missing else "PASS", current_missing,
                f"{confidence_col}: {current_missing}/{int(current_mask.sum())} missing in latest date cohort; "
                f"{legacy_missing} missing in older rows")
            add("candidate_confidence_range", "WARN" if out_of_range else "PASS", out_of_range,
                "confidence should be on a 0-100 scale after normalization")
        else:
            add("candidate_confidence_current_cohort", "WARN", len(candidates),
                "no recognized confidence column exists in candidate history")

        suspicious = [col for col in candidates.columns if col.lower() != "target_date" and re.search(
            r"(actual|target|future|next_day|forward_return|label)", col, re.I
        )]
        add("candidate_future_named_fields", "REVIEW" if suspicious else "PASS", len(suspicious),
            "field names require manual provenance review: " + ", ".join(suspicious)
            if suspicious else "no obviously future/label-named fields found")

        pred_col = "prediction_timestamp" if "prediction_timestamp" in candidates else (
            "created_at" if "created_at" in candidates else None
        )
        if pred_col:
            pred_ts = pd.to_datetime(candidates[pred_col], errors="coerce", utc=True)
            bad_ts = int(pred_ts.isna().sum())
            add("candidate_prediction_time_coverage", "WARN" if bad_ts else "PASS", bad_ts,
                f"{pred_col}: missing/unparseable timestamps prevent precise as-of checks")
            news_cols = [col for col in candidates.columns if re.search(
                r"(news|article).*(published|timestamp|datetime)|(published|timestamp|datetime).*(news|article)", col, re.I
            )]
            future_news = 0
            for col in news_cols:
                published = pd.to_datetime(candidates[col], errors="coerce", utc=True)
                future_news += int((published.notna() & pred_ts.notna() & (published > pred_ts)).sum())
            add("news_timestamp_after_prediction", "WARN" if future_news else "PASS", future_news,
                "news publication timestamps later than prediction timestamp; only verifiable where timestamp columns exist")
        else:
            add("candidate_prediction_time_coverage", "REVIEW", len(candidates),
                "no prediction_timestamp/created_at field available for as-of verification")
    return pd.DataFrame(rows, columns=["check", "status", "affected_rows", "detail"])


def regime_accuracy_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Describe model errors by recorded regime; no regime-specific tuning."""
    columns = ["market_regime", "rows", "sessions", "direction_accuracy_pct",
               "close_mape_pct", "baseline_mape_pct", "mape_improvement_pct", "evidence_status"]
    if evaluations.empty or candidates.empty:
        return pd.DataFrame(columns=columns)
    if not {"prediction_date", "symbol"}.issubset(evaluations.columns) or "symbol" not in candidates:
        return pd.DataFrame(columns=columns)
    e = _keys(evaluations, "prediction_date")
    candidate_date = _candidate_date_column(candidates)
    if candidate_date is None:
        return pd.DataFrame(columns=columns)
    c = _keys(candidates, candidate_date)
    c["market_regime"] = candidates.apply(_extract_regime, axis=1)
    c["prediction_date"] = _parse_dates(candidates.get("prediction_date", candidates[candidate_date])).dt.normalize()
    if candidate_date == "target_date" and "prediction_date" not in candidates:
        c["prediction_date"] = _parse_dates(candidates[candidate_date]).dt.normalize()
    c["symbol"] = candidates["symbol"].astype(str).str.upper().str.strip()
    # Preserve UNKNOWN rows when no verified regime is recorded; the report
    # should make the evidence gap visible rather than silently omit it.
    metric_cols = ["close_direction_correct", "close_abs_pct_error", "baseline_close_abs_pct_error"]
    for col in metric_cols:
        if col not in e:
            e[col] = np.nan
        e[col] = pd.to_numeric(e[col], errors="coerce")
    join_date = "prediction_date" if "prediction_date" in candidates else candidate_date
    c["join_date"] = _parse_dates(candidates[join_date]).dt.normalize()
    # Evaluation prediction_date corresponds to the date the forecast was made.
    merged = e.merge(c[["join_date", "symbol", "market_regime"]].drop_duplicates(["join_date", "symbol"], keep="last"),
                     left_on=["prediction_date", "symbol"], right_on=["join_date", "symbol"], how="left")
    merged["market_regime"] = merged["market_regime"].fillna("UNKNOWN").astype(str).str.upper()
    out = []
    for regime, g in merged.groupby("market_regime", dropna=False):
        direction = g["close_direction_correct"].dropna()
        model = g["close_abs_pct_error"].dropna() * 100
        baseline = g["baseline_close_abs_pct_error"].dropna() * 100
        m = float(model.mean()) if len(model) else np.nan
        b = float(baseline.mean()) if len(baseline) else np.nan
        sessions = int(g["prediction_date"].nunique())
        out.append({
            "market_regime": str(regime), "rows": int(len(g)), "sessions": sessions,
            "direction_accuracy_pct": float(direction.mean() * 100) if len(direction) else np.nan,
            "close_mape_pct": m, "baseline_mape_pct": b,
            "mape_improvement_pct": float((b - m) / b * 100) if np.isfinite(m) and np.isfinite(b) and b > 0 else np.nan,
            # UNKNOWN is not a verified market regime and must never be treated
            # as evidence for regime-specific conclusions.
            "evidence_status": (
                "UNKNOWN_REGIME" if str(regime).upper() in {"UNKNOWN", "", "NAN", "NONE"}
                else "EVIDENCE_ONLY" if sessions >= 10 and len(g) >= 30
                else "INSUFFICIENT_SAMPLE"
            ),
        })
    return pd.DataFrame(out, columns=columns).sort_values("market_regime")


def run() -> dict:
    DATA.mkdir(exist_ok=True)
    evaluations = read_csv("evaluations.csv")
    candidates = read_csv("prediction_candidates_history.csv")
    integrity = integrity_report(evaluations, candidates)
    regimes = regime_accuracy_report(evaluations, candidates)
    integrity.to_csv(DATA / "phase27_data_integrity.csv", index=False)
    regimes.to_csv(DATA / "phase27_regime_accuracy.csv", index=False)
    warnings = int(integrity["status"].isin(["WARN", "BLOCKED", "REVIEW"]).sum()) if not integrity.empty else 1
    def input_fingerprint(name: str) -> dict:
        path = DATA / name
        if not path.exists():
            return {"exists": False, "sha256": None, "bytes": 0}
        raw = path.read_bytes()
        return {"exists": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    summary = {
        "phase": 27,
        "as_of": str(pd.Timestamp.now().date()),
        "evaluation_rows": int(len(evaluations)),
        "candidate_rows": int(len(candidates)),
        "inputs": {
            "data/evaluations.csv": input_fingerprint("evaluations.csv"),
            "data/prediction_candidates_history.csv": input_fingerprint("prediction_candidates_history.csv"),
        },
        "integrity_checks": int(len(integrity)),
        "warnings_or_blocks": warnings,
        "regime_buckets": int(len(regimes)),
        "regime_buckets_with_sufficient_evidence": int((regimes["evidence_status"] == "EVIDENCE_ONLY").sum()) if not regimes.empty else 0,
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "production_champion": "V1",
        "reports": ["data/phase27_data_integrity.csv", "data/phase27_regime_accuracy.csv"],
        "note": "Diagnostics are descriptive and analysis-only. Input SHA-256 fingerprints identify the exact source files; UNKNOWN regimes never count as verified evidence. Legacy confidence gaps are reported separately from the latest cohort.",
    }
    (DATA / "phase27_data_quality_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    run()
