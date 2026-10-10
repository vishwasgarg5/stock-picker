from __future__ import annotations

"""Phase 27: targeted data-integrity and accuracy-recovery diagnostics.

Reports are analysis-only. This module never changes ranking, model weights,
risk settings, production champion, or trading decisions.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    return x


def integrity_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Quantify duplicate keys, confidence coverage, and suspicious future fields."""
    rows = []

    def add(check: str, status: str, count: int, detail: str) -> None:
        rows.append({"check": check, "status": status, "affected_rows": int(count), "detail": detail})

    for label, frame in (("evaluation", evaluations), ("candidate_history", candidates)):
        if frame.empty:
            add(f"{label}_data", "BLOCKED", 0, "input file is missing or empty")
            continue
        if not {"prediction_date", "symbol"}.issubset(frame.columns):
            add(f"{label}_key_columns", "BLOCKED", 0, "prediction_date and symbol are required for reliable joins")
            continue
        x = _keys(frame, "prediction_date")
        invalid = int((x["prediction_date"].isna() | x["symbol"].isin(["", "NAN", "NONE"])).sum())
        duplicate_mask = x.duplicated(["prediction_date", "symbol"], keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_excess = int(x.duplicated(["prediction_date", "symbol"]).sum())
        add(f"{label}_invalid_keys", "WARN" if invalid else "PASS", invalid,
            "rows with invalid prediction_date or blank symbol")
        add(f"{label}_duplicate_key_rows", "WARN" if duplicate_rows else "PASS", duplicate_rows,
            f"{duplicate_excess} excess rows across duplicate prediction_date/symbol keys; inspect before deduplicating")

    if not candidates.empty:
        confidence_col = next((c for c in (
            "adjusted_confidence_score", "confidence_score", "confidence"
        ) if c in candidates.columns), None)
        if confidence_col:
            vals = pd.to_numeric(candidates[confidence_col], errors="coerce")
            nonmissing = vals.dropna()
            if len(nonmissing) and nonmissing.between(0, 1).mean() >= 0.8:
                vals = vals * 100
            missing = int(vals.isna().sum())
            out_of_range = int((vals.notna() & ~vals.between(0, 100)).sum())
            add("candidate_confidence_coverage", "WARN" if missing else "PASS", missing,
                f"{confidence_col}: {missing}/{len(candidates)} missing or nonnumeric values")
            add("candidate_confidence_range", "WARN" if out_of_range else "PASS", out_of_range,
                "confidence should be on a 0-100 scale after normalization")
        else:
            add("candidate_confidence_coverage", "WARN", len(candidates),
                "no recognized confidence column exists in candidate history")

        suspicious = [c for c in candidates.columns if c.lower() != "target_date" and re.search(
            r"(actual|target|future|next_day|forward_return|label)", c, re.I
        )]
        add("candidate_future_named_fields", "REVIEW" if suspicious else "PASS", len(suspicious),
            "field names require manual provenance review: " + ", ".join(suspicious)
            if suspicious else "no obviously future/label-named fields found")

        if "prediction_timestamp" in candidates:
            pred_ts = pd.to_datetime(candidates["prediction_timestamp"], errors="coerce", utc=True)
            timestamp_bad = int(pred_ts.isna().sum())
            add("prediction_timestamp_coverage", "WARN" if timestamp_bad else "PASS", timestamp_bad,
                "missing/unparseable prediction_timestamp prevents precise as-of leakage verification")
            news_cols = [c for c in candidates.columns if re.search(
                r"(news|article).*(published|timestamp|datetime)|(published|timestamp|datetime).*(news|article)", c, re.I
            )]
            future_news = 0
            for col in news_cols:
                published = pd.to_datetime(candidates[col], errors="coerce", utc=True)
                future_news += int((published.notna() & pred_ts.notna() & (published > pred_ts)).sum())
            add("news_timestamp_after_prediction", "WARN" if future_news else "PASS", future_news,
                "news publication timestamps later than prediction_timestamp")
    return pd.DataFrame(rows, columns=["check", "status", "affected_rows", "detail"])


def regime_accuracy_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Describe model errors by available market regime; no regime-specific tuning."""
    columns = ["market_regime", "rows", "sessions", "direction_accuracy_pct",
               "close_mape_pct", "baseline_mape_pct", "mape_improvement_pct", "evidence_status"]
    if evaluations.empty or candidates.empty:
        return pd.DataFrame(columns=columns)
    required = {"prediction_date", "symbol"}
    if not required.issubset(evaluations.columns) or not required.issubset(candidates.columns):
        return pd.DataFrame(columns=columns)
    e = _keys(evaluations, "prediction_date")
    c = _keys(candidates, "prediction_date")
    regime_col = next((cname for cname in ("index_market_regime", "market_regime", "regime") if cname in c), None)
    if regime_col is None:
        return pd.DataFrame(columns=columns)
    metric_cols = ["close_direction_correct", "close_abs_pct_error", "baseline_close_abs_pct_error"]
    for col in metric_cols:
        if col not in e:
            e[col] = np.nan
        e[col] = pd.to_numeric(e[col], errors="coerce")
    c = c[["prediction_date", "symbol", regime_col]].drop_duplicates(["prediction_date", "symbol"], keep="last")
    merged = e.merge(c, on=["prediction_date", "symbol"], how="left")
    merged[regime_col] = merged[regime_col].fillna("UNKNOWN").astype(str).str.upper()
    out = []
    for regime, g in merged.groupby(regime_col, dropna=False):
        direction = g["close_direction_correct"].dropna()
        model = g["close_abs_pct_error"].dropna() * 100
        baseline = g["baseline_close_abs_pct_error"].dropna() * 100
        m = float(model.mean()) if len(model) else np.nan
        b = float(baseline.mean()) if len(baseline) else np.nan
        out.append({
            "market_regime": str(regime), "rows": int(len(g)),
            "sessions": int(g["prediction_date"].nunique()),
            "direction_accuracy_pct": float(direction.mean() * 100) if len(direction) else np.nan,
            "close_mape_pct": m, "baseline_mape_pct": b,
            "mape_improvement_pct": float((b-m)/b*100) if np.isfinite(m) and np.isfinite(b) and b > 0 else np.nan,
            "evidence_status": "EVIDENCE_ONLY" if g["prediction_date"].nunique() >= 10 and len(g) >= 30 else "INSUFFICIENT_SAMPLE",
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
    warnings = int(integrity["status"].isin(["WARN", "BLOCKED"]).sum()) if not integrity.empty else 1
    summary = {
        "phase": 27,
        "as_of": str(pd.Timestamp.now().date()),
        "evaluation_rows": int(len(evaluations)),
        "candidate_rows": int(len(candidates)),
        "integrity_checks": int(len(integrity)),
        "warnings_or_blocks": warnings,
        "regime_buckets": int(len(regimes)),
        "regime_buckets_with_sufficient_evidence": int((regimes["evidence_status"] == "EVIDENCE_ONLY").sum()) if not regimes.empty else 0,
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "production_champion": "V1",
        "reports": ["data/phase27_data_integrity.csv", "data/phase27_regime_accuracy.csv"],
        "note": "Resolve duplicate and timestamp warnings at source; regime metrics are descriptive and do not change selection.",
    }
    (DATA / "phase27_data_quality_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    run()
