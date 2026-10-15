from __future__ import annotations

"""Phase 119: locked prospective validation of forecast improvements.

The evaluation cohort is fixed before new outcomes arrive. Historical sessions
at or before the lock date cannot count as prospective evidence. Analysis only.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
SUMMARY_OUT = DATA / "phase119_prospective_validation.json"
DAILY_OUT = DATA / "phase119_prospective_validation_by_session.csv"
# Frozen after the last target session present when this protocol was introduced.
LOCK_DATE = pd.Timestamp("2026-10-14")
MIN_PROSPECTIVE_SESSIONS = 20
REQUIRED = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}


def _mape(actual: pd.Series, pred: pd.Series) -> float | None:
    valid = actual.notna() & pred.notna() & actual.gt(0)
    if not valid.any():
        return None
    return float((pred[valid] - actual[valid]).abs().div(actual[valid]).mean() * 100)


def evaluate_prospective(scores: pd.DataFrame, lock_date: pd.Timestamp = LOCK_DATE) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED.difference(scores.columns))
    empty = pd.DataFrame()
    if missing:
        return empty, {"status": "INSUFFICIENT_DATA", "missing_columns": missing,
            "lock_date": str(pd.Timestamp(lock_date).date()), "prospective_sessions": 0,
            "production_model_changed": False, "automatic_promotion": False}
    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    for col in ("base_close", "predicted_close", "actual_close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=list(REQUIRED))
    x = x[(x["target_date"] > x["prediction_date"]) & x["base_close"].gt(0) & x["actual_close"].gt(0)]
    x = x.drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
    # Only predictions made strictly after the frozen lock date can enter.
    prospective = x.loc[x["prediction_date"] > pd.Timestamp(lock_date).normalize()].copy()
    daily_rows = []
    for pred_date, batch in prospective.groupby("prediction_date", sort=True):
        daily_rows.append({
            "prediction_date": str(pd.Timestamp(pred_date).date()),
            "target_date": str(batch["target_date"].min().date()),
            "rows": int(len(batch)),
            "model_mape_pct": _mape(batch["actual_close"], batch["predicted_close"]),
            "baseline_mape_pct": _mape(batch["actual_close"], batch["base_close"]),
            "model_beats_baseline": bool(_mape(batch["actual_close"], batch["predicted_close"]) is not None
                and _mape(batch["actual_close"], batch["base_close"]) is not None
                and _mape(batch["actual_close"], batch["predicted_close"]) < _mape(batch["actual_close"], batch["base_close"])),
        })
    daily = pd.DataFrame(daily_rows, columns=["prediction_date", "target_date", "rows", "model_mape_pct", "baseline_mape_pct", "model_beats_baseline"])
    model = _mape(prospective["actual_close"], prospective["predicted_close"]) if not prospective.empty else None
    baseline = _mape(prospective["actual_close"], prospective["base_close"]) if not prospective.empty else None
    sessions = int(prospective["target_date"].nunique())
    enough = sessions >= MIN_PROSPECTIVE_SESSIONS
    summary: dict[str, Any] = {
        "status": "READY_FOR_REVIEW" if enough else ("WAITING_FOR_PROSPECTIVE_SESSIONS" if prospective.empty or sessions < MIN_PROSPECTIVE_SESSIONS else "INSUFFICIENT_DATA"),
        "lock_date": str(pd.Timestamp(lock_date).date()),
        "cohort_rule": "Include only rows whose prediction_date is strictly later than lock_date; cohort membership is not backfilled with pre-lock predictions.",
        "prospective_rows": int(len(prospective)),
        "prospective_symbols": int(prospective["symbol"].nunique()) if not prospective.empty else 0,
        "prospective_sessions": sessions,
        "minimum_sessions_before_review": MIN_PROSPECTIVE_SESSIONS,
        "model_mape_pct": model,
        "previous_close_baseline_mape_pct": baseline,
        "delta_mape_pct_points_model_minus_baseline": round(model - baseline, 6) if model is not None and baseline is not None else None,
        "sessions_model_beats_baseline": int(daily["model_beats_baseline"].sum()) if not daily.empty else 0,
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "promotion_allowed": False,
        "interpretation": "No model change is authorized by this report. A review threshold is not a promotion gate; review must also check data quality, drawdown, and independent evidence.",
    }
    return daily, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    daily, summary = evaluate_prospective(scores)
    daily.to_csv(DAILY_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
