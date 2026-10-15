from __future__ import annotations

"""Phase 117: point-in-time forecast bias correction experiment (analysis only).

At each prediction date, estimate pooled return bias from realized outcomes whose
target date is strictly earlier than the prediction date. Correct the raw model
return by subtracting that historical bias. No production forecast is changed.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
DAILY_OUT = DATA / "phase117_bias_correction_by_session.csv"
SUMMARY_OUT = DATA / "phase117_bias_correction_summary.json"
MIN_TRAIN_ROWS = 30
MIN_TRAIN_SESSIONS = 5


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & a.ne(0)
    if not valid.any():
        return None
    return float((p[valid] - a[valid]).abs().div(a[valid].abs()).mean() * 100.0)


def evaluate_bias_correction(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "missing_columns": missing, "scored_rows": 0,
            "strict_no_lookahead": True, "production_model_changed": False,
            "production_champion": "V1", "v2_mode": "SHADOW_ONLY", "automatic_promotion": False,
        }

    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    for col in ("base_close", "predicted_close", "actual_close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=list(required)).copy()
    x = x[(x["target_date"] > x["prediction_date"]) & x["base_close"].gt(0) & x["actual_close"].gt(0)]
    duplicate_rows = int(x.duplicated(["prediction_date", "target_date", "symbol"], keep="last").sum())
    x = x.drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
    x["raw_return"] = x["predicted_close"] / x["base_close"] - 1.0
    x["actual_return"] = x["actual_close"] / x["base_close"] - 1.0
    x["realized_return_bias"] = x["raw_return"] - x["actual_return"]

    output = []
    for pred_date, batch in x.groupby("prediction_date", sort=True):
        # Strict as-of condition: only labels already realized before prediction.
        prior = x.loc[x["target_date"] < pred_date]
        train_sessions = int(prior["target_date"].nunique())
        enough = len(prior) >= MIN_TRAIN_ROWS and train_sessions >= MIN_TRAIN_SESSIONS
        bias = float(prior["realized_return_bias"].mean()) if enough else 0.0
        corrected = batch["base_close"] * (1.0 + batch["raw_return"] - bias)
        baseline_mape = _mape(batch["actual_close"], batch["base_close"])
        raw_mape = _mape(batch["actual_close"], batch["predicted_close"])
        corrected_mape = _mape(batch["actual_close"], corrected)
        output.append({
            "prediction_date": str(pd.Timestamp(pred_date).date()),
            "target_date": str(pd.to_datetime(batch["target_date"]).min().date()),
            "rows": int(len(batch)),
            "training_rows": int(len(prior)),
            "training_sessions": train_sessions,
            "latest_training_target_date": str(prior["target_date"].max().date()) if not prior.empty else None,
            "correction_applied": bool(enough),
            "estimated_return_bias_pct": bias * 100.0,
            "baseline_mape_pct": baseline_mape,
            "raw_model_mape_pct": raw_mape,
            "bias_corrected_mape_pct": corrected_mape,
            "corrected_beats_baseline": bool(corrected_mape is not None and baseline_mape is not None and corrected_mape < baseline_mape),
        })

    daily = pd.DataFrame(output).sort_values("prediction_date").reset_index(drop=True)
    x["correction"] = x["prediction_date"].map({
        r["prediction_date"]: r["estimated_return_bias_pct"] / 100.0 for r in output
    })
    x["bias_corrected_close"] = x["base_close"] * (1.0 + x["raw_return"] - x["correction"])
    baseline = _mape(x["actual_close"], x["base_close"])
    raw = _mape(x["actual_close"], x["predicted_close"])
    corrected = _mape(x["actual_close"], x["bias_corrected_close"])
    improvement = (baseline - corrected) / baseline * 100.0 if baseline and corrected is not None else None
    summary: dict[str, Any] = {
        "status": "SCORED" if len(x) >= 20 else ("SMALL_SAMPLE" if len(x) else "INSUFFICIENT_DATA"),
        "scored_rows": int(len(x)),
        "target_sessions": int(x["target_date"].nunique()),
        "prediction_sessions": int(x["prediction_date"].nunique()),
        "duplicate_rows_excluded": duplicate_rows,
        "baseline_close_mape_pct": baseline,
        "raw_model_close_mape_pct": raw,
        "walk_forward_bias_corrected_close_mape_pct": corrected,
        "relative_improvement_vs_previous_close_pct": improvement,
        "sessions_with_correction": int(daily["correction_applied"].sum()),
        "sessions_without_enough_training_history": int((~daily["correction_applied"]).sum()),
        "sessions_corrected_model_beats_baseline": int(daily["corrected_beats_baseline"].sum()),
        "correction_method": "Subtract pooled mean raw-return minus realized-return bias estimated only from rows whose target_date is strictly before prediction_date; no hyperparameter search.",
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "minimum_training_sessions": MIN_TRAIN_SESSIONS,
        "strict_no_lookahead": bool((daily["latest_training_target_date"].dropna().map(pd.Timestamp) < daily.loc[daily["latest_training_target_date"].notna(), "prediction_date"].map(pd.Timestamp)).all()),
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "promotion_allowed": False,
        "interpretation": "Research-only single-holdout result. Any apparent gain must persist across additional untouched sessions before promotion consideration.",
    }
    return daily, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    daily, summary = evaluate_bias_correction(scores)
    daily.to_csv(DAILY_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
