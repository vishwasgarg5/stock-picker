from __future__ import annotations

"""Phase 123: leakage-safe magnitude calibration of saved out-of-sample forecasts.

This is an analysis-only post-processor. It never edits production predictions,
model artifacts, rankings, risk limits, or promotion state.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
SUMMARY_OUT = DATA / "phase123_magnitude_calibration.json"
DAILY_OUT = DATA / "phase123_magnitude_calibration_by_session.csv"
REQUIRED = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}
MIN_TRAIN_ROWS = 30
MIN_TRAIN_SESSIONS = 5
MAX_SCALE = 5.0
MAX_ABS_RETURN = 0.40
MIN_PROSPECTIVE_SESSIONS = 17


def _metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "mape_pct": None, "directional_accuracy_pct": None,
                "signed_bias_pct_points": None, "mean_abs_predicted_move_pct": None,
                "mean_abs_realized_move_pct": None, "magnitude_ratio": None}
    actual = frame["actual_close"].to_numpy(dtype=float)
    base = frame["base_close"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    actual_return = actual / base - 1.0
    predicted_return = predicted / base - 1.0
    return {
        "rows": int(len(frame)),
        "mape_pct": float(np.mean(np.abs(predicted - actual) / actual) * 100.0),
        "directional_accuracy_pct": float(np.mean(np.sign(predicted_return) == np.sign(actual_return)) * 100.0),
        "signed_bias_pct_points": float(np.mean(predicted_return - actual_return) * 100.0),
        "mean_abs_predicted_move_pct": float(np.mean(np.abs(predicted_return)) * 100.0),
        "mean_abs_realized_move_pct": float(np.mean(np.abs(actual_return)) * 100.0),
        "magnitude_ratio": (
            float(np.mean(np.abs(predicted_return)) / np.mean(np.abs(actual_return)) )
            if np.mean(np.abs(actual_return)) > 0 else None
        ),
    }


def evaluate_magnitude_calibration(
    scores: pd.DataFrame,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_train_sessions: int = MIN_TRAIN_SESSIONS,
    max_scale: float = MAX_SCALE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED.difference(scores.columns))
    if missing:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "missing_columns": missing,
                                "production_model_changed": False, "automatic_promotion": False}
    if min_train_rows < 1 or min_train_sessions < 1 or max_scale <= 0:
        raise ValueError("Calibration training requirements and max_scale must be positive")
    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    for column in ["base_close", "predicted_close", "actual_close"]:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.dropna(subset=list(REQUIRED))
    finite = np.isfinite(x[["base_close", "predicted_close", "actual_close"]].to_numpy(dtype=float)).all(axis=1)
    x = x.loc[finite].copy()
    x = x[(x["target_date"] > x["prediction_date"]) & x["base_close"].gt(0)
          & x["predicted_close"].gt(0) & x["actual_close"].gt(0)]
    key_columns = ["prediction_date", "target_date", "symbol"]
    if x.duplicated(key_columns).any():
        raise ValueError("Duplicate prediction_date/target_date/symbol keys; refusing ambiguous calibration")
    x["raw_return"] = x["predicted_close"] / x["base_close"] - 1.0
    x["actual_return"] = x["actual_close"] / x["base_close"] - 1.0
    if not np.isfinite(x[["raw_return", "actual_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite derived return; refusing calibration")
    x = x.sort_values(["prediction_date", "target_date", "symbol"]).reset_index(drop=True)

    daily: list[dict[str, Any]] = []
    scored: list[pd.DataFrame] = []
    for prediction_date, batch in x.groupby("prediction_date", sort=True):
        # Only outcomes fully observed before this prediction date may fit alpha.
        prior = x.loc[x["target_date"] < prediction_date]
        prior_sessions = int(prior["target_date"].nunique())
        enough_history = len(prior) >= min_train_rows and prior_sessions >= min_train_sessions
        alpha = 1.0
        if enough_history:
            p = prior["raw_return"].to_numpy(dtype=float)
            y = prior["actual_return"].to_numpy(dtype=float)
            denominator = float(np.dot(p, p))
            if denominator > 1e-12:
                alpha = float(np.dot(p, y) / denominator)
            alpha = float(np.clip(alpha, 0.0, max_scale))
        part = batch.copy()
        part["raw_forecast_close"] = part["predicted_close"]
        calibrated_return = np.clip(part["raw_return"].to_numpy(dtype=float) * alpha,
                                    -MAX_ABS_RETURN, MAX_ABS_RETURN)
        part["calibrated_forecast_close"] = part["base_close"].to_numpy(dtype=float) * (1.0 + calibrated_return)
        part["baseline_close"] = part["base_close"]
        scored.append(part)
        daily.append({
            "prediction_date": str(prediction_date.date()),
            "target_date": str(batch["target_date"].min().date()),
            "rows": int(len(batch)),
            "training_rows": int(len(prior)),
            "training_target_sessions": prior_sessions,
            "latest_training_target_date": str(prior["target_date"].max().date()) if not prior.empty else None,
            "no_lookahead": bool(prior.empty or (prior["target_date"] < prediction_date).all()),
            "calibration_applied": bool(enough_history),
            "scale_alpha": alpha if enough_history else None,
            "raw_mape_pct": _metrics(part, "raw_forecast_close")["mape_pct"],
            "calibrated_mape_pct": _metrics(part, "calibrated_forecast_close")["mape_pct"],
            "baseline_mape_pct": _metrics(part, "baseline_close")["mape_pct"],
        })
    all_rows = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    if all_rows.empty:
        return pd.DataFrame(daily), {"status": "INSUFFICIENT_DATA", "prospective_sessions": 0,
                                     "production_model_changed": False, "automatic_promotion": False}
    sessions = int(all_rows["target_date"].nunique())
    raw_metrics = _metrics(all_rows, "raw_forecast_close")
    calibrated_metrics = _metrics(all_rows, "calibrated_forecast_close")
    baseline_metrics = _metrics(all_rows, "baseline_close")
    prospective = all_rows.loc[all_rows["prediction_date"] > pd.Timestamp("2026-10-14")]
    summary = {
        "phase": 123,
        "title": "Leakage-safe forecast magnitude calibration",
        "status": "EVALUATED" if sessions else "INSUFFICIENT_DATA",
        "scored_rows": int(len(all_rows)),
        "target_sessions": sessions,
        "prospective_rows_after_2026_10_14": int(len(prospective)),
        "prospective_target_sessions_after_2026_10_14": int(prospective["target_date"].nunique()),
        "minimum_prospective_sessions_for_review": MIN_PROSPECTIVE_SESSIONS,
        "raw_incumbent": raw_metrics,
        "calibrated_incumbent": calibrated_metrics,
        "previous_close_baseline": baseline_metrics,
        "calibration": {
            "method": "single pooled least-squares return scale, fit only on prior saved out-of-sample forecasts whose target_date is strictly earlier than prediction_date",
            "minimum_training_rows": min_train_rows,
            "minimum_training_target_sessions": min_train_sessions,
            "maximum_scale": max_scale,
            "maximum_absolute_calibrated_return": MAX_ABS_RETURN,
            "fallback_without_history": "identity scale (alpha=1)",
        },
        "strict_no_lookahead": bool(pd.Series([row["no_lookahead"] for row in daily]).all()),
        "sessions_with_calibration": int(sum(row["calibration_applied"] for row in daily)),
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "challenger_shadow_only": True,
        "production_champion": "V1",
        "automatic_promotion": False,
        "promotion_allowed": False,
        "decision": "Calibration is research-only. It must beat both raw incumbent and previous-close baseline across untouched chronological folds and fresh prospective sessions before any promotion review.",
    }
    return pd.DataFrame(daily), summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    daily, summary = evaluate_magnitude_calibration(scores)
    daily.to_csv(DAILY_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
