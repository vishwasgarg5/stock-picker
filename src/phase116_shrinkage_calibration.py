from __future__ import annotations

"""Phase 116: strictly chronological baseline-anchored forecast calibration.

This report is analysis-only. It never edits production predictions, ranking,
risk limits, or promotion decisions. The shrinkage coefficient for each
prediction date is chosen only from outcomes known before that prediction date.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
DAILY_OUT = DATA / "phase116_shrinkage_calibration_by_session.csv"
SUMMARY_OUT = DATA / "phase116_shrinkage_calibration_summary.json"
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
MIN_TRAIN_SESSIONS = 5
MIN_TRAIN_ROWS = 30


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & a.ne(0) & p.notna()
    if not valid.any():
        return None
    return float((p[valid] - a[valid]).abs().div(a[valid].abs()).mean() * 100.0)


def evaluate_shrinkage(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "missing_columns": missing, "scored_rows": 0,
            "production_model_changed": False, "production_champion": "V1",
            "v2_mode": "SHADOW_ONLY", "automatic_promotion": False,
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
    x = x.sort_values(["prediction_date", "symbol", "target_date"]).reset_index(drop=True)
    if x.empty:
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "missing_columns": [], "scored_rows": 0,
            "production_model_changed": False, "production_champion": "V1",
            "v2_mode": "SHADOW_ONLY", "automatic_promotion": False,
        }

    rows: list[dict[str, Any]] = []
    # Calibrate once per prediction date. Every training label must have been
    # realized before the current prediction date; current/future outcomes never
    # participate in choosing alpha.
    for pred_date, batch in x.groupby("prediction_date", sort=True):
        prior = x.loc[x["target_date"] < pred_date].copy()
        prior_sessions = int(prior["target_date"].nunique())
        enough_history = len(prior) >= MIN_TRAIN_ROWS and prior_sessions >= MIN_TRAIN_SESSIONS
        alpha = 0.0
        train_mape = _mape(prior["actual_close"], prior["base_close"]) if not prior.empty else None
        if enough_history:
            best_loss = float("inf")
            for candidate_alpha in ALPHAS:
                train_prediction = prior["base_close"] + candidate_alpha * (prior["predicted_close"] - prior["base_close"])
                loss = _mape(prior["actual_close"], train_prediction)
                if loss is not None and loss < best_loss - 1e-12:
                    best_loss, alpha = loss, float(candidate_alpha)

        calibrated = batch["base_close"] + alpha * (batch["predicted_close"] - batch["base_close"])
        rows.append({
            "prediction_date": str(pd.Timestamp(pred_date).date()),
            "target_date": str(pd.to_datetime(batch["target_date"]).min().date()),
            "rows": int(len(batch)),
            "training_rows": int(len(prior)),
            "training_sessions": prior_sessions,
            "calibration_used": bool(enough_history),
            "selected_alpha": alpha,
            "baseline_mape_pct": _mape(batch["actual_close"], batch["base_close"]),
            "raw_model_mape_pct": _mape(batch["actual_close"], batch["predicted_close"]),
            "calibrated_mape_pct": _mape(batch["actual_close"], calibrated),
            "raw_model_beats_baseline": bool((_mape(batch["actual_close"], batch["predicted_close"]) or float("inf")) < (_mape(batch["actual_close"], batch["base_close"]) or float("inf"))),
            "calibrated_beats_baseline": bool((_mape(batch["actual_close"], calibrated) or float("inf")) < (_mape(batch["actual_close"], batch["base_close"]) or float("inf"))),
        })

    daily = pd.DataFrame(rows).sort_values("prediction_date").reset_index(drop=True)
    baseline_mape = _mape(x["actual_close"], x["base_close"])
    model_mape = _mape(x["actual_close"], x["predicted_close"])
    # Recreate calibrated predictions using the coefficient selected per date.
    alpha_map = {r["prediction_date"]: r["selected_alpha"] for r in rows}
    x["_alpha"] = x["prediction_date"].map(lambda d: alpha_map[str(pd.Timestamp(d).date())])
    x["_calibrated"] = x["base_close"] + x["_alpha"] * (x["predicted_close"] - x["base_close"])
    calibrated_mape = _mape(x["actual_close"], x["_calibrated"])
    relative_improvement = (
        (baseline_mape - calibrated_mape) / baseline_mape * 100.0
        if baseline_mape is not None and baseline_mape > 0 and calibrated_mape is not None else None
    )
    summary = {
        "status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE",
        "scored_rows": int(len(x)),
        "target_sessions": int(x["target_date"].nunique()),
        "prediction_sessions": int(x["prediction_date"].nunique()),
        "duplicate_rows_excluded": duplicate_rows,
        "baseline_close_mape_pct": baseline_mape,
        "raw_model_close_mape_pct": model_mape,
        "walk_forward_calibrated_close_mape_pct": calibrated_mape,
        "relative_calibrated_improvement_vs_baseline_pct": relative_improvement,
        "sessions_with_calibration": int(daily["calibration_used"].sum()),
        "sessions_alpha_zero": int(daily["selected_alpha"].eq(0).sum()),
        "sessions_calibrated_model_beats_baseline": int(daily["calibrated_beats_baseline"].sum()),
        "calibration": "grid search alpha=0..1 by 0.05 on prior realized outcomes only; forecast = previous_close + alpha*(raw_model - previous_close)",
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "minimum_training_sessions": MIN_TRAIN_SESSIONS,
        "strict_no_lookahead": True,
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "promotion_allowed": False,
        "interpretation": "Analysis only. A calibrated result that does not beat the previous-close baseline is not evidence of forecast improvement.",
    }
    return daily, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    daily, summary = evaluate_shrinkage(scores)
    daily.to_csv(DAILY_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
