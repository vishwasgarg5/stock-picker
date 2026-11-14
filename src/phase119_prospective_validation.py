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
MIN_TRAIN_ROWS = 30
MIN_TRAIN_SESSIONS = 5
SHRINKAGE_ALPHAS = [i / 20 for i in range(21)]
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
    numeric_columns = ["base_close", "predicted_close", "actual_close"]
    for col in numeric_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    finite_inputs = pd.DataFrame(pd.notna(x[numeric_columns]) & (x[numeric_columns].abs() != float("inf")), index=x.index, columns=numeric_columns).all(axis=1)
    invalid_non_finite_rows = int((x[numeric_columns].notna().any(axis=1) & ~finite_inputs).sum())
    x = x.loc[finite_inputs].dropna(subset=list(REQUIRED))
    x = x[(x["target_date"] > x["prediction_date"]) & x["base_close"].gt(0) & x["predicted_close"].gt(0) & x["actual_close"].gt(0)]
    x = x.drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
    # Only predictions made strictly after the frozen lock date can enter.
    prospective = x.loc[x["prediction_date"] > pd.Timestamp(lock_date).normalize()].copy()
    # Evaluate a pooled bias-corrected variant on the exact same locked cohort.
    # For each prediction date, train only on labels whose target_date is strictly
    # earlier than that prediction date; pre-lock rows may train the estimator but
    # can never enter the prospective scoring cohort.
    min_train_rows = 30
    min_train_sessions = 5
    import numpy as np
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x["raw_return"] = x["predicted_close"] / x["base_close"] - 1.0
        x["actual_return"] = x["actual_close"] / x["base_close"] - 1.0
        x["realized_return_bias"] = x["raw_return"] - x["actual_return"]
    derived_columns = ["raw_return", "actual_return", "realized_return_bias"]
    finite_derived = pd.DataFrame(np.isfinite(x[derived_columns]), index=x.index, columns=derived_columns).all(axis=1)
    invalid_non_finite_rows += int((~finite_derived).sum())
    x = x.loc[finite_derived].copy()
    prospective = x.loc[x["prediction_date"] > pd.Timestamp(lock_date).normalize()].copy()
    daily_rows = []
    corrected_parts = []
    shrinkage_parts = []
    for pred_date, batch in prospective.groupby("prediction_date", sort=True):
        prior = x.loc[x["target_date"] < pred_date]
        train_sessions = int(prior["target_date"].nunique())
        correction_applied = len(prior) >= min_train_rows and train_sessions >= min_train_sessions
        estimated_bias = float(prior["realized_return_bias"].mean()) if correction_applied else 0.0
        corrected_close = batch["predicted_close"] - (batch["base_close"] * estimated_bias)
        # Select a baseline-anchored shrinkage weight using only prior realized labels.
        # Alpha=0 is the previous-close baseline and remains available at every date.
        selected_alpha = 0.0
        if correction_applied:
            best_loss = float("inf")
            for candidate_alpha in SHRINKAGE_ALPHAS:
                train_prediction = prior["base_close"] + candidate_alpha * (prior["predicted_close"] - prior["base_close"])
                loss = _mape(prior["actual_close"], train_prediction)
                if loss is not None and loss < best_loss - 1e-12:
                    best_loss, selected_alpha = loss, candidate_alpha
        shrinkage_close = batch["base_close"] + selected_alpha * (batch["predicted_close"] - batch["base_close"])
        corrected_parts.append(pd.DataFrame({
            "actual_close": batch["actual_close"].to_numpy(),
            "base_close": batch["base_close"].to_numpy(),
            "raw_close": batch["predicted_close"].to_numpy(),
            "corrected_close": corrected_close.to_numpy(),
        }))
        shrinkage_parts.append(pd.DataFrame({
            "actual_close": batch["actual_close"].to_numpy(),
            "shrinkage_close": shrinkage_close.to_numpy(),
        }))
        corrected_mape = _mape(batch["actual_close"], corrected_close)
        shrinkage_mape = _mape(batch["actual_close"], shrinkage_close)
        daily_rows.append({
            "prediction_date": str(pd.Timestamp(pred_date).date()),
            "target_date": str(batch["target_date"].min().date()),
            "rows": int(len(batch)),
            "model_mape_pct": _mape(batch["actual_close"], batch["predicted_close"]),
            "baseline_mape_pct": _mape(batch["actual_close"], batch["base_close"]),
            "model_beats_baseline": bool(_mape(batch["actual_close"], batch["predicted_close"]) is not None
                and _mape(batch["actual_close"], batch["base_close"]) is not None
                and _mape(batch["actual_close"], batch["predicted_close"]) < _mape(batch["actual_close"], batch["base_close"])),
            "training_rows": int(len(prior)),
            "training_sessions": train_sessions,
            "latest_training_target_date": str(prior["target_date"].max().date()) if not prior.empty else None,
            "correction_applied": bool(correction_applied),
            "estimated_return_bias_pct": estimated_bias * 100.0,
            "bias_corrected_mape_pct": corrected_mape,
            "bias_corrected_beats_baseline": bool(corrected_mape is not None
                and _mape(batch["actual_close"], batch["base_close"]) is not None
                and corrected_mape < _mape(batch["actual_close"], batch["base_close"])),
            "shrinkage_alpha": float(selected_alpha),
            "shrinkage_mape_pct": shrinkage_mape,
            "shrinkage_beats_baseline": bool(shrinkage_mape is not None
                and _mape(batch["actual_close"], batch["base_close"]) is not None
                and shrinkage_mape < _mape(batch["actual_close"], batch["base_close"])),
        })
    daily = pd.DataFrame(daily_rows)
    model = _mape(prospective["actual_close"], prospective["predicted_close"]) if not prospective.empty else None
    baseline = _mape(prospective["actual_close"], prospective["base_close"]) if not prospective.empty else None
    corrected_all = pd.concat(corrected_parts, ignore_index=True) if corrected_parts else pd.DataFrame()
    corrected_model = _mape(corrected_all["actual_close"], corrected_all["corrected_close"]) if not corrected_all.empty else None
    shrinkage_all = pd.concat(shrinkage_parts, ignore_index=True) if shrinkage_parts else pd.DataFrame()
    shrinkage_model = _mape(shrinkage_all["actual_close"], shrinkage_all["shrinkage_close"]) if not shrinkage_all.empty else None
    sessions = int(prospective["target_date"].nunique())
    enough = sessions >= MIN_PROSPECTIVE_SESSIONS
    summary: dict[str, Any] = {
        "status": "READY_FOR_REVIEW" if enough else ("WAITING_FOR_PROSPECTIVE_SESSIONS" if prospective.empty or sessions < MIN_PROSPECTIVE_SESSIONS else "INSUFFICIENT_DATA"),
        "lock_date": str(pd.Timestamp(lock_date).date()),
        "cohort_rule": "Include only rows whose prediction_date is strictly later than lock_date; cohort membership is not backfilled with pre-lock predictions.",
        "prospective_rows": int(len(prospective)),
        "invalid_non_finite_rows": invalid_non_finite_rows,
        "prospective_symbols": int(prospective["symbol"].nunique()) if not prospective.empty else 0,
        "prospective_sessions": sessions,
        "minimum_sessions_before_review": MIN_PROSPECTIVE_SESSIONS,
        "model_mape_pct": model,
        "previous_close_baseline_mape_pct": baseline,
        "delta_mape_pct_points_model_minus_baseline": round(model - baseline, 6) if model is not None and baseline is not None else None,
        "sessions_model_beats_baseline": int(daily["model_beats_baseline"].sum()) if not daily.empty else 0,
        "bias_corrected_mape_pct": corrected_model,
        "delta_mape_pct_points_bias_corrected_minus_baseline": round(corrected_model - baseline, 6) if corrected_model is not None and baseline is not None else None,
        "sessions_bias_corrected_beats_baseline": int(daily["bias_corrected_beats_baseline"].sum()) if not daily.empty else 0,
        "walk_forward_shrinkage_mape_pct": shrinkage_model,
        "delta_mape_pct_points_shrinkage_minus_baseline": round(shrinkage_model - baseline, 6) if shrinkage_model is not None and baseline is not None else None,
        "sessions_shrinkage_beats_baseline": int(daily["shrinkage_beats_baseline"].sum()) if not daily.empty else 0,
        "sessions_with_nonzero_shrinkage": int(daily["shrinkage_alpha"].gt(0).sum()) if not daily.empty else 0,
        "shrinkage_training_rule": "Select alpha from 0.00 to 1.00 in 0.05 steps by prior-session MAPE only; require at least 30 prior rows and 5 realized target sessions; alpha=0 falls back to previous close.",
        "sessions_with_bias_correction": int(daily["correction_applied"].sum()) if not daily.empty else 0,
        "sessions_without_enough_bias_training_history": int((~daily["correction_applied"]).sum()) if not daily.empty else 0,
        "bias_correction_training_rule": "Pooled raw-return minus realized-return bias; only rows with target_date strictly before prediction_date; minimum 30 rows and 5 realized target sessions; fallback to raw forecast otherwise.",
        "strict_no_lookahead_for_bias_correction": bool(
            daily.empty
            or (
                daily.loc[daily["correction_applied"], "latest_training_target_date"].notna().all()
                and (
                    pd.to_datetime(daily.loc[daily["correction_applied"], "latest_training_target_date"])
                    < pd.to_datetime(daily.loc[daily["correction_applied"], "prediction_date"])
                ).all()
            )
        ),
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
