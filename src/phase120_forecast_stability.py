from __future__ import annotations

"""Phase 120: forecast and walk-forward bias-correction stability audit.

Analysis-only comparison of raw forecasts, a strictly point-in-time bias-corrected
forecast, and previous-close baseline on the same scored rows. No production
forecast, ranking, risk setting, or promotion decision is modified.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
SESSION_OUT = DATA / "phase120_forecast_stability_by_session.csv"
SYMBOL_OUT = DATA / "phase120_forecast_stability_by_symbol.csv"
REGIME_OUT = DATA / "phase120_forecast_stability_by_regime.csv"
SUMMARY_OUT = DATA / "phase120_forecast_stability_summary.json"
MIN_TRAIN_ROWS = 30
MIN_TRAIN_SESSIONS = 5
MIN_SYMBOL_ROWS = 3
REQUIRED = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & a.gt(0) & p.notna()
    if not valid.any():
        return None
    return float((p[valid] - a[valid]).abs().div(a[valid]).mean() * 100.0)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0, "raw_model_mape_pct": None, "bias_corrected_mape_pct": None,
            "baseline_mape_pct": None, "raw_minus_baseline_mape_pct_points": None,
            "corrected_minus_baseline_mape_pct_points": None,
            "correction_improves_raw": False, "corrected_beats_baseline": False,
        }
    raw = _mape(frame["actual_close"], frame["predicted_close"])
    corrected = _mape(frame["actual_close"], frame["bias_corrected_close"])
    baseline = _mape(frame["actual_close"], frame["base_close"])
    return {
        "rows": int(len(frame)),
        "raw_model_mape_pct": round(raw, 6) if raw is not None else None,
        "bias_corrected_mape_pct": round(corrected, 6) if corrected is not None else None,
        "baseline_mape_pct": round(baseline, 6) if baseline is not None else None,
        "raw_minus_baseline_mape_pct_points": round(raw - baseline, 6) if raw is not None and baseline is not None else None,
        "corrected_minus_baseline_mape_pct_points": round(corrected - baseline, 6) if corrected is not None and baseline is not None else None,
        "correction_improves_raw": bool(corrected is not None and raw is not None and corrected < raw),
        "corrected_beats_baseline": bool(corrected is not None and baseline is not None and corrected < baseline),
    }


def audit_forecast_stability(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED.difference(scores.columns))
    empty = pd.DataFrame()
    if missing:
        return empty, empty, empty, {
            "status": "INSUFFICIENT_DATA", "missing_columns": missing, "scored_rows": 0,
            "strict_no_lookahead": True, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "automatic_promotion": False, "promotion_allowed": False,
        }

    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    for col in ("base_close", "predicted_close", "actual_close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=list(REQUIRED)).copy()
    x = x[
        (x["target_date"] > x["prediction_date"])
        & x["base_close"].gt(0)
        & x["predicted_close"].gt(0)
        & x["actual_close"].gt(0)
        & x["symbol"].ne("")
    ].copy()
    duplicate_rows = int(x.duplicated(["prediction_date", "target_date", "symbol"], keep="last").sum())
    x = x.drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
    x = x.sort_values(["prediction_date", "target_date", "symbol"]).reset_index(drop=True)

    if x.empty:
        return empty, empty, empty, {
            "status": "INSUFFICIENT_DATA", "missing_columns": [], "scored_rows": 0,
            "duplicate_rows_excluded": duplicate_rows, "strict_no_lookahead": True,
            "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "automatic_promotion": False, "promotion_allowed": False,
        }

    x["raw_return"] = x["predicted_close"] / x["base_close"] - 1.0
    x["actual_return"] = x["actual_close"] / x["base_close"] - 1.0
    x["realized_return_bias"] = x["raw_return"] - x["actual_return"]

    session_rows: list[dict[str, Any]] = []
    no_lookahead_checks: list[bool] = []
    for pred_date, batch in x.groupby("prediction_date", sort=True):
        # Outcomes are eligible for training only when fully realized before prediction.
        prior = x.loc[x["target_date"] < pred_date]
        training_sessions = int(prior["target_date"].nunique())
        enough_history = len(prior) >= MIN_TRAIN_ROWS and training_sessions >= MIN_TRAIN_SESSIONS
        bias = float(prior["realized_return_bias"].mean()) if enough_history else 0.0
        corrected = batch["base_close"] * (1.0 + batch["raw_return"] - bias)
        scored = batch.copy()
        scored["bias_corrected_close"] = corrected
        latest_prior = prior["target_date"].max() if not prior.empty else pd.NaT
        no_lookahead_checks.append(pd.isna(latest_prior) or latest_prior < pred_date)
        metrics = _metrics(scored)
        session_rows.append({
            "prediction_date": str(pd.Timestamp(pred_date).date()),
            "target_date": str(batch["target_date"].min().date()),
            "rows": int(len(batch)),
            "training_rows": int(len(prior)),
            "training_sessions": training_sessions,
            "latest_training_target_date": str(latest_prior.date()) if not pd.isna(latest_prior) else None,
            "correction_applied": bool(enough_history),
            "estimated_return_bias_pct": round(bias * 100.0, 6),
            **metrics,
        })
        x.loc[batch.index, "bias_corrected_close"] = corrected

    by_session = pd.DataFrame(session_rows)
    x["realized_abs_move_pct"] = (x["actual_close"] / x["base_close"] - 1.0).abs() * 100.0
    x["move_regime"] = pd.cut(
        x["realized_abs_move_pct"],
        bins=[-np.inf, 1.0, 2.5, np.inf],
        labels=["QUIET_LT_1PCT", "NORMAL_1_TO_2_5PCT", "VOLATILE_GE_2_5PCT"],
    )

    symbol_rows: list[dict[str, Any]] = []
    for symbol, part in x.groupby("symbol", sort=True):
        if len(part) >= MIN_SYMBOL_ROWS:
            symbol_rows.append({"symbol": symbol, **_metrics(part)})
    by_symbol = pd.DataFrame(symbol_rows)
    if not by_symbol.empty:
        by_symbol = by_symbol.sort_values(
            ["corrected_minus_baseline_mape_pct_points", "rows"], ascending=[False, False]
        ).reset_index(drop=True)

    regime_rows: list[dict[str, Any]] = []
    for regime, part in x.groupby("move_regime", observed=True, sort=True):
        regime_rows.append({"regime": str(regime), **_metrics(part)})
    by_regime = pd.DataFrame(regime_rows)

    baseline = _mape(x["actual_close"], x["base_close"])
    raw = _mape(x["actual_close"], x["predicted_close"])
    corrected = _mape(x["actual_close"], x["bias_corrected_close"])
    strict_no_lookahead = bool(all(no_lookahead_checks))
    summary: dict[str, Any] = {
        "status": "SCORED" if len(x) >= 20 else ("SMALL_SAMPLE" if len(x) else "INSUFFICIENT_DATA"),
        "scored_rows": int(len(x)),
        "symbols": int(x["symbol"].nunique()),
        "target_sessions": int(x["target_date"].nunique()),
        "prediction_sessions": int(x["prediction_date"].nunique()),
        "duplicate_rows_excluded": duplicate_rows,
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "minimum_training_sessions": MIN_TRAIN_SESSIONS,
        "minimum_symbol_rows_reported": MIN_SYMBOL_ROWS,
        "raw_model_mape_pct": raw,
        "walk_forward_bias_corrected_mape_pct": corrected,
        "previous_close_baseline_mape_pct": baseline,
        "raw_minus_baseline_mape_pct_points": round(raw - baseline, 6) if raw is not None and baseline is not None else None,
        "corrected_minus_baseline_mape_pct_points": round(corrected - baseline, 6) if corrected is not None and baseline is not None else None,
        "correction_improves_raw": bool(corrected is not None and raw is not None and corrected < raw),
        "corrected_beats_baseline": bool(corrected is not None and baseline is not None and corrected < baseline),
        "sessions_with_correction": int(by_session["correction_applied"].sum()),
        "sessions_without_enough_training_history": int((~by_session["correction_applied"]).sum()),
        "sessions_raw_model_beats_baseline": int((by_session["raw_minus_baseline_mape_pct_points"] < 0).sum()),
        "sessions_corrected_model_beats_baseline": int((by_session["corrected_minus_baseline_mape_pct_points"] < 0).sum()),
        "sessions_correction_improves_raw": int(by_session["correction_improves_raw"].sum()),
        "strict_no_lookahead": strict_no_lookahead,
        "regime_definition": "Absolute realized close move from base_close: quiet <1%, normal 1%-<2.5%, volatile >=2.5%. Descriptive only.",
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_promotion": False,
        "promotion_allowed": False,
        "interpretation": "Analysis only. Compare all forecast variants on the same valid rows. Segment metrics are descriptive, not causal; do not exclude symbols based on small samples. Prospective evidence and separate risk/accounting gates remain required before any promotion review.",
    }
    return by_session, by_symbol, by_regime, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    by_session, by_symbol, by_regime, summary = audit_forecast_stability(scores)
    by_session.to_csv(SESSION_OUT, index=False)
    by_symbol.to_csv(SYMBOL_OUT, index=False)
    by_regime.to_csv(REGIME_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
