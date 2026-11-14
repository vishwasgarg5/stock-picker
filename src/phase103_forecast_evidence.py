from __future__ import annotations

"""Phase 103: chronological forecast evidence and session-cluster uncertainty.

This is an evaluation-only report over already-recorded forecasts. It does not
refit a model, change production predictions, or authorize promotion.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def evaluate_forecast_evidence(scores: pd.DataFrame, bootstrap_repeats: int = 2000) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"target_date", "symbol", "close_ape", "baseline_close_ape", "close_direction_correct"}
    if scores.empty or not required.issubset(scores.columns):
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA",
            "missing_columns": sorted(required - set(scores.columns)),
            "scored_rows": 0,
            "promotion_allowed": False,
        }
    x = scores.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype("string").str.strip().str.upper()
    valid_keys = x["target_date"].notna() & x["symbol"].notna() & x["symbol"].ne("")
    duplicate_keys = x.loc[valid_keys].duplicated(["target_date", "symbol"], keep=False)
    duplicate_key_rows = int(duplicate_keys.sum())
    if duplicate_key_rows:
        return pd.DataFrame(), {
            "status": "BLOCKED_AMBIGUOUS_KEYS",
            "scored_rows": 0,
            "duplicate_forecast_key_rows": duplicate_key_rows,
            "promotion_allowed": False,
            "production_champion": "V1",
            "v2_mode": "SHADOW_ONLY",
            "production_model_changed": False,
            "interpretation": "Duplicate target_date/symbol forecast rows are ambiguous and are not silently overweighted.",
        }
    for col in ("close_ape", "baseline_close_ape", "close_direction_correct"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    ape_values = x[["close_ape", "baseline_close_ape"]].to_numpy(dtype=float)
    finite_ape = pd.Series(np.isfinite(ape_values).all(axis=1), index=x.index)
    nonnegative_ape = x["close_ape"].ge(0) & x["baseline_close_ape"].ge(0)
    direction_valid = x["close_direction_correct"].isna() | x["close_direction_correct"].isin([0, 1])
    invalid_metric = ~(finite_ape & nonnegative_ape & direction_valid)
    invalid_metric_rows = int(invalid_metric.sum())
    if invalid_metric_rows:
        return pd.DataFrame(), {
            "status": "BLOCKED_INVALID_METRICS",
            "scored_rows": 0,
            "invalid_metric_rows": invalid_metric_rows,
            "promotion_allowed": False,
            "production_champion": "V1",
            "v2_mode": "SHADOW_ONLY",
            "production_model_changed": False,
            "interpretation": "Non-finite or negative APE values and non-binary direction labels are invalid evidence and are not silently scored.",
        }
    x = x.dropna(subset=["target_date", "close_ape", "baseline_close_ape"])
    if x.empty:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "promotion_allowed": False}
    x["model_beats_baseline"] = x["close_ape"] < x["baseline_close_ape"]
    x["ape_improvement"] = x["baseline_close_ape"] - x["close_ape"]
    daily = x.groupby("target_date", as_index=False).agg(
        rows=("symbol", "count"),
        close_mape_pct=("close_ape", lambda s: float(s.mean() * 100)),
        baseline_close_mape_pct=("baseline_close_ape", lambda s: float(s.mean() * 100)),
        direction_accuracy_pct=("close_direction_correct", lambda s: float(s.mean() * 100) if s.notna().any() else np.nan),
        model_beats_baseline_rate_pct=("model_beats_baseline", lambda s: float(s.mean() * 100)),
        mean_ape_improvement_pct_points=("ape_improvement", lambda s: float(s.mean() * 100)),
    ).sort_values("target_date").reset_index(drop=True)
    daily["rolling_5_session_close_mape_pct"] = daily["close_mape_pct"].rolling(5, min_periods=3).mean()
    daily["rolling_5_session_baseline_mape_pct"] = daily["baseline_close_mape_pct"].rolling(5, min_periods=3).mean()
    daily["rolling_5_session_direction_accuracy_pct"] = daily["direction_accuracy_pct"].rolling(5, min_periods=3).mean()

    # Cluster bootstrap on session means avoids treating stocks from one date as independent.
    session_delta = daily["mean_ape_improvement_pct_points"].dropna().to_numpy(dtype=float)
    ci_low = ci_high = None
    if len(session_delta) >= 2 and bootstrap_repeats > 0:
        rng = np.random.default_rng(103)
        sampled = rng.choice(session_delta, size=(bootstrap_repeats, len(session_delta)), replace=True).mean(axis=1)
        ci_low, ci_high = [float(v) for v in np.quantile(sampled, [0.025, 0.975])]
    model_mape = float(x["close_ape"].mean() * 100)
    baseline_mape = float(x["baseline_close_ape"].mean() * 100)
    rel_improvement = float((baseline_mape - model_mape) / baseline_mape * 100) if baseline_mape > 0 else None
    n_sessions = int(x["target_date"].nunique())
    direction = x["close_direction_correct"].dropna()
    summary: dict[str, Any] = {
        "status": "DESCRIPTIVE_ONLY" if n_sessions >= 20 else "INSUFFICIENT_HOLDOUT",
        "scored_rows": int(len(x)),
        "unique_symbols": int(x["symbol"].nunique()),
        "target_sessions": n_sessions,
        "model_close_mape_pct": model_mape,
        "previous_close_baseline_mape_pct": baseline_mape,
        "relative_mape_improvement_pct": rel_improvement,
        "direction_accuracy_pct": float(direction.mean() * 100) if len(direction) else None,
        "sessions_model_mape_beats_baseline": int((daily["mean_ape_improvement_pct_points"] > 0).sum()),
        "sessions_baseline_beats_model": int((daily["mean_ape_improvement_pct_points"] < 0).sum()),
        "session_cluster_bootstrap_95pct_ci_ape_improvement_pct_points": [ci_low, ci_high],
        "bootstrap_repeats": int(bootstrap_repeats if ci_low is not None else 0),
        "uncertainty_method": "deterministic bootstrap resampling target-session mean APE differences",
        "promotion_allowed": False,
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "interpretation": "Positive APE improvement favors the recorded model over previous-close baseline; descriptive evidence only, not executable P&L.",
    }
    return daily, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "phase33_chronological_prediction_scores.csv"
    try:
        scores = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    daily, summary = evaluate_forecast_evidence(scores)
    daily.to_csv(DATA / "phase103_forecast_session_evidence.csv", index=False)
    (DATA / "phase103_forecast_evidence_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
