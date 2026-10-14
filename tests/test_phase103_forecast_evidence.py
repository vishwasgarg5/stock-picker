import pandas as pd

from src.phase103_forecast_evidence import evaluate_forecast_evidence


def test_phase103_session_cluster_evidence_and_safety_flags():
    rows = []
    for day in range(1, 9):
        date = f"2026-01-{day:02d}"
        for symbol, model, baseline, correct in [
            ("AAA", 0.01, 0.02, 1),
            ("BBB", 0.03, 0.02, 0),
        ]:
            rows.append({
                "target_date": date, "symbol": symbol, "close_ape": model,
                "baseline_close_ape": baseline, "close_direction_correct": correct,
            })
    daily, summary = evaluate_forecast_evidence(pd.DataFrame(rows), bootstrap_repeats=200)
    assert len(daily) == 8
    assert summary["target_sessions"] == 8
    assert summary["status"] == "INSUFFICIENT_HOLDOUT"
    assert summary["promotion_allowed"] is False
    assert summary["production_champion"] == "V1"
    assert summary["v2_mode"] == "SHADOW_ONLY"
    assert summary["production_model_changed"] is False
    assert len(summary["session_cluster_bootstrap_95pct_ci_ape_improvement_pct_points"]) == 2


def test_phase103_missing_inputs_are_not_reported_as_success():
    daily, summary = evaluate_forecast_evidence(pd.DataFrame())
    assert daily.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["promotion_allowed"] is False
