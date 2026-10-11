import pandas as pd

from src.phase41_43_diagnostics import (
    audit_selection_funnel,
    compare_selected_rejected,
    fixed_grid_threshold_sensitivity,
)


def test_phase41_does_not_fabricate_missing_gate_counts():
    funnel, summary = audit_selection_funnel(pd.DataFrame(), pd.DataFrame([
        {"signal": "SKIP"}, {"signal": "SKIP"}, {"signal": "BUY"}
    ]))
    assert summary["status"] == "FALLBACK_HISTORY_ONLY"
    assert summary["observed_stages"] == 0
    assert "cannot attribute" in funnel.iloc[0]["detail"]


def test_phase41_checks_funnel_count_consistency():
    source = pd.DataFrame([
        {"stage": "confidence", "input_rows": 10, "output_rows": 7, "rejected_rows": 3},
        {"stage": "phase2", "input_rows": 7, "output_rows": 4, "rejected_rows": 3},
    ])
    out, summary = audit_selection_funnel(source, pd.DataFrame())
    assert summary["status"] == "OBSERVED"
    assert out["count_consistent"].all()


def test_phase42_compares_selected_and_rejected_to_actuals():
    candidates = pd.DataFrame([
        {"target_date": "2026-09-02", "symbol": "abc", "base_close": 100, "predicted_close": 102, "phase2_selected": 1},
        {"target_date": "2026-09-02", "symbol": "xyz", "base_close": 100, "predicted_close": 98, "phase2_selected": 0},
    ])
    actuals = pd.DataFrame([
        {"date": "2026-09-02", "symbol": "ABC", "close": 103, "open": 101},
        {"date": "2026-09-02", "symbol": "XYZ", "close": 99, "open": 100},
    ])
    report, summary = compare_selected_rejected(candidates, actuals)
    assert summary["matched_rows"] == 2
    assert set(report["group"]) == {"SELECTED", "REJECTED"}
    assert summary["production_settings_changed"] is False


def test_phase43_threshold_sensitivity_is_chronological_and_non_deploying():
    candidates = pd.DataFrame([
        {"prediction_date": f"2026-09-{d:02d}", "target_date": f"2026-09-{d+1:02d}",
         "symbol": "ABC", "base_close": 100+d, "predicted_close": 101+d,
         "confidence_v3": float(d)}
        for d in range(1, 10)
    ])
    actuals = pd.DataFrame([
        {"date": f"2026-09-{d:02d}", "symbol": "ABC", "open": 100+d, "close": 101+d}
        for d in range(2, 11)
    ])
    report, summary = fixed_grid_threshold_sensitivity(candidates, actuals)
    assert len(report) == 8
    assert set(report["period"]) == {"EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT"}
    assert summary["thresholds_deployed"] is False
    assert summary["promotion_allowed"] is False
