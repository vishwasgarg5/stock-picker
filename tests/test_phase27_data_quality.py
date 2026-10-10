import pandas as pd

from src.phase27_data_quality import integrity_report, regime_accuracy_report


def test_integrity_report_flags_duplicate_keys_and_missing_confidence():
    evaluations = pd.DataFrame([
        {"prediction_date": "2026-10-01", "symbol": "abc"},
        {"prediction_date": "2026-10-01", "symbol": "ABC"},
    ])
    candidates = pd.DataFrame([
        {"prediction_date": "2026-10-01", "symbol": "abc", "confidence_score": None,
         "target_return": 0.04},
        {"prediction_date": "2026-10-01", "symbol": "ABC", "confidence_score": 90,
         "target_return": 0.02},
    ])
    out = integrity_report(evaluations, candidates).set_index("check")
    assert out.loc["evaluation_duplicate_key_rows", "affected_rows"] == 2
    assert out.loc["candidate_history_duplicate_key_rows", "affected_rows"] == 2
    assert out.loc["candidate_confidence_coverage", "affected_rows"] == 1
    assert out.loc["candidate_future_named_fields", "status"] == "REVIEW"


def test_regime_report_is_descriptive_and_marks_small_samples():
    evaluations = pd.DataFrame([
        {"prediction_date": "2026-10-01", "symbol": "ABC",
         "close_direction_correct": 1, "close_abs_pct_error": 0.01,
         "baseline_close_abs_pct_error": 0.02}
    ])
    candidates = pd.DataFrame([
        {"prediction_date": "2026-10-01", "symbol": "ABC", "market_regime": "BEAR"}
    ])
    out = regime_accuracy_report(evaluations, candidates)
    assert out.iloc[0]["market_regime"] == "BEAR"
    assert out.iloc[0]["direction_accuracy_pct"] == 100
    assert out.iloc[0]["evidence_status"] == "INSUFFICIENT_SAMPLE"
