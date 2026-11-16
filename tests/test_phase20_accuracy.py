import pandas as pd

from src.phase20_accuracy import _feature_report, _merge_evidence, _symbol_report, _slice_report


def test_phase20_merges_prediction_features_and_evaluation():
    evaluations = pd.DataFrame([{
        "prediction_date": "2026-10-01", "target_date": "2026-10-02",
        "symbol": "ABC", "rank": 1, "close_abs_pct_error": 0.01,
        "baseline_close_abs_pct_error": 0.02, "close_direction_correct": 1,
    }])
    candidates = pd.DataFrame([{
        "prediction_date": "2026-10-01", "symbol": "abc",
        "technical_score": 70, "stock_news_score": 0.5, "confidence_tier": "HIGH",
    }])
    merged = _merge_evidence(evaluations, candidates)
    assert len(merged) == 1
    assert merged.iloc[0]["technical_score"] == 70
    assert merged.iloc[0]["close_error_pct"] == 1.0
    assert merged.iloc[0]["direction_correct"] == 1


def test_phase20_flags_repeated_poor_symbol_results():
    rows = []
    for day in range(4):
        rows.append({
            "prediction_date": f"2026-10-0{day+1}", "symbol": "BAD",
            "direction_correct": 0, "close_error_pct": 8.0,
            "baseline_close_error_pct": 4.0,
        })
    report = _symbol_report(pd.DataFrame(rows))
    assert bool(report.iloc[0]["instability_flag"]) is True
    assert "direction_accuracy_below_40pct" in report.iloc[0]["diagnostic"]


def test_phase20_slice_report_is_descriptive():
    df = pd.DataFrame([
        {"prediction_date": "2026-10-01", "rank": 1, "direction_correct": 1,
         "close_error_pct": 1.0, "baseline_close_error_pct": 2.0, "stock_news_score": 0.5},
        {"prediction_date": "2026-10-02", "rank": 2, "direction_correct": 0,
         "close_error_pct": 3.0, "baseline_close_error_pct": 2.0, "stock_news_score": -0.5},
    ])
    report = _slice_report(df)
    assert not report.empty
    assert {"dimension", "bucket", "direction_accuracy_pct", "mape_improvement_pct"}.issubset(report.columns)


def test_phase20_feature_report_requires_evidence():
    df = pd.DataFrame([{
        "prediction_date": "2026-10-01", "direction_correct": 1,
        "technical_score": 50,
    }])
    report = _feature_report(df)
    assert not report.empty
    assert not report["stable_candidate"].any()
    assert report["status"].eq("COLLECTING_MINIMUM_5_SESSIONS").all()
