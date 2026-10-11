import pandas as pd

from src.phase28_30_foundation import (
    accuracy_metrics,
    data_quality_report,
    historical_dataset_audit,
    parse_dates,
)


def test_parse_dates_handles_mixed_legacy_formats():
    values = pd.Series(["2026-10-09", "2026-10-09 00:00:00", "not-a-date"])
    parsed = parse_dates(values)
    assert parsed.iloc[0] == pd.Timestamp("2026-10-09")
    assert parsed.iloc[1] == pd.Timestamp("2026-10-09")
    assert pd.isna(parsed.iloc[2])


def test_data_quality_excludes_invalid_keys_from_duplicate_count():
    evaluations = pd.DataFrame([
        {"prediction_date": "2026-10-01", "target_date": "2026-10-02",
         "symbol": "abc", "predicted_close": 10, "actual_close": 11},
        {"prediction_date": "2026-10-01 00:00:00", "target_date": "2026-10-02",
         "symbol": "ABC", "predicted_close": 10, "actual_close": 11},
    ])
    candidates = pd.DataFrame([
        {"prediction_date": "2026-10-01", "target_date": None, "symbol": "ABC"},
        {"prediction_date": "2026-10-01 00:00:00", "target_date": None, "symbol": "ABC"},
    ])
    report = data_quality_report(evaluations, candidates)
    duplicate_eval = report[(report.dataset == "evaluations") & (report.check == "business_keys_unique")].iloc[0]
    invalid_candidates = report[(report.dataset == "candidates") & (report.check == "business_keys_valid")].iloc[0]
    duplicate_candidates = report[(report.dataset == "candidates") & (report.check == "business_keys_unique")].iloc[0]
    assert duplicate_eval.affected_rows == 2
    assert invalid_candidates.affected_rows == 2
    assert duplicate_candidates.affected_rows == 0


def test_dataset_audit_reports_mixed_date_coverage_and_duplicates():
    evaluations = pd.DataFrame([
        {"prediction_date": "2026-10-01", "target_date": "2026-10-02", "symbol": "abc"},
        {"prediction_date": "2026-10-01 00:00:00", "target_date": "2026-10-02", "symbol": "ABC"},
    ])
    candidates = pd.DataFrame([
        {"prediction_date": "2026-10-01", "target_date": "2026-10-02", "symbol": "abc"},
        {"prediction_date": "2026-10-01 00:00:00", "target_date": "2026-10-03", "symbol": "xyz"},
    ])
    audit = historical_dataset_audit(evaluations, candidates).set_index("dataset")
    assert audit.loc["evaluations", "prediction_date_coverage_pct"] == 100
    assert audit.loc["evaluations", "duplicate_business_key_rows"] == 2
    assert audit.loc["prediction_candidates_history", "audit_status"] == "PASS"


def test_accuracy_metrics_compare_forecast_and_baseline_without_promoting_model():
    evaluations = pd.DataFrame([
        {"prediction_date": "2026-10-01", "target_date": "2026-10-02",
         "predicted_open": 101, "actual_open": 100, "baseline_open_abs_pct_error": 0.02,
         "predicted_high": 111, "actual_high": 110, "baseline_high_abs_pct_error": 0.02,
         "predicted_low": 91, "actual_low": 90, "baseline_low_abs_pct_error": 0.02,
         "predicted_close": 101, "actual_close": 100, "baseline_close_abs_pct_error": 0.02,
         "predicted_close_direction": 1, "actual_close_direction": 1},
    ])
    metrics = accuracy_metrics(evaluations)
    close = metrics[(metrics.scope == "overall") & (metrics.field == "close")].iloc[0]
    assert close.n == 1
    assert close.mae == 1
    assert close.mape_pct == 1
    assert close.baseline_mape_pct == 2
    assert close.mape_improvement_pct == 50
    assert close.direction_accuracy_pct == 100
    assert close.evidence_status == "INSUFFICIENT_SAMPLE"
