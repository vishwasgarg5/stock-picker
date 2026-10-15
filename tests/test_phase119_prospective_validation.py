import pandas as pd

from src.phase119_prospective_validation import evaluate_prospective


def _row(prediction_date, target_date, symbol="AAA", actual=102.0, predicted=101.0):
    return {
        "prediction_date": prediction_date,
        "target_date": target_date,
        "symbol": symbol,
        "base_close": 100.0,
        "predicted_close": predicted,
        "actual_close": actual,
    }


def test_locked_cohort_excludes_all_pre_lock_predictions():
    scores = pd.DataFrame([
        _row("2026-10-13", "2026-10-14", actual=110, predicted=100),
        _row("2026-10-15", "2026-10-16", actual=102, predicted=101),
    ])
    daily, summary = evaluate_prospective(scores, pd.Timestamp("2026-10-14"))

    assert summary["prospective_rows"] == 1
    assert summary["prospective_sessions"] == 1
    assert summary["status"] == "WAITING_FOR_PROSPECTIVE_SESSIONS"
    assert daily.iloc[0]["prediction_date"] == "2026-10-15"
    assert summary["automatic_promotion"] is False
    assert summary["production_model_changed"] is False


def test_prospective_review_waits_for_minimum_session_count():
    rows = []
    dates = pd.bdate_range("2026-10-15", periods=21)
    for i in range(20):
        rows.append(_row(dates[i].strftime("%Y-%m-%d"), dates[i + 1].strftime("%Y-%m-%d")))
    daily, summary = evaluate_prospective(pd.DataFrame(rows), pd.Timestamp("2026-10-14"))

    assert summary["prospective_sessions"] == 20
    assert summary["status"] == "READY_FOR_REVIEW"
    assert summary["promotion_allowed"] is False


def test_missing_schema_is_safe():
    daily, summary = evaluate_prospective(pd.DataFrame([{"symbol": "AAA"}]))

    assert daily.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["automatic_promotion"] is False
