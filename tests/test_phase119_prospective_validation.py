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
    assert "bias_corrected_mape_pct" in summary
    assert summary["promotion_allowed"] is False


def test_prospective_review_waits_for_minimum_session_count():
    rows = []
    dates = pd.bdate_range("2026-10-15", periods=21)
    for i in range(20):
        rows.append(_row(dates[i].strftime("%Y-%m-%d"), dates[i + 1].strftime("%Y-%m-%d")))
    daily, summary = evaluate_prospective(pd.DataFrame(rows), pd.Timestamp("2026-10-14"))

    assert summary["prospective_sessions"] == 20
    assert summary["status"] == "READY_FOR_REVIEW"
    assert summary["promotion_allowed"] is False
    assert "walk_forward_shrinkage_mape_pct" in summary
    assert "sessions_with_nonzero_shrinkage" in summary


def test_prospective_shrinkage_uses_prior_labels_and_can_fall_back_to_baseline():
    rows = []
    dates = pd.bdate_range("2026-10-01", periods=11)
    for i in range(8):
        for symbol in ("AAA", "BBB", "CCC", "DDD"):
            rows.append(_row(dates[i].strftime("%Y-%m-%d"), dates[i + 1].strftime("%Y-%m-%d"),
                             symbol=symbol, actual=100.0, predicted=105.0))
    # The prospective forecast is strictly after the frozen lock date and all
    # training labels have matured before it is issued.
    for symbol in ("AAA", "BBB", "CCC", "DDD"):
        rows.append(_row("2026-10-15", "2026-10-16", symbol=symbol, actual=100.0, predicted=105.0))

    daily, summary = evaluate_prospective(pd.DataFrame(rows), pd.Timestamp("2026-10-14"))

    assert summary["prospective_sessions"] == 1
    assert int(daily.iloc[0]["training_rows"]) >= 30
    assert int(daily.iloc[0]["training_sessions"]) >= 5
    assert pd.Timestamp(daily.iloc[0]["latest_training_target_date"]) < pd.Timestamp(daily.iloc[0]["prediction_date"])
    assert summary["walk_forward_shrinkage_mape_pct"] == summary["previous_close_baseline_mape_pct"]
    assert summary["sessions_with_nonzero_shrinkage"] == 0
    assert daily.iloc[0]["shrinkage_alpha"] == 0.0
    assert not bool(daily.iloc[0]["shrinkage_beats_baseline"])
    assert summary["production_model_changed"] is False
    assert summary["promotion_allowed"] is False


def test_missing_schema_is_safe():
    daily, summary = evaluate_prospective(pd.DataFrame([{"symbol": "AAA"}]))

    assert daily.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["automatic_promotion"] is False


def test_bias_correction_uses_only_prior_realized_sessions_and_same_locked_cohort():
    rows = []
    dates = pd.bdate_range("2026-10-01", periods=35)
    for i in range(34):
        for symbol in ("AAA", "BBB", "CCC", "DDD"):
            rows.append(_row(
                dates[i].strftime("%Y-%m-%d"),
                dates[i + 1].strftime("%Y-%m-%d"),
                symbol=symbol, actual=103.0, predicted=105.0,
            ))
    daily, summary = evaluate_prospective(pd.DataFrame(rows), pd.Timestamp("2026-10-14"))

    corrected = daily[daily["correction_applied"]]
    assert not corrected.empty
    assert (pd.to_numeric(corrected["training_rows"]) >= 30).all()
    assert (pd.to_numeric(corrected["training_sessions"]) >= 5).all()
    assert summary["sessions_with_bias_correction"] == len(corrected)
    assert summary["strict_no_lookahead_for_bias_correction"] is True
    assert summary["production_model_changed"] is False
    assert summary["automatic_promotion"] is False
    assert summary["promotion_allowed"] is False



def test_prospective_validation_rejects_non_finite_and_non_positive_forecasts():
    rows = [
        _row("2026-10-15", "2026-10-16", predicted=float("inf")),
        _row("2026-10-15", "2026-10-16", symbol="BBB", predicted=-1.0),
        _row("2026-10-15", "2026-10-16", symbol="CCC", actual=float("-inf")),
        _row("2026-10-15", "2026-10-16", symbol="DDD"),
    ]
    daily, summary = evaluate_prospective(pd.DataFrame(rows), pd.Timestamp("2026-10-14"))

    assert summary["invalid_non_finite_rows"] == 2
    assert summary["prospective_rows"] == 1
    assert summary["prospective_rows"] == int(daily["rows"].sum())


def test_prospective_validation_rejects_overflowed_derived_returns():
    row = _row("2026-10-15", "2026-10-16", predicted=1e308)
    row["base_close"] = 1e-320
    daily, summary = evaluate_prospective(pd.DataFrame([row]), pd.Timestamp("2026-10-14"))

    assert summary["invalid_non_finite_rows"] == 1
    assert summary["prospective_rows"] == 0
    assert daily.empty
