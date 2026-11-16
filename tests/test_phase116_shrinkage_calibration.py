import pandas as pd

from src.phase116_shrinkage_calibration import evaluate_shrinkage


def _scores(model_price: float, actual_price: float = 105.0) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-01", periods=12)
    for i, prediction_date in enumerate(dates[:-1]):
        target_date = dates[i + 1]
        for symbol in ("AAA", "BBB", "CCC", "DDD"):
            rows.append({
                "prediction_date": prediction_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "base_close": 100.0,
                "predicted_close": model_price,
                "actual_close": actual_price,
            })
    return pd.DataFrame(rows)


def test_walk_forward_calibration_only_uses_prior_realized_sessions_and_can_improve():
    daily, summary = evaluate_shrinkage(_scores(105.0, 105.0))

    assert summary["strict_no_lookahead"] is True
    assert summary["production_model_changed"] is False
    assert summary["promotion_allowed"] is False
    assert summary["production_champion"] == "V1"
    assert summary["v2_mode"] == "SHADOW_ONLY"
    assert summary["sessions_with_calibration"] > 0
    assert summary["walk_forward_calibrated_close_mape_pct"] < summary["baseline_close_mape_pct"]
    calibrated = daily[daily["calibration_used"]]
    assert (calibrated["training_sessions"] >= 5).all()
    assert (calibrated["training_rows"] >= 30).all()
    assert (pd.to_datetime(calibrated["latest_training_target_date"]) < pd.to_datetime(calibrated["prediction_date"])).all()
    assert (calibrated["selected_alpha"] == 1.0).all()


def test_walk_forward_calibration_falls_back_to_baseline_when_model_is_worse():
    daily, summary = evaluate_shrinkage(_scores(110.0, 100.0))

    assert summary["walk_forward_calibrated_close_mape_pct"] == 0.0
    assert summary["baseline_close_mape_pct"] == 0.0
    assert summary["sessions_alpha_zero_including_fallback"] == len(daily)
    assert summary["calibrated_sessions_alpha_zero"] == summary["sessions_with_calibration"]
    assert (daily["selected_alpha"] == 0.0).all()


def test_walk_forward_calibration_marks_missing_schema_insufficient():
    daily, summary = evaluate_shrinkage(pd.DataFrame([{"symbol": "AAA"}]))

    assert daily.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["production_model_changed"] is False
    assert summary["promotion_allowed"] is False
