import pandas as pd

from src.phase117_bias_correction import evaluate_bias_correction


def _biased_scores() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=12)
    rows = []
    for i, prediction_date in enumerate(dates[:-1]):
        for symbol in ("AAA", "BBB", "CCC", "DDD"):
            rows.append({
                "prediction_date": prediction_date.strftime("%Y-%m-%d"),
                "target_date": dates[i + 1].strftime("%Y-%m-%d"),
                "symbol": symbol,
                "base_close": 100.0,
                "predicted_close": 105.0,
                "actual_close": 103.0,
            })
    return pd.DataFrame(rows)


def test_walk_forward_bias_correction_learns_only_past_bias_and_improves():
    daily, summary = evaluate_bias_correction(_biased_scores())

    assert summary["strict_no_lookahead"] is True
    assert summary["production_model_changed"] is False
    assert summary["promotion_allowed"] is False
    assert summary["production_champion"] == "V1"
    assert summary["v2_mode"] == "SHADOW_ONLY"
    assert summary["sessions_with_correction"] > 0
    assert summary["walk_forward_bias_corrected_close_mape_pct"] < summary["raw_model_close_mape_pct"]
    assert summary["walk_forward_bias_corrected_close_mape_pct"] < summary["baseline_close_mape_pct"]
    corrected = daily[daily["correction_applied"]]
    assert (corrected["training_rows"] >= 30).all()
    assert (corrected["training_sessions"] >= 5).all()
    assert (pd.to_datetime(corrected["latest_training_target_date"]) < pd.to_datetime(corrected["prediction_date"])).all()


def test_walk_forward_bias_correction_falls_back_without_history():
    scores = _biased_scores().iloc[:8].copy()
    daily, summary = evaluate_bias_correction(scores)

    assert summary["sessions_with_correction"] == 0
    assert summary["walk_forward_bias_corrected_close_mape_pct"] == summary["raw_model_close_mape_pct"]
    assert summary["production_model_changed"] is False


def test_walk_forward_bias_correction_missing_schema_is_safe():
    daily, summary = evaluate_bias_correction(pd.DataFrame([{"symbol": "AAA"}]))

    assert daily.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["promotion_allowed"] is False
