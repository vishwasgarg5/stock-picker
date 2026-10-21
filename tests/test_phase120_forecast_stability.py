import pandas as pd

from src.phase120_forecast_stability import audit_forecast_stability


def _scores() -> pd.DataFrame:
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


def test_stability_audit_compares_all_variants_on_same_rows_and_corrects_past_bias():
    sessions, symbols, regimes, summary = audit_forecast_stability(_scores())

    assert summary["status"] == "SCORED"
    assert summary["scored_rows"] == 44
    assert summary["strict_no_lookahead"] is True
    assert summary["production_champion"] == "V1"
    assert summary["v2_mode"] == "SHADOW_ONLY"
    assert summary["production_model_changed"] is False
    assert summary["ranking_or_risk_changed"] is False
    assert summary["automatic_promotion"] is False
    assert summary["promotion_allowed"] is False
    assert summary["walk_forward_bias_corrected_mape_pct"] < summary["raw_model_mape_pct"]
    assert summary["walk_forward_bias_corrected_mape_pct"] < summary["previous_close_baseline_mape_pct"]
    assert sessions["rows"].sum() == summary["scored_rows"]
    assert (sessions["training_rows"] >= 30).any()
    corrected = sessions[sessions["correction_applied"]]
    assert (corrected["training_rows"] >= 30).all()
    assert (corrected["training_sessions"] >= 5).all()
    assert (
        pd.to_datetime(corrected["latest_training_target_date"])
        < pd.to_datetime(corrected["prediction_date"])
    ).all()
    assert set(regimes["regime"]) == {"VOLATILE_GE_2_5PCT"}
    assert set(symbols["symbol"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_stability_audit_falls_back_to_raw_forecast_when_history_is_small():
    scores = _scores().iloc[:8].copy()
    sessions, _, _, summary = audit_forecast_stability(scores)

    assert summary["sessions_with_correction"] == 0
    assert summary["walk_forward_bias_corrected_mape_pct"] == summary["raw_model_mape_pct"]
    assert summary["production_model_changed"] is False
    assert (sessions["correction_applied"] == False).all()  # noqa: E712


def test_stability_audit_excludes_duplicate_and_temporally_invalid_rows():
    scores = _scores()
    duplicate = scores.iloc[[0]].copy()
    invalid = scores.iloc[[1]].copy()
    invalid["target_date"] = invalid["prediction_date"]
    scores = pd.concat([scores, duplicate, invalid], ignore_index=True)

    sessions, _, _, summary = audit_forecast_stability(scores)

    assert summary["duplicate_rows_excluded"] == 1
    assert summary["scored_rows"] == 44
    assert sessions["rows"].sum() == 44
    assert summary["strict_no_lookahead"] is True


def test_stability_audit_missing_schema_is_safe():
    sessions, symbols, regimes, summary = audit_forecast_stability(pd.DataFrame([{"symbol": "AAA"}]))

    assert sessions.empty
    assert symbols.empty
    assert regimes.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["production_model_changed"] is False
    assert summary["promotion_allowed"] is False
