import pandas as pd

from src.phase97_forecast_quality import build_quality_audit


def test_quality_audit_compares_model_with_baselines_and_excludes_temporal_leakage():
    scores = pd.DataFrame([
        {
            "prediction_date": "2026-10-06", "target_date": "2026-10-07", "symbol": "AAA",
            "base_close": 60.0, "predicted_close": 71.0, "actual_close": 72.0, "period": "RISK_ON",
        },
        {
            "prediction_date": "2026-10-07", "target_date": "2026-10-07", "symbol": "AAA",
            "base_close": 72.0, "predicted_close": 75.0, "actual_close": 72.0, "period": "RISK_ON",
        },
    ])
    history = pd.DataFrame([
        {"date": f"2026-10-0{day}", "symbol": "AAA", "close": close}
        for day, close in enumerate([50, 52, 54, 56, 58, 60], start=1)
    ])

    summary, by_symbol, by_segment = build_quality_audit(scores, history)

    assert summary["scored_rows"] == 1
    assert summary["rejected_temporal_rows"] == 1
    assert summary["momentum_baseline_scored_rows"] == 1
    assert summary["five_session_momentum_baseline"]["close_mape_pct"] == 0.0
    # A no-change baseline is neutral, not a directional forecast.
    assert summary["previous_close_baseline"]["directional_accuracy_pct"] is None
    assert summary["production_champion"] == "V1"
    assert summary["v2_mode"] == "SHADOW_ONLY"
    assert summary["production_model_changed"] is False
    assert len(by_symbol) == 1
    assert by_symbol.iloc[0]["symbol"] == "AAA"
    assert len(by_segment) == 1


def test_quality_audit_marks_missing_schema_insufficient():
    summary, by_symbol, by_segment = build_quality_audit(
        pd.DataFrame([{"symbol": "AAA"}]), pd.DataFrame()
    )

    assert summary["status"] == "INSUFFICIENT_DATA"
    assert "prediction_date" in summary["missing_columns"]
    assert by_symbol.empty
    assert by_segment.empty


def test_momentum_baseline_is_unavailable_without_six_prior_closes():
    scores = pd.DataFrame([{
        "prediction_date": "2026-10-03", "target_date": "2026-10-04", "symbol": "AAA",
        "base_close": 12.0, "predicted_close": 12.5, "actual_close": 12.2,
    }])
    history = pd.DataFrame([
        {"date": "2026-10-01", "symbol": "AAA", "close": 10.0},
        {"date": "2026-10-02", "symbol": "AAA", "close": 11.0},
        {"date": "2026-10-03", "symbol": "AAA", "close": 12.0},
    ])

    summary, _, _ = build_quality_audit(scores, history)

    assert summary["scored_rows"] == 1
    assert summary["momentum_baseline_scored_rows"] == 0
    assert summary["five_session_momentum_baseline"]["scored_rows"] == 0
    assert summary["production_model_changed"] is False
