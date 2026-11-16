import pandas as pd

from src.phase118_forecast_error_diagnosis import diagnose_forecast_errors


def test_diagnosis_segments_by_symbol_and_realized_move_without_changing_production():
    rows = []
    dates = pd.bdate_range("2026-01-01", periods=5)
    for i in range(4):
        for symbol, actual in (("QUIET", 100.5), ("VOLATILE", 104.0)):
            rows.append({
                "prediction_date": dates[i].strftime("%Y-%m-%d"),
                "target_date": dates[i + 1].strftime("%Y-%m-%d"),
                "symbol": symbol,
                "base_close": 100.0,
                "predicted_close": 101.0,
                "actual_close": actual,
            })
    symbols, regimes, summary = diagnose_forecast_errors(pd.DataFrame(rows))

    assert summary["status"] == "SMALL_SAMPLE"
    assert summary["scored_rows"] == 8
    assert set(symbols["symbol"]) == {"QUIET", "VOLATILE"}
    assert set(regimes["regime"]) == {"QUIET_LT_1PCT", "VOLATILE_GE_2_5PCT"}
    assert summary["production_model_changed"] is False
    assert summary["automatic_promotion"] is False


def test_diagnosis_missing_schema_is_safe():
    symbols, regimes, summary = diagnose_forecast_errors(pd.DataFrame([{"symbol": "AAA"}]))

    assert symbols.empty
    assert regimes.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["automatic_promotion"] is False



def test_diagnosis_rejects_non_finite_and_non_positive_forecast_prices():
    dates = pd.bdate_range("2026-01-01", periods=4)
    rows = []
    for i in range(3):
        rows.append({
            "prediction_date": dates[i].strftime("%Y-%m-%d"),
            "target_date": dates[i + 1].strftime("%Y-%m-%d"),
            "symbol": f"S{i}",
            "base_close": 100.0,
            "predicted_close": [float("inf"), -1.0, 101.0][i],
            "actual_close": [102.0, 102.0, float("-inf")][i],
        })
    symbols, regimes, summary = diagnose_forecast_errors(pd.DataFrame(rows))

    assert summary["invalid_non_finite_rows"] == 2
    assert summary["scored_rows"] == 0
    assert symbols.empty
    assert regimes.empty


def test_diagnosis_rejects_overflowed_realized_move():
    row = {
        "prediction_date": "2026-01-01",
        "target_date": "2026-01-02",
        "symbol": "AAA",
        "base_close": 1e-320,
        "predicted_close": 1.0,
        "actual_close": 1e308,
    }
    symbols, regimes, summary = diagnose_forecast_errors(pd.DataFrame([row]))

    assert summary["invalid_non_finite_rows"] == 1
    assert summary["scored_rows"] == 0
    assert symbols.empty
    assert regimes.empty
