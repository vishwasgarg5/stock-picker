import pandas as pd

from src.phase32_40_diagnostics import (
    _phase32_funnel,
    _phase35_trade_metrics,
    score_historical_predictions,
)


def _candidates():
    return pd.DataFrame([
        {"prediction_date": "2026-09-01", "target_date": "2026-09-02", "symbol": "ABC",
         "base_close": 100, "predicted_open": 101, "predicted_high": 105,
         "predicted_low": 99, "predicted_close": 102, "prediction_spread": 0.01},
        {"prediction_date": "2026-09-02", "target_date": "2026-09-03", "symbol": "ABC",
         "base_close": 102, "predicted_open": 103, "predicted_high": 107,
         "predicted_low": 101, "predicted_close": 104, "prediction_spread": 0.02},
        {"prediction_date": "2026-09-03", "target_date": "2026-09-04", "symbol": "ABC",
         "base_close": 104, "predicted_open": 104, "predicted_high": 108,
         "predicted_low": 102, "predicted_close": 105, "prediction_spread": 0.03},
        {"prediction_date": "2026-09-04", "target_date": "2026-09-05", "symbol": "ABC",
         "base_close": 105, "predicted_open": 106, "predicted_high": 109,
         "predicted_low": 103, "predicted_close": 107, "prediction_spread": 0.01},
        {"prediction_date": "2026-09-05", "target_date": "2026-09-06", "symbol": "ABC",
         "base_close": 107, "predicted_open": 107, "predicted_high": 111,
         "predicted_low": 105, "predicted_close": 108, "prediction_spread": 0.02},
    ])


def _actuals():
    return pd.DataFrame([
        {"date": "2026-09-02", "symbol": "abc", "open": 100, "high": 104, "low": 98, "close": 101},
        {"date": "2026-09-03", "symbol": "ABC", "open": 102, "high": 106, "low": 100, "close": 103},
        {"date": "2026-09-04", "symbol": "ABC", "open": 103, "high": 107, "low": 101, "close": 104},
        {"date": "2026-09-05", "symbol": "ABC", "open": 105, "high": 108, "low": 102, "close": 106},
        {"date": "2026-09-06", "symbol": "ABC", "open": 106, "high": 110, "low": 104, "close": 107},
    ])


def test_phase33_scores_saved_predictions_in_chronological_order():
    scored, summary = score_historical_predictions(_candidates(), _actuals(), min_holdout_dates=1)
    assert len(scored) == 5
    assert scored["target_date"].is_monotonic_increasing
    assert scored["period"].iloc[-1] == "CHRONOLOGICAL_HOLDOUT"
    assert summary["method"].startswith("chronological replay")
    assert summary["promotion_allowed"] is False


def test_phase33_rejects_prediction_date_not_before_target_date():
    candidates = _candidates()
    candidates.loc[0, "prediction_date"] = "2026-09-02"
    scored, summary = score_historical_predictions(candidates, _actuals(), min_holdout_dates=1)
    assert summary["invalid_prediction_order_rows"] == 1
    assert len(scored) == 4


def test_phase33_empty_inputs_are_insufficient_not_success():
    scored, summary = score_historical_predictions(pd.DataFrame(), pd.DataFrame())
    assert scored.empty
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["scored_rows"] == 0


def test_phase35_uses_net_profit_loss_without_double_subtracting_costs():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 10, "profit_loss": 95, "trading_cost": 5},
        {"signal": "BUY", "quantity": 5, "profit_loss": -25, "trading_cost": 2},
        {"signal": "SKIP", "quantity": 0, "profit_loss": 0, "trading_cost": 0},
    ])
    out = _phase35_trade_metrics(trades, "V2")
    assert out["executed_trades"] == 2
    assert out["net_pnl"] == 70
    assert out["recorded_costs"] == 7
    assert "does not subtract costs a second time" in out["costs_note"]


def test_phase32_funnel_does_not_claim_internal_stages_from_trade_history():
    trades = pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "shadow_selection_filter"},
        {"signal": "BUY", "no_trade_reason": ""},
    ])
    funnel, summary = _phase32_funnel(trades)
    assert summary["status"] == "FALLBACK_HISTORY_ONLY"
    assert summary["buy_rows"] == 1
    assert any("cannot identify every internal filter stage" in x for x in funnel["detail"])


def test_phase32_no_trade_history_marks_funnel_as_awaiting_observation(monkeypatch, tmp_path):
    import src.phase32_40_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
    funnel, summary = _phase32_funnel(pd.DataFrame())
    assert summary["status"] == "AWAITING_NEXT_V2_RUN"
    assert "Awaiting next V2 run" in funnel.iloc[0]["detail"]


def test_phase33_requires_minimum_holdout_dates_and_rows():
    scored, summary = score_historical_predictions(
        _candidates(), _actuals(), min_holdout_dates=10
    )
    assert len(scored) == 5
    assert summary["holdout_target_dates"] < 10
    assert summary["status"] == "INSUFFICIENT_HOLDOUT"
