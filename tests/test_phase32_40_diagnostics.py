import pandas as pd

from src.phase32_40_diagnostics import (
    _phase32_funnel,
    _phase35_trade_metrics,
    score_historical_predictions,
    promotion_gate,
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
    assert "does not subtract costs a second time" in out["costs_note"] or "stress cases subtract only additional assumed costs" in out["costs_note"]
    assert out["net_pnl_with_50pct_extra_cost_stress"] == 66.5
    assert out["net_pnl_with_100pct_extra_cost_stress"] == 63


def test_phase32_funnel_does_not_claim_internal_stages_from_trade_history(monkeypatch, tmp_path):
    import src.phase32_40_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
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



def test_phase36_zero_v2_holdout_pnl_cannot_pass_even_if_v1_loses():
    result = promotion_gate(
        matched_sessions=30,
        v2_executed_trades=100,
        v1_holdout_net_pnl=-500.0,
        v2_holdout_net_pnl=0.0,
    )
    assert result["status"] == "NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE"
    assert result["production_champion"] == "V1"
    assert result["v2_promoted"] is False


def test_phase36_requires_minimum_sessions_and_trade_count():
    result = promotion_gate(
        matched_sessions=19,
        v2_executed_trades=49,
        v1_holdout_net_pnl=100.0,
        v2_holdout_net_pnl=200.0,
    )
    assert result["status"] == "NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE"
    assert result["automatic_promotion"] is False



def test_phase35_flags_missing_gross_and_cost_ledger():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 10, "profit_loss": 95},
    ])
    out = _phase35_trade_metrics(trades, "V1")
    assert out["pnl_reconciliation_status"] == "MISSING_GROSS_OR_COST_LEDGER"
    assert out["pnl_reconciliation_incomplete_rows"] == 1


def test_phase35_flags_gross_net_cost_mismatch():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 10, "gross_profit_loss": 100,
         "trading_cost": 5, "profit_loss": 98},
    ])
    out = _phase35_trade_metrics(trades, "V1")
    assert out["pnl_reconciliation_status"] == "MISMATCH_GROSS_MINUS_COSTS_VS_NET"
    assert out["pnl_reconciliation_delta"] == -3.0


def test_phase35_marks_reconciled_ledger():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 10, "gross_profit_loss": 100,
         "trading_cost": 5, "profit_loss": 95},
    ])
    out = _phase35_trade_metrics(trades, "V1")
    assert out["pnl_reconciliation_status"] == "RECONCILED"
    assert out["pnl_reconciliation_delta"] == 0.0


def test_phase33_uses_recorded_evaluation_actuals_when_ohlcv_is_missing():
    from src.phase32_40_diagnostics import _actuals_for_scoring

    evaluations = pd.DataFrame([
        {"target_date": "2026-09-02", "symbol": "ABC", "actual_open": 100,
         "actual_high": 104, "actual_low": 98, "actual_close": 101},
    ])
    actuals, source = _actuals_for_scoring(pd.DataFrame(), evaluations)
    assert source.startswith("evaluations.csv actual OHLC fallback")
    assert list(actuals.columns) == ["date", "symbol", "open", "high", "low", "close"]
    assert actuals.iloc[0]["close"] == 101


def test_phase33_supplements_partial_ohlcv_with_missing_evaluation_dates():
    from src.phase32_40_diagnostics import _actuals_for_scoring

    ohlcv = pd.DataFrame([
        {"date": "2026-09-02", "symbol": "ABC", "open": 100, "high": 104, "low": 98, "close": 101},
    ])
    evaluations = pd.DataFrame([
        {"target_date": "2026-09-02", "symbol": "ABC", "actual_open": 100, "actual_high": 104,
         "actual_low": 98, "actual_close": 101},
        {"target_date": "2026-09-03", "symbol": "ABC", "actual_open": 102, "actual_high": 106,
         "actual_low": 100, "actual_close": 103},
    ])
    actuals, source = _actuals_for_scoring(ohlcv, evaluations)
    assert len(actuals) == 2
    assert "supplementing missing date/symbol keys" in source
    assert set(pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")) == {"2026-09-02", "2026-09-03"}


def test_phase35_falls_back_to_shares_when_quantity_column_is_blank():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": float("nan"), "shares": 2,
         "profit_loss": 90, "gross_profit_loss": 100, "costs": 10},
    ])
    out = _phase35_trade_metrics(trades, "V1")
    assert out["executed_trades"] == 1
    assert out["gross_pnl"] == 100
    assert out["net_pnl"] == 90
    assert out["recorded_costs"] == 10
    assert out["pnl_reconciliation_status"] == "RECONCILED"


def test_phase35_does_not_treat_blank_ledger_values_as_zero():
    trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 1, "profit_loss": 90,
         "gross_profit_loss": float("nan"), "trading_cost": 10},
    ])
    out = _phase35_trade_metrics(trades, "V1")
    assert out["pnl_reconciliation_status"] == "MISSING_GROSS_OR_COST_LEDGER"



def test_phase33_falls_back_to_recorded_evaluation_forecasts_when_history_match_is_too_narrow():
    from src.phase32_40_diagnostics import _score_with_evaluation_fallback

    candidates = _candidates().iloc[:1].copy()
    actuals = _actuals()
    evaluations = _candidates().merge(
        actuals.rename(columns={"date": "target_date", "open": "actual_open",
                                "high": "actual_high", "low": "actual_low", "close": "actual_close"}),
        on=["target_date", "symbol"],
        how="inner",
    )
    scored, summary = _score_with_evaluation_fallback(
        candidates, evaluations, actuals, "ohlcv.csv", min_holdout_dates=1
    )
    assert summary["unique_target_dates"] == 5
    assert len(scored) == 5
    assert summary["prediction_source"].startswith("evaluations.csv paired saved forecasts/outcomes")
    assert summary["actuals_source"] == "evaluations.csv paired actual OHLC"
