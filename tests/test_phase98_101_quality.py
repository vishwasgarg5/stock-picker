import pandas as pd

from src.phase98_101_quality import (
    FEATURES,
    _diagnose_forecasts,
    _feature_panel,
    _paper_performance,
    _selection_outcomes,
    _walk_forward_candidate,
)


def test_feature_panel_is_point_in_time_and_labels_next_observed_session():
    history = pd.DataFrame([
        {"date": f"2026-01-{day:02d}", "symbol": "AAA", "close": 100 + day, "volume": 1000 + 10 * day}
        for day in range(1, 36)
    ])

    panel = _feature_panel(history)

    assert not panel.empty
    assert set(FEATURES).issubset(panel.columns)
    valid = panel.dropna(subset=["target_date", "target_return"])
    assert (pd.to_datetime(valid["date"]) < pd.to_datetime(valid["target_date"])).all()
    last = panel.sort_values("date").iloc[-1]
    assert pd.isna(last["target_date"])


def test_forecast_root_cause_groups_symbol_and_move_bucket():
    scores = pd.DataFrame([
        {"prediction_date": "2026-01-01", "target_date": "2026-01-02", "symbol": "AAA",
         "base_close": 100, "predicted_close": 110, "actual_close": 100, "period": "RISK_OFF"},
        {"prediction_date": "2026-01-02", "target_date": "2026-01-03", "symbol": "BBB",
         "base_close": 100, "predicted_close": 99, "actual_close": 100, "period": "RISK_ON"},
    ])

    by_symbol, by_segment, summary = _diagnose_forecasts(scores)

    assert summary["scored_rows"] == 2
    assert summary["overprediction_rows"] == 1
    assert "volatility_bucket" in by_segment.columns
    assert set(by_symbol["symbol"]) == {"AAA", "BBB"}


def test_walk_forward_candidate_refuses_insufficient_training_history():
    history = pd.DataFrame([
        {"date": f"2026-01-{day:02d}", "symbol": "AAA", "close": 100 + day, "volume": 1000 + day}
        for day in range(1, 35)
    ])
    panel = _feature_panel(history)
    scores = pd.DataFrame([{
        "prediction_date": "2026-02-05", "target_date": "2026-02-06", "symbol": "AAA",
        "base_close": 135, "actual_close": 136,
    }])

    candidate, summary = _walk_forward_candidate(scores, panel)

    assert candidate.empty
    assert summary["status"] == "INSUFFICIENT_HISTORY"
    assert summary["model_refit_production"] is False


def test_paper_performance_uses_net_pnl_and_does_not_subtract_costs_twice():
    trades = pd.DataFrame([
        {"target_date": "2026-01-02", "signal": "BUY", "profit_loss": 90.0, "costs": 10.0},
        {"target_date": "2026-01-02", "signal": "BUY", "profit_loss": -30.0, "costs": 10.0},
        {"target_date": "2026-01-02", "signal": "SKIP", "profit_loss": 0.0, "costs": 0.0},
    ])

    daily, summary = _paper_performance(trades, "test")

    assert summary["executed_trade_rows"] == 2
    assert summary["net_pnl_sum"] == 60.0
    assert summary["reported_costs_sum_not_subtracted_again"] == 20.0
    assert daily.iloc[0]["net_pnl"] == 60.0


def test_selection_outcomes_labels_skips_as_counterfactual_not_trades():
    v2 = pd.DataFrame([
        {"target_date": "2026-01-02", "symbol": "AAA", "signal": "BUY", "reference_price": 100},
        {"target_date": "2026-01-02", "symbol": "BBB", "signal": "SKIP", "reference_price": 100},
    ])
    history = pd.DataFrame([
        {"date": "2026-01-02", "symbol": "AAA", "close": 102, "volume": 1000},
        {"date": "2026-01-02", "symbol": "BBB", "close": 98, "volume": 1000},
    ])

    grouped, summary = _selection_outcomes(v2, history)

    assert summary["rows_with_actuals"] == 2
    assert summary["buy_rows"] == 1
    assert summary["skipped_rows"] == 1
    assert "counterfactuals" in summary["interpretation"]
    assert set(grouped["signal_group"]) == {"SELECTED_BUY", "REJECTED_OR_SKIPPED"}
