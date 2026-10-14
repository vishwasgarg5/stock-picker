import pandas as pd

from src.phase104_loss_concentration import evaluate_loss_concentration


def test_phase104_loss_concentration_preserves_pnl_and_flags_small_samples():
    trades = pd.DataFrame([
        {"target_date": "2026-01-01", "symbol": "AAA", "signal": "BUY", "exit_reason": "stop", "profit_loss": -40.0},
        {"target_date": "2026-01-01", "symbol": "AAA", "signal": "BUY", "exit_reason": "target", "profit_loss": 20.0},
        {"target_date": "2026-01-02", "symbol": "BBB", "signal": "BUY", "exit_reason": "stop", "profit_loss": -10.0},
        {"target_date": "2026-01-02", "symbol": "SKIP", "signal": "SKIP", "exit_reason": "none", "profit_loss": 0.0},
    ])
    by_symbol, by_reason, daily, summary = evaluate_loss_concentration(trades)
    assert summary["executed_trade_rows"] == 3
    assert summary["net_pnl_sum"] == -30.0
    assert summary["gross_loss_magnitude"] == 50.0
    assert summary["costs_subtracted_again"] is False
    assert summary["production_model_changed"] is False
    assert summary["ranking_or_risk_changed"] is False
    assert by_symbol.iloc[0]["symbol"] == "AAA"
    assert by_symbol.iloc[0]["minimum_sample_met_5_trades"] == False
    assert set(by_reason["exit_reason_group"]) == {"stop", "target"}
    assert daily.iloc[-1]["cumulative_net_pnl"] == -30.0


def test_phase104_empty_or_invalid_inputs_never_authorize_changes():
    _, _, _, empty = evaluate_loss_concentration(pd.DataFrame())
    assert empty["status"] == "INSUFFICIENT_DATA"
    assert empty["production_model_changed"] is False
    invalid = pd.DataFrame([{"signal": "BUY", "symbol": "AAA"}])
    _, _, _, summary = evaluate_loss_concentration(invalid)
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["automatic_promotion"] is False
