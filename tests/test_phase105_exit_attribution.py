import pandas as pd

from src.phase105_exit_attribution import evaluate_exit_attribution


def test_phase105_audits_unknown_reasons_without_inference_or_policy_changes():
    trades = pd.DataFrame([
        {"symbol": "AAA", "signal": "BUY", "exit_reason": "stop_loss", "profit_loss": -40.0, "exit_price": 90.0},
        {"symbol": "BBB", "signal": "BUY", "exit_reason": "", "profit_loss": -20.0, "exit_price": 80.0},
        {"symbol": "CCC", "signal": "BUY", "exit_reason": "target", "profit_loss": 30.0, "exit_price": 120.0},
        {"symbol": "SKIP", "signal": "SKIP", "exit_reason": "unknown", "profit_loss": 0.0},
    ])
    grouped, summary = evaluate_exit_attribution(trades)
    assert summary["executed_trade_rows"] == 3
    assert summary["unknown_or_unrecorded_reason_rows"] == 1
    assert summary["unknown_reason_gross_loss"] == 20.0
    assert summary["unknown_reason_share_of_gross_loss_pct"] == (20.0 / 60.0 * 100)
    assert summary["metadata_non_null_counts_unknown_reason_trades"]["exit_price"] == 1
    assert summary["missing_exit_reasons_inferred"] is False
    assert summary["costs_subtracted_again"] is False
    assert summary["production_model_changed"] is False
    assert summary["ranking_or_risk_changed"] is False
    assert summary["automatic_promotion"] is False
    assert set(grouped["exit_reason_group"]) == {"stop_loss", "UNKNOWN", "target"}


def test_phase105_fails_closed_for_empty_or_missing_columns():
    _, empty = evaluate_exit_attribution(pd.DataFrame())
    assert empty["status"] == "INSUFFICIENT_DATA"
    assert empty["automatic_promotion"] is False

    _, invalid = evaluate_exit_attribution(pd.DataFrame([{"signal": "BUY", "symbol": "AAA"}]))
    assert invalid["status"] == "INSUFFICIENT_DATA"
    assert invalid["production_champion"] == "V1"
    assert invalid["v2_mode"] == "SHADOW_ONLY"
