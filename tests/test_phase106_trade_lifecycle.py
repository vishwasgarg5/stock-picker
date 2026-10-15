import pandas as pd

from src.phase106_trade_lifecycle import evaluate_trade_lifecycle


def test_phase106_measures_lifecycle_gaps_without_mutating_or_inventing_facts():
    trades = pd.DataFrame([
        {"symbol": "AAA", "entry_price": 100, "exit_price": 90, "entry_date": "2026-01-01",
         "exit_date": "2026-01-02", "quantity": 2, "profit_loss": -20, "exit_reason": None,
         "trade_status": "CLOSED"},
        {"symbol": "AAA", "entry_price": 100, "exit_price": 0, "entry_date": "2026-01-01",
         "exit_date": "2025-12-31", "quantity": 0, "profit_loss": -5, "exit_reason": "stop",
         "trade_status": "CLOSED"},
        {"symbol": "BBB", "entry_price": None, "exit_price": None, "entry_date": "2026-01-03",
         "exit_date": None, "quantity": 1, "profit_loss": 0, "exit_reason": None,
         "trade_status": "OPEN"},
    ])
    original = trades.copy(deep=True)
    result = evaluate_trade_lifecycle(trades)

    assert result["rows"] == 3
    assert result["field_presence"]["entry_price"] is True
    assert result["field_non_null_counts"]["exit_reason"] == 1
    assert result["invalid_price_rows"] == 1
    assert result["invalid_quantity_rows"] == 1
    assert result["entry_after_exit_rows"] == 1
    assert result["duplicate_symbol_entry_date_rows"] == 2
    assert result["closed_rows_missing_exit_price"] == 0
    assert result["missing_facts_inferred"] is False
    assert result["production_model_changed"] is False
    pd.testing.assert_frame_equal(trades, original)


def test_phase106_empty_and_missing_fields_fail_closed():
    empty = evaluate_trade_lifecycle(pd.DataFrame())
    assert empty["status"] == "INSUFFICIENT_DATA"
    assert empty["automatic_promotion"] is False

    result = evaluate_trade_lifecycle(pd.DataFrame([{"symbol": "AAA", "signal": "BUY"}]))
    assert result["status"] == "SMALL_SAMPLE"
    assert "entry_price" in result["missing_core_fields"]
    assert result["missing_facts_inferred"] is False
    assert result["v2_mode"] == "SHADOW_ONLY"


def test_phase106_treats_buy_with_recorded_exit_as_simulated_outcome_not_broker_lifecycle():
    result = evaluate_trade_lifecycle(pd.DataFrame([
        {"symbol": "AAA", "target_date": "2026-01-02", "signal": "BUY",
         "entry_price": 100, "exit_price": 105, "shares": 2, "profit_loss": 10}
    ]))
    assert result["closed_trade_rows_detected"] == 1
    assert result["exit_date_recorded"] is False
    assert result["source_record_type"] == "simulated_outcome_ledger"
    assert "not an inferred broker exit timestamp" in result["exit_date_semantics"]
