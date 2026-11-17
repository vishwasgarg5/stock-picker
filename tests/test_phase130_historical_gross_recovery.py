import pandas as pd

from src.phase130_historical_gross_recovery import evaluate_historical_gross_recovery


def test_recovers_only_source_validated_gross_fields_without_net_or_exit_inference():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 110.0, "profit_loss": 20.0,
        "shares": None, "allocated_capital": None, "gross_profit_loss": None,
        "costs": None, "quantity": 2.0, "position_value": 200.0,
        "planned_capital": 190.0, "reference_price": 95.0, "return_pct": 10.0,
        "pre_split_quantity": 2.0, "split_ratio": 1.0,
    }])
    detail, summary = evaluate_historical_gross_recovery(trades)
    assert summary["gross_source_recoverable_rows"] == 1
    assert detail.iloc[0]["validated_shares"] == 2.0
    assert detail.iloc[0]["validated_allocated_capital"] == 200.0
    assert detail.iloc[0]["validated_gross_profit_loss"] == 20.0
    assert summary["costs_recovered_rows"] == 0
    assert summary["net_profit_loss_recovered_rows"] == 0
    assert summary["exit_reason_recovered_rows"] == 0
    assert summary["canonical_ledger_backfilled"] is False


def test_rejects_source_identity_mismatch():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 110.0, "profit_loss": 21.0, "return_pct": 10.5,
        "shares": None, "allocated_capital": None, "gross_profit_loss": None,
        "costs": None, "quantity": 2.0, "position_value": 200.0,
        "planned_capital": 190.0, "reference_price": 95.0,
        "pre_split_quantity": 2.0, "split_ratio": 1.0,
    }])
    _, summary = evaluate_historical_gross_recovery(trades)
    assert summary["gross_source_recoverable_rows"] == 0
    assert summary["gross_source_unresolved_rows"] == 1


def test_recovers_legacy_fixed_notional_shares_only_when_pnl_and_return_agree():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "OLD", "target_date": "2026-09-29",
        "entry_price": 100.0, "exit_price": 110.0, "profit_loss": 2000.0,
        "return_pct": 10.0, "shares": None, "allocated_capital": None,
        "gross_profit_loss": None, "costs": None, "position_value": 20000.0,
    }])
    detail, summary = evaluate_historical_gross_recovery(trades)
    assert summary["gross_source_recoverable_rows"] == 1
    assert abs(detail.iloc[0]["validated_shares"] - 200.0) < 1e-9
    assert detail.iloc[0]["validated_allocated_capital"] == 20000.0
    assert detail.iloc[0]["validated_gross_profit_loss"] == 2000.0


def test_accepts_valid_flat_price_trade_without_dividing_by_zero():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "FLAT", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 100.0, "profit_loss": 0.0,
        "return_pct": 0.0, "shares": None, "allocated_capital": None,
        "gross_profit_loss": None, "costs": None, "position_value": 20000.0,
    }])
    detail, summary = evaluate_historical_gross_recovery(trades)
    assert summary["gross_source_recoverable_rows"] == 1
    assert detail.iloc[0]["validated_shares"] == 200.0
    assert detail.iloc[0]["validated_gross_profit_loss"] == 0.0


def test_does_not_recover_zero_allocation_rows():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ZERO", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 100.0, "profit_loss": 0.0,
        "return_pct": 0.0, "shares": None, "allocated_capital": None,
        "gross_profit_loss": None, "costs": None, "position_value": 0.0,
    }])
    _, summary = evaluate_historical_gross_recovery(trades)
    assert summary["gross_source_recoverable_rows"] == 0
    assert summary["gross_source_unresolved_rows"] == 1



def test_validates_canonical_share_capital_gross_and_return_identities():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "CANON", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 110.0, "profit_loss": 19.0,
        "return_pct": 9.5, "shares": 2.0, "allocated_capital": 200.0,
        "gross_profit_loss": 20.0, "costs": 1.0, "position_value": None,
    }])
    original = trades.copy(deep=True)
    detail, summary = evaluate_historical_gross_recovery(trades)
    assert summary["canonical_complete_rows"] == 1
    assert summary["canonical_source_validated_rows"] == 1
    assert summary["gross_source_unresolved_rows"] == 0
    assert detail.iloc[0]["recovery_status"] == "CANONICAL_FIELDS_SOURCE_VALIDATED"
    assert detail.iloc[0]["validated_shares"] == 2.0
    assert detail.iloc[0]["validated_allocated_capital"] == 200.0
    assert detail.iloc[0]["validated_gross_profit_loss"] == 20.0
    pd.testing.assert_frame_equal(trades, original)


def test_canonical_field_presence_does_not_hide_identity_mismatch():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "BAD", "target_date": "2026-11-16",
        "entry_price": 100.0, "exit_price": 110.0, "profit_loss": 18.0,
        "return_pct": 9.0, "shares": 2.0, "allocated_capital": 200.0,
        "gross_profit_loss": 19.0, "costs": 1.0, "position_value": None,
    }])
    detail, summary = evaluate_historical_gross_recovery(trades)
    assert summary["canonical_complete_rows"] == 1
    assert summary["canonical_source_validated_rows"] == 0
    assert summary["gross_source_unresolved_rows"] == 1
    assert detail.iloc[0]["recovery_status"] == "CANONICAL_FIELDS_PRESENT_IDENTITY_MISMATCH"
    assert pd.isna(detail.iloc[0]["validated_gross_profit_loss"])
