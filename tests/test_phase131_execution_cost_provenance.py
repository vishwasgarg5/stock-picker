import pandas as pd

from src.phase131_execution_cost_provenance import evaluate_execution_cost_provenance


def test_costs_are_source_verified_only_with_complete_components_and_reference():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "target_date": "2026-11-16",
        "quantity": 2, "gross_profit_loss": 20, "costs": 1.0, "profit_loss": 19,
        "brokerage": 0.2, "stt": 0.3, "gst": 0.1, "exchange_charges": 0.1,
        "sebi_charges": 0.1, "stamp_duty": 0.1, "other_charges": 0.1,
        "cost_evidence_reference": "contract-note-123",
    }])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert detail.iloc[0]["status"] == "SOURCE_REFERENCED_COMPONENTS_RECONCILED"
    assert summary["source_referenced_rows"] == 1
    assert summary["status"] == "PASS"
    assert summary["costs_inferred"] is False
    assert summary["canonical_ledger_mutated"] is False


def test_recorded_cost_identity_without_source_reference_is_review_only():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "quantity": 1,
        "gross_profit_loss": 10, "costs": 0.5, "profit_loss": 9.5,
    }])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert detail.iloc[0]["status"] == "LEDGER_ARITHMETIC_ONLY_SOURCE_NOT_ESTABLISHED"
    assert summary["status"] == "REVIEW"
    assert summary["source_referenced_rows"] == 0


def test_missing_costs_are_blocked_and_never_reconstructed_from_gross_and_net():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "quantity": 1,
        "gross_profit_loss": 10, "costs": None, "profit_loss": 9.5,
    }])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert pd.isna(detail.iloc[0]["recorded_costs"])
    assert detail.iloc[0]["status"] == "BLOCKED_MISSING_RECORDED_ACCOUNTING"
    assert summary["blocked_rows"] == 1
    assert summary["costs_inferred"] is False


def test_cost_mismatch_and_conflicting_cost_columns_fail():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "quantity": 1,
        "gross_profit_loss": 10, "costs": 1, "trading_cost": 2,
        "profit_loss": 9,
    }])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert detail.iloc[0]["status"] == "FAIL_RECONCILIATION_MISMATCH"
    assert summary["mismatch_rows"] == 1
    assert summary["status"] == "FAIL"


def test_skips_are_excluded_and_input_is_not_mutated():
    trades = pd.DataFrame([
        {"signal": "SKIP", "symbol": "XYZ", "quantity": 0, "costs": None},
        {"signal": "BUY", "symbol": "ABC", "quantity": 1,
         "gross_profit_loss": 10, "costs": 1, "profit_loss": 9},
    ])
    original = trades.copy(deep=True)
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert len(detail) == 1
    assert summary["executed_rows"] == 1
    pd.testing.assert_frame_equal(trades, original)



def test_legacy_recorded_cost_matches_simulation_rate_but_is_not_broker_evidence():
    trades = pd.DataFrame([{
        "signal": "BUY", "symbol": "ABC", "target_date": "2026-11-16",
        "shares": 10, "allocated_capital": 10000,
        "gross_profit_loss": 25, "costs": 10, "profit_loss": 15,
    }])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert detail.iloc[0]["recorded_cost_rate"] == 0.001
    assert bool(detail.iloc[0]["matches_current_simulation_rate"]) is True
    assert summary["configured_cost_model"]["rate"] == 0.001
    assert summary["configured_cost_model"]["actual_broker_cost_evidence"] is False
    assert summary["status"] == "REVIEW"


def test_all_skip_ledger_reports_no_executed_trades():
    trades = pd.DataFrame([{"signal": "SKIP", "quantity": 0, "costs": None}])
    detail, summary = evaluate_execution_cost_provenance(trades)
    assert detail.empty
    assert summary["status"] == "NO_EXECUTED_TRADES"
    assert summary["executed_rows"] == 0
