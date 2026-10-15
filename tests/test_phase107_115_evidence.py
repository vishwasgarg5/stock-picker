import pandas as pd

from src.phase107_115_evidence import (
    phase107_pnl_reconciliation,
    phase108_loss_concentration,
    phase109_selection_funnel,
    phase113_sustained_evidence,
    phase114_promotion_gate,
)


def test_phase107_reconciles_recorded_accounting_without_mutation(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([
        {"signal": "BUY", "entry_price": 100, "exit_price": 110, "shares": 2,
         "allocated_capital": 200, "gross_profit_loss": 20, "costs": 1,
         "profit_loss": 19, "return_pct": 10},
        {"signal": "SKIP", "entry_price": None, "exit_price": None, "shares": 0,
         "allocated_capital": 0, "gross_profit_loss": 0, "costs": 0,
         "profit_loss": 0, "return_pct": 0},
    ])
    original = trades.copy(deep=True)
    result = phase107_pnl_reconciliation(trades)
    assert result["status"] == "PASS"
    assert result["buy_rows"] == 1
    pd.testing.assert_frame_equal(trades, original)


def test_phase107_blocks_accounting_mismatch(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{"signal": "BUY", "entry_price": 100, "exit_price": 110,
        "shares": 2, "allocated_capital": 200, "gross_profit_loss": 20, "costs": 1,
        "profit_loss": 17, "return_pct": 10}])
    assert phase107_pnl_reconciliation(trades)["status"] == "BLOCKED"


def test_phase108_and_109_are_diagnostic_only(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "symbol": "AAA", "target_date": "2026-01-01", "exit_reason": "stop_loss", "profit_loss": -5},
        {"signal": "BUY", "symbol": "BBB", "target_date": "2026-01-01", "exit_reason": "target", "profit_loss": 8},
    ])
    summary = phase108_loss_concentration(v1)
    assert summary["gross_loss"] == 5
    v2 = pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "", "phase4_reject_reason": "minimum_expected_return",
         "confidence_pct": 0.2, "symbol": "AAA", "target_date": "2026-01-01"},
        {"signal": "BUY", "no_trade_reason": "", "phase4_reject_reason": "",
         "confidence_pct": 0.9, "symbol": "BBB", "target_date": "2026-01-01"},
    ])
    pd.DataFrame([{"stage": "phase2_selected_gate", "input_rows": 10,
                   "output_rows": 2, "rejected_rows": 8}]).to_csv(
        tmp_path / "phase32_v2_selection_funnel.csv", index=False
    )
    funnel = phase109_selection_funnel(v2)
    assert funnel["skip_rows_without_reason"] == 1
    assert funnel["phase4_reject_reason_counts"] == {"minimum_expected_return": 1}
    assert funnel["confidence_column_diagnostics"]["confidence_pct"]["non_null_rows"] == 2
    assert funnel["funnel_stages"][0]["rejected_rows"] == 8
    assert funnel["production_model_changed"] is False


def test_phase113_requires_sustained_v2_evidence_and_gate_never_promotes(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([{"signal": "BUY", "target_date": "2026-01-01", "profit_loss": 1}])
    v2 = pd.DataFrame([{"signal": "BUY", "target_date": "2026-01-01", "profit_loss": 1}])
    sustained = phase113_sustained_evidence(v1, v2)
    assert sustained["status"] == "INSUFFICIENT_EVIDENCE"
    gate = phase114_promotion_gate(
        {"status": "PASS"}, {"status": "SCORED", "skip_rows_without_reason": 0},
        {"status": "SCORED"}, {"status": "SCORED"}, sustained
    )
    assert gate["status"] == "INSUFFICIENT_EVIDENCE"
    assert gate["automatic_promotion"] is False
    assert gate["v2_remains_shadow_only"] is True

def test_phase107_localizes_missing_accounting_by_field_and_session_without_inference(tmp_path, monkeypatch):
    import json
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-09-29", "entry_price": 100, "exit_price": 105,
         "shares": None, "allocated_capital": None, "gross_profit_loss": None, "costs": None,
         "profit_loss": 5, "return_pct": 5},
        {"signal": "BUY", "target_date": "2026-10-01", "entry_price": 100, "exit_price": 105,
         "shares": 2, "allocated_capital": 200, "gross_profit_loss": 10, "costs": 1,
         "profit_loss": 9, "return_pct": 4.5},
    ])
    original = trades.copy(deep=True)
    result = phase107_pnl_reconciliation(trades)
    assert result["status"] == "BLOCKED"
    assert result["missing_values_by_field"]["shares"] == 1
    assert result["missing_values_by_field"]["allocated_capital"] == 1
    assert result["incomplete_rows_by_session"] == [
        {"target_date": "2026-09-29", "buy_rows": 1, "incomplete_rows": 1}
    ]
    assert result["record_vintage_classification"] == "UNCLASSIFIED_NO_SCHEMA_VERSION_OR_CREATION_TIMESTAMP"
    assert result["missing_facts_inferred"] is False
    pd.testing.assert_frame_equal(trades, original)
    persisted = json.loads((tmp_path / "phase107_pnl_reconciliation_summary.json").read_text())
    assert persisted["policy"].find("not treated as proof") >= 0

