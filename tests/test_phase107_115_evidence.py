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
         "profit_loss": 19, "return_pct": 9.5},
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
    assert funnel["gate_overlap_diagnostics"]["phase2_flag_persisted_in_decision_ledger"] is False
    assert funnel["gate_overlap_diagnostics"]["phase2_gate_output_rows"] == 0
    assert funnel["skip_reasons_by_phase2_flag"] == {}
    assert funnel["skip_reason_cross_tab_status"].startswith("UNAVAILABLE_PHASE2_FLAG_NOT_PERSISTED")


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
    assert result["record_vintage_classification"] == "EXPLICIT_METADATA_WHEN_AVAILABLE; OTHERWISE_UNCLASSIFIED"
    assert result["incomplete_rows_by_record_vintage"] == {"UNCLASSIFIED_NO_EXPLICIT_METADATA": 1}
    assert result["missing_facts_inferred"] is False
    pd.testing.assert_frame_equal(trades, original)
    persisted = json.loads((tmp_path / "phase107_pnl_reconciliation_summary.json").read_text())
    assert persisted["policy"].find("not treated as proof") >= 0



def test_phase111_drawdown_includes_starting_capital_and_reports_risk_metrics(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": -10000.0},
        {"signal": "BUY", "target_date": "2026-01-05", "profit_loss": 2000.0},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase111_chronological_replay(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["max_drawdown_pct_from_100k_start"] == -10.0
    assert result["v1"]["win_rate_pct"] == 50.0
    assert result["v1"]["profit_factor"] == 0.2
    assert result["v2"]["sessions"] == 0
    assert result["v2"]["max_drawdown_pct_from_100k_start"] is None


def test_phase107_blocks_return_percentage_mismatch(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "entry_price": 100, "exit_price": 110, "shares": 10,
        "allocated_capital": 1000, "gross_profit_loss": 100, "costs": 10,
        "profit_loss": 90, "return_pct": 10,
    }])
    result = phase107_pnl_reconciliation(trades)
    assert result["status"] == "BLOCKED"
    assert result["reconciliation_mismatch_rows"] == 1
    assert result["max_abs_return_delta_pct_points"] == 1.0


def test_phase110_blocks_duplicate_forecast_keys_instead_of_overweighting_them(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([
        {"symbol": "AAA", "target_date": "2026-01-02", "predicted_close": 105, "base_close": 100},
        {"symbol": "AAA", "target_date": "2026-01-02", "predicted_close": 106, "base_close": 100},
    ])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02", "open": 100, "high": 107, "low": 99, "close": 104,
    }])
    monkeypatch.setattr(module, "_read", lambda name, **kwargs: pred.copy() if name == "predictions.csv" else history.copy())
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_AMBIGUOUS_KEYS"
    assert result["duplicate_prediction_key_rows"] == 2
    assert result["duplicate_actual_key_rows"] == 0
    assert result["metrics"] == {}


def test_phase109_identifies_earliest_zero_buy_funnel_bottleneck(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "confidence_percentile_gate", "input_rows": 10, "output_rows": 5, "rejected_rows": 5},
        {"stage": "phase2_selected_gate", "input_rows": 5, "output_rows": 0, "rejected_rows": 5},
        {"stage": "phase4_eligibility_gate", "input_rows": 0, "output_rows": 0, "rejected_rows": 0},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    v2 = pd.DataFrame([{
        "signal": "SKIP", "no_trade_reason": "phase2_not_selected",
        "confidence_pct": 0.9, "symbol": "AAA", "target_date": "2026-01-02",
    }])
    result = phase109_selection_funnel(v2)
    assert result["buy_rows"] == 0
    assert result["zero_buy_bottleneck"] == "PHASE2_SELECTED_GATE_ZERO_OUTPUT"
    assert result["required_gate_stages_present"] is True
