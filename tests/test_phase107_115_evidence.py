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
    assert funnel["gate_overlap_diagnostics"]["phase2_gate_output_rows"] == 2
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
    assert round(result["v1"]["max_drawdown_pct_from_100k_start"], 8) == -10.0
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
    monkeypatch.setattr(module, "_read", lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame())
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_AMBIGUOUS_KEYS"
    assert result["duplicate_evaluation_key_rows"] == 2
    assert result.get("duplicate_actual_key_rows", 0) == 0
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


def test_phase114_requires_exit_attribution_and_never_auto_promotes(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    recon = {"status": "PASS"}
    funnel = {
        "status": "SCORED", "skip_rows_without_reason": 0,
        "existing_funnel_valid": True, "required_gate_stages_present": True,
    }
    forecast = {"status": "SCORED"}
    replay = {"status": "SCORED"}
    sustained = {"status": "SCORED"}
    loss = {"status": "SCORED"}
    incomplete_attribution = {"status": "SCORED", "unknown_or_unrecorded_reason_rows": 1}
    blocked = phase114_promotion_gate(
        recon, funnel, forecast, replay, sustained, incomplete_attribution, loss
    )
    assert blocked["gates"]["exit_reason_attribution_complete"] is False
    assert blocked["automatic_promotion"] is False
    complete_attribution = {"status": "SCORED", "unknown_or_unrecorded_reason_rows": 0}
    review = phase114_promotion_gate(
        recon, funnel, forecast, replay, sustained, complete_attribution, loss
    )
    assert review["status"] == "REVIEW_REQUIRED"
    assert review["gates"]["loss_concentration_evidence_available"] is True
    assert review["automatic_promotion"] is False
    assert review["production_changes_authorized"] is False


def test_phase107_validates_fixed_share_source_evidence_without_backfilling(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "target_date": "2026-10-08",
        "entry_price": 348.5, "exit_price": 356.6499938964844,
        "quantity": 58, "position_value": 20213.0,
        "planned_capital": 19949.1, "reference_price": 343.95,
        "pre_split_quantity": 58, "split_ratio": 1.0,
        "profit_loss": 472.69964599609375, "return_pct": 2.338592222807568,
        "shares": None, "allocated_capital": None,
        "gross_profit_loss": None, "costs": None,
    }])
    original = trades.copy(deep=True)
    result = phase107_pnl_reconciliation(trades)
    evidence = result["fixed_share_source_evidence"]
    assert evidence["status"] == "GROSS_SOURCE_EVIDENCE_VALIDATED"
    assert evidence["candidate_rows"] == 1
    assert evidence["validated_rows"] == 1
    assert evidence["mismatch_rows"] == 0
    assert evidence["costs_modeled"] is False
    assert evidence["net_profit_loss_verified"] is False
    assert evidence["source_ledger_mutated"] is False
    assert result["status"] == "BLOCKED"
    pd.testing.assert_frame_equal(trades, original)


def test_phase107_fixed_share_source_evidence_rejects_inconsistent_quantity(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "entry_price": 100, "exit_price": 110,
        "quantity": 2, "position_value": 200, "planned_capital": 190,
        "reference_price": 95, "pre_split_quantity": 2, "split_ratio": 1,
        "profit_loss": 25, "return_pct": 12.5,
    }])
    result = phase107_pnl_reconciliation(trades)
    evidence = result["fixed_share_source_evidence"]
    assert evidence["candidate_rows"] == 1
    assert evidence["validated_rows"] == 0
    assert evidence["mismatch_rows"] == 1
    assert evidence["status"] == "NO_VALID_SOURCE_EVIDENCE"



def test_v2_17_sessions_and_100_trades_meet_sample_thresholds(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    dates_v2 = pd.date_range("2026-01-01", periods=17, freq="B")
    v2 = pd.DataFrame([
        {"signal": "BUY", "target_date": dates_v2[i % 17], "profit_loss": 1.0}
        for i in range(100)
    ])
    v1_dates = pd.date_range("2025-10-01", periods=60, freq="B")
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": date, "profit_loss": 1.0}
        for date in v1_dates
    ])

    sustained = phase113_sustained_evidence(v1, v2)
    assert sustained["status"] == "SCORED"
    assert sustained["v2"]["executed_rows"] == 100
    assert sustained["v2"]["distinct_sessions"] == 17
    assert sustained["v2"]["minimum_trade_count_met"] is True
    assert sustained["v2"]["minimum_session_count_met"] is True
    assert sustained["minimum_distinct_sessions"] == 17

    replay = module.phase111_chronological_replay(v1, v2)
    assert replay["status"] == "SCORED"
    assert replay["minimum_v1_sessions_for_evidence"] == 60
    assert replay["minimum_v2_sessions_for_evidence"] == 17


def test_v2_17_sessions_without_100_trades_does_not_meet_sustained_gate(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    dates = pd.date_range("2026-01-01", periods=17, freq="B")
    v2 = pd.DataFrame([
        {"signal": "BUY", "target_date": dates[i % 17], "profit_loss": 1.0}
        for i in range(99)
    ])
    result = phase113_sustained_evidence(pd.DataFrame(), v2)
    assert result["v2"]["minimum_session_count_met"] is True
    assert result["v2"]["minimum_trade_count_met"] is False
    assert result["status"] == "INSUFFICIENT_EVIDENCE"



def test_phase110_blocks_non_finite_or_non_positive_forecast_prices(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([
        {"symbol": "AAA", "target_date": "2026-01-02", "predicted_close": float("inf"), "base_close": 100},
    ])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02", "open": 100, "high": 107, "low": 99, "close": 104,
    }])
    monkeypatch.setattr(module, "_read", lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame())
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_INVALID_METRICS"
    assert result["invalid_metric_rows"] == 1
    assert result["metrics"] == {}


def test_phase111_blocks_non_finite_replay_pnl(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": float("inf")},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase111_chronological_replay(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["valid"] is False
    assert result["v1"]["invalid_metric_rows"] == 1
    assert result["v1"]["net_pnl"] is None



def test_phase107_rejects_incomplete_split_source_metadata(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "entry_price": 100, "exit_price": 110,
        "quantity": 2, "position_value": 200, "planned_capital": 190,
        "reference_price": 95, "pre_split_quantity": None, "split_ratio": 1,
        "profit_loss": 20, "return_pct": 10,
    }])
    result = phase107_pnl_reconciliation(trades)
    evidence = result["fixed_share_source_evidence"]
    assert evidence["validated_rows"] == 0
    assert evidence["invalid_split_metadata_rows"] == 1
    assert evidence["invalid_or_incomplete_source_rows"] == 1


def test_phase107_rejects_unpaired_split_source_metadata(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "entry_price": 100, "exit_price": 110,
        "quantity": 2, "position_value": 200, "planned_capital": 190,
        "reference_price": 95, "pre_split_quantity": 2,
        "profit_loss": 20, "return_pct": 10,
    }])
    result = phase107_pnl_reconciliation(trades)
    evidence = result["fixed_share_source_evidence"]
    assert evidence["validated_rows"] == 0
    assert evidence["invalid_split_metadata_rows"] == 1



def test_phase107_blocks_non_finite_canonical_accounting_values(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([{
        "signal": "BUY", "entry_price": 100, "exit_price": 110, "shares": 2,
        "allocated_capital": 200, "gross_profit_loss": float("inf"), "costs": 1,
        "profit_loss": 19, "return_pct": 9.5,
    }])
    result = phase107_pnl_reconciliation(trades)
    assert result["status"] == "BLOCKED"
    assert result["invalid_non_finite_rows"] == 1
    assert result["invalid_non_finite_values_by_field"]["gross_profit_loss"] == 1


def test_phase108_blocks_non_finite_pnl_instead_of_scoring_losses(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    trades = pd.DataFrame([
        {"signal": "BUY", "symbol": "AAA", "profit_loss": float("-inf")},
        {"signal": "BUY", "symbol": "BBB", "profit_loss": 10},
    ])
    result = phase108_loss_concentration(trades)
    assert result["status"] == "BLOCKED_INVALID_METRICS"
    assert result["invalid_non_finite_rows"] == 1
    assert result["gross_loss"] is None
    assert result["group_reports"] == []



def test_phase110_blocks_non_finite_base_close_before_directional_scoring(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([{
        "symbol": "AAA", "target_date": "2026-01-02",
        "predicted_close": 105, "base_close": float("inf"),
    }])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02",
        "open": 100, "high": 107, "low": 99, "close": 104,
    }])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame(),
    )
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_INVALID_METRICS"
    assert result["invalid_metric_rows"] == 1
    assert result["metrics"] == {}
    assert result["close_directional_accuracy_pct"] is None



def test_phase109_blocks_malformed_persisted_funnel_counts(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "confidence_percentile_gate", "input_rows": 10, "output_rows": "invalid", "rejected_rows": 5},
        {"stage": "phase2_selected_gate", "input_rows": 5, "output_rows": 2.5, "rejected_rows": 2},
        {"stage": "phase4_eligibility_gate", "input_rows": 2, "output_rows": 1, "rejected_rows": 1},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    v2 = pd.DataFrame([{"signal": "SKIP", "no_trade_reason": "phase2_not_selected"}])
    result = module.phase109_selection_funnel(v2)
    assert result["existing_funnel_valid"] is False
    assert "invalid_output_rows" in result["funnel_validation_errors"]
    assert result["required_gate_stages_present"] is False
    assert result["zero_buy_bottleneck"] == "INCOMPLETE_FUNNEL_STAGE_DIAGNOSTICS"



def test_phase111_blocks_unparseable_pnl_and_missing_session_dates(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": "not-a-number"},
        {"signal": "BUY", "target_date": "invalid-date", "profit_loss": 25},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase111_chronological_replay(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["valid"] is False
    assert result["v1"]["invalid_metric_rows"] == 2
    assert result["v1"]["invalid_date_rows"] == 1
    assert result["v1"]["invalid_pnl_rows"] == 1
    assert result["v1"]["net_pnl"] is None



def test_phase110_blocks_overflowed_derived_forecast_metrics(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([{
        "symbol": "AAA", "target_date": "2026-01-02",
        "predicted_close": 1e308, "base_close": 100,
    }])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02",
        "open": 100, "high": 1e308, "low": 1e-308, "close": 1e-308,
    }])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame(),
    )
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_INVALID_METRICS"
    assert result["metrics"] == {}
    assert result["close_directional_accuracy_pct"] is None



def test_phase110_blocks_invalid_actual_ohlc_relationships(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([{
        "symbol": "AAA", "target_date": "2026-01-02", "predicted_close": 105, "base_close": 100,
    }])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02",
        "open": 100, "high": 98, "low": 99, "close": 104,
    }])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame(),
    )
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_INVALID_OHLC"
    assert result["invalid_actual_ohlc_rows"] == 1
    assert result["metrics"] == {}


def test_phase110_blocks_unparseable_forecast_dates(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pred = pd.DataFrame([{
        "symbol": "AAA", "target_date": "not-a-date", "predicted_close": 105, "base_close": 100,
    }])
    history = pd.DataFrame([{
        "symbol": "AAA", "date": "2026-01-02",
        "open": 100, "high": 107, "low": 99, "close": 104,
    }])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: pred.merge(history.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"}), on=["symbol", "target_date"], how="left") if name == "evaluations.csv" else pd.DataFrame(),
    )
    result = module.phase110_forecast_accuracy()
    assert result["status"] == "BLOCKED_INVALID_KEYS"
    assert result["invalid_evaluation_key_rows"] == 1
    assert result["metrics"] == {}



def test_phase109_accepts_unclassified_phase2_metadata_rows(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "phase2_join", "input_rows": 10, "output_rows": 8, "rejected_rows": None},
        {"stage": "confidence_percentile_gate", "input_rows": 8, "output_rows": 4, "rejected_rows": 4},
        {"stage": "phase2_selected_gate", "input_rows": 4, "output_rows": 2, "rejected_rows": 2},
        {"stage": "phase4_eligibility_gate", "input_rows": 2, "output_rows": 1, "rejected_rows": 1},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    v2 = pd.DataFrame([{"signal": "SKIP", "no_trade_reason": "phase2_not_selected"}])
    result = module.phase109_selection_funnel(v2)
    assert result["existing_funnel_valid"] is True
    assert result["funnel_validation_errors"] == []
    assert result["funnel_stages"][0]["rejected_rows"] == 0
    assert result["funnel_stages"][0]["unclassified_rows"] == 2
    assert result["funnel_stages"][1]["unclassified_rows"] == 0
    assert result["required_gate_stages_present"] is True


def test_phase109_blocks_funnel_output_greater_than_input(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "confidence_percentile_gate", "input_rows": 2, "output_rows": 3, "rejected_rows": -1},
        {"stage": "phase2_selected_gate", "input_rows": 3, "output_rows": 2, "rejected_rows": 1},
        {"stage": "phase4_eligibility_gate", "input_rows": 2, "output_rows": 1, "rejected_rows": 1},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    v2 = pd.DataFrame([{"signal": "SKIP", "no_trade_reason": "phase2_not_selected"}])
    result = module.phase109_selection_funnel(v2)
    assert result["existing_funnel_valid"] is False
    assert "output_rows_exceed_input_rows" in result["funnel_validation_errors"]



def test_phase113_blocks_invalid_executed_session_dates(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": 10},
        {"signal": "BUY", "target_date": "not-a-date", "profit_loss": 20},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase113_sustained_evidence(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["valid"] is False
    assert result["v1"]["invalid_session_date_rows"] == 1


def test_phase114_requires_loss_concentration_evidence(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    recon = {"status": "PASS"}
    funnel = {
        "status": "SCORED", "skip_rows_without_reason": 0,
        "existing_funnel_valid": True, "required_gate_stages_present": True,
    }
    forecast = {"status": "SCORED"}
    replay = {"status": "SCORED"}
    sustained = {"status": "SCORED"}
    attribution = {"status": "SCORED", "unknown_or_unrecorded_reason_rows": 0}
    missing_loss_evidence = phase114_promotion_gate(
        recon, funnel, forecast, replay, sustained, attribution,
        {"status": "SMALL_SAMPLE"},
    )
    assert missing_loss_evidence["status"] == "INSUFFICIENT_EVIDENCE"
    assert missing_loss_evidence["gates"]["loss_concentration_evidence_available"] is False
    assert "loss_concentration_evidence_available" in missing_loss_evidence["unmet_gates"]
    assert missing_loss_evidence["automatic_promotion"] is False


def test_phase112_requires_index_dates_to_match_latest_ohlcv_session(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    intelligence = pd.DataFrame([
        {"index": "NIFTY50", "as_of": "2026-11-13"},
        {"index": "BANKNIFTY", "as_of": "2026-11-12"},
    ])
    history = pd.DataFrame([{"date": "2026-11-13"}])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: intelligence.copy() if name == "index_intelligence_summary.csv" else history.copy(),
    )
    result = module.phase112_market_intelligence()
    assert result["status"] == "STALE_DATA"
    assert result["reference_ohlcv_session"] == "2026-11-13"
    assert result["stale_index_rows"] == 1


def test_phase112_rejects_invalid_or_duplicate_index_metadata(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    intelligence = pd.DataFrame([
        {"index": "NIFTY50", "as_of": "not-a-date"},
        {"index": "NIFTY50", "as_of": "2026-11-13"},
    ])
    history = pd.DataFrame([{"date": "2026-11-13"}])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: intelligence.copy() if name == "index_intelligence_summary.csv" else history.copy(),
    )
    result = module.phase112_market_intelligence()
    assert result["status"] == "BLOCKED_INVALID_DATA"
    assert result["invalid_as_of_rows"] == 1
    assert result["duplicate_index_rows"] == 2


def test_phase112_scores_aligned_index_session_metadata(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    intelligence = pd.DataFrame([
        {"index": "NIFTY50", "as_of": "2026-11-13"},
        {"index": "BANKNIFTY", "as_of": "2026-11-13"},
    ])
    history = pd.DataFrame([{"date": "2026-11-13"}])
    monkeypatch.setattr(
        module, "_read",
        lambda name, **kwargs: intelligence.copy() if name == "index_intelligence_summary.csv" else history.copy(),
    )
    result = module.phase112_market_intelligence()
    assert result["status"] == "SCORED"
    assert result["stale_index_rows"] == 0
    assert result["future_index_rows"] == 0


def test_phase109_reports_missing_rejection_metadata_without_inventing_rejections(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "phase2_join", "input_rows": 12, "output_rows": 9, "rejected_rows": None},
        {"stage": "confidence_percentile_gate", "input_rows": 9, "output_rows": 5, "rejected_rows": 4},
        {"stage": "phase2_selected_gate", "input_rows": 5, "output_rows": 3, "rejected_rows": 2},
        {"stage": "phase4_eligibility_gate", "input_rows": 3, "output_rows": 2, "rejected_rows": 1},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    result = module.phase109_selection_funnel(pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected"}
    ]))
    phase2_join = result["funnel_stages"][0]
    assert result["existing_funnel_valid"] is True
    assert phase2_join["rejected_rows"] == 0
    assert phase2_join["unclassified_rows"] == 3
    assert result["funnel_stages"][1]["unclassified_rows"] == 0



def test_phase111_blocks_finite_trade_values_that_overflow_daily_aggregation(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": 1e308},
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": 1e308},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase111_chronological_replay(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["valid"] is False
    assert result["v1"]["invalid_derived_metric"] == "daily_session_pnl_overflow"
    assert result["v1"]["net_pnl"] is None


def test_phase111_blocks_equity_overflow_across_valid_sessions(tmp_path, monkeypatch):
    import src.phase107_115_evidence as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    v1 = pd.DataFrame([
        {"signal": "BUY", "target_date": "2026-01-02", "profit_loss": 1e308},
        {"signal": "BUY", "target_date": "2026-01-05", "profit_loss": 1e308},
    ])
    v2 = pd.DataFrame(columns=["signal", "target_date", "profit_loss"])
    result = module.phase111_chronological_replay(v1, v2)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["v1"]["valid"] is False
    assert result["v1"]["invalid_derived_metric"] == "equity_or_aggregate_overflow"
    assert result["v1"]["max_drawdown_pct_from_100k_start"] is None
