import json

import pandas as pd

from src import phase58_62_evidence as evidence


def test_accounting_diagnostics_flags_missing_costs_without_imputation(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "BUY", "symbol": "ABC", "target_date": "2026-10-12", "shares": 2,
         "entry_price": 10, "exit_price": 12, "gross_profit_loss": 4,
         "costs": None, "profit_loss": 4},
        {"signal": "SKIP", "symbol": "XYZ", "target_date": "2026-10-12", "shares": 0,
         "entry_price": None, "exit_price": None, "gross_profit_loss": 0,
         "costs": 0, "profit_loss": 0},
    ]).to_csv(tmp_path / "paper_trades.csv", index=False)
    summary, detail = evidence.accounting_diagnostics()
    assert summary["executed_rows_examined"] == 1
    assert summary["missing_cost_rows"] == 1
    assert summary["values_imputed"] is False
    assert detail.iloc[0]["missing_costs"]


def test_forecast_diagnostics_compares_baseline_by_date_and_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": "2026-10-09", "symbol": "ABC", "close_ape": 0.01, "baseline_close_ape": 0.02},
        {"target_date": "2026-10-12", "symbol": "XYZ", "close_ape": 0.03, "baseline_close_ape": 0.02},
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    summary, dates, symbols = evidence.forecast_diagnostics()
    assert summary["scored_rows"] == 2
    assert summary["beats_baseline"] is False
    assert len(dates) == 2
    assert set(symbols["symbol"]) == {"ABC", "XYZ"}


def test_v2_missing_skip_reasons_are_reported_not_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "", "phase4_reject_reason": "minimum_expected_return"},
        {"signal": "BUY", "no_trade_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    summary, breakdown = evidence.v2_diagnostics()
    assert summary["status"] == "INCOMPLETE_REASONS"
    assert summary["missing_skip_reason_rows"] == 1
    assert summary["production_filters_changed"] is False
    assert len(breakdown) == 1


def test_promotion_gate_is_never_auto_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "BUY", "symbol": "ABC", "target_date": "2026-10-12", "shares": 1,
         "entry_price": 10, "exit_price": 11, "gross_profit_loss": 1, "costs": 0.1, "profit_loss": 0.9},
    ]).to_csv(tmp_path / "paper_trades.csv", index=False)
    pd.DataFrame([
        {"target_date": "2026-10-12", "symbol": "ABC", "close_ape": 0.01, "baseline_close_ape": 0.02},
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected", "phase4_reject_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    (tmp_path / "phase52_promotion_gate.json").write_text(json.dumps({"promotion_safe": True}))
    report = evidence.build_reports()
    assert report["phase62_promotion"]["production_champion"] == "V1"
    assert report["phase62_promotion"]["v2_status"] == "SHADOW_ONLY"
    assert report["phase62_promotion"]["automatic_promotion"] is False
    assert report["phase62_promotion"]["production_settings_changed"] is False
    assert (tmp_path / "phase58_62_summary.json").exists()
