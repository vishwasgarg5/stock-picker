import json

import pandas as pd

from src.phase48_52_evidence import _ledger, build_reports


def test_ledger_does_not_impute_missing_accounting_fields():
    frame = pd.DataFrame([
        {"symbol": "ABC", "target_date": "2026-10-01", "signal": "BUY",
         "shares": 2, "quantity": None, "gross_profit_loss": 12.0,
         "costs": None, "profit_loss": 12.0},
        {"symbol": "XYZ", "target_date": "2026-10-01", "signal": "SKIP",
         "shares": 0, "quantity": 0, "gross_profit_loss": 0.0,
         "costs": 0.0, "profit_loss": 0.0},
    ])
    summary, detail = _ledger(frame, "V1", "gross_profit_loss", "costs", "profit_loss")
    assert summary["executed_trades"] == 1
    assert summary["incomplete_rows"] == 1
    assert summary["ledger_status"] == "INCOMPLETE_OR_MISMATCH"
    assert summary["missing_values_imputed"] is False
    assert detail.iloc[0]["status"] == "MISSING_ACCOUNTING_FIELD"


def test_ledger_reconciles_gross_less_costs_to_net():
    frame = pd.DataFrame([
        {"symbol": "ABC", "target_date": "2026-10-01", "signal": "BUY",
         "shares": 2, "gross_profit_loss": 12.0, "costs": 2.0, "profit_loss": 10.0},
    ])
    summary, detail = _ledger(frame, "V1", "gross_profit_loss", "costs", "profit_loss")
    assert summary["ledger_status"] == "RECONCILED"
    assert summary["net_pnl"] == 10.0
    assert abs(detail.iloc[0]["reconciliation_delta"]) < 0.001


def test_phase48_52_outputs_keep_v1_champion_and_no_auto_promotion(tmp_path, monkeypatch):
    import src.phase48_52_evidence as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    pd.DataFrame([
        {"target_date": "2026-10-01", "close_ape": 0.01, "baseline_close_ape": 0.02}
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    pd.DataFrame([
        {"target_date": "2026-10-01", "v1_net_pnl": 1.0, "v2_net_pnl": 0.0}
    ]).to_csv(tmp_path / "phase31_chronological_comparison.csv", index=False)
    pd.DataFrame([
        {"signal": "SKIP", "quantity": 0, "symbol": "ABC", "target_date": "2026-10-02",
         "gross_profit_loss": 0.0, "trading_cost": 0.0, "profit_loss": 0.0,
         "no_trade_reason": "phase4_ineligible", "phase4_reject_reason": "minimum_expected_return"}
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([
        {"signal": "BUY", "shares": 1, "symbol": "ABC", "target_date": "2026-10-01",
         "gross_profit_loss": 2.0, "costs": 1.0, "profit_loss": 1.0}
    ]).to_csv(tmp_path / "paper_trades.csv", index=False)

    summary = build_reports()
    gate = json.loads((tmp_path / "phase52_promotion_gate.json").read_text())
    assert summary["production_champion"] == "V1"
    assert summary["v2_promoted"] is False
    assert gate["automatic_promotion"] is False
    assert gate["production_settings_changed"] is False
    assert (tmp_path / "phase48_ledger_reconciliation.csv").exists()
    assert (tmp_path / "phase49_forecast_accuracy.csv").exists()
    assert (tmp_path / "phase50_v2_rejection_reasons.csv").exists()
    assert (tmp_path / "phase51_return_stress.csv").exists()
    assert (tmp_path / "phase49_forecast_by_symbol.csv").exists()
    assert (tmp_path / "phase50_v2_rejection_summary.json").exists()

def test_ledger_uses_rowwise_quantity_fallback_and_reconstructs_gross():
    frame = pd.DataFrame([
        {"symbol": "OLD", "target_date": "2026-10-01", "signal": "BUY",
         "quantity": None, "shares": 2, "entry_price": 10.0, "exit_price": 15.0,
         "gross_profit_loss": None, "costs": 1.0, "profit_loss": 9.0},
        {"symbol": "NEW", "target_date": "2026-10-02", "signal": "BUY",
         "quantity": 3, "shares": None, "entry_price": 10.0, "exit_price": 12.0,
         "gross_profit_loss": 6.0, "costs": 1.0, "profit_loss": 5.0},
    ])
    summary, detail = _ledger(frame, "V1", "gross_profit_loss", "costs", "profit_loss")
    assert summary["executed_trades"] == 2
    assert summary["gross_rows_reconstructed_from_prices"] == 1
    assert summary["ledger_status"] == "RECONCILED"
    assert detail["quantity_used"].tolist() == [2.0, 3.0]
    assert detail.iloc[0]["gross_pnl_source"] == "DERIVED_FROM_RECORDED_ENTRY_EXIT_AND_QUANTITY"
