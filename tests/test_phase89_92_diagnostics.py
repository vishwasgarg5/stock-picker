import json

import pandas as pd

from src import phase89_92_diagnostics as diag


def test_receipt_reconciliation_flags_repeated_hash_with_distinct_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"message_hash": "same", "delivery_status": "TELEGRAM_API_CONFIRMED", "telegram_message_id": "10"},
        {"message_hash": "same", "delivery_status": "TELEGRAM_API_CONFIRMED", "telegram_message_id": "11"},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, detail = diag.phase89_receipt_reconciliation()
    assert report["multiple_confirmed_message_id_groups"] == 1
    assert set(detail["classification"]) == {"MULTIPLE_CONFIRMED_MESSAGE_IDS"}
    assert report["send_logic_changed"] is False


def test_segmented_forecast_compares_each_symbol_to_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": "2026-10-01", "symbol": "AAA", "close_ape": .01, "baseline_close_ape": .02},
        {"target_date": "2026-10-01", "symbol": "BBB", "close_ape": .03, "baseline_close_ape": .02},
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    report, detail = diag.phase90_segmented_forecast()
    assert report["symbols_scored"] == 2
    assert report["symbols_model_beats_baseline"] == 1
    assert set(detail["symbol"]) == {"AAA", "BBB"}


def test_v2_rejection_analysis_is_diagnostic_only(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase4_ineligible"},
        {"signal": "SKIP", "no_trade_reason": "phase4_ineligible"},
        {"signal": "BUY", "no_trade_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([
        {"stage": "gate", "input_rows": 3, "output_rows": 1, "rejected_rows": 2, "detail": "test"}
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    report, reasons, funnel = diag.phase91_v2_rejection_analysis()
    assert report["buy_rows"] == 1
    assert report["skip_rows"] == 2
    assert reasons.iloc[0]["diagnostic_reason"] == "phase4_ineligible"
    assert report["risk_thresholds_changed"] is False


def test_storage_readiness_does_not_modify_source(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol\n2026-10-01,AAA\n", encoding="utf-8")
    before = source.read_bytes()
    report = diag.phase92_storage_migration_readiness()
    assert report["migration_executed"] is False
    assert report["source_modified"] is False
    assert source.read_bytes() == before


def test_run_writes_reports_and_preserves_safety_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase4_ineligible"}
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    report = diag.run()
    saved = json.loads((tmp_path / "phase89_92_diagnostics.json").read_text(encoding="utf-8"))
    assert saved["phases"] == [89, 90, 91, 92]
    assert report["safety"]["production_champion"] == "V1"
    assert report["safety"]["v2_mode"] == "SHADOW_ONLY"
    assert report["safety"]["automatic_promotion"] is False
