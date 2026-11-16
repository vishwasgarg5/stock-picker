import json

import pandas as pd

from src import phase77_80_diagnostics as diag


def test_delivery_audit_separates_confirmed_legacy_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "a", "telegram_message_id": "1"},
        {"delivery_status": "LEGACY_UNVERIFIED", "message_hash": "b", "telegram_message_id": ""},
        {"delivery_status": "", "message_hash": "c", "telegram_message_id": ""},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, rows = diag.delivery_audit()
    assert report["rows"] == 3
    assert report["confirmed"] == 1
    assert report["legacy_unverified"] == 1
    assert report["missing_status"] == 1
    assert report["status"] == "RECEIPT_AUDIT_INCOMPLETE"
    assert len(rows) == 3


def test_delivery_audit_detects_duplicate_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "same"},
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "same"},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, rows = diag.delivery_audit()
    assert report["duplicates"] == 1
    assert rows["duplicate_message_key"].all()


def test_v2_funnel_reports_largest_drop_without_changing_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected"},
        {"signal": "SKIP", "no_trade_reason": "confidence_percentile"},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([
        {"stage": "phase2", "input_rows": 100, "output_rows": 20},
        {"stage": "confidence", "input_rows": 20, "output_rows": 5},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    report, funnel = diag.v2_funnel_audit()
    assert report["buy_rows"] == 0
    assert report["largest_recorded_drop_stage"] == "phase2"
    assert report["production_filters_changed"] is False
    assert report["thresholds_deployed"] is False
    assert len(funnel) == 2


def test_forecast_evidence_uses_paired_rows_and_recent_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": "2026-10-01", "close_ape": 0.01, "baseline_close_ape": 0.02},
        {"target_date": "2026-10-02", "close_ape": 0.03, "baseline_close_ape": 0.02},
        {"target_date": "2026-10-03", "close_ape": 0.01, "baseline_close_ape": 0.02},
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    report, dates = diag.forecast_evidence()
    assert report["scored_rows"] == 3
    assert report["beats_baseline"] is True
    assert report["same_scored_rows"] is True
    assert report["recent_sessions"] == 3
    assert len(dates) == 3


def test_storage_plan_is_non_destructive(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    path = tmp_path / "ohlcv.csv"
    path.write_text("date,symbol,close\n2026-10-01,ABC,10\n", encoding="utf-8")
    before = path.read_bytes()
    report = diag.storage_migration_plan()
    assert report["migration_executed"] is False
    assert report["source_modified"] is False
    assert path.read_bytes() == before


def test_run_writes_report_with_safety_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    report = diag.run()
    saved = json.loads((tmp_path / "phase77_80_diagnostics.json").read_text())
    assert saved["phases"] == [77, 78, 79, 80]
    assert report["safety"]["production_champion"] == "V1"
    assert report["safety"]["v2_mode"] == "SHADOW_ONLY"
    assert report["safety"]["automatic_promotion"] is False
