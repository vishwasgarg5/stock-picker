import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

from src import phase81_84_diagnostics as diag


def test_receipt_reconciliation_flags_legacy_duplicates_and_missing_id(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "x", "telegram_message_id": "1"},
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "x", "telegram_message_id": ""},
        {"delivery_status": "LEGACY_UNVERIFIED", "message_hash": "y", "telegram_message_id": ""},
        {"delivery_status": "", "message_hash": "z", "telegram_message_id": ""},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, rows = diag.receipt_reconciliation()
    assert report["rows"] == 4
    assert report["confirmed"] == 2
    assert report["legacy_unverified"] == 1
    assert report["unknown_status"] == 1
    assert report["duplicate_hash_rows"] == 2
    assert report["confirmed_missing_message_id"] == 1
    assert report["status"] == "RECEIPTS_NEED_RECONCILIATION"
    assert len(rows) == 4


def test_v2_diagnosis_is_descriptive_and_preserves_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected"},
        {"signal": "SKIP", "no_trade_reason": "confidence_percentile"},
        {"signal": "BUY", "no_trade_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([
        {"stage": "phase2_selected_gate", "input_rows": 100, "output_rows": 20},
        {"stage": "confidence", "input_rows": 20, "output_rows": 5},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    report, funnel = diag.v2_rejection_diagnosis()
    assert report["buy_rows"] == 1
    assert report["skip_rows"] == 2
    assert report["production_filters_changed"] is False
    assert report["thresholds_deployed"] is False
    assert funnel.iloc[0]["drop_rows"] == 80


def test_forecast_evidence_bootstraps_by_date_and_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": f"2026-10-{day:02d}", "close_ape": 0.01, "baseline_close_ape": 0.02}
        for day in range(1, 9)
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    first, by_date = diag.forecast_evidence()
    second, _ = diag.forecast_evidence()
    assert first["scored_rows"] == 8
    assert first["date_block_bootstrap_95pct_ci_delta_pct_points"] == second["date_block_bootstrap_95pct_ci_delta_pct_points"]
    assert first["beats_baseline"] is True
    assert len(by_date) == 8


def test_storage_prototype_roundtrips_and_leaves_source_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol,open,high,low,close\n2026-10-01,ABC,10,12,9,11\n", encoding="utf-8")
    before = source.read_bytes()
    report = diag.storage_prototype()
    assert report["status"] == "ROUNDTRIP_PARITY_PASS"
    assert report["source_unchanged"] is True
    assert report["prototype_written_to_repo"] is False
    assert source.read_bytes() == before


def test_run_writes_report_with_nonpromotion_safety_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    report = diag.run()
    saved = json.loads((tmp_path / "phase81_84_diagnostics.json").read_text())
    assert saved["phases"] == [81, 82, 83, 84]
    assert report["safety"]["production_champion"] == "V1"
    assert report["safety"]["v2_mode"] == "SHADOW_ONLY"
    assert report["safety"]["automatic_promotion"] is False
    assert report["safety"]["source_ohlcv_modified"] is False
