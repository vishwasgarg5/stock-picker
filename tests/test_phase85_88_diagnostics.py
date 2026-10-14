import json
from pathlib import Path

import pandas as pd

from src import phase85_88_diagnostics as diag


def test_receipt_anomalies_are_flagged_not_assumed_to_be_duplicate_sends(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "same", "telegram_message_id": "1"},
        {"delivery_status": "TELEGRAM_API_CONFIRMED", "message_hash": "same", "telegram_message_id": "2"},
        {"delivery_status": "LEGACY_UNVERIFIED", "message_hash": "", "telegram_message_id": ""},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, detail = diag.receipt_anomaly_classification()
    assert report["repeated_hash_rows"] == 2
    assert report["missing_hash_rows"] == 1
    assert report["messages_resubmitted"] is False
    assert set(detail["classification"]) == {"REPEATED_HASH_REVIEW", "HASH_MISSING"}


def test_receipt_schema_audit_reports_missing_columns_without_changing_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([{"delivery_status": "LEGACY_UNVERIFIED"}]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, detail = diag.receipt_schema_readiness()
    assert report["send_path_changed"] is False
    assert report["receipt_writer_changed"] is False
    assert len(detail) == 4
    assert not bool(detail.loc[detail["file"] == "telegram_sent.csv", "schema_ready_for_future_receipts"].iloc[0])


def test_rolling_forecast_windows_are_chronological_and_descriptive(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": f"2026-09-{day:02d}", "close_ape": 0.01, "baseline_close_ape": 0.02}
        for day in range(1, 9)
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    report, windows = diag.rolling_forecast_evidence()
    assert report["target_dates"] == 8
    assert len(windows) == 4
    assert windows.iloc[0]["window_start"] == "2026-09-01"
    assert windows.iloc[-1]["window_end"] == "2026-09-08"
    assert report["model_or_threshold_changed"] is False


def test_storage_benchmark_preserves_source_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    source = tmp_path / "ohlcv.csv"
    source.write_text(
        "date,symbol,open,high,low,close\n2026-09-01,ABC,10,12,9,11\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    report = diag.storage_performance_benchmark()
    assert report["status"] == "BENCHMARK_PARITY_PASS"
    assert report["source_unchanged"] is True
    assert report["production_reader_switched"] is False
    assert source.read_bytes() == before


def test_report_has_all_safety_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    report = diag.run()
    saved = json.loads((tmp_path / "phase85_88_diagnostics.json").read_text(encoding="utf-8"))
    assert saved["phases"] == [85, 86, 87, 88]
    assert report["safety"]["production_champion"] == "V1"
    assert report["safety"]["v2_mode"] == "SHADOW_ONLY"
    assert report["safety"]["automatic_promotion"] is False
    assert report["safety"]["source_ohlcv_modified"] is False
