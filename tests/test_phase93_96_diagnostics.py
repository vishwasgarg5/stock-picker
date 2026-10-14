import pandas as pd
from src import phase93_96_diagnostics as diag


def test_receipt_duplicate_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"message_hash": "abc", "delivery_status": "TELEGRAM_API_CONFIRMED", "telegram_message_id": 10, "telegram_date_utc": 1791905727},
        {"message_hash": "abc", "delivery_status": "TELEGRAM_API_CONFIRMED", "telegram_message_id": 12, "telegram_date_utc": 1791911312},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)
    report, groups = diag.phase93_receipt_forensics()
    assert report["repeated_hash_groups"] == 1
    assert report["groups_with_multiple_message_ids"] == 1


def test_forecast_error_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": "2026-10-01", "symbol": "AAA", "close_ape": .03, "baseline_close_ape": .01, "period": "RISK_OFF"},
        {"target_date": "2026-10-02", "symbol": "BBB", "close_ape": .01, "baseline_close_ape": .02, "period": "RISK_ON"},
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    report, segments = diag.phase94_forecast_error_segments()
    assert report["symbols_worse_than_baseline"] == 1
    assert report["worst_symbols"][0]["symbol"] == "AAA"


def test_v2_earliest_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([
        {"stage": "load", "input_rows": 10, "output_rows": 10, "rejected_rows": 0},
        {"stage": "actual_data_join", "input_rows": 10, "output_rows": 7, "rejected_rows": 3},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    report, funnel = diag.phase95_v2_earliest_rejection()
    assert report["earliest_blocking_stage"] == "actual_data_join"


def test_storage_preflight_is_non_destructive(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol,close\n2026-10-01,AAA,10\n", encoding="utf-8")
    before = source.read_bytes()
    report = diag.phase96_reversible_storage_preflight()
    assert report["first_100_rows_parity"]
    assert report["source_unchanged"]
    assert report["migration_executed"] is False
