import pandas as pd
from src import phase73_76_diagnostics as diag


def test_v2_zero_buy_reasons_are_reported_without_changing_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([{"signal":"SKIP","no_trade_reason":"phase2_not_selected"}, {"signal":"SKIP","no_trade_reason":"phase2_not_selected"}]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([{"stage":"phase2_selected_gate","input_rows":5,"output_rows":0}]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)
    summary, detail = diag.v2_diagnostic()
    assert summary["buy_rows"] == 0
    assert summary["skip_reason_counts"]["phase2_not_selected"] == 2
    assert summary["production_filters_changed"] is False
    assert "FUNNEL_STAGE" in set(detail["primary_reason"])


def test_directional_recent_window_is_chronological(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    pd.DataFrame([{"target_date":f"2026-10-{d:02d}","close_direction_correct":v} for d,v in [(1,1),(2,0),(3,0),(4,1),(5,0),(6,0)]]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)
    summary, dates = diag.direction_diagnostic()
    assert summary["scored_rows"] == 6
    # The latest five dates are Oct 2–6: only Oct 4 is correct (1/5 = 20%).
    assert summary["recent_5_directional_accuracy_pct"] == 20.0
    assert summary["status"] == "REVIEW_DIRECTION_QUALITY"
    assert len(dates) == 6


def test_storage_audit_does_not_modify_source(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    path = tmp_path / "ohlcv.csv"
    path.write_text("date,symbol,close\n2026-10-10,ABC,10\n2026-10-10,ABC,10\n", encoding="utf-8")
    before = path.read_bytes()
    summary = diag.storage_audit()
    assert summary["rows"] == 2
    assert summary["duplicate_symbol_date_keys"] == 1
    assert summary["storage_changed"] is False
    assert path.read_bytes() == before


def test_missing_directional_evidence_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "DATA", tmp_path)
    summary, detail = diag.direction_diagnostic()
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["directional_accuracy_pct"] is None
    assert detail.empty
