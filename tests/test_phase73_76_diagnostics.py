import pandas as pd

from src import phase73_76_diagnostics as diagnostics


def test_v2_zero_buy_reports_reasons_and_never_changes_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected", "phase4_reject_reason": ""},
        {"signal": "SKIP", "no_trade_reason": "phase2_not_selected", "phase4_reject_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    pd.DataFrame([
        {"stage": "confidence_percentile_gate", "input_rows": 10, "output_rows": 5},
        {"stage": "phase2_selected_gate", "input_rows": 5, "output_rows": 0},
    ]).to_csv(tmp_path / "phase32_v2_selection_funnel.csv", index=False)

    summary, breakdown = diagnostics.v2_zero_buy_diagnostic()

    assert summary["zero_buy"] is True
    assert summary["buy_rows"] == 0
    assert summary["primary_reason_counts"]["phase2_not_selected"] == 2
    assert summary["production_filters_changed"] is False
    assert set(breakdown["primary_reason"]) == {"phase2_not_selected", "FUNNEL_STAGE"}


def test_directional_quality_uses_chronological_recent_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": f"2026-10-{day:02d}", "close_direction_correct": value}
        for day, value in [(1, 1), (2, 0), (3, 0), (4, 1), (5, 0), (6, 0)]
    ]).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)

    summary, by_date = diagnostics.directional_quality_diagnostic()

    assert summary["scored_rows"] == 6
    assert summary["target_dates"] == 6
    assert summary["recent_5_directional_accuracy_pct"] == 40.0
    assert summary["status"] == "REVIEW_DIRECTION_QUALITY"
    assert len(by_date) == 6


def test_storage_audit_warns_but_never_modifies_file(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
    path = tmp_path / "ohlcv.csv"
    path.write_text("date,symbol,close\n2026-10-10,ABC,10\n2026-10-10,ABC,10\n", encoding="utf-8")
    before = path.read_bytes()

    summary = diagnostics.storage_audit()

    assert summary["status"] == "WITHIN_GITHUB_RECOMMENDED_SIZE"
    assert summary["rows"] == 2
    assert summary["duplicate_symbol_date_keys"] == 1
    assert path.read_bytes() == before
    assert summary["storage_changed"] is False


def test_missing_directional_evidence_is_not_treated_as_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "DATA", tmp_path)
    summary, detail = diagnostics.directional_quality_diagnostic()
    assert summary["status"] == "INSUFFICIENT_DATA"
    assert summary["directional_accuracy_pct"] is None
    assert detail.empty
