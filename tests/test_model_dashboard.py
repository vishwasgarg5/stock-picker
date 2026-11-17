import json

import pandas as pd

from src.model_dashboard import build_dashboard


def test_dashboard_uses_persisted_evidence_and_never_invents_green_status(tmp_path, monkeypatch):
    import src.model_dashboard as dashboard

    monkeypatch.setattr(dashboard, "DATA", tmp_path)
    (tmp_path / "predictions.csv").write_text(
        "prediction_date,symbol,target_date,rank,base_close,predicted_close\n"
        "2026-11-16,AAA,2026-11-17,1,100,102\n"
        "2026-11-16,BBB,2026-11-17,2,200,201\n",
        encoding="utf-8",
    )
    (tmp_path / "phase114_promotion_readiness_summary.json").write_text(
        json.dumps({"status": "INSUFFICIENT_EVIDENCE", "unmet_gates": ["pnl_reconciliation_pass"]}),
        encoding="utf-8",
    )
    (tmp_path / "historical_coverage_audit.json").write_text(
        json.dumps({"status": "REVIEW_REQUIRED", "universe_symbols": 2, "symbols_with_history": 2}),
        encoding="utf-8",
    )
    payload = build_dashboard()

    assert payload["predictions"]["count"] == 2
    assert payload["predictions"]["top10"][0]["symbol"] == "AAA"
    assert payload["production"]["promotion_status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["governance"]["unmet_gates"] == ["pnl_reconciliation_pass"]
    assert payload["data_health"]["calendar"]["status"] == "REVIEW"
    assert payload["data_health"]["index"]["status"] == "REVIEW"
    assert payload["reconciliation"]["status"] == "NOT_AVAILABLE"


def test_dashboard_handles_missing_or_malformed_artifacts(tmp_path, monkeypatch):
    import src.model_dashboard as dashboard

    monkeypatch.setattr(dashboard, "DATA", tmp_path)
    (tmp_path / "phase110_forecast_accuracy_summary.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "paper_trades.csv").write_text("signal,profit_loss\nBUY,10\nSKIP,0\n", encoding="utf-8")
    payload = build_dashboard()

    assert payload["predictions"]["count"] == 0
    assert payload["paper_trading"]["v1"]["buy_rows"] == 1
    assert payload["paper_trading"]["v1"]["skip_rows"] == 1
    assert payload["forecast"]["status"] == "NOT_AVAILABLE"

def test_dashboard_marks_old_calendar_refresh_for_review(tmp_path, monkeypatch):
    import src.model_dashboard as dashboard

    monkeypatch.setattr(dashboard, "DATA", tmp_path)
    (tmp_path / "nse_market_calendar.json").write_text(
        json.dumps({"updated_at_utc": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )
    payload = build_dashboard()
    assert payload["data_health"]["calendar"]["status"] == "REVIEW"
