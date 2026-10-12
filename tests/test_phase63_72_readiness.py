import json

import pandas as pd

from src import phase63_72_readiness as audit


def test_prediction_audit_blocks_invalid_top10_and_keeps_candidate_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DATA", tmp_path)
    rows = []
    for rank in range(1, 11):
        rows.append({
            "symbol": f"S{rank}", "target_date": "2026-10-13", "rank": rank,
            "predicted_open": 100, "predicted_high": 105, "predicted_low": 95,
            "predicted_close": 102,
        })
    rows[0]["predicted_high"] = 90
    pd.DataFrame(rows).to_csv(tmp_path / "predictions.csv", index=False)

    result = audit.prediction_report_audit()

    assert result["status"] == "BLOCK"
    assert result["invalid_ohlc_rows"] == 1
    assert result["ranked_candidates_are_not_buy_approval"] is True


def test_v2_readiness_never_automatically_promotes(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DATA", tmp_path)
    pd.DataFrame([
        {"signal": "SKIP", "target_date": "2026-10-12", "no_trade_reason": "risk_off"},
        {"signal": "BUY", "target_date": "2026-10-13", "no_trade_reason": ""},
    ]).to_csv(tmp_path / "paper_trades_v2.csv", index=False)
    (tmp_path / "phase52_promotion_gate.json").write_text(
        json.dumps({"promotion_safe": True}), encoding="utf-8"
    )

    result = audit.v2_evidence_audit()

    assert result["buy_rows"] == 1
    assert result["promotion_ready"] is False
    assert result["production_champion"] == "V1"
    assert result["v2_mode"] == "SHADOW_ONLY"
    assert result["automatic_promotion"] is False


def test_forecast_audit_reports_recent_sessions_and_per_date_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DATA", tmp_path)
    rows = []
    for day, model_ape, baseline_ape in [
        ("2026-10-01", .010, .011),
        ("2026-10-02", .012, .011),
        ("2026-10-03", .009, .010),
        ("2026-10-04", .011, .010),
        ("2026-10-05", .008, .009),
        ("2026-10-06", .013, .010),
    ]:
        rows.append({
            "target_date": day, "close_ape": model_ape,
            "baseline_close_ape": baseline_ape, "close_direction_correct": 1,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "phase33_chronological_prediction_scores.csv", index=False)

    result = audit.forecast_baseline_audit()

    assert result["recent_5_sessions"] == 5
    assert result["recent_5_sessions_scored_rows"] == 5
    assert result["recent_5_sessions_error_delta_pct_points"] > 0
    assert result["recent_5_sessions_beats_baseline"] is False
    assert len(result["per_date_comparison"]) == 6
    assert result["production_model_changed"] is False


def test_telegram_receipt_audit_ignores_blank_legacy_keys_and_counts_real_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DATA", tmp_path)
    pd.DataFrame([
        {"target_date": "", "message_hash": "", "delivery_status": "LEGACY_UNVERIFIED"},
        {"target_date": "2026-10-13", "message_hash": "abc", "delivery_status": "TELEGRAM_API_CONFIRMED"},
        {"target_date": "2026-10-13", "message_hash": "abc", "delivery_status": "TELEGRAM_API_CONFIRMED"},
        {"target_date": "2026-10-13", "message_hash": "def", "delivery_status": "LEGACY_UNVERIFIED"},
    ]).to_csv(tmp_path / "telegram_sent.csv", index=False)

    result = audit.telegram_receipt_audit()

    assert result["rows"] == 4
    assert result["rows_with_usable_message_keys"] == 3
    assert result["api_confirmed_rows"] == 2
    assert result["duplicate_message_keys"] == 1
    assert result["by_file"]["telegram_sent.csv"]["rows_missing_message_key"] == 1
