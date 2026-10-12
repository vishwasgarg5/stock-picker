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
