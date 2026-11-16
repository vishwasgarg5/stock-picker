import pandas as pd

from src.phase31_walk_forward import chronological_comparison

from src.phase21_26 import (
    direction_calibration, leakage_audit, matched_evidence, drift_report,
    news_impact_report,
)


def test_direction_calibration_reports_without_changing_production():
    e = pd.DataFrame([
        {"prediction_date":"2026-09-01","symbol":"ABC","close_direction_correct":1,
         "close_abs_pct_error":0.01,"baseline_close_abs_pct_error":0.02},
        {"prediction_date":"2026-09-02","symbol":"XYZ","close_direction_correct":0,
         "close_abs_pct_error":0.03,"baseline_close_abs_pct_error":0.02},
    ])
    c = pd.DataFrame([
        {"prediction_date":"2026-09-01","symbol":"ABC","confidence_score":90},
        {"prediction_date":"2026-09-02","symbol":"XYZ","confidence_score":30},
    ])
    out = direction_calibration(e,c)
    assert set(out["confidence_bucket"]) >= {"LOW_0_40","VERY_HIGH_80_PLUS"}
    assert set(out["calibration_status"]) == {"INSUFFICIENT_SAMPLE"}


def test_leakage_audit_flags_bad_date_order_and_duplicates():
    e = pd.DataFrame([
        {"prediction_date":"2026-09-02","target_date":"2026-09-01","symbol":"ABC"},
        {"prediction_date":"2026-09-02","target_date":"2026-09-03","symbol":"ABC"},
    ])
    c = pd.DataFrame([{"prediction_date":"2026-09-02","symbol":"ABC"}])
    h = pd.DataFrame([{"date":"2026-09-01","symbol":"ABC"}])
    out = leakage_audit(e,c,h).set_index("check")
    assert out.loc["prediction_before_target","affected_rows"] == 1
    assert out.loc["unique_prediction_key","affected_rows"] == 1


def test_matched_evidence_uses_only_common_sessions_and_keeps_v1_champion():
    v1 = pd.DataFrame([{"target_date":"2026-09-01","daily_profit_loss":10},
                       {"target_date":"2026-09-02","daily_profit_loss":-5}])
    v2 = pd.DataFrame([{"target_date":"2026-09-02","daily_profit_loss":4},
                       {"target_date":"2026-09-03","daily_profit_loss":7}])
    out = matched_evidence(v1,v2,pd.DataFrame(),pd.DataFrame()).iloc[0]
    assert out["common_sessions"] == 1
    assert out["v1_net_pnl"] == -5
    assert out["v2_net_pnl"] == 4
    assert out["production_champion"] == "V1"


def test_drift_report_has_safe_fallback():
    out = drift_report(pd.DataFrame())
    assert out.iloc[0]["drift_status"] == "INSUFFICIENT_DATA"


def test_news_impact_is_analysis_only():
    e = pd.DataFrame([{"prediction_date":"2026-09-01","symbol":"ABC",
                       "close_direction_correct":1,"close_abs_pct_error":0.01}])
    c = pd.DataFrame([{"prediction_date":"2026-09-01","symbol":"ABC","stock_news_score":0.5}])
    out = news_impact_report(e,c)
    assert out.iloc[0]["news_signal"] == "stock_news_score"
    assert out.iloc[0]["status"] == "INSUFFICIENT_SAMPLE"



def test_phase31_chronological_replay_sorts_dates_and_keeps_v1_champion():
    v1 = pd.DataFrame([
        {"target_date": "2026-10-03", "daily_profit_loss": 30},
        {"target_date": "2026-10-01", "daily_profit_loss": 10},
        {"target_date": "2026-10-02", "daily_profit_loss": -5},
    ])
    v2 = pd.DataFrame([
        {"target_date": "2026-10-02", "daily_profit_loss": 0},
        {"target_date": "2026-10-03", "daily_profit_loss": 0},
        {"target_date": "2026-10-01", "daily_profit_loss": 0},
    ])
    v2_trades = pd.DataFrame([
        {"signal": "SKIP", "quantity": 0, "profit_loss": 0},
    ])
    daily, summary = chronological_comparison(v1, v2, pd.DataFrame(), v2_trades)

    assert daily["target_date"].is_monotonic_increasing
    assert daily["period"].tolist() == ["EARLY_CONTEXT", "EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT"]
    assert summary["matched_sessions"] == 3
    assert summary["v2_executed_trades"] == 0
    assert summary["evidence_gate"] == "COLLECTING_MATCHED_SESSIONS"
    assert summary["production_champion"] == "V1"
    assert summary["v2_promoted"] is False


def test_phase31_evidence_gate_requires_sessions_and_executed_v2_trades():
    dates = pd.date_range("2026-09-01", periods=25, freq="B")
    v1 = pd.DataFrame({"target_date": dates, "daily_profit_loss": [10.0] * len(dates)})
    v2 = pd.DataFrame({"target_date": dates, "daily_profit_loss": [12.0] * len(dates)})
    v2_trades = pd.DataFrame([
        {"signal": "BUY", "quantity": 1, "profit_loss": 1.0}
        for _ in range(50)
    ])
    _, summary = chronological_comparison(v1, v2, pd.DataFrame(), v2_trades)

    assert summary["matched_sessions"] == 25
    assert summary["v2_executed_trades"] == 50
    assert summary["evidence_gate"] == "EVIDENCE_SUFFICIENT_FOR_REVIEW"
    assert summary["v2_promoted"] is False


def test_phase31_empty_history_cannot_pass_evidence_gate():
    daily, summary = chronological_comparison(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert daily.empty
    assert summary["matched_sessions"] == 0
    assert summary["evidence_gate"] == "INSUFFICIENT_EVIDENCE"
    assert summary["v2_promoted"] is False
