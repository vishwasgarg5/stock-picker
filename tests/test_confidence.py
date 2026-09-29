import pandas as pd
import pytest

from src.confidence import run_confidence_analysis


def test_confidence_bucket_calibration_is_monotonic(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=12)
    rows = []
    for i, d in enumerate(dates):
        for j in range(10):
            score = 10 + j * 9
            close = 100.0 + (1.0 if j >= 5 else -1.0)
            rows.append({
                "prediction_date": d,
                "target_date": d,
                "symbol": f"S{i}_{j}",
                "rank": j + 1,
                "score": float(j),
                "confidence_score": float(score),
                "predicted_close": 101.0,
                "base_close": 100.0,
            })
    candidates = pd.DataFrame(rows)
    history = []
    for _, r in candidates.iterrows():
        history.append({
            "date": r["target_date"],
            "symbol": r["symbol"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + (1.0 if r["confidence_score"] >= 55 else -1.0),
        })
    history = pd.DataFrame(history)

    monkeypatch.setattr("src.confidence.CANDIDATES", tmp_path / "prediction_candidates_history.csv")
    monkeypatch.setattr("src.confidence.HISTORY", tmp_path / "ohlcv.csv")
    monkeypatch.setattr("src.confidence.OUTPUT", tmp_path / "confidence_analysis.csv")
    monkeypatch.setattr("src.confidence.SELECTION_OUTPUT", tmp_path / "selection_validation.csv")
    candidates.to_csv(tmp_path / "prediction_candidates_history.csv", index=False)
    history.to_csv(tmp_path / "ohlcv.csv", index=False)

    out = run_confidence_analysis()

    assert not out.empty
    assert out["confidence_direction_spearman"].iloc[0] > 0
    assert out["confidence_profit_spearman"].iloc[0] > 0
