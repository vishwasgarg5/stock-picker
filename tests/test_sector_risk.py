import pandas as pd
import pytest

from src.ranking_diagnostics import run_sector_risk_validation


def test_sector_risk_uses_point_in_time_snapshot():
    date = pd.Timestamp("2026-01-05")
    candidates = pd.DataFrame({
        "prediction_date": [date] * 5,
        "symbol": ["A", "B", "C", "D", "E"],
        "rank": [1, 2, 3, 4, 5],
    })
    fundamentals = pd.DataFrame({
        "as_of_date": [pd.Timestamp("2026-01-01")] * 5,
        "symbol": ["A", "B", "C", "D", "E"],
        "sector": ["Tech", "Tech", "Tech", "Bank", "Energy"],
    })

    out = run_sector_risk_validation(candidates, fundamentals, max_sector_limit_pct=40.0)
    row = out[out["group"].eq("TOP5")].iloc[0]

    assert row["top_sector"] == "Tech"
    assert row["top_sector_weight_pct"] == pytest.approx(60.0)
    assert row["sector_hhi"] == pytest.approx(4400.0)
    assert bool(row["concentration_breach"]) is True
