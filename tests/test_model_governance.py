import pandas as pd

from src.model_governance import _target_diagnostics


def test_target_threshold_diagnostics_are_monotonic_in_coverage():
    rows = []
    for i in range(30):
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "symbol": "A",
            "close": 100.0 * (1.01 ** i),
        })
    out = _target_diagnostics(pd.DataFrame(rows))
    assert not out.empty
    assert out["coverage_pct"].is_monotonic_decreasing
    assert 0.15 in out["threshold_pct"].round(2).tolist()
