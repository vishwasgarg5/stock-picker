import pandas as pd

from src.phase4_optimizer import _regime, _regime_multipliers, _symbol_stats


def test_phase4_regime_parameters_are_defined():
    for regime in ["BULL", "BEAR", "NEUTRAL"]:
        p = _regime_multipliers(regime)
        assert 0 < p["confidence"] <= 1
        assert p["expected_return"] > 0
        assert p["stop"] > 0
        assert p["target"] > 0


def test_phase4_symbol_stats_is_shrinkage_safe():
    e = pd.DataFrame({
        "symbol": ["ABC"] * 4 + ["XYZ"] * 12,
        "target_date": list(pd.date_range("2026-09-01", periods=4))
            + list(pd.date_range("2026-09-01", periods=12)),
        "close_direction_correct": [1, 0, 1, 0] + [1] * 12,
        "actual_open": [100.0] * 16,
        "actual_close": [101.0, 99.0, 101.0, 99.0] + [101.0] * 12,
    })
    s = _symbol_stats(e)
    assert set(s["symbol"]) == {"ABC", "XYZ"}
    assert s["observations"].min() >= 4
    assert s["loss_streak"].max() >= 0


def test_phase4_regime_returns_neutral_without_history():
    h = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "close": range(100, 110)})
    assert _regime(h) == "NEUTRAL"


def test_phase4_missing_metadata_defaults_neutral():
    import pandas as pd
    from src.phase4_optimizer import apply_phase4
    x = pd.DataFrame({
        "symbol": ["ABC"], "phase2_score": [1.0],
        "trade_quality_probability": [0.8],
        "confidence_pct": [0.9], "expected_return_pct": [1.0],
    })
    y = apply_phase4(x)
    assert bool(y.iloc[0]["phase4_eligible"])
    assert float(y.iloc[0]["phase4_weight"]) == 1.0


def test_phase4_missing_scoring_columns_are_safe():
    from src.phase4_optimizer import apply_phase4
    x = pd.DataFrame({"symbol": ["ABC"]})
    y = apply_phase4(x)
    assert len(y) == 1
    assert float(y.iloc[0]["symbol_weight"]) == 1.0
    assert not bool(y.iloc[0]["phase4_eligible"])
