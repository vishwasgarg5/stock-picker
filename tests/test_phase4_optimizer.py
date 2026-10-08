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


def test_index_intelligence_overlay_is_conservative():
    from src.index_intelligence import apply_index_overlay
    x = pd.DataFrame({"symbol": ["ABC"], "phase4_score": [10.0], "score": [5.0]})
    y = apply_index_overlay(x)
    assert y.iloc[0]["index_market_regime"] in {"BULL", "BEAR", "NEUTRAL"}
    assert 0 < float(y.iloc[0]["index_risk_multiplier"]) <= 1.0
    assert float(y.iloc[0]["phase4_score"]) <= 10.0


def test_index_news_reason_scoring_is_deterministic():
    from src.index_intelligence import score_news
    score, headline, sentiment, url, reasons = score_news([
        {"headline": "Nifty falls as crude oil rises and RBI tightens policy", "url": "x"}
    ])
    assert score < 0
    assert sentiment == "BEARISH"
    assert headline
    assert url == "x"
    assert reasons and reasons[0]["event"] in {"RBI / rates", "Crude oil", "General market", "Inflation", "Geopolitics"}


def test_market_intelligence_score_is_bounded():
    from src.index_intelligence import _market_score
    out = pd.DataFrame({"index": ["NIFTY50"], "direction": ["BUY"], "confidence": [0.8]})
    score, level = _market_score(out, 2.0, 80.0)
    assert 0.0 <= score <= 100.0
    assert level in {"RISK_ON", "NEUTRAL", "RISK_OFF"}


def test_market_score_excludes_missing_index_data():
    from src.index_intelligence import _market_score
    out = pd.DataFrame(columns=["index", "direction", "confidence"])
    score, level, confidence, quality, available = _market_score(out, -6.8, 28.33)
    assert not available
    assert score < 35
    assert level == "RISK_OFF"
    assert quality == "LOW"
    assert confidence > 0
