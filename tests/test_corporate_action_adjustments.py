import pandas as pd
import pytest

from src.pipeline import (
    _add_split_adjusted_columns,
    _normalise_history,
    add_targets,
    features,
    market_regime,
    technical_score,
)


def split_history():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "symbol": ["ABC"] * 3,
        "open": [199.0, 99.0, 100.0],
        "high": [202.0, 101.0, 102.0],
        "low": [198.0, 98.0, 99.0],
        "close": [200.0, 100.0, 101.0],
        "volume": [1000.0, 2000.0, 2100.0],
        "dividends": [0.0, 0.0, 0.0],
        "stock_splits": [0.0, 2.0, 0.0],
    })


def test_split_adjustment_rebases_pre_split_ohlcv_but_keeps_raw_prices():
    adjusted = _add_split_adjusted_columns(split_history())

    assert adjusted.loc[0, "close"] == 200.0
    assert adjusted.loc[0, "split_adjusted_close"] == 100.0
    assert adjusted.loc[1, "split_adjusted_close"] == 100.0
    assert adjusted.loc[0, "split_adjusted_volume"] == 2000.0
    assert adjusted.loc[1, "split_adjusted_volume"] == 2000.0


def test_target_returns_do_not_treat_stock_split_as_market_loss():
    targets = add_targets(split_history())

    assert targets.loc[0, "target_close_return"] == pytest.approx(0.0)
    assert targets.loc[1, "target_close_return"] == pytest.approx(0.01)


def test_features_use_split_adjusted_close_for_returns():
    result = features(split_history())

    assert result.loc[1, "return_1d"] == pytest.approx(0.0)
    assert result.loc[2, "return_1d"] == pytest.approx(0.01)
    assert result.loc[1, "close"] == 100.0


def test_dividend_metadata_does_not_change_split_only_adjustment():
    history = split_history()
    history.loc[1, "dividends"] = 5.0

    adjusted = _add_split_adjusted_columns(history)

    assert adjusted.loc[0, "split_adjusted_close"] == 100.0
    assert adjusted.loc[1, "split_adjusted_close"] == 100.0


def test_invalid_split_ratio_fails_closed():
    history = split_history()
    history.loc[1, "stock_splits"] = -2.0

    with pytest.raises(RuntimeError, match="invalid stock split ratio"):
        _add_split_adjusted_columns(history)


def test_yahoo_action_fields_are_preserved_by_history_normalizer():
    raw = pd.DataFrame({
        "Open": [100.0],
        "High": [102.0],
        "Low": [99.0],
        "Close": [101.0],
        "Volume": [2000],
        "Dividends": [1.5],
        "Stock Splits": [2.0],
    }, index=pd.DatetimeIndex(["2026-01-02"], name="Date"))

    normalized = _normalise_history(raw, "ABC")

    assert normalized.loc[0, "dividends"] == 1.5
    assert normalized.loc[0, "stock_splits"] == 2.0


def test_market_regime_uses_split_adjusted_close():
    dates = pd.bdate_range("2026-01-01", periods=60)
    adjusted_prices = pd.Series(range(100, 160), dtype=float)
    raw_prices = adjusted_prices.copy()
    raw_prices.iloc[40:] = raw_prices.iloc[40:] / 2.0
    history = pd.DataFrame({
        "date": dates,
        "symbol": "ABC",
        "open": raw_prices,
        "high": raw_prices * 1.01,
        "low": raw_prices * 0.99,
        "close": raw_prices,
        "volume": 1000.0,
        "stock_splits": [0.0] * 39 + [2.0] + [0.0] * 20,
    })

    assert market_regime(history) == "BULL"


def test_technical_score_compares_adjusted_close_to_adjusted_ema():
    latest = pd.DataFrame({
        "return_20d": [0.1, 0.0],
        "return_5d": [0.05, 0.0],
        "volume_ratio": [1.5, 1.0],
        "rsi14": [60.0, 40.0],
        "close": [50.0, 50.0],
        "split_adjusted_close": [100.0, 50.0],
        "ema20": [90.0, 90.0],
        "ema50": [80.0, 80.0],
    })

    scores = technical_score(latest)

    assert scores.iloc[0] > scores.iloc[1]
