import pandas as pd

from src.pipeline import _select_completed_market_date, validate_data_quality


def _rows(date, symbols, invalid=False):
    rows = []
    for i, symbol in enumerate(symbols):
        close = 100.0 + i
        high = close + 1.0
        low = close - 1.0
        if invalid:
            high, low = close - 2.0, close + 2.0
        rows.append({
            "date": date,
            "symbol": symbol,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        })
    return rows


def test_partial_current_session_falls_back_to_previous_clean_session():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    today = pd.Timestamp.now().normalize()
    hist = pd.DataFrame(_rows(today, symbols, invalid=True) + _rows(yesterday, symbols))
    assert _select_completed_market_date(hist, symbols) == yesterday


def test_completed_session_quality_gate_uses_clean_session():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    session = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    hist = pd.DataFrame(_rows(session, symbols))
    result = validate_data_quality(hist, symbols, "test")
    assert result["latest_date"] == session
    assert result["coverage"] == 1.0
