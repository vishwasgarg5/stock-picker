from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src import system_health as health


def _rows(date, symbols, invalid=False):
    rows = []
    for i, symbol in enumerate(symbols):
        close = 100.0 + i
        high, low = close + 1.0, close - 1.0
        if invalid:
            high, low = close - 2.0, close + 2.0
        rows.append({"date": date, "symbol": symbol, "open": close, "high": high,
                     "low": low, "close": close, "volume": 1000})
    return rows


def test_health_ignores_partial_current_session():
    symbols = [f"S{i:02d}" for i in range(20)]
    today = pd.Timestamp("2026-10-14")
    yesterday = pd.Timestamp("2026-10-13")
    hist = pd.DataFrame(_rows(yesterday, symbols) + _rows(today, symbols[:2]))
    now = datetime(2026, 10, 14, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert health._select_completed_market_date(hist, symbols, now) == yesterday


def test_health_rejects_current_session_before_market_close():
    symbols = [f"S{i:02d}" for i in range(20)]
    today = pd.Timestamp("2026-10-14")
    hist = pd.DataFrame(_rows(today, symbols))
    now = datetime(2026, 10, 14, 14, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert pd.isna(health._select_completed_market_date(hist, symbols, now))


def test_health_falls_back_when_latest_completed_session_has_invalid_ohlc():
    symbols = [f"S{i:02d}" for i in range(20)]
    newest = pd.Timestamp("2026-10-13")
    prior = pd.Timestamp("2026-10-12")
    hist = pd.DataFrame(_rows(prior, symbols) + _rows(newest, symbols, invalid=True))
    now = datetime(2026, 10, 14, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert health._select_completed_market_date(hist, symbols, now) == prior
