import pandas as pd

from src.pipeline import _merge_candidate_history, _select_completed_market_date, validate_data_quality


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



def test_non_positive_open_and_low_prices_are_rejected():
    symbol = "AAA"
    fallback = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
    newer = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    clean = pd.DataFrame(_rows(fallback, [symbol]))
    corrupt = pd.DataFrame(_rows(newer, [symbol]))
    corrupt.loc[0, "open"] = 0.0
    corrupt.loc[0, "low"] = 0.0

    hist = pd.concat([clean, corrupt], ignore_index=True)

    assert _select_completed_market_date(hist, [symbol]) == fallback


def test_non_finite_ohlc_prices_are_rejected():
    symbol = "AAA"
    fallback = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
    newer = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    clean = pd.DataFrame(_rows(fallback, [symbol]))
    corrupt = pd.DataFrame(_rows(newer, [symbol]))
    corrupt.loc[0, "high"] = float("inf")

    hist = pd.concat([clean, corrupt], ignore_index=True)

    assert _select_completed_market_date(hist, [symbol]) == fallback



def test_candidate_history_preserves_repeated_target_date_across_prediction_sessions():
    existing = pd.DataFrame([
        {"prediction_date": "2026-11-12", "target_date": "2026-11-16", "symbol": "MCX", "score": 70.0},
    ])
    current = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 80.0},
    ])

    result = _merge_candidate_history(existing, current)

    assert len(result) == 2
    assert set(result["prediction_date"]) == {"2026-11-12", "2026-11-13"}
    assert result["symbol"].tolist() == ["MCX", "MCX"]


def test_candidate_history_rerun_refreshes_same_prediction_session():
    existing = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 70.0},
    ])
    rerun = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 85.0},
    ])

    result = _merge_candidate_history(existing, rerun)

    assert len(result) == 1
    assert result.iloc[0]["score"] == 85.0
