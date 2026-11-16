import pandas as pd

from src import repair_historical_gaps as repair


def test_find_missing_sessions_returns_exact_nse_calendar_gaps(monkeypatch):
    sessions = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    monkeypatch.setattr(repair, "_expected_sessions", lambda start, end: sessions)
    history = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA"],
        "date": ["2024-01-01", "2024-01-03", "2024-01-04"],
    })
    result = repair.find_missing_sessions(pd.DataFrame({"symbol": ["AAA", "BBB"]}), history)

    assert result["AAA"] == [pd.Timestamp("2024-01-02")]
    assert result["BBB"] == []


def test_find_missing_sessions_does_not_invent_history_for_empty_symbol():
    result = repair.find_missing_sessions(
        pd.DataFrame({"symbol": ["AAA"]}),
        pd.DataFrame(columns=["symbol", "date"]),
    )
    assert result == {"AAA": []}
