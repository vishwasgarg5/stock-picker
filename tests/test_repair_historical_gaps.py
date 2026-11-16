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


def test_nse_bhavcopy_url_uses_legacy_archive_before_udiff_cutover():
    assert repair.nse_bhavcopy_url(pd.Timestamp("2024-07-05")) == (
        "https://archives.nseindia.com/content/historical/EQUITIES/2024/JUL/cm05JUL2024bhav.csv.zip"
    )


def test_nse_bhavcopy_url_uses_udiff_archive_on_cutover():
    assert repair.nse_bhavcopy_url(pd.Timestamp("2024-07-08")) == (
        "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip"
    )


def test_normalise_legacy_nse_bhavcopy_maps_equity_ohlcv_and_filters_other_series():
    raw = pd.DataFrame({
        "SYMBOL": ["AAA", "AAA", "BBB"],
        "SERIES": ["EQ", "BE", "SM"],
        "OPEN": [10, 10, 20],
        "HIGH": [12, 12, 22],
        "LOW": [9, 9, 19],
        "CLOSE": [11, 11, 21],
        "TOTTRDQTY": [100, 90, 200],
    })
    result = repair.normalise_nse_bhavcopy(raw, pd.Timestamp("2024-07-05"))

    assert result[["symbol", "open", "high", "low", "close", "volume"]].to_dict("records") == [
        {"symbol": "AAA", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100}
    ]
    assert result.iloc[0]["date"] == pd.Timestamp("2024-07-05")


def test_normalise_udiff_nse_bhavcopy_maps_modern_fields():
    raw = pd.DataFrame({
        "TckrSymb": ["AAA"],
        "SctySrs": ["EQ"],
        "OpnPric": [10],
        "HghPric": [12],
        "LwPric": [9],
        "ClsPric": [11],
        "TtlTradgVol": [100],
    })
    result = repair.normalise_nse_bhavcopy(raw, pd.Timestamp("2024-07-08"))

    assert result.iloc[0]["symbol"] == "AAA"
    assert result.iloc[0]["volume"] == 100
