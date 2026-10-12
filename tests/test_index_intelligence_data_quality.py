import numpy as np
import pandas as pd

from src import index_intelligence as ii


def test_short_history_uses_full_downloaded_index_series(tmp_path, monkeypatch):
    monkeypatch.setattr(ii, "HISTORY", tmp_path / "ohlcv.csv")
    monkeypatch.setattr(ii, "SUMMARY", tmp_path / "index_intelligence_summary.csv")
    monkeypatch.setattr(ii, "CONFIG", tmp_path / "index_intelligence_config.json")
    monkeypatch.setattr(ii, "fetch_market_news", lambda: [])
    monkeypatch.setattr(ii, "score_news", lambda news: (0.0, "", "NEUTRAL", "", []))
    monkeypatch.setattr(ii, "_breadth", lambda hist: (50.0, 0.5))

    dates = pd.date_range("2026-07-01", periods=60, freq="D")
    downloaded = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    calls = []

    def fake_download(name):
        calls.append(name)
        return downloaded

    monkeypatch.setattr(ii, "_download_index_series", fake_download)
    hist = pd.DataFrame([{
        "date": "2026-10-12", "symbol": "NIFTYAUTO", "close": 120.0
    }])

    out = ii.build_index_intelligence(hist)
    auto = out.loc[out["index"].eq("NIFTYAUTO")].iloc[0]

    assert "NIFTYAUTO" in calls
    assert np.isfinite(float(auto["change_1d_pct"]))
    assert np.isfinite(float(auto["return_5d_pct"]))
    assert np.isfinite(float(auto["return_20d_pct"]))


def test_market_score_does_not_call_missing_returns_complete_coverage():
    rows = []
    for name in ii.INDEXES:
        rows.append({
            "index": name,
            "direction": "HOLD",
            "confidence": 0.5,
            "change_1d_pct": -0.2,
            "return_5d_pct": -0.5,
            "return_20d_pct": -1.0,
        })
    rows[-1]["return_20d_pct"] = np.nan

    score, risk, confidence, quality, all_available = ii._market_score(
        pd.DataFrame(rows), news_score=0.0, breadth_pct=25.0
    )

    assert np.isfinite(score)
    assert not all_available
    assert quality != "HIGH"


def test_financial_index_download_tries_fallback_ticker(monkeypatch):
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            calls.append(self.ticker)
            if self.ticker == "^CNXFINANCE":
                return pd.DataFrame()
            dates = pd.date_range("2026-07-01", periods=30, freq="D")
            return pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates)

    monkeypatch.setattr(ii.yf, "Ticker", FakeTicker)
    series = ii._download_index_series("NIFTYFIN")

    assert calls == ["^CNXFINANCE", "^CNXFIN"]
    assert len(series) >= 21
    assert np.isfinite(float(series.iloc[-1]))
