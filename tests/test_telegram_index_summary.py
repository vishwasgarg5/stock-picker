import json

import pandas as pd

from src import telegram


def test_index_market_lines_show_all_five_indices_and_market_context(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "DATA", tmp_path)
    (tmp_path / "index_intelligence_config.json").write_text(json.dumps({
        "market_regime": "BEAR",
        "risk_level": "RISK_OFF",
        "market_intelligence_score": 31.5,
        "index_data_available": True,
        "data_quality": "HIGH",
        "news_headline": "Crude oil rises & markets fall",
    }))
    pd.DataFrame([
        {
            "index": name,
            "as_of": "2026-10-09",
            "last_close": 25000 + i,
            "change_1d_pct": -0.5,
            "return_5d_pct": -1.2,
            "return_20d_pct": -3.4,
            "regime": "BEAR",
        }
        for i, name in enumerate(
            ["NIFTY50", "BANKNIFTY", "NIFTYIT", "NIFTYAUTO", "NIFTYFIN"]
        )
    ]).to_csv(tmp_path / "index_intelligence_summary.csv", index=False)

    lines = telegram._index_market_lines(pd.Timestamp("2026-10-10"))
    message = "\n".join(lines)

    for label in ["NIFTY 50", "BANK NIFTY", "NIFTY IT", "NIFTY AUTO", "NIFTY FIN"]:
        assert label in message
    assert "RISK_OFF" in message
    assert "25000.00" in message
    assert "Crude oil rises &amp; markets fall" in message
    assert "Index data as of:" in message
    assert "All indices available:" in message


def test_index_market_lines_mark_missing_indices_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "DATA", tmp_path)
    lines = telegram._index_market_lines(pd.Timestamp("2026-10-10"))
    message = "\n".join(lines)

    assert "NIFTY 50" in message
    assert "BANK NIFTY" in message
    assert "NO DATA" in message
    assert "Index data unavailable" in message
