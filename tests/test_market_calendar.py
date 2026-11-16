from datetime import date

import pytest

from src import market_calendar as calendar


def _payload():
    return {
        "exchange": "NSE",
        "segment": "CM",
        "holidays": [
            {"date": "2026-01-26", "name": "Republic Day", "type": "FULL_DAY_HOLIDAY"}
        ],
        "special_sessions": [
            {"date": "2026-11-08", "name": "Diwali Muhurat Trading", "status": "CONFIRMED",
             "open_time": "14:20", "close_time": "15:30"}
        ],
    }


def test_weekend_is_closed_by_default():
    assert calendar.session_for("2026-11-14", _payload())["kind"] == "WEEKEND"
    assert not calendar.is_trading_session("2026-11-14", _payload())


def test_weekday_exchange_holiday_is_closed():
    result = calendar.session_for("2026-01-26", _payload())
    assert result["kind"] == "FULL_DAY_HOLIDAY"
    assert not result["is_open"]


def test_confirmed_special_session_overrides_weekend():
    result = calendar.session_for("2026-11-08", _payload())
    assert result["is_open"]
    assert result["kind"] == "SPECIAL_SESSION"
    assert result["open_time"] == "14:20"


def test_unconfirmed_special_session_does_not_open_market():
    payload = _payload()
    payload["special_sessions"][0]["status"] = "PROPOSED"
    assert not calendar.is_trading_session("2026-11-08", payload)


def test_next_trading_session_skips_weekends_and_holidays():
    payload = _payload()
    result = calendar.next_trading_session("2026-01-23", payload)
    assert result == date(2026, 1, 27)


def test_nse_holiday_api_parser_handles_cm_segment():
    rows = {"CM": [
        {"tradingDate": "26-Jan-2026", "description": "Republic Day"},
        {"tradingDate": "03-Mar-2026", "description": "Holi"},
    ]}
    result = calendar._extract_cm_holidays(rows)
    assert [x["date"] for x in result] == ["2026-01-26", "2026-03-03"]


def test_nse_holiday_api_parser_fails_closed_on_unknown_schema():
    with pytest.raises(ValueError):
        calendar._extract_cm_holidays({"unknown": []})
