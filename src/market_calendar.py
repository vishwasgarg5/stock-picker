from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CALENDAR_FILE = DATA_DIR / "nse_market_calendar.json"
AUDIT_FILE = DATA_DIR / "nse_market_calendar_audit.json"
NSE_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_HOLIDAY_PAGE = "https://www.nseindia.com/resources/exchange-communication-holidays"
NSE_CIRCULAR_ARCHIVE = "https://www.nseindia.com/market-data/circulars"
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_OPEN = "09:15"
DEFAULT_CLOSE = "15:30"


def _iso_day(value: Any) -> str:
    raw = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed.date().isoformat()


def _calendar_payload() -> dict:
    if not CALENDAR_FILE.exists():
        return {"schema_version": 1, "exchange": "NSE", "segment": "CM", "holidays": [], "special_sessions": []}
    payload = json.loads(CALENDAR_FILE.read_text(encoding="utf-8"))
    if payload.get("exchange") != "NSE" or payload.get("segment") != "CM":
        raise ValueError("Calendar must explicitly target NSE Capital Market (equities) segment")
    return payload


def _extract_cm_holidays(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        # NSE's holiday-master API groups entries by segment; CM is the cash/equity market.
        rows = payload.get("CM", payload.get("cm", payload.get("Capital Market", [])))
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("NSE holiday API did not return a non-empty CM holiday list")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = next((row[k] for k in ("tradingDate", "date", "holidayDate", "trading_date") if row.get(k)), None)
        if raw_date is None:
            continue
        description = next((row[k] for k in ("description", "tradingDateDescription", "holiday", "name") if row.get(k)), "NSE trading holiday")
        result.append({"date": _iso_day(raw_date), "name": str(description).strip(), "type": "FULL_DAY_HOLIDAY", "source": NSE_HOLIDAY_API})
    if not result:
        raise ValueError("Could not parse any NSE CM holiday dates from the official response")
    return sorted({row["date"]: row for row in result}.values(), key=lambda x: x["date"])


def refresh_official_calendar(timeout: int = 25) -> dict:
    """Refresh official NSE CM holidays; preserve separately verified special-session overrides.

    The old calendar is never overwritten when the official source is unavailable,
    malformed, or returns an unexpected schema. This makes the caller fail closed.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 StockPicker market-calendar updater",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    response = requests.get(NSE_HOLIDAY_API, headers=headers, timeout=timeout)
    response.raise_for_status()
    official_holidays = _extract_cm_holidays(response.json())
    old = _calendar_payload()
    special = old.get("special_sessions", [])
    # Special sessions are separate from holidays: they explicitly override a
    # weekend/full-day holiday only when supported by a published exchange circular.
    payload = {
        "schema_version": 1,
        "exchange": "NSE",
        "segment": "CM",
        "timezone": "Asia/Kolkata",
        "updated_at_utc": datetime.now().astimezone().isoformat(),
        "source": NSE_HOLIDAY_API,
        "source_page": NSE_HOLIDAY_PAGE,
        "holidays": official_holidays,
        "special_sessions": special,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="nse-calendar-", suffix=".json", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, CALENDAR_FILE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    audit = {
        "status": "REFRESHED",
        "refreshed_at_utc": datetime.now().astimezone().isoformat(),
        "source": NSE_HOLIDAY_API,
        "holiday_count": len(official_holidays),
        "special_session_count": len(special),
    }
    AUDIT_FILE.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _entry_date(entry: dict) -> date:
    return date.fromisoformat(str(entry["date"]))


def session_for(day: date | datetime | str, payload: dict | None = None) -> dict:
    candidate = date.fromisoformat(day[:10]) if isinstance(day, str) else (day.date() if isinstance(day, datetime) else day)
    calendar = payload or _calendar_payload()
    special = [x for x in calendar.get("special_sessions", []) if _entry_date(x) == candidate and x.get("status", "CONFIRMED") == "CONFIRMED"]
    if special:
        return {"is_open": True, "kind": "SPECIAL_SESSION", **special[-1]}
    holiday = next((x for x in calendar.get("holidays", []) if _entry_date(x) == candidate), None)
    if holiday:
        return {"is_open": False, "kind": "FULL_DAY_HOLIDAY", **holiday}
    if candidate.weekday() >= 5:
        return {"is_open": False, "kind": "WEEKEND", "date": candidate.isoformat(), "name": "Weekend"}
    return {
        "is_open": True, "kind": "NORMAL_SESSION", "date": candidate.isoformat(),
        "name": "Normal NSE equity session", "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
    }


def is_trading_session(day: date | datetime | str, payload: dict | None = None) -> bool:
    return bool(session_for(day, payload)["is_open"])


def next_trading_session(day: date | datetime | str, payload: dict | None = None, max_days: int = 370) -> date:
    start = date.fromisoformat(day[:10]) if isinstance(day, str) else (day.date() if isinstance(day, datetime) else day)
    for offset in range(1, max_days + 1):
        candidate = start + timedelta(days=offset)
        if is_trading_session(candidate, payload):
            return candidate
    raise RuntimeError(f"No NSE trading session found within {max_days} days after {start}")


def check_today(github_output: str | None = None, refresh: bool = True) -> dict:
    if refresh:
        refresh_official_calendar()
    today = datetime.now(IST).date()
    session = session_for(today)
    result = {"date": today.isoformat(), **session}
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"should_run={'true' if session['is_open'] else 'false'}\n")
            handle.write(f"session_kind={session['kind']}\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-today", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--github-output")
    args = parser.parse_args()
    if args.check_today:
        check_today(args.github_output, refresh=not args.no_refresh)
    else:
        print(json.dumps(refresh_official_calendar(), indent=2))


if __name__ == "__main__":
    main()
