"""Small, side-effect-free validators for historical OHLCV rows."""

from datetime import date, datetime
from math import isfinite
from typing import Mapping

from .ohlcv_contract import REQUIRED_OHLCV_COLUMNS


def validate_ohlcv_row(row: Mapping[str, object]) -> list[str]:
    """Return all detected row errors; an empty list means the row is valid."""
    errors: list[str] = []

    missing = [name for name in REQUIRED_OHLCV_COLUMNS if name not in row]
    if missing:
        return [f"missing required column: {name}" for name in missing]

    raw_date = row["date"]
    try:
        if isinstance(raw_date, datetime):
            parsed_date = raw_date.date()
        elif isinstance(raw_date, date):
            parsed_date = raw_date
        else:
            parsed_date = date.fromisoformat(str(raw_date)[:10])
        if parsed_date.isoformat() != str(raw_date)[:10]:
            errors.append("date is not canonical YYYY-MM-DD")
    except (TypeError, ValueError):
        errors.append("date is not a valid YYYY-MM-DD date")

    symbol = str(row["symbol"]).strip()
    if not symbol:
        errors.append("symbol is empty")

    numeric: dict[str, float] = {}
    for name in ("open", "high", "low", "close", "volume"):
        try:
            value = float(row[name])
            if not isfinite(value):
                errors.append(f"{name} is not finite")
            elif name == "volume" and value < 0:
                errors.append("volume is negative")
            elif name != "volume" and value <= 0:
                errors.append(f"{name} must be positive")
            numeric[name] = value
        except (TypeError, ValueError):
            errors.append(f"{name} is not numeric")

    if all(name in numeric and isfinite(numeric[name]) for name in ("open", "high", "low", "close")):
        if numeric["high"] < max(numeric["open"], numeric["close"], numeric["low"]):
            errors.append("high is below another OHLC value")
        if numeric["low"] > min(numeric["open"], numeric["close"], numeric["high"]):
            errors.append("low is above another OHLC value")

    return errors
