"""Small, side-effect-free validators for historical OHLCV rows."""

from datetime import date, datetime
from math import isfinite
from typing import Mapping

from .ohlcv_contract import REQUIRED_OHLCV_COLUMNS


def parse_canonical_date(value: object) -> date:
    """Parse a date without accepting timestamps, prefixes, whitespace, or suffixes."""
    if isinstance(value, datetime):
        raise ValueError("datetime values are not canonical date-only values")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("date must be a YYYY-MM-DD string")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date is not canonical YYYY-MM-DD")
    return parsed


def validate_ohlcv_row(row: Mapping[str, object]) -> list[str]:
    """Return all detected row errors; an empty list means the row is valid."""
    errors: list[str] = []

    missing = [name for name in REQUIRED_OHLCV_COLUMNS if name not in row]
    if missing:
        return [f"missing required column: {name}" for name in missing]

    try:
        parse_canonical_date(row["date"])
    except (TypeError, ValueError):
        errors.append("date is not a valid canonical YYYY-MM-DD date")

    raw_symbol = row["symbol"]
    if raw_symbol is None:
        errors.append("symbol is missing")
    elif not isinstance(raw_symbol, str):
        errors.append("symbol must be a string")
    elif not raw_symbol.strip():
        errors.append("symbol is empty")
    elif raw_symbol != raw_symbol.strip():
        errors.append("symbol contains surrounding whitespace")

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
