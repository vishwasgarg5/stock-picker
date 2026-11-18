"""Point-in-time-safe split adjustment for isolated research inputs.

Raw OHLCV and dividend fields are retained. Only pre-effective-date prices
and volumes are rebased for splits occurring later in the supplied history.
Dividend cash amounts are deliberately not folded into prices.
"""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable, Mapping


PRICE_COLUMNS = ("open", "high", "low", "close")


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _number(value: Any, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {field}") from exc
    if not isfinite(result):
        raise ValueError(f"row {row_number}: non-finite {field}")
    return result


def add_split_adjusted_columns(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copied rows with split-adjusted OHLCV; never mutate source rows."""
    copied = [dict(row) for row in rows]
    indexed: list[tuple[date, str, int, dict[str, Any], float]] = []
    seen: set[tuple[str, date]] = set()

    for index, row in enumerate(copied):
        if "date" not in row or "symbol" not in row:
            raise ValueError(f"row {index + 1}: missing date or symbol")
        symbol = str(row["symbol"]).strip().upper()
        if not symbol:
            raise ValueError(f"row {index + 1}: blank symbol")
        session = _as_date(row["date"])
        key = (symbol, session)
        if key in seen:
            raise ValueError(f"row {index + 1}: duplicate symbol/date {symbol} {session}")
        seen.add(key)
        ratio = _number(row.get("stock_splits", 0) or 0, "stock_splits", index + 1)
        if ratio < 0:
            raise ValueError(f"row {index + 1}: negative stock_splits")
        for column in PRICE_COLUMNS:
            price = _number(row.get(column), column, index + 1)
            if price <= 0:
                raise ValueError(f"row {index + 1}: non-positive {column}")
        volume = _number(row.get("volume"), "volume", index + 1)
        if volume < 0:
            raise ValueError(f"row {index + 1}: negative volume")
        indexed.append((session, symbol, index, row, ratio))

    by_symbol: dict[str, list[tuple[date, int, dict[str, Any], float]]] = {}
    for session, symbol, index, row, ratio in indexed:
        by_symbol.setdefault(symbol, []).append((session, index, row, ratio))

    adjusted_by_index: dict[int, dict[str, Any]] = {}
    for symbol, symbol_rows in by_symbol.items():
        future_factor = 1.0
        factors: dict[int, float] = {}
        for session, index, row, ratio in sorted(symbol_rows, key=lambda item: item[0], reverse=True):
            factors[index] = future_factor
            if ratio > 0:
                future_factor /= ratio
        for session, index, row, ratio in symbol_rows:
            factor = factors[index]
            adjusted = dict(row)
            for column in PRICE_COLUMNS:
                adjusted[f"split_adjusted_{column}"] = _number(row[column], column, index + 1) * factor
            adjusted["split_adjusted_volume"] = _number(row["volume"], "volume", index + 1) / factor
            adjusted_by_index[index] = adjusted

    return [adjusted_by_index[index] for index in range(len(copied))]
