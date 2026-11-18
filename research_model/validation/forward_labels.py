"""Deterministic forward labels over explicitly supplied, verified price rows.

This is a pure research utility. It does not load market data, authorize
training, or verify corporate-action evidence. Callers must supply prices
whose split-adjustment provenance satisfies the TargetContract.
"""
from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any, Mapping, Sequence

from research_model.validation.target_contract import TargetContract


def _as_date(value: Any, *, row_index: int) -> date:
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index}: invalid observation date {value!r}") from exc


def _positive_price(value: Any, *, field: str, row_index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"row {row_index}: {field} must be a finite positive number")
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index}: {field} must be a finite positive number") from exc
    if not isfinite(price) or price <= 0:
        raise ValueError(f"row {row_index}: {field} must be a finite positive number")
    return price


def generate_forward_labels(
    rows: Sequence[Mapping[str, Any]],
    contract: TargetContract,
    *,
    symbol_column: str = "symbol",
    date_column: str = "date",
) -> list[dict[str, Any]]:
    """Label each decision row using the price exactly N sessions ahead.

    The entry price is read from the decision row; the exit price is read from
    the same symbol's row N observations later. Rows are never joined across
    symbols. Incomplete trailing horizons are omitted. Output is sorted by
    symbol then decision date and carries target_end for purged splitting.

    Only split-adjusted price returns are implemented. Dividend-inclusive
    cash-flow accounting needs an explicit event/entitlement schedule and is
    intentionally rejected rather than approximated.
    """
    if not isinstance(contract, TargetContract):
        raise TypeError("contract must be a validated TargetContract")
    if contract.return_basis != "split_adjusted_price":
        raise ValueError(
            "cash_dividend_inclusive label generation is not implemented; "
            "an evidenced cash-flow schedule is required"
        )
    if contract.dividend_treatment != "excluded":
        raise ValueError("split_adjusted_price labels must exclude dividends")

    grouped: dict[str, list[tuple[date, int, Mapping[str, Any]]]] = {}
    seen: set[tuple[str, date]] = set()
    for index, row in enumerate(rows):
        symbol = row.get(symbol_column)
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"row {index}: {symbol_column} must be a non-empty string")
        if date_column not in row:
            raise ValueError(f"row {index}: missing {date_column}")
        observed = _as_date(row[date_column], row_index=index)
        key = (symbol, observed)
        if key in seen:
            raise ValueError(f"duplicate symbol/date key: {symbol} {observed}")
        seen.add(key)
        grouped.setdefault(symbol, []).append((observed, index, row))

    labels: list[dict[str, Any]] = []
    for symbol in sorted(grouped):
        observations = sorted(grouped[symbol], key=lambda item: item[0])
        for position, (decision_date, source_index, decision_row) in enumerate(observations):
            target_position = position + contract.horizon_sessions
            if target_position >= len(observations):
                continue
            target_date, target_source_index, target_row = observations[target_position]
            if contract.entry_price_field not in decision_row:
                raise ValueError(
                    f"row {source_index}: missing entry price field "
                    f"{contract.entry_price_field!r}"
                )
            if contract.exit_price_field not in target_row:
                raise ValueError(
                    f"row {target_source_index}: missing exit price field "
                    f"{contract.exit_price_field!r}"
                )
            if contract.decision_price_field not in decision_row:
                raise ValueError(
                    f"row {source_index}: missing decision price field "
                    f"{contract.decision_price_field!r}"
                )
            decision_price = _positive_price(
                decision_row[contract.decision_price_field],
                field=contract.decision_price_field,
                row_index=source_index,
            )
            entry = _positive_price(
                decision_row[contract.entry_price_field],
                field=contract.entry_price_field,
                row_index=source_index,
            )
            exit_price = _positive_price(
                target_row[contract.exit_price_field],
                field=contract.exit_price_field,
                row_index=target_source_index,
            )
            labels.append({
                symbol_column: symbol,
                "decision_date": decision_date.isoformat(),
                "decision_price": decision_price,
                "entry_price": entry,
                "exit_price": exit_price,
                contract.label_end_field: target_date.isoformat(),
                "target_return": exit_price / entry - 1.0,
                "horizon_sessions": contract.horizon_sessions,
                "return_basis": contract.return_basis,
                "dividend_treatment": contract.dividend_treatment,
                "label_status": "SYNTHETIC_OR_CALLER_SUPPLIED_INPUTS_NOT_EVIDENCE_VERIFIED",
            })
    return labels
