"""Read-only audit of embedded corporate-action fields and provenance.

This audit does not verify actions against an exchange/vendor source and does
not create adjusted prices or total-return data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ACTION_COLUMNS = ("dividends", "stock_splits")
PROVENANCE_CANDIDATES = (
    "source", "source_name", "source_url", "vendor", "provider",
    "action_source", "exchange", "retrieved_at", "effective_date",
)


def _audit_action(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    events = 0
    invalid = 0
    negative = 0
    symbols: set[str] = set()
    dates: list[str] = []
    values: Counter[str] = Counter()
    for number, row in enumerate(rows, start=2):
        raw = (row.get(column) or "").strip()
        if not raw:
            raw = "0"
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
            invalid += 1
            continue
        if value < 0:
            negative += 1
            continue
        if value == 0:
            continue
        events += 1
        symbol = (row.get("symbol") or "").strip().upper()
        raw_date = (row.get("date") or "").strip()
        try:
            event_date = date.fromisoformat(raw_date[:10]).isoformat()
        except ValueError:
            invalid += 1
            continue
        symbols.add(symbol)
        dates.append(event_date)
        values[format(value, ".12g")] += 1
    return {
        "event_rows": events,
        "unique_symbols": len(symbols),
        "first_event_date": min(dates) if dates else None,
        "last_event_date": max(dates) if dates else None,
        "distinct_values": len(values),
        "most_common_values": [
            {"value": value, "rows": count}
            for value, count in values.most_common(10)
        ],
        "invalid_values": invalid,
        "negative_values_held_for_review": negative,
    }


def audit_corporate_actions(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = [column for column in ACTION_COLUMNS if column not in columns]
    provenance_columns = [column for column in PROVENANCE_CANDIDATES if column in columns]
    provenance_values: dict[str, dict[str, int]] = {}
    for column in provenance_columns:
        counts = Counter(
            (row.get(column) or "").strip()
            for row in rows
            if (row.get(column) or "").strip()
        )
        provenance_values[column] = dict(counts.most_common(20))
    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "row_count": len(rows),
        "action_columns_present": [column for column in ACTION_COLUMNS if column in columns],
        "action_columns_missing": missing,
        "actions": {
            column: _audit_action(rows, column)
            for column in ACTION_COLUMNS if column in columns
        },
        "provenance_columns_present": provenance_columns,
        "provenance_column_value_samples": provenance_values,
        "event_level_provenance_status": (
            "FIELDS_PRESENT_BUT_UNVERIFIED"
            if provenance_columns else "NO_EVENT_LEVEL_PROVENANCE_FIELDS"
        ),
        "exchange_independent_verification": "NOT_PERFORMED",
        "dividend_total_return_convention": "NOT_DEFINED",
        "input_modified": False,
        "limitations": [
            "Non-zero embedded values are observations, not independently verified corporate actions.",
            "Source/provider fields, if present, are descriptive metadata and do not prove authenticity.",
            "Negative action values are held for review rather than silently interpreted.",
            "Dividend cash flows are not converted into reinvested total returns.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_corporate_actions(args.csv_path)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
