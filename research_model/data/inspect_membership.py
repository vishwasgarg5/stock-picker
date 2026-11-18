"""Read-only inspection of index constituent membership evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"index", "symbol", "as_of_date", "weight_method", "source"}


def inspect_membership_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    dates: set[str] = set()
    indexes: set[str] = set()
    symbols: set[str] = set()
    methods: Counter[str] = Counter()
    duplicate_keys = 0
    invalid_rows = 0
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    rows = 0

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"Membership source missing required columns: {missing}")
        if len(columns) != len(set(columns)):
            raise ValueError("Membership source contains duplicate column names")
        for row in reader:
            rows += 1
            line = rows + 1
            row_errors: list[str] = []
            if None in row:
                row_errors.append("extra CSV fields")
            index = (row.get("index") or "").strip().upper()
            symbol = (row.get("symbol") or "").strip().upper()
            raw_date = (row.get("as_of_date") or "").strip()
            method = (row.get("weight_method") or "").strip()
            evidence_source = (row.get("source") or "").strip()
            if not index:
                row_errors.append("missing index")
            if not symbol:
                row_errors.append("missing symbol")
            try:
                parsed = date.fromisoformat(raw_date)
                if parsed.isoformat() != raw_date:
                    raise ValueError
            except ValueError:
                row_errors.append("invalid or non-canonical as_of_date")
                parsed = None
            if not method:
                row_errors.append("missing weight_method")
            if not evidence_source:
                row_errors.append("missing source")
            if row_errors:
                invalid_rows += 1
                errors.extend(f"row {line}: {message}" for message in row_errors)
                continue

            assert parsed is not None
            dates.add(parsed.isoformat())
            indexes.add(index)
            symbols.add(symbol)
            methods[method] += 1
            key = (index, symbol, parsed.isoformat())
            if key in seen:
                duplicate_keys += 1
                errors.append(f"row {line}: duplicate index/symbol/date key {key}")
            seen.add(key)

    ordered_dates = sorted(dates)
    if invalid_rows or duplicate_keys:
        status = "BLOCKED_INVALID_MEMBERSHIP_ROWS"
    elif not ordered_dates:
        status = "BLOCKED_NO_VALID_MEMBERSHIP_DATES"
    elif len(ordered_dates) == 1:
        status = "SNAPSHOT_ONLY_NOT_HISTORICAL_MEMBERSHIP"
    else:
        status = "MULTI_SNAPSHOT_REQUIRES_MEMBERSHIP_INTERVALS"
    return {
        "status": status,
        "source_path": source.as_posix(),
        "row_count": rows,
        "valid_row_count": rows - invalid_rows,
        "invalid_row_count": invalid_rows,
        "errors": errors[:200],
        "errors_truncated": len(errors) > 200,
        "distinct_as_of_dates": ordered_dates,
        "snapshot_count": len(ordered_dates),
        "index_count": len(indexes),
        "indexes": sorted(indexes),
        "unique_symbol_count": len(symbols),
        "weight_method_counts": dict(sorted(methods.items())),
        "duplicate_index_symbol_date_keys": duplicate_keys,
        "continuous_membership_intervals_present": False,
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_membership_file(args.csv_path)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
