"""Streaming audit of the committed OHLCV source; never modifies input data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .ohlcv_contract import REQUIRED_OHLCV_COLUMNS


def inspect_ohlcv_file(path: str | Path) -> dict[str, Any]:
    """Inspect a CSV in one pass and return a deterministic source manifest."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    digest = hashlib.sha256()
    with source.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)

    row_count = 0
    valid_date_rows = 0
    invalid_dates = 0
    invalid_numeric_rows = 0
    invalid_ohlc_rows = 0
    duplicate_symbol_dates = 0
    symbols: dict[str, dict[str, Any]] = {}
    quarter_rows: Counter[str] = Counter()
    seen_keys: set[tuple[str, str]] = set()
    error_counts: Counter[str] = Counter()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_columns = [name for name in REQUIRED_OHLCV_COLUMNS if name not in columns]
        if missing_columns:
            raise ValueError(f"OHLCV source missing required columns: {missing_columns}")

        for row in reader:
            row_count += 1
            symbol = str(row.get("symbol") or "").strip()
            raw_date = str(row.get("date") or "").strip()
            try:
                parsed_date = date.fromisoformat(raw_date[:10])
                valid_date_rows += 1
                quarter = (parsed_date.month - 1) // 3 + 1
                quarter_rows[f"{parsed_date.year:04d}-Q{quarter}"] += 1
            except ValueError:
                invalid_dates += 1
                error_counts["invalid_date"] += 1
                continue

            key = (symbol, parsed_date.isoformat())
            if key in seen_keys:
                duplicate_symbol_dates += 1
                error_counts["duplicate_symbol_date"] += 1
            else:
                seen_keys.add(key)

            stats = symbols.setdefault(
                symbol,
                {"rows": 0, "first_date": parsed_date.isoformat(), "last_date": parsed_date.isoformat()},
            )
            stats["rows"] += 1
            stats["first_date"] = min(stats["first_date"], parsed_date.isoformat())
            stats["last_date"] = max(stats["last_date"], parsed_date.isoformat())

            numeric: dict[str, float] = {}
            numeric_ok = True
            for column in ("open", "high", "low", "close", "volume"):
                try:
                    value = float(row[column])
                    if value != value or value in (float("inf"), float("-inf")):
                        raise ValueError
                    numeric[column] = value
                    if column == "volume" and value < 0:
                        numeric_ok = False
                    if column != "volume" and value <= 0:
                        numeric_ok = False
                except (TypeError, ValueError, KeyError):
                    numeric_ok = False
            if not numeric_ok:
                invalid_numeric_rows += 1
                error_counts["invalid_numeric_or_nonpositive_price"] += 1
            elif (
                numeric["high"] < max(numeric["open"], numeric["close"], numeric["low"])
                or numeric["low"] > min(numeric["open"], numeric["close"], numeric["high"])
            ):
                invalid_ohlc_rows += 1
                error_counts["inconsistent_ohlc"] += 1

    dates = [v["first_date"] for v in symbols.values()] + [v["last_date"] for v in symbols.values()]
    return {
        "source_path": source.as_posix(),
        "source_bytes": source.stat().st_size,
        "source_sha256": digest.hexdigest(),
        "columns": columns,
        "required_columns": list(REQUIRED_OHLCV_COLUMNS),
        "optional_corporate_action_columns_present": [
            name for name in ("dividends", "stock_splits", "adj_close", "adjusted_close") if name in columns
        ],
        "row_count": row_count,
        "symbol_count": len(symbols),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "valid_date_rows": valid_date_rows,
        "invalid_dates": invalid_dates,
        "invalid_numeric_rows": invalid_numeric_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "duplicate_symbol_dates": duplicate_symbol_dates,
        "quarter_row_counts": dict(sorted(quarter_rows.items())),
        "symbols": dict(sorted(symbols.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, help="Write the full manifest to this generated path")
    args = parser.parse_args()
    report = inspect_ohlcv_file(args.csv_path)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "symbols"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
