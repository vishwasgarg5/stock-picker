"""Explain raw OHLCV symbols absent from the current reviewed universe.

This is a diagnostic only: it never drops rows, expands the universe, or grants
historical eligibility. Every raw-only symbol needs a point-in-time disposition.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


def _symbols(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a symbol column")
        return {(row.get("symbol") or "").strip() for row in reader if (row.get("symbol") or "").strip()}


def diagnose_raw_symbol_history(
    ohlcv_path: str | Path, universe_path: str | Path, coverage_path: str | Path
) -> dict[str, Any]:
    raw_path = Path(ohlcv_path)
    current_universe = _symbols(Path(universe_path))
    audited_symbols = _symbols(Path(coverage_path))
    counts: Counter[str] = Counter()
    first: dict[str, str] = {}
    last: dict[str, str] = {}
    invalid_dates = 0

    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"date", "symbol"}.issubset(reader.fieldnames):
            raise ValueError("OHLCV source must contain date and symbol columns")
        for line_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip()
            value = row.get("date") or ""
            try:
                parsed = date.fromisoformat(value)
                if parsed.isoformat() != value:
                    raise ValueError("noncanonical date")
            except ValueError:
                invalid_dates += 1
                continue
            if not symbol:
                continue
            counts[symbol] += 1
            if symbol not in first or value < first[symbol]:
                first[symbol] = value
            if symbol not in last or value > last[symbol]:
                last[symbol] = value

    raw_symbols = set(counts)
    raw_only = sorted(raw_symbols - audited_symbols)
    return {
        "status": "REVIEW_REQUIRED" if raw_only or invalid_dates else "MATCH",
        "raw_row_count": sum(counts.values()),
        "raw_symbol_count": len(raw_symbols),
        "current_universe_symbol_count": len(current_universe),
        "coverage_audit_symbol_count": len(audited_symbols),
        "invalid_date_rows": invalid_dates,
        "raw_only_symbol_count": len(raw_only),
        "raw_only_symbols": [
            {
                "symbol": symbol,
                "row_count": counts[symbol],
                "first_date": first[symbol],
                "last_date": last[symbol],
                "in_current_universe": symbol in current_universe,
                "disposition": "REQUIRES_POINT_IN_TIME_HISTORICAL_UNIVERSE_REVIEW",
            }
            for symbol in raw_only
        ],
        "interpretation": (
            "Diagnostic only. Raw-only rows are retained. Do not add symbols to the "
            "current universe or discard historical rows without dated, authoritative evidence."
        ),
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--coverage-audit", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnose_raw_symbol_history(args.ohlcv, args.universe, args.coverage_audit)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
