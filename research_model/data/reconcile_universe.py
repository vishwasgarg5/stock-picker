"""Compare symbols in the raw OHLCV file with the reviewed 500-symbol coverage universe."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def reconcile_universe(
    ohlcv_path: str | Path, audit_csv_path: str | Path, universe_csv_path: str | Path
) -> dict[str, Any]:
    """Report raw-only and audit-only symbols plus listing-provenance exceptions."""
    raw_counts: Counter[str] = Counter()
    with Path(ohlcv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"date", "symbol"}.issubset(reader.fieldnames):
            raise ValueError("OHLCV source must contain date and symbol columns")
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if symbol:
                raw_counts[symbol] += 1

    audit_symbols: set[str] = set()
    unknown: list[str] = []
    prelisting: list[dict[str, str]] = []
    with Path(audit_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "listing_history_status", "first_date", "listing_date"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Coverage audit missing required columns: {sorted(required)}")
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            audit_symbols.add(symbol)
            status = (row.get("listing_history_status") or "").strip()
            if status == "LISTING_DATE_UNKNOWN":
                unknown.append(symbol)
            elif status == "DATA_PRECEDES_LISTING_DATE":
                prelisting.append({
                    "symbol": symbol,
                    "first_date": row.get("first_date", ""),
                    "listing_date": row.get("listing_date", ""),
                })

    current_universe: set[str] = set()
    with Path(universe_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError("Current universe file must contain a symbol column")
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if symbol:
                current_universe.add(symbol)

    raw_symbols = set(raw_counts)
    raw_only = sorted(raw_symbols - audit_symbols)
    audit_only = sorted(audit_symbols - raw_symbols)
    raw_outside_universe = sorted(raw_symbols - current_universe)
    current_universe_missing_ohlcv = sorted(current_universe - raw_symbols)
    universe_audit_mismatch = sorted(current_universe ^ audit_symbols)
    return {
        "status": "REVIEW_REQUIRED" if (
            raw_outside_universe or current_universe_missing_ohlcv or universe_audit_mismatch or unknown or prelisting
        ) else "MATCH",
        "raw_ohlcv_symbol_count": len(raw_symbols),
        "current_universe_symbol_count": len(current_universe),
        "current_universe_missing_ohlcv": current_universe_missing_ohlcv,
        "raw_symbols_outside_current_universe": raw_outside_universe,
        "current_universe_not_in_coverage_audit": sorted(current_universe - audit_symbols),
        "coverage_audit_not_in_current_universe": sorted(audit_symbols - current_universe),
        "coverage_audit_symbol_count": len(audit_symbols),
        "raw_ohlcv_row_count": sum(raw_counts.values()),
        "raw_only_symbols": raw_only,
        "audit_only_symbols": audit_only,
        "raw_only_symbol_row_counts": {symbol: raw_counts[symbol] for symbol in raw_only},
        "symbols_without_verified_listing_date": sorted(unknown),
        "data_precedes_listing_date": sorted(prelisting, key=lambda item: item["symbol"]),
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", required=True, type=Path)
    parser.add_argument("--coverage-audit", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = reconcile_universe(args.ohlcv, args.coverage_audit, args.universe)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
