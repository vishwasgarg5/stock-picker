"""Compare symbols in the raw OHLCV file with the reviewed 500-symbol coverage universe."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_observed_sessions(value: str, symbol: str) -> int:
    """Parse an exact non-negative integer count; never truncate or coerce."""
    raw = (value or "").strip()
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"Invalid observed_sessions for {symbol}: {value!r}")
    return int(raw)


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
    audit_counts: dict[str, int] = {}
    has_observed_counts = False
    unknown: list[str] = []
    prelisting: list[dict[str, str]] = []
    with Path(audit_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "listing_history_status", "first_date", "listing_date"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Coverage audit missing required columns: {sorted(required)}")
        has_observed_counts = "observed_sessions" in (reader.fieldnames or [])
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            if symbol in audit_symbols:
                raise ValueError(f"Duplicate symbol in coverage audit: {symbol}")
            audit_symbols.add(symbol)
            if has_observed_counts:
                audit_counts[symbol] = _parse_observed_sessions(
                    row.get("observed_sessions") or "", symbol
                )
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
    excluded_synthetic: set[str] = set()
    with Path(universe_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError("Current universe file must contain a symbol column")
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            # Match the established V1 universe policy: DUMMY* rows are
            # synthetic placeholders, never tradable securities.
            if symbol.upper().startswith("DUMMY"):
                excluded_synthetic.add(symbol)
                continue
            current_universe.add(symbol)

    raw_symbols = set(raw_counts)
    raw_only = sorted(raw_symbols - audit_symbols)
    audit_only = sorted(audit_symbols - raw_symbols)
    raw_outside_universe = sorted(raw_symbols - current_universe)
    current_universe_missing_ohlcv = sorted(current_universe - raw_symbols)
    universe_audit_mismatch = sorted(current_universe ^ audit_symbols)
    common_symbols = raw_symbols & audit_symbols
    count_mismatches = [
        {
            "symbol": symbol,
            "raw_rows": raw_counts[symbol],
            "audit_rows": audit_counts[symbol],
            "difference": raw_counts[symbol] - audit_counts[symbol],
        }
        for symbol in sorted(common_symbols)
        if has_observed_counts and raw_counts[symbol] != audit_counts[symbol]
    ]
    audit_row_total = sum(audit_counts.values()) if has_observed_counts else None
    return {
        "status": "REVIEW_REQUIRED" if (
            raw_outside_universe or current_universe_missing_ohlcv or universe_audit_mismatch
            or unknown or prelisting or count_mismatches
        ) else "MATCH",
        "raw_ohlcv_symbol_count": len(raw_symbols),
        "current_universe_symbol_count": len(current_universe),
        "excluded_synthetic_universe_symbols": sorted(excluded_synthetic),
        "current_universe_missing_ohlcv": current_universe_missing_ohlcv,
        "raw_symbols_outside_current_universe": raw_outside_universe,
        "current_universe_not_in_coverage_audit": sorted(current_universe - audit_symbols),
        "coverage_audit_not_in_current_universe": sorted(audit_symbols - current_universe),
        "coverage_audit_symbol_count": len(audit_symbols),
        "raw_ohlcv_row_count": sum(raw_counts.values()),
        "coverage_audit_observed_row_count": audit_row_total,
        "common_symbol_row_count_mismatch_count": len(count_mismatches),
        "common_symbol_row_count_mismatches": count_mismatches,
        "common_symbol_row_count_difference_total": sum(item["difference"] for item in count_mismatches),
        "raw_only_row_count": sum(raw_counts[symbol] for symbol in raw_only),
        "audit_only_observed_row_count": (
            sum(audit_counts.get(symbol, 0) for symbol in audit_only) if has_observed_counts else None
        ),
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
