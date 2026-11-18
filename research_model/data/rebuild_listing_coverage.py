"""Rebuild listing coverage audit in a temporary output without mutating source data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.historical_coverage_audit import audit_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--listing-master", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    universe = pd.read_csv(args.universe, dtype={"symbol": str})
    history = pd.read_csv(
        args.ohlcv,
        usecols=["date", "symbol", "open", "high", "low", "close", "volume"],
        dtype={"symbol": str},
        parse_dates=["date"],
    )
    listing_master = pd.read_csv(args.listing_master, dtype=str)
    summary, detail = audit_history(universe, history, listing_master)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_csv, index=False)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "symbols": summary["universe_symbols"],
        "listing_history_status_counts": summary.get("listing_history_status_counts", {}),
        "output_csv": str(args.output_csv),
        "source_modified": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
