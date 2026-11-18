"""Independent integrity verification for a materialized OHLCV partition archive."""

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .manifest import sha256_file
from .partition_keys import partition_key
from .validate_ohlcv import validate_ohlcv_row


def verify_partition_archive(output_root: str | Path) -> dict[str, Any]:
    """Verify every manifest-listed partition and aggregate archive invariants."""
    root = Path(output_root).resolve()
    manifest_path = root / "manifests" / "partition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"partition manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("unsupported or missing manifest_version")
    if manifest.get("write_applied") is not True:
        raise ValueError("manifest does not certify an applied partition write")

    schema = manifest.get("schema")
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    if not isinstance(schema, list) or not required.issubset(schema):
        raise ValueError("manifest schema is missing required OHLCV columns")
    if len(schema) != len(set(schema)):
        raise ValueError("manifest schema contains duplicate columns")

    entries = manifest.get("partitions")
    if not isinstance(entries, list) or len(entries) != manifest.get("partition_count"):
        raise ValueError("partition_count does not match manifest partition entries")

    seen_keys: set[tuple[str, str]] = set()
    total_rows = 0
    total_symbols: set[str] = set()
    verified = []

    for entry in entries:
        key = entry.get("key")
        expected_rel = f"partitioned/ohlcv/{key}/part-000.csv"
        if entry.get("path") != expected_rel:
            raise ValueError(f"unexpected partition path for key {key!r}")
        relative = Path(expected_rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe partition path in manifest")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"partition file missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {key}")

        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != schema:
                raise ValueError(f"schema mismatch for {key}")
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"{key} line {line_number}: excess CSV values")
                errors = validate_ohlcv_row(row)
                if errors:
                    raise ValueError(f"{key} line {line_number}: " + "; ".join(errors))
                parsed = date.fromisoformat(row["date"])
                if row["date"] != parsed.isoformat() or partition_key(parsed) != key:
                    raise ValueError(f"{key} line {line_number}: date is noncanonical or in wrong partition")
                symbol = row["symbol"]
                if symbol != symbol.strip():
                    raise ValueError(f"{key} line {line_number}: symbol whitespace")
                identity = (row["date"], symbol)
                if identity in seen_keys:
                    raise ValueError(f"duplicate symbol/date key {symbol}/{row['date']}")
                seen_keys.add(identity)
                for action in ("dividends", "stock_splits"):
                    if action in row and row[action] not in ("", None):
                        try:
                            value = float(row[action])
                        except (TypeError, ValueError):
                            raise ValueError(f"{key} line {line_number}: {action} is not numeric")
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"{key} line {line_number}: {action} must be finite and nonnegative")
                rows.append(row)

        if not rows:
            raise ValueError(f"empty partition: {key}")
        ordered = sorted(rows, key=lambda row: (row["date"], row["symbol"]))
        if rows != ordered:
            raise ValueError(f"partition is not deterministically sorted: {key}")
        dates = [row["date"] for row in rows]
        symbols = {row["symbol"] for row in rows}
        observed = {
            "row_count": len(rows),
            "symbol_count": len(symbols),
            "min_date": min(dates),
            "max_date": max(dates),
        }
        for field, value in observed.items():
            if entry.get(field) != value:
                raise ValueError(f"{key}: manifest {field} mismatch")
        total_rows += len(rows)
        total_symbols.update(symbols)
        verified.append({"key": key, **observed, "sha256": actual_hash})

    source = manifest.get("source", {})
    if total_rows != source.get("row_count"):
        raise ValueError("aggregate row count does not match source manifest")
    if len(total_symbols) != source.get("symbol_count"):
        raise ValueError("aggregate symbol count does not match source manifest")

    all_dates = [identity[0] for identity in seen_keys]
    if min(all_dates) != source.get("min_date") or max(all_dates) != source.get("max_date"):
        raise ValueError("aggregate date range does not match source manifest")

    return {
        "status": "PASS",
        "partition_count": len(verified),
        "row_count": total_rows,
        "symbol_count": len(total_symbols),
        "partitions": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", help="materialized archive root")
    args = parser.parse_args()
    print(json.dumps(verify_partition_archive(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
