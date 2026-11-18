"""Fail-closed, deterministic year/quarter partitioning for OHLCV CSV files."""

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .manifest import sha256_file, write_json
from .partition_keys import partition_key
from .validate_ohlcv import validate_ohlcv_row


def _read_and_validate(source: Path) -> tuple[list[str], dict[str, list[dict[str, str]]], dict[str, Any]]:
    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    symbols: set[str] = set()
    dates: list[str] = []

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("source CSV has no header")
        columns = list(reader.fieldnames)
        if len(columns) != len(set(columns)):
            raise ValueError("source CSV contains duplicate column names")

        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"line {line_number}: row has more values than the header")
            errors = validate_ohlcv_row(row)
            if errors:
                raise ValueError(f"line {line_number}: " + "; ".join(errors))

            raw_date = row["date"]
            parsed_date = date.fromisoformat(raw_date)
            symbol = row["symbol"].strip()
            if raw_date != parsed_date.isoformat():
                raise ValueError(f"line {line_number}: date is not canonical YYYY-MM-DD")
            if symbol != row["symbol"]:
                raise ValueError(f"line {line_number}: symbol contains surrounding whitespace")

            key = (raw_date, symbol)
            if key in seen:
                raise ValueError(f"line {line_number}: duplicate symbol/date key {symbol}/{raw_date}")
            seen.add(key)

            for action_column in ("dividends", "stock_splits"):
                if action_column in row and row[action_column] not in (None, ""):
                    try:
                        action_value = float(row[action_column])
                    except (TypeError, ValueError):
                        raise ValueError(f"line {line_number}: {action_column} is not numeric")
                    if not math.isfinite(action_value) or action_value < 0:
                        raise ValueError(f"line {line_number}: {action_column} must be finite and nonnegative")

            partitions[partition_key(parsed_date)].append(dict(row))
            symbols.add(symbol)
            dates.append(raw_date)
            row_count += 1

    if row_count == 0:
        raise ValueError("source CSV contains no data rows")

    for rows in partitions.values():
        rows.sort(key=lambda item: (item["date"], item["symbol"]))

    source_info = {
        "row_count": row_count,
        "symbol_count": len(symbols),
        "min_date": min(dates),
        "max_date": max(dates),
    }
    return columns, dict(partitions), source_info


def partition_ohlcv(
    input_path: str | Path,
    output_root: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate and plan partitions; write only when apply=True.

    Writes into a new directory via a temporary sibling and atomic rename.
    Existing output paths are never overwritten or cleaned up.
    """
    source = Path(input_path).resolve()
    output = Path(output_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or source.is_relative_to(output):
        raise ValueError("output root must not contain or equal the source file")

    columns, partitions, source_info = _read_and_validate(source)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            **source_info,
        },
        "schema": columns,
        "partition_count": len(partitions),
        "partitions": [],
        "write_applied": bool(apply),
    }

    for key in sorted(partitions):
        rows = partitions[key]
        dates = [row["date"] for row in rows]
        manifest["partitions"].append({
            "key": key,
            "path": f"partitioned/ohlcv/{key}/part-000.csv",
            "row_count": len(rows),
            "symbol_count": len({row["symbol"] for row in rows}),
            "min_date": min(dates),
            "max_date": max(dates),
        })

    if not apply:
        return manifest
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for partition in manifest["partitions"]:
            key = partition["key"]
            destination = temporary / "partitioned" / "ohlcv" / key / "part-000.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(partitions[key])
            partition["sha256"] = sha256_file(destination)

        manifest_path = temporary / "manifests" / "partition_manifest.json"
        write_json(manifest_path, manifest)
        os.replace(temporary, output)
    except Exception:
        # The temporary directory is unique to this invocation; do not touch user paths.
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source OHLCV CSV (read-only)")
    parser.add_argument("--output-root", required=True, help="new output directory; must not exist")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write partitions; without this flag, perform validation and print a plan only",
    )
    args = parser.parse_args()
    result = partition_ohlcv(args.input, args.output_root, apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
