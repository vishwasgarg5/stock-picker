"""Fail-closed, deterministic, bounded-memory OHLCV quarter partitioning."""

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from .manifest import sha256_file, write_json
from .partition_keys import partition_key
from .validate_ohlcv import validate_ohlcv_row


def _validated_rows(source: Path, database: Path) -> tuple[list[str], dict[str, Any]]:
    """Stream validated input rows to a disk-backed SQLite staging table."""
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "CREATE TABLE rows (partition_key TEXT NOT NULL, date TEXT NOT NULL, "
            "symbol TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(date, symbol))"
        )
        connection.execute("CREATE TABLE symbols (symbol TEXT PRIMARY KEY)")
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
                symbol = row["symbol"]
                if raw_date != parsed_date.isoformat():
                    raise ValueError(f"line {line_number}: date is not canonical YYYY-MM-DD")
                for action_column in ("dividends", "stock_splits"):
                    if action_column in row and row[action_column] not in (None, ""):
                        try:
                            action_value = float(row[action_column])
                        except (TypeError, ValueError):
                            raise ValueError(f"line {line_number}: {action_column} is not numeric")
                        if not math.isfinite(action_value) or action_value < 0:
                            raise ValueError(
                                f"line {line_number}: {action_column} must be finite and nonnegative"
                            )
                try:
                    connection.execute(
                        "INSERT INTO rows VALUES (?, ?, ?, ?)",
                        (partition_key(parsed_date), raw_date, symbol,
                         json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False)),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"line {line_number}: duplicate symbol/date key {symbol}/{raw_date}"
                    ) from exc
                connection.execute("INSERT OR IGNORE INTO symbols VALUES (?)", (symbol,))
                row_count += 1
        if row_count == 0:
            raise ValueError("source CSV contains no data rows")

        partition_rows = connection.execute(
            "SELECT partition_key, COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) "
            "FROM rows GROUP BY partition_key ORDER BY partition_key"
        ).fetchall()
        source_info = {
            "row_count": row_count,
            "symbol_count": connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            "min_date": connection.execute("SELECT MIN(date) FROM rows").fetchone()[0],
            "max_date": connection.execute("SELECT MAX(date) FROM rows").fetchone()[0],
        }
        return columns, {
            "connection": connection,
            "partition_rows": partition_rows,
            "source_info": source_info,
        }
    except Exception:
        connection.close()
        raise


def partition_ohlcv(
    input_path: str | Path,
    output_root: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate and plan partitions; write only when apply=True.

    Input is streamed into a disk-backed staging database, so Python memory use
    does not grow with the number of OHLCV rows. Output is sorted by date/symbol,
    written under a unique temporary sibling, and atomically renamed. Existing
    output paths are never overwritten or cleaned up.
    """
    source = Path(input_path).resolve()
    output = Path(output_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or source.is_relative_to(output):
        raise ValueError("output root must not contain or equal the source file")
    if apply and output.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    if apply and not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")

    with tempfile.TemporaryDirectory(prefix="research-ohlcv-stage-") as staging:
        database = Path(staging) / "rows.sqlite3"
        columns, staged = _validated_rows(source, database)
        connection: sqlite3.Connection = staged["connection"]
        source_info = staged["source_info"]
        partitions = [
            {
                "key": key,
                "path": f"partitioned/ohlcv/{key}/part-000.csv",
                "row_count": count,
                "symbol_count": symbol_count,
                "min_date": min_date,
                "max_date": max_date,
            }
            for key, count, symbol_count, min_date, max_date in staged["partition_rows"]
        ]
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
            "partitions": partitions,
            "write_applied": bool(apply),
            "partition_strategy": "DISK_BACKED_SQLITE_STREAMING",
        }

        if not apply:
            connection.close()
            return manifest

        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        try:
            for partition in partitions:
                key = partition["key"]
                destination = temporary / "partitioned" / "ohlcv" / key / "part-000.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                cursor = connection.execute(
                    "SELECT payload FROM rows WHERE partition_key=? ORDER BY date, symbol", (key,)
                )
                with destination.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
                    writer.writeheader()
                    for (payload,) in cursor:
                        writer.writerow(json.loads(payload))
                partition["sha256"] = sha256_file(destination)

            manifest_path = temporary / "manifests" / "partition_manifest.json"
            write_json(manifest_path, manifest)
            connection.close()
            os.replace(temporary, output)
        except Exception:
            connection.close()
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source OHLCV CSV (read-only)")
    parser.add_argument("--output-root", required=True, help="new output directory; must not exist")
    parser.add_argument(
        "--apply", action="store_true",
        help="write partitions; without this flag, perform validation and print a plan only",
    )
    args = parser.parse_args()
    result = partition_ohlcv(args.input, args.output_root, apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
