"""Exact, disk-backed reconciliation of a source CSV and its partition archive."""

import argparse
import csv
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .manifest import sha256_file
from .verify_partitions import verify_partition_archive


def reconcile_source_to_partitions(
    source_path: str | Path, archive_root: str | Path
) -> dict[str, Any]:
    """Prove every source row occurs exactly once and unchanged in the archive.

    CSV field strings are compared exactly. SQLite keeps the source index on disk,
    avoiding an in-memory copy of the full historical dataset.
    """
    source = Path(source_path).resolve()
    root = Path(archive_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    archive_check = verify_partition_archive(root)
    manifest_path = root / "manifests" / "partition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = manifest.get("source", {})
    actual_hash = sha256_file(source)
    if actual_hash != source_manifest.get("sha256"):
        raise ValueError("source SHA-256 does not match partition manifest")

    schema = manifest["schema"]
    source_rows = 0
    partition_rows = 0
    with tempfile.TemporaryDirectory(prefix="research-partition-reconcile-") as temp_dir:
        db_path = Path(temp_dir) / "source.sqlite3"
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "CREATE TABLE source_rows (date TEXT NOT NULL, symbol TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(date, symbol))"
            )
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != schema:
                    raise ValueError("source schema does not match partition manifest")
                for line_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"source line {line_number}: malformed CSV row")
                    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    try:
                        connection.execute(
                            "INSERT INTO source_rows(date, symbol, payload) VALUES (?, ?, ?)",
                            (row["date"], row["symbol"], payload),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"duplicate source symbol/date key {row['symbol']}/{row['date']}"
                        ) from exc
                    source_rows += 1
            connection.commit()

            for entry in manifest["partitions"]:
                path = root / entry["path"]
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames != schema:
                        raise ValueError(f"partition schema mismatch: {entry['key']}")
                    for line_number, row in enumerate(reader, start=2):
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(
                                f"{entry['key']} line {line_number}: malformed CSV row"
                            )
                        key = (row["date"], row["symbol"])
                        found = connection.execute(
                            "SELECT payload FROM source_rows WHERE date = ? AND symbol = ?",
                            key,
                        ).fetchone()
                        if found is None:
                            raise ValueError(
                                f"partition row has no unmatched source row: {key[1]}/{key[0]}"
                            )
                        payload = json.dumps(
                            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                        )
                        if payload != found[0]:
                            raise ValueError(
                                f"row content differs from source: {key[1]}/{key[0]}"
                            )
                        connection.execute(
                            "DELETE FROM source_rows WHERE date = ? AND symbol = ?", key
                        )
                        partition_rows += 1
            connection.commit()
            remaining = connection.execute(
                "SELECT COUNT(*) FROM source_rows"
            ).fetchone()[0]

    if remaining:
        raise ValueError(f"{remaining} source rows are missing from the partition archive")
    if source_rows != partition_rows or source_rows != source_manifest.get("row_count"):
        raise ValueError("source and partition row totals do not reconcile")

    return {
        "status": "PASS",
        "source_sha256": actual_hash,
        "source_rows": source_rows,
        "partition_rows": partition_rows,
        "exact_row_match": True,
        "archive_verification": archive_check["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="original source OHLCV CSV (read-only)")
    parser.add_argument("--archive-root", required=True, help="materialized partition archive")
    args = parser.parse_args()
    print(json.dumps(
        reconcile_source_to_partitions(args.source, args.archive_root),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
