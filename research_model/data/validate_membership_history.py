"""Validate evidence-linked point-in-time index constituent membership intervals.

The validator checks declared membership intervals and their document provenance.
It never infers membership between snapshots or grants historical-model approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "index", "isin", "symbol", "valid_from", "valid_to", "source_id",
    "source_document_sha256", "evidence_reference", "verification_status",
    "reviewed_by", "reviewed_at_utc",
)
STATUSES = {"UNREVIEWED", "PENDING", "INDEPENDENTLY_VERIFIED", "REJECTED"}
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def valid_isin(value: str) -> bool:
    if not ISIN_RE.fullmatch(value):
        return False
    expanded = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in value)
    total = 0
    double = False
    for char in reversed(expanded):
        number = int(char) * (2 if double else 1)
        total += number // 10 + number % 10
        double = not double
    return total % 10 == 0


def _parse_date(value: str, field: str, row: int, errors: list[str], required: bool = False) -> date | None:
    value = (value or "").strip()
    if not value:
        if required:
            errors.append(f"row {row}: missing {field}")
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        errors.append(f"row {row}: invalid {field}; expected YYYY-MM-DD")
        return None
    if parsed.isoformat() != value:
        errors.append(f"row {row}: non-canonical {field}")
    return parsed


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat((value or "").strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed


def validate_membership_history(
    path: str | Path, source_documents: str | Path | None = None
) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    documents: dict[str, list[dict[str, str]]] | None = None
    if source_documents is not None:
        documents = defaultdict(list)
        with Path(source_documents).open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                digest = (item.get("sha256") or "").strip().lower()
                if digest:
                    documents[digest].append({
                        "source_id": (item.get("source_id") or "").strip(),
                        "capture_status": (item.get("capture_status") or "").strip().upper(),
                    })

    by_index_isin: dict[tuple[str, str], list[tuple[date, date | None, int]]] = defaultdict(list)
    by_index_symbol: dict[tuple[str, str], list[tuple[date, date | None, int, str]]] = defaultdict(list)
    valid_rows = 0
    for number, row in enumerate(rows, start=2):
        before = len(errors)
        index = (row.get("index") or "").strip().upper()
        isin = (row.get("isin") or "").strip().upper()
        symbol = (row.get("symbol") or "").strip().upper()
        source_id = (row.get("source_id") or "").strip()
        digest = (row.get("source_document_sha256") or "").strip().lower()
        status = (row.get("verification_status") or "").strip().upper()

        if not index:
            errors.append(f"row {number}: missing index")
        if not symbol:
            errors.append(f"row {number}: missing symbol")
        if not valid_isin(isin):
            errors.append(f"row {number}: invalid ISIN")
        start = _parse_date(row.get("valid_from", ""), "valid_from", number, errors, required=True)
        end = _parse_date(row.get("valid_to", ""), "valid_to", number, errors)
        if start and end and end < start:
            errors.append(f"row {number}: valid_to precedes valid_from")
        if not source_id:
            errors.append(f"row {number}: missing source_id")
        if not SHA_RE.fullmatch(digest):
            errors.append(f"row {number}: source_document_sha256 must be 64 lowercase hexadecimal characters")
        if not (row.get("evidence_reference") or "").strip():
            errors.append(f"row {number}: missing evidence_reference")
        if status not in STATUSES:
            errors.append(f"row {number}: invalid verification_status")

        reviewed_by = (row.get("reviewed_by") or "").strip()
        reviewed_at = (row.get("reviewed_at_utc") or "").strip()
        if status in {"INDEPENDENTLY_VERIFIED", "REJECTED"}:
            if not reviewed_by or not _parse_utc(reviewed_at):
                errors.append(f"row {number}: {status} requires reviewer and UTC review timestamp")
        elif reviewed_by or reviewed_at:
            errors.append(f"row {number}: reviewer metadata requires INDEPENDENTLY_VERIFIED or REJECTED")

        if documents is not None and SHA_RE.fullmatch(digest):
            matches = [item for item in documents.get(digest, []) if item["source_id"] == source_id]
            if not matches:
                errors.append(f"row {number}: source_id and document hash are absent from supplied source-document manifest")
            elif status == "INDEPENDENTLY_VERIFIED" and not any(
                item["capture_status"] == "INDEPENDENTLY_REVIEWED" for item in matches
            ):
                errors.append(f"row {number}: independently verified membership requires independently reviewed source document")

        if start and index and valid_isin(isin):
            by_index_isin[(index, isin)].append((start, end, number))
            if symbol:
                by_index_symbol[(index, symbol)].append((start, end, number, isin))
        if len(errors) == before:
            valid_rows += 1

    def overlaps(entries: list[tuple[date, date | None, int]]) -> list[tuple[int, int]]:
        active: list[tuple[date | None, int]] = []
        found: list[tuple[int, int]] = []
        for start, end, row_number in sorted(entries, key=lambda item: item[0]):
            active = [(prior_end, prior_row) for prior_end, prior_row in active
                      if prior_end is None or prior_end >= start]
            found.extend((row_number, prior_row) for _, prior_row in active)
            active.append((end, row_number))
        return found

    isin_overlaps = 0
    for (index, isin), entries in by_index_isin.items():
        for current, prior in overlaps(entries):
            errors.append(f"row {current}: overlapping membership interval for {index} ISIN {isin} with row {prior}")
            isin_overlaps += 1

    symbol_overlaps = 0
    for (index, symbol), entries in by_index_symbol.items():
        intervals = [(start, end, row) for start, end, row, _ in entries]
        for current, prior in overlaps(intervals):
            current_isin = next(item[3] for item in entries if item[2] == current)
            prior_isin = next(item[3] for item in entries if item[2] == prior)
            if current_isin != prior_isin:
                errors.append(f"row {current}: overlapping index symbol {index}/{symbol} maps to different ISINs with row {prior}")
                symbol_overlaps += 1

    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "required_columns_missing": missing,
        "structurally_valid_rows": valid_rows,
        "index_isin_count": len(by_index_isin),
        "overlapping_index_isin_intervals": isin_overlaps,
        "overlapping_index_symbol_intervals": symbol_overlaps,
        "source_document_manifest_checked": source_documents is not None,
        "independent_verification": "NOT_PERFORMED_BY_THIS_VALIDATOR",
        "approval": "HUMAN_REVIEW_REQUIRED_EVEN_IF_STRUCTURALLY_VALID",
        "error_count": len(errors),
        "errors": errors[:200],
        "errors_truncated": len(errors) > 200,
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--source-documents", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_membership_history(args.csv_path, args.source_documents)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
