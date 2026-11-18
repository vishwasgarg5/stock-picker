"""Validate an evidence-linked, point-in-time security identifier history.

This validator checks declared identifier intervals only. It does not infer
renames, join different ISINs, or establish source authenticity.
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
    "venue", "source_id", "isin", "symbol", "series", "security_code", "valid_from", "valid_to",
    "source_name", "source_url", "source_document_sha256", "retrieved_at_utc",
    "evidence_reference", "verification_status", "reviewed_by", "reviewed_at_utc",
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
    for ch in reversed(expanded):
        n = int(ch) * (2 if double else 1)
        total += n // 10 + n % 10
        double = not double
    return total % 10 == 0


def parse_date(value: str, field: str, row: int, errors: list[str], required: bool = False) -> date | None:
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


def parse_utc(value: str, field: str, row: int, errors: list[str]) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"row {row}: invalid {field}; expected ISO-8601 UTC timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"row {row}: {field} must be UTC")
    return parsed


def validate_identity_history(path: str | Path, source_documents: str | Path | None = None) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    errors: list[str] = []
    reviewed_document_hashes: dict[str, set[str]] | None = None
    source_document_urls: dict[str, set[str]] = {}
    source_document_metadata: dict[str, list[dict[str, str]]] = defaultdict(list)
    if source_documents is not None:
        reviewed_document_hashes = defaultdict(set)
        with Path(source_documents).open("r", encoding="utf-8-sig", newline="") as manifest_handle:
            for document in csv.DictReader(manifest_handle):
                digest = (document.get("sha256") or "").strip().lower()
                if digest:
                    reviewed_document_hashes[digest].add((document.get("capture_status") or "").strip().upper())
                    source_document_urls.setdefault(digest, set()).update({
                        (document.get("canonical_url") or "").strip(),
                        (document.get("retrieved_url") or "").strip(),
                    })
                    source_document_metadata[digest].append({
                        "source_id": (document.get("source_id") or "").strip(),
                        "venue": (document.get("venue") or "").strip().upper(),
                    })
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    intervals: dict[tuple[str, str], list[tuple[date, date | None, int]]] = defaultdict(list)
    valid_rows = 0
    for number, row in enumerate(rows, start=2):
        before = len(errors)
        venue = (row.get("venue") or "").strip().upper()
        source_id = (row.get("source_id") or "").strip()
        if venue not in {"NSE", "BSE"}:
            errors.append(f"row {number}: venue must be NSE or BSE")
        if not source_id:
            errors.append(f"row {number}: missing source_id")
        isin = (row.get("isin") or "").strip().upper()
        symbol = (row.get("symbol") or "").strip().upper()
        series = (row.get("series") or "").strip().upper()
        if not valid_isin(isin):
            errors.append(f"row {number}: invalid ISO 6166 ISIN/checksum")
        if not symbol:
            errors.append(f"row {number}: missing historical symbol")
        if not series:
            errors.append(f"row {number}: missing historical series")
        start = parse_date(row.get("valid_from", ""), "valid_from", number, errors, True)
        end = parse_date(row.get("valid_to", ""), "valid_to", number, errors)
        if start and end and end < start:
            errors.append(f"row {number}: valid_to precedes valid_from")

        if not (row.get("source_name") or "").strip():
            errors.append(f"row {number}: missing source_name")
        source_url = (row.get("source_url") or "").strip()
        if not source_url.startswith("https://"):
            errors.append(f"row {number}: source_url must use HTTPS")
        digest = (row.get("source_document_sha256") or "").strip().lower()
        if not SHA_RE.fullmatch(digest):
            errors.append(f"row {number}: source_document_sha256 must be 64 hex characters")
        elif reviewed_document_hashes is not None:
            document_statuses = reviewed_document_hashes.get(digest)
            if not document_statuses:
                errors.append(f"row {number}: source_document_sha256 is absent from supplied source-document manifest")
            elif "INDEPENDENTLY_REVIEWED" not in document_statuses:
                errors.append(f"row {number}: referenced source document is not independently reviewed")
            if digest in source_document_metadata and not any(
                item["source_id"] == source_id and item["venue"] == venue
                for item in source_document_metadata[digest]
            ):
                errors.append(f"row {number}: source_id and venue do not match the referenced source-document manifest entry")
        parse_utc(row.get("retrieved_at_utc", ""), "retrieved_at_utc", number, errors)
        if not (row.get("retrieved_at_utc") or "").strip():
            errors.append(f"row {number}: missing retrieved_at_utc")
        if not (row.get("evidence_reference") or "").strip():
            errors.append(f"row {number}: missing evidence_reference")

        status = (row.get("verification_status") or "").strip().upper()
        if status not in STATUSES:
            errors.append(f"row {number}: invalid verification_status")
        reviewed_by = (row.get("reviewed_by") or "").strip()
        reviewed_at_raw = (row.get("reviewed_at_utc") or "").strip()
        reviewed_at = parse_utc(reviewed_at_raw, "reviewed_at_utc", number, errors) if reviewed_at_raw else None
        if status == "INDEPENDENTLY_VERIFIED" and (not reviewed_by or not reviewed_at):
            errors.append(f"row {number}: verified identity requires reviewer and UTC review time")
        if status != "INDEPENDENTLY_VERIFIED" and (reviewed_by or reviewed_at_raw):
            errors.append(f"row {number}: reviewer metadata requires INDEPENDENTLY_VERIFIED")

        if reviewed_document_hashes is not None and SHA_RE.fullmatch(digest):
            if digest in reviewed_document_hashes and source_url not in source_document_urls.get(digest, set()):
                errors.append(f"row {number}: source_url does not match the referenced source-document manifest entry")

        if start and valid_isin(isin):
            intervals[(venue, isin)].append((start, end, number))
        if len(errors) == before:
            valid_rows += 1

    overlap_count = 0
    for (venue, isin), entries in intervals.items():
        entries.sort(key=lambda item: item[0])
        previous_end: date | None = None
        previous_row: int | None = None
        for start, end, row_number in entries:
            if previous_row is not None and (previous_end is None or start <= previous_end):
                errors.append(
                    f"row {row_number}: overlapping validity interval for {venue} ISIN {isin} "
                    f"with row {previous_row}"
                )
                overlap_count += 1
            if previous_row is None or previous_end is not None and (end is None or end > previous_end):
                previous_end = end
                previous_row = row_number

    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "required_columns_missing": missing,
        "structurally_valid_rows": valid_rows,
        "isin_count": len(intervals),
        "overlapping_intervals": overlap_count,
        "identity_policy": "VENUE_SCOPED_ISIN_INTERVALS; DIFFERENT_VENUES_REMAIN_DISTINCT; DIFFERENT_ISINS_NEVER_AUTO_LINKED",
        "source_document_manifest": Path(source_documents).as_posix() if source_documents is not None else None,
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
    parser.add_argument("--source-documents", type=Path, help="Optional captured-document manifest to bind identity evidence to reviewed source artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_identity_history(args.csv_path, args.source_documents)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
