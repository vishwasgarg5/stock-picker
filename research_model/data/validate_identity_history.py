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
    source_documents_by_hash: dict[str, list[dict[str, str]]] | None = None
    if source_documents is not None:
        source_documents_by_hash = defaultdict(list)
        with Path(source_documents).open("r", encoding="utf-8-sig", newline="") as manifest_handle:
            for document in csv.DictReader(manifest_handle):
                digest = (document.get("sha256") or "").strip().lower()
                if digest:
                    source_documents_by_hash[digest].append({
                        "capture_status": (document.get("capture_status") or "").strip().upper(),
                        "source_id": (document.get("source_id") or "").strip(),
                        "venue": (document.get("venue") or "").strip().upper(),
                        "canonical_url": (document.get("canonical_url") or "").strip(),
                        "retrieved_url": (document.get("retrieved_url") or "").strip(),
                    })
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    intervals: dict[tuple[str, str], list[tuple[date, date | None, int]]] = defaultdict(list)
    symbol_intervals: dict[tuple[str, str, str], list[tuple[date, date | None, int, str]]] = defaultdict(list)
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
        elif source_documents_by_hash is not None:
            documents = source_documents_by_hash.get(digest, [])
            if not documents:
                errors.append(f"row {number}: source_document_sha256 is absent from supplied source-document manifest")
            else:
                matching_identity = [
                    item for item in documents
                    if item["source_id"] == source_id and item["venue"] == venue
                ]
                if not matching_identity:
                    errors.append(f"row {number}: source_id and venue do not match the referenced source-document manifest entry")
                elif not any(item["capture_status"] == "INDEPENDENTLY_REVIEWED" for item in matching_identity):
                    errors.append(f"row {number}: referenced source document is not independently reviewed")
                elif not any(
                    item["capture_status"] == "INDEPENDENTLY_REVIEWED"
                    and source_url in {item["canonical_url"], item["retrieved_url"]}
                    for item in matching_identity
                ):
                    errors.append(f"row {number}: source_url does not match the independently reviewed source-document manifest entry")
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

        if start and valid_isin(isin):
            intervals[(venue, isin)].append((start, end, number))
        if start and venue in {"NSE", "BSE"} and symbol and series:
            symbol_intervals[(venue, symbol, series)].append((start, end, number, isin))
        if len(errors) == before:
            valid_rows += 1

    def count_overlaps(entries: list[tuple[date, date | None, int]]) -> list[tuple[int, int]]:
        """Return row pairs whose inclusive validity intervals overlap."""
        ordered = sorted(entries, key=lambda item: item[0])
        overlaps: list[tuple[int, int]] = []
        active: list[tuple[date | None, int]] = []
        for start, end, row_number in ordered:
            active = [
                (active_end, active_row)
                for active_end, active_row in active
                if active_end is None or active_end >= start
            ]
            overlaps.extend((row_number, active_row) for _, active_row in active)
            active.append((end, row_number))
        return overlaps

    overlap_count = 0
    isin_overlap_count = 0
    for (venue, isin), entries in intervals.items():
        for current_row, previous_row in count_overlaps(entries):
            errors.append(
                f"row {current_row}: overlapping validity interval for {venue} ISIN {isin} "
                f"with row {previous_row}"
            )
            overlap_count += 1
            isin_overlap_count += 1

    symbol_overlap_count = 0
    for (venue, symbol, series), entries in symbol_intervals.items():
        # A venue/symbol/series tuple must resolve to at most one ISIN on any
        # date. Different ISINs are never joined; ambiguous concurrent claims
        # are rejected rather than silently choosing one.
        interval_rows = [(start, end, row_number) for start, end, row_number, _ in entries]
        for current_row, previous_row in count_overlaps(interval_rows):
            current_isin = next(item[3] for item in entries if item[2] == current_row)
            previous_isin = next(item[3] for item in entries if item[2] == previous_row)
            if current_isin != previous_isin:
                errors.append(
                    f"row {current_row}: overlapping symbol/series identity for {venue} "
                    f"{symbol}.{series} maps to different ISINs {current_isin} and "
                    f"{previous_isin} (row {previous_row})"
                )
                overlap_count += 1
                symbol_overlap_count += 1

    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "required_columns_missing": missing,
        "structurally_valid_rows": valid_rows,
        "venue_isin_count": len(intervals),
        # Backward-compatible alias; counts are venue/ISIN pairs, not global ISINs.
        "isin_count": len(intervals),
        "overlapping_intervals": overlap_count,
        "overlapping_venue_isin_intervals": isin_overlap_count,
        "overlapping_venue_symbol_series_intervals": symbol_overlap_count,
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
