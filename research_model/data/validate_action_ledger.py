"""Validate a canonical, evidence-linked corporate-action event ledger.

This is a structural/provenance-completeness gate, not an exchange verification
service. It never promotes an event to verified based only on self-asserted CSV
fields and never changes OHLCV or corporate-action source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "isin", "symbol", "series", "security_code", "action_type", "ex_date",
    "announcement_date", "record_date", "payment_date", "cash_amount",
    "currency", "split_numerator", "split_denominator", "source_name",
    "source_url", "source_document_sha256", "retrieved_at_utc",
    "evidence_reference", "verification_status", "reviewed_by", "reviewed_at_utc",
)
ACTION_TYPES = {"DIVIDEND", "STOCK_SPLIT"}
VERIFICATION_STATUSES = {"UNREVIEWED", "PENDING", "INDEPENDENTLY_VERIFIED", "REJECTED"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def _parse_date(value: str, field: str, row_number: int, errors: list[str], required: bool = False) -> date | None:
    value = (value or "").strip()
    if not value:
        if required:
            errors.append(f"row {row_number}: missing {field}")
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        errors.append(f"row {row_number}: invalid {field} (expected YYYY-MM-DD)")
        return None
    if parsed.isoformat() != value:
        errors.append(f"row {row_number}: non-canonical {field}")
    return parsed


def _parse_utc(value: str, field: str, row_number: int, errors: list[str], required: bool = False) -> datetime | None:
    value = (value or "").strip()
    if not value:
        if required:
            errors.append(f"row {row_number}: missing {field}")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"row {row_number}: invalid {field} (expected ISO-8601 timestamp)")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"row {row_number}: {field} must include UTC timezone")
    return parsed


def _valid_isin(value: str) -> bool:
    if not _ISIN.fullmatch(value):
        return False
    digits = ""
    for char in value:
        digits += str(ord(char) - 55) if char.isalpha() else char
    total = 0
    double = False
    for char in reversed(digits):
        number = int(char) * (2 if double else 1)
        total += number // 10 + number % 10
        double = not double
    return total % 10 == 0


def validate_event_ledger(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
    errors: list[str] = []
    if missing_columns:
        errors.append("missing required columns: " + ", ".join(missing_columns))

    seen: set[tuple[str, str, str]] = set()
    type_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    provenance_complete = 0
    structurally_valid = 0

    for row_number, row in enumerate(rows, start=2):
        row_errors_before = len(errors)
        isin = (row.get("isin") or "").strip().upper()
        symbol = (row.get("symbol") or "").strip().upper()
        series = (row.get("series") or "").strip().upper()
        action_type = (row.get("action_type") or "").strip().upper()
        if not _valid_isin(isin):
            errors.append(f"row {row_number}: isin must be a valid ISO 6166 ISIN with checksum")
        if not symbol:
            errors.append(f"row {row_number}: missing historical symbol")
        if not series:
            errors.append(f"row {row_number}: missing historical series")
        if action_type not in ACTION_TYPES:
            errors.append(f"row {row_number}: action_type must be DIVIDEND or STOCK_SPLIT")
        else:
            type_counts[action_type] += 1

        ex_date = _parse_date(row.get("ex_date", ""), "ex_date", row_number, errors, True)
        _parse_date(row.get("announcement_date", ""), "announcement_date", row_number, errors)
        _parse_date(row.get("record_date", ""), "record_date", row_number, errors)
        _parse_date(row.get("payment_date", ""), "payment_date", row_number, errors)

        if action_type == "DIVIDEND":
            try:
                amount = float((row.get("cash_amount") or "").strip())
                if not math.isfinite(amount) or amount <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"row {row_number}: DIVIDEND requires finite positive cash_amount")
            if not (row.get("currency") or "").strip():
                errors.append(f"row {row_number}: DIVIDEND requires currency")
            if (row.get("split_numerator") or "").strip() or (row.get("split_denominator") or "").strip():
                errors.append(f"row {row_number}: DIVIDEND must not carry split ratio fields")
        elif action_type == "STOCK_SPLIT":
            try:
                numerator = int((row.get("split_numerator") or "").strip())
                denominator = int((row.get("split_denominator") or "").strip())
                if numerator <= 0 or denominator <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"row {row_number}: STOCK_SPLIT requires positive integer split_numerator and split_denominator")
            if (row.get("cash_amount") or "").strip() or (row.get("currency") or "").strip():
                errors.append(f"row {row_number}: STOCK_SPLIT must not carry cash amount or currency")

        source_name = (row.get("source_name") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        digest = (row.get("source_document_sha256") or "").strip().lower()
        retrieved = _parse_utc(row.get("retrieved_at_utc", ""), "retrieved_at_utc", row_number, errors, True)
        evidence = (row.get("evidence_reference") or "").strip()
        if not source_name:
            errors.append(f"row {row_number}: missing source_name")
        if not source_url.startswith(("https://", "http://")):
            errors.append(f"row {row_number}: source_url must be an http(s) URL")
        if not _SHA256.fullmatch(digest):
            errors.append(f"row {row_number}: source_document_sha256 must be 64 lowercase/uppercase hex characters")
        if not evidence:
            errors.append(f"row {row_number}: missing evidence_reference")

        verification = (row.get("verification_status") or "").strip().upper()
        if verification not in VERIFICATION_STATUSES:
            errors.append(f"row {row_number}: invalid verification_status")
        else:
            verification_counts[verification] += 1
        reviewed_by = (row.get("reviewed_by") or "").strip()
        reviewed_at = _parse_utc(row.get("reviewed_at_utc", ""), "reviewed_at_utc", row_number, errors)
        if verification == "INDEPENDENTLY_VERIFIED" and (not reviewed_by or reviewed_at is None):
            errors.append(f"row {row_number}: INDEPENDENTLY_VERIFIED requires reviewed_by and UTC reviewed_at_utc")
        if verification != "INDEPENDENTLY_VERIFIED" and (reviewed_by or (row.get("reviewed_at_utc") or "").strip()):
            errors.append(f"row {row_number}: reviewer metadata requires INDEPENDENTLY_VERIFIED status")

        if isin and action_type in ACTION_TYPES and ex_date:
            key = (isin, action_type, ex_date.isoformat())
            if key in seen:
                errors.append(f"row {row_number}: duplicate ISIN/action_type/ex_date {key}")
            seen.add(key)

        if len(errors) == row_errors_before:
            structurally_valid += 1
            if source_name and source_url.startswith(("https://", "http://")) and _SHA256.fullmatch(digest) and retrieved and evidence:
                provenance_complete += 1

    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "required_columns_missing": missing_columns,
        "action_type_counts": dict(type_counts),
        "verification_status_counts": dict(verification_counts),
        "structurally_valid_rows": structurally_valid,
        "provenance_complete_rows": provenance_complete,
        "identity_key": "ISIN/action_type/ex_date",
        "identity_fields": ["isin", "symbol", "series", "security_code"],
        "error_count": len(errors),
        "errors": errors[:200],
        "errors_truncated": len(errors) > 200,
        "independent_verification": "NOT_PERFORMED_BY_THIS_VALIDATOR",
        "approval": "HUMAN_REVIEW_REQUIRED_EVEN_IF_STRUCTURALLY_VALID",
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_event_ledger(args.csv_path)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
