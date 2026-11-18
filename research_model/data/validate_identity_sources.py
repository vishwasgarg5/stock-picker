"""Validate the registry of candidate authoritative identity sources.

Registry entries describe where evidence may be discovered. They are not
security-identity evidence and never approve an ISIN/symbol mapping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REQUIRED_COLUMNS = (
    "source_id", "venue", "source_class", "source_name", "canonical_url",
    "authority_scope", "available_fields", "effective_date_basis",
    "retrieval_method", "review_status", "reviewed_by", "reviewed_at_utc", "notes",
)
VENUES = {"NSE", "BSE", "ISSUER", "OTHER"}
SOURCE_CLASSES = {
    "EXCHANGE_SECURITY_MASTER", "EXCHANGE_CIRCULAR", "LISTING_NOTICE",
    "ISSUER_REGULATORY_FILING", "OTHER_PRIMARY_SOURCE",
}
REVIEW_STATUSES = {"DISCOVERY_ONLY", "REVIEWED_AUTHORITATIVE", "REJECTED"}


def _utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def validate_identity_sources(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    seen: set[str] = set()
    valid_rows = 0
    for line, row in enumerate(rows, start=2):
        before = len(errors)
        source_id = (row.get("source_id") or "").strip()
        if not source_id:
            errors.append(f"row {line}: missing source_id")
        elif source_id in seen:
            errors.append(f"row {line}: duplicate source_id {source_id}")
        seen.add(source_id)

        venue = (row.get("venue") or "").strip().upper()
        if venue not in VENUES:
            errors.append(f"row {line}: invalid venue")
        source_class = (row.get("source_class") or "").strip().upper()
        if source_class not in SOURCE_CLASSES:
            errors.append(f"row {line}: invalid source_class")
        for field in ("source_name", "authority_scope", "available_fields",
                      "effective_date_basis", "retrieval_method"):
            if not (row.get(field) or "").strip():
                errors.append(f"row {line}: missing {field}")

        raw_url = (row.get("canonical_url") or "").strip()
        parsed_url = urlparse(raw_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            errors.append(f"row {line}: canonical_url must be an absolute HTTPS URL")

        status = (row.get("review_status") or "").strip().upper()
        if status not in REVIEW_STATUSES:
            errors.append(f"row {line}: invalid review_status")
        reviewer = (row.get("reviewed_by") or "").strip()
        reviewed_at = (row.get("reviewed_at_utc") or "").strip()
        if status == "REVIEWED_AUTHORITATIVE":
            if not reviewer or not reviewed_at or not _utc_timestamp(reviewed_at):
                errors.append(f"row {line}: reviewed source requires reviewer and UTC review timestamp")
        elif reviewer or reviewed_at:
            errors.append(f"row {line}: reviewer metadata requires REVIEWED_AUTHORITATIVE")

        if len(errors) == before:
            valid_rows += 1

    return {
        "status": "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "required_columns_missing": missing,
        "structurally_valid_rows": valid_rows,
        "unique_source_ids": len(seen),
        "error_count": len(errors),
        "errors": errors[:200],
        "errors_truncated": len(errors) > 200,
        "registry_policy": "DISCOVERY_REFERENCE_ONLY; SOURCE_REVIEW_DOES_NOT_VALIDATE_SECURITY_MAPPINGS",
        "identity_mapping_approval": "NOT_GRANTED",
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_identity_sources(args.csv_path)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
