"""Validate provenance metadata and optional byte integrity of archived source documents.

A verified archive proves only that captured bytes match their manifest. It does
not establish source authority, document completeness, or mapping correctness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

REQUIRED_COLUMNS = (
    "document_id", "source_id", "venue", "source_class", "document_title",
    "canonical_url", "retrieved_url", "publication_date", "effective_from",
    "effective_to", "retrieved_at_utc", "content_type", "byte_length",
    "sha256", "archive_path", "evidence_locator", "capture_method",
    "capture_status", "reviewed_by", "reviewed_at_utc", "notes",
)
VENUES = {"NSE", "BSE", "ISSUER", "OTHER"}
SOURCE_CLASSES = {
    "EXCHANGE_SECURITY_MASTER", "EXCHANGE_CIRCULAR", "LISTING_NOTICE",
    "ISSUER_REGULATORY_FILING", "OTHER_PRIMARY_SOURCE",
}
CAPTURE_STATUSES = {"CAPTURED_UNREVIEWED", "INDEPENDENTLY_REVIEWED", "REJECTED"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def _utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _safe_archive_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def validate_source_archive(csv_path: str | Path, archive_root: str | Path | None = None, source_registry: str | Path | None = None) -> dict[str, Any]:
    manifest = Path(csv_path)
    registered_source_ids: set[str] | None = None
    if source_registry is not None:
        with Path(source_registry).open("r", encoding="utf-8-sig", newline="") as registry_handle:
            registered_source_ids = {((item.get("source_id") or "").strip()) for item in csv.DictReader(registry_handle)}
    root = Path(archive_root).resolve() if archive_root is not None else None
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    errors: list[str] = []
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))
    seen: set[str] = set()
    byte_verified = 0

    for line, row in enumerate(rows, start=2):
        before = len(errors)
        doc_id = (row.get("document_id") or "").strip()
        if not doc_id:
            errors.append(f"row {line}: missing document_id")
        elif doc_id in seen:
            errors.append(f"row {line}: duplicate document_id {doc_id}")
        seen.add(doc_id)

        source_id = (row.get("source_id") or "").strip()
        if not source_id:
            errors.append(f"row {line}: missing source_id (must reference source registry)")
        elif registered_source_ids is not None and source_id not in registered_source_ids:
            errors.append(f"row {line}: source_id is not present in the supplied source registry")
        if (row.get("venue") or "").strip().upper() not in VENUES:
            errors.append(f"row {line}: invalid venue")
        if (row.get("source_class") or "").strip().upper() not in SOURCE_CLASSES:
            errors.append(f"row {line}: invalid source_class")
        for field in ("document_title", "evidence_locator", "capture_method", "content_type"):
            if not (row.get(field) or "").strip():
                errors.append(f"row {line}: missing {field}")

        for field in ("canonical_url", "retrieved_url"):
            parsed = urlparse((row.get(field) or "").strip())
            if parsed.scheme != "https" or not parsed.netloc:
                errors.append(f"row {line}: {field} must be an absolute HTTPS URL")

        retrieved_at = (row.get("retrieved_at_utc") or "").strip()
        if not _utc(retrieved_at):
            errors.append(f"row {line}: retrieved_at_utc must be timezone-aware UTC")
        for field in ("publication_date", "effective_from", "effective_to"):
            value = (row.get(field) or "").strip()
            if value and not _date(value):
                errors.append(f"row {line}: {field} must use YYYY-MM-DD")
        start, end = (row.get("effective_from") or "").strip(), (row.get("effective_to") or "").strip()
        if start and end and _date(start) and _date(end) and end < start:
            errors.append(f"row {line}: effective_to precedes effective_from")

        try:
            size = int((row.get("byte_length") or "").strip())
            if size <= 0:
                raise ValueError
        except ValueError:
            size = -1
            errors.append(f"row {line}: byte_length must be a positive integer")
        digest = (row.get("sha256") or "").strip().lower()
        if not _SHA256.fullmatch(digest):
            errors.append(f"row {line}: sha256 must be 64 lowercase hexadecimal characters")

        relative = (row.get("archive_path") or "").strip()
        if not _safe_archive_path(relative):
            errors.append(f"row {line}: archive_path must be a safe relative path")
        elif root is not None:
            target = (root / relative).resolve()
            if root not in target.parents:
                errors.append(f"row {line}: archive_path escapes archive root")
            elif not target.is_file():
                errors.append(f"row {line}: archived document is missing")
            else:
                payload = target.read_bytes()
                if len(payload) != size:
                    errors.append(f"row {line}: archived byte length mismatch")
                if hashlib.sha256(payload).hexdigest() != digest:
                    errors.append(f"row {line}: archived SHA-256 mismatch")
                if len(errors) == before:
                    byte_verified += 1

        status = (row.get("capture_status") or "").strip().upper()
        if status not in CAPTURE_STATUSES:
            errors.append(f"row {line}: invalid capture_status")
        reviewer, reviewed_at = (row.get("reviewed_by") or "").strip(), (row.get("reviewed_at_utc") or "").strip()
        if status == "INDEPENDENTLY_REVIEWED":
            if not reviewer or not _utc(reviewed_at):
                errors.append(f"row {line}: independently reviewed document requires reviewer and UTC review timestamp")
        elif reviewer or reviewed_at:
            errors.append(f"row {line}: reviewer metadata requires INDEPENDENTLY_REVIEWED")

    return {
        "status": "REVIEW_REQUIRED", "manifest_path": manifest.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "archive_root_checked": str(root) if root is not None else None,
        "source_registry_checked": str(source_registry) if source_registry is not None else None,
        "document_count": len(rows), "required_columns_missing": missing,
        "unique_document_ids": len(seen), "byte_verified_documents": byte_verified,
        "error_count": len(errors), "errors": errors[:200],
        "errors_truncated": len(errors) > 200,
        "archive_integrity": "CHECKED" if root is not None else "NOT_CHECKED",
        "source_authority": "NOT_ESTABLISHED", "identity_mapping_approval": "NOT_GRANTED",
        "input_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--source-registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_source_archive(args.csv_path, args.archive_root, args.source_registry)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
