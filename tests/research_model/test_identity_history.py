"""Populated-data tests for source-manifest-bound identity history."""
import csv
from pathlib import Path

from research_model.data.validate_identity_history import (
    REQUIRED_COLUMNS,
    validate_identity_history,
)

DIGEST = "a" * 64
IDENTITY = {
    "venue": "NSE",
    "source_id": "NSE_DAILY_SECURITY_MASTER",
    "isin": "INE002A01018",
    "symbol": "RELIANCE",
    "series": "EQ",
    "security_code": "500325",
    "valid_from": "2024-02-05",
    "valid_to": "",
    "source_name": "NSE daily security master fixture",
    "source_url": "https://example.test/security-master.csv",
    "source_document_sha256": DIGEST,
    "retrieved_at_utc": "2026-11-18T06:00:00Z",
    "evidence_reference": "fixture row 2",
    "verification_status": "PENDING",
    "reviewed_by": "",
    "reviewed_at_utc": "",
}


def write_csv(path: Path, columns: list[str], row: dict[str, str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    return path


def manifest_row(**overrides: str) -> dict[str, str]:
    row = {
        "sha256": DIGEST,
        "capture_status": "INDEPENDENTLY_REVIEWED",
        "source_id": "NSE_DAILY_SECURITY_MASTER",
        "venue": "NSE",
        "canonical_url": "https://example.test/security-master.csv",
        "retrieved_url": "https://example.test/security-master.csv",
    }
    return {**row, **overrides}


def validate(tmp_path: Path, identity: dict[str, str], manifest: dict[str, str]):
    identity_path = write_csv(tmp_path / "identity.csv", list(REQUIRED_COLUMNS), identity)
    manifest_path = write_csv(tmp_path / "manifest.csv", list(manifest), manifest)
    return validate_identity_history(identity_path, manifest_path)


def test_valid_populated_identity_row_is_linked_to_reviewed_manifest(tmp_path):
    report = validate(tmp_path, IDENTITY, manifest_row())
    assert report["row_count"] == 1
    assert report["structurally_valid_rows"] == 1
    assert report["error_count"] == 0
    assert report["source_document_manifest_checked"] is True


def test_rejects_source_id_or_venue_mismatch(tmp_path):
    report = validate(tmp_path, IDENTITY, manifest_row(source_id="BSE_SECURITY_MASTER"))
    assert any("source_id and venue do not match" in error for error in report["errors"])


def test_rejects_unreviewed_source_document(tmp_path):
    report = validate(tmp_path, IDENTITY, manifest_row(capture_status="DISCOVERY_ONLY"))
    assert any("not independently reviewed" in error for error in report["errors"])


def test_rejects_missing_document_digest_from_manifest(tmp_path):
    report = validate(tmp_path, IDENTITY, manifest_row(sha256="b" * 64))
    assert any("absent from supplied source-document manifest" in error for error in report["errors"])


def test_rejects_source_url_mismatch(tmp_path):
    report = validate(tmp_path, IDENTITY, manifest_row(canonical_url="https://example.test/other.csv",
                                                       retrieved_url="https://example.test/other.csv"))
    assert any("source_url does not match" in error for error in report["errors"])


def test_manifest_fields_cannot_be_combined_across_duplicate_digest_rows(tmp_path):
    identity_path = write_csv(tmp_path / "identity.csv", list(REQUIRED_COLUMNS), IDENTITY)
    manifest_path = tmp_path / "manifest.csv"
    columns = ["sha256", "capture_status", "source_id", "venue", "canonical_url", "retrieved_url"]
    rows = [
        manifest_row(capture_status="DISCOVERY_ONLY"),
        manifest_row(source_id="OTHER_SOURCE",
                     canonical_url="https://example.test/other.csv",
                     retrieved_url="https://example.test/other.csv"),
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    report = validate_identity_history(identity_path, manifest_path)
    assert report["error_count"] > 0
    assert any("not independently reviewed" in error for error in report["errors"])
