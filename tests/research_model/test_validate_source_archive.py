from __future__ import annotations

import csv
import hashlib

from research_model.data.validate_source_archive import REQUIRED_COLUMNS, validate_source_archive


def make_row(**overrides):
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update({
        "document_id": "NSE_CM_2026_11_18",
        "source_id": "NSE_CM_MASTER",
        "venue": "NSE",
        "source_class": "EXCHANGE_SECURITY_MASTER",
        "document_title": "Historical CM security master",
        "canonical_url": "https://example.com/master",
        "retrieved_url": "https://example.com/archive/master.csv",
        "publication_date": "2026-11-17",
        "effective_from": "2026-11-17",
        "effective_to": "2026-11-17",
        "retrieved_at_utc": "2026-11-18T01:02:03Z",
        "content_type": "text/csv",
        "byte_length": "4",
        "sha256": hashlib.sha256(b"test").hexdigest(),
        "archive_path": "nse/2026/master.csv",
        "evidence_locator": "Full document; rows 1-20",
        "capture_method": "Direct HTTPS retrieval; original bytes preserved",
        "capture_status": "CAPTURED_UNREVIEWED",
        "reviewed_by": "",
        "reviewed_at_utc": "",
        "notes": "Synthetic test fixture only",
    })
    row.update(overrides)
    return row


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_empty_template_is_valid_but_never_approved(tmp_path):
    path = tmp_path / "documents.csv"
    write_manifest(path, [])
    report = validate_source_archive(path)
    assert report["document_count"] == 0
    assert report["error_count"] == 0
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["identity_mapping_approval"] == "NOT_GRANTED"
    assert report["source_authority"] == "NOT_ESTABLISHED"


def test_archive_bytes_are_verified_against_size_and_sha256(tmp_path):
    archive = tmp_path / "archive"
    (archive / "nse/2026").mkdir(parents=True)
    (archive / "nse/2026/master.csv").write_bytes(b"test")
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row()])
    report = validate_source_archive(manifest, archive)
    assert report["error_count"] == 0
    assert report["byte_verified_documents"] == 1
    assert report["archive_integrity"] == "CHECKED"
    assert report["identity_mapping_approval"] == "NOT_GRANTED"


def test_tampered_archived_bytes_are_rejected(tmp_path):
    archive = tmp_path / "archive"
    (archive / "nse/2026").mkdir(parents=True)
    (archive / "nse/2026/master.csv").write_bytes(b"evil")
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row()])
    report = validate_source_archive(manifest, archive)
    assert any("SHA-256 mismatch" in error for error in report["errors"])


def test_missing_archived_document_is_rejected(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row()])
    report = validate_source_archive(manifest, archive)
    assert any("document is missing" in error for error in report["errors"])


def test_unsafe_archive_path_is_rejected(tmp_path):
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row(archive_path="../outside.csv")])
    report = validate_source_archive(manifest)
    assert any("safe relative path" in error for error in report["errors"])


def test_publication_and_effective_dates_are_separate_from_retrieval_time(tmp_path):
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row(
        publication_date="2026-11-18", effective_from="2026-11-19", effective_to="2026-11-19",
        retrieved_at_utc="2026-11-18T01:02:03Z",
    )])
    report = validate_source_archive(manifest)
    assert report["error_count"] == 0


def test_invalid_effective_interval_is_rejected(tmp_path):
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row(effective_from="2026-11-20", effective_to="2026-11-19")])
    report = validate_source_archive(manifest)
    assert any("effective_to precedes" in error for error in report["errors"])


def test_independent_review_requires_reviewer_metadata(tmp_path):
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row(capture_status="INDEPENDENTLY_REVIEWED")])
    report = validate_source_archive(manifest)
    assert any("requires reviewer and UTC review timestamp" in error for error in report["errors"])


def test_duplicate_document_ids_are_rejected(tmp_path):
    manifest = tmp_path / "documents.csv"
    write_manifest(manifest, [make_row(), make_row()])
    report = validate_source_archive(manifest)
    assert any("duplicate document_id" in error for error in report["errors"])


def test_document_source_id_must_exist_in_supplied_registry(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(source_id="UNKNOWN_SOURCE")])
    registry.write_text(
        "source_id,venue,source_class,source_name,canonical_url,authority_scope,available_fields,effective_date_basis,retrieval_method,review_status,reviewed_by,reviewed_at_utc,notes\n",
        encoding="utf-8",
    )
    report = validate_source_archive(manifest, source_registry=registry)
    assert any("not present in the supplied source registry" in error for error in report["errors"])



def write_registry(path, **overrides):
    row = {
        "source_id": "NSE_CM_MASTER",
        "venue": "NSE",
        "source_class": "EXCHANGE_SECURITY_MASTER",
        "canonical_url": "https://example.com/master",
        "review_status": "DISCOVERY_ONLY",
        "reviewed_by": "",
        "reviewed_at_utc": "",
    }
    row.update(overrides)
    path.write_text(
        "source_id,venue,source_class,canonical_url,review_status,reviewed_by,reviewed_at_utc\n"
        + ",".join(row[field] for field in (
            "source_id", "venue", "source_class", "canonical_url",
            "review_status", "reviewed_by", "reviewed_at_utc"
        )) + "\n",
        encoding="utf-8",
    )


def test_document_metadata_must_match_registered_source(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row()])
    write_registry(
        registry, review_status="REVIEWED_AUTHORITATIVE",
        reviewed_by="independent-reviewer", reviewed_at_utc="2026-11-18T02:00:00Z",
    )
    report = validate_source_archive(manifest, source_registry=registry)
    assert report["error_count"] == 0
    assert report["registered_source_count"] == 1


def test_document_venue_must_match_registered_source(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(venue="BSE")])
    write_registry(registry)
    report = validate_source_archive(manifest, source_registry=registry)
    assert any("venue does not match source registry" in error for error in report["errors"])


def test_document_source_class_must_match_registered_source(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(source_class="EXCHANGE_CIRCULAR")])
    write_registry(registry)
    report = validate_source_archive(manifest, source_registry=registry)
    assert any("source_class does not match source registry" in error for error in report["errors"])


def test_document_canonical_url_must_match_registered_source(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(canonical_url="https://example.com/other")])
    write_registry(registry)
    report = validate_source_archive(manifest, source_registry=registry)
    assert any("canonical_url does not match source registry" in error for error in report["errors"])


def test_duplicate_registry_source_ids_are_rejected(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row()])
    registry.write_text(
        "source_id,venue,source_class,canonical_url\n"
        "NSE_CM_MASTER,NSE,EXCHANGE_SECURITY_MASTER,https://example.com/master\n"
        "NSE_CM_MASTER,NSE,EXCHANGE_SECURITY_MASTER,https://example.com/master\n",
        encoding="utf-8",
    )
    report = validate_source_archive(manifest, source_registry=registry)
    assert any("duplicate source_id" in error for error in report["errors"])



def test_independently_reviewed_document_requires_authoritative_registry_entry(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(
        capture_status="INDEPENDENTLY_REVIEWED",
        reviewed_by="document-reviewer",
        reviewed_at_utc="2026-11-18T03:00:00Z",
    )])
    write_registry(registry)
    report = validate_source_archive(manifest, source_registry=registry)
    assert any(
        "requires REVIEWED_AUTHORITATIVE source registry entry" in error
        for error in report["errors"]
    )


def test_authoritative_registry_entry_requires_reviewer_metadata(tmp_path):
    manifest = tmp_path / "documents.csv"
    registry = tmp_path / "registry.csv"
    write_manifest(manifest, [make_row(
        capture_status="INDEPENDENTLY_REVIEWED",
        reviewed_by="document-reviewer",
        reviewed_at_utc="2026-11-18T03:00:00Z",
    )])
    write_registry(registry, review_status="REVIEWED_AUTHORITATIVE")
    report = validate_source_archive(manifest, source_registry=registry)
    assert any(
        "requires reviewer and UTC review timestamp" in error
        for error in report["errors"]
    )
