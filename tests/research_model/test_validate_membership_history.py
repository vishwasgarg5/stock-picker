from __future__ import annotations

import csv
import hashlib

from research_model.data.validate_membership_history import REQUIRED_COLUMNS, validate_membership_history
from research_model.data.validate_source_archive import REQUIRED_COLUMNS as DOCUMENT_COLUMNS


def membership_row(**updates):
    row = {
        "index": "NIFTY50", "isin": "INE009A01021", "symbol": "INFY",
        "valid_from": "2020-01-01", "valid_to": "2022-12-31",
        "source_id": "NSE_INDEX_CIRCULAR", "source_document_sha256": "b" * 64,
        "evidence_reference": "circular page 2, constituent table row 1",
        "verification_status": "PENDING", "reviewed_by": "", "reviewed_at_utc": "",
    }
    row.update(updates)
    return row


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_documents(path, capture_status="INDEPENDENTLY_REVIEWED"):
    row = {
        "document_id": "DOC-1", "source_id": "NSE_INDEX_CIRCULAR", "venue": "NSE",
        "source_class": "EXCHANGE_CIRCULAR", "document_title": "Index change circular",
        "canonical_url": "https://example.com/circular", "retrieved_url": "https://example.com/circular",
        "publication_date": "2020-01-01", "effective_from": "2020-01-01", "effective_to": "",
        "retrieved_at_utc": "2020-01-02T00:00:00Z", "content_type": "application/pdf",
        "byte_length": "100", "sha256": "b" * 64, "archive_path": "nse/index.pdf",
        "evidence_locator": "page 2", "capture_method": "manual",
        "capture_status": capture_status,
        "reviewed_by": "reviewer" if capture_status == "INDEPENDENTLY_REVIEWED" else "",
        "reviewed_at_utc": "2020-01-03T00:00:00Z" if capture_status == "INDEPENDENTLY_REVIEWED" else "",
        "notes": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCUMENT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_empty_schema_is_valid_but_never_approved(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [])
    report = validate_membership_history(path)
    assert report["error_count"] == 0
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["approval"] == "HUMAN_REVIEW_REQUIRED_EVEN_IF_STRUCTURALLY_VALID"


def test_valid_interval_passes_structural_checks(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [membership_row()])
    report = validate_membership_history(path)
    assert report["error_count"] == 0
    assert report["structurally_valid_rows"] == 1


def test_overlapping_membership_intervals_for_same_index_isin_fail(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [
        membership_row(valid_to="2021-06-30"),
        membership_row(valid_from="2021-06-01", valid_to=""),
    ])
    report = validate_membership_history(path)
    assert report["overlapping_index_isin_intervals"] == 1
    assert any("overlapping membership interval" in error for error in report["errors"])


def test_overlapping_index_symbol_cannot_map_to_different_isins(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [
        membership_row(valid_to="2021-06-30"),
        membership_row(isin="INE115A01026", valid_from="2021-06-01", valid_to=""),
    ])
    report = validate_membership_history(path)
    assert report["overlapping_index_symbol_intervals"] == 1


def test_sequential_membership_intervals_are_allowed(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [
        membership_row(valid_to="2021-05-31"),
        membership_row(valid_from="2021-06-01", valid_to=""),
    ])
    report = validate_membership_history(path)
    assert report["error_count"] == 0


def test_independently_verified_interval_requires_reviewed_document(tmp_path):
    path, docs = tmp_path / "membership.csv", tmp_path / "documents.csv"
    write_rows(path, [membership_row(
        verification_status="INDEPENDENTLY_VERIFIED", reviewed_by="reviewer",
        reviewed_at_utc="2020-01-04T00:00:00Z",
    )])
    write_documents(docs, capture_status="CAPTURED_UNREVIEWED")
    report = validate_membership_history(path, docs)
    assert any("requires independently reviewed source document" in error for error in report["errors"])


def test_independently_verified_interval_binds_to_manifest_source_and_hash(tmp_path):
    path, docs = tmp_path / "membership.csv", tmp_path / "documents.csv"
    write_rows(path, [membership_row(
        verification_status="INDEPENDENTLY_VERIFIED", reviewed_by="reviewer",
        reviewed_at_utc="2020-01-04T00:00:00Z",
    )])
    write_documents(docs)
    report = validate_membership_history(path, docs)
    assert report["error_count"] == 0
    assert report["source_document_manifest_checked"]



def test_manifest_cannot_claim_review_without_reviewer_metadata(tmp_path):
    path, docs = tmp_path / "membership.csv", tmp_path / "documents.csv"
    write_rows(path, [membership_row(
        verification_status="INDEPENDENTLY_VERIFIED", reviewed_by="reviewer",
        reviewed_at_utc="2020-01-04T00:00:00Z",
    )])
    write_documents(docs)
    content = docs.read_text(encoding="utf-8")
    content = content.replace(
        "reviewer,2020-01-03T00:00:00Z",
        ",",
    )
    docs.write_text(content, encoding="utf-8")
    report = validate_membership_history(path, docs)
    assert any("INDEPENDENTLY_REVIEWED requires reviewer" in error for error in report["errors"])
    assert any("requires independently reviewed source document" in error for error in report["errors"])


def test_manifest_rejects_duplicate_document_ids(tmp_path):
    path, docs = tmp_path / "membership.csv", tmp_path / "documents.csv"
    write_rows(path, [membership_row()])
    write_documents(docs)
    content = docs.read_text(encoding="utf-8")
    docs.write_text(content + content.splitlines(keepends=True)[1], encoding="utf-8")
    report = validate_membership_history(path, docs)
    assert any("duplicate document_id DOC-1" in error for error in report["errors"])

def test_invalid_isin_and_reversed_dates_fail(tmp_path):
    path = tmp_path / "membership.csv"
    write_rows(path, [membership_row(isin="INE009A01022", valid_from="2022-01-01", valid_to="2021-01-01")])
    report = validate_membership_history(path)
    assert any("invalid ISIN" in error for error in report["errors"])
    assert any("valid_to precedes valid_from" in error for error in report["errors"])
