from __future__ import annotations

import csv

from research_model.data.validate_identity_sources import validate_identity_sources

COLUMNS = [
    "source_id", "venue", "source_class", "source_name", "canonical_url",
    "authority_scope", "available_fields", "effective_date_basis",
    "retrieval_method", "review_status", "reviewed_by", "reviewed_at_utc", "notes",
]


def write_registry(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(**overrides):
    row = dict(
        source_id="NSE_CM_MASTER", venue="NSE",
        source_class="EXCHANGE_SECURITY_MASTER", source_name="NSE historical CM security master",
        canonical_url="https://www.nseindia.com/all-reports",
        authority_scope="Exchange-published security master fields only",
        available_fields="symbol; series; ISIN; security code",
        effective_date_basis="Snapshot date; interval boundaries require source-specific confirmation",
        retrieval_method="Manual archived source retrieval with immutable document capture",
        review_status="DISCOVERY_ONLY", reviewed_by="", reviewed_at_utc="", notes="Discovery pointer only",
    )
    row.update(overrides)
    return row


def test_empty_registry_template_is_valid_but_not_approved(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [])
    report = validate_identity_sources(path)
    assert report["row_count"] == 0
    assert report["error_count"] == 0
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["identity_mapping_approval"] == "NOT_GRANTED"


def test_valid_discovery_source_is_structurally_accepted(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row()])
    report = validate_identity_sources(path)
    assert report["structurally_valid_rows"] == 1
    assert report["error_count"] == 0
    assert report["identity_mapping_approval"] == "NOT_GRANTED"


def test_duplicate_source_ids_are_rejected(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row(), valid_row()])
    report = validate_identity_sources(path)
    assert any("duplicate source_id" in error for error in report["errors"])


def test_non_https_url_is_rejected(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row(canonical_url="http://example.com/source")])
    report = validate_identity_sources(path)
    assert any("absolute HTTPS" in error for error in report["errors"])


def test_reviewed_source_requires_reviewer_and_utc_timestamp(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row(review_status="REVIEWED_AUTHORITATIVE")])
    report = validate_identity_sources(path)
    assert any("requires reviewer and UTC review timestamp" in error for error in report["errors"])


def test_reviewed_source_accepts_explicit_utc_review_metadata(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row(
        review_status="REVIEWED_AUTHORITATIVE", reviewed_by="independent-reviewer",
        reviewed_at_utc="2026-11-18T00:00:00Z",
    )])
    report = validate_identity_sources(path)
    assert report["error_count"] == 0
    assert report["identity_mapping_approval"] == "NOT_GRANTED"


def test_registry_does_not_accept_source_as_identity_evidence(tmp_path):
    path = tmp_path / "sources.csv"
    write_registry(path, [valid_row()])
    report = validate_identity_sources(path)
    assert report["registry_policy"].startswith("DISCOVERY_REFERENCE_ONLY")
    assert report["identity_mapping_approval"] == "NOT_GRANTED"
