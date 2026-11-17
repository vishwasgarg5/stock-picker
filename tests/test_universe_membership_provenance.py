from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.universe_membership_provenance import (
    MembershipEvidenceError, validate_membership_records,
)


def record(**overrides):
    item = {
        "schema_version": "1.0.0", "universe": "NIFTY50", "symbol": "AAA",
        "exchange": "NSE", "asset_class": "EQUITY",
        "effective_from": "2024-03-28", "effective_to": "2024-09-26",
        "published_at": "2024-03-15T12:30:00+05:30",
        "source": "https://www.niftyindices.com/Index_Constituent",
        "source_document_sha256": "a" * 64,
    }
    item.update(overrides)
    return item


def test_accepts_effective_dated_membership_with_publication_provenance():
    result = validate_membership_records([record()])
    assert result["status"] == "PASS"
    assert result["record_count"] == 1
    assert result["contract_validated"] is True
    assert result["point_in_time_ready"] is False
    assert result["historical_membership_inferred"] is False


def test_rejects_missing_effective_date():
    item = record()
    del item["effective_from"]
    with pytest.raises(MembershipEvidenceError, match="effective_from"):
        validate_membership_records([item])


def test_rejects_publication_timestamp_without_timezone():
    with pytest.raises(MembershipEvidenceError, match="timezone"):
        validate_membership_records([record(published_at="2024-03-15T12:30:00")])


def test_rejects_effective_end_before_start():
    with pytest.raises(MembershipEvidenceError, match="precedes"):
        validate_membership_records([record(effective_to="2024-03-01")])


def test_rejects_duplicate_membership_start():
    with pytest.raises(MembershipEvidenceError, match="duplicate"):
        validate_membership_records([record(), record()])


def test_rejects_overlapping_membership_intervals():
    first = record(effective_from="2024-01-01", effective_to="2024-06-30")
    second = record(effective_from="2024-06-15", effective_to="2024-12-31")
    with pytest.raises(MembershipEvidenceError, match="overlapping"):
        validate_membership_records([first, second])


def test_rejects_unsupported_exchange_and_unknown_fields():
    with pytest.raises(MembershipEvidenceError, match="unsupported exchange"):
        validate_membership_records([record(exchange="UNKNOWN")])
    with pytest.raises(MembershipEvidenceError, match="unknown"):
        validate_membership_records([record(unverified_weight="10")])


def test_rejects_missing_or_malformed_source_document_hash():
    item = record()
    del item["source_document_sha256"]
    with pytest.raises(MembershipEvidenceError, match="source_document_sha256"):
        validate_membership_records([item])
    with pytest.raises(MembershipEvidenceError, match="64-character SHA-256"):
        validate_membership_records([record(source_document_sha256="not-a-hash")])


def test_rejects_non_https_source_document():
    with pytest.raises(MembershipEvidenceError, match="HTTPS"):
        validate_membership_records([record(source="http://example.com/release.pdf")])
