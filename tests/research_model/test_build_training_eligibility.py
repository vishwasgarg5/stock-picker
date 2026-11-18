import csv
import hashlib
from pathlib import Path

from research_model.data.build_training_eligibility import build_training_eligibility


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_missing_evidence_quarantines_every_row_and_preserves_source(tmp_path):
    source = tmp_path / "ohlcv.csv"
    source.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,AAA,10,11,9,10,100\n"
        "2024-01-03,RAW,10,11,9,10,100\n",
        encoding="utf-8",
    )
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    listing = tmp_path / "listing.csv"
    write_csv(listing, ["symbol", "listing_history_status", "listing_date"], [
        {"symbol": "AAA", "listing_history_status": "STARTS_AT_FIRST_POSSIBLE_SESSION", "listing_date": "2024-01-02"},
        {"symbol": "RAW", "listing_history_status": "STARTS_AT_FIRST_POSSIBLE_SESSION", "listing_date": "2024-01-02"},
    ])
    identity = tmp_path / "identity.csv"
    write_csv(identity, ["symbol", "isin", "series", "valid_from", "valid_to", "verification_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"], [])
    source_coverage = tmp_path / "source.csv"
    write_csv(source_coverage, ["symbol", "venue", "source_id", "valid_from", "valid_to", "source_document_sha256", "coverage_status", "reviewed_by", "reviewed_at_utc"], [])
    actions = tmp_path / "actions.csv"
    write_csv(actions, ["isin", "valid_from", "valid_to", "coverage_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"], [])
    docs = tmp_path / "docs.csv"
    write_csv(docs, ["sha256", "capture_status", "archive_path"], [])
    raw = tmp_path / "raw.csv"
    write_csv(raw, ["symbol", "historical_disposition"], [
        {"symbol": "RAW", "historical_disposition": "PENDING"},
    ])
    output = tmp_path / "eligibility.csv"
    summary = build_training_eligibility(source, listing, identity, source_coverage, actions, docs, raw, output)
    assert summary["source_row_count"] == 2
    assert summary["eligible_row_count"] == 0
    assert summary["quarantined_row_count"] == 2
    assert "IDENTITY_INTERVAL_MISSING_OR_AMBIGUOUS" in summary["reason_counts"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["decision"] == "QUARANTINED" for row in rows)


def test_verified_evidence_can_approve_single_row(tmp_path):
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol,open,high,low,close,volume\n2024-01-02,AAA,10,11,9,10,100\n", encoding="utf-8")
    listing = tmp_path / "listing.csv"
    write_csv(listing, ["symbol", "listing_history_status", "listing_date"], [
        {"symbol": "AAA", "listing_history_status": "STARTS_AT_FIRST_POSSIBLE_SESSION", "listing_date": "2024-01-02"},
    ])
    digest = "a" * 64
    docs = tmp_path / "docs.csv"
    write_csv(docs, ["sha256", "capture_status", "archive_path"], [
        {"sha256": digest, "capture_status": "INDEPENDENTLY_REVIEWED", "archive_path": "nse/doc.pdf"},
    ])
    common = {"source_document_sha256": digest, "reviewed_by": "reviewer", "reviewed_at_utc": "2024-01-03T00:00:00Z"}
    identity = tmp_path / "identity.csv"
    write_csv(identity, ["symbol", "isin", "series", "valid_from", "valid_to", "verification_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"], [
        {"symbol": "AAA", "isin": "INE000A01000", "series": "EQ", "valid_from": "2024-01-01", "valid_to": "", "verification_status": "INDEPENDENTLY_VERIFIED", **common},
    ])
    source_coverage = tmp_path / "source.csv"
    write_csv(source_coverage, ["symbol", "venue", "source_id", "valid_from", "valid_to", "source_document_sha256", "coverage_status", "reviewed_by", "reviewed_at_utc"], [
        {"symbol": "AAA", "venue": "NSE", "source_id": "NSE_EOD", "valid_from": "2024-01-01", "valid_to": "", "coverage_status": "INDEPENDENTLY_VERIFIED", **common},
    ])
    actions = tmp_path / "actions.csv"
    write_csv(actions, ["isin", "valid_from", "valid_to", "coverage_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"], [
        {"isin": "INE000A01000", "valid_from": "2024-01-01", "valid_to": "", "coverage_status": "INDEPENDENTLY_VERIFIED", **common},
    ])
    raw = tmp_path / "raw.csv"
    write_csv(raw, ["symbol", "historical_disposition"], [])
    output = tmp_path / "eligibility.csv"
    result = build_training_eligibility(source, listing, identity, source_coverage, actions, docs, raw, output)
    assert result["eligible_row_count"] == 1
    assert result["quarantined_row_count"] == 0
    assert len(result["eligibility_manifest_sha256"]) == 64


def test_gap_review_and_prelisting_anomaly_rows_remain_quarantined(tmp_path):
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol,open,high,low,close,volume\n2024-01-02,GAP,10,11,9,10,100\n2024-01-02,PRE,10,11,9,10,100\n", encoding="utf-8")
    listing = tmp_path / "listing.csv"
    write_csv(listing, ["symbol", "listing_history_status", "listing_date"], [
        {"symbol": "GAP", "listing_history_status": "STARTS_AFTER_LISTING_DATE_REVIEW", "listing_date": "2024-01-01"},
        {"symbol": "PRE", "listing_history_status": "DATA_PRECEDES_LISTING_DATE", "listing_date": "2024-01-01"},
    ])
    for name, fields in [
        ("identity.csv", ["symbol", "isin", "series", "valid_from", "valid_to", "verification_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"]),
        ("source.csv", ["symbol", "venue", "source_id", "valid_from", "valid_to", "source_document_sha256", "coverage_status", "reviewed_by", "reviewed_at_utc"]),
        ("actions.csv", ["isin", "valid_from", "valid_to", "coverage_status", "source_document_sha256", "reviewed_by", "reviewed_at_utc"]),
        ("docs.csv", ["sha256", "capture_status", "archive_path"]),
        ("raw.csv", ["symbol", "historical_disposition"]),
    ]:
        write_csv(tmp_path / name, fields, [])
    result = build_training_eligibility(source, listing, tmp_path/"identity.csv", tmp_path/"source.csv", tmp_path/"actions.csv", tmp_path/"docs.csv", tmp_path/"raw.csv", tmp_path/"out.csv")
    assert result["eligible_row_count"] == 0
    assert result["quarantined_row_count"] == 2
    assert "LISTING_GAP_REVIEW_REQUIRED" in result["reason_counts"]
    assert "PRELISTING_ANOMALY_SYMBOL_REQUIRES_REVIEW" in result["reason_counts"]
