import csv
import hashlib
import json

import pytest

from research_model.data.verify_training_eligibility import verify_training_eligibility


FIELDS = [
    "source_row_number", "date", "symbol", "isin", "venue", "source_id",
    "identity_source_document_sha256", "ohlcv_source_document_sha256",
    "identity_valid_from", "identity_valid_to", "corporate_action_coverage_sha256",
    "decision", "reason_codes", "source_ohlcv_sha256",
]


def fixture(tmp_path):
    source = tmp_path / "ohlcv.csv"
    source.write_text("date,symbol,open,high,low,close,volume\n2024-01-02,AAA,1,2,1,2,10\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "eligibility.csv"
    row = {
        "source_row_number": "2", "date": "2024-01-02", "symbol": "AAA",
        "isin": "", "venue": "", "source_id": "",
        "identity_source_document_sha256": "", "ohlcv_source_document_sha256": "",
        "identity_valid_from": "", "identity_valid_to": "",
        "corporate_action_coverage_sha256": "", "decision": "QUARANTINED",
        "reason_codes": "IDENTITY_MISSING", "source_ohlcv_sha256": digest,
    }
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "source_sha256": digest,
        "eligibility_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_row_count": 1, "eligible_row_count": 0, "quarantined_row_count": 1,
    }), encoding="utf-8")
    return source, manifest, summary


def test_valid_quarantine_manifest_passes(tmp_path):
    source, manifest, summary = fixture(tmp_path)
    result = verify_training_eligibility(source, manifest, summary)
    assert result["status"] == "PASS"
    assert result["quarantined_row_count"] == 1


def test_manifest_byte_tampering_is_detected(tmp_path):
    source, manifest, summary = fixture(tmp_path)
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    result = verify_training_eligibility(source, manifest, summary)
    assert result["status"] == "FAIL"
    assert any("manifest_sha256" in error for error in result["errors"])


def test_eligible_row_without_evidence_references_is_rejected(tmp_path):
    source, manifest, summary = fixture(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    rows[0]["decision"] = "ELIGIBLE"
    rows[0]["reason_codes"] = ""
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary.write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "eligibility_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_row_count": 1, "eligible_row_count": 1, "quarantined_row_count": 0,
    }), encoding="utf-8")
    result = verify_training_eligibility(source, manifest, summary)
    assert result["status"] == "FAIL"
    assert any("missing required evidence references" in error for error in result["errors"])


def test_summary_count_tampering_is_detected(tmp_path):
    source, manifest, summary = fixture(tmp_path)
    payload = json.loads(summary.read_text())
    payload["quarantined_row_count"] = 0
    summary.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_training_eligibility(source, manifest, summary)
    assert result["status"] == "FAIL"
    assert any("quarantined_row_count" in error for error in result["errors"])



def test_manifest_observation_must_match_source_symbol_and_date(tmp_path):
    source, manifest, summary = fixture(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    rows[0]["symbol"] = "WRONG"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = json.loads(summary.read_text())
    payload["eligibility_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_training_eligibility(source, manifest, summary)
    assert result["status"] == "FAIL"
    assert any("symbol does not match source" in error for error in result["errors"])
