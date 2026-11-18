"""Tests for point-in-time security identity history validation."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.validate_identity_history import REQUIRED_COLUMNS, validate_identity_history
from research_model.data.validate_source_archive import REQUIRED_COLUMNS as DOCUMENT_COLUMNS


def identity_row(**updates):
    row = {
        "venue": "NSE", "source_id": "NSE_DAILY_MII_SECURITY_FILE",
        "isin": "INE009A01021", "symbol": "INFY", "series": "EQ",
        "security_code": "500209", "valid_from": "2020-01-01", "valid_to": "2022-12-31",
        "source_name": "Official exchange master", "source_url": "https://example.invalid/master",
        "source_document_sha256": "b" * 64, "retrieved_at_utc": "2023-01-01T00:00:00Z",
        "evidence_reference": "master-row-1", "verification_status": "PENDING",
        "reviewed_by": "", "reviewed_at_utc": "",
    }
    row.update(updates)
    return row


class IdentityHistoryTests(unittest.TestCase):
    def write_rows(self, directory, rows):
        path = Path(directory) / "identity.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_document_manifest(self, directory, capture_status="INDEPENDENTLY_REVIEWED", digest="b" * 64):
        path = Path(directory) / "documents.csv"
        row = {
            "document_id": "DOC-001", "source_id": "NSE_DAILY_MII_SECURITY_FILE",
            "venue": "NSE", "source_class": "EXCHANGE_SECURITY_MASTER",
            "document_title": "Dated NSE security master",
            "canonical_url": "https://example.invalid/master",
            "retrieved_url": "https://example.invalid/master",
            "publication_date": "2023-01-01", "effective_from": "2023-01-01",
            "effective_to": "", "retrieved_at_utc": "2023-01-02T00:00:00Z",
            "content_type": "text/csv", "byte_length": "10", "sha256": digest,
            "archive_path": "nse/master.csv", "evidence_locator": "row 1",
            "capture_method": "manual", "capture_status": capture_status,
            "reviewed_by": "reviewer" if capture_status == "INDEPENDENTLY_REVIEWED" else "",
            "reviewed_at_utc": "2023-01-03T00:00:00Z" if capture_status == "INDEPENDENTLY_REVIEWED" else "",
            "notes": "",
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DOCUMENT_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_identity_must_reference_manifested_independently_reviewed_document(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.write_rows(directory, [identity_row()])
            manifest = self.write_document_manifest(directory)
            report = validate_identity_history(identity, manifest)
            self.assertEqual(report["error_count"], 0)
            self.assertTrue(report["source_document_manifest_checked"])

    def test_unreviewed_manifest_document_cannot_support_identity_row(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.write_rows(directory, [identity_row()])
            manifest = self.write_document_manifest(directory, capture_status="CAPTURED_UNREVIEWED")
            report = validate_identity_history(identity, manifest)
            self.assertTrue(any("not independently reviewed" in error for error in report["errors"]))

    def test_identity_document_hash_must_exist_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.write_rows(directory, [identity_row(source_document_sha256="c" * 64)])
            manifest = self.write_document_manifest(directory)
            report = validate_identity_history(identity, manifest)
            self.assertTrue(any("absent from supplied source-document manifest" in error for error in report["errors"]))

    def test_empty_schema_is_review_required_without_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, []))
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["row_count"], 0)
            self.assertEqual(report["error_count"], 0)
            self.assertEqual(report["independent_verification"], "NOT_PERFORMED_BY_THIS_VALIDATOR")

    def test_sequential_symbol_intervals_for_same_isin_are_allowed(self):
        first = identity_row()
        second = identity_row(
            symbol="INFY2", valid_from="2023-01-01", valid_to="",
            evidence_reference="master-row-2",
        )
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, [first, second]))
            self.assertEqual(report["overlapping_intervals"], 0)
            self.assertEqual(report["structurally_valid_rows"], 2)

    def test_overlapping_intervals_for_same_isin_are_flagged(self):
        first = identity_row(valid_to="2023-06-30")
        second = identity_row(symbol="OLDINFY", valid_from="2023-06-01", valid_to="")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, [first, second]))
            self.assertEqual(report["overlapping_intervals"], 1)
            self.assertTrue(any("overlapping validity interval" in error for error in report["errors"]))

    def test_bad_isin_and_reversed_dates_are_rejected(self):
        row = identity_row(isin="INE009A01022", valid_from="2024-01-01", valid_to="2023-01-01")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, [row]))
            self.assertGreaterEqual(report["error_count"], 2)
            self.assertEqual(report["structurally_valid_rows"], 0)

    def test_missing_evidence_is_rejected(self):
        row = identity_row(source_url="", source_document_sha256="bad", evidence_reference="")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, [row]))
            self.assertEqual(report["structurally_valid_rows"], 0)
            self.assertGreaterEqual(report["error_count"], 3)



    def test_same_isin_can_have_independent_nse_and_bse_intervals(self):
        nse = identity_row()
        bse = identity_row(
            venue="BSE", source_id="BSE_STANDARD_SECURITY_MASTER_FORMAT",
            symbol="500209", security_code="500209",
        )
        with tempfile.TemporaryDirectory() as directory:
            report = validate_identity_history(self.write_rows(directory, [nse, bse]))
            self.assertEqual(report["overlapping_intervals"], 0)
            self.assertEqual(report["structurally_valid_rows"], 2)

    def test_identity_venue_and_source_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.write_rows(directory, [identity_row(venue="BSE")])
            manifest = self.write_document_manifest(directory)
            report = validate_identity_history(identity, manifest)
            self.assertTrue(any(
                "source_id and venue do not match" in error
                for error in report["errors"]
            ))


if __name__ == "__main__":
    unittest.main()
