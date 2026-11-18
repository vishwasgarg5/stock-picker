"""Tests for the evidence-linked corporate-action ledger contract."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.validate_action_ledger import REQUIRED_COLUMNS, validate_event_ledger


def valid_dividend():
    return {
        "isin": "INE009A01021", "symbol": "INFY", "series": "EQ", "security_code": "500209", "action_type": "DIVIDEND", "ex_date": "2024-01-10",
        "announcement_date": "2023-12-20", "record_date": "2024-01-11",
        "payment_date": "2024-01-25", "cash_amount": "2.5", "currency": "INR",
        "split_numerator": "", "split_denominator": "", "source_name": "Exchange filing",
        "source_url": "https://example.invalid/filing", "source_document_sha256": "a" * 64,
        "retrieved_at_utc": "2024-01-01T10:00:00Z", "evidence_reference": "filing-id-123",
        "verification_status": "UNREVIEWED", "reviewed_by": "", "reviewed_at_utc": "",
    }


class ActionLedgerTests(unittest.TestCase):
    def _write(self, directory, rows):
        path = Path(directory) / "actions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_structurally_complete_is_still_not_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [valid_dividend()])
            before = path.read_bytes()
            report = validate_event_ledger(path)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["structurally_valid_rows"], 1)
            self.assertEqual(report["provenance_complete_rows"], 1)
            self.assertEqual(report["approval"], "HUMAN_REVIEW_REQUIRED_EVEN_IF_STRUCTURALLY_VALID")
            self.assertEqual(report["independent_verification"], "NOT_PERFORMED_BY_THIS_VALIDATOR")
            self.assertEqual(path.read_bytes(), before)

    def test_split_requires_explicit_positive_integer_ratio(self):
        row = valid_dividend()
        row.update({
            "action_type": "STOCK_SPLIT", "cash_amount": "", "currency": "",
            "split_numerator": "2", "split_denominator": "1",
        })
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [row]))
            self.assertEqual(report["action_type_counts"], {"STOCK_SPLIT": 1})
            self.assertEqual(report["structurally_valid_rows"], 1)

    def test_duplicate_symbol_type_ex_date_is_rejected(self):
        first = valid_dividend()
        second = dict(first)
        second["cash_amount"] = "3"
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [first, second]))
            self.assertEqual(report["structurally_valid_rows"], 1)
            self.assertTrue(any("duplicate ISIN/action_type/ex_date" in error for error in report["errors"]))

    def test_missing_evidence_and_bad_dates_are_rejected(self):
        row = valid_dividend()
        row.update({"source_url": "", "source_document_sha256": "bad", "ex_date": "10-01-2024"})
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [row]))
            self.assertEqual(report["structurally_valid_rows"], 0)
            self.assertGreaterEqual(report["error_count"], 3)

    def test_verified_claim_requires_reviewer_metadata_but_still_not_auto_approved(self):
        row = valid_dividend()
        row.update({
            "verification_status": "INDEPENDENTLY_VERIFIED",
            "reviewed_by": "reviewer-id", "reviewed_at_utc": "2024-01-02T00:00:00Z",
        })
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [row]))
            self.assertEqual(report["structurally_valid_rows"], 1)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")


    def test_invalid_isin_checksum_is_rejected(self):
        row = valid_dividend()
        row["isin"] = "INE009A01022"
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [row]))
            self.assertEqual(report["structurally_valid_rows"], 0)
            self.assertTrue(any("valid ISO 6166 ISIN with checksum" in error for error in report["errors"]))

    def test_historical_symbol_and_series_are_required(self):
        row = valid_dividend()
        row["symbol"] = ""
        row["series"] = ""
        with tempfile.TemporaryDirectory() as directory:
            report = validate_event_ledger(self._write(directory, [row]))
            self.assertEqual(report["structurally_valid_rows"], 0)
            self.assertTrue(any("missing historical symbol" in error for error in report["errors"]))
            self.assertTrue(any("missing historical series" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
