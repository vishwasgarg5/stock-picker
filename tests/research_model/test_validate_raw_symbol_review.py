import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_model.data.validate_raw_symbol_review import validate_raw_symbol_review


class RawSymbolReviewTests(unittest.TestCase):
    def _fixtures(self, root):
        diagnostic = root / "diagnostic.json"
        ledger = root / "ledger.csv"
        documents = root / "documents.csv"
        diagnostic.write_text(json.dumps({"raw_only_symbols": [
            {"symbol": "AAA", "row_count": 2, "first_date": "2024-01-01",
             "last_date": "2024-01-02", "in_current_universe": False}
        ]}), encoding="utf-8")
        with documents.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["document_id"])
            writer.writerow(["DOC-1"])
        return diagnostic, ledger, documents

    def test_blank_template_is_pending_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            with ledger.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["symbol", "raw_row_count", "first_observed_date",
                    "last_observed_date", "in_current_universe", "historical_disposition",
                    "source_document_id", "reviewed_by", "reviewed_at_utc", "review_notes"])
                writer.writerow(["AAA", "2", "2024-01-01", "2024-01-02", "false",
                    "PENDING", "", "", "", ""])
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["pending_symbols"], ["AAA"])

    def test_evidence_backed_disposition_passes_structural_review(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            with ledger.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["symbol", "raw_row_count", "first_observed_date",
                    "last_observed_date", "in_current_universe", "historical_disposition",
                    "source_document_id", "reviewed_by", "reviewed_at_utc", "review_notes"])
                writer.writerow(["AAA", "2", "2024-01-01", "2024-01-02", "false",
                    "EXCLUDE_FROM_RESEARCH_WITH_EVIDENCE", "DOC-1", "reviewer",
                    "2026-11-18T08:00:00Z", "Exchange archive confirms scope"])
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEWED")

    def test_stale_counts_and_missing_document_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            with ledger.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["symbol", "raw_row_count", "first_observed_date",
                    "last_observed_date", "in_current_universe", "historical_disposition",
                    "source_document_id", "reviewed_by", "reviewed_at_utc", "review_notes"])
                writer.writerow(["AAA", "3", "2024-01-01", "2024-01-02", "false",
                    "INCLUDE_IN_HISTORICAL_UNIVERSE", "MISSING", "reviewer",
                    "2026-11-18T08:00:00Z", "Evidence"])
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertTrue(any("raw_row_count" in error for error in report["errors"]))
            self.assertTrue(any("source_document_id" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
