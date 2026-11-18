import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_model.data.validate_raw_symbol_review import validate_raw_symbol_review


FIELDS = [
    "symbol", "raw_row_count", "first_observed_date", "last_observed_date",
    "in_current_universe", "historical_disposition", "source_document_id",
    "source_document_sha256", "reviewed_by", "reviewed_at_utc", "review_notes",
]
DIGEST = "a" * 64


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
            writer.writerow(["document_id", "capture_status", "sha256", "archive_path"])
            writer.writerow(["DOC-1", "INDEPENDENTLY_REVIEWED", DIGEST, "research_model/archive/DOC-1.html"])
        return diagnostic, ledger, documents

    def _write_ledger(self, path, disposition="PENDING", document_id="", digest="",
                      reviewer="", reviewed_at="", notes=""):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\\n")
            writer.writeheader()
            writer.writerow({
                "symbol": "AAA", "raw_row_count": "2",
                "first_observed_date": "2024-01-01", "last_observed_date": "2024-01-02",
                "in_current_universe": "false", "historical_disposition": disposition,
                "source_document_id": document_id, "source_document_sha256": digest,
                "reviewed_by": reviewer, "reviewed_at_utc": reviewed_at, "review_notes": notes,
            })

    def test_blank_template_is_pending_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            self._write_ledger(ledger)
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["pending_symbols"], ["AAA"])

    def test_evidence_backed_exclusion_passes_structural_review(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            self._write_ledger(ledger, "EXCLUDE_FROM_RESEARCH_WITH_EVIDENCE",
                               "DOC-1", DIGEST, "reviewer", "2026-11-18T08:00:00Z",
                               "Exchange archive confirms scope")
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEWED")

    def test_evidence_backed_include_passes_structural_review(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            self._write_ledger(ledger, "INCLUDE_IN_HISTORICAL_UNIVERSE",
                               "DOC-1", DIGEST, "reviewer", "2026-11-18T08:00:00Z",
                               "Exchange archive confirms historical listing")
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEWED")

    def test_document_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            self._write_ledger(ledger, "INCLUDE_IN_HISTORICAL_UNIVERSE",
                               "DOC-1", "b" * 64, "reviewer", "2026-11-18T08:00:00Z",
                               "Digest mismatch test")
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertTrue(any("must match" in error for error in report["errors"]))

    def test_unreviewed_source_document_cannot_support_disposition(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            with documents.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["document_id", "capture_status", "sha256", "archive_path"])
                writer.writerow(["DOC-1", "CAPTURED", DIGEST, "archive/doc.html"])
            self._write_ledger(ledger, "EXCLUDE_FROM_RESEARCH_WITH_EVIDENCE",
                               "DOC-1", DIGEST, "reviewer", "2026-11-18T08:00:00Z", "Evidence")
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertTrue(any("independently reviewed" in error for error in report["errors"]))

    def test_stale_counts_and_missing_document_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic, ledger, documents = self._fixtures(Path(directory))
            with ledger.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\\n")
                writer.writeheader()
                writer.writerow({
                    "symbol": "AAA", "raw_row_count": "3",
                    "first_observed_date": "2024-01-01", "last_observed_date": "2024-01-02",
                    "in_current_universe": "false", "historical_disposition": "INCLUDE_IN_HISTORICAL_UNIVERSE",
                    "source_document_id": "MISSING", "source_document_sha256": "",
                    "reviewed_by": "reviewer", "reviewed_at_utc": "2026-11-18T08:00:00Z",
                    "review_notes": "Evidence",
                })
            report = validate_raw_symbol_review(diagnostic, ledger, documents)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertTrue(any("raw_row_count" in error for error in report["errors"]))
            self.assertTrue(any("source_document_id" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
