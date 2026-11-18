"""Tests for the read-only corporate-action provenance audit."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.audit_corporate_actions import audit_corporate_actions


class CorporateActionAuditTests(unittest.TestCase):
    def test_counts_events_and_never_claims_source_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ohlcv.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "date", "symbol", "dividends", "stock_splits", "source"
                ])
                writer.writeheader()
                writer.writerows([
                    {"date": "2024-01-01", "symbol": "AAA", "dividends": "0", "stock_splits": "0", "source": "vendor-x"},
                    {"date": "2024-01-02", "symbol": "AAA", "dividends": "2.5", "stock_splits": "0", "source": "vendor-x"},
                    {"date": "2024-01-03", "symbol": "BBB", "dividends": "0", "stock_splits": "2", "source": "vendor-y"},
                ])
            before = path.read_bytes()
            report = audit_corporate_actions(path)
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["actions"]["dividends"]["event_rows"], 1)
            self.assertEqual(report["actions"]["stock_splits"]["event_rows"], 1)
            self.assertEqual(report["event_level_provenance_status"], "FIELDS_PRESENT_BUT_UNVERIFIED")
            self.assertEqual(report["exchange_independent_verification"], "NOT_PERFORMED")
            self.assertEqual(path.read_bytes(), before)

    def test_missing_provenance_is_explicit_and_invalid_values_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ohlcv.csv"
            path.write_text(
                "date,symbol,dividends,stock_splits\n"
                "2024-01-01,AAA,NaN,-1\n"
                "2024-01-02,BBB,abc,0\n",
                encoding="utf-8",
            )
            report = audit_corporate_actions(path)
            self.assertEqual(report["event_level_provenance_status"], "NO_EVENT_LEVEL_PROVENANCE_FIELDS")
            self.assertEqual(report["actions"]["dividends"]["invalid_values"], 1)
            self.assertEqual(report["actions"]["stock_splits"]["negative_values_held_for_review"], 1)

    def test_missing_action_columns_do_not_get_assumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.csv"
            path.write_text("date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
            report = audit_corporate_actions(path)
            self.assertEqual(report["action_columns_missing"], ["dividends", "stock_splits"])
            self.assertEqual(report["actions"], {})


if __name__ == "__main__":
    unittest.main()
