"""Tests for historical index-membership evidence inspection."""

import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.inspect_membership import inspect_membership_file


class InspectMembershipTests(unittest.TestCase):
    def test_single_date_is_never_misclassified_as_historical_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "index", "symbol", "as_of_date", "weight_pct", "weight_method", "ffmc", "source"
                ])
                writer.writeheader()
                writer.writerows([
                    {"index": "NIFTY50", "symbol": "AAA", "as_of_date": "2026-11-17", "weight_method": "ESTIMATED_FFMC_PROXY", "source": "official"},
                    {"index": "NIFTY50", "symbol": "BBB", "as_of_date": "2026-11-17", "weight_method": "ESTIMATED_FFMC_PROXY", "source": "official"},
                ])
            before = path.read_bytes()
            report = inspect_membership_file(path)
            self.assertEqual(report["status"], "SNAPSHOT_ONLY_NOT_HISTORICAL_MEMBERSHIP")
            self.assertEqual(report["snapshot_count"], 1)
            self.assertEqual(report["unique_symbol_count"], 2)
            self.assertFalse(report["continuous_membership_intervals_present"])
            self.assertEqual(path.read_bytes(), before)

    def test_multiple_snapshots_still_require_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_text(
                "index,symbol,as_of_date,weight_method,source\n"
                "NIFTY50,AAA,2025-06-01,OFFICIAL,exchange\n"
                "NIFTY50,AAA,2026-06-01,OFFICIAL,exchange\n",
                encoding="utf-8",
            )
            report = inspect_membership_file(path)
            self.assertEqual(report["status"], "MULTI_SNAPSHOT_REQUIRES_MEMBERSHIP_INTERVALS")
            self.assertFalse(report["continuous_membership_intervals_present"])



    def test_invalid_dates_and_incomplete_rows_are_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_text(
                "index,symbol,as_of_date,weight_method,source\n"
                "NIFTY50,AAA,2026-11-17T00:00:00Z,OFFICIAL,exchange\n"
                "NIFTY50,,2026-11-17,OFFICIAL,exchange\n",
                encoding="utf-8",
            )
            report = inspect_membership_file(path)
            self.assertEqual(report["status"], "BLOCKED_INVALID_MEMBERSHIP_ROWS")
            self.assertEqual(report["invalid_row_count"], 2)
            self.assertEqual(report["snapshot_count"], 0)


if __name__ == "__main__":
    unittest.main()
