"""Tests for the read-only historical OHLCV source inspector."""

import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.inspect_ohlcv import inspect_ohlcv_file


class InspectOhlcvTests(unittest.TestCase):
    def test_reports_date_quarter_and_duplicates_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            rows = [
                ["2024-03-28", "AAA", "10", "12", "9", "11", "100"],
                ["2024-04-01", "AAA", "11", "13", "10", "12", "120"],
                ["2024-04-01", "AAA", "11", "13", "10", "12", "120"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
                writer.writerows(rows)

            before = path.read_bytes()
            result = inspect_ohlcv_file(path)

            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["symbol_count"], 1)
            self.assertEqual(result["first_date"], "2024-03-28")
            self.assertEqual(result["last_date"], "2024-04-01")
            self.assertEqual(result["quarter_row_counts"], {"2024-Q1": 1, "2024-Q2": 2})
            self.assertEqual(result["duplicate_symbol_dates"], 1)
            self.assertFalse(result["input_modified"])
            self.assertEqual(path.read_bytes(), before)

    def test_missing_required_columns_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("date,symbol,close\n2024-01-01,AAA,10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                inspect_ohlcv_file(path)



    def test_rejects_date_prefixes_and_counts_missing_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            path.write_text(
                "date,symbol,open,high,low,close,volume\n"
                "2024-01-01T00:00:00Z,AAA,10,11,9,10,100\n"
                "2024-01-02,,10,11,9,10,100\n",
                encoding="utf-8",
            )
            report = inspect_ohlcv_file(path)
            self.assertEqual(report["invalid_dates"], 1)
            self.assertEqual(report["invalid_symbols"], 1)
            self.assertEqual(report["symbol_count"], 0)


if __name__ == "__main__":
    unittest.main()
