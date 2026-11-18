"""Tests for raw-vs-reviewed historical universe reconciliation."""

import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.reconcile_universe import reconcile_universe


class ReconcileUniverseTests(unittest.TestCase):
    def test_reports_raw_only_symbols_and_listing_provenance_exceptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ohlcv = root / "ohlcv.csv"
            audit = root / "audit.csv"
            universe = root / "universe.csv"
            with ohlcv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
                writer.writerows([
                    ["2024-01-01", "AAA", 1, 2, 1, 2, 10],
                    ["2024-01-02", "EXTRA", 1, 2, 1, 2, 10],
                ])
            with audit.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "symbol", "listing_history_status", "first_date", "listing_date"
                ])
                writer.writeheader()
                writer.writerows([
                    {"symbol": "AAA", "listing_history_status": "LISTING_DATE_UNKNOWN", "first_date": "2024-01-01"},
                    {"symbol": "BBB", "listing_history_status": "DATA_PRECEDES_LISTING_DATE", "first_date": "2020-01-01", "listing_date": "2021-01-01"},
                ])

            universe.write_text("symbol,company_name\nAAA,Alpha\nBBB,Beta\n", encoding="utf-8")
            result = reconcile_universe(ohlcv, audit, universe)

            self.assertEqual(result["status"], "REVIEW_REQUIRED")
            self.assertEqual(result["raw_only_symbols"], ["EXTRA"])
            self.assertEqual(result["audit_only_symbols"], ["BBB"])
            self.assertEqual(result["current_universe_symbol_count"], 2)
            self.assertEqual(result["raw_symbols_outside_current_universe"], ["EXTRA"])
            self.assertEqual(result["current_universe_missing_ohlcv"], ["BBB"])
            self.assertEqual(result["current_universe_not_in_coverage_audit"], [])
            self.assertEqual(result["symbols_without_verified_listing_date"], ["AAA"])
            self.assertEqual(result["data_precedes_listing_date"], [
                {"symbol": "BBB", "first_date": "2020-01-01", "listing_date": "2021-01-01"}
            ])
            self.assertFalse(result["input_modified"])

    def test_exact_clean_universe_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ohlcv = root / "ohlcv.csv"
            audit = root / "audit.csv"
            universe = root / "universe.csv"
            ohlcv.write_text("date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
            audit.write_text(
                "symbol,listing_history_status,first_date,listing_date\nAAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01\n",
                encoding="utf-8",
            )
            universe.write_text("symbol,company_name\nAAA,Alpha\n", encoding="utf-8")
            self.assertEqual(reconcile_universe(ohlcv, audit, universe)["status"], "MATCH")


    def test_excludes_synthetic_dummy_rows_from_tradable_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ohlcv = root / "ohlcv.csv"
            audit = root / "audit.csv"
            universe = root / "universe.csv"
            ohlcv.write_text("date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
            audit.write_text(
                "symbol,listing_history_status,first_date,listing_date\n"
                "AAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01\n",
                encoding="utf-8",
            )
            universe.write_text(
                "symbol,company_name\nAAA,Alpha\nDUMMYHEG,Synthetic placeholder\n",
                encoding="utf-8",
            )

            result = reconcile_universe(ohlcv, audit, universe)

            self.assertEqual(result["status"], "MATCH")
            self.assertEqual(result["current_universe_symbol_count"], 1)
            self.assertEqual(result["excluded_synthetic_universe_symbols"], ["DUMMYHEG"])
            self.assertEqual(result["current_universe_missing_ohlcv"], [])
            self.assertEqual(result["current_universe_not_in_coverage_audit"], [])
            self.assertFalse(result["input_modified"])


if __name__ == "__main__":
    unittest.main()
