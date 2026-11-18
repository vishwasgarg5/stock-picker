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


    def test_reports_per_symbol_row_count_drift_without_mutating_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ohlcv, audit, universe = root / "ohlcv.csv", root / "audit.csv", root / "universe.csv"
            ohlcv.write_text(
                "date,symbol\n2024-01-01,AAA\n2024-01-02,AAA\n2024-01-03,BBB\n",
                encoding="utf-8",
            )
            audit.write_text(
                "symbol,listing_history_status,first_date,listing_date,observed_sessions\n"
                "AAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01,1\n"
                "BBB,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-03,2024-01-03,1\n",
                encoding="utf-8",
            )
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")

            result = reconcile_universe(ohlcv, audit, universe)

            self.assertEqual(result["status"], "REVIEW_REQUIRED")
            self.assertEqual(result["coverage_audit_observed_row_count"], 2)
            self.assertEqual(result["raw_ohlcv_row_count"], 3)
            self.assertEqual(result["common_symbol_row_count_mismatch_count"], 1)
            self.assertEqual(result["common_symbol_row_count_mismatches"], [
                {"symbol": "AAA", "raw_rows": 2, "audit_rows": 1, "difference": 1}
            ])
            self.assertEqual(result["common_symbol_row_count_difference_total"], 1)
            self.assertFalse(result["input_modified"])



    def test_rejects_fractional_or_negative_observed_session_counts(self):
        for bad_count in ("1.5", "-1"):
            with self.subTest(observed_sessions=bad_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                ohlcv, audit, universe = root / "ohlcv.csv", root / "audit.csv", root / "universe.csv"
                ohlcv.write_text("date,symbol\\n2024-01-01,AAA\\n", encoding="utf-8")
                audit.write_text(
                    "symbol,listing_history_status,first_date,listing_date,observed_sessions\\n"
                    f"AAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01,{bad_count}\\n",
                    encoding="utf-8",
                )
                universe.write_text("symbol\\nAAA\\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid observed_sessions"):
                    reconcile_universe(ohlcv, audit, universe)

    def test_rejects_duplicate_symbols_in_coverage_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ohlcv, audit, universe = root / "ohlcv.csv", root / "audit.csv", root / "universe.csv"
            ohlcv.write_text("date,symbol\\n2024-01-01,AAA\\n", encoding="utf-8")
            audit.write_text(
                "symbol,listing_history_status,first_date,listing_date,observed_sessions\\n"
                "AAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01,1\\n"
                "AAA,STARTS_AT_FIRST_POSSIBLE_SESSION,2024-01-01,2024-01-01,2\\n",
                encoding="utf-8",
            )
            universe.write_text("symbol\\nAAA\\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate symbol"):
                reconcile_universe(ohlcv, audit, universe)



if __name__ == "__main__":
    unittest.main()
