"""Tests for historical OHLCV and coverage-audit reconciliation."""

import unittest

from research_model.data.reconcile_coverage import reconcile_coverage


class ReconcileCoverageTests(unittest.TestCase):
    def test_mismatch_is_review_required_and_preserves_both_counts(self):
        scan = {
            "source_path": "data/ohlcv.csv",
            "source_sha256": "abc",
            "row_count": 958475,
            "symbol_count": 528,
            "first_date": "2018-01-01",
            "last_date": "2026-11-17",
        }
        audit = {
            "status": "REVIEW_REQUIRED",
            "total_valid_rows": 915478,
            "universe_symbols": 500,
            "history_start": "2018-01-01",
            "history_end": "2026-11-17",
            "listing_history_status_counts": {"DATA_PRECEDES_LISTING_DATE": 6},
            "symbols_without_verified_listing_date": ["BAGMANE", "BIRET", "EMBASSY"],
        }

        result = reconcile_coverage(scan, audit)

        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["differences"]["row_count"]["fresh_scan"], 958475)
        self.assertEqual(result["differences"]["row_count"]["prior_audit"], 915478)
        self.assertEqual(result["differences"]["symbol_count"]["fresh_scan"], 528)
        self.assertEqual(result["differences"]["symbol_count"]["prior_audit"], 500)
        self.assertEqual(result["prior_listing_history_status_counts"]["DATA_PRECEDES_LISTING_DATE"], 6)

    def test_exact_match_requires_prior_audit_pass(self):
        scan = {
            "row_count": 10,
            "symbol_count": 2,
            "first_date": "2024-01-01",
            "last_date": "2024-12-31",
        }
        audit = {
            "status": "PASS",
            "total_valid_rows": 10,
            "universe_symbols": 2,
            "history_start": "2024-01-01",
            "history_end": "2024-12-31",
        }
        self.assertEqual(reconcile_coverage(scan, audit)["status"], "MATCH")


if __name__ == "__main__":
    unittest.main()
