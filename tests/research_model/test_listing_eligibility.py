"""Tests for conservative listing-date eligibility decisions."""

import csv
import tempfile
import unittest
from pathlib import Path

from research_model.data.listing_eligibility import (
    audit_listing_eligibility,
    assess_listing_row,
    load_listing_evidence,
)


class ListingEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "coverage.csv"
        rows = [
            {"symbol": "AAA", "listing_history_status": "LISTING_DATE_UNKNOWN", "listing_date": ""},
            {"symbol": "BBB", "listing_history_status": "DATA_PRECEDES_LISTING_DATE", "listing_date": "2020-01-02"},
            {"symbol": "CCC", "listing_history_status": "STARTS_AFTER_LISTING_DATE_REVIEW", "listing_date": "2021-03-04"},
            {"symbol": "DDD", "listing_history_status": "STARTS_AT_FIRST_POSSIBLE_SESSION", "listing_date": "2022-05-06"},
        ]
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.evidence = load_listing_evidence(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_listing_date_blocks_every_row(self):
        self.assertEqual(
            assess_listing_row("AAA", "2024-01-01", self.evidence),
            {"eligible": False, "reason": "LISTING_DATE_UNVERIFIED"},
        )

    def test_prelisting_row_is_held_out_but_later_row_is_eligible(self):
        before = assess_listing_row("BBB", "2020-01-01", self.evidence)
        after = assess_listing_row("BBB", "2020-01-02", self.evidence)
        self.assertFalse(before["eligible"])
        self.assertEqual(before["reason"], "BEFORE_VERIFIED_LISTING_DATE")
        self.assertTrue(after["eligible"])
        self.assertTrue(after["coverage_review_required"])

    def test_gap_review_does_not_mistake_nontrading_gap_for_bad_rows(self):
        result = assess_listing_row("CCC", "2021-03-04", self.evidence)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reason"], "ELIGIBLE_LISTING_DATE_GAP_REVIEW")
        self.assertTrue(result["coverage_review_required"])

    def test_missing_symbol_evidence_fails_closed(self):
        self.assertEqual(
            assess_listing_row("MISSING", "2024-01-01", self.evidence),
            {"eligible": False, "reason": "NO_LISTING_EVIDENCE"},
        )

    def test_summary_exposes_blockers_and_does_not_change_audit(self):
        original = self.path.read_bytes()
        report = audit_listing_eligibility(self.path)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["status_counts"], {
            "verified": 1, "gap_review": 1, "prelisting_anomaly": 1, "unknown": 1
        })
        self.assertEqual(report["symbols_blocked_entirely"], ["AAA"])
        self.assertEqual(report["symbols_with_prelisting_rows_to_hold_out"], ["BBB"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_duplicate_symbols_are_rejected(self):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("AAA,LISTING_DATE_UNKNOWN,\n")
        with self.assertRaisesRegex(ValueError, "duplicate symbol"):
            load_listing_evidence(self.path)


if __name__ == "__main__":
    unittest.main()
