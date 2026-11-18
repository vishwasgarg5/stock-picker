"""Conservative point-in-time listing-date eligibility policy for research rows."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


UNKNOWN = "LISTING_DATE_UNKNOWN"
PRELISTING = "DATA_PRECEDES_LISTING_DATE"
GAP_REVIEW = "STARTS_AFTER_LISTING_DATE_REVIEW"
VERIFIED = "STARTS_AT_FIRST_POSSIBLE_SESSION"


def load_listing_evidence(audit_path: str | Path) -> dict[str, dict[str, str]]:
    """Load one reviewed listing-evidence row per symbol; reject ambiguous evidence."""
    evidence: dict[str, dict[str, str]] = {}
    with Path(audit_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "listing_history_status", "listing_date"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"coverage audit must contain {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip()
            status = (row.get("listing_history_status") or "").strip()
            listing_date = (row.get("listing_date") or "").strip()
            if not symbol:
                raise ValueError(f"audit line {line_number}: empty symbol")
            if symbol in evidence:
                raise ValueError(f"audit line {line_number}: duplicate symbol {symbol}")
            if status == UNKNOWN:
                if listing_date:
                    raise ValueError(f"{symbol}: unknown listing date must be blank")
            elif status in {PRELISTING, GAP_REVIEW, VERIFIED}:
                if not listing_date:
                    raise ValueError(f"{symbol}: verified listing status has no listing_date")
                try:
                    parsed = date.fromisoformat(listing_date)
                except ValueError as exc:
                    raise ValueError(f"{symbol}: invalid listing_date {listing_date}") from exc
                if parsed.isoformat() != listing_date:
                    raise ValueError(f"{symbol}: listing_date is not canonical")
            else:
                raise ValueError(f"{symbol}: unsupported listing status {status!r}")
            evidence[symbol] = {"status": status, "listing_date": listing_date}
    return evidence


def assess_listing_row(
    symbol: str, session_date: str, evidence: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Return explicit eligibility; unknown and pre-listing rows fail closed."""
    item = evidence.get(symbol)
    if item is None:
        return {"eligible": False, "reason": "NO_LISTING_EVIDENCE"}
    status = item["status"]
    if status == UNKNOWN:
        return {"eligible": False, "reason": "LISTING_DATE_UNVERIFIED"}
    parsed_session = date.fromisoformat(session_date)
    listing_date = date.fromisoformat(item["listing_date"])
    if parsed_session < listing_date:
        return {"eligible": False, "reason": "BEFORE_VERIFIED_LISTING_DATE"}
    if status == GAP_REVIEW:
        return {
            "eligible": True,
            "reason": "ELIGIBLE_LISTING_DATE_GAP_REVIEW",
            "coverage_review_required": True,
        }
    if status == PRELISTING:
        return {
            "eligible": True,
            "reason": "ELIGIBLE_ON_OR_AFTER_LISTING_DATE_PRELISTING_ROWS_HELD_OUT",
            "coverage_review_required": True,
        }
    return {"eligible": True, "reason": "ELIGIBLE_VERIFIED_LISTING_DATE"}


def audit_listing_eligibility(audit_path: str | Path) -> dict[str, Any]:
    """Summarize fail-closed symbol eligibility without modifying source files."""
    evidence = load_listing_evidence(audit_path)
    counts = {
        "verified": sum(item["status"] == VERIFIED for item in evidence.values()),
        "gap_review": sum(item["status"] == GAP_REVIEW for item in evidence.values()),
        "prelisting_anomaly": sum(item["status"] == PRELISTING for item in evidence.values()),
        "unknown": sum(item["status"] == UNKNOWN for item in evidence.values()),
    }
    return {
        "status": "BLOCKED" if counts["unknown"] or counts["prelisting_anomaly"] else (
            "REVIEW_REQUIRED" if counts["gap_review"] else "PASS"
        ),
        "symbol_count": len(evidence),
        "status_counts": counts,
        "symbols_blocked_entirely": sorted(
            symbol for symbol, item in evidence.items() if item["status"] == UNKNOWN
        ),
        "symbols_with_prelisting_rows_to_hold_out": sorted(
            symbol for symbol, item in evidence.items() if item["status"] == PRELISTING
        ),
        "symbols_with_listing_gap_review": sorted(
            symbol for symbol, item in evidence.items() if item["status"] == GAP_REVIEW
        ),
        "policy": {
            "unknown_listing_date": "exclude all rows until independently verified",
            "prelisting_anomaly": "exclude rows before listing_date; review symbol provenance",
            "listing_gap_review": "allow rows on/after verified listing_date but retain coverage warning",
            "source_mutated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_audit", help="reviewed historical coverage audit CSV")
    args = parser.parse_args()
    print(json.dumps(audit_listing_eligibility(args.coverage_audit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
