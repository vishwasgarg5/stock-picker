"""Compare a fresh OHLCV scan with the repository's prior coverage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def reconcile_coverage(scan: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Return explicit differences; never silently replace either evidence source."""
    comparisons = {
        "row_count": {
            "fresh_scan": scan.get("row_count"),
            "prior_audit": audit.get("total_valid_rows"),
        },
        "symbol_count": {
            "fresh_scan": scan.get("symbol_count"),
            "prior_audit": audit.get("universe_symbols"),
        },
        "first_date": {
            "fresh_scan": scan.get("first_date"),
            "prior_audit": audit.get("history_start"),
        },
        "last_date": {
            "fresh_scan": scan.get("last_date"),
            "prior_audit": audit.get("history_end"),
        },
    }
    differences = {
        name: values
        for name, values in comparisons.items()
        if values["fresh_scan"] != values["prior_audit"]
    }
    return {
        "status": "MATCH" if not differences and audit.get("status") == "PASS" else "REVIEW_REQUIRED",
        "fresh_source_sha256": scan.get("source_sha256"),
        "fresh_source_path": scan.get("source_path"),
        "prior_audit_status": audit.get("status"),
        "comparisons": comparisons,
        "differences": differences,
        "prior_listing_history_status_counts": audit.get("listing_history_status_counts", {}),
        "prior_symbols_without_verified_listing_date": audit.get("symbols_without_verified_listing_date", []),
        "prior_total_valid_rows": audit.get("total_valid_rows"),
        "fresh_total_rows": scan.get("row_count"),
        "note": (
            "A mismatch is an evidence gap, not proof that either source is wrong. "
            "Review source version, universe definition, and audit generation time before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    scan = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    print(json.dumps(reconcile_coverage(scan, audit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
