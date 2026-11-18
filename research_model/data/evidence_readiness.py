"""Single fail-closed gate for point-in-time historical evidence readiness.

This gate is deliberately stricter than unit-test success. It only returns
READY_FOR_RESEARCH when source bytes, source authority, event/identity/membership
evidence, listing eligibility, and reconciliation reports all meet their
acceptance criteria. It does not authorize production or live trading.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_FILES = {
    "ohlcv": "ohlcv-source-manifest.json",
    "corporate_action_audit": "corporate-action-audit.json",
    "corporate_action_validation": "corporate-action-ledger-validation.json",
    "identity_history_validation": "security-identity-history-validation.json",
    "identity_source_validation": "identity-source-registry-validation.json",
    "source_document_validation": "historical-source-document-validation.json",
    "universe_reconciliation": "historical-universe-reconciliation.json",
    "raw_symbol_review": "raw-symbol-review-validation.json",
    "listing_eligibility": "listing-eligibility-audit.json",
    "membership_snapshot": "membership-evidence-audit.json",
    "membership_history_validation": "index-membership-history-validation.json",
    "coverage_reconciliation": "coverage-reconciliation.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence report must contain a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"manifest has missing or duplicate headers: {path}")
        return list(reader)


def _utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _common_symbol_coverage_matches(coverage: dict[str, Any], universe: dict[str, Any]) -> tuple[bool, str]:
    """Accept exact shared-symbol coverage while separately blocking raw-only symbols."""
    if coverage.get("status") == "MATCH":
        return True, "fresh OHLCV scan matches prior coverage audit"
    differences = coverage.get("differences") or {}
    if set(differences) != {"row_count", "symbol_count"}:
        return False, f"status={coverage.get('status')}; differences={differences}"
    try:
        rows, symbols = differences["row_count"], differences["symbol_count"]
        row_delta = int(rows["fresh_scan"]) - int(rows["prior_audit"])
        symbol_delta = int(symbols["fresh_scan"]) - int(symbols["prior_audit"])
        raw_only_rows = int(universe["raw_only_row_count"])
        raw_only_symbols = universe.get("raw_only_symbols") or []
        raw_only_count = len(raw_only_symbols)
        common_mismatches = int(universe["common_symbol_row_count_mismatch_count"])
        symbol_count_delta = (
            int(universe["raw_ohlcv_symbol_count"])
            - int(universe["coverage_audit_symbol_count"])
        )
    except (KeyError, TypeError, ValueError):
        return False, f"coverage/universe reconciliation fields are incomplete; differences={differences}"
    audit_only = universe.get("audit_only_symbols") or []
    reconciled = (
        row_delta == raw_only_rows
        and symbol_delta == raw_only_count
        and symbol_count_delta == raw_only_count
        and common_mismatches == 0
        and not audit_only
    )
    if reconciled:
        return True, (
            f"shared-symbol row counts match; {raw_only_count} raw-only symbols "
            f"account for all {raw_only_rows} excess rows and remain subject to separate review"
        )
    return False, (
        f"status={coverage.get('status')}; differences={differences}; "
        f"raw_only_symbols={raw_only_count}; raw_only_rows={raw_only_rows}; "
        f"common_symbol_mismatches={common_mismatches}; audit_only={len(audit_only)}"
    )


def _listing_eligibility_is_acceptable(listing: dict[str, Any]) -> bool:
    """Allow documented post-listing coverage gaps, never unknown/pre-listing rows."""
    counts = listing.get("status_counts")
    if not isinstance(counts, dict):
        return False
    # REVIEW_REQUIRED is expected when first observed bars start after listing.
    # The row-level policy permits post-listing observations but preserves the warning.
    return (
        listing.get("status") in {"PASS", "REVIEW_REQUIRED"}
        and counts.get("unknown") == 0
        and counts.get("prelisting_anomaly") == 0
    )


def evaluate_evidence_readiness(report_dir: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    reports_root = Path(report_dir)
    root = Path(repo_root)
    reports = {name: _read_json(reports_root / filename) for name, filename in REPORT_FILES.items()}
    registry_rows = _read_csv(root / "research_model/schemas/identity_source_registry.csv")
    document_rows = _read_csv(root / "research_model/schemas/historical_source_documents.csv")
    identity_rows = _read_csv(root / "research_model/schemas/security_identity_history.csv")
    action_rows = _read_csv(root / "research_model/schemas/corporate_action_events.csv")
    membership_rows = _read_csv(root / "research_model/schemas/index_membership_history.csv")

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    ohlcv = reports["ohlcv"]
    ohlcv_errors = {
        key: ohlcv.get(key, 0)
        for key in ("invalid_dates", "invalid_symbols", "malformed_rows", "invalid_numeric_rows",
                    "invalid_ohlc_rows", "duplicate_symbol_dates")
    }
    add("ohlcv_integrity", bool(ohlcv.get("row_count")) and all(v == 0 for v in ohlcv_errors.values()),
        f"rows={ohlcv.get('row_count')}; errors={ohlcv_errors}")

    action_audit = reports["corporate_action_audit"]
    action_reports = action_audit.get("actions", {})
    action_fields_clean = (
        not action_audit.get("action_columns_missing")
        and all(
            action_reports.get(field, {}).get("invalid_values", 0) == 0
            and action_reports.get(field, {}).get("negative_values_held_for_review", 0) == 0
            for field in ("dividends", "stock_splits")
        )
    )
    add("embedded_corporate_action_field_integrity", action_fields_clean,
        "embedded action fields contain no malformed or negative values" if action_fields_clean
        else "embedded action fields contain malformed or negative values")

    source_errors = reports["identity_source_validation"].get("error_count", 0)
    registry_by_id: dict[str, dict[str, str]] = {}
    registry_unique = True
    for row in registry_rows:
        source_id = (row.get("source_id") or "").strip()
        if not source_id or source_id in registry_by_id:
            registry_unique = False
        registry_by_id[source_id] = row
    add("source_registry_structure", source_errors == 0 and registry_unique,
        f"registry_rows={len(registry_rows)}; validation_errors={source_errors}")

    docs_errors = reports["source_document_validation"].get("error_count", 0)
    docs_by_hash: dict[str, list[dict[str, str]]] = {}
    docs_authoritative = bool(document_rows)
    for row in document_rows:
        digest = (row.get("sha256") or "").strip().lower()
        source_id = (row.get("source_id") or "").strip()
        source = registry_by_id.get(source_id, {})
        status = (row.get("capture_status") or "").strip().upper()
        reviewed = bool((row.get("reviewed_by") or "").strip()) and _utc(row.get("reviewed_at_utc") or "")
        source_reviewed = (
            (source.get("review_status") or "").strip().upper() == "REVIEWED_AUTHORITATIVE"
            and bool((source.get("reviewed_by") or "").strip())
            and _utc(source.get("reviewed_at_utc") or "")
        )
        if not digest or status != "INDEPENDENTLY_REVIEWED" or not reviewed or not source_reviewed:
            docs_authoritative = False
        docs_by_hash.setdefault(digest, []).append(row)
    archive = reports["source_document_validation"]
    archive_bytes = (
        archive.get("archive_integrity") == "CHECKED"
        and archive.get("byte_verified_documents") == len(document_rows)
        and len(document_rows) > 0
    )
    add("captured_source_documents", docs_errors == 0 and docs_authoritative and archive_bytes,
        f"documents={len(document_rows)}; byte_verified={archive.get('byte_verified_documents')}; "
        f"archive_integrity={archive.get('archive_integrity')}; authoritative_reviewed={docs_authoritative}")

    identity_validation = reports["identity_history_validation"]
    identity_verified = bool(identity_rows) and all(
        (row.get("verification_status") or "").strip().upper() == "INDEPENDENTLY_VERIFIED"
        and bool((row.get("reviewed_by") or "").strip())
        and _utc(row.get("reviewed_at_utc") or "")
        and any(
            (doc.get("source_id") or "").strip() == (row.get("source_id") or "").strip()
            and (doc.get("capture_status") or "").strip().upper() == "INDEPENDENTLY_REVIEWED"
            for doc in docs_by_hash.get((row.get("source_document_sha256") or "").strip().lower(), [])
        )
        for row in identity_rows
    )
    add("point_in_time_security_identity", identity_validation.get("error_count") == 0
        and identity_validation.get("structurally_valid_rows") == len(identity_rows)
        and identity_verified,
        f"identity_rows={len(identity_rows)}; verified={identity_verified}; "
        f"errors={identity_validation.get('error_count')}")

    action_validation = reports["corporate_action_validation"]
    actions_verified = bool(action_rows) and all(
        (row.get("verification_status") or "").strip().upper() == "INDEPENDENTLY_VERIFIED"
        and bool((row.get("reviewed_by") or "").strip())
        and _utc(row.get("reviewed_at_utc") or "")
        and any(
            (row.get("source_url") or "").strip() in {
                (doc.get("canonical_url") or "").strip(),
                (doc.get("retrieved_url") or "").strip(),
            }
            and (registry_by_id.get((doc.get("source_id") or "").strip(), {}).get("source_name") or "").strip()
                == (row.get("source_name") or "").strip()
            for doc in docs_by_hash.get((row.get("source_document_sha256") or "").strip().lower(), [])
        )
        for row in action_rows
    )
    add("corporate_action_evidence", action_validation.get("error_count") == 0
        and action_validation.get("structurally_valid_rows") == len(action_rows)
        and actions_verified,
        f"event_rows={len(action_rows)}; verified={actions_verified}; "
        f"errors={action_validation.get('error_count')}")

    membership_validation = reports["membership_history_validation"]
    membership_verified = bool(membership_rows) and all(
        (row.get("verification_status") or "").strip().upper() == "INDEPENDENTLY_VERIFIED"
        and bool((row.get("reviewed_by") or "").strip())
        and _utc(row.get("reviewed_at_utc") or "")
        and any(
            (doc.get("source_id") or "").strip() == (row.get("source_id") or "").strip()
            and (doc.get("capture_status") or "").strip().upper() == "INDEPENDENTLY_REVIEWED"
            for doc in docs_by_hash.get((row.get("source_document_sha256") or "").strip().lower(), [])
        )
        for row in membership_rows
    )
    add("historical_index_membership_intervals", membership_validation.get("error_count") == 0
        and membership_validation.get("structurally_valid_rows") == len(membership_rows)
        and membership_verified,
        f"interval_rows={len(membership_rows)}; verified={membership_verified}; "
        f"errors={membership_validation.get('error_count')}")

    listing = reports["listing_eligibility"]
    add("listing_date_eligibility", _listing_eligibility_is_acceptable(listing),
        f"status={listing.get('status')}; counts={listing.get('status_counts')}")

    coverage = reports["coverage_reconciliation"]
    universe = reports["universe_reconciliation"]
    coverage_matches, coverage_detail = _common_symbol_coverage_matches(coverage, universe)
    add("historical_coverage_reconciliation", coverage_matches, coverage_detail)

    raw_review = reports["raw_symbol_review"]
    add("raw_only_symbol_evidence_review", raw_review.get("status") == "REVIEWED",
        f"status={raw_review.get('status')}; pending={len(raw_review.get('pending_symbols', []))}; "
        f"errors={raw_review.get('error_count')}")

    add("universe_reconciliation", universe.get("status") == "MATCH",
        f"status={universe.get('status')}; raw_only={len(universe.get('raw_only_symbols', []))}; "
        f"audit_only={len(universe.get('audit_only_symbols', []))}")

    passed = all(item["passed"] for item in checks)
    return {
        "status": "READY_FOR_RESEARCH" if passed else "BLOCKED",
        "approval_scope": "HISTORICAL_RESEARCH_DATA_ONLY_NOT_PRODUCTION_OR_LIVE_TRADING",
        "check_count": len(checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "blocked_checks": sum(not item["passed"] for item in checks),
        "checks": checks,
        "source_row_counts": {
            "source_registry": len(registry_rows),
            "captured_documents": len(document_rows),
            "security_identity_intervals": len(identity_rows),
            "corporate_action_events": len(action_rows),
            "index_membership_intervals": len(membership_rows),
        },
        "human_review_note": (
            "Reviewer fields are auditable attestations, not cryptographic proof that an independent review occurred. "
            "The accountable research owner must retain the original documents and review trail."
        ),
        "training_authorized": passed,
        "production_authorized": False,
        "live_trading_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_evidence_readiness(args.report_dir, args.repo_root)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if report["status"] != "READY_FOR_RESEARCH":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
