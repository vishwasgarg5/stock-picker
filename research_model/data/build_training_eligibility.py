"""Build a fail-closed, per-observation training eligibility ledger.

The output preserves every source row and marks it ELIGIBLE or QUARANTINED.
It never mutates the OHLCV source. Missing provenance or coverage is a
quarantine reason, not an implicit pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .listing_eligibility import load_listing_evidence, UNKNOWN, PRELISTING, GAP_REVIEW, VERIFIED


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV headers: {path}")
        return list(reader)


def _utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _valid_interval(row: dict[str, str], day: date) -> bool:
    try:
        start = date.fromisoformat(row["valid_from"])
        end = date.fromisoformat(row["valid_to"]) if row.get("valid_to") else date.max
    except (KeyError, ValueError):
        return False
    return start <= day <= end


def _reviewed(row: dict[str, str], docs: dict[str, dict[str, str]]) -> bool:
    digest = (row.get("source_document_sha256") or "").strip().lower()
    doc = docs.get(digest)
    return bool(
        doc
        and len(digest) == 64
        and doc.get("capture_status", "").upper() == "INDEPENDENTLY_REVIEWED"
        and doc.get("sha256", "").lower() == digest
        and doc.get("archive_path", "").strip()
        and row.get("reviewed_by", "").strip()
        and _utc(row.get("reviewed_at_utc", ""))
    )


def _active(rows: list[dict[str, str]], day: date, key: str, value: str) -> list[dict[str, str]]:
    return [r for r in rows if r.get(key, "").strip() == value and _valid_interval(r, day)]


def build_training_eligibility(
    ohlcv_path: str | Path,
    listing_audit_path: str | Path,
    identity_path: str | Path,
    source_coverage_path: str | Path,
    action_coverage_path: str | Path,
    documents_path: str | Path,
    raw_only_review_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one auditable decision row per OHLCV row; missing evidence fails closed."""
    source = Path(ohlcv_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    listings = load_listing_evidence(listing_audit_path)
    identities = _read_rows(identity_path)
    source_coverage = _read_rows(source_coverage_path)
    action_coverage = _read_rows(action_coverage_path)
    documents = {r.get("sha256", "").strip().lower(): r for r in _read_rows(documents_path) if r.get("sha256")}
    raw_review = _read_rows(raw_only_review_path)
    raw_only = {
        r.get("symbol", "").strip()
        for r in raw_review
        if r.get("historical_disposition", "").strip().upper() not in {"INCLUDE_VERIFIED"}
    }

    counts: dict[str, int] = {}
    eligible = quarantined = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_row_number", "date", "symbol", "isin", "venue", "source_id",
        "source_document_sha256", "identity_valid_from", "identity_valid_to",
        "corporate_action_coverage_sha256", "decision", "reason_codes",
        "source_ohlcv_sha256",
    ]
    with source.open("r", encoding="utf-8-sig", newline="") as src, output.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"OHLCV source must contain {sorted(required)}")
        writer = csv.DictWriter(dst, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for line, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip()
            day_text = (row.get("date") or "").strip()
            reasons: list[str] = []
            isin = venue = source_id = source_hash_ref = identity_from = identity_to = action_hash = ""
            try:
                day = date.fromisoformat(day_text)
            except ValueError:
                day = None
                reasons.append("INVALID_OBSERVATION_DATE")

            listing = listings.get(symbol)
            if symbol in raw_only:
                reasons.append("RAW_ONLY_SYMBOL_NOT_APPROVED")
            if listing is None:
                reasons.append("NO_LISTING_EVIDENCE")
            elif listing["status"] == UNKNOWN:
                reasons.append("LISTING_DATE_UNVERIFIED")
            elif day and day < date.fromisoformat(listing["listing_date"]):
                reasons.append("BEFORE_VERIFIED_LISTING_DATE")
            elif listing["status"] == PRELISTING:
                # Post-listing rows may proceed only if all independent identity,
                # source and action coverage checks below also pass.
                reasons.append("PRELISTING_ANOMALY_SYMBOL_REQUIRES_REVIEW")
            elif listing["status"] == GAP_REVIEW:
                reasons.append("LISTING_GAP_REVIEW_REQUIRED")

            identity_matches = _active(identities, day, "symbol", symbol) if day else []
            identity_matches = [
                r for r in identity_matches
                if r.get("verification_status", "").upper() == "INDEPENDENTLY_VERIFIED"
                and r.get("series", "").strip()
            ]
            if len(identity_matches) != 1:
                reasons.append("IDENTITY_INTERVAL_MISSING_OR_AMBIGUOUS")
            else:
                identity = identity_matches[0]
                isin = identity.get("isin", "").strip()
                identity_from = identity.get("valid_from", "")
                identity_to = identity.get("valid_to", "")
                identity_doc_hash = identity.get("source_document_sha256", "").lower()
                if not _reviewed(identity, documents) or identity_doc_hash != identity.get("source_document_sha256", "").lower():
                    reasons.append("IDENTITY_SOURCE_DOCUMENT_NOT_VERIFIED")
                else:
                    source_hash_ref = identity_doc_hash

                source_matches = _active(source_coverage, day, "symbol", symbol)
                source_matches = [
                    r for r in source_matches
                    if r.get("coverage_status", "").upper() == "INDEPENDENTLY_VERIFIED"
                    and r.get("isin", isin).strip() in {"", isin}
                    and r.get("venue", "").strip()
                    and r.get("source_id", "").strip()
                    and _reviewed(r, documents)
                    and r.get("source_document_sha256", "").lower() == identity_doc_hash
                ]
                if len(source_matches) != 1:
                    reasons.append("OHLCV_ROW_SOURCE_PROVENANCE_MISSING_OR_AMBIGUOUS")
                else:
                    venue = source_matches[0]["venue"].strip()
                    source_id = source_matches[0]["source_id"].strip()

                action_matches = _active(action_coverage, day, "isin", isin) if isin else []
                action_matches = [
                    r for r in action_matches
                    if r.get("coverage_status", "").upper() == "INDEPENDENTLY_VERIFIED"
                    and _reviewed(r, documents)
                ]
                if len(action_matches) != 1:
                    reasons.append("CORPORATE_ACTION_COVERAGE_MISSING_OR_AMBIGUOUS")
                else:
                    action_hash = action_matches[0].get("source_document_sha256", "").lower()

            decision = "ELIGIBLE" if not reasons else "QUARANTINED"
            if decision == "ELIGIBLE":
                eligible += 1
            else:
                quarantined += 1
            for reason in set(reasons):
                counts[reason] = counts.get(reason, 0) + 1
            writer.writerow({
                "source_row_number": line,
                "date": day_text,
                "symbol": symbol,
                "isin": isin,
                "venue": venue,
                "source_id": source_id,
                "source_document_sha256": source_hash_ref,
                "identity_valid_from": identity_from,
                "identity_valid_to": identity_to,
                "corporate_action_coverage_sha256": action_hash,
                "decision": decision,
                "reason_codes": ";".join(sorted(set(reasons))),
                "source_ohlcv_sha256": source_hash,
            })

    manifest_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "status": "PASS" if quarantined == 0 else "REVIEW_REQUIRED",
        "source_path": source.as_posix(),
        "source_sha256": source_hash,
        "source_row_count": eligible + quarantined,
        "eligible_row_count": eligible,
        "quarantined_row_count": quarantined,
        "reason_counts": dict(sorted(counts.items())),
        "eligibility_manifest_path": output.as_posix(),
        "eligibility_manifest_sha256": manifest_hash,
        "source_modified": False,
        "policy": "Only rows with independently verified listing, identity, row-source provenance, and corporate-action coverage are eligible.",
    }
    if summary_path:
        target = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--listing-audit", required=True)
    parser.add_argument("--identity-history", required=True)
    parser.add_argument("--source-coverage", required=True)
    parser.add_argument("--action-coverage", required=True)
    parser.add_argument("--source-documents", required=True)
    parser.add_argument("--raw-only-review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    report = build_training_eligibility(
        args.ohlcv, args.listing_audit, args.identity_history, args.source_coverage,
        args.action_coverage, args.source_documents, args.raw_only_review,
        args.output, args.summary,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
