"""Validate evidence-backed dispositions for raw-only historical OHLCV symbols."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED = {
    "symbol", "raw_row_count", "first_observed_date", "last_observed_date",
    "in_current_universe", "historical_disposition", "source_document_id",
    "reviewed_by", "reviewed_at_utc", "review_notes",
}
DISPOSITIONS = {"PENDING", "INCLUDE_IN_HISTORICAL_UNIVERSE", "EXCLUDE_FROM_RESEARCH_WITH_EVIDENCE"}


def _utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def validate_raw_symbol_review(
    diagnostic_path: str | Path, ledger_path: str | Path, documents_path: str | Path
) -> dict[str, Any]:
    diagnostic = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
    expected = {row["symbol"]: row for row in diagnostic.get("raw_only_symbols", [])}
    with Path(documents_path).open("r", encoding="utf-8-sig", newline="") as handle:
        docs_reader = csv.DictReader(handle)
        if not docs_reader.fieldnames or "document_id" not in docs_reader.fieldnames:
            raise ValueError("source document manifest must contain document_id")
        document_ids = {(row.get("document_id") or "").strip() for row in docs_reader}
    with Path(ledger_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"review ledger missing columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)

    seen: set[str] = set()
    errors: list[str] = []
    pending: list[str] = []
    approved: list[str] = []
    for row in rows:
        symbol = (row.get("symbol") or "").strip()
        if not symbol or symbol in seen:
            errors.append(f"blank or duplicate ledger symbol: {symbol!r}")
            continue
        seen.add(symbol)
        source = expected.get(symbol)
        if source is None:
            errors.append(f"ledger symbol is not in current raw-only diagnostic: {symbol}")
            continue
        expected_values = {
            "raw_row_count": str(source["row_count"]),
            "first_observed_date": source["first_date"],
            "last_observed_date": source["last_date"],
            "in_current_universe": str(source["in_current_universe"]).lower(),
        }
        for field, value in expected_values.items():
            if (row.get(field) or "").strip() != value:
                errors.append(f"{symbol}: {field} does not match current diagnostic")
        disposition = (row.get("historical_disposition") or "").strip().upper()
        if disposition not in DISPOSITIONS:
            errors.append(f"{symbol}: unsupported historical_disposition {disposition!r}")
            continue
        if disposition == "PENDING":
            pending.append(symbol)
            continue
        document_id = (row.get("source_document_id") or "").strip()
        if not document_id or document_id not in document_ids:
            errors.append(f"{symbol}: disposition requires a source_document_id in the document manifest")
        if not (row.get("reviewed_by") or "").strip():
            errors.append(f"{symbol}: disposition requires reviewed_by")
        if not _utc(row.get("reviewed_at_utc") or ""):
            errors.append(f"{symbol}: disposition requires UTC reviewed_at_utc")
        if not (row.get("review_notes") or "").strip():
            errors.append(f"{symbol}: disposition requires review_notes")
        approved.append(symbol)

    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing ledger rows: {','.join(missing)}")
    status = "REVIEWED" if not errors and not pending and len(seen) == len(expected) else "REVIEW_REQUIRED"
    return {
        "status": status,
        "expected_raw_only_symbol_count": len(expected),
        "ledger_row_count": len(rows),
        "pending_symbols": sorted(pending),
        "approved_symbols": sorted(approved),
        "missing_symbols": missing,
        "error_count": len(errors),
        "errors": errors,
        "input_modified": False,
        "interpretation": (
            "Human review ledger only. A disposition never changes or deletes OHLCV rows and does not "
            "by itself establish historical index membership or authorize training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-documents", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_raw_symbol_review(args.diagnostic, args.ledger, args.source_documents)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if report["status"] != "REVIEWED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
