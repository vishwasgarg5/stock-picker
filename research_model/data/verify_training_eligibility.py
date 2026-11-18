"""Independently verify a row-level training eligibility manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_eligibility(source_path: str | Path, manifest_path: str | Path,
                                summary_path: str | Path) -> dict[str, Any]:
    """Recompute hashes and compare each manifest observation to its source row."""
    source, manifest, summary_file = map(Path, (source_path, manifest_path, summary_path))
    errors: list[str] = []
    source_hash = _sha256(source)
    manifest_hash = _sha256(manifest)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("training eligibility summary must be a JSON object")
    if summary.get("source_sha256") != source_hash:
        errors.append("summary source_sha256 does not match source bytes")
    if summary.get("eligibility_manifest_sha256") != manifest_hash:
        errors.append("summary eligibility_manifest_sha256 does not match manifest bytes")

    required = {
        "source_row_number", "date", "symbol", "isin", "venue", "source_id",
        "identity_source_document_sha256", "ohlcv_source_document_sha256",
        "identity_valid_from", "identity_valid_to", "corporate_action_coverage_sha256",
        "decision", "reason_codes", "source_ohlcv_sha256",
    }
    eligible = quarantined = source_rows = 0
    seen_numbers: set[int] = set()
    sentinel = object()
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle, manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as manifest_handle:
        source_reader = csv.DictReader(source_handle)
        manifest_reader = csv.DictReader(manifest_handle)
        if not source_reader.fieldnames or not {"date", "symbol"}.issubset(source_reader.fieldnames):
            errors.append("source OHLCV headers are missing date/symbol")
        fields = set(manifest_reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            errors.append(f"manifest missing required columns: {missing}")

        for physical_row, pair in enumerate(zip_longest(source_reader, manifest_reader, fillvalue=sentinel), start=2):
            source_row, row = pair
            if source_row is sentinel:
                errors.append(f"manifest has an extra row at physical row {physical_row}")
                continue
            source_rows += 1
            if row is sentinel:
                errors.append(f"manifest is missing source row {physical_row}")
                continue

            if row.get("date", "") != source_row.get("date", ""):
                errors.append(f"row {physical_row}: manifest date does not match source observation")
            if row.get("symbol", "").strip() != source_row.get("symbol", "").strip():
                errors.append(f"row {physical_row}: manifest symbol does not match source observation")
            try:
                number = int(row.get("source_row_number", ""))
            except (TypeError, ValueError):
                errors.append(f"row {physical_row}: invalid source_row_number")
                number = -1
            if number in seen_numbers:
                errors.append(f"row {physical_row}: duplicate source_row_number {number}")
            seen_numbers.add(number)
            if number != physical_row:
                errors.append(f"row {physical_row}: source_row_number must be sequential and match source row")
            if row.get("source_ohlcv_sha256") != source_hash:
                errors.append(f"row {physical_row}: source_ohlcv_sha256 mismatch")

            decision = row.get("decision", "")
            reasons = [item for item in row.get("reason_codes", "").split(";") if item]
            if decision == "ELIGIBLE":
                eligible += 1
                required_refs = (
                    "isin", "venue", "source_id", "identity_source_document_sha256",
                    "ohlcv_source_document_sha256", "identity_valid_from",
                    "corporate_action_coverage_sha256",
                )
                if reasons:
                    errors.append(f"row {physical_row}: eligible row has reason_codes")
                if any(not row.get(key, "").strip() for key in required_refs):
                    errors.append(f"row {physical_row}: eligible row is missing required evidence references")
            elif decision == "QUARANTINED":
                quarantined += 1
                if not reasons:
                    errors.append(f"row {physical_row}: quarantined row has no reason_codes")
            else:
                errors.append(f"row {physical_row}: invalid decision {decision!r}")

    manifest_rows = eligible + quarantined
    if summary.get("source_row_count") != source_rows:
        errors.append("summary source_row_count does not match source row count")
    if summary.get("eligible_row_count") != eligible:
        errors.append("summary eligible_row_count does not match manifest")
    if summary.get("quarantined_row_count") != quarantined:
        errors.append("summary quarantined_row_count does not match manifest")
    if len(seen_numbers) != manifest_rows:
        errors.append("manifest source row numbers are not unique")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "source_row_count": source_rows,
        "manifest_row_count": manifest_rows,
        "eligible_row_count": eligible,
        "quarantined_row_count": quarantined,
        "error_count": len(errors),
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_training_eligibility(args.source, args.manifest, args.summary)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
