"""Strict validation for externally sourced, effective-dated universe membership.

This module does not infer historical membership from current constituent snapshots.
Only records carrying explicit effective dates, publication timestamps and source
provenance are accepted. It is a research contract, not a membership data source.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = (
    "schema_version", "universe", "symbol", "exchange", "asset_class",
    "effective_from", "published_at", "source",
)
OPTIONAL_FIELDS = ("effective_to",)
ALLOWED_EXCHANGES = {"NSE", "BSE", "MCX"}
SCHEMA_VERSION = "1.0.0"


class MembershipEvidenceError(ValueError):
    """Raised when membership evidence is incomplete or contradictory."""


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise MembershipEvidenceError(f"{field} is required")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MembershipEvidenceError(f"{field} must be an ISO YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise MembershipEvidenceError(f"{field} must use canonical YYYY-MM-DD format")
    return parsed


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MembershipEvidenceError(f"{field} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MembershipEvidenceError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MembershipEvidenceError(f"{field} must include a timezone offset")
    return parsed


def validate_membership_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate records and reject duplicate/overlapping intervals per universe-symbol."""
    accepted: list[dict[str, Any]] = []
    intervals: dict[tuple[str, str], list[tuple[date, date | None]]] = {}
    seen: set[tuple[str, str, date]] = set()
    count = 0

    for number, original in enumerate(records, start=1):
        count += 1
        if not isinstance(original, dict):
            raise MembershipEvidenceError(f"record {number} must be an object")
        missing = set(REQUIRED_FIELDS) - set(original)
        extra = set(REQUIRED_FIELDS + OPTIONAL_FIELDS) ^ set(REQUIRED_FIELDS + OPTIONAL_FIELDS)
        unknown = set(original) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
        if missing or unknown:
            raise MembershipEvidenceError(
                f"record {number} fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if original["schema_version"] != SCHEMA_VERSION:
            raise MembershipEvidenceError(f"record {number} has unsupported schema_version")
        universe = str(original["universe"]).strip().upper()
        symbol = str(original["symbol"]).strip().upper()
        exchange = str(original["exchange"]).strip().upper()
        asset_class = str(original["asset_class"]).strip()
        source = str(original["source"]).strip()
        if not universe or not symbol or not asset_class or not source:
            raise MembershipEvidenceError(f"record {number} has empty identity or source provenance")
        if exchange not in ALLOWED_EXCHANGES:
            raise MembershipEvidenceError(f"record {number} has unsupported exchange: {exchange}")
        start = _parse_date(original["effective_from"], "effective_from")
        end = _parse_date(original["effective_to"], "effective_to") if original.get("effective_to") else None
        if end is not None and end < start:
            raise MembershipEvidenceError(f"record {number} effective_to precedes effective_from")
        _parse_timestamp(original["published_at"], "published_at")
        key = (universe, symbol, start)
        if key in seen:
            raise MembershipEvidenceError(f"duplicate membership start for {universe}/{symbol}/{start}")
        seen.add(key)
        intervals.setdefault((universe, symbol), []).append((start, end))
        accepted.append({
            "schema_version": SCHEMA_VERSION, "universe": universe, "symbol": symbol,
            "exchange": exchange, "asset_class": asset_class, "effective_from": start.isoformat(),
            "effective_to": end.isoformat() if end else None,
            "published_at": str(original["published_at"]), "source": source,
        })

    for identity, spans in intervals.items():
        ordered = sorted(spans, key=lambda span: span[0])
        for previous, current in zip(ordered, ordered[1:]):
            previous_start, previous_end = previous
            current_start, _ = current
            if previous_end is None or current_start <= previous_end:
                raise MembershipEvidenceError(
                    f"overlapping effective intervals for {identity[0]}/{identity[1]}"
                )

    return {
        "status": "PASS" if count else "BLOCKED",
        "record_count": count,
        "validated_records": accepted,
        "point_in_time_ready": bool(count),
        "historical_membership_inferred": False,
    }


def validate_membership_csv(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"status": "BLOCKED", "record_count": 0,
                "error": f"membership evidence file not found: {source}",
                "point_in_time_ready": False, "historical_membership_inferred": False}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        return validate_membership_records(records)
    except (OSError, UnicodeDecodeError, csv.Error, MembershipEvidenceError) as exc:
        return {"status": "BLOCKED", "record_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "point_in_time_ready": False, "historical_membership_inferred": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/universe_membership_evidence.csv")
    parser.add_argument("--output", default="audit/broker_universe/membership_validation.json")
    args = parser.parse_args()
    result = validate_membership_csv(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "validated_records"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
