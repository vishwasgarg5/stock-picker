"""Broker-inspired, fail-closed data contracts for Stock Picker.

These helpers are deliberately additive: they validate externally supplied
market-data evidence but do not fetch data, mutate OHLCV/trade ledgers, change
model features, or authorize trading/promotion.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0.0"
ALLOWED_EXCHANGES = frozenset({"NSE", "BSE", "MCX"})
REQUIRED_RESEARCH_GATES = (
    "data_quality_passed",
    "point_in_time_passed",
    "chronological_validation_passed",
    "benchmark_comparison_passed",
    "costs_and_slippage_included",
    "risk_limits_passed",
    "paper_trading_reconciled",
)


class DataContractError(ValueError):
    """Raised when external data or evidence violates a research contract."""


def _aware_timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DataContractError(f"{field} must include a timezone")
    return parsed


def _iso_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be an ISO date") from exc


def validate_ohlcv_record(row: Mapping[str, Any]) -> None:
    """Validate a normalized OHLCV record without coercing or modifying it."""
    required = {
        "schema_version", "date", "symbol", "exchange", "asset_class",
        "open", "high", "low", "close", "volume", "source", "retrieved_at",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise DataContractError(f"OHLCV missing required fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise DataContractError("Unsupported OHLCV schema version")
    _iso_date(row["date"], "date")
    _aware_timestamp(row["retrieved_at"], "retrieved_at")
    if not str(row["symbol"]).strip() or not str(row["asset_class"]).strip():
        raise DataContractError("symbol and asset_class must be non-empty")
    exchange = str(row["exchange"]).strip().upper()
    if exchange not in ALLOWED_EXCHANGES:
        raise DataContractError(f"Unsupported exchange: {exchange}")
    if not str(row["source"]).strip():
        raise DataContractError("source must be non-empty")

    prices = [row[name] for name in ("open", "high", "low", "close")]
    for value in prices:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataContractError("OHLC prices must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise DataContractError("OHLC prices must be positive and finite")
    if row["high"] < max(prices) or row["low"] > min(prices):
        raise DataContractError("OHLC high/low range is inconsistent")

    volume = row["volume"]
    if isinstance(volume, bool) or not isinstance(volume, (int, float)):
        raise DataContractError("volume must be numeric")
    if not math.isfinite(volume) or volume < 0:
        raise DataContractError("volume must be non-negative and finite")


def validate_unique_sessions(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject duplicate symbol/exchange/session keys and out-of-order rows."""
    seen: set[tuple[str, str, date]] = set()
    previous: dict[tuple[str, str], date] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        exchange = str(row.get("exchange", "")).strip().upper()
        if not symbol or exchange not in ALLOWED_EXCHANGES:
            raise DataContractError("Every row needs a symbol and supported exchange")
        session = _iso_date(row.get("date"), "date")
        key = (symbol, exchange, session)
        if key in seen:
            raise DataContractError(f"Duplicate symbol/exchange/session: {key}")
        seen.add(key)
        instrument = (symbol, exchange)
        if instrument in previous and session < previous[instrument]:
            raise DataContractError(f"Non-chronological rows for {symbol}/{exchange}")
        previous[instrument] = session


def validate_point_in_time(feature_available_at: str, decision_at: str) -> None:
    """Reject any feature that was not available by the decision timestamp."""
    available = _aware_timestamp(feature_available_at, "feature_available_at")
    decision = _aware_timestamp(decision_at, "decision_at")
    if available > decision:
        raise DataContractError("Look-ahead detected: feature unavailable at decision time")


def _safe_relative_path(value: Any, field: str) -> str:
    raw = str(value or "").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DataContractError(f"{field} must be a safe relative POSIX path")
    return raw


def build_ingestion_manifest(
    *,
    dataset: str,
    source_id: str,
    source_url: str,
    exchange: str,
    session_date: str,
    retrieved_at: str,
    raw_bytes: bytes,
    normalized_bytes: bytes,
    raw_file: str,
    normalized_file: str,
    record_count: int,
    series_filter: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic hash manifest; does not write or mutate source files."""
    parsed_url = urlparse(str(source_url))
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise DataContractError("source_url must be an HTTPS URL")
    if not str(dataset).strip() or not str(source_id).strip():
        raise DataContractError("dataset and source_id must be non-empty")
    exchange = str(exchange).strip().upper()
    if exchange not in ALLOWED_EXCHANGES:
        raise DataContractError(f"Unsupported exchange: {exchange}")
    _iso_date(session_date, "session_date")
    _aware_timestamp(retrieved_at, "retrieved_at")
    if not isinstance(raw_bytes, bytes) or not isinstance(normalized_bytes, bytes):
        raise DataContractError("raw and normalized payloads must be bytes")
    if not raw_bytes or not normalized_bytes:
        raise DataContractError("raw and normalized payloads must be non-empty")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 1:
        raise DataContractError("record_count must be a positive integer")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset).strip(),
        "source_id": str(source_id).strip(),
        "source_url": str(source_url),
        "exchange": exchange,
        "session_date": str(session_date),
        "retrieved_at": str(retrieved_at),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "raw_file": _safe_relative_path(raw_file, "raw_file"),
        "normalized_file": _safe_relative_path(normalized_file, "normalized_file"),
        "record_count": record_count,
        "research_only": True,
    }
    if series_filter is not None:
        manifest["series_filter"] = str(series_filter)
    return manifest


def verify_ingestion_manifest(
    manifest: Mapping[str, Any],
    *,
    raw_bytes: bytes | None = None,
    normalized_bytes: bytes | None = None,
) -> None:
    """Validate manifest structure and, when supplied, verify both payload hashes."""
    required = {
        "schema_version", "dataset", "source_id", "source_url", "exchange",
        "session_date", "retrieved_at", "raw_sha256", "normalized_sha256",
        "raw_file", "normalized_file", "record_count", "research_only",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise DataContractError(f"Ingestion manifest missing fields: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["research_only"] is not True:
        raise DataContractError("Manifest schema or research-only policy invalid")
    if str(manifest["exchange"]).upper() not in ALLOWED_EXCHANGES:
        raise DataContractError("Manifest exchange is unsupported")
    parsed_url = urlparse(str(manifest["source_url"]))
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise DataContractError("Manifest source_url must be HTTPS")
    _iso_date(manifest["session_date"], "session_date")
    _aware_timestamp(manifest["retrieved_at"], "retrieved_at")
    _safe_relative_path(manifest["raw_file"], "raw_file")
    _safe_relative_path(manifest["normalized_file"], "normalized_file")
    if isinstance(manifest["record_count"], bool) or not isinstance(manifest["record_count"], int) or manifest["record_count"] < 1:
        raise DataContractError("Manifest record_count must be a positive integer")
    for field in ("raw_sha256", "normalized_sha256"):
        digest = str(manifest[field])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DataContractError(f"{field} must be a lowercase SHA-256 digest")
    for field, payload in (("raw_sha256", raw_bytes), ("normalized_sha256", normalized_bytes)):
        if payload is not None and hashlib.sha256(payload).hexdigest() != manifest[field]:
            raise DataContractError(f"{field} does not match supplied payload")


def require_research_gates(gates: Mapping[str, Any]) -> None:
    """Fail closed: missing, false, or non-boolean readiness flags block."""
    missing = [name for name in REQUIRED_RESEARCH_GATES if name not in gates]
    failed = [name for name in REQUIRED_RESEARCH_GATES if gates.get(name) is not True]
    if missing or failed:
        raise DataContractError(f"Readiness blocked; missing={missing}, failed={failed}")


def self_check() -> dict[str, Any]:
    """Small deterministic smoke check used by CI; creates no evidence claims."""
    sample = {
        "schema_version": SCHEMA_VERSION, "date": "2026-11-06",
        "symbol": "RELIANCE", "exchange": "NSE", "asset_class": "equity_cash",
        "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0,
        "volume": 1000, "source": "self-check",
        "retrieved_at": "2026-11-06T12:00:00+05:30",
    }
    validate_ohlcv_record(sample)
    validate_unique_sessions([sample])
    validate_point_in_time("2026-11-06T09:00:00+05:30", "2026-11-06T09:15:00+05:30")
    raw, normalized = b"raw-fixture", b"normalized-fixture"
    manifest = build_ingestion_manifest(
        dataset="nse_cm_ohlcv", source_id="NSE_CM_UDIFF_BHAVCOPY",
        source_url="https://www.nseindia.com/test", exchange="NSE",
        session_date="2026-11-06", retrieved_at="2026-11-06T12:00:00+05:30",
        raw_bytes=raw, normalized_bytes=normalized,
        raw_file="raw/session.zip", normalized_file="normalized/session.csv",
        record_count=1, series_filter="EQ",
    )
    verify_ingestion_manifest(manifest, raw_bytes=raw, normalized_bytes=normalized)
    require_research_gates({name: True for name in REQUIRED_RESEARCH_GATES})
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "checks": [
            "ohlcv_contract", "unique_chronological_sessions", "point_in_time",
            "immutable_hash_manifest", "fail_closed_research_gates",
        ],
        "production_behaviour_changed": False,
        "ledger_mutated": False,
        "promotion_authorized": False,
        "live_execution_enabled": False,
    }


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, sort_keys=True))
