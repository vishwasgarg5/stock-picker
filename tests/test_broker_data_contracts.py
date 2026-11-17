from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.broker_data_contracts import (
    DataContractError,
    REQUIRED_RESEARCH_GATES,
    build_ingestion_manifest,
    require_research_gates,
    self_check,
    validate_ohlcv_record,
    validate_point_in_time,
    validate_unique_sessions,
    verify_ingestion_manifest,
)


def row(day="2026-11-06", symbol="RELIANCE"):
    return {
        "schema_version": "1.0.0", "date": day, "symbol": symbol,
        "exchange": "NSE", "asset_class": "equity_cash",
        "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0,
        "volume": 1000, "source": "test",
        "retrieved_at": "2026-11-06T12:00:00+05:30",
    }


def manifest(raw=b"raw", normalized=b"normalized"):
    return build_ingestion_manifest(
        dataset="nse_cm_ohlcv", source_id="NSE_CM_UDIFF_BHAVCOPY",
        source_url="https://www.nseindia.com/test", exchange="NSE",
        session_date="2026-11-06", retrieved_at="2026-11-06T12:00:00+05:30",
        raw_bytes=raw, normalized_bytes=normalized,
        raw_file="raw/session.zip", normalized_file="normalized/session.csv",
        record_count=1, series_filter="EQ",
    )


def test_ci_self_check_is_research_only():
    result = self_check()
    assert result["status"] == "PASS"
    assert result["production_behaviour_changed"] is False
    assert result["ledger_mutated"] is False
    assert result["promotion_authorized"] is False
    assert result["live_execution_enabled"] is False


@pytest.mark.parametrize("field,value", [
    ("high", 98.0), ("low", 106.0), ("open", float("nan")),
    ("close", True), ("volume", -1), ("volume", float("inf")),
    ("exchange", "UNKNOWN"), ("source", ""),
])
def test_ohlcv_rejects_invalid_values(field, value):
    item = row()
    item[field] = value
    with pytest.raises(DataContractError):
        validate_ohlcv_record(item)


def test_ohlcv_rejects_missing_fields_and_naive_timestamp():
    item = row()
    del item["source"]
    with pytest.raises(DataContractError):
        validate_ohlcv_record(item)
    item = row()
    item["retrieved_at"] = "2026-11-06T12:00:00"
    with pytest.raises(DataContractError):
        validate_ohlcv_record(item)


def test_unique_sessions_reject_duplicates_and_out_of_order_rows():
    with pytest.raises(DataContractError, match="Duplicate"):
        validate_unique_sessions([row(), row()])
    with pytest.raises(DataContractError, match="Non-chronological"):
        validate_unique_sessions([row("2026-11-06"), row("2026-11-05")])


def test_point_in_time_rejects_lookahead_and_naive_timestamps():
    with pytest.raises(DataContractError, match="Look-ahead"):
        validate_point_in_time("2026-11-06T10:00:00+05:30", "2026-11-06T09:15:00+05:30")
    with pytest.raises(DataContractError, match="timezone"):
        validate_point_in_time("2026-11-06T09:00:00", "2026-11-06T09:15:00+05:30")


def test_manifest_hashes_verify_and_tampering_is_rejected():
    item = manifest()
    verify_ingestion_manifest(item, raw_bytes=b"raw", normalized_bytes=b"normalized")
    with pytest.raises(DataContractError, match="raw_sha256"):
        verify_ingestion_manifest(item, raw_bytes=b"tampered", normalized_bytes=b"normalized")


@pytest.mark.parametrize("kwargs", [
    {"source_url": "http://example.com/data"},
    {"raw_file": "../escape.zip"},
    {"normalized_file": "/tmp/data.csv"},
    {"record_count": 0},
    {"exchange": "UNKNOWN"},
])
def test_manifest_builder_rejects_unsafe_or_invalid_inputs(kwargs):
    values = {
        "dataset": "nse_cm_ohlcv", "source_id": "NSE_CM_UDIFF_BHAVCOPY",
        "source_url": "https://www.nseindia.com/test", "exchange": "NSE",
        "session_date": "2026-11-06", "retrieved_at": "2026-11-06T12:00:00+05:30",
        "raw_bytes": b"raw", "normalized_bytes": b"normalized",
        "raw_file": "raw/session.zip", "normalized_file": "normalized/session.csv",
        "record_count": 1,
    }
    values.update(kwargs)
    with pytest.raises(DataContractError):
        build_ingestion_manifest(**values)


def test_manifest_verifier_rejects_research_policy_and_digest_shape():
    item = manifest()
    invalid = copy.deepcopy(item)
    invalid["research_only"] = False
    with pytest.raises(DataContractError):
        verify_ingestion_manifest(invalid)
    invalid = copy.deepcopy(item)
    invalid["raw_sha256"] = "not-a-hash"
    with pytest.raises(DataContractError):
        verify_ingestion_manifest(invalid)


def test_readiness_gates_fail_closed_for_missing_false_or_non_boolean():
    with pytest.raises(DataContractError):
        require_research_gates({})
    gates = {name: True for name in REQUIRED_RESEARCH_GATES}
    gates[REQUIRED_RESEARCH_GATES[0]] = 1
    with pytest.raises(DataContractError):
        require_research_gates(gates)
    require_research_gates({name: True for name in REQUIRED_RESEARCH_GATES})
