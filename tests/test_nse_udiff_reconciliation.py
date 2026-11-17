from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.broker_data_contracts import DataContractError
from src.nse_udiff_reconciliation import (
    SOURCE_ID,
    extract_csv,
    normalize_bhavcopy,
    run,
    write_research_snapshot,
)

HEADER = "TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol"


def csv_payload(*rows):
    return (HEADER + "\n" + "\n".join(rows) + "\n").encode()


def zip_payload(payload):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bhavcopy.csv", payload)
    return output.getvalue()


def test_extracts_exactly_one_csv_from_official_zip():
    payload = csv_payload("2026-11-06,RELIANCE,EQ,100,105,99,103,1000")
    assert extract_csv(zip_payload(payload)) == payload


def test_rejects_zip_with_multiple_csv_files():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("one.csv", b"a")
        archive.writestr("two.csv", b"b")
    with pytest.raises(DataContractError, match="one CSV"):
        extract_csv(output.getvalue())


def test_normalizes_eq_rows_and_excludes_other_series():
    payload = csv_payload(
        "2026-11-06,RELIANCE,EQ,100,105,99,103,1000",
        "2026-11-06,RELIANCE,BE,100,105,99,103,20",
    )
    rows = normalize_bhavcopy(payload, date(2026, 11, 6), "2026-11-06T12:00:00+00:00")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "RELIANCE"
    assert rows[0]["source"] == SOURCE_ID
    assert rows[0]["volume"] == 1000


@pytest.mark.parametrize("payload,expected", [
    (b"TradDt,TckrSymb\n2026-11-06,RELIANCE", "columns missing"),
    (csv_payload("2026-11-05,RELIANCE,EQ,100,105,99,103,1000"), "Requested"),
    (csv_payload("2026-11-06,RELIANCE,EQ,100,98,99,103,1000"), "OHLC"),
    (csv_payload("2026-11-06,,EQ,100,105,99,103,1000"), "symbol"),
    (csv_payload("2026-11-06,RELIANCE,EQ,100,105,99,103,1000",
                 "2026-11-06,RELIANCE,EQ,100,105,99,103,1000"), "Duplicate"),
])
def test_normalizer_fails_closed(payload, expected):
    with pytest.raises(DataContractError, match=expected):
        normalize_bhavcopy(payload, date(2026, 11, 6), "2026-11-06T12:00:00+00:00")


def test_writes_raw_normalized_and_hash_manifest_only_under_audit_root(tmp_path):
    payload = csv_payload("2026-11-06,RELIANCE,EQ,100,105,99,103,1000")
    archive = zip_payload(payload)
    rows = normalize_bhavcopy(payload, date(2026, 11, 6), "2026-11-06T12:00:00+00:00")
    manifest = write_research_snapshot(date(2026, 11, 6), archive, payload, rows, tmp_path / "audit")
    assert manifest["source_id"] == SOURCE_ID
    assert manifest["research_only"] is True
    assert (tmp_path / "audit" / manifest["raw_file"]).is_file()
    assert (tmp_path / "audit" / manifest["normalized_file"]).is_file()
    assert (tmp_path / "audit" / "manifests" / next((tmp_path / "audit" / "manifests").iterdir()).name).is_file()
    assert not (tmp_path / "data" / "ohlcv.csv").exists()


def test_failed_live_source_is_recorded_as_blocked_without_mutating_data(tmp_path, monkeypatch):
    import src.nse_udiff_reconciliation as module

    def fail(_target, _lookback=10):
        raise DataContractError("source unavailable")

    monkeypatch.setattr(module, "latest_available_session", fail)
    result = run(date(2026, 11, 6), tmp_path / "audit")
    assert result["status"] == "BLOCKED"
    assert result["canonical_ohlcv_mutated"] is False
    assert result["paper_trade_ledger_mutated"] is False
    stored = json.loads((tmp_path / "audit" / "status.json").read_text())
    assert stored["status"] == "BLOCKED"
