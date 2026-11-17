from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.broker_data_contracts import DataContractError
from src.nse_universe_observation import (
    SOURCE_ID, normalize_nifty50, verify_observation_manifest, write_observation,
)


def payload():
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["Symbol", "Company Name", "Industry", "Series", "ISIN Code"])
    writer.writeheader()
    for i in range(50):
        writer.writerow({"Symbol": f"SYM{i:02d}", "Company Name": f"Company {i}",
                         "Industry": "Test", "Series": "EQ", "ISIN Code": f"INE{i:06d}"})
    return out.getvalue().encode()


def test_normalizer_requires_exactly_50_unique_symbols():
    raw = payload()
    rows = normalize_nifty50(raw)
    assert len(rows) == 50
    assert rows[0]["symbol"] == "SYM00"
    with pytest.raises(DataContractError, match="Expected 50"):
        normalize_nifty50(raw.replace(b"SYM49", b"SYM48"))


def test_observation_manifest_hashes_and_pit_boundary(tmp_path):
    raw = payload()
    rows = normalize_nifty50(raw)
    manifest = write_observation(rows, raw, tmp_path / "audit", "2026-11-17T08:00:00+00:00")
    manifest_path = next((tmp_path / "audit" / "manifests").glob("*.json"))
    verified = verify_observation_manifest(manifest_path, tmp_path / "audit")
    assert verified["status"] == "PASS"
    assert verified["record_count"] == 50
    assert verified["pit_eligible"] is False
    assert manifest["source_id"] == SOURCE_ID


def test_observation_rejects_raw_file_tampering(tmp_path):
    raw = payload()
    rows = normalize_nifty50(raw)
    manifest = write_observation(rows, raw, tmp_path / "audit", "2026-11-17T08:00:00+00:00")
    (tmp_path / "audit" / manifest["raw_file"]).write_bytes(b"tampered")
    manifest_path = next((tmp_path / "audit" / "manifests").glob("*.json"))
    with pytest.raises(DataContractError, match="raw_sha256"):
        verify_observation_manifest(manifest_path, tmp_path / "audit")


def test_observation_rejects_missing_files_and_pit_promotion(tmp_path):
    raw = payload()
    rows = normalize_nifty50(raw)
    manifest = write_observation(rows, raw, tmp_path / "audit", "2026-11-17T08:00:00+00:00")
    (tmp_path / "audit" / manifest["normalized_file"]).unlink()
    manifest_path = next((tmp_path / "audit" / "manifests").glob("*.json"))
    with pytest.raises(DataContractError, match="normalized_file is missing"):
        verify_observation_manifest(manifest_path, tmp_path / "audit")
