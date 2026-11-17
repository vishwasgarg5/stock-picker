"""Research-only snapshots of the current official NSE Nifty 50 universe.

These are observation-time records, never historical membership. They are stored
under audit/ and explicitly remain ineligible for point-in-time backtests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.broker_data_contracts import DataContractError, _aware_timestamp, _safe_relative_path

SOURCE_ID = "NSE_INDICES_NIFTY50_CONSTITUENTS"
SOURCE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
USER_AGENT = "StockPicker-research-audit/1.0"
FIELDS = ("symbol", "company_name", "industry", "series", "isin")
REQUIRED_MANIFEST_FIELDS = {
    "schema_version", "dataset", "universe", "exchange", "source_id", "source_url",
    "observed_at", "raw_sha256", "normalized_sha256", "record_count",
    "pit_eligible", "pit_reason", "raw_file", "normalized_file",
}


def normalize_nifty50(payload: bytes) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    except UnicodeDecodeError as exc:
        raise DataContractError("NSE Nifty 50 CSV is not valid UTF-8") from exc
    if not reader.fieldnames or "Symbol" not in reader.fieldnames:
        raise DataContractError("NSE Nifty 50 CSV is missing the Symbol column")
    rows, seen = [], set()
    for raw in reader:
        symbol = str(raw.get("Symbol") or "").strip().upper()
        if not symbol:
            raise DataContractError("NSE Nifty 50 snapshot contains an empty symbol")
        if symbol in seen:
            raise DataContractError(f"Duplicate Nifty 50 symbol: {symbol}")
        seen.add(symbol)
        rows.append({
            "symbol": symbol,
            "company_name": str(raw.get("Company Name") or "").strip(),
            "industry": str(raw.get("Industry") or "").strip(),
            "series": str(raw.get("Series") or "").strip(),
            "isin": str(raw.get("ISIN Code") or "").strip(),
        })
    if len(rows) != 50:
        raise DataContractError(f"Expected 50 Nifty 50 constituents, received {len(rows)}")
    return rows


def fetch_nifty50(timeout: int = 30) -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={
        "User-Agent": USER_AGENT, "Accept": "text/csv,*/*", "Accept-Encoding": "identity",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise DataContractError(f"NSE returned HTTP {response.status}")
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise DataContractError(f"NSE Nifty 50 download failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataContractError(f"NSE Nifty 50 download failed: {exc}") from exc
    if not payload:
        raise DataContractError("NSE returned an empty Nifty 50 snapshot")
    return payload


def _contained_file(root: Path, relative: str, field: str) -> Path:
    _safe_relative_path(relative, field)
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise DataContractError(f"{field} is missing or outside the audit root")
    return path


def verify_observation_manifest(manifest_path: Path, audit_root: Path) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    path = Path(manifest_path).resolve()
    if root not in path.parents or not path.is_file():
        raise DataContractError("Observation manifest is missing or outside the audit root")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Cannot read observation manifest: {exc}") from exc
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    extra = set(manifest) - REQUIRED_MANIFEST_FIELDS
    if missing or extra:
        raise DataContractError(f"Observation manifest fields invalid; missing={sorted(missing)}, extra={sorted(extra)}")
    if manifest["schema_version"] != "1.0.0" or manifest["dataset"] != "universe_observation":
        raise DataContractError("Unsupported universe observation schema")
    if manifest["universe"] != "NIFTY50" or manifest["exchange"] != "NSE":
        raise DataContractError("Unexpected universe or exchange")
    if manifest["source_id"] != SOURCE_ID or manifest["source_url"] != SOURCE_URL:
        raise DataContractError("Universe observation source provenance is invalid")
    if manifest["pit_eligible"] is not False or not str(manifest["pit_reason"]).strip():
        raise DataContractError("Current universe observation must remain PIT-ineligible")
    _aware_timestamp(manifest["observed_at"], "observed_at")
    if isinstance(manifest["record_count"], bool) or not isinstance(manifest["record_count"], int) or manifest["record_count"] != 50:
        raise DataContractError("Nifty 50 observation record_count must equal 50")
    raw_path = _contained_file(root, manifest["raw_file"], "raw_file")
    normalized_path = _contained_file(root, manifest["normalized_file"], "normalized_file")
    raw = raw_path.read_bytes()
    normalized = normalized_path.read_bytes()
    for field, payload in (("raw_sha256", raw), ("normalized_sha256", normalized)):
        digest = hashlib.sha256(payload).hexdigest()
        if manifest[field] != digest:
            raise DataContractError(f"{field} does not match persisted observation file")
    rows = normalize_nifty50(normalized)
    if len(rows) != manifest["record_count"]:
        raise DataContractError("Observation record count does not match normalized file")
    return {"status": "PASS", "record_count": len(rows),
            "raw_sha256_verified": True, "normalized_sha256_verified": True,
            "pit_eligible": False}


def write_observation(rows: list[dict[str, str]], raw: bytes, audit_root: Path,
                      observed_at: str | None = None) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    _aware_timestamp(observed_at, "observed_at")
    if len(rows) != 50 or len({row.get("symbol") for row in rows}) != 50:
        raise DataContractError("Observation must contain 50 unique symbols")
    normalized_io = io.StringIO(newline="")
    writer = csv.DictWriter(normalized_io, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    normalized = normalized_io.getvalue().encode("utf-8")
    stamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
    revision = hashlib.sha256(raw).hexdigest()[:12]
    raw_name = f"raw/nse/nifty50_{stamp}_{revision}.csv"
    normalized_name = f"normalized/nifty50_{stamp}_{revision}.csv"
    manifest_name = f"manifests/nifty50_observation_{stamp}_{revision}.json"
    manifest = {
        "schema_version": "1.0.0", "dataset": "universe_observation",
        "universe": "NIFTY50", "exchange": "NSE", "source_id": SOURCE_ID,
        "source_url": SOURCE_URL, "observed_at": observed_at,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "record_count": len(rows), "pit_eligible": False,
        "pit_reason": "Current constituent snapshot is an observation, not effective-dated historical membership.",
        "raw_file": raw_name, "normalized_file": normalized_name,
    }
    for relative, payload in ((raw_name, raw), (normalized_name, normalized),
                              (manifest_name, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())):
        _safe_relative_path(relative, "output_file")
        target = (root / relative).resolve()
        if root not in target.parents:
            raise DataContractError("Observation output path escaped audit root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    verify_observation_manifest(root / manifest_name, root)
    return manifest


def run(audit_root: Path = Path("audit/broker_universe")) -> dict[str, Any]:
    root = Path(audit_root)
    try:
        raw = fetch_nifty50()
        rows = normalize_nifty50(raw)
        manifest = write_observation(rows, raw, root)
        result = {"status": "PASS", "source_id": SOURCE_ID, "manifest": manifest,
                  "research_only": True, "pit_eligible": False,
                  "canonical_universe_mutated": False, "historical_membership_created": False}
    except Exception as exc:
        result = {"status": "BLOCKED", "source_id": SOURCE_ID,
                  "error": f"{type(exc).__name__}: {exc}", "research_only": True,
                  "pit_eligible": False, "canonical_universe_mutated": False,
                  "historical_membership_created": False}
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="audit/broker_universe")
    args = parser.parse_args()
    run(Path(args.output_root))


if __name__ == "__main__":
    main()
