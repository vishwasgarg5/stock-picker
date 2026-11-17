"""Optional official NSE UDiFF ingestion for independent research verification.

This feed is stored under audit/ only. It never writes Stock Picker's canonical
OHLCV, universe, paper-trade ledger, forecasts, or production state.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.broker_data_contracts import (
    DataContractError,
    build_ingestion_manifest,
    validate_ohlcv_record,
    validate_unique_sessions,
)

SOURCE_ID = "NSE_CM_UDIFF_BHAVCOPY"
ARCHIVE_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
REQUIRED_COLUMNS = {
    "TradDt", "TckrSymb", "SctySrs", "OpnPric", "HghPric",
    "LwPric", "ClsPric", "TtlTradgVol",
}
USER_AGENT = "StockPicker-research-audit/1.0"


def download_archive(url: str, attempts: int = 3, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/zip,application/octet-stream,*/*",
            "Accept-Encoding": "identity",
        },
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise DataContractError(f"NSE returned HTTP {response.status}")
                payload = response.read()
                if not payload:
                    raise DataContractError("NSE returned an empty archive")
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            if exc.code in (403, 429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            raise DataContractError(f"NSE download failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            raise DataContractError(f"NSE download failed: {exc}") from exc
    raise DataContractError("NSE download exhausted retries")


def extract_csv(archive: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise DataContractError(f"Expected one CSV in NSE ZIP, found {len(names)}")
            return zipped.read(names[0])
    except (zipfile.BadZipFile, OSError) as exc:
        raise DataContractError("NSE bhavcopy is not a valid ZIP archive") from exc


def normalize_bhavcopy(payload: bytes, expected_session: date, retrieved_at: str) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise DataContractError(f"NSE UDiFF columns missing: {sorted(missing)}")
        records = []
        for raw in reader:
            if str(raw.get("SctySrs", "")).strip().upper() != "EQ":
                continue
            symbol = str(raw.get("TckrSymb", "")).strip().upper()
            if not symbol:
                raise DataContractError("NSE bhavcopy contains an empty EQ symbol")
            try:
                session = date.fromisoformat(str(raw["TradDt"]).strip())
                prices = [float(raw[name]) for name in ("OpnPric", "HghPric", "LwPric", "ClsPric")]
                volume = float(raw["TtlTradgVol"])
            except (TypeError, ValueError, KeyError) as exc:
                raise DataContractError(f"Invalid date or numeric OHLCV for {symbol}") from exc
            if session != expected_session:
                raise DataContractError(
                    f"Requested {expected_session}, archive contains {session}"
                )
            volume_value = int(volume) if volume.is_integer() else volume
            record = {
                "schema_version": "1.0.0", "date": session.isoformat(),
                "symbol": symbol, "exchange": "NSE", "asset_class": "equity_cash",
                "open": prices[0], "high": prices[1], "low": prices[2],
                "close": prices[3], "volume": volume_value,
                "source": SOURCE_ID, "retrieved_at": retrieved_at,
            }
            validate_ohlcv_record(record)
            records.append(record)
    except UnicodeDecodeError as exc:
        raise DataContractError("NSE CSV is not UTF-8/UTF-8-SIG text") from exc
    if not records:
        raise DataContractError("NSE bhavcopy contained no EQ records")
    validate_unique_sessions(records)
    return records


def latest_available_session(target: date, lookback_days: int = 10) -> tuple[date, bytes]:
    """Move backwards only on official 404 responses; never switch data vendors."""
    for offset in range(lookback_days + 1):
        session = target - timedelta(days=offset)
        url = ARCHIVE_TEMPLATE.format(yyyymmdd=session.strftime("%Y%m%d"))
        try:
            return session, download_archive(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
    raise DataContractError(
        f"No NSE UDiFF archive found within {lookback_days + 1} calendar days ending {target}"
    )


def write_research_snapshot(
    session: date, archive: bytes, csv_payload: bytes, records: list[dict], output_root: Path
) -> dict[str, Any]:
    """Persist raw, normalized and manifest files only below the supplied audit root."""
    output_root = Path(output_root).resolve()
    raw_bytes = archive
    normalized_bytes = csv_payload
    stamp = datetime.now(timezone.utc).isoformat()
    revision = __import__("hashlib").sha256(raw_bytes).hexdigest()[:12]
    raw_name = f"raw/nse_cm_bhavcopy_{session:%Y%m%d}_{revision}.zip"
    normalized_name = f"normalized/nse_equity_{session.isoformat()}_{revision}.csv"
    manifest_name = f"manifests/nse_cm_{session.isoformat()}_{revision}.json"
    manifest = build_ingestion_manifest(
        dataset="nse_cash_equity_ohlcv", source_id=SOURCE_ID,
        source_url=ARCHIVE_TEMPLATE.format(yyyymmdd=session.strftime("%Y%m%d")),
        exchange="NSE", session_date=session.isoformat(), retrieved_at=stamp,
        raw_bytes=raw_bytes, normalized_bytes=normalized_bytes,
        raw_file=raw_name, normalized_file=normalized_name,
        record_count=len(records), series_filter="EQ",
    )
    for relative, payload in (
        (raw_name, raw_bytes), (normalized_name, normalized_bytes),
        (manifest_name, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
    ):
        target = (output_root / relative).resolve()
        if output_root not in target.parents:
            raise DataContractError("Audit output path escaped its root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return manifest


def run(target: date | None = None, output_root: Path = Path("audit/broker_udiff")) -> dict[str, Any]:
    target = target or datetime.now(timezone.utc).date()
    output_root = Path(output_root)
    try:
        session, archive = latest_available_session(target)
        csv_payload = extract_csv(archive)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        records = normalize_bhavcopy(csv_payload, session, retrieved_at)
        manifest = write_research_snapshot(session, archive, csv_payload, records, output_root)
        result = {
            "status": "PASS", "source_id": SOURCE_ID,
            "session_date": session.isoformat(), "record_count": len(records),
            "manifest": manifest, "research_only": True,
            "canonical_ohlcv_mutated": False, "paper_trade_ledger_mutated": False,
            "production_behaviour_changed": False,
        }
    except Exception as exc:
        result = {
            "status": "BLOCKED", "source_id": SOURCE_ID,
            "error": f"{type(exc).__name__}: {exc}",
            "research_only": True, "canonical_ohlcv_mutated": False,
            "paper_trade_ledger_mutated": False, "production_behaviour_changed": False,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "status.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Latest acceptable NSE session (YYYY-MM-DD)")
    parser.add_argument("--output-root", default="audit/broker_udiff")
    args = parser.parse_args()
    requested = date.fromisoformat(args.date) if args.date else None
    run(requested, Path(args.output_root))


if __name__ == "__main__":
    main()
