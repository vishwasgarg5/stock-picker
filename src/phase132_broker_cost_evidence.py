from __future__ import annotations

"""Phase 132: hash-verified broker-cost evidence manifest.

A matching hash establishes file integrity only; it does not authenticate a broker,
contract note, or manually transcribed fee values. No ledger fields are backfilled.
"""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVIDENCE_ROOT = DATA / "execution_cost_evidence"
MANIFEST = DATA / "execution_cost_evidence_manifest.csv"
COMPONENTS = (
    "brokerage", "stt", "gst", "exchange_charges",
    "sebi_charges", "stamp_duty", "other_charges",
)
MANIFEST_COLUMNS = (
    "symbol", "target_date", "broker", "broker_trade_id", "document_path",
    "document_sha256", *COMPONENTS, "total_cost",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_manifest(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS), "MANIFEST_MISSING"
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=MANIFEST_COLUMNS), "MANIFEST_UNREADABLE"
    missing = sorted(set(MANIFEST_COLUMNS).difference(frame.columns))
    if missing:
        return frame, "MANIFEST_SCHEMA_INVALID:" + ",".join(missing)
    return frame, None


def evaluate_broker_cost_evidence(
    trades: pd.DataFrame,
    manifest: pd.DataFrame,
    evidence_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify manifest rows against ledger arithmetic, fee components, and file hashes."""
    detail_rows: list[dict[str, Any]] = []
    required_missing = sorted(set(MANIFEST_COLUMNS).difference(manifest.columns))
    if required_missing:
        return pd.DataFrame(), {
            "status": "BLOCKED_INVALID_MANIFEST_SCHEMA",
            "manifest_rows": int(len(manifest)),
            "missing_manifest_columns": required_missing,
            "document_hashes_verified": 0,
            "ledger_matches": 0,
            "costs_inferred": False,
            "canonical_ledger_mutated": False,
        }

    if manifest.empty:
        return pd.DataFrame(columns=[
            "manifest_row", "symbol", "target_date", "broker_trade_id",
            "document_path", "document_sha256", "calculated_component_total",
            "manifest_total_cost", "ledger_cost", "document_hash_status",
            "status",
        ]), {
            "status": "NO_EVIDENCE_MANIFEST_ROWS",
            "manifest_rows": 0,
            "document_hashes_verified": 0,
            "ledger_matches": 0,
            "costs_inferred": False,
            "canonical_ledger_mutated": False,
            "policy": "No broker documents were supplied. The audit creates no costs and does not promote simulation assumptions into broker evidence.",
        }

    if trades.empty:
        trade_rows = pd.DataFrame()
    else:
        signal = trades.get("signal", pd.Series("BUY", index=trades.index)).astype(str).str.upper().str.strip()
        qty = pd.to_numeric(trades.get("quantity", pd.Series(np.nan, index=trades.index)), errors="coerce")
        shares = pd.to_numeric(trades.get("shares", pd.Series(np.nan, index=trades.index)), errors="coerce")
        qty = qty.where(qty.gt(0), shares)
        trade_rows = trades.loc[signal.isin(["BUY", "LONG", "EXECUTED"]) & qty.gt(0)].copy()
        trade_rows["_manifest_date"] = pd.to_datetime(trade_rows.get("target_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        trade_rows["_manifest_symbol"] = trade_rows.get("symbol", pd.Series("", index=trade_rows.index)).astype(str).str.strip().str.upper()

    seen_keys: set[tuple[str, str, str]] = set()
    assigned_ledger_rows: set[Any] = set()
    for index, record in manifest.iterrows():
        symbol = str(record.get("symbol", "")).strip().upper()
        date = pd.to_datetime(record.get("target_date", ""), errors="coerce")
        date_text = date.strftime("%Y-%m-%d") if not pd.isna(date) else ""
        broker_trade_id = str(record.get("broker_trade_id", "")).strip()
        broker = str(record.get("broker", "")).strip()
        key = (symbol, date_text, broker_trade_id)
        reasons: list[str] = []

        if not symbol or not date_text or not broker_trade_id or not broker:
            reasons.append("MISSING_TRADE_IDENTITY_OR_BROKER")
        if key in seen_keys:
            reasons.append("DUPLICATE_MANIFEST_IDENTITY")
        seen_keys.add(key)

        relative_path = Path(str(record.get("document_path", "")).strip())
        if not str(relative_path) or relative_path.is_absolute() or ".." in relative_path.parts:
            reasons.append("INVALID_DOCUMENT_PATH")
            document = None
        else:
            document = (evidence_root / relative_path).resolve()
            root_resolved = evidence_root.resolve()
            if root_resolved not in document.parents:
                reasons.append("DOCUMENT_PATH_OUTSIDE_EVIDENCE_ROOT")
                document = None

        supplied_hash = str(record.get("document_sha256", "")).strip().lower()
        actual_hash = None
        hash_status = "MISSING"
        if document is not None and document.is_file():
            digest = hashlib.sha256()
            try:
                with document.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_hash = digest.hexdigest()
                hash_status = "MATCH" if supplied_hash == actual_hash else "MISMATCH"
            except OSError:
                hash_status = "UNREADABLE"
        else:
            hash_status = "FILE_MISSING"
        if hash_status != "MATCH":
            reasons.append("DOCUMENT_HASH_" + hash_status)

        component_values = {name: _finite(record.get(name)) for name in COMPONENTS}
        if any(value is None or value < 0 for value in component_values.values()):
            reasons.append("MISSING_OR_INVALID_FEE_COMPONENT")
        component_total = sum(value for value in component_values.values() if value is not None)
        total_cost = _finite(record.get("total_cost"))
        if total_cost is None or total_cost < 0:
            reasons.append("INVALID_MANIFEST_TOTAL")
        elif all(value is not None for value in component_values.values()) and abs(component_total - total_cost) > 0.02:
            reasons.append("COMPONENT_TOTAL_MISMATCH")

        candidates = pd.DataFrame()
        if not trade_rows.empty and symbol and date_text:
            candidates = trade_rows.loc[
                trade_rows["_manifest_symbol"].eq(symbol)
                & trade_rows["_manifest_date"].eq(date_text)
            ]
        if len(candidates) == 0:
            reasons.append("LEDGER_TRADE_NOT_FOUND")
            ledger_cost = None
        elif len(candidates) > 1:
            reasons.append("LEDGER_TRADE_AMBIGUOUS")
            ledger_cost = None
        else:
            ledger_index = candidates.index[0]
            ledger_row = candidates.iloc[0]
            if ledger_index in assigned_ledger_rows:
                reasons.append("LEDGER_ROW_ALREADY_MAPPED")
            assigned_ledger_rows.add(ledger_index)
            ledger_cost = next((
                _finite(ledger_row[name])
                for name in ("costs", "trading_cost", "transaction_costs")
                if name in ledger_row.index and _finite(ledger_row[name]) is not None
            ), None)
            if ledger_cost is None:
                reasons.append("LEDGER_COST_MISSING")
            elif total_cost is not None and abs(ledger_cost - total_cost) > 0.02:
                reasons.append("LEDGER_COST_MISMATCH")

        detail_rows.append({
            "manifest_row": str(index),
            "symbol": symbol,
            "target_date": date_text,
            "broker_trade_id": broker_trade_id,
            "document_path": str(relative_path),
            "document_sha256": actual_hash,
            "document_hash_status": hash_status,
            "calculated_component_total": component_total,
            "manifest_total_cost": total_cost,
            "ledger_cost": ledger_cost,
            "status": "PASS_INTEGRITY_AND_LEDGER_MATCH" if not reasons else "BLOCKED:" + ";".join(reasons),
        })

    detail = pd.DataFrame(detail_rows)
    passed = int(detail["status"].eq("PASS_INTEGRITY_AND_LEDGER_MATCH").sum())
    hash_verified = int(detail["document_hash_status"].eq("MATCH").sum())
    blocked = int(len(detail) - passed)
    summary = {
        "status": "PASS" if len(detail) and passed == len(detail) else "BLOCKED",
        "manifest_rows": int(len(detail)),
        "document_hashes_verified": hash_verified,
        "ledger_matches": passed,
        "blocked_rows": blocked,
        "source_authenticity_verified": False,
        "manual_fee_transcription_independently_verified": False,
        "costs_inferred": False,
        "costs_estimated": False,
        "canonical_ledger_mutated": False,
        "policy": "PASS means the referenced file exists, its SHA-256 matches the manifest, its declared fee components sum to the declared total, and that total matches one unambiguous ledger trade. It does not authenticate the broker document or independently verify manually transcribed values.",
    }
    return detail, summary


def build_report() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST, index=False)
    manifest, read_error = _read_manifest(MANIFEST)
    try:
        trades = pd.read_csv(DATA / "paper_trades.csv", low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        trades = pd.DataFrame()
    detail, summary = evaluate_broker_cost_evidence(trades, manifest, EVIDENCE_ROOT)
    if read_error:
        summary = {**summary, "status": "BLOCKED", "manifest_error": read_error}
    detail.to_csv(DATA / "phase132_broker_cost_evidence_detail.csv", index=False)
    report = {**summary, "manifest": "data/execution_cost_evidence_manifest.csv",
              "evidence_directory": "data/execution_cost_evidence/"}
    (DATA / "phase132_broker_cost_evidence_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    build_report()
