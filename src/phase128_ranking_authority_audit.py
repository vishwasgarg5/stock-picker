from __future__ import annotations
"""Phase 128: reconcile ranking evidence with the explicit promotion authority; audit-only."""
import json
from pathlib import Path
import pandas as pd
from .pipeline import DATA

READINESS_FILE = DATA / "phase114_promotion_readiness_summary.json"
LEGACY_SUMMARY_FILE = DATA / "ranking_model_validation_summary.csv"
OUTPUT_FILE = DATA / "phase128_ranking_authority_audit.json"


def audit_ranking_authority(readiness: dict, legacy_summary: pd.DataFrame) -> dict:
    """Treat the consolidated readiness artifact as authority; legacy metrics never authorize."""
    gates = readiness.get("gates") if isinstance(readiness.get("gates"), dict) else {}
    all_gates_pass = bool(gates) and all(value is True for value in gates.values())
    explicit_authorization = readiness.get("production_changes_authorized") is True
    authorization_signal = explicit_authorization and all_gates_pass

    legacy_claims = []
    if not legacy_summary.empty:
        row = legacy_summary.iloc[-1]
        for field in ("promotion_evidence", "production_ready"):
            value = row.get(field)
            if str(value).strip().lower() in {"true", "1", "yes"}:
                legacy_claims.append(field)
        status = str(row.get("status", "")).strip().lower()
        if status in {"promote", "promotion_ready", "production_ready", "diagnostic_pass"}:
            legacy_claims.append(f"status:{status}")

    conflict = bool(legacy_claims) and not authorization_signal
    return {
        "phase": 128,
        "status": (
            "AUTHORITY_CONFLICT_FAIL_CLOSED" if conflict else
            "HUMAN_REVIEW_AUTHORITY_PRESENT" if authorization_signal else
            "NO_PROMOTION_AUTHORITY"
        ),
        "readiness_status": str(readiness.get("status", "MISSING")),
        "production_champion": str(readiness.get("production_champion", "UNKNOWN")),
        "v2_mode": str(readiness.get("v2_mode", "UNKNOWN")),
        "explicit_production_authorization": explicit_authorization,
        "all_readiness_gates_pass": all_gates_pass,
        "legacy_positive_claims": legacy_claims,
        "legacy_summary_rows": int(len(legacy_summary)),
        "legacy_claims_are_authoritative": False,
        "legacy_summary_interpretation": "UNVERIFIED_HISTORICAL_DIAGNOSTIC_ONLY" if legacy_claims else "NO_POSITIVE_LEGACY_CLAIMS",
        "authority_conflict": conflict,
        "production_changes_authorized": authorization_signal,
        "automatic_promotion": False,
        "production_model_changed": False,
        "decision": "FAIL_CLOSED_KEEP_EXISTING_CHAMPION" if conflict or not authorization_signal else "HUMAN_REVIEW_REQUIRED",
        "note": "Legacy ranking metrics and status labels are evidence only; they cannot override consolidated readiness or authorize automatic promotion."
    }


def run() -> dict:
    if not READINESS_FILE.exists():
        readiness = {}
    else:
        readiness = json.loads(READINESS_FILE.read_text(encoding="utf-8"))
    legacy = pd.read_csv(LEGACY_SUMMARY_FILE) if LEGACY_SUMMARY_FILE.exists() else pd.DataFrame()
    result = audit_ranking_authority(readiness, legacy)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
