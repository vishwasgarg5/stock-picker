"""Tests for the single fail-closed historical evidence readiness gate."""

import json

from research_model.data.evidence_readiness import REPORT_FILES, evaluate_evidence_readiness


def test_empty_evidence_manifests_never_pass_readiness(tmp_path):
    repo = tmp_path / "repo"
    schemas = repo / "research_model" / "schemas"
    schemas.mkdir(parents=True)
    (schemas / "identity_source_registry.csv").write_text(
        "source_id,review_status,reviewed_by,reviewed_at_utc\n", encoding="utf-8"
    )
    (schemas / "historical_source_documents.csv").write_text(
        "document_id,source_id,sha256,capture_status,reviewed_by,reviewed_at_utc\n",
        encoding="utf-8",
    )
    (schemas / "security_identity_history.csv").write_text(
        "source_id,source_document_sha256,verification_status,reviewed_by,reviewed_at_utc\n",
        encoding="utf-8",
    )
    (schemas / "corporate_action_events.csv").write_text(
        "source_document_sha256,verification_status,reviewed_by,reviewed_at_utc\n",
        encoding="utf-8",
    )
    (schemas / "index_membership_history.csv").write_text(
        "source_id,source_document_sha256,verification_status,reviewed_by,reviewed_at_utc\n",
        encoding="utf-8",
    )

    reports = tmp_path / "reports"
    reports.mkdir()
    values = {
        "ohlcv": {"row_count": 10, "invalid_dates": 0, "invalid_symbols": 0,
                  "malformed_rows": 0, "invalid_numeric_rows": 0, "invalid_ohlc_rows": 0,
                  "duplicate_symbol_dates": 0},
        "corporate_action_audit": {"dividends": {"invalid_values": 0, "negative_values_held_for_review": 0},
                                  "stock_splits": {"invalid_values": 0, "negative_values_held_for_review": 0}},
        "corporate_action_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "identity_history_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "identity_source_validation": {"error_count": 0},
        "source_document_validation": {"error_count": 0, "archive_integrity": "NOT_CHECKED",
                                       "byte_verified_documents": 0},
        "universe_reconciliation": {"status": "MATCH"},
        "listing_eligibility": {"status": "PASS"},
        "membership_snapshot": {"status": "SNAPSHOT_ONLY_NOT_HISTORICAL_MEMBERSHIP"},
        "membership_history_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "coverage_reconciliation": {"status": "MATCH", "differences": {}},
    }
    for key, filename in REPORT_FILES.items():
        (reports / filename).write_text(json.dumps(values[key]), encoding="utf-8")

    result = evaluate_evidence_readiness(reports, repo)
    assert result["status"] == "BLOCKED"
    assert result["training_authorized"] is False
    assert result["production_authorized"] is False
    assert result["live_trading_authorized"] is False
    assert result["blocked_checks"] > 0
    assert result["source_row_counts"]["captured_documents"] == 0
