"""Tests for the single fail-closed historical evidence readiness gate."""

import json

from research_model.data.evidence_readiness import (
    REPORT_FILES,
    _common_symbol_coverage_matches,
    evaluate_evidence_readiness,
)


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
        "corporate_action_audit": {"action_columns_missing": [], "actions": {
            "dividends": {"invalid_values": 0, "negative_values_held_for_review": 0},
            "stock_splits": {"invalid_values": 0, "negative_values_held_for_review": 0}}},
        "corporate_action_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "identity_history_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "identity_source_validation": {"error_count": 0},
        "source_document_validation": {"error_count": 0, "archive_integrity": "NOT_CHECKED",
                                       "byte_verified_documents": 0},
        "universe_reconciliation": {"status": "MATCH"},
        "raw_symbol_review": {"status": "REVIEW_REQUIRED", "pending_symbols": ["AAA"], "error_count": 0},
        "listing_eligibility": {"status": "PASS"},
        "membership_snapshot": {"status": "SNAPSHOT_ONLY_NOT_HISTORICAL_MEMBERSHIP"},
        "membership_history_validation": {"error_count": 0, "structurally_valid_rows": 0},
        "coverage_reconciliation": {"status": "MATCH", "differences": {}},
        "training_eligibility": {
            "status": "REVIEW_REQUIRED", "source_row_count": 10,
            "eligible_row_count": 0, "quarantined_row_count": 10,
            "source_sha256": "a" * 64, "eligibility_manifest_sha256": "b" * 64,
        },
    }
    for key, filename in REPORT_FILES.items():
        (reports / filename).write_text(json.dumps(values[key]), encoding="utf-8")

    result = evaluate_evidence_readiness(reports, repo)
    assert result["status"] == "BLOCKED"
    assert result["readiness_tiers"]["tier1_structural_exploration_authorized"] is True
    assert result["readiness_tiers"]["tier2_reduced_universe_training_authorized"] is False
    assert "row_level_training_eligibility_manifest" in result["readiness_tiers"]["tier2_missing_requirements"]
    assert result["readiness_tiers"]["tier3_promotion_evidence_authorized"] is False
    assert result["training_authorized"] is False
    assert result["production_authorized"] is False
    assert result["live_trading_authorized"] is False
    assert result["blocked_checks"] > 0
    assert result["source_row_counts"]["captured_documents"] == 0


def test_coverage_reconciliation_accepts_only_explained_raw_only_delta():
    coverage = {
        "status": "REVIEW_REQUIRED",
        "prior_audit_status": "REVIEW_REQUIRED",
        "differences": {
            "row_count": {"fresh_scan": 958475, "prior_audit": 915478},
            "symbol_count": {"fresh_scan": 528, "prior_audit": 500},
        },
    }
    universe = {
        "raw_ohlcv_symbol_count": 528,
        "coverage_audit_symbol_count": 500,
        "raw_only_symbols": [f"RAW{i}" for i in range(28)],
        "raw_only_row_count": 42997,
        "audit_only_symbols": [],
        "common_symbol_row_count_mismatch_count": 0,
    }

    passed, detail = _common_symbol_coverage_matches(coverage, universe)

    assert passed is True
    assert "shared-symbol row counts match" in detail
    assert "separate review" in detail


def test_coverage_reconciliation_rejects_unexplained_or_shared_symbol_drift():
    coverage = {
        "status": "REVIEW_REQUIRED",
        "prior_audit_status": "REVIEW_REQUIRED",
        "differences": {
            "row_count": {"fresh_scan": 958476, "prior_audit": 915478},
            "symbol_count": {"fresh_scan": 528, "prior_audit": 500},
        },
    }
    universe = {
        "raw_ohlcv_symbol_count": 528,
        "coverage_audit_symbol_count": 500,
        "raw_only_symbols": [f"RAW{i}" for i in range(28)],
        "raw_only_row_count": 42997,
        "audit_only_symbols": [],
        "common_symbol_row_count_mismatch_count": 0,
    }

    passed, _ = _common_symbol_coverage_matches(coverage, universe)

    assert passed is False


def test_coverage_reconciliation_rejects_any_common_symbol_row_mismatch():
    coverage = {
        "status": "REVIEW_REQUIRED",
        "prior_audit_status": "REVIEW_REQUIRED",
        "differences": {
            "row_count": {"fresh_scan": 958475, "prior_audit": 915478},
            "symbol_count": {"fresh_scan": 528, "prior_audit": 500},
        },
    }
    universe = {
        "raw_ohlcv_symbol_count": 528,
        "coverage_audit_symbol_count": 500,
        "raw_only_symbols": [f"RAW{i}" for i in range(28)],
        "raw_only_row_count": 42997,
        "audit_only_symbols": [],
        "common_symbol_row_count_mismatch_count": 1,
    }

    passed, _ = _common_symbol_coverage_matches(coverage, universe)

    assert passed is False

def test_listing_gate_allows_post_listing_gaps_but_preserves_review_status():
    from research_model.data.evidence_readiness import _listing_eligibility_is_acceptable

    assert _listing_eligibility_is_acceptable({
        "status": "REVIEW_REQUIRED",
        "status_counts": {
            "verified": 492,
            "gap_review": 8,
            "prelisting_anomaly": 0,
            "unknown": 0,
        },
    })


def test_listing_gate_still_blocks_prelisting_or_unknown_symbols():
    from research_model.data.evidence_readiness import _listing_eligibility_is_acceptable

    assert not _listing_eligibility_is_acceptable({
        "status": "BLOCKED",
        "status_counts": {
            "verified": 486,
            "gap_review": 8,
            "prelisting_anomaly": 6,
            "unknown": 0,
        },
    })
    assert not _listing_eligibility_is_acceptable({
        "status": "BLOCKED",
        "status_counts": {
            "verified": 499,
            "gap_review": 0,
            "prelisting_anomaly": 0,
            "unknown": 1,
        },
    })


def test_listing_gate_rejects_missing_status_counts():
    from research_model.data.evidence_readiness import _listing_eligibility_is_acceptable

    assert not _listing_eligibility_is_acceptable({"status": "PASS"})

