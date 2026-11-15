import pandas as pd

from src.phase128_ranking_authority_audit import audit_ranking_authority


def _readiness(authorized=False, all_pass=False):
    return {
        "status": "READY" if all_pass else "INSUFFICIENT_EVIDENCE",
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_changes_authorized": authorized,
        "gates": {"pnl": all_pass, "chronological": all_pass},
    }


def test_legacy_promote_claim_cannot_override_readiness():
    legacy = pd.DataFrame([{
        "promotion_evidence": True, "production_ready": True, "status": "promote"
    }])
    result = audit_ranking_authority(_readiness(), legacy)
    assert result["authority_conflict"]
    assert not result["production_changes_authorized"]
    assert result["decision"] == "FAIL_CLOSED_KEEP_EXISTING_CHAMPION"
    assert result["production_champion"] == "V1"


def test_no_legacy_claim_still_requires_explicit_authorization():
    result = audit_ranking_authority(_readiness(), pd.DataFrame([{"status": "hold"}]))
    assert not result["production_changes_authorized"]
    assert result["status"] == "NO_PROMOTION_AUTHORITY"


def test_authorization_requires_every_gate_and_explicit_flag():
    legacy = pd.DataFrame([{"promotion_evidence": True, "status": "promote"}])
    result = audit_ranking_authority(_readiness(authorized=True, all_pass=False), legacy)
    assert not result["production_changes_authorized"]
    assert result["authority_conflict"]


def test_authorized_readiness_still_does_not_auto_promote():
    result = audit_ranking_authority(
        _readiness(authorized=True, all_pass=True),
        pd.DataFrame([{"promotion_evidence": True, "status": "promote"}]),
    )
    assert result["production_changes_authorized"]
    assert result["status"] == "HUMAN_REVIEW_AUTHORITY_PRESENT"
    assert result["automatic_promotion"] is False
    assert result["production_model_changed"] is False
