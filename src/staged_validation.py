from __future__ import annotations

"""Fail-closed staged validation readiness contract."""
REQUIRED_PREREQUISITES=(
 "point_in_time_security_master_verified",
 "inactive_and_delisted_history_verified",
 "corporate_action_lineage_verified",
 "feature_publication_and_first_seen_times_verified",
 "label_contract_frozen",
 "chronological_purged_validation_defined",
 "cost_and_slippage_evidence_reviewed",
 "development_sample_design_preregistered",
 "model_artifact_frozen",
 "final_holdout_protocol_unchanged",
)


def assess_staged_validation(prerequisites: dict, *, stage: str) -> dict:
    if stage not in {"DEVELOPMENT","BROAD_EVALUATION","CONFIRMATORY_HOLDOUT"}:
        raise ValueError("unknown validation stage")
    missing=[key for key in REQUIRED_PREREQUISITES if prerequisites.get(key) is not True]
    if stage=="DEVELOPMENT":
        # Development work may begin only after the data, label and sampling contract is defensible.
        required=REQUIRED_PREREQUISITES[:7]+(REQUIRED_PREREQUISITES[7],)
    elif stage=="BROAD_EVALUATION":
        required=list(REQUIRED_PREREQUISITES)
    else:
        required=list(REQUIRED_PREREQUISITES)
    blockers=[key for key in required if prerequisites.get(key) is not True]
    return {"stage":stage,"status":"READY_FOR_INDEPENDENT_REVIEW" if not blockers else "RESEARCH_HOLD",
            "passed_prerequisites":[key for key in required if prerequisites.get(key) is True],
            "blockers":blockers,"promotion_eligible":False,"final_holdout_unlocked":False,
            "live_execution_enabled":False,
            "note":"Readiness is not authorization. Existing repository governance and human review remain mandatory."}
