import hashlib

import pandas as pd

from src.phase132_broker_cost_evidence import COMPONENTS, MANIFEST_COLUMNS, evaluate_broker_cost_evidence


def _manifest_row(tmp_path, content=b"broker contract note"):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    document = evidence / "note-1.pdf"
    document.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    row = {
        "symbol": "ABC", "target_date": "2026-11-16", "broker": "Example Broker",
        "broker_trade_id": "TRD-1", "document_path": "note-1.pdf",
        "document_sha256": digest, "total_cost": "1.00",
    }
    row.update({name: "0.00" for name in COMPONENTS})
    row["brokerage"] = "1.00"
    return evidence, pd.DataFrame([row], columns=MANIFEST_COLUMNS)


def _trade():
    return pd.DataFrame([{
        "symbol": "ABC", "target_date": "2026-11-16", "signal": "BUY",
        "quantity": 1, "costs": 1.0,
    }])


def test_hash_components_and_unique_ledger_match_pass_integrity_only(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    detail, summary = evaluate_broker_cost_evidence(_trade(), manifest, evidence)
    assert detail.iloc[0]["status"] == "PASS_INTEGRITY_AND_LEDGER_MATCH"
    assert summary["document_hashes_verified"] == 1
    assert summary["ledger_matches"] == 1
    assert summary["source_authenticity_verified"] is False
    assert summary["canonical_ledger_mutated"] is False


def test_modified_document_fails_hash_verification(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    (evidence / "note-1.pdf").write_bytes(b"modified document")
    detail, summary = evaluate_broker_cost_evidence(_trade(), manifest, evidence)
    assert "DOCUMENT_HASH_MISMATCH" in detail.iloc[0]["status"]
    assert summary["blocked_rows"] == 1


def test_component_sum_mismatch_is_blocked(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    manifest.loc[0, "brokerage"] = "0.50"
    detail, _ = evaluate_broker_cost_evidence(_trade(), manifest, evidence)
    assert "COMPONENT_TOTAL_MISMATCH" in detail.iloc[0]["status"]


def test_ledger_cost_mismatch_is_blocked(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    trades = _trade()
    trades.loc[0, "costs"] = 2.0
    detail, _ = evaluate_broker_cost_evidence(trades, manifest, evidence)
    assert "LEDGER_COST_MISMATCH" in detail.iloc[0]["status"]


def test_ambiguous_ledger_match_is_blocked(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    trades = pd.concat([_trade(), _trade()], ignore_index=True)
    detail, _ = evaluate_broker_cost_evidence(trades, manifest, evidence)
    assert "LEDGER_TRADE_AMBIGUOUS" in detail.iloc[0]["status"]


def test_path_traversal_is_rejected(tmp_path):
    evidence, manifest = _manifest_row(tmp_path)
    manifest.loc[0, "document_path"] = "../outside.pdf"
    detail, _ = evaluate_broker_cost_evidence(_trade(), manifest, evidence)
    assert "INVALID_DOCUMENT_PATH" in detail.iloc[0]["status"]


def test_empty_manifest_does_not_invent_cost_evidence(tmp_path):
    manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    detail, summary = evaluate_broker_cost_evidence(_trade(), manifest, tmp_path)
    assert detail.empty
    assert summary["status"] == "NO_EVIDENCE_MANIFEST_ROWS"
    assert summary["costs_inferred"] is False
